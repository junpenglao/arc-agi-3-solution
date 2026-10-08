"""Tests for the H200 launcher plan without model files or a GPU."""

from __future__ import annotations

from pathlib import Path

import argparse
from dataclasses import replace
import hashlib
import http.client
import json
import signal
import socket
import stat
import subprocess
import threading
import zipfile

import pytest

from serving.franzen_h200 import (
    RuntimePaths,
    build_server_argv,
    h200_environment,
    main,
    offline_install_commands,
    parse_gpu_metrics,
    prepare_draft_view,
    validate_fr_spec_assets,
    _precache_paths,
    _extract_uv,
    _find_unique,
    _indexed_shards,
    _launch_and_wait,
    _record_gpu_metrics,
    _run_logged,
    _runtime_environment,
    _setup_venv,
    _sglang_wheel_version,
    _verify_installed_sglang,
    _write_launch_metadata,
    validate_gpu_admission,
)


class _FakeProcess:
    def __init__(self, poll_results: tuple[int | None, ...]) -> None:
        self.pid = 12345
        self.poll_results = poll_results
        self.poll_count = 0
        self.returncode: int | None = None
        self.waited = False

    def poll(self) -> int | None:
        result = self.poll_results[min(self.poll_count, len(self.poll_results) - 1)]
        self.poll_count += 1
        if result is not None:
            self.returncode = result
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        self.waited = True
        return self.returncode or 0


def test_server_argv_preserves_notebook_serving_controls(tmp_path: Path) -> None:
    args = build_server_argv(
        sglang=tmp_path / "venv/bin/sglang",
        model_dir=tmp_path / "target",
        draft_view=tmp_path / "draft-view",
        token_map=tmp_path / "wheels/hot_tokens_64k.pt",
        port=8001,
        chat_template=tmp_path / "target/chat_template.jinja",
    )

    assert args[0:2] == (str(tmp_path / "venv/bin/sglang"), "serve")
    assert _option_value(args, "--context-length") == "139264"
    assert _option_value(args, "--max-running-requests") == "10"
    assert _option_value(args, "--kv-cache-dtype") == "fp8_e4m3"
    assert _option_value(args, "--mem-fraction-static") == "0.96"
    assert _option_value(args, "--speculative-num-steps") == "3"
    assert _option_value(args, "--speculative-num-draft-tokens") == "4"
    assert _option_value(args, "--speculative-draft-model-path") == str(
        tmp_path / "draft-view",
    )
    assert _option_value(args, "--speculative-token-map") == str(
        tmp_path / "wheels/hot_tokens_64k.pt",
    )
    assert "--ple-offload-embedding" in args
    assert _option_value(args, "--speculative-accept-threshold-single") == "1.0"
    assert _option_value(args, "--speculative-accept-threshold-acc") == "1.0"
    assert "--chat-template" in args
    from serving.franzen_h200 import GPU_STATIC_MEMORY_FRACTION

    assert GPU_STATIC_MEMORY_FRACTION == 0.96


def test_launcher_process_group_annotations_match_binary_streams() -> None:
    from serving import franzen_h200

    assert franzen_h200._terminate_process_group.__annotations__["process"] == (
        "subprocess.Popen[bytes]"
    )
    assert franzen_h200._signal_process_group.__annotations__["process"] == (
        "subprocess.Popen[bytes]"
    )
    assert "supervise the H200 server until it exits" in franzen_h200.main.__doc__


def test_server_argv_can_bind_to_the_private_host(tmp_path: Path) -> None:
    args = build_server_argv(
        sglang=tmp_path / "venv/bin/sglang",
        model_dir=tmp_path / "target",
        draft_view=tmp_path / "draft-view",
        token_map=tmp_path / "map.pt",
        port=8001,
        chat_template=None,
        bind_host="10.15.0.15",
    )

    assert args[args.index("--host") + 1] == "10.15.0.15"


@pytest.mark.parametrize(
    ("family", "host", "address"),
    [
        (socket.AF_INET, "127.0.0.1", ("127.0.0.1", 8001)),
        (socket.AF_INET6, "::1", ("::1", 8001, 0, 0)),
    ],
)
def test_port_probe_uses_resolved_ipv4_or_ipv6_socket(
    family: socket.AddressFamily,
    host: str,
    address: tuple[object, ...],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    sockets: list[tuple[int, tuple[object, ...]]] = []

    class UnoccupiedSocket:
        def __init__(self, socket_family: int) -> None:
            self.family = socket_family

        def __enter__(self) -> UnoccupiedSocket:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def connect_ex(self, sockaddr: tuple[object, ...]) -> int:
            sockets.append((self.family, sockaddr))
            return 111

    monkeypatch.setattr(
        franzen_h200.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [(family, socket.SOCK_STREAM, 0, "", address)],
    )
    monkeypatch.setattr(
        franzen_h200.socket,
        "socket",
        lambda socket_family, *_args: UnoccupiedSocket(socket_family),
    )

    franzen_h200._require_unoccupied_port(host, 8001)

    assert sockets == [(family, address)]


def test_port_probe_normalizes_invalid_host_and_rejects_occupied_ipv6(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    monkeypatch.setattr(
        franzen_h200.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(socket.gaierror("bad host")),
    )
    with pytest.raises(ValueError, match="bind host"):
        franzen_h200._require_unoccupied_port("bad-host", 8001)

    class OccupiedSocket:
        def __enter__(self) -> OccupiedSocket:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def connect_ex(self, _sockaddr: tuple[object, ...]) -> int:
            return 0

    monkeypatch.setattr(
        franzen_h200.socket,
        "getaddrinfo",
        lambda *_args, **_kwargs: [
            (socket.AF_INET6, socket.SOCK_STREAM, 0, "", ("::1", 8001, 0, 0)),
        ],
    )
    monkeypatch.setattr(
        franzen_h200.socket,
        "socket",
        lambda *_args: OccupiedSocket(),
    )
    with pytest.raises(RuntimeError, match="Port 8001 is occupied"):
        franzen_h200._require_unoccupied_port("::1", 8001)


@pytest.mark.parametrize("timeout", [0, -1])
def test_startup_timeout_must_be_positive(timeout: int) -> None:
    from serving.franzen_h200 import _validate_startup_timeout

    with pytest.raises(ValueError, match="Startup timeout must be positive"):
        _validate_startup_timeout(timeout)


def test_prepare_only_path_resolution_does_not_require_fr_spec_map(
    tmp_path: Path,
) -> None:
    from serving.franzen_h200 import _add_arguments, _resolve_paths

    wheelhouse = tmp_path / "wheelhouse"
    wheels = wheelhouse / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "sglang-test.whl").write_bytes(b"wheel")
    (wheelhouse / "requirements.lock").write_text("locked")
    parser = argparse.ArgumentParser()
    _add_arguments(parser)
    flags = parser.parse_args(
        [
            "--wheelhouse",
            str(wheelhouse),
            "--work-dir",
            str(tmp_path / "work"),
            "--prepare-only",
        ],
    )

    paths = _resolve_paths(flags)

    assert paths.token_map is None


def test_missing_shared_cache_binding_fails_before_workdir_or_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = replace(
        _runtime_paths(tmp_path),
        target_dir=tmp_path / "target",
        draft_dir=tmp_path / "draft",
    )
    paths.work_dir.rmdir()
    monkeypatch.setattr(franzen_h200, "_resolve_paths", lambda _flags: paths)
    monkeypatch.setattr(franzen_h200, "h200_environment", lambda **_kwargs: {})
    monkeypatch.setattr(franzen_h200, "_require_unoccupied_port", lambda *_args: None)
    monkeypatch.setattr(
        franzen_h200,
        "_setup_venv",
        lambda *_args: (_ for _ in ()).throw(AssertionError("setup must not run")),
    )

    with pytest.raises(RuntimeError, match="shared XDG_CACHE_HOME"):
        main(
            [
                "--target-dir",
                "/target",
                "--draft-dir",
                "/draft",
                "--wheelhouse",
                "/wheelhouse",
                "--work-dir",
                str(paths.work_dir),
                "--port",
                "8001",
            ],
        )

    assert not paths.work_dir.exists()


def test_explicit_triton_verify_override_changes_only_one_argv_pair(
    tmp_path: Path,
) -> None:
    arguments = {
        "sglang": tmp_path / "venv/bin/sglang",
        "model_dir": tmp_path / "target",
        "draft_view": tmp_path / "draft-view",
        "token_map": tmp_path / "tokens.pt",
        "port": 8001,
        "chat_template": None,
    }
    base = build_server_argv(**arguments)
    retry = build_server_argv(**arguments, linear_attn_verify_backend="triton")

    assert retry[:-2] == base
    assert retry[-2:] == ("--linear-attn-verify-backend", "triton")
    assert _option_value(retry, "--mamba-ssm-dtype") == "bfloat16"
    assert _option_value(retry, "--mamba-backend") == "flashinfer"
    assert _option_value(retry, "--linear-attn-decode-backend") == "flashinfer"
    assert _option_value(retry, "--linear-attn-prefill-backend") == "flashinfer"
    assert _option_value(retry, "--gdn-mtp-cache-mode") == "none"
    assert _option_value(retry, "--kv-cache-dtype") == "fp8_e4m3"
    assert _option_value(retry, "--speculative-num-steps") == "3"
    assert _option_value(retry, "--speculative-num-draft-tokens") == "4"
    assert _option_value(retry, "--speculative-draft-kv-cache-dtype") == "fp8_e4m3"


def test_qsa_scratch_patch_is_exact_hashed_opt_in_and_idempotent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    source = (
        franzen_h200._QSA_FORWARD_EXTEND_ANCHOR
        + franzen_h200._QSA_STORE_ANCHOR * 2
        + franzen_h200._QSA_SCRATCH_ANCHOR
    )
    expected = franzen_h200._apply_qsa_patch_text(source)
    monkeypatch.setattr(franzen_h200, "QSA_FP8_PATCH_BEFORE_SHA256", _text_digest(source))
    monkeypatch.setattr(franzen_h200, "QSA_FP8_PATCH_AFTER_SHA256", _text_digest(expected))
    path = tmp_path / "qwen_sparse_attn_backend.py"
    path.write_text(source)

    untouched = franzen_h200.apply_qsa_fp8_scratch_patch(path, enabled=False)
    assert untouched["status"] == "unpatched"
    assert path.read_text() == source

    applied = franzen_h200.apply_qsa_fp8_scratch_patch(path, enabled=True)
    assert applied["status"] == "applied"
    assert path.read_text() == expected
    assert expected.count("k_scale=1.0") == 2
    assert expected.count("v_scale=1.0") == 2
    assert "q.dtype if k_buffer.dtype == torch.float8_e4m3fn" in expected
    repeated = franzen_h200.apply_qsa_fp8_scratch_patch(path, enabled=True)
    assert repeated["status"] == "already_applied"
    with pytest.raises(ValueError, match="selector is required"):
        franzen_h200.apply_qsa_fp8_scratch_patch(path, enabled=False)


def test_qsa_scratch_patch_refuses_an_unknown_source_hash(tmp_path: Path) -> None:
    from serving import franzen_h200

    path = tmp_path / "qwen_sparse_attn_backend.py"
    path.write_text("other source")

    with pytest.raises(ValueError, match="source hash differs"):
        franzen_h200.apply_qsa_fp8_scratch_patch(path, enabled=True)


def test_qsa_mtp_hole_patch_is_pinned_to_real_v1_source_and_idempotent(
    tmp_path: Path,
) -> None:
    from serving.franzen_h200 import (
        QSA_MTP_HOLE_AFTER_SHA256,
        QSA_MTP_HOLE_BEFORE_SHA256,
        apply_qsa_fp8_scratch_patch,
        apply_qsa_mtp_hole_compaction_patch,
    )

    fixture = Path(__file__).parent / "fixtures/qwen_sparse_attn_backend-v1.py.txt"
    source = fixture.read_text()
    assert _digest(fixture) == QSA_MTP_HOLE_BEFORE_SHA256
    assert QSA_MTP_HOLE_BEFORE_SHA256 == "e5e08c37c603b2977d4b93ce1395185bc595be029ed5cb84ee976f3acb221d2c"
    path = tmp_path / "qwen_sparse_attn_backend.py"
    path.write_text(source)

    untouched = apply_qsa_mtp_hole_compaction_patch(path, enabled=False)
    assert untouched["status"] == "unpatched"
    assert path.read_text() == source

    applied = apply_qsa_mtp_hole_compaction_patch(path, enabled=True)
    assert applied["status"] == "applied"
    assert _digest(path) == QSA_MTP_HOLE_AFTER_SHA256
    patched = path.read_text()
    sort_position = patched.index(
        "sort_keys = columns + invalid.to(columns.dtype) * topk",
    )
    assert patched.index("trtllm_decode = _resolve_trtllm_sparse_decode()") < sort_position
    assert sort_position < patched.index(
        "qwen_sparse_fa2_cu_seqlens_triton(",
        sort_position,
    )
    repeated = apply_qsa_mtp_hole_compaction_patch(path, enabled=True)
    assert repeated["status"] == "already_applied"
    v1_receipt = apply_qsa_fp8_scratch_patch(path, enabled=True)
    assert v1_receipt["status"] == "already_applied"
    assert v1_receipt["effective_source_sha256"] == QSA_MTP_HOLE_AFTER_SHA256
    with pytest.raises(ValueError, match="selector is required"):
        apply_qsa_mtp_hole_compaction_patch(path, enabled=False)


def test_qsa_mtp_hole_selector_requires_v1_scratch_selector() -> None:
    from serving.franzen_h200 import main

    with pytest.raises(ValueError, match="requires --qsa-fp8-compute-scratch"):
        main(
            [
                "--wheelhouse",
                "/missing/wheelhouse",
                "--work-dir",
                "/tmp/unused-qsa-test",
                "--qsa-mtp-hole-compaction",
            ],
        )


def _text_digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_h200_environment_is_explicit_without_disabling_mtp_or_fp8() -> None:
    environment = h200_environment(
        base={"PATH": "/usr/bin", "TORCH_CUDA_ARCH_LIST": "12.0"},
    )

    assert environment["TORCH_CUDA_ARCH_LIST"] == "9.0"
    assert environment["SGLANG_ENABLE_SM120_LOWM_BF16_GEMM"] == "0"
    assert environment["SGLANG_SM120_ONLINE_MXFP8"] == "0"
    assert environment["SGLANG_SM120_LOWM_FP8_WEIGHT"] == "0"
    assert environment["SGLANG_SM120_LM_HEAD_FP8"] == "0"
    assert environment["PATH"] == "/usr/bin"
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["UV_OFFLINE"] == "1"
    args = build_server_argv(
        sglang=Path("sglang"),
        model_dir=Path("target"),
        draft_view=Path("draft"),
        token_map=Path("hot_tokens_64k.pt"),
        port=8001,
        chat_template=None,
    )
    assert _option_value(args, "--kv-cache-dtype") == "fp8_e4m3"
    assert _option_value(args, "--speculative-num-steps") == "3"


def test_offline_install_plan_never_uses_an_index(tmp_path: Path) -> None:
    commands = offline_install_commands(
        uv=tmp_path / "bin/uv",
        python=tmp_path / "venv/bin/python",
        venv=tmp_path / "venv",
        wheels=tmp_path / "bundle/wheels",
        lock=tmp_path / "bundle/requirements.lock",
        sglang_wheel=tmp_path / "bundle/wheels/sglang.whl",
    )

    assert commands[0][-4:] == (
        "venv",
        "--python",
        str(tmp_path / "venv/bin/python"),
        str(tmp_path / "venv"),
    )
    install = commands[1]
    assert "--no-index" in install
    assert "--find-links" in install
    assert "-r" in install
    assert commands[2][-3:] == (
        "--reinstall",
        "--no-deps",
        str(tmp_path / "bundle/wheels/sglang.whl"),
    )
    assert all("index-url" not in item for command in commands for item in command)


def test_gpu_metrics_are_parsed_and_sm90_is_required() -> None:
    metrics = parse_gpu_metrics(
        "0, NVIDIA H200, 9.0, 143771, 1024, 142747, 2, 120.5, 32\n",
    )

    assert metrics[0]["name"] == "NVIDIA H200"
    assert metrics[0]["compute_capability"] == "9.0"
    assert metrics[0]["memory_total_mib"] == 143771

    with pytest.raises(ValueError, match="exactly one NVIDIA H200"):
        parse_gpu_metrics("0, NVIDIA RTX PRO 6000, 12.0, 100000, 0, 100000, 0, 0, 30\n")


def test_gpu_admission_requires_one_slurm_h200_with_static_headroom() -> None:
    gpus = parse_gpu_metrics("0, NVIDIA H200, 9.0, 143771, 615, 143156, 0, 75.72, 28\n")
    admission = validate_gpu_admission(
        gpus,
        environment={"SLURM_JOB_ID": "38654", "SLURM_JOB_GPUS": "0"},
    )

    assert admission["slurm_job_id"] == "38654"
    assert admission["minimum_free_memory_mib"] == 138021
    with pytest.raises(RuntimeError, match="requires 138021 MiB"):
        validate_gpu_admission(
            parse_gpu_metrics("0, NVIDIA H200, 9.0, 143771, 6000, 137771, 0, 75.72, 28\n"),
            environment={"SLURM_JOB_ID": "38654", "SLURM_JOB_GPUS": "0"},
        )
    with pytest.raises(RuntimeError, match="one GPU in an active Slurm job"):
        validate_gpu_admission(gpus, environment={})


def test_final_launch_admission_records_a_fresh_gpu_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    gpus = parse_gpu_metrics("0, NVIDIA H200, 9.0, 143771, 615, 143156, 0, 75.72, 28\n")
    monkeypatch.setattr(franzen_h200, "_query_gpu", lambda: gpus)

    measured, admission = franzen_h200._refresh_launch_admission(
        paths=paths,
        environment={"SLURM_JOB_ID": "38699", "SLURM_JOB_GPUS": "0"},
    )

    assert measured == gpus
    assert admission["slurm_job_id"] == "38699"
    record = json.loads(paths.metrics.read_text())
    assert record["phase"] == "prelaunch"
    assert record["admission"] == admission


def test_required_finder_has_non_optional_success_contract(tmp_path: Path) -> None:
    required = tmp_path / "required"
    required.write_text("value")

    assert _find_unique(tmp_path, "required") == required
    with pytest.raises(FileNotFoundError):
        _find_unique(tmp_path, "missing")


def test_sglang_wheel_metadata_must_match_pinned_version(tmp_path: Path) -> None:
    wheel = tmp_path / "sglang-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "sglang-0.5.dist-info/METADATA",
            "Name: sglang\nVersion: 0.5.19+gd00d88efc8d6\n\n",
        )

    assert _sglang_wheel_version(wheel) == "0.5.19+gd00d88efc8d6"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(
            "sglang-0.5.dist-info/METADATA",
            "Name: sglang\nVersion: 0.5.19\n\n",
        )
    with pytest.raises(ValueError, match="Expected SGLang"):
        _sglang_wheel_version(wheel)


def test_installed_sglang_must_match_the_pinned_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    monkeypatch.setattr(
        franzen_h200,
        "_run_logged",
        lambda *_args: "Name: sglang\nVersion: 0.5.19+gd00d88efc8d6\n",
    )

    _verify_installed_sglang(paths, {})

    monkeypatch.setattr(
        franzen_h200,
        "_run_logged",
        lambda *_args: "Name: sglang\nVersion: 0.5.19\n",
    )
    with pytest.raises(RuntimeError, match="Installed SGLang differs"):
        _verify_installed_sglang(paths, {})


def test_reused_venv_keeps_original_install_receipt_immutable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    uv_wheel = paths.wheels / "uv-test.whl"
    sglang_wheel = paths.wheels / "sglang-test.whl"
    with zipfile.ZipFile(uv_wheel, "w") as archive:
        archive.writestr("uv", b"\x7fELFsynthetic uv")
    with zipfile.ZipFile(sglang_wheel, "w") as archive:
        archive.writestr(
            "sglang-0.5.dist-info/METADATA",
            "Name: sglang\nVersion: 0.5.19+gd00d88efc8d6\n\n",
        )
    paths.lock.write_text("locked")
    paths.venv.mkdir(parents=True)
    paths.python.parent.mkdir(parents=True)
    paths.python.write_text("python")
    paths.sglang.write_text("sglang")
    identity = {
        "uv_wheel_sha256": _digest(uv_wheel),
        "sglang_wheel_sha256": _digest(sglang_wheel),
        "requirements_lock_sha256": _digest(paths.lock),
        "python": franzen_h200.platform.python_version(),
        "sglang_version": franzen_h200.SGLANG_VERSION,
        "sglang_revision": franzen_h200.SGLANG_REVISION,
    }
    marker = paths.work_dir / "installed-bundle.json"
    marker.write_text(json.dumps(identity))
    paths.install_metadata.write_text(json.dumps({"identity": identity, "mode": "installed"}))
    before = _digest(paths.install_metadata)
    monkeypatch.setattr(franzen_h200.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(franzen_h200, "_verify_installed_sglang", lambda *_args: None)

    receipt = _setup_venv(paths, h200_environment(base={}))

    assert receipt["mode"] == "reused"
    assert _digest(paths.install_metadata) == before


def test_workdir_lock_rejects_a_second_launcher_for_same_state(
    tmp_path: Path,
) -> None:
    from serving.franzen_h200 import _workdir_lock

    paths = _runtime_paths(tmp_path)
    with _workdir_lock(paths):
        with pytest.raises(RuntimeError, match="work directory is already locked"):
            with _workdir_lock(paths):
                raise AssertionError("a second launcher must not enter shared setup")


@pytest.mark.parametrize("fail_at", ["receipt", "marker"])
def test_install_completion_marker_is_last_publication_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    fail_at: str,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    marker = paths.work_dir / "installed-bundle.json"
    identity = {"bundle": "pinned"}
    receipt = {"identity": identity, "mode": "installed"}
    original_write_json = franzen_h200._write_json
    failure_path = paths.install_metadata if fail_at == "receipt" else marker

    def write_json(path: Path, value: object) -> None:
        if path == failure_path:
            raise OSError(f"synthetic {fail_at} publication failure")
        original_write_json(path, value)

    monkeypatch.setattr(franzen_h200, "_write_json", write_json)
    with pytest.raises(RuntimeError, match="incomplete"):
        franzen_h200._publish_install_state(paths, identity, receipt)

    assert marker.exists() is False
    assert paths.install_metadata.exists() is (fail_at == "marker")


def test_launch_receipt_records_controls_hardware_and_separate_clocks(
    tmp_path: Path,
) -> None:
    paths = _runtime_paths(tmp_path)
    paths.install_metadata.write_text(json.dumps({"mode": "installed"}))
    gpu = parse_gpu_metrics("0, NVIDIA H200, 9.0, 143771, 615, 143156, 0, 75.72, 28\n")
    admission = validate_gpu_admission(
        gpu,
        environment={"SLURM_JOB_ID": "38654", "SLURM_JOB_GPUS": "0"},
    )

    _write_launch_metadata(
        paths=paths,
        argv=("sglang", "serve", "--host", "10.15.0.15"),
        environment={"XDG_CACHE_HOME": "/shared/cache", "TORCH_CUDA_ARCH_LIST": "9.0"},
        draft_view=tmp_path / "draft-view",
        started=100.0,
        startup_timeout=720,
        token_map_digest="a" * 64,
        tokenizer_digest="b" * 64,
        target_shards=4,
        install_receipt={"mode": "reused"},
        gpu_metrics=gpu,
        admission=admission,
        setup_started=110.0,
        setup_completed=120.0,
        target_config_sha256="c" * 64,
        target_index_sha256="d" * 64,
        draft_config_sha256="e" * 64,
        draft_index_sha256="f" * 64,
        linear_attn_verify_backend="triton",
        qsa_patch={
            "enabled": True,
            "status": "applied",
            "path": "/pinned/qwen_sparse_attn_backend.py",
            "before_sha256": "b" * 64,
            "after_sha256": "a" * 64,
        },
        qsa_mtp_hole_patch={
            "enabled": True,
            "status": "applied",
            "path": "/pinned/qwen_sparse_attn_backend.py",
            "before_sha256": "a" * 64,
            "after_sha256": "c" * 64,
        },
    )

    receipt = json.loads(paths.metadata.read_text())
    assert receipt["gpu_preflight"][0]["memory_free_mib"] == 143156
    assert receipt["launch_admission"]["slurm_job_id"] == "38654"
    assert receipt["managed_environment"]["XDG_CACHE_HOME"] == "/shared/cache"
    assert receipt["clocks"]["notebook_start_epoch"] == 100.0
    assert receipt["clocks"]["launcher_setup_seconds"] == 10.0
    assert any(
        "--linear-attn-verify-backend triton" in difference
        for difference in receipt["differences_from_notebook"]
    )
    assert receipt["qsa_fp8_compute_scratch_patch"]["after_sha256"] == "a" * 64
    assert any(
        "QSA FP8-to-query-dtype" in difference
        for difference in receipt["differences_from_notebook"]
    )
    assert receipt["qsa_mtp_hole_compaction_patch"]["after_sha256"] == "c" * 64
    assert receipt["command"].endswith("10.15.0.15")


def test_startup_timeout_terminates_and_reaps_owned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    process = _FakeProcess((None, None, None))
    signals: list[int] = []
    group_alive = [True]

    class ControlledClock:
        current = 10.0

        def monotonic(self) -> float:
            return self.current

        def sleep(self, seconds: float) -> None:
            self.current += seconds

    clock = ControlledClock()
    monkeypatch.setattr(franzen_h200.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(franzen_h200, "_query_gpu", lambda: [])
    monkeypatch.setattr(
        franzen_h200.urllib.request,
        "urlopen",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("not ready")),
    )
    monkeypatch.setattr(franzen_h200.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(franzen_h200.time, "sleep", clock.sleep)
    monkeypatch.setattr(franzen_h200, "_process_group_exists", lambda _process: group_alive[0])

    def signal_group(_process, sig: signal.Signals) -> None:
        signals.append(process.pid)
        group_alive[0] = False
        process.returncode = -int(sig)

    monkeypatch.setattr(
        franzen_h200,
        "_signal_process_group",
        signal_group,
    )

    result = _launch_and_wait(
        paths=paths,
        argv=("sglang", "serve", "--port", "8001"),
        environment={},
        notebook_start_epoch=1.0,
        startup_timeout=1,
        precache_cancel=threading.Event(),
        cancellation=threading.Event(),
        bind_host="10.15.0.15",
    )

    assert result == 2
    assert signals == [process.pid]
    assert process.waited
    assert json.loads((paths.work_dir / "launch-result.json").read_text())["server_left_running"] is False


def test_exited_leader_still_reaps_live_owned_process_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    process = _FakeProcess((0,))
    group_alive = True
    signals: list[signal.Signals] = []

    def signal_group(_process, sig: signal.Signals) -> None:
        nonlocal group_alive
        signals.append(sig)
        group_alive = False

    monkeypatch.setattr(franzen_h200.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(franzen_h200, "_query_gpu", lambda: [])
    monkeypatch.setattr(franzen_h200, "_process_group_exists", lambda _process: group_alive, raising=False)
    monkeypatch.setattr(franzen_h200, "_signal_process_group", signal_group)

    result = _launch_and_wait(
        paths=paths,
        argv=("sglang", "serve", "--port", "8001"),
        environment={},
        notebook_start_epoch=1.0,
        startup_timeout=30,
        precache_cancel=threading.Event(),
        cancellation=threading.Event(),
        bind_host="127.0.0.1",
    )

    assert result == 2
    assert signals == [signal.SIGTERM]
    assert json.loads((paths.work_dir / "launch-result.json").read_text())["server_left_running"] is False


def test_process_group_escalates_when_descendant_ignores_term(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    process = _FakeProcess((0,))
    group_alive = True
    signals: list[signal.Signals] = []

    def signal_group(_process, sig: signal.Signals) -> None:
        nonlocal group_alive
        signals.append(sig)
        if sig == signal.SIGKILL:
            group_alive = False

    monkeypatch.setattr(franzen_h200, "GROUP_TERMINATION_GRACE_SECONDS", 0, raising=False)
    monkeypatch.setattr(franzen_h200, "_process_group_exists", lambda _process: group_alive, raising=False)
    monkeypatch.setattr(franzen_h200, "_signal_process_group", signal_group)
    monkeypatch.setattr(franzen_h200.time, "sleep", lambda _seconds: None)

    franzen_h200._terminate_process_group(process)

    assert signals == [signal.SIGTERM, signal.SIGKILL]


def test_launch_receipt_reports_a_process_group_survivor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    process = _FakeProcess((0,))
    signals: list[signal.Signals] = []
    monkeypatch.setattr(franzen_h200.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(franzen_h200, "_query_gpu", lambda: [])
    monkeypatch.setattr(franzen_h200, "_process_group_exists", lambda _process: True)
    monkeypatch.setattr(
        franzen_h200,
        "_signal_process_group",
        lambda _process, sig: signals.append(sig),
    )
    monkeypatch.setattr(franzen_h200, "GROUP_TERMINATION_GRACE_SECONDS", 0)
    monkeypatch.setattr(franzen_h200, "GROUP_KILL_WAIT_SECONDS", 0)

    result = _launch_and_wait(
        paths=paths,
        argv=("sglang", "serve", "--port", "8001"),
        environment={},
        notebook_start_epoch=1.0,
        startup_timeout=30,
        precache_cancel=threading.Event(),
        cancellation=threading.Event(),
        bind_host="127.0.0.1",
    )

    launch_receipt = json.loads((paths.work_dir / "launch-result.json").read_text())
    pid_receipt = json.loads(paths.pid_file.read_text())
    assert result == 2
    assert signals == [signal.SIGTERM, signal.SIGKILL]
    assert launch_receipt["server_left_running"] is True
    assert pid_receipt["status"] == "group_survivors"


def test_post_spawn_receipt_failure_still_cleans_owned_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    process = _FakeProcess((None,))
    group_alive = True
    signals: list[signal.Signals] = []
    original_write_json = franzen_h200._write_json
    fail_pid_receipt = True

    def write_json(path: Path, payload: object) -> None:
        nonlocal fail_pid_receipt
        if path == paths.pid_file and fail_pid_receipt:
            fail_pid_receipt = False
            raise OSError("synthetic PID receipt failure")
        original_write_json(path, payload)

    def signal_group(_process, sig: signal.Signals) -> None:
        nonlocal group_alive
        signals.append(sig)
        group_alive = False

    monkeypatch.setattr(franzen_h200.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(franzen_h200, "_process_group_exists", lambda _process: group_alive, raising=False)
    monkeypatch.setattr(franzen_h200, "_signal_process_group", signal_group)
    monkeypatch.setattr(franzen_h200, "_write_json", write_json)

    with pytest.raises(OSError, match="synthetic PID receipt failure"):
        _launch_and_wait(
            paths=paths,
            argv=("sglang", "serve", "--port", "8001"),
            environment={},
            notebook_start_epoch=1.0,
            startup_timeout=30,
            precache_cancel=threading.Event(),
            cancellation=threading.Event(),
            bind_host="127.0.0.1",
        )

    assert signals == [signal.SIGTERM]


def test_readiness_does_not_stop_supervision_or_gpu_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    process = _FakeProcess((None, None, 0))
    health_urls: list[str] = []
    metrics_started = threading.Event()
    metrics_stopped = threading.Event()
    metrics_was_live_for_group_cleanup: list[bool] = []
    group_alive = [True]

    class Ready:
        status = 200

        def __enter__(self) -> Ready:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(franzen_h200.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(franzen_h200, "_query_gpu", lambda: [])
    monkeypatch.setattr(franzen_h200, "_process_group_exists", lambda _process: group_alive[0])

    def signal_group(_process, _sig: signal.Signals) -> None:
        metrics_was_live_for_group_cleanup.append(not metrics_stopped.is_set())
        group_alive[0] = False

    monkeypatch.setattr(franzen_h200, "_signal_process_group", signal_group)
    monkeypatch.setattr(
        franzen_h200,
        "_record_gpu_metrics",
        lambda _path, stop: (metrics_started.set(), stop.wait(), metrics_stopped.set()),
    )
    monkeypatch.setattr(
        franzen_h200.urllib.request,
        "urlopen",
        lambda url, **kwargs: health_urls.append(url) or Ready(),
    )
    monkeypatch.setattr(franzen_h200.time, "sleep", lambda _seconds: None)

    result = _launch_and_wait(
        paths=paths,
        argv=("sglang", "serve", "--port", "8001"),
        environment={},
        notebook_start_epoch=1.0,
        startup_timeout=30,
        precache_cancel=threading.Event(),
        cancellation=threading.Event(),
        bind_host="10.15.0.15",
    )

    assert result == 0
    assert len(health_urls) == 1
    assert health_urls[0] == "http://10.15.0.15:8001/health"
    assert process.poll_count >= 3
    assert process.waited
    assert metrics_started.is_set()
    assert metrics_stopped.is_set()
    assert metrics_was_live_for_group_cleanup == [True]


def test_readiness_retries_malformed_http_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    process = _FakeProcess((None, None, 0))
    responses: list[object] = [
        http.client.BadStatusLine("malformed"),
    ]
    request_count = 0

    class Ready:
        status = 200

        def __enter__(self) -> Ready:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    responses.append(Ready())

    def open_health(*_args, **_kwargs):
        nonlocal request_count
        request_count += 1
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(franzen_h200.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(franzen_h200, "_query_gpu", lambda: [])
    monkeypatch.setattr(franzen_h200, "_process_group_exists", lambda _process: False)
    monkeypatch.setattr(franzen_h200, "_signal_process_group", lambda *_args: None)
    monkeypatch.setattr(
        franzen_h200,
        "_record_gpu_metrics",
        lambda _path, stop: stop.wait(),
    )
    monkeypatch.setattr(franzen_h200.urllib.request, "urlopen", open_health)
    monkeypatch.setattr(franzen_h200.time, "sleep", lambda _seconds: None)

    result = _launch_and_wait(
        paths=paths,
        argv=("sglang", "serve", "--port", "8001"),
        environment={},
        notebook_start_epoch=1.0,
        startup_timeout=30,
        precache_cancel=threading.Event(),
        cancellation=threading.Event(),
        bind_host="127.0.0.1",
    )

    assert result == 0
    assert request_count == 2


def test_cancellation_reaps_the_owned_process_and_cancels_precache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    process = _FakeProcess((None, None))
    cancellation = threading.Event()
    cancellation.set()
    precache_cancel = threading.Event()
    signals: list[int] = []
    group_alive = [True]
    monkeypatch.setattr(franzen_h200.subprocess, "Popen", lambda *args, **kwargs: process)
    monkeypatch.setattr(franzen_h200, "_query_gpu", lambda: [])
    monkeypatch.setattr(franzen_h200, "_process_group_exists", lambda _process: group_alive[0])

    def signal_group(_process, sig: signal.Signals) -> None:
        signals.append(process.pid)
        group_alive[0] = False
        process.returncode = -int(sig)

    monkeypatch.setattr(
        franzen_h200,
        "_signal_process_group",
        signal_group,
    )

    result = _launch_and_wait(
        paths=paths,
        argv=("sglang", "serve", "--port", "8001"),
        environment={},
        notebook_start_epoch=1.0,
        startup_timeout=30,
        precache_cancel=precache_cancel,
        cancellation=cancellation,
        bind_host="127.0.0.1",
    )

    assert result == 130
    assert signals == [process.pid]
    assert process.waited
    assert precache_cancel.is_set()


def test_draft_view_skips_unrelated_config_and_checks_links(tmp_path: Path) -> None:
    target = tmp_path / "target"
    source = tmp_path / "draft"
    work = tmp_path / "work"
    target.mkdir()
    source.mkdir()
    (target / "tokenizer.json").write_text("tokenizer")
    (target / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "auto-round", "bits": 4}}))
    candidate = source / "valid-draft"
    candidate.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "unquantized"}))
    (candidate / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "mtp_routed_experts": {
                            "weights": {"num_bits": 4, "group_size": 32, "symmetric": True},
                            "targets": ["RoutedExperts"],
                        },
                    },
                    "ignore": [],
                },
            },
        ),
    )
    (candidate / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.layers.0.mlp.experts.0.up_proj.weight": "model-00001.safetensors"}}),
    )
    (candidate / "model-00001.safetensors").write_bytes(b"synthetic shard")

    view = prepare_draft_view(target=target, source_root=source, work_root=work)

    assert (view / "model-00001.safetensors").is_symlink()
    assert (view / "model-00001.safetensors").resolve() == candidate / "model-00001.safetensors"
    assert json.loads((view / "config.json").read_text())["quantization_config"]["config_groups"]["mtp_routed_experts"]["weights"]["group_size"] == 32
    assert not (work / "model-00001.safetensors").exists()
    (view / "stale-extra.bin").write_bytes(b"stale")
    with pytest.raises(ValueError, match="unexpected draft-view inventory"):
        prepare_draft_view(target=target, source_root=source, work_root=work)
    (view / "stale-extra.bin").unlink()

    next_shard = candidate / "model-00002.safetensors"
    next_shard.write_bytes(b"second shard")
    (candidate / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"mtp.layers.0.mlp.experts.0.up_proj.weight": "model-00002.safetensors"}}),
    )
    updated_view = prepare_draft_view(target=target, source_root=source, work_root=work)
    assert updated_view != view
    assert (updated_view / "model-00002.safetensors").resolve() == next_shard
    assert not (updated_view / "model-00001.safetensors").exists()

    bad_target = updated_view / "model-00002.safetensors"
    bad_target.unlink()
    wrong_shard = candidate / "wrong.safetensors"
    wrong_shard.write_bytes(b"different shard")
    bad_target.symlink_to(wrong_shard)
    with pytest.raises(ValueError, match="Unexpected existing draft-view file"):
        prepare_draft_view(target=target, source_root=source, work_root=work)


def test_shard_index_rejects_non_string_values(tmp_path: Path) -> None:
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"layer.weight": 7}}),
    )

    with pytest.raises(ValueError, match="string"):
        _indexed_shards(tmp_path)


def test_cli_help_exposes_required_paths(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--target-dir" in output
    assert "--draft-dir" in output
    assert "--wheelhouse" in output
    assert "--work-dir" in output
    assert "--port" in output
    assert "--prepare-only" in output
    assert "--bind-host" in output
    assert "--qsa-fp8-compute-scratch" in output


def test_prepare_only_needs_no_model_paths_or_gpu(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    wheelhouse = tmp_path / "bundle"
    wheels = wheelhouse / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "sglang-test.whl").touch()
    (wheels / "uv-test.whl").touch()
    (wheelhouse / "requirements.lock").write_text("locked")
    (wheelhouse / "hot_tokens_64k.pt").write_bytes(b"map")
    calls: list[Path] = []

    def setup_venv(paths: franzen_h200.RuntimePaths, environment: dict[str, str]) -> dict[str, object]:
        del environment
        calls.append(paths.venv)
        return {"mode": "test"}

    def reject_gpu_probe() -> list[dict[str, object]]:
        raise AssertionError("prepare-only must not inspect a GPU")

    monkeypatch.setattr(franzen_h200, "_setup_venv", setup_venv)
    monkeypatch.setattr(franzen_h200, "_query_gpu", reject_gpu_probe)
    work_dir = tmp_path / "prepared"

    result = main(
        [
            "--prepare-only",
            "--wheelhouse",
            str(wheelhouse),
            "--work-dir",
            str(work_dir),
        ],
    )

    assert result == 0
    assert calls == [work_dir / "venv"]


def test_invalid_model_config_fails_before_gpu_probe_or_install(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    wheelhouse = tmp_path / "bundle"
    wheels = wheelhouse / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "sglang-test.whl").touch()
    (wheelhouse / "requirements.lock").write_text("locked")
    (wheelhouse / "hot_tokens_64k.pt").write_bytes(b"map")
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    (target / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "auto-round", "bits": 8}}),
    )
    shared_cache = tmp_path / "shared-cache"
    shared_cache.mkdir()
    monkeypatch.setattr(
        franzen_h200,
        "h200_environment",
        lambda *, base: {**base, "XDG_CACHE_HOME": str(shared_cache)},
    )
    monkeypatch.setattr(franzen_h200, "_require_unoccupied_port", lambda *_args: None)
    monkeypatch.setattr(
        franzen_h200,
        "_query_gpu",
        lambda: (_ for _ in ()).throw(AssertionError("invalid assets precede GPU probe")),
    )
    monkeypatch.setattr(
        franzen_h200,
        "_setup_venv",
        lambda *_args: (_ for _ in ()).throw(AssertionError("invalid assets precede install")),
    )

    with pytest.raises(ValueError, match="AutoRound INT4"):
        main(
            [
                "--target-dir",
                str(target),
                "--draft-dir",
                str(draft),
                "--wheelhouse",
                str(wheelhouse),
                "--work-dir",
                str(tmp_path / "work"),
                "--port",
                "8001",
            ],
        )


def test_full_serve_orchestration_rechecks_admission_then_launches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    wheelhouse = tmp_path / "bundle"
    wheels = wheelhouse / "wheels"
    wheels.mkdir(parents=True)
    (wheels / "sglang-test.whl").write_bytes(b"wheel")
    (wheelhouse / "requirements.lock").write_text("locked")
    token_map = wheelhouse / "hot_tokens_64k.pt"
    token_map.write_bytes(b"token map")
    target = tmp_path / "target"
    target.mkdir()
    tokenizer = target / "tokenizer.json"
    tokenizer.write_bytes(b"tokenizer")
    (target / "config.json").write_text(
        json.dumps({"quantization_config": {"quant_method": "auto-round", "bits": 4}}),
    )
    (target / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"target.weight": "target.safetensors"}}),
    )
    (target / "target.safetensors").write_bytes(b"target shard")
    draft = tmp_path / "draft"
    candidate = draft / "candidate"
    candidate.mkdir(parents=True)
    (candidate / "config.json").write_text(
        json.dumps(
            {
                "quantization_config": {
                    "quant_method": "compressed-tensors",
                    "config_groups": {
                        "mtp_routed_experts": {
                            "weights": {
                                "num_bits": 4,
                                "group_size": 32,
                                "symmetric": True,
                            },
                            "targets": ["RoutedExperts"],
                        },
                    },
                    "ignore": [],
                },
            },
        ),
    )
    (candidate / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "mtp.layers.0.mlp.experts.0.up_proj.weight": "draft.safetensors",
                },
            },
        ),
    )
    (candidate / "draft.safetensors").write_bytes(b"draft shard")
    shared_cache = tmp_path / "shared-cache"
    shared_cache.mkdir()
    monkeypatch.setenv("XDG_CACHE_HOME", str(shared_cache))
    monkeypatch.setenv("SLURM_JOB_ID", "38700")
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")
    monkeypatch.setattr(franzen_h200, "TOKEN_MAP_SHA256", _digest(token_map))
    monkeypatch.setattr(franzen_h200, "TOKENIZER_SHA256", _digest(tokenizer))
    monkeypatch.setattr(franzen_h200, "_require_unoccupied_port", lambda *_args: None)
    gpu = parse_gpu_metrics("0, NVIDIA H200, 9.0, 143771, 615, 143156, 0, 75.72, 28\n")
    gpu_calls = 0

    def query_gpu() -> list[dict[str, object]]:
        nonlocal gpu_calls
        gpu_calls += 1
        return gpu

    events: list[str] = []
    monkeypatch.setattr(franzen_h200, "_query_gpu", query_gpu)
    monkeypatch.setattr(
        franzen_h200,
        "_precache_paths",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        franzen_h200,
        "_setup_venv",
        lambda *_args: events.append("setup") or {"mode": "installed"},
    )
    monkeypatch.setattr(
        franzen_h200,
        "apply_qsa_fp8_scratch_patch",
        lambda *_args, **_kwargs: {"enabled": False, "status": "unpatched"},
    )
    monkeypatch.setattr(
        franzen_h200,
        "apply_qsa_mtp_hole_compaction_patch",
        lambda *_args, **_kwargs: {"enabled": False, "status": "unpatched"},
    )
    monkeypatch.setattr(
        franzen_h200,
        "_qsa_source_file",
        lambda _venv: tmp_path / "qwen_sparse_attn_backend.py",
    )
    monkeypatch.setattr(
        franzen_h200,
        "_prepare_cuda",
        lambda _paths: (tmp_path / "cuda", "gcc", "g++"),
    )
    monkeypatch.setattr(
        franzen_h200,
        "_runtime_environment",
        lambda *, base, **_kwargs: base,
    )
    monkeypatch.setattr(franzen_h200, "_run_logged", lambda *_args: "nvcc")

    def write_metadata(**kwargs: object) -> None:
        events.append("metadata")
        admission = kwargs["admission"]
        assert isinstance(admission, dict)
        events.append(str(admission["slurm_job_id"]))

    def launch(**_kwargs: object) -> int:
        events.append("launch")
        return 0

    monkeypatch.setattr(franzen_h200, "_write_launch_metadata", write_metadata)
    monkeypatch.setattr(franzen_h200, "_launch_and_wait", launch)
    monkeypatch.setattr(franzen_h200.signal, "signal", lambda *_args: None)

    result = main(
        [
            "--target-dir",
            str(target),
            "--draft-dir",
            str(draft),
            "--wheelhouse",
            str(wheelhouse),
            "--work-dir",
            str(tmp_path / "work"),
            "--port",
            "8001",
        ],
    )

    assert result == 0
    assert gpu_calls == 2
    assert events == ["setup", "metadata", "38700", "launch"]


def test_fr_spec_assets_are_hash_checked(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    token_map = tmp_path / "hot_tokens_64k.pt"
    tokenizer = tmp_path / "tokenizer.json"
    token_map.write_bytes(b"map")
    tokenizer.write_bytes(b"tokenizer")
    monkeypatch.setattr(franzen_h200, "TOKEN_MAP_SHA256", _digest(token_map))
    monkeypatch.setattr(franzen_h200, "TOKENIZER_SHA256", _digest(tokenizer))

    assert validate_fr_spec_assets(token_map=token_map, tokenizer=tokenizer) == (
        _digest(token_map),
        _digest(tokenizer),
    )
    token_map.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hot-token map"):
        validate_fr_spec_assets(token_map=token_map, tokenizer=tokenizer)


def test_precache_reads_files_without_modifying_inputs(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    source = dataset / "shard.bin"
    source.write_bytes(b"offline fixture")
    original = source.read_bytes()
    log = tmp_path / "precache.jsonl"

    _precache_paths((dataset,), threads=1, log=log)

    assert source.read_bytes() == original
    record = json.loads(log.read_text())
    assert record["bytes"] == len(original)
    assert record["errors"] == []


def test_precache_cancellation_returns_without_reading_remaining_chunks(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    (dataset / "shard.bin").write_bytes(b"offline fixture")
    cancellation = threading.Event()
    cancellation.set()

    _precache_paths(
        (dataset,),
        threads=2,
        log=tmp_path / "precache.jsonl",
        cancellation=cancellation,
    )

    record = json.loads((tmp_path / "precache.jsonl").read_text())
    assert record["cancelled"] is True


def test_runtime_environment_keeps_shared_cache_paths_and_single_cuda_prefix(
    tmp_path: Path,
) -> None:
    paths = _runtime_paths(tmp_path)
    cuda_home = tmp_path / "cuda"
    shared_cache = tmp_path / "shared-xdg"
    shared_cache.mkdir()
    environment = _runtime_environment(
        base={
            "HF_HOME": "/shared/hf",
            "XDG_CACHE_HOME": str(shared_cache),
            "TRITON_CACHE_DIR": "/shared/triton",
            "TORCHINDUCTOR_CACHE_DIR": "/shared/inductor",
            "PATH": "/usr/bin",
            "LD_LIBRARY_PATH": "/usr/lib",
            "CC": "gcc",
            "CXX": "g++",
            "CUDAHOSTCXX": "g++",
        },
        paths=paths,
        cuda_home=cuda_home,
        c_compiler="gcc",
        cxx_compiler="g++",
    )

    assert environment["HF_HOME"] == "/shared/hf"
    assert environment["XDG_CACHE_HOME"] == str(shared_cache)
    assert environment["TRITON_CACHE_DIR"] == "/shared/triton"
    assert environment["TORCHINDUCTOR_CACHE_DIR"] == "/shared/inductor"
    assert environment["PATH"].count(str(cuda_home / "bin")) == 1
    assert environment["LD_LIBRARY_PATH"].split(":").count(str(cuda_home / "lib")) == 1


def test_uv_binary_reuse_restores_executable_mode(tmp_path: Path) -> None:
    wheel = tmp_path / "uv.whl"
    binary = b"\x7fELFsynthetic uv binary"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("uv", binary)
    destination = tmp_path / "uv"
    destination.write_bytes(binary)
    destination.chmod(stat.S_IRUSR | stat.S_IWUSR)

    _extract_uv(wheel, destination)

    assert destination.stat().st_mode & stat.S_IXUSR


def test_cuda_driver_link_repairs_dangling_symlink_and_is_repeatable(
    tmp_path: Path,
) -> None:
    from serving.franzen_h200 import _ensure_libcuda_link

    lib = tmp_path / "cuda/lib"
    drivers = tmp_path / "drivers"
    lib.mkdir(parents=True)
    drivers.mkdir()
    driver = drivers / "libcuda.so.1"
    driver.write_text("driver")
    link = lib / "libcuda.so"
    link.symlink_to("missing-driver.so")

    _ensure_libcuda_link(lib, driver_locations=(drivers,))

    assert link.is_symlink()
    assert link.resolve() == driver.resolve()
    _ensure_libcuda_link(lib, driver_locations=(drivers,))
    assert link.resolve() == driver.resolve()


def test_metrics_continue_after_nvidia_smi_process_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    class OneTick(threading.Event):
        def __init__(self) -> None:
            super().__init__()
            self.ticks = 0

        def wait(self, timeout: float | None = None) -> bool:
            del timeout
            self.ticks += 1
            return self.ticks > 1

    monkeypatch.setattr(
        franzen_h200,
        "_query_gpu",
        lambda: (_ for _ in ()).throw(
            subprocess.CalledProcessError(1, ["nvidia-smi"], stderr="driver lost"),
        ),
    )
    path = tmp_path / "metrics.jsonl"

    _record_gpu_metrics(path, OneTick())

    record = json.loads(path.read_text())
    assert record["phase"] == "runtime_error"
    assert "driver lost" in record["error"]


def test_gpu_query_has_a_finite_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    observed: dict[str, object] = {}

    def run(command, **kwargs):
        observed["command"] = command
        observed.update(kwargs)
        return subprocess.CompletedProcess(command, 0, stdout="0, H200, 9.0, 100, 1, 99, 0, 40, 30\n")

    monkeypatch.setattr(franzen_h200.subprocess, "run", run)

    franzen_h200._query_gpu()

    assert observed["timeout"] == franzen_h200.GPU_QUERY_TIMEOUT_SECONDS


def test_metrics_record_nvidia_smi_timeout_and_continue(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    class OneTick(threading.Event):
        def __init__(self) -> None:
            super().__init__()
            self.ticks = 0

        def wait(self, timeout: float | None = None) -> bool:
            del timeout
            self.ticks += 1
            return self.ticks > 1

    monkeypatch.setattr(
        franzen_h200,
        "_query_gpu",
        lambda: (_ for _ in ()).throw(
            subprocess.TimeoutExpired(["nvidia-smi"], 30),
        ),
    )
    path = tmp_path / "metrics.jsonl"

    _record_gpu_metrics(path, OneTick())

    record = json.loads(path.read_text())
    assert record["phase"] == "runtime_error"
    assert "timed out" in record["error"]


@pytest.mark.parametrize(("returncode", "expected"), [(-9, 137), (-15, 143), (0, 0), (2, 2)])
def test_signal_exit_status_is_normalized_for_shell(returncode: int, expected: int) -> None:
    from serving.franzen_h200 import _normalize_server_exit_code

    assert _normalize_server_exit_code(returncode) == expected


def test_setup_failure_reports_bounded_command_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    class FailedProcess:
        def wait(self, *, timeout: float) -> int:
            del timeout
            return 1

        def kill(self) -> None:
            raise AssertionError("failed command should not be killed")

    def popen(_command, *, stdout, **_kwargs):
        stdout.write("stdout evidence\nstderr evidence\n")
        stdout.flush()
        return FailedProcess()

    monkeypatch.setattr(franzen_h200.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="stderr evidence"):
        _run_logged(["false"], {}, paths)


def test_setup_command_timeout_kills_child_after_streaming_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)
    timeouts: list[float] = []

    class TimedOutProcess:
        def wait(self, *, timeout: float) -> int:
            timeouts.append(timeout)
            if len(timeouts) == 1:
                raise subprocess.TimeoutExpired(["synthetic"], timeout)
            return -9

        def kill(self) -> None:
            return None

    process = TimedOutProcess()

    def popen(_command, *, stdout, **_kwargs):
        stdout.write("streamed before timeout\n")
        stdout.flush()
        return process

    monkeypatch.setattr(franzen_h200.subprocess, "Popen", popen)

    with pytest.raises(RuntimeError, match="timed out"):
        _run_logged(["synthetic"], {}, paths)

    assert timeouts == [franzen_h200.SETUP_COMMAND_TIMEOUT_SECONDS, 10]
    assert "streamed before timeout" in paths.log.read_text()


def test_setup_command_keyboard_interrupt_reaps_its_child(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from serving import franzen_h200

    paths = _runtime_paths(tmp_path)

    class InterruptedProcess:
        killed = False
        waited = False

        def wait(self, *, timeout: float) -> int:
            del timeout
            if not self.killed:
                raise KeyboardInterrupt
            self.waited = True
            return -9

        def kill(self) -> None:
            self.killed = True

    process = InterruptedProcess()

    def popen(_command, *, stdout, **_kwargs):
        stdout.write("setup output before interrupt\n")
        stdout.flush()
        return process

    monkeypatch.setattr(franzen_h200.subprocess, "Popen", popen)

    with pytest.raises(KeyboardInterrupt):
        _run_logged(["synthetic"], {}, paths)

    assert process.killed
    assert process.waited


def _option_value(arguments: tuple[str, ...], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _runtime_paths(root: Path) -> RuntimePaths:
    work = root / "work"
    work.mkdir(exist_ok=True)
    return RuntimePaths(
        target_dir=None,
        draft_dir=None,
        wheelhouse=root,
        wheels=root,
        lock=root / "requirements.lock",
        token_map=root / "tokens.pt",
        work_dir=work,
        venv=work / "venv",
        python=work / "venv/bin/python",
        sglang=work / "venv/bin/sglang",
        log=work / "serve.log",
        metadata=work / "run_config.json",
        install_metadata=work / "offline-install.json",
        metrics=work / "gpu-metrics.jsonl",
        pid_file=work / "server.pid",
        lock_file=work / ".serve.lock",
    )
