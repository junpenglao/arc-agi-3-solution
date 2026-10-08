"""Launch the pinned Franzen SGLang server on one H200."""

from __future__ import annotations

import argparse
import csv
import email.parser
import hashlib
import json
import math
import os
import platform
import queue
import re
import shlex
import shutil
import signal
import socket
import stat
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol, cast


NOTEBOOK_SOURCE_SHA256 = "25879d2fee20cbf91a4ccd684477ad5db69f0d2e36d81a1de1dd49bb68e4210d"
TOKEN_MAP_SHA256 = "becfa41d394b86c26c632bea8f3c6ea64bbb76d7b238d8673c06afae21269f25"
TOKENIZER_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
SGLANG_REVISION = "d00d88efc8d6281b12be4f4073126aec95038c55"
SGLANG_VERSION = "0.5.19+gd00d88efc8d6"
EXPERT_TARGET = (
    r"re:^mtp\.layers\.0\.mlp\.experts\.[0-9]+\.(gate_proj|up_proj|down_proj)$"
)
DENSE_IGNORE = r"re:^(?!mtp\.layers\.0\.mlp\.experts(?:\.|$)).*"
GPU_QUERY = (
    "index,name,compute_cap,memory.total,memory.used,memory.free,"
    "utilization.gpu,power.draw,temperature.gpu"
)
PRECACHE_BLOCK_BYTES = 32 * 1024 * 1024
GPU_STATIC_MEMORY_FRACTION = 0.96
QSA_FP8_PATCH_BEFORE_SHA256 = "2ce24d66d6a0bff0e22ff0819291649a4169937f32c5c28241215c6eb119ad54"
QSA_FP8_PATCH_AFTER_SHA256 = "e5e08c37c603b2977d4b93ce1395185bc595be029ed5cb84ee976f3acb221d2c"
_QSA_FORWARD_EXTEND_ANCHOR = "    def forward_extend(\n"
_QSA_FORWARD_EXTEND_REPLACEMENT = '''    @staticmethod
    def _require_unit_qsa_kv_scales(kwargs):
        for name in ("k_scale", "v_scale"):
            scale = kwargs.get(name, 1.0)
            if type(scale) not in (int, float) or scale != 1.0:
                raise ValueError(
                    "SM90 QSA FP8 compute scratch supports unit K/V cache scales only"
                )

    def forward_extend(
'''
_QSA_STORE_ANCHOR = '''        if save_kv_cache:
            self.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )
'''
_QSA_STORE_REPLACEMENT = '''        if save_kv_cache:
            self._require_unit_qsa_kv_scales(kwargs)
            self.token_to_kv_pool.set_kv_buffer(
                layer,
                forward_batch.out_cache_loc,
                k,
                v,
                k_scale=1.0,
                v_scale=1.0,
            )
'''
_QSA_SCRATCH_ANCHOR = '''        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            k_buffer.dtype,
            k_buffer.device,
        )
'''
_QSA_SCRATCH_REPLACEMENT = '''        packed_k, packed_v = self._get_fa2_scratch(
            scratch_capacity,
            k_buffer.shape[1],
            k_buffer.shape[2],
            q.dtype if k_buffer.dtype == torch.float8_e4m3fn else k_buffer.dtype,
            k_buffer.device,
        )
'''


class Flags(Protocol):
    """Parsed launcher arguments."""

    target_dir: Path | None
    draft_dir: Path | None
    wheelhouse: Path
    work_dir: Path
    port: int | None
    startup_timeout: int
    notebook_start_epoch: float | None
    prepare_only: bool
    qsa_fp8_compute_scratch: bool
    bind_host: str
    linear_attn_verify_backend: Literal["triton"] | None


@dataclass(frozen=True, slots=True, kw_only=True)
class RuntimePaths:
    """Resolved paths owned by one server launch."""

    target_dir: Path | None
    draft_dir: Path | None
    wheelhouse: Path
    wheels: Path
    lock: Path
    token_map: Path
    work_dir: Path
    venv: Path
    python: Path
    sglang: Path
    log: Path
    metadata: Path
    install_metadata: Path
    metrics: Path
    pid_file: Path


def build_server_argv(
    *,
    sglang: Path,
    model_dir: Path,
    draft_view: Path,
    token_map: Path,
    port: int,
    chat_template: Path | None,
    bind_host: str = "127.0.0.1",
    linear_attn_verify_backend: Literal["triton"] | None = None,
) -> tuple[str, ...]:
    """Build the notebook-equivalent command without changing model controls."""
    graph_bs = (1, 2, 4, 7, 8, 9, 10)
    arguments = [
        str(sglang),
        "serve",
        "--model-path",
        str(model_dir),
        "--load-format",
        "safetensors",
        "--model-loader-extra-config",
        '{"enable_multithread_load":false}',
        "--served-model-name",
        "flashnext",
        "--host",
        bind_host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "bfloat16",
        "--quantization",
        "auto-round",
        "--kv-cache-dtype",
        "fp8_e4m3",
        "--mem-fraction-static",
        "0.96",
        "--context-length",
        "139264",
        "--page-size",
        "64",
        "--max-running-requests",
        "10",
        "--chunked-prefill-size",
        "8192",
        "--max-prefill-tokens",
        "16384",
        "--cuda-graph-max-bs-decode",
        "10",
        "--cuda-graph-bs-decode",
        *map(str, graph_bs),
        "--mamba-ssm-dtype",
        "bfloat16",
        "--max-mamba-cache-size",
        "60",
        "--mamba-radix-cache-strategy",
        "extra_buffer",
        "--mamba-track-interval",
        "64",
        "--mamba-backend",
        "flashinfer",
        "--linear-attn-decode-backend",
        "flashinfer",
        "--linear-attn-prefill-backend",
        "flashinfer",
        "--moe-runner-backend",
        "auto",
        "--ple-offload-embedding",
        "--trust-remote-code",
        "--mm-feature-transport",
        "cpu",
        "--image-processor-backend",
        "pil",
        "--reasoning-parser",
        "qwen3",
        "--tool-call-parser",
        "qwen3_coder",
        "--default-chat-template-kwargs",
        '{"preserve_thinking":true}',
        "--watchdog-timeout",
        "1800",
        "--schedule-policy",
        "lpm",
        "--warmups",
        "structured_output",
        "--enable-cache-report",
        "--enable-metrics",
        "--enable-request-time-stats-logging",
        "--weight-loader-prefetch-checkpoints",
        "--weight-loader-drop-cache-after-load",
        "--gdn-mtp-cache-mode",
        "none",
        "--speculative-algorithm",
        "NEXTN",
        "--speculative-num-steps",
        "3",
        "--speculative-eagle-topk",
        "1",
        "--speculative-num-draft-tokens",
        "4",
        "--speculative-draft-model-path",
        str(draft_view),
        "--speculative-draft-model-quantization",
        "compressed-tensors",
        "--speculative-moe-runner-backend",
        "auto",
        "--speculative-draft-kv-cache-dtype",
        "fp8_e4m3",
        "--speculative-accept-threshold-single",
        "1.0",
        "--speculative-accept-threshold-acc",
        "1.0",
        "--speculative-token-map",
        str(token_map),
    ]
    if chat_template is not None:
        arguments.extend(("--chat-template", str(chat_template)))
    if linear_attn_verify_backend is not None:
        arguments.extend(
            ("--linear-attn-verify-backend", linear_attn_verify_backend),
        )
    return tuple(arguments)


def h200_environment(*, base: dict[str, str]) -> dict[str, str]:
    """Route build/JIT targets to SM90 and explicitly disable SM120-only paths."""
    environment = dict(base)
    environment.pop("PYTHONPATH", None)
    environment.update(
        {
            "PYTHONNOUSERSITE": "1",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "UV_OFFLINE": "1",
            "UV_PYTHON_DOWNLOADS": "never",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "TORCH_CUDA_ARCH_LIST": "9.0",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "TOKENIZERS_PARALLELISM": "false",
            "MAX_JOBS": "8",
            "CMAKE_BUILD_PARALLEL_LEVEL": "8",
            "FLASHINFER_NINJA_JOBS": "8",
            "FLASHINFER_NVCC_THREADS": "2",
            "TORCHINDUCTOR_COMPILE_THREADS": "8",
            "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM": "0",
            "SGLANG_SM120_ONLINE_MXFP8": "0",
            "SGLANG_SM120_LOWM_FP8_WEIGHT": "0",
            "SGLANG_SM120_LM_HEAD_FP8": "0",
            "SGLANG_MAMBA_CONV_DTYPE": "bfloat16",
            "SGLANG_NUMA_BIND_V2": "false",
            "SGLANG_MM_PREPROCESS_DEVICE": "cpu",
            "NUMPY_MADVISE_HUGEPAGE": "0",
        },
    )
    return environment


def offline_install_commands(
    *,
    uv: Path,
    python: Path,
    venv: Path,
    wheels: Path,
    lock: Path,
    sglang_wheel: Path,
) -> tuple[tuple[str, ...], ...]:
    """Build the notebook's local-wheel-only venv/install command sequence."""
    target_python = venv / "bin/python"
    install = (
        str(uv),
        "pip",
        "install",
        "--python",
        str(target_python),
        "--no-index",
        "--find-links",
        str(wheels),
    )
    return (
        (str(uv), "venv", "--python", str(python), str(venv)),
        (*install, "-r", str(lock)),
        (*install, "--reinstall", "--no-deps", str(sglang_wheel)),
    )


def parse_gpu_metrics(output: str) -> list[dict[str, object]]:
    """Parse nvidia-smi rows and require exactly one H200 at capability 9.0."""
    rows: list[dict[str, object]] = []
    for fields in csv.reader(output.splitlines(), skipinitialspace=True):
        if len(fields) != 9:
            raise ValueError(f"Unexpected nvidia-smi row: {fields}")
        index, name, capability, total, used, free, utilization, power, temperature = (
            field.strip() for field in fields
        )
        try:
            row: dict[str, object] = {
                "index": int(index),
                "name": name,
                "compute_capability": capability,
                "memory_total_mib": int(total),
                "memory_used_mib": int(used),
                "memory_free_mib": int(free),
                "utilization_percent": _number_or_none(utilization),
                "power_watts": _number_or_none(power),
                "temperature_celsius": _number_or_none(temperature),
            }
        except ValueError as error:
            raise ValueError(f"Invalid nvidia-smi row: {fields}") from error
        rows.append(row)
    if len(rows) != 1 or rows[0]["compute_capability"] != "9.0" or "H200" not in cast("str", rows[0]["name"]):
        raise ValueError(
            "Expected exactly one NVIDIA H200 with compute capability 9.0; "
            f"found {[(row['name'], row['compute_capability']) for row in rows]}.",
        )
    return rows


def prepare_draft_view(
    *,
    target: Path,
    source_root: Path,
    work_root: Path,
) -> Path:
    """Create the notebook's metadata-only compressed-tensors draft view."""
    target = target.resolve()
    source_root = source_root.resolve()
    configs = ([source_root / "config.json"] if (source_root / "config.json").is_file() else [])
    configs.extend(
        path
        for path in sorted(source_root.rglob("config.json"))
        if path not in configs
    )
    candidates: list[tuple[Path, dict[str, object]]] = []
    for config_path in configs:
        config = _read_object(config_path)
        quantization_value = config.get("quantization_config")
        if not isinstance(quantization_value, dict):
            continue
        quantization = cast("dict[str, object]", quantization_value)
        group_value = quantization.get("config_groups")
        if not isinstance(group_value, dict):
            continue
        groups = cast("dict[str, object]", group_value)
        if (
            quantization.get("quant_method") == "compressed-tensors"
            and "mtp_routed_experts"
            in groups
        ):
            candidates.append((config_path.parent, config))
    if len(candidates) != 1:
        raise ValueError(
            "Expected one INT4 g32 MTP checkpoint under "
            f"{source_root}; found {len(candidates)}.",
        )
    source, config = candidates[0]
    index, shards = _indexed_shards(source)
    if not any(name.startswith("mtp.layers.0.mlp.experts.") for name in index):
        raise ValueError("Draft index does not contain the expected MTP experts")
    quantization = _object(config["quantization_config"])
    groups = _object(quantization["config_groups"])
    group = _object(groups["mtp_routed_experts"])
    weights = _object(group["weights"])
    if (
        weights.get("num_bits"),
        weights.get("group_size"),
        weights.get("symmetric"),
    ) != (4, 32, True):
        raise ValueError("Expected symmetric INT4 group32 draft experts")
    if group.get("targets") not in (["RoutedExperts"], [EXPERT_TARGET]):
        raise ValueError(f"Unexpected draft target rules: {group.get('targets')}")
    if quantization.get("ignore", []) not in ([], [DENSE_IGNORE]):
        raise ValueError("Unexpected draft ignore rules")
    adapted = json.loads(json.dumps(config))
    adapted_quantization = _object(adapted["quantization_config"])
    adapted_groups = _object(adapted_quantization["config_groups"])
    adapted_group = _object(adapted_groups["mtp_routed_experts"])
    adapted_group["targets"] = [EXPERT_TARGET]
    adapted_quantization["ignore"] = [DENSE_IGNORE]
    identity = json.dumps(
        {
            "source": str(source.resolve()),
            "target": str(target),
            "config": adapted,
        },
        sort_keys=True,
    )
    suffix = hashlib.sha256(identity.encode()).hexdigest()[:16]
    view = work_root / f"draft-view-{suffix}"
    view.mkdir(parents=True, exist_ok=True)
    config_path = view / "config.json"
    config_text = json.dumps(adapted, indent=2) + "\n"
    if config_path.exists() or config_path.is_symlink():
        if config_path.is_symlink() or config_path.read_text() != config_text:
            raise ValueError(f"Unexpected existing draft-view config: {config_path}")
    else:
        config_path.write_text(config_text)
    links = {name: source / name for name in shards}
    links["model.safetensors.index.json"] = source / "model.safetensors.index.json"
    for filename in (
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "generation_config.json",
    ):
        if (target / filename).is_file():
            links[filename] = target / filename
    for filename, source_path in links.items():
        destination = view / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink() or destination.resolve() != source_path.resolve():
                raise ValueError(f"Unexpected existing draft-view file: {destination}")
        else:
            destination.symlink_to(source_path.resolve())
    return view


def main(argv: list[str] | None = None) -> int:
    """Run the H200 serving launcher; return zero only after health is ready."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    _add_arguments(parser)
    flags = cast(Flags, parser.parse_args(argv))
    paths = _resolve_paths(flags)
    paths.work_dir.mkdir(parents=True, exist_ok=True)
    runtime_environment = h200_environment(base=dict(os.environ))
    if flags.prepare_only:
        if flags.qsa_fp8_compute_scratch:
            raise ValueError("QSA source patches apply only to a serve launch")
        _precache_paths(
            (paths.wheelhouse,),
            threads=16,
            log=paths.work_dir / "precache.log",
        )
        receipt = _setup_venv(paths, runtime_environment)
        print(
            f"Offline server venv {receipt['mode']}: {paths.venv}; "
            f"setup receipt: {paths.install_metadata}",
        )
        return 0
    if paths.target_dir is None or paths.draft_dir is None or flags.port is None:
        raise ValueError(
            "Serving requires --target-dir, --draft-dir and --port; "
            "use --prepare-only for CPU venv setup.",
        )
    started = (
        flags.notebook_start_epoch
        if flags.notebook_start_epoch is not None
        else time.time()
    )
    _require_unoccupied_port(flags.bind_host, flags.port)
    setup_started = time.time()
    target_config = _read_object(paths.target_dir / "config.json")
    quantization = _object(target_config.get("quantization_config"))
    if quantization.get("quant_method") != "auto-round" or quantization.get("bits") != 4:
        raise ValueError("Target must be the pinned AutoRound INT4 checkpoint")
    _, target_shards = _indexed_shards(paths.target_dir)
    token_map_digest, tokenizer_digest = validate_fr_spec_assets(
        token_map=paths.token_map,
        tokenizer=paths.target_dir / "tokenizer.json",
    )
    draft_view = prepare_draft_view(
        target=paths.target_dir,
        source_root=paths.draft_dir,
        work_root=paths.work_dir,
    )
    gpu = _query_gpu()
    admission = validate_gpu_admission(gpu, environment=os.environ)
    _append_jsonl(
        paths.metrics,
        {"utc": _utc_now(), "phase": "preflight", "gpus": gpu, "admission": admission},
    )
    _precache_paths((paths.wheelhouse,), threads=16, log=paths.work_dir / "precache.log")
    install_receipt = _setup_venv(paths, runtime_environment)
    qsa_source = _qsa_source_file(paths.venv)
    qsa_patch = apply_qsa_fp8_scratch_patch(
        qsa_source,
        enabled=flags.qsa_fp8_compute_scratch,
    )
    cuda_home, c_compiler, cxx_compiler = _prepare_cuda(paths)
    runtime_environment = _runtime_environment(
        base=runtime_environment,
        paths=paths,
        cuda_home=cuda_home,
        c_compiler=c_compiler,
        cxx_compiler=cxx_compiler,
    )
    _run_logged([str(cuda_home / "bin/nvcc"), "--version"], runtime_environment, paths)
    setup_completed = time.time()
    chat_template = paths.target_dir / "chat_template.jinja"
    server_argv = build_server_argv(
        sglang=paths.sglang,
        model_dir=paths.target_dir,
        draft_view=draft_view,
        token_map=paths.token_map,
        port=flags.port,
        chat_template=chat_template if chat_template.is_file() else None,
        bind_host=flags.bind_host,
        linear_attn_verify_backend=flags.linear_attn_verify_backend,
    )
    _write_launch_metadata(
        paths=paths,
        argv=server_argv,
        environment=runtime_environment,
        draft_view=draft_view,
        started=started,
        startup_timeout=flags.startup_timeout,
        token_map_digest=token_map_digest,
        tokenizer_digest=tokenizer_digest,
        target_shards=len(target_shards),
        install_receipt=install_receipt,
        gpu_metrics=gpu,
        admission=admission,
        setup_started=setup_started,
        setup_completed=setup_completed,
        target_config_sha256=_sha256(paths.target_dir / "config.json"),
        target_index_sha256=_sha256(paths.target_dir / "model.safetensors.index.json"),
        draft_config_sha256=_sha256(draft_view / "config.json"),
        draft_index_sha256=_sha256(draft_view / "model.safetensors.index.json"),
        linear_attn_verify_backend=flags.linear_attn_verify_backend,
        qsa_patch=qsa_patch,
    )
    precache_cancel = threading.Event()
    model_precache = threading.Thread(
        target=_precache_paths,
        kwargs={
            "paths": (paths.target_dir, paths.draft_dir),
            "threads": 3,
            "delay_seconds": 60,
            "log": paths.work_dir / "precache.log",
            "cancellation": precache_cancel,
        },
        name="franzen-h200-model-precache",
        daemon=True,
    )
    model_precache.start()
    cancellation = threading.Event()
    previous_handlers = {
        signum: signal.getsignal(signum)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    for signum in previous_handlers:
        signal.signal(signum, lambda _signum, _frame: cancellation.set())
    try:
        return _launch_and_wait(
            paths=paths,
            argv=server_argv,
            environment=runtime_environment,
            notebook_start_epoch=started,
            startup_timeout=flags.startup_timeout,
            precache_cancel=precache_cancel,
            cancellation=cancellation,
            bind_host=flags.bind_host,
        )
    finally:
        precache_cancel.set()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


def validate_fr_spec_assets(*, token_map: Path, tokenizer: Path) -> tuple[str, str]:
    """Verify the pinned FR-Spec map/tokenizer pair and return both digests."""
    token_map_digest = _sha256(token_map)
    tokenizer_digest = _sha256(tokenizer)
    if token_map_digest != TOKEN_MAP_SHA256:
        raise ValueError("The hot-token map differs from the pinned Pennyroyal map")
    if tokenizer_digest != TOKENIZER_SHA256:
        raise ValueError("The target tokenizer differs from the pinned FR-Spec tokenizer")
    return token_map_digest, tokenizer_digest


def apply_qsa_fp8_scratch_patch(path: Path, *, enabled: bool) -> dict[str, object]:
    """Apply only the source-pinned QSA SM90 FP8-gather patch when selected."""
    source = path.read_text()
    before_sha256 = _sha256(path)
    if before_sha256 == QSA_FP8_PATCH_AFTER_SHA256:
        if not enabled:
            raise ValueError("QSA source is patched; the QSA compatibility selector is required")
        return {
            "enabled": True,
            "status": "already_applied",
            "path": str(path),
            "before_sha256": QSA_FP8_PATCH_BEFORE_SHA256,
            "after_sha256": QSA_FP8_PATCH_AFTER_SHA256,
        }
    if before_sha256 != QSA_FP8_PATCH_BEFORE_SHA256:
        raise ValueError(
            f"QSA source hash differs from pinned base: {before_sha256}",
        )
    receipt: dict[str, object] = {
        "enabled": enabled,
        "status": "unpatched",
        "path": str(path),
        "before_sha256": before_sha256,
        "after_sha256": before_sha256,
    }
    if not enabled:
        return receipt
    patched_source = _apply_qsa_patch_text(source)
    after_sha256 = hashlib.sha256(patched_source.encode()).hexdigest()
    if after_sha256 != QSA_FP8_PATCH_AFTER_SHA256:
        raise RuntimeError(
            "QSA patch output hash differs from its reviewed after seal: "
            f"{after_sha256}",
        )
    temporary = path.with_name(f".{path.name}.qsa-patch.pending")
    file_mode = stat.S_IMODE(path.stat().st_mode)
    try:
        temporary.write_text(patched_source)
        temporary.chmod(file_mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    if _sha256(path) != after_sha256:
        raise RuntimeError("QSA source did not retain the reviewed after seal")
    return {
        "enabled": True,
        "status": "applied",
        "path": str(path),
        "before_sha256": before_sha256,
        "after_sha256": after_sha256,
    }


def _apply_qsa_patch_text(source: str) -> str:
    replacements = (
        (_QSA_FORWARD_EXTEND_ANCHOR, _QSA_FORWARD_EXTEND_REPLACEMENT, 1),
        (_QSA_STORE_ANCHOR, _QSA_STORE_REPLACEMENT, 2),
        (_QSA_SCRATCH_ANCHOR, _QSA_SCRATCH_REPLACEMENT, 1),
    )
    for original, replacement, expected_count in replacements:
        count = source.count(original)
        if count != expected_count:
            raise ValueError(
                "QSA source patch anchor count differs: "
                f"expected {expected_count}, found {count}",
            )
        source = source.replace(original, replacement)
    return source


def _qsa_source_file(venv: Path) -> Path:
    candidates = sorted(
        venv.glob(
            "lib/python*/site-packages/sglang/srt/layers/attention/"
            "qwen_sparse_attn_backend.py",
        ),
    )
    if len(candidates) != 1:
        raise ValueError(
            "Expected one installed qwen_sparse_attn_backend.py, "
            f"found {len(candidates)} under {venv}",
        )
    return candidates[0]


def validate_gpu_admission(
    gpus: list[dict[str, object]],
    *,
    environment: dict[str, str],
) -> dict[str, object]:
    """Require one H200 in one active Slurm allocation with launch headroom."""
    job_id = environment.get("SLURM_JOB_ID", "")
    allocation = environment.get("SLURM_STEP_GPUS") or environment.get("SLURM_JOB_GPUS", "")
    allocated = [device for device in allocation.split(",") if device.strip()]
    if not job_id or len(allocated) != 1:
        raise RuntimeError("Server admission requires one GPU in an active Slurm job")
    if len(gpus) != 1 or not isinstance(gpus[0].get("memory_total_mib"), int) or not isinstance(gpus[0].get("memory_free_mib"), int):
        raise ValueError("Server admission requires one measured H200 memory row")
    gpu = gpus[0]
    memory_total = cast("int", gpu["memory_total_mib"])
    memory_free = cast("int", gpu["memory_free_mib"])
    minimum_free = math.ceil(memory_total * GPU_STATIC_MEMORY_FRACTION)
    if memory_free < minimum_free:
        raise RuntimeError(
            f"H200 has {memory_free} MiB free; launch requires {minimum_free} MiB "
            f"from the pinned {GPU_STATIC_MEMORY_FRACTION:.0%} static memory fraction",
        )
    return {
        "slurm_job_id": job_id,
        "allocated_gpu_ids": allocated,
        "selected_gpu": gpu,
        "minimum_free_memory_mib": minimum_free,
        "free_memory_threshold_fraction": GPU_STATIC_MEMORY_FRACTION,
    }


def _precache_paths(
    paths: tuple[Path, ...],
    *,
    threads: int,
    log: Path,
    delay_seconds: int = 0,
    cancellation: threading.Event | None = None,
) -> None:
    cancellation = cancellation or threading.Event()
    if cancellation.wait(delay_seconds):
        _append_jsonl(
            log,
            {"utc": _utc_now(), "phase": "precache", "cancelled": True},
        )
        return
    files: list[tuple[Path, int, int]] = []
    total_bytes = 0
    for path in paths:
        if path.is_file():
            candidates = (path,)
        else:
            candidates = (
                Path(root) / name
                for root, _, names in os.walk(path)
                for name in names
            )
        for candidate in candidates:
            try:
                size = candidate.stat().st_size
            except OSError:
                continue
            total_bytes += size
            files.extend(
                (candidate, offset, min(PRECACHE_BLOCK_BYTES, size - offset))
                for offset in range(0, max(size, 1), PRECACHE_BLOCK_BYTES)
            )
    started = time.monotonic()
    pending: queue.Queue[tuple[Path, int, int]] = queue.Queue()
    for chunk in files:
        pending.put(chunk)
    errors: list[str] = []

    def read_chunks() -> None:
        while not cancellation.is_set():
            try:
                chunk = pending.get(timeout=0.1)
            except queue.Empty:
                return
            error = _read_cache_chunk(chunk)
            if error is not None:
                errors.append(error)
            pending.task_done()

    workers = [
        threading.Thread(target=read_chunks, name="asset-precache", daemon=True)
        for _ in range(threads)
    ]
    for worker in workers:
        worker.start()
    while not cancellation.is_set() and any(worker.is_alive() for worker in workers):
        for worker in workers:
            worker.join(timeout=0.1)
    _append_jsonl(
        log,
        {
            "utc": _utc_now(),
            "phase": "precache",
            "paths": [str(path) for path in paths],
            "files": len({path for path, _, _ in files}),
            "bytes": total_bytes,
            "threads": threads,
            "elapsed_seconds": time.monotonic() - started,
            "errors": errors,
            "cancelled": cancellation.is_set(),
        },
    )


def _read_cache_chunk(chunk: tuple[Path, int, int]) -> str | None:
    path, offset, length = chunk
    try:
        with path.open("rb") as source:
            source.seek(offset)
            remaining = length
            while remaining:
                data = source.read(min(8 * 1024 * 1024, remaining))
                if not data:
                    break
                remaining -= len(data)
    except OSError as error:
        return f"{path}: {error}"
    return None


def _add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the launcher's explicit local inputs."""
    parser.add_argument("--target-dir", type=Path)
    parser.add_argument("--draft-dir", type=Path)
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--port", type=int)
    parser.add_argument("--startup-timeout", type=int, default=720)
    parser.add_argument("--notebook-start-epoch", type=float)
    parser.add_argument("--bind-host", default="127.0.0.1")
    parser.add_argument(
        "--linear-attn-verify-backend",
        choices=("triton",),
        default=None,
    )
    parser.add_argument("--qsa-fp8-compute-scratch", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")


def _resolve_paths(flags: Flags) -> RuntimePaths:
    target_dir = (
        flags.target_dir.expanduser().resolve(strict=True)
        if flags.target_dir is not None
        else None
    )
    draft_dir = (
        flags.draft_dir.expanduser().resolve(strict=True)
        if flags.draft_dir is not None
        else None
    )
    wheelhouse = flags.wheelhouse.expanduser().resolve(strict=True)
    work_dir = flags.work_dir.expanduser().resolve()
    wheels = _find_unique(wheelhouse, "wheels", directory=True)
    lock = _find_unique(wheelhouse, "requirements.lock")
    candidates = sorted(wheels.glob("sglang-*.whl"))
    if len(candidates) != 1:
        raise ValueError(f"Expected one SGLang wheel, found {candidates}")
    token_map = _find_optional(wheelhouse, "hot_tokens_64k.pt")
    if token_map is None:
        token_map = _find_unique(wheelhouse, "flash-next-64k.pt")
    venv = work_dir / "venv"
    return RuntimePaths(
        target_dir=target_dir,
        draft_dir=draft_dir,
        wheelhouse=wheelhouse,
        wheels=wheels,
        lock=lock,
        token_map=token_map,
        work_dir=work_dir,
        venv=venv,
        python=venv / "bin/python",
        sglang=venv / "bin/sglang",
        log=work_dir / "serve.log",
        metadata=work_dir / "run_config.json",
        install_metadata=work_dir / "offline-install.json",
        metrics=work_dir / "gpu-metrics.jsonl",
        pid_file=work_dir / "server.pid",
    )


def _setup_venv(
    paths: RuntimePaths,
    environment: dict[str, str],
) -> dict[str, object]:
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("The Pennyroyal wheelhouse requires Linux x86_64")
    if sys.version_info < (3, 10):
        raise RuntimeError("The Pennyroyal wheelhouse requires CPython >=3.10")
    uv_wheels = sorted(paths.wheels.glob("uv-*.whl"))
    sglang_wheels = sorted(paths.wheels.glob("sglang-*.whl"))
    if len(uv_wheels) != 1 or len(sglang_wheels) != 1:
        raise ValueError("Expected exactly one bundled uv wheel and SGLang wheel")
    wheel_version = _sglang_wheel_version(sglang_wheels[0])
    uv = paths.work_dir / "bin/uv"
    uv.parent.mkdir(parents=True, exist_ok=True)
    _extract_uv(uv_wheels[0], uv)
    marker = paths.work_dir / "installed-bundle.json"
    identity = {
        "uv_wheel_sha256": _sha256(uv_wheels[0]),
        "sglang_wheel_sha256": _sha256(sglang_wheels[0]),
        "requirements_lock_sha256": _sha256(paths.lock),
        "python": platform.python_version(),
        "sglang_version": wheel_version,
        "sglang_revision": SGLANG_REVISION,
    }
    if paths.python.is_file() and paths.sglang.is_file() and marker.is_file():
        if _read_object(marker) != identity:
            raise RuntimeError(
                f"Existing offline venv identity differs; use a fresh --work-dir: {paths.venv}",
            )
        mode = "reused"
    else:
        if paths.venv.exists() or marker.exists():
            raise RuntimeError(
                f"Incomplete or unverified offline venv; use a fresh --work-dir: {paths.venv}",
            )
        mode = "installed"
    commands = offline_install_commands(
        uv=uv,
        python=Path(sys.executable),
        venv=paths.venv,
        wheels=paths.wheels,
        lock=paths.lock,
        sglang_wheel=sglang_wheels[0],
    )
    if mode == "installed":
        for command in commands:
            _run_logged(list(command), environment, paths)
        _run_logged([str(paths.python), "-m", "pip", "check"], environment, paths)
        _verify_installed_sglang(paths, environment)
        _write_json(marker, identity)
    else:
        _verify_installed_sglang(paths, environment)
    if mode == "installed":
        receipt: dict[str, object] = {
            "mode": "installed",
            "offline": True,
            "installed_utc": _utc_now(),
            "identity": identity,
            "commands": [list(command) for command in commands]
            + [[str(paths.python), "-m", "pip", "check"]],
            "installer": str(uv),
            "installer_sha256": _sha256(uv),
            "marker": str(marker),
        }
        if paths.install_metadata.exists():
            raise RuntimeError(f"Refusing to overwrite install receipt: {paths.install_metadata}")
        _write_json(paths.install_metadata, receipt)
        return receipt
    install_receipt = _read_object(paths.install_metadata)
    if install_receipt.get("identity") != identity:
        raise RuntimeError("Immutable install receipt differs from the installed venv")
    return {
        "mode": "reused",
        "install_receipt": str(paths.install_metadata),
        "install_receipt_sha256": _sha256(paths.install_metadata),
        "identity": identity,
    }


def _sglang_wheel_version(wheel: Path) -> str:
    with zipfile.ZipFile(wheel) as archive:
        metadata_files = [
            name
            for name in archive.namelist()
            if name.count("/") == 1 and name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError(f"Expected one SGLang wheel METADATA file in {wheel}")
        metadata = email.parser.BytesParser().parsebytes(
            archive.read(metadata_files[0]),
        )
    name = re.sub(r"[-_.]+", "-", metadata.get("Name", "")).lower()
    version = metadata.get("Version", "")
    if name != "sglang" or version != SGLANG_VERSION:
        raise ValueError(
            f"Expected SGLang {SGLANG_VERSION}, found {name} {version} in {wheel}",
        )
    if not version.endswith(f"+g{SGLANG_REVISION[:12]}"):
        raise ValueError("SGLang wheel version does not identify the pinned source revision")
    return version


def _verify_installed_sglang(paths: RuntimePaths, environment: dict[str, str]) -> None:
    output = _run_logged(
        [str(paths.python), "-m", "pip", "show", "sglang"],
        environment,
        paths,
    )
    version = next(
        (line.partition(":")[2].strip() for line in output.splitlines() if line.startswith("Version:")),
        "",
    )
    if version != SGLANG_VERSION or not version.endswith(f"+g{SGLANG_REVISION[:12]}"):
        raise RuntimeError(
            f"Installed SGLang differs from pinned {SGLANG_VERSION}/{SGLANG_REVISION}: {version}",
        )


def _extract_uv(wheel: Path, destination: Path) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = [
            name
            for name in archive.namelist()
            if Path(name).name == "uv" and not name.endswith("/")
        ]
        if len(members) != 1:
            raise ValueError("Cannot identify one bundled uv executable")
        binary = archive.read(members[0])
    if not binary.startswith(b"\x7fELF"):
        raise ValueError("The bundled uv executable is not a Linux ELF binary")
    if destination.exists():
        if _sha256(destination) != hashlib.sha256(binary).hexdigest():
            raise ValueError(f"Unexpected existing uv executable: {destination}")
        destination.chmod(0o755)
        return
    temporary = destination.with_name(f".{destination.name}.pending")
    try:
        temporary.write_bytes(binary)
        temporary.chmod(0o755)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_cuda(paths: RuntimePaths) -> tuple[Path, str, str]:
    candidates = sorted(paths.venv.glob("lib/python*/site-packages/nvidia/cu13"))
    if len(candidates) != 1 or not (candidates[0] / "bin/nvcc").is_file():
        raise RuntimeError("Bundled CUDA 13 nvcc is missing from the offline venv")
    cuda_home = candidates[0]
    lib = cuda_home / "lib"
    lib64 = cuda_home / "lib64"
    if not lib64.exists() and not lib64.is_symlink():
        lib64.symlink_to("lib")
    for shared_object in sorted(lib.glob("*.so.*")):
        link = lib / re.sub(r"\.so\..*$", ".so", shared_object.name)
        if not link.exists() and not link.is_symlink():
            link.symlink_to(shared_object.name)
    if not (lib / "libcudart.so").exists():
        raise RuntimeError("Bundled CUDA runtime library libcudart.so is missing")
    if not (lib / "libcuda.so").exists():
        locations = (
            "/usr/local/nvidia/lib64",
            "/usr/local/nvidia/lib",
            "/usr/lib/x86_64-linux-gnu",
            "/usr/lib64",
        )
        drivers = [
            path
            for location in locations
            for path in sorted(Path(location).glob("libcuda.so*"))
            if path.is_file()
        ]
        if not drivers:
            raise RuntimeError("NVIDIA driver libcuda.so was not found in standard locations")
        (lib / "libcuda.so").symlink_to(drivers[0].resolve())
    compiler = next(
        (
            name
            for name in ("g++-15", "g++-14", "g++-13", "g++-12", "g++-11", "g++")
            if shutil.which(name)
        ),
        None,
    )
    if compiler is None:
        raise RuntimeError("Host C++ compiler not found")
    c_compiler = compiler.replace("g++", "gcc")
    if shutil.which(c_compiler) is None:
        raise RuntimeError(f"Matching C compiler missing: {c_compiler}")
    return cuda_home, c_compiler, compiler


def _runtime_environment(
    *,
    base: dict[str, str],
    paths: RuntimePaths,
    cuda_home: Path,
    c_compiler: str,
    cxx_compiler: str,
) -> dict[str, str]:
    environment = dict(base)
    cache_root = environment.get("XDG_CACHE_HOME")
    if not cache_root:
        raise RuntimeError("Bind the provisioned shared XDG_CACHE_HOME before serving")
    cache = Path(cache_root)
    shared_cache_defaults = {
        "HF_HOME": cache / "huggingface",
        "TORCH_HOME": cache / "torch",
        "CUDA_CACHE_PATH": cache / "cuda",
        "TORCHINDUCTOR_CACHE_DIR": cache / "torchinductor",
        "TRITON_CACHE_DIR": cache / "triton",
        "FLASHINFER_WORKSPACE_BASE": cache / "flashinfer",
        "SGLANG_CACHE_DIR": cache / "sglang",
        "SGLANG_JIT_CACHE_DIR": cache / "sglang/jit",
    }
    for name, default in shared_cache_defaults.items():
        if not environment.get(name):
            environment[name] = str(default)
    environment.update(
        {
            "CUDA_HOME": str(cuda_home),
            "CUDA_PATH": str(cuda_home),
            "CUDACXX": str(cuda_home / "bin/nvcc"),
            "PATH": f"{cuda_home / 'bin'}:{paths.venv / 'bin'}:{environment.get('PATH', '')}",
            "LD_LIBRARY_PATH": (
                f"{cuda_home / 'lib64'}:{cuda_home / 'lib'}:"
                + environment.get("LD_LIBRARY_PATH", "")
            ),
            "CC": c_compiler,
            "CXX": cxx_compiler,
            "CUDAHOSTCXX": cxx_compiler,
            "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
        },
    )
    return environment


def _write_launch_metadata(
    *,
    paths: RuntimePaths,
    argv: tuple[str, ...],
    environment: dict[str, str],
    draft_view: Path,
    started: float,
    startup_timeout: int,
    token_map_digest: str,
    tokenizer_digest: str,
    target_shards: int,
    install_receipt: dict[str, object],
    gpu_metrics: list[dict[str, object]],
    admission: dict[str, object],
    setup_started: float,
    setup_completed: float,
    target_config_sha256: str,
    target_index_sha256: str,
    draft_config_sha256: str,
    draft_index_sha256: str,
    linear_attn_verify_backend: Literal["triton"] | None,
    qsa_patch: dict[str, object],
) -> None:
    controlled_names = (
        "PYTHONNOUSERSITE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_DATASETS_OFFLINE",
        "UV_OFFLINE",
        "UV_PYTHON_DOWNLOADS",
        "PYTORCH_CUDA_ALLOC_CONF",
        "TORCH_CUDA_ARCH_LIST",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "MAX_JOBS",
        "CMAKE_BUILD_PARALLEL_LEVEL",
        "FLASHINFER_NINJA_JOBS",
        "FLASHINFER_NVCC_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
        "SGLANG_ENABLE_SM120_LOWM_BF16_GEMM",
        "SGLANG_SM120_ONLINE_MXFP8",
        "SGLANG_SM120_LOWM_FP8_WEIGHT",
        "SGLANG_SM120_LM_HEAD_FP8",
        "SGLANG_MAMBA_CONV_DTYPE",
        "SGLANG_NUMA_BIND_V2",
        "SGLANG_MM_PREPROCESS_DEVICE",
        "NUMPY_MADVISE_HUGEPAGE",
        "CUDA_HOME",
        "CUDA_PATH",
        "CUDACXX",
        "CC",
        "CXX",
        "CUDAHOSTCXX",
        "HF_HOME",
        "HF_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "HF_DATASETS_CACHE",
        "XDG_CACHE_HOME",
        "TORCH_HOME",
        "CUDA_CACHE_PATH",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "FLASHINFER_WORKSPACE_BASE",
        "SGLANG_CACHE_DIR",
        "SGLANG_JIT_CACHE_DIR",
        "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN",
    )
    managed_environment = {
        key: environment[key] for key in controlled_names if key in environment
    }
    differences = [
        "Notebook targets RTX PRO 6000 SM120; this route requires H200 SM90.",
        "TORCH_CUDA_ARCH_LIST changes from 12.0 to 9.0.",
        "Four SM120-only optimization flags are explicitly set to 0.",
        "MTP NEXTN settings and FP8 KV/draft cache settings remain enabled.",
        "Setup and notebook clocks are recorded separately; readiness timeout starts at the Popen monotonic clock.",
    ]
    if linear_attn_verify_backend is not None:
        differences.append(
            f"Explicit compatibility override: --linear-attn-verify-backend {linear_attn_verify_backend}.",
        )
    if qsa_patch["enabled"]:
        differences.append(
            "Explicit source-pinned QSA FP8-to-query-dtype selected-K/V scratch patch applied.",
        )
    metadata = {
        "schema": "franzen-h200-launch.v1",
        "created_utc": _utc_now(),
        "notebook_source_sha256": NOTEBOOK_SOURCE_SHA256,
        "sglang_revision": SGLANG_REVISION,
        "sglang_version": SGLANG_VERSION,
        "route": "H200-SM90",
        "expected_gpu": {"name": "NVIDIA H200", "compute_capability": "9.0"},
        "differences_from_notebook": differences,
        "linear_attn_verify_backend_override": linear_attn_verify_backend,
        "qsa_fp8_compute_scratch_patch": qsa_patch,
        "clocks": {
            "notebook_start_epoch": started,
            "launcher_setup_started_epoch": setup_started,
            "launcher_setup_completed_epoch": setup_completed,
            "launcher_setup_seconds": setup_completed - setup_started,
            "server_process_started_at": "recorded after Popen in launch-result.json",
            "startup_deadline_origin": "server Popen monotonic clock",
        },
        "startup_timeout_seconds": startup_timeout,
        "input_precache": {
            "wheelhouse_threads": 16,
            "model_and_draft_threads": 3,
            "model_and_draft_delay_seconds": 60,
            "log": str(paths.work_dir / "precache.log"),
        },
        "paths": {
            "target": str(paths.target_dir),
            "draft_source": str(paths.draft_dir),
            "draft_view": str(draft_view),
            "wheelhouse": str(paths.wheelhouse),
            "work_dir": str(paths.work_dir),
            "server_log": str(paths.log),
            "gpu_metrics": str(paths.metrics),
        },
        "assets": {
            "token_map_sha256": token_map_digest,
            "tokenizer_sha256": tokenizer_digest,
            "target_shard_count": target_shards,
            "target_config_sha256": target_config_sha256,
            "target_index_sha256": target_index_sha256,
            "draft_view_config_sha256": draft_config_sha256,
            "draft_view_index_sha256": draft_index_sha256,
        },
        "gpu_preflight": gpu_metrics,
        "launch_admission": admission,
        "offline_install": {
            "metadata_path": str(paths.install_metadata),
            "metadata_sha256": _sha256(paths.install_metadata),
            "receipt": install_receipt,
        },
        "argv": list(argv),
        "command": shlex.join(argv),
        "managed_environment": managed_environment,
    }
    _write_json(paths.metadata, metadata)


def _launch_and_wait(
    *,
    paths: RuntimePaths,
    argv: tuple[str, ...],
    environment: dict[str, str],
    notebook_start_epoch: float,
    startup_timeout: int,
    precache_cancel: threading.Event,
    cancellation: threading.Event,
    bind_host: str,
) -> int:
    log_offset = paths.log.stat().st_size if paths.log.exists() else 0
    server_url = _health_url(bind_host, int(argv[argv.index("--port") + 1]))
    with paths.log.open("ab", buffering=0) as log_file:
        process = subprocess.Popen(
            argv,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    process_start_epoch = time.time()
    process_start_monotonic = time.monotonic()
    startup_deadline = process_start_monotonic + startup_timeout
    _write_json(paths.pid_file, {"pid": process.pid, "argv": list(argv)})
    stop_metrics = threading.Event()
    metrics_thread = threading.Thread(
        target=_record_gpu_metrics,
        args=(paths.metrics, stop_metrics),
        name="franzen-h200-gpu-metrics",
        daemon=True,
    )
    metrics_thread.start()
    last_report = 0.0
    ready = False
    status = "running"
    try:
        while process.poll() is None:
            if cancellation.is_set():
                status = "cancelled"
                break
            now = time.monotonic()
            if not ready:
                if now >= startup_deadline:
                    status = "startup_timeout"
                    break
                try:
                    with urllib.request.urlopen(server_url, timeout=5) as response:
                        ready = response.status == 200
                except (OSError, urllib.error.URLError):
                    ready = False
                if ready:
                    _append_jsonl(
                        paths.metrics,
                        {
                            "utc": _utc_now(),
                            "phase": "ready",
                            "server_url": server_url.removesuffix("/health") + "/v1",
                        },
                    )
                    print(
                        f"READY pid={process.pid} url="
                        f"{server_url.removesuffix('/health')}/v1",
                        flush=True,
                    )
                elif now - last_report >= 30:
                    _append_jsonl(
                        paths.metrics,
                        {
                            "utc": _utc_now(),
                            "phase": "startup_wait",
                            "elapsed_seconds": now - process_start_monotonic,
                        },
                    )
                    last_report = now
            time.sleep(0.5 if ready else 3)
        if status == "running":
            status = "exited_before_ready" if not ready else "exited"
    except KeyboardInterrupt:
        cancellation.set()
        status = "cancelled"
    except BaseException:
        status = "error"
        raise
    finally:
        precache_cancel.set()
        if process.poll() is None:
            _terminate_process_group(process)
        try:
            return_code = process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _signal_process_group(process, signal.SIGKILL)
            return_code = process.wait()
        stop_metrics.set()
        metrics_thread.join(timeout=5)
        _write_json(
            paths.pid_file,
            {"pid": process.pid, "status": "reaped", "returncode": return_code},
        )
    result = {
        "status": status,
        "utc": _utc_now(),
        "pid": process.pid,
        "health_url": server_url,
        "notebook_start_epoch": notebook_start_epoch,
        "process_start_epoch": process_start_epoch,
        "process_start_monotonic": process_start_monotonic,
        "startup_deadline_monotonic": startup_deadline,
        "startup_timeout_seconds": startup_timeout,
        "startup_elapsed_seconds": time.monotonic() - process_start_monotonic,
        "server_left_running": False,
        "returncode": return_code,
    }
    _write_json(paths.work_dir / "launch-result.json", result)
    _show_log_tail(paths.log, offset=log_offset)
    if status == "exited" and return_code == 0:
        return 0
    print(
        f"Server lifecycle ended ({status}, returncode={return_code}); owned process group was reaped. "
        f"Inspect {paths.log} and {paths.metadata}.",
        file=sys.stderr,
    )
    if status == "cancelled":
        return 130
    if status in ("startup_timeout", "exited_before_ready"):
        return 2
    return return_code if return_code != 0 else 2


def _terminate_process_group(process: subprocess.Popen[str]) -> None:
    _signal_process_group(process, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        _signal_process_group(process, signal.SIGKILL)
        process.wait()


def _signal_process_group(process: subprocess.Popen[str], sig: signal.Signals) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def _health_url(host: str, port: int) -> str:
    url_host = f"[{host}]" if ":" in host else host
    return f"http://{url_host}:{port}/health"


def _record_gpu_metrics(path: Path, stop: threading.Event) -> None:
    while not stop.wait(30):
        try:
            _append_jsonl(
                path,
                {"utc": _utc_now(), "phase": "runtime", "gpus": _query_gpu()},
            )
        except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
            _append_jsonl(
                path,
                {
                    "utc": _utc_now(),
                    "phase": "runtime_error",
                    "error": _exception_message(error),
                },
            )


def _query_gpu() -> list[dict[str, object]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={GPU_QUERY}",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_gpu_metrics(result.stdout)


def _run_logged(
    command: list[str],
    environment: dict[str, str],
    paths: RuntimePaths,
) -> str:
    with paths.log.open("a", encoding="utf-8") as log_file:
        log_file.write(f"$ {shlex.join(command)}\n")
        log_file.flush()
        result = subprocess.run(
            command,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        log_file.write(result.stdout)
        log_file.write(result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"Command failed ({result.returncode}): {shlex.join(command)}; "
            f"see {paths.log}\n{_log_tail(paths.log)}",
        )
    return result.stdout


def _exception_message(error: Exception) -> str:
    if isinstance(error, subprocess.CalledProcessError):
        detail = error.stderr or error.output
        if isinstance(detail, bytes):
            detail = detail.decode(errors="replace")
        if detail:
            return f"{error}: {detail}"
    return str(error)


def _require_unoccupied_port(host: str, port: int) -> None:
    if port < 1 or port > 65535:
        raise ValueError(f"Port is outside the valid range: {port}")
    with socket.socket() as probe:
        if probe.connect_ex((host, port)) == 0:
            raise RuntimeError(
                f"Port {port} is occupied; stop the existing server before rerunning",
            )


def _show_log_tail(path: Path, *, offset: int) -> None:
    if not path.is_file():
        return
    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        end = log_file.tell()
        log_file.seek(max(offset, end - 20000))
        print(log_file.read().decode(errors="replace"))


def _log_tail(path: Path, *, limit: int = 20000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as log_file:
        log_file.seek(0, os.SEEK_END)
        log_file.seek(max(0, log_file.tell() - limit))
        return log_file.read().decode(errors="replace")


def _find_unique(
    root: Path,
    name: str,
    *,
    directory: bool = False,
) -> Path:
    direct = root if root.name == name else root / name
    valid = Path.is_dir if directory else Path.is_file
    if valid(direct):
        return direct
    hits = sorted(path for path in root.rglob(name) if valid(path))
    if len(hits) > 1:
        raise ValueError(f"Multiple {name} matches under {root}; select a precise wheelhouse")
    if hits:
        return hits[0]
    raise FileNotFoundError(f"{name} not found under {root}")


def _find_optional(root: Path, name: str) -> Path | None:
    direct = root if root.name == name else root / name
    if direct.is_file():
        return direct
    hits = sorted(path for path in root.rglob(name) if path.is_file())
    if len(hits) > 1:
        raise ValueError(f"Multiple {name} matches under {root}; select a precise wheelhouse")
    return hits[0] if hits else None


def _indexed_shards(root: Path) -> tuple[dict[str, object], list[str]]:
    index = _read_object(root / "model.safetensors.index.json")
    weight_map = _object(index.get("weight_map"))
    if not all(
        isinstance(key, str) and isinstance(name, str)
        for key, name in weight_map.items()
    ):
        raise ValueError("Checkpoint shard index keys and values must be strings")
    names = sorted(set(cast("str", name) for name in weight_map.values()))
    if not names:
        raise ValueError(f"Empty weight index: {root}")
    for name in names:
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Invalid shard path in index: {name}")
        shard = root / relative
        if not shard.is_file() or shard.stat().st_size < 8:
            raise ValueError(f"Missing or empty checkpoint shard: {shard}")
    return weight_map, names


def _read_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return cast("dict[str, object]", value)


def _object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object, got {type(value).__name__}")
    return cast("dict[str, object]", value)


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(f".{path.name}.pending")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _append_jsonl(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(value, sort_keys=True) + "\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number_or_none(value: str) -> float | int | None:
    if value in ("N/A", "[N/A]", "Not Supported"):
        return None
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError as error:
            raise ValueError(f"Invalid nvidia-smi metric: {value}") from error


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()
