"""Run a bounded numerical probe against the installed, patched QSA kernels."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

from serving.franzen_h200 import QSA_FP8_PATCH_AFTER_SHA256


@dataclass(frozen=True, slots=True)
class ProbeCase:
    name: str
    sequence_lengths: tuple[int, ...]
    selected_indices: tuple[tuple[int, ...], ...]
    caller: str

    @property
    def query_rows(self) -> int:
        return len(self.sequence_lengths)


class _Flags(Protocol):
    vendor_file: Path
    output: Path
    run: bool


class _Pool:
    def __init__(self, key: object, value: object) -> None:
        self.key = key
        self.value = value

    def get_key_buffer(self, layer_id: int) -> object:
        del layer_id
        return self.key

    def get_value_buffer(self, layer_id: int) -> object:
        del layer_id
        return self.value


class _TargetVerifyMode:
    @staticmethod
    def is_target_verify() -> bool:
        return True

    @staticmethod
    def is_draft_extend_v2() -> bool:
        return False


def probe_case_specs() -> tuple[ProbeCase, ...]:
    """Return the bounded batch shapes and masks exercised by the GPU probe."""
    return (
        ProbeCase(
            name="decode-one",
            sequence_lengths=(5,),
            selected_indices=((0, 4, 2, 5, -1),),
            caller="forward_decode",
        ),
        ProbeCase(
            name="target-verify-four",
            sequence_lengths=(4, 5, 6, 7),
            selected_indices=(
                (0, 3, 1, -1, 99),
                (4, 0, 3, 2, 7),
                (-1, -1, -1, -1, -1),
                (6, 2, 5, 4, 99),
            ),
            caller="forward_extend",
        ),
        ProbeCase(
            name="draft-decode-four",
            sequence_lengths=(6, 7, 8, 8),
            selected_indices=(
                (5, 1, 3, -1, 99),
                (6, 0, 4, 2, 99),
                (7, 1, 5, 3, 0),
                (7, 6, 2, -1, 99),
            ),
            caller="forward_decode",
        ),
    )


def _valid_prefix_counts(
    sequence_lengths: tuple[int, ...],
    selected_indices: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    counts: list[int] = []
    for sequence_length, row_indices in zip(sequence_lengths, selected_indices, strict=True):
        valid_count = 0
        invalid_seen = False
        for index in row_indices:
            if 0 <= index < sequence_length:
                if invalid_seen:
                    raise ValueError(
                        "QSA selected indices must be valid indices first; "
                        "valid indices must be a prefix of each row",
                    )
                valid_count += 1
            else:
                invalid_seen = True
        counts.append(valid_count)
    return tuple(counts)


def main(argv: list[str] | None = None) -> int:
    """Run the explicit GPU probe and write a durable JSON receipt."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").strip())
    parser.add_argument("--vendor-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="store_true")
    flags = cast(_Flags, parser.parse_args(argv))
    if not flags.run:
        parser.error("this probe performs GPU work; pass --run explicitly after review")
    _validate_vendor_file(flags.vendor_file)
    receipt = _run_probe(flags.vendor_file, diagnostic_output=flags.output)
    flags.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = flags.output.with_name(f".{flags.output.name}.pending")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(flags.output)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


def _validate_vendor_file(path: Path) -> None:
    digest = _sha256(path)
    if digest != QSA_FP8_PATCH_AFTER_SHA256:
        raise ValueError(
            "The probe requires the explicit QSA scratch patch; "
            f"vendor source SHA-256 is {digest}",
        )


def _run_probe(vendor_file: Path, *, diagnostic_output: Path) -> dict[str, object]:
    import torch

    from sglang.srt.layers.attention import qwen_sparse_attn_backend as qsa

    module_path = Path(cast(str, qsa.__file__)).resolve()
    if module_path != vendor_file.resolve():
        raise RuntimeError(
            f"Imported QSA backend {module_path} differs from selected source {vendor_file}",
        )
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible H200 GPU, found {torch.cuda.device_count()}")
    device_name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    if "H200" not in device_name or capability != (9, 0):
        raise RuntimeError(
            f"Expected NVIDIA H200 SM90, found {device_name} {capability}",
        )
    if qsa._resolve_trtllm_sparse_decode() is not None:
        raise RuntimeError("Probe expected the pinned SM90 packed-varlen fallback")
    qsa.QwenSparseAttnBackend._require_unit_qsa_kv_scales({})
    try:
        qsa.QwenSparseAttnBackend._require_unit_qsa_kv_scales({"k_scale": 0.5})
    except ValueError:
        nonunit_scale_refused = True
    else:
        nonunit_scale_refused = False
    if not nonunit_scale_refused:
        raise AssertionError("The patched QSA path accepted a non-unit KV scale")

    torch.manual_seed(38660)
    torch.cuda.manual_seed_all(38660)
    key_cache, value_cache = _fp8_cache(torch, device_name)
    initial_key = key_cache.clone()
    initial_value = value_cache.clone()
    varlen = qsa._resolve_flash_attn_varlen_func()
    result_rows = []
    for case in probe_case_specs():
        diagnostics: dict[str, object] = {
            "initial_key_cache": initial_key,
            "initial_value_cache": initial_value,
            "case_sequence_lengths": list(case.sequence_lengths),
            "case_selected_indices": [list(row) for row in case.selected_indices],
        }
        try:
            output, selected = _run_case(
                torch=torch,
                qsa=qsa,
                varlen=varlen,
                case=case,
                key_cache=key_cache,
                value_cache=value_cache,
                diagnostics=diagnostics,
            )
            diagnostics.update(selected)
            diagnostics["gather_verified"] = False
            reference = _bf16_attention_reference(
                torch=torch,
                queries=selected["queries"],
                key_cache=initial_key,
                value_cache=initial_value,
                sequence_lengths=case.sequence_lengths,
                selected_indices=case.selected_indices,
                scale=selected["scale"],
            )
            diagnostics["output"] = output
            diagnostics["reference"] = reference
            max_abs_error = float((output.float() - reference.float()).abs().max())
            diagnostics["max_abs_error"] = max_abs_error
            _assert_gathered_scratch(torch, selected, initial_key, initial_value, case)
            diagnostics["gather_verified"] = True
            torch.testing.assert_close(output, reference, rtol=0.04, atol=0.04)
        except Exception as error:
            _write_failure_diagnostic(
                diagnostic_output,
                vendor_file=str(module_path),
                vendor_sha256=_sha256(module_path),
                gpu={"name": device_name, "capability": list(capability)},
                case=case.name,
                error=error,
                evidence=_snapshot_evidence(torch, diagnostics),
            )
            raise
        result_rows.append(
            {
                "case": case.name,
                "caller": case.caller,
                "query_rows": case.query_rows,
                "sequence_lengths": list(case.sequence_lengths),
                "selected_indices": [list(row) for row in case.selected_indices],
                "output_max_abs_error": max_abs_error,
                "scratch_dtype": str(selected["scratch_k"].dtype),
                "masked_counts": selected["valid_counts"].cpu().tolist(),
            },
        )
    if not torch.equal(key_cache, initial_key) or not torch.equal(value_cache, initial_value):
        raise AssertionError("QSA gather modified the persistent FP8 K/V pool")
    bad_control = _known_bad_fp8_scratch_control(
        torch=torch,
        qsa=qsa,
        varlen=varlen,
        key_cache=key_cache,
        value_cache=value_cache,
    )
    _require_bad_control_rejected(bad_control)
    return {
        "schema": "franzen-qsa-sm90-fp8-scratch-probe.v1",
        "vendor_file": str(module_path),
        "vendor_sha256": _sha256(module_path),
        "gpu": {"name": device_name, "capability": list(capability)},
        "pool_dtype": str(key_cache.dtype),
        "ssm_or_weights_changed": False,
        "nonunit_scale_refused": nonunit_scale_refused,
        "persistent_fp8_pool_unchanged": True,
        "cases": result_rows,
        "known_bad_raw_fp8_scratch_control": bad_control,
    }


def _run_case(
    *,
    torch,
    qsa,
    varlen,
    case: ProbeCase,
    key_cache,
    value_cache,
    diagnostics: dict[str, object],
) -> tuple[object, dict[str, object]]:
    expected_valid_counts = _valid_prefix_counts(
        case.sequence_lengths,
        case.selected_indices,
    )
    query_rows = case.query_rows
    query_heads = 2
    kv_heads = 1
    head_dim = 64
    max_context = int(key_cache.shape[0])
    queries = (
        torch.arange(query_rows * query_heads * head_dim, dtype=torch.float32)
        .reshape(query_rows, query_heads, head_dim)
        .sub(64)
        .div(64)
        .to(device="cuda", dtype=torch.bfloat16)
    )
    sequence_lengths = torch.tensor(case.sequence_lengths, dtype=torch.int32, device="cuda")
    indices = torch.tensor(case.selected_indices, dtype=torch.int32, device="cuda")
    row_requests = torch.zeros(query_rows, dtype=torch.int32, device="cuda")
    req_to_token = torch.arange(max_context, dtype=torch.int32, device="cuda").reshape(1, -1)
    metadata = SimpleNamespace(
        sequence_lengths=sequence_lengths,
        row_req_pool_indices=row_requests,
        is_cuda_graph=False,
        fa2_valid_counts=None,
        fa2_cu_seqlens_k=None,
        fa2_cu_seqlens_q=None,
    )
    pool = _Pool(key_cache, value_cache)
    backend = qsa.QwenSparseAttnBackend.__new__(qsa.QwenSparseAttnBackend)
    backend.token_to_kv_pool = pool
    backend.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)
    backend.forward_metadata = metadata
    backend._cuda_graph_max_tokens = 0
    backend._fa2_scratch = {}
    mode = _TargetVerifyMode() if case.caller == "forward_extend" else None
    forward_batch = SimpleNamespace(
        req_pool_indices=row_requests,
        forward_mode=mode,
        out_cache_loc=torch.zeros(query_rows, dtype=torch.int64, device="cuda"),
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=query_heads,
        head_dim=head_dim,
        scaling=head_dim**-0.5,
    )
    valid_counts = torch.empty(query_rows, dtype=torch.int32, device="cuda")
    cu_seqlens_k = torch.empty(query_rows + 1, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.arange(query_rows + 1, dtype=torch.int32, device="cuda")
    qsa.qwen_sparse_fa2_cu_seqlens_triton(
        sequence_lengths,
        indices,
        valid_counts,
        cu_seqlens_k,
        query_rows,
        indices.shape[1],
    )
    diagnostics.update(
        {
            "queries": queries,
            "query_scale": layer.scaling,
            "persistent_key_cache": key_cache,
            "persistent_value_cache": value_cache,
            "indices": indices,
            "sequence_lengths": sequence_lengths,
            "valid_counts": valid_counts,
            "expected_valid_counts": list(expected_valid_counts),
            "cu_seqlens_k": cu_seqlens_k,
            "cu_seqlens_q": cu_seqlens_q,
        },
    )
    if valid_counts.cpu().tolist() != list(expected_valid_counts):
        raise AssertionError("QSA valid-count kernel differs from the fixture mask contract")
    try:
        if case.caller == "forward_extend":
            output = backend.forward_extend(
                queries,
                key_cache[:query_rows],
                value_cache[:query_rows],
                layer,
                forward_batch,
                save_kv_cache=False,
                topk_indices=indices,
            )
        else:
            output = backend.forward_decode(
                queries,
                key_cache[:query_rows],
                value_cache[:query_rows],
                layer,
                forward_batch,
                save_kv_cache=False,
                topk_indices=indices,
            )
    finally:
        scratch = backend._fa2_scratch.get(
            (kv_heads, head_dim, queries.dtype, queries.device),
        )
        if scratch is not None:
            selected_scratch_rows = int(valid_counts.sum().item())
            diagnostics["scratch_k"] = scratch[0][:selected_scratch_rows]
            diagnostics["scratch_v"] = scratch[1][:selected_scratch_rows]
    scratch_k, scratch_v = backend._fa2_scratch[(kv_heads, head_dim, queries.dtype, queries.device)]
    selected_scratch_rows = int(valid_counts.sum().item())
    scratch_k = scratch_k[:selected_scratch_rows]
    scratch_v = scratch_v[:selected_scratch_rows]
    return output.reshape_as(queries), {
        "queries": queries,
        "indices": indices,
        "sequence_lengths": sequence_lengths,
        "valid_counts": valid_counts,
        "cu_seqlens_k": cu_seqlens_k,
        "cu_seqlens_q": cu_seqlens_q,
        "scratch_k": scratch_k,
        "scratch_v": scratch_v,
        "scale": layer.scaling,
    }


def _fp8_cache(torch, device_name: str) -> tuple[object, object]:
    del device_name
    values = (
        torch.arange(8 * 64, dtype=torch.float32)
        .reshape(8, 1, 64)
        .remainder(29)
        .sub(14)
        .div(8)
        .to(device="cuda", dtype=torch.float8_e4m3fn)
    )
    return values.clone(), values.flip(0).clone()


def _bf16_attention_reference(
    *,
    torch,
    queries,
    key_cache,
    value_cache,
    sequence_lengths: tuple[int, ...],
    selected_indices: tuple[tuple[int, ...], ...],
    scale: float,
) -> object:
    reference = torch.zeros_like(queries, dtype=torch.float32)
    group_size = queries.shape[1] // key_cache.shape[1]
    for row, (length, selected) in enumerate(zip(sequence_lengths, selected_indices, strict=True)):
        valid = [index for index in selected if 0 <= index < length]
        if not valid:
            continue
        keys = key_cache[valid].to(torch.bfloat16).float()
        values = value_cache[valid].to(torch.bfloat16).float()
        for head in range(queries.shape[1]):
            kv_head = head // group_size
            scores = queries[row, head].float() @ keys[:, kv_head, :].T
            probabilities = torch.softmax(scores * scale, dim=0)
            reference[row, head] = probabilities @ values[:, kv_head, :]
    return reference.to(torch.bfloat16)


def _assert_gathered_scratch(torch, selected, key_cache, value_cache, case) -> None:
    offset = 0
    for row, (length, indices) in enumerate(zip(case.sequence_lengths, case.selected_indices, strict=True)):
        valid = [index for index in indices if 0 <= index < length]
        count = len(valid)
        expected_key = key_cache[valid].to(torch.bfloat16)
        expected_value = value_cache[valid].to(torch.bfloat16)
        actual_key = selected["scratch_k"][offset : offset + count]
        actual_value = selected["scratch_v"][offset : offset + count]
        if not torch.equal(actual_key, expected_key) or not torch.equal(actual_value, expected_value):
            raise AssertionError(f"Gathered selected K/V differ in {case.name} row {row}")
        offset += count
    if selected["scratch_k"].dtype != torch.bfloat16 or selected["scratch_v"].dtype != torch.bfloat16:
        raise AssertionError("QSA temporary selected-K/V scratch must use query dtype")


def _known_bad_fp8_scratch_control(
    *,
    torch,
    qsa,
    varlen,
    key_cache,
    value_cache,
) -> dict[str, object]:
    case = probe_case_specs()[0]
    queries = torch.ones((1, 2, 64), dtype=torch.bfloat16, device="cuda")
    seq_lens = torch.tensor(case.sequence_lengths, dtype=torch.int32, device="cuda")
    indices = torch.tensor(case.selected_indices, dtype=torch.int32, device="cuda")
    req_to_token = torch.arange(key_cache.shape[0], dtype=torch.int32, device="cuda").reshape(1, -1)
    req_indices = torch.zeros(1, dtype=torch.int32, device="cuda")
    valid_counts = torch.empty(1, dtype=torch.int32, device="cuda")
    cu_k = torch.empty(2, dtype=torch.int32, device="cuda")
    cu_q = torch.arange(2, dtype=torch.int32, device="cuda")
    qsa.qwen_sparse_fa2_cu_seqlens_triton(seq_lens, indices, valid_counts, cu_k, 1, indices.shape[1])
    bad_k = torch.empty(
        (indices.shape[0] * indices.shape[1], key_cache.shape[1], key_cache.shape[2]),
        dtype=key_cache.dtype,
        device="cuda",
    )
    bad_v = torch.empty_like(bad_k)
    qsa.qwen_sparse_kv_extraction_compact_triton(
        key_cache, value_cache, req_to_token, req_indices, indices, seq_lens,
        cu_k, bad_k, bad_v, 1, indices.shape[1],
    )
    try:
        bad_output = varlen(
            q=queries,
            k=bad_k,
            v=bad_v,
            cu_seqlens_q=cu_q,
            cu_seqlens_k=cu_k,
            max_seqlen_q=1,
            max_seqlen_k=indices.shape[1],
            softmax_scale=64**-0.5,
            causal=True,
        )
    except Exception as error:
        return {"rejected": True, "error": f"{type(error).__name__}: {error}"}
    reference = _bf16_attention_reference(
        torch=torch,
        queries=queries,
        key_cache=key_cache,
        value_cache=value_cache,
        sequence_lengths=case.sequence_lengths,
        selected_indices=case.selected_indices,
        scale=64**-0.5,
    )
    max_error = float(
        (bad_output.reshape_as(reference).float() - reference.float()).abs().max(),
    )
    return {
        "rejected": not torch.allclose(
            bad_output.reshape_as(reference).float(),
            reference.float(),
            rtol=0.04,
            atol=0.04,
        ),
        "output_dtype": str(bad_output.dtype),
        "reference_max_abs_error": max_error,
    }


def _require_bad_control_rejected(result: dict[str, object]) -> None:
    if result.get("rejected") is not True:
        raise AssertionError("known-bad FP8 scratch control was not rejected")


def _write_failure_diagnostic(
    output: Path,
    *,
    vendor_file: str,
    vendor_sha256: str,
    gpu: dict[str, object],
    case: str,
    error: Exception,
    evidence: dict[str, object],
) -> None:
    receipt = {
        "schema": "franzen-qsa-sm90-fp8-scratch-probe.v1",
        "status": "failed",
        "vendor_file": vendor_file,
        "vendor_sha256": vendor_sha256,
        "gpu": gpu,
        "case": case,
        "failure": {"type": type(error).__name__, "message": str(error)},
        "case_evidence": evidence,
    }
    _write_receipt(output, receipt)


def _snapshot_evidence(torch, evidence: dict[str, object]) -> dict[str, object]:
    def snapshot(value: object) -> object:
        if isinstance(value, torch.Tensor):
            cpu_value = value.detach().cpu()
            visible = cpu_value.float() if cpu_value.is_floating_point() else cpu_value
            return {
                "dtype": str(cpu_value.dtype),
                "shape": list(cpu_value.shape),
                "values": visible.tolist(),
            }
        if isinstance(value, dict):
            return {str(name): snapshot(item) for name, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [snapshot(item) for item in value]
        if value is None or type(value) in (str, int, float, bool):
            return value
        return repr(value)

    return {name: snapshot(value) for name, value in evidence.items()}


def _write_receipt(output: Path, receipt: dict[str, object]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.pending")
    temporary.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
