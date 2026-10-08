"""CPU tests for the gated standalone QSA verification probe."""

from __future__ import annotations

import json

import pytest

from serving.qsa_probe.runner import (
    _require_bad_control_rejected,
    _valid_prefix_counts,
    _write_failure_diagnostic,
    main,
    probe_case_specs,
)


def test_probe_matrix_covers_decode_target_verify_draft_and_empty_mask() -> None:
    cases = probe_case_specs()

    assert [(case.name, case.query_rows, case.caller) for case in cases] == [
        ("decode-one", 1, "forward_decode"),
        ("target-verify-four", 4, "forward_extend"),
        ("draft-decode-four", 4, "forward_decode"),
    ]
    assert (-1, -1, -1, -1, -1) in cases[1].selected_indices
    assert any(
        index >= length
        for row, length in zip(
            cases[1].selected_indices,
            cases[1].sequence_lengths,
            strict=True,
        )
        for index in row
    )
    assert [
        _valid_prefix_counts(case.sequence_lengths, case.selected_indices)
        for case in cases
    ] == [(3,), (3, 4, 0, 4), (3, 4, 5, 3)]


def test_probe_mask_contract_rejects_valid_index_after_invalid_slot() -> None:
    with pytest.raises(ValueError, match="valid indices must be a prefix"):
        _valid_prefix_counts((5,), ((0, -1, 2, 4, 5),))


def test_probe_requires_explicit_execution_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--vendor-file",
                "/site-packages/sglang/srt/layers/attention/qwen_sparse_attn_backend.py",
                "--output",
                "/opt/scratch/runs/arcagi3/franzen-repro-20261008/qsa-probe.json",
            ],
        )

    assert error.value.code == 2
    assert "--run" in capsys.readouterr().err


def test_probe_help_is_safe_without_loading_vendor_runtime() -> None:
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0


def test_probe_requires_known_bad_fp8_control_to_fail() -> None:
    with pytest.raises(AssertionError, match="known-bad FP8 scratch control"):
        _require_bad_control_rejected({"rejected": False})

    _require_bad_control_rejected({"rejected": True})


def test_probe_writes_case_evidence_before_propagating_failure(tmp_path) -> None:
    output = tmp_path / "probe-failure.json"
    evidence = {
        "query": {"dtype": "torch.bfloat16", "shape": [1, 2, 64], "values": [[[0.0]]] },
        "cache": {"dtype": "torch.float8_e4m3fn", "shape": [8, 1, 64], "values": [[[1.0]]] },
        "indices": [[0, 4, 5, -1, 2]],
        "valid_counts": [3],
        "gathered_scratch": [[[1.0]]],
        "output": [[[0.2]]],
        "reference": [[[0.1]]],
    }

    _write_failure_diagnostic(
        output,
        vendor_file="/runtime/qwen_sparse_attn_backend.py",
        vendor_sha256="a" * 64,
        gpu={"name": "NVIDIA H200", "capability": [9, 0]},
        case="decode-one",
        error=AssertionError("attention output differs"),
        evidence=evidence,
    )

    receipt = json.loads(output.read_text())
    assert receipt["status"] == "failed"
    assert receipt["failure"] == {
        "type": "AssertionError",
        "message": "attention output differs",
    }
    assert receipt["case_evidence"]["valid_counts"] == [3]
    assert receipt["case_evidence"]["gathered_scratch"] == [[[1.0]]]
    assert receipt["case_evidence"]["output"] == [[[0.2]]]
    assert receipt["case_evidence"]["reference"] == [[[0.1]]]
