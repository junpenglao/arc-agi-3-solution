"""CPU tests for the gated standalone QSA verification probe."""

from __future__ import annotations

import pytest

from serving.qsa_probe.runner import (
    _require_bad_control_rejected,
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
