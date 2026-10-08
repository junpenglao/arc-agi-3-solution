from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path
from typing import Callable
from unittest.mock import Mock

import inference.framework.run as run_module
import pytest
from inference.framework.roster import (
    COMMUNITY_GAME_COUNT,
    COMMUNITY_MANIFEST_SHA256,
    load_game_roster,
)
from inference.framework.run import (
    _make_games,
    _make_solver,
    _optional_positive_float,
    _resolve_game_ids,
)


def _write_community_fixture(root: Path) -> tuple[Path, Path, list[str]]:
    environments_dir = root / "environments"
    roster: list[dict[str, str]] = []
    game_ids = [
        f"{('zz', 'zy', 'zx')[index // 100]}{index % 100:02d}"
        for index in range(COMMUNITY_GAME_COUNT)
    ]
    game_ids[0] = "wtw01s01"
    source_counts = {"ext_arc3games": 1, "ext_arcinteractive": 166, "ext_arcwitness": 101}
    game_index = 0
    for source, source_count in source_counts.items():
        for _ in range(source_count):
            game_id = game_ids[game_index]
            game_index += 1
            game_dir = environments_dir / source / game_id
            game_dir.mkdir(parents=True)
            game_bytes = f"# pinned synthetic fixture for {game_id}\n".encode()
            metadata_bytes = json.dumps(
                {"game_id": game_id, "class_name": "Game"}
            ).encode()
            (game_dir / "game.py").write_bytes(game_bytes)
            (game_dir / "metadata.json").write_bytes(metadata_bytes)
            roster.append(
                {
                    "source": source,
                    "game_id": game_id,
                    "game_sha256": hashlib.sha256(game_bytes).hexdigest(),
                    "metadata_sha256": hashlib.sha256(metadata_bytes).hexdigest(),
                }
            )
    manifest = root / "manifest.json"
    manifest.write_text(json.dumps({"games": roster}), encoding="utf-8")
    return manifest, environments_dir, game_ids


def _roster_args(**updates: object) -> Namespace:
    values: dict[str, object] = {
        "roster_group": "community268",
        "roster_manifest": "/pinned/manifest.json",
        "roster_game_ids": None,
        "roster_asset_transport": "local",
        "environments_dir": "/pinned/environments",
        "game": "",
        "dataset": "",
        "include_tags": "",
        "exclude_tags": "",
        "n_passes": 1,
        "analyzer_save_request_logs": True,
        "list_games": False,
        "max_actions": None,
        "max_generated_tokens_per_game": None,
        "max_runtime_minutes": 20.0,
        "max_experiment_runtime_minutes": 30.0,
        "max_experiment_runtime_hours": None,
        "simulate_competition_arcade": False,
        "kaggle_make_share_version": False,
        "kaggle_duck_public_harness": False,
        "deployment_target": "inline",
    }
    values.update(updates)
    return Namespace(**values)


def _run_config_args(args: Namespace, *, environments_dir: str) -> Namespace:
    values = vars(args).copy()
    values.update(
        agent="coverage-test",
        model="test-model",
        dataset="",
        include_tags="",
        exclude_tags="",
        environments_dir=environments_dir,
        experiments_dir="",
        experiment_dir="",
        analyzer_save_request_logs=True,
        pass_offset=0,
        concurrent_jobs=1,
        deployment_target="inline",
        deployment_wait=False,
        deployment_source_repos="",
        slurm_start_local_server=False,
        slurm_gpu="B200",
        slurm_gpu_count=1,
        slurm_time="01:00:00",
        slurm_image="",
        slurm_partition="",
        slurm_nodelist="",
        slurm_extra_sbatch_flags="",
        max_actions=None,
        max_generated_tokens_per_game=None,
    )
    return Namespace(**values)


def test_community_roster_preserves_all_manifest_ids_and_game_api_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, environments_dir, expected_ids = _write_community_fixture(tmp_path)
    game_ids = load_game_roster(
        "community268",
        manifest_path=manifest,
        environments_dir=environments_dir,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )

    games = _make_games(game_ids, environments_dir=str(environments_dir))

    assert game_ids == expected_ids
    assert len(games) == COMMUNITY_GAME_COUNT
    assert [game.env_name for game in games] == expected_ids
    assert all(game.arcade_spec.environments_dir == str(environments_dir) for game in games)
    assert all(game.arcade_spec.operation_mode.name == "OFFLINE" for game in games)

    load_roster = Mock(return_value=game_ids)
    monkeypatch.setattr(run_module, "load_game_roster", load_roster)
    args = _roster_args(roster_manifest=str(manifest), environments_dir=str(environments_dir))
    assert _resolve_game_ids(args) == expected_ids
    assert args.roster_manifest_sha256 == COMMUNITY_MANIFEST_SHA256
    assert args.roster_baseline_kind == "synthetic_certificate_derived_not_human"
    assert args.roster_full_game_count == COMMUNITY_GAME_COUNT
    assert args.roster_selected_game_ids == expected_ids
    assert args.roster_selection_kind is None
    load_roster.assert_called_once_with(
        "community268", manifest_path=manifest, environments_dir=environments_dir
    )


def test_roster_enforcement_keeps_uncapped_traced_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _roster_args(
        roster_group="public25",
        roster_manifest="",
        environments_dir="/mounted/public25",
        agent="franzen-community-repro",
        model="flashnext",
        analyzer_timeout=900,
        concurrent_jobs=10,
        slurm_start_local_server=False,
        slurm_gpu_count=1,
    )
    expected_ids = ["tn36-ef4dde99"]
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=expected_ids))

    game_ids = _resolve_game_ids(args)
    solver = _make_solver(args, run_dir=tmp_path, max_runtime_minutes_per_game=483.6)

    assert game_ids == expected_ids
    assert solver.concurrency == 10
    assert solver.analyzer_timeout == 900
    assert solver.max_actions_per_game is None
    assert solver.max_generated_tokens_per_game is None
    assert solver.save_request_logs is True
    assert args.roster_manifest_sha256 is None


def test_community_roster_selects_json_ids_in_manifest_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, environments_dir, expected_ids = _write_community_fixture(tmp_path)
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=expected_ids))
    args = _roster_args(
        roster_manifest=str(manifest),
        environments_dir=str(environments_dir),
        roster_game_ids=json.dumps([expected_ids[3], expected_ids[0]]),
    )

    selected_ids = _resolve_game_ids(args)

    assert selected_ids == [expected_ids[0], expected_ids[3]]
    assert args.roster_full_game_count == COMMUNITY_GAME_COUNT
    assert args.roster_selected_game_ids == selected_ids
    assert args.roster_selection_kind == "coverage_supplement"

    config_args = _run_config_args(args, environments_dir=str(environments_dir))
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_module._write_run_config(
        config_args,
        run_dir=run_dir,
        game_ids=selected_ids,
        max_experiment_runtime_minutes=30.0,
        max_runtime_minutes_per_game=20.0,
        max_runtime_minutes_per_game_source="explicit",
        wave_count=1,
    )
    config = json.loads((run_dir / run_module.RUN_CONFIG_FILENAME).read_text())

    assert config["roster_manifest_sha256"] == COMMUNITY_MANIFEST_SHA256
    assert config["roster_full_game_count"] == COMMUNITY_GAME_COUNT
    assert config["roster_selected_game_ids"] == selected_ids
    assert config["roster_selected_game_count"] == len(selected_ids)
    assert config["roster_selection_kind"] == "coverage_supplement"
    assert config["game_count"] == len(selected_ids)
    assert config["score_baseline_kind"] == "synthetic_certificate_derived_not_human"

    csv_args = _roster_args(
        roster_manifest=str(manifest),
        environments_dir=str(environments_dir),
        roster_game_ids=f"{expected_ids[3]},{expected_ids[0]}",
    )
    assert _resolve_game_ids(csv_args) == selected_ids


@pytest.mark.parametrize(
    "selection,match",
    [
        ("", "must not be empty"),
        ("[]", "must not be empty"),
        ("zz00,zz00", "duplicate"),
        ("unknown-game", "Unknown community game id"),
    ],
)
def test_community_roster_rejects_invalid_game_id_selections(
    selection: str,
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_ids = ["zz00", "zy00"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=expected_ids))
    args = _roster_args(roster_manifest=str(manifest), roster_game_ids=selection)

    with pytest.raises(ValueError, match=match):
        _resolve_game_ids(args)


def test_coverage_supplement_listing_keeps_roster_validation_exemptions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected_ids = ["zz00", "zy00", "zx00"]
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=expected_ids))
    args = _roster_args(
        roster_manifest=str(manifest),
        roster_game_ids='["zx00", "zz00"]',
        list_games=True,
        n_passes=4,
        analyzer_save_request_logs=False,
        max_actions=12,
        max_generated_tokens_per_game=1200,
    )

    assert _resolve_game_ids(args) == ["zz00", "zx00"]


@pytest.mark.parametrize(
    "updates,match",
    [
        ({"roster_group": "public25"}, "only valid with --roster-group community268"),
        ({"max_actions": 1}, "uncapped"),
        ({"max_generated_tokens_per_game": 1}, "uncapped"),
        ({"n_passes": 2}, "exactly one pass"),
    ],
)
def test_coverage_supplement_requires_community_uncapped_single_pass(
    updates: dict[str, object],
    match: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=["zz00"]))
    args = _roster_args(
        roster_manifest="" if updates.get("roster_group") == "public25" else str(manifest),
        roster_game_ids='["zz00"]',
        **updates,
    )

    with pytest.raises(ValueError, match=match):
        _resolve_game_ids(args)


def test_coverage_supplement_still_verifies_every_community_asset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, environments_dir, expected_ids = _write_community_fixture(tmp_path)
    unselected_game_path = (
        environments_dir / "ext_arcinteractive" / expected_ids[1] / "game.py"
    )
    unselected_game_path.write_text("# changed outside the selected subset\n", encoding="utf-8")
    expected_manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()

    def load_fixture_roster(
        group: str,
        *,
        manifest_path: Path | None,
        environments_dir: Path | None,
    ) -> list[str]:
        return load_game_roster(
            group,
            manifest_path=manifest_path,
            environments_dir=environments_dir,
            expected_manifest_sha256=expected_manifest_sha256,
        )

    monkeypatch.setattr(run_module, "load_game_roster", load_fixture_roster)
    args = _roster_args(
        roster_manifest=str(manifest),
        environments_dir=str(environments_dir),
        roster_game_ids=expected_ids[0],
    )

    with pytest.raises(ValueError, match="game.py hash mismatch"):
        _resolve_game_ids(args)


def test_community_roster_rejects_asset_hash_drift(tmp_path: Path) -> None:
    manifest, environments_dir, _ = _write_community_fixture(tmp_path)
    game_path = environments_dir / "ext_arc3games" / "wtw01s01" / "game.py"
    game_path.write_text("# changed after pinning\n", encoding="utf-8")

    with pytest.raises(ValueError, match="game.py hash mismatch"):
        load_game_roster(
            "community268",
            manifest_path=manifest,
            environments_dir=environments_dir,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_public25_roster_is_stable_and_manifest_independent(tmp_path: Path) -> None:
    game_ids = load_game_roster("public25", manifest_path=None, environments_dir=tmp_path)

    assert len(game_ids) == 25
    assert len(set(game_ids)) == 25
    assert game_ids[0] == "tn36-ef4dde99"


def test_public25_roster_requires_explicit_asset_directory() -> None:
    with pytest.raises(ValueError, match="environments-dir"):
        load_game_roster("public25", manifest_path=None, environments_dir=None)


def test_community_roster_requires_pinned_manifest(tmp_path: Path) -> None:
    manifest, environments_dir, _ = _write_community_fixture(tmp_path)

    with pytest.raises(ValueError, match="manifest SHA256 mismatch"):
        load_game_roster(
            "community268",
            manifest_path=manifest,
            environments_dir=environments_dir,
            expected_manifest_sha256=COMMUNITY_MANIFEST_SHA256,
        )


def test_roster_runner_rejects_multiple_passes() -> None:
    args = _roster_args(n_passes=2)

    with pytest.raises(ValueError, match="exactly one pass"):
        _resolve_game_ids(args)


def test_roster_runner_requires_request_logs(tmp_path: Path) -> None:
    args = _roster_args(
        roster_manifest="manifest.json",
        environments_dir=str(tmp_path),
        analyzer_save_request_logs=False,
    )

    with pytest.raises(ValueError, match="--save-request-logs"):
        _resolve_game_ids(args)


def test_roster_listing_does_not_require_run_trace_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_ids = ["zz00"]
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=expected_ids))
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    args = _roster_args(
        roster_manifest=str(manifest),
        environments_dir="/pinned/environments",
        n_passes=4,
        analyzer_save_request_logs=False,
        list_games=True,
    )

    assert _resolve_game_ids(args) == expected_ids


@pytest.mark.parametrize("cap_name,cap_value", [("max_actions", 10), ("max_generated_tokens_per_game", 1)])
def test_roster_runner_rejects_trajectory_caps(
    cap_name: str, cap_value: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _roster_args(**{cap_name: cap_value})
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=["tl01"]))

    with pytest.raises(ValueError, match="uncapped"):
        _resolve_game_ids(args)


@pytest.mark.parametrize(
    "updates",
    [
        {"simulate_competition_arcade": True},
        {"kaggle_make_share_version": True},
        {"kaggle_duck_public_harness": True},
        {"deployment_target": "kaggle"},
        {"deployment_target": "slurm", "roster_asset_transport": "local"},
    ],
)
def test_roster_runner_rejects_mode_or_transport_conflicts(
    updates: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _roster_args(**updates)
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=["tl01"]))

    with pytest.raises(ValueError, match="(?i)roster"):
        _resolve_game_ids(args)


def test_mounted_slurm_roster_transport_is_accepted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _roster_args(
        deployment_target="slurm",
        roster_asset_transport="mounted",
        roster_manifest=str(tmp_path / "manifest.json"),
        environments_dir=str(tmp_path),
    )
    Path(args.roster_manifest).write_text("{}", encoding="utf-8")
    monkeypatch.setattr(run_module, "load_game_roster", Mock(return_value=["tl01"]))

    assert _resolve_game_ids(args) == ["tl01"]


def test_cli_registers_roster_and_mount_controls() -> None:
    parser = run_module.argparse.ArgumentParser()
    run_module._add_roster_arguments(parser)

    args = parser.parse_args(
        [
            "--roster-group",
            "community268",
            "--roster-asset-transport",
            "mounted",
            "--roster-game-ids",
            '["zz00"]',
        ]
    )

    assert args.roster_group == "community268"
    assert args.roster_asset_transport == "mounted"
    assert args.roster_game_ids == '["zz00"]'


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), 0.0, -1.0])
def test_runtime_limit_requires_finite_positive_value(value: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        _optional_positive_float(value, option_name="--max-runtime-minutes")


def test_runtime_limit_accepts_finite_positive_value() -> None:
    assert _optional_positive_float(1.5, option_name="--max-runtime-minutes") == 1.5


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda payload: payload["games"].pop(), "must contain 268 games"),
        (lambda payload: payload["games"][0].update(source="unknown"), "Unsupported community source"),
        (lambda payload: payload["games"][0].update(game_id="../bad"), "Invalid community game id"),
        (lambda payload: payload["games"][0].update(game_sha256=3), "Invalid game_sha256"),
    ],
)
def test_community_roster_rejects_malformed_manifest_entries(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
    match: str,
) -> None:
    manifest, environments_dir, _ = _write_community_fixture(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    mutation(payload)
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises((TypeError, ValueError), match=match):
        load_game_roster(
            "community268",
            manifest_path=manifest,
            environments_dir=environments_dir,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_community_roster_rejects_missing_metadata_asset(tmp_path: Path) -> None:
    manifest, environments_dir, _ = _write_community_fixture(tmp_path)
    (environments_dir / "ext_arc3games" / "wtw01s01" / "metadata.json").unlink()

    with pytest.raises(ValueError, match="asset is missing"):
        load_game_roster(
            "community268",
            manifest_path=manifest,
            environments_dir=environments_dir,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )


def test_community_roster_normalizes_hyphenated_game_id_consistently(
    tmp_path: Path,
) -> None:
    manifest, environments_dir, game_ids = _write_community_fixture(tmp_path)
    old_dir = environments_dir / "ext_arc3games" / game_ids[0]
    new_id = f"{game_ids[0]}-version1"
    new_dir = environments_dir / "ext_arc3games" / new_id
    old_dir.rename(new_dir)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    entry = payload["games"][0]
    entry["game_id"] = new_id
    metadata_path = new_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["game_id"] = game_ids[0]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    entry["metadata_sha256"] = hashlib.sha256(metadata_path.read_bytes()).hexdigest()
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    resolved = load_game_roster(
        "community268",
        manifest_path=manifest,
        environments_dir=environments_dir,
        expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    )

    assert resolved[0] == new_id


def test_community_roster_rejects_invalid_json(tmp_path: Path) -> None:
    manifest, environments_dir, _ = _write_community_fixture(tmp_path)
    manifest.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid roster JSON"):
        load_game_roster(
            "community268",
            manifest_path=manifest,
            environments_dir=environments_dir,
            expected_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        )
