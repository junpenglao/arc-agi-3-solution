"""Resolve the fixed public or manifest-pinned community game roster."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from inference.framework.kaggle import DUCK_HARNESS_PUBLIC_GAME_IDS

COMMUNITY_GAME_COUNT = 268
COMMUNITY_SCORE_BASELINE_KIND = "synthetic_certificate_derived_not_human"
COMMUNITY_MANIFEST_SHA256 = (
    "3807564d2d7e1ed08d5397dd01093cd3eb44a8decf8bb9b03d7fb654a94cb263"
)
_COMMUNITY_SOURCE_COUNTS = {
    "ext_arc3games": 1,
    "ext_arcinteractive": 166,
    "ext_arcwitness": 101,
}
_GAME_ID = re.compile(r"[a-z0-9]{4,}(?:-[a-z0-9]+)?\Z")


@dataclass(frozen=True, slots=True)
class _CommunityGame:
    """Manifest identity and exact asset hashes for one community game."""

    source: str
    game_id: str
    game_sha256: str
    metadata_sha256: str


def load_game_roster(
    group: str,
    *,
    manifest_path: Path | None,
    environments_dir: Path | None,
    expected_manifest_sha256: str = COMMUNITY_MANIFEST_SHA256,
) -> list[str]:
    """Return stable game ids after verifying the pinned community assets."""
    if environments_dir is None or not environments_dir.is_dir():
        raise ValueError(f"--environments-dir must name an existing directory for {group}.")
    if group == "public25":
        if manifest_path is not None:
            raise ValueError("--roster-manifest is only valid for community268.")
        if len(DUCK_HARNESS_PUBLIC_GAME_IDS) != 25:
            raise ValueError("The pinned public roster must contain exactly 25 ids.")
        return list(DUCK_HARNESS_PUBLIC_GAME_IDS)
    if group != "community268":
        raise ValueError(f"Unknown roster group: {group!r}.")
    if manifest_path is None:
        raise ValueError(
            "community268 requires --roster-manifest."
        )

    manifest_bytes = manifest_path.read_bytes()
    actual_manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise ValueError(
            "Community roster manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_manifest_sha256}."
        )
    try:
        payload = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{manifest_path}: invalid roster JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("games"), list):
        raise TypeError("Community roster manifest must contain a games list.")
    games = [_parse_community_game(value) for value in payload["games"]]
    game_ids = [game.game_id for game in games]
    if len(games) != COMMUNITY_GAME_COUNT:
        raise ValueError(
            f"Community roster must contain {COMMUNITY_GAME_COUNT} games; "
            f"got {len(games)}."
        )
    if len(set(game_ids)) != len(game_ids):
        raise ValueError("Community roster contains duplicate game ids.")
    source_counts = Counter(game.source for game in games)
    if dict(source_counts) != _COMMUNITY_SOURCE_COUNTS:
        raise ValueError(f"Unexpected community source distribution: {dict(source_counts)}.")
    _verify_community_assets(games, environments_dir)
    return game_ids


def _parse_community_game(value: object) -> _CommunityGame:
    if not isinstance(value, dict):
        raise TypeError("Every community roster entry must be an object.")
    source = value.get("source")
    game_id = value.get("game_id")
    game_sha256 = value.get("game_sha256")
    metadata_sha256 = value.get("metadata_sha256")
    if source not in _COMMUNITY_SOURCE_COUNTS:
        raise ValueError(f"Unsupported community source: {source!r}.")
    if not isinstance(game_id, str) or _GAME_ID.fullmatch(game_id) is None:
        raise ValueError(f"Invalid community game id: {game_id!r}.")
    return _CommunityGame(
        source=source,
        game_id=game_id,
        game_sha256=_validated_sha256(game_sha256, "game_sha256", game_id),
        metadata_sha256=_validated_sha256(metadata_sha256, "metadata_sha256", game_id),
    )


def _verify_community_assets(games: list[_CommunityGame], root: Path) -> None:
    if not root.is_dir():
        raise ValueError(f"Community environments directory does not exist: {root}.")
    for game in games:
        game_dir = root / game.source / game.game_id
        game_path = game_dir / "game.py"
        metadata_path = game_dir / "metadata.json"
        _verify_file_hash(game_path, game.game_sha256, label="game.py")
        _verify_file_hash(metadata_path, game.metadata_sha256, label="metadata.json")
        try:
            metadata = json.loads(metadata_path.read_bytes())
        except json.JSONDecodeError as exc:
            raise ValueError(f"{metadata_path}: invalid metadata JSON.") from exc
        if not isinstance(metadata, dict) or metadata.get("class_name") != "Game":
            raise ValueError(f"{metadata_path} must declare class_name=Game.")
        metadata_game_id = metadata.get("game_id")
        if not isinstance(metadata_game_id, str) or _normalize_game_id(metadata_game_id) != _normalize_game_id(game.game_id):
            raise ValueError(f"{metadata_path} does not identify {game.game_id}.")


def _verify_file_hash(path: Path, expected: str, *, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"Pinned community asset is missing: {path}.")
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual != expected:
        raise ValueError(
            f"{label} hash mismatch for {path}: expected {expected}, got {actual}."
        )


def _validated_sha256(value: object, name: str, game_id: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid {name} for community game {game_id}.")
    return value


def _normalize_game_id(game_id: str) -> str:
    return game_id.split("-", 1)[0]
