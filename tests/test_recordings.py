import json
from pathlib import Path

import pytest

from motiverooms.recordings import BUNDLED_RECORDING_ID, load_recording, recording_catalog


def test_replay_available_without_local_runs(tmp_path):
    catalog = recording_catalog(tmp_path)
    assert catalog["default_id"] == BUNDLED_RECORDING_ID
    recording = load_recording(catalog["default_id"], tmp_path)
    assert recording["summary"] == catalog["recordings"][0]
    assert recording["outcome"] == "success"
    assert len(recording["frames"]) == recording["summary"]["steps"] + 1
    assert recording["messages"]


def test_bundled_replay_is_packaged_for_vercel():
    root = Path(__file__).resolve().parents[1]
    import tomllib

    package = tomllib.loads((root / "pyproject.toml").read_text())
    assert "data/*.json" in package["tool"]["setuptools"]["package-data"]["motiverooms"]
    config = json.loads((root / "vercel.json").read_text())
    assert "motiverooms/data/**" in config["functions"]["api/index.py"]["includeFiles"]


def test_bundled_lookup_does_not_allow_arbitrary_paths(tmp_path):
    with pytest.raises(ValueError, match="Recording not found"):
        load_recording("bundled/../../.env", tmp_path)
