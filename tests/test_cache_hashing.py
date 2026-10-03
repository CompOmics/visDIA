import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from mumdia_viewer.data import Cache, open_results
from mumdia_viewer.data.cache import ENV_VAR, default_cache_root
from mumdia_viewer.data.hashing import blake3_file, verify


def test_cache_root_follows_the_environment(monkeypatch, tmp_path: Path):
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "c"))
    assert default_cache_root() == tmp_path / "c"
    monkeypatch.delenv(ENV_VAR)
    assert default_cache_root().name == "mumdia-viewer"


def test_arrays_and_json_round_trip(tmp_path: Path):
    cache = Cache(tmp_path / "c")
    cache.save_arrays("b3:abcdef", "idx", a=np.arange(5))
    cache.save_json("b3:abcdef", "meta", {"x": 1})
    fresh = Cache(tmp_path / "c")
    assert np.array_equal(fresh.load_arrays("b3:abcdef", "idx")["a"], np.arange(5))
    assert fresh.load_json("b3:abcdef", "meta") == {"x": 1}
    assert fresh.load_arrays("b3:other", "idx") is None


def test_cache_refuses_to_write_inside_a_run_directory(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    cache = Cache(run / "cache")  # a misconfigured cache inside the run directory
    cache.forbid(run)
    assert not cache.writable
    cache.save_arrays("b3:ab", "x", a=np.arange(3))  # kept in memory only
    assert not list(run.rglob("*.npz"))
    assert np.array_equal(cache.load_arrays("b3:ab", "x")["a"], np.arange(3))


def test_unwritable_cache_degrades_to_memory(tmp_path: Path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    cache = Cache(blocker / "sub")
    assert not cache.writable
    cache.save_json("b3:ab", "m", [1])
    assert cache.load_json("b3:ab", "m") == [1]


def test_blake3_matches_the_recorded_content_hash(open_fixture):
    rs = open_fixture("single")
    for kind in ("psms_scored", "run_windows", "peptide_quant"):
        artifact = rs.runs[0].artifact(kind)
        assert blake3_file(artifact.path) == artifact.content_hash


def test_verify_detects_a_changed_file(fixture_dir, tmp_path: Path):
    copy = tmp_path / "run"
    shutil.copytree(fixture_dir("single"), copy)
    cache = Cache(tmp_path / "cache")
    rs = open_results(copy, cache=cache)
    artifact = rs.runs[0].artifact("peptide_quant")
    assert verify(artifact, cache).verdict == "match"
    with open(artifact.path, "ab") as fh:
        fh.write(b"tail")
    st = os.stat(artifact.path)
    os.utime(artifact.path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    assert verify(artifact, cache).verdict == "mismatch"


@pytest.mark.parametrize("kind", ["features", "psms_competed"])
def test_hard_linked_features_share_one_hash(open_fixture, kind):
    run = open_fixture("single").runs[0]
    assert run.artifact(kind).content_hash == run.artifact("features").content_hash
