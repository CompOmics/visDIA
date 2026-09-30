from pathlib import Path

from mumdia_viewer.data.paths import (
    PathResolver,
    common_prefix,
    is_absolute_recorded,
    is_windows_origin,
    recorded_out_dir,
    split_parts,
)


def test_split_parts_handles_both_separators():
    assert split_parts("C:\\Users\\x\\res/r0/peptide_quant.parquet") == [
        "C:",
        "Users",
        "x",
        "res",
        "r0",
        "peptide_quant.parquet",
    ]
    assert split_parts("/home/robbin/out//psms.parquet") == [
        "home",
        "robbin",
        "out",
        "psms.parquet",
    ]
    assert split_parts("./out/./x.parquet") == ["out", "x.parquet"]


def test_path_kinds():
    assert (
        is_absolute_recorded("C:/x")
        and is_absolute_recorded("/x")
        and is_absolute_recorded("\\\\s\\x")
    )
    assert not is_absolute_recorded("test_data/fixture.fasta")
    assert is_windows_origin("C:/x") and is_windows_origin("out\\x") and not is_windows_origin("/x")


def test_recorded_out_dir_long_forms_only():
    assert recorded_out_dir(["mumdia", "run", "--out-dir", "C:/o", "--threads", "2"]) == "C:/o"
    assert recorded_out_dir(["mumdia", "run", "--out-dir=rel/o"]) == "rel/o"
    assert recorded_out_dir(["mumdia", "run", "-o", "x"]) is None
    assert recorded_out_dir(None) is None


def test_common_prefix():
    assert common_prefix(["/a/b/c.parquet", "/a/b/spectra/d.parquet"]) == "a/b"
    assert common_prefix(["C:\\A\\b\\c.parquet", "c:/a/B/d.parquet"]) == "C:/A/b"
    assert common_prefix(["x.parquet"]) is None


def test_rerooting_strips_the_recorded_out_dir(tmp_path: Path):
    (tmp_path / "spectra").mkdir()
    (tmp_path / "spectra" / "spectra_ms2.parquet").write_bytes(b"x")
    resolver = PathResolver(tmp_path, "C:\\Users\\old\\Run")
    got = resolver.resolve("c:/users/OLD/run/spectra/spectra_ms2.parquet")
    assert got.how == "rerooted" and got.path == tmp_path / "spectra" / "spectra_ms2.parquet"
    assert got.inside_root


def test_missing_inside_root_never_falls_back_to_the_old_location(tmp_path: Path):
    old = tmp_path / "old"
    old.mkdir()
    (old / "features.parquet").write_bytes(b"old run")
    new = tmp_path / "new"
    new.mkdir()
    resolver = PathResolver(new, str(old))
    got = resolver.resolve(str(old / "features.parquet"))
    assert got.how == "missing" and got.path is None and got.inside_root


def test_relative_out_dir_is_stripped_component_wise(tmp_path: Path):
    (tmp_path / "psms_scored.parquet").write_bytes(b"x")
    resolver = PathResolver(tmp_path, "out")
    assert resolver.resolve("out/psms_scored.parquet").path == tmp_path / "psms_scored.parquet"
    assert resolver.resolve("other/psms_scored.parquet").how == "missing"


def test_outside_paths_use_remaps_then_the_recorded_path(tmp_path: Path):
    lib = tmp_path / "libs" / "lib_frag.parquet"
    lib.parent.mkdir()
    lib.write_bytes(b"x")
    resolver = PathResolver(tmp_path / "run", "D:/runs/a", remaps={"D:\\libs": tmp_path / "libs"})
    got = resolver.resolve("d:/LIBS/lib_frag.parquet")
    assert got.how == "remapped" and got.path == lib and not got.inside_root
    assert resolver.resolve(str(lib)).how == "as_recorded"
    # A relative path outside the out-dir has no known base: missing unless remapped.
    assert resolver.resolve("test_data/fixture.fasta").how == "missing"
