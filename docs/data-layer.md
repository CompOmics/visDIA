# The data layer of mumdia-viewer

`mumdia_viewer.data` is a plain Python API over the outputs of MuMDIA. It returns
pandas DataFrames, numpy arrays, dataclasses and dicts, and it does not import the user
interface, so a notebook or the desktop application can use it on its own. This page
describes how it finds, reads and labels the engine's artifacts. The rules come from the
engine's source and data dictionary (MuMDIA v0.5.0) and were checked on real outputs.

## Principles

1. **Read-only.** Nothing is written to a run directory, a library or an input. Derived
   data (candidate indexes, hash verdicts, DuckDB spill files) goes to the viewer's cache
   (see [Cache](#cache)). `features.parquet` and `psms_competed.parquet` can be one
   hard-linked file, so an edit of either would change both; neither is ever opened for
   writing.
2. **No recomputed results.** The viewer counts the engine's own q columns; it does not
   recompute a q value, a score or a quantity and present it as the engine's. Every
   derived number is labelled with how it was computed.
3. **Every count names its unit.** A count is a triple of row unit, distinct key and q
   column, for example `81,312 peptides (unique base_peptide_id, peptide_q_value <= 0.01)`.
4. **Refuse what is not known.** An unknown schema version is refused with a message that
   names the file, the artifact, the version found and the versions supported. Unknown
   extra columns are ignored. Ion mobility is optional.
5. **Never load a whole large table.** Per-candidate reads use a candidate index and read
   one row group with a column projection. Aggregates run in DuckDB over the parquet files
   with projection.
6. **No long-lived file handles.** On Windows an open handle blocks the engine from
   replacing or deleting a file (the engine publishes every artifact by rename). Footers
   are cached; a file is opened only for the duration of one read.

## Opening a directory

```python
from mumdia_viewer.data import open_results

rs = open_results("path/to/run-or-experiment")
rs.kind          # "run" or "experiment"
rs.runs          # list of Run; run.index is the `source` value of its rows
rs.scored        # the pooled scored table (psms_scored, or scored_combined)
rs.notices       # what the user should know: moved paths, missing files, ...
```

A directory is classified by its manifest:

| file | kind | runs |
|---|---|---|
| `experiment_manifest.json` | experiment | `experiment.runs`; run `i` is `<root>/<runs[i]>` and holds the rows with `source == i` |
| `manifest.json` | single run | one run; grouped when `groups/plan.json` exists or a key ends in `[gNN]` |

An experiment manifest records only the pooled scored tables, the per-run `scored`,
`peptide_quant` and `protein_group_quant` tables and the LFQ matrix. Every other per-run
artifact (chromatograms, features, spectra, windows, seeds) is found by its fixed file
name and described by its own `<file>.report.json`.

### Moved directories

The engine records artifact paths as `<--out-dir as typed>/<relative path>`, which may be
absolute or relative and on Windows mixes separators. The resolver strips the recorded
`--out-dir` (taken from `cli_args`, compared without case for Windows paths) and joins the
rest to the directory that was opened. If that file does not exist, the artifact is
missing: the old location is never read in its place, because it may belong to another
run. Paths outside the run directory (an input library, a FASTA file) are looked up
through user-supplied root remaps, then as recorded. A bare file name is never searched
for, because file names repeat across runs and bands.

### Grouped runs

With `groups.window_groups > 1` the run holds one directory per band under `groups/`.
By default (`groups.pool_chromatograms` false) there is no run-level chromatogram table:
the band tables, minus each band's overlap losers, are the run's chromatograms. The loser
file's footer key `mumdia.overlap_losers.band_tables` lists the band tables in pool order,
and the `band` column is a position in that list, not the `gNN` number (bands skipped at
run time shift the positions). A candidate must yield rows from exactly one table.
Band `features` and `psms_extracted` are usually deleted after pooling
(`groups.delete_band_intermediates`) although the manifest still lists them; they are
reported as "deleted after pooling", not as errors.

## Schema versions

A parquet footer carries no schema name or version. The version of an artifact is read
from its manifest record, then from its report. When the two disagree, the report wins,
because it is written together with the file (a single-stage re-run rewrites the table and
its report but not the manifest); the disagreement is reported, and the file is refused
when either version is unsupported. The few files the engine writes with neither record
(the per-run LFQ siblings, the multi-head library table, the MBR transfer table) get a
version inferred from their columns, labelled as inferred; an ion-mobility column in a
schema whose 3D versions have none infers the diaPASEF version.

| schema | versions read | notes |
|---|---|---|
| `psms_scored` | 4; 3 (pre-release builds, no `selected_peak_rank`) | |
| `psms_competed` | 3, 4; 2 (pre-release) | |
| `features` | 1 (float64), 2 (float32) | |
| `psms_extracted` | 2; 1 (pre-release); 4 (diaPASEF); 5, 6, 8, 9 (ZT Scan); 7 | 5: Q1 profile, 6: per-fragment Q1 decisions, 8: precursor and multi-scale Q1 scores, 9: separable RT x Q1 model; 7: MS1 isotope envelope |
| `chromatograms` | 1, 2; 3, 4 (diaPASEF: per-point 1/K0); 5, 6 (ZT Scan: one point per Q1 cycle and `q1_class`) | 3/4 only when `features.im_features` or retrace is on; 5/6 under `extract.q1_traces` |
| `spectra_ms1`, `spectra_ms2`, `isolation_windows`, `fragment_library_precursors`, `seed_psms` | 1; 2 (diaPASEF); `spectra_ms2` 3 (ZT Scan: `q1_cycle`, `q1_step`) | |
| `peptide_quant` | 2; 3 (`first_pass_id`), 4 (`q1_quant`) | |
| `protein_group_quant` | 2 | |
| `fragment_quant` | 1; 5, 6 (the cross-run export; 6 under a pattern rule) | an experiment's run tables are the cross-run step's first pass |
| `quant_fragments` | 1 | written under `quant.fragment_detail` |
| every other artifact | 1 | |

The diaPASEF versions are what MuMDIA ≥0.7 writes for a native Bruker `.d` (timsrust) and
are read by default (`schemas.DIAPASEF`). Versions written only by the old ion-mobility
engine branch (spectra v3, `psms_extracted` v3 and v5) are refused unless
`allow_unreleased=True`. MuMDIA 0.12 reuses two of those numbers for SCIEX ZT Scan DIA
(`spectra_ms2` v3, `psms_extracted` v5); a file of either is read as the ZT Scan version
only when it holds the ZT Scan columns (`q1_cycle` and `q1_step`, `q1_corr`;
`schemas.ZTSCAN_MARKERS`), so the ion-mobility branch's files are still refused.

## Cache

Derived data is stored under `MUMDIA_VIEWER_CACHE_DIR`, else `%LOCALAPPDATA%\mumdia-viewer`
(Windows), `~/Library/Caches/mumdia-viewer` (macOS), `$XDG_CACHE_HOME/mumdia-viewer` or
`~/.cache/mumdia-viewer`. Entries are keyed by the artifact's recorded blake3 content hash,
so an entry can never describe a different file. A file without a recorded hash is keyed
by a fingerprint of its size, modification time and footer. The cache refuses to write
inside an opened run directory. The engine's own cache (`MUMDIA_CACHE_DIR`,
`%LOCALAPPDATA%\mumdia\cache`) is a different directory.

Nothing is created on disk before the first write, and a cache root inside an opened
directory is refused before it could create anything there.

Opening a directory never hashes a file. `hashing.verify` compares a file with its
recorded hash on request and caches the verdict by (path, size, modification time).

## DuckDB

Every result set has its own in-memory DuckDB database with a private spill directory,
removed when the database closes: spill files are not safe to share between instances.
The spill directory is under the cache directory, else under the system's temporary
directory; when neither can hold a private directory, spilling is disabled rather than
left at DuckDB's default, `.tmp` in the working directory. Every thread (a Dash
callback) uses its own cursor. After each query the cursor runs a trivial statement, which
ends the previous scan: DuckDB otherwise keeps a parquet file open, which on Windows
blocks the engine from replacing it. Paths given to `read_parquet(?)` have their glob
characters escaped, because DuckDB reads the parameter as a glob: a directory named
`run[1]` would otherwise match `run1` and read another run's table.

The database runs on the usable CPUs, at most 8 threads. Its memory limit
(`duck.resolve_memory_limit`) is `MUMDIA_VIEWER_MEMORY_LIMIT` when it is set, a size in
DuckDB's units (`8GB`, `1.5GiB`, `800MB`) or a share of the physical memory (`50%`); the
command line's `--memory-limit` sets that variable. Otherwise it is a quarter of the
physical memory, at least 1 GiB and at most 16 GiB, where the physical memory is the
machine's or the process's smaller cgroup limit. An explicit value is not clamped. The
limit was a fixed 512 MB before, and a desktop user's protein-group table of an 8.5M-row
scored table failed against it. An out-of-memory error keeps its DuckDB type and its
message states the limit, whether spilling is possible and how to raise the limit.

Two settings are global, because a plain `SET` would apply to the base connection and
not to the cursors that run the queries: the parquet metadata cache, and
`disable_parquet_prefetching`. DuckDB 2 prefetches the projected columns of whole row
groups on every thread, and MuMDIA writes about a million rows per row group; on an
8.5M-row scored table those buffers alone filled the 488 MiB of the old limit, so every
identification table failed there with DuckDB 2.0.0. DuckDB 1 prefetches remote files
only. `preserve_insertion_order` keeps its default: some queries rely on file order
without an `ORDER BY` (the first row of a fragment name wins in
`quant_fragments._fragment_quant_areas`).

## Memory

The fixed caches are the row-group cache (256 MB) and the spectrum cache (96 MB); DuckDB's
limit follows the machine (see DuckDB above) and is a ceiling, not a reservation: on the
fleet's large tables the identification tables used tens to a few hundred MB of it.
`runtime.configure_environment()`, which the command-line entry
point calls before numpy and pyarrow are imported, sets one OpenBLAS thread and Arrow's
system allocator: on the Astral single run the system allocator halved the private memory
after 21 precursor details (to 0.64 to 0.72 GB) for about 17% slower warm details.

## Per-candidate reads

Every candidate-keyed table stores the rows of one candidate contiguously, in ascending
`candidate_id` in all but one case: the pooled chromatograms of overlapping bands are not
globally sorted when a later band wins a candidate. No writer records the order in the
footer, so the candidate index is built from the `candidate_id` column, one row group at a
time, and the order is checked while it is built. The index maps a candidate to its global
row range and to its row-group segments. On the Astral single run the chromatogram index
(10.1 million rows) builds in 0.09 s, and one candidate reads in about 13 ms.

Tables whose `candidate_id` equals the row index (`run_windows`, the library precursor
tables) need no index. The footer statistics detect this; because statistics cannot see a
permutation inside one row group, every such read checks the id it returns.

Chromatograms layout 2 stores the retention-time axis once per candidate per row group and
restarts that rule at every row group. A candidate is therefore decoded one row-group
segment at a time, starting at its first row in each row group.

## Identification counts

| unit | distinct key | q column | notes |
|---|---|---|---|
| PSMs | rows | `q_value` | pooled over every row of the rescore |
| precursors | `(peptidoform, charge)` | `precursor_q` | a base-peptide unit when `compete.group_by = base_peptide` |
| peptides | `base_peptide_id` | `peptide_q_value` | picked target-decoy competition |
| protein groups | `protein_group` (non-empty) | `pg_q_value` | |
| PSMs of one run | rows with `source = i` | `run_psm_q` | the only per-run q column |

The grouped columns (`precursor_q`, `peptide_q_value`, `pg_q_value`) hold a value on the
winning row of each group and 1.0 on every other row. In an experiment the groups are
experiment-wide, so a per-run count on them would keep only about one run in `n_runs` of
the rows; per-run counts use `run_psm_q`. At a threshold of 1.0 a winner cannot be told
from a loser, so grouped counts require a threshold below 1.

The winner of a group follows the engine: the highest score, a decoy on an exact score
tie, then the earlier row of the pooled table. In entrapment mode decoys do not compete
and an entrapment row wins a tie. This rule reproduces the engine's sparse columns on
every fixture and on the Astral run.

The peptide and protein-group tables keep their working set small on large tables. Only
the groups shown are counted, and a protein group is counted under an integer id rather
than its string: a distinct count keyed by the group string held a copy of it per
(group, peptide). When the rows of each group must be ranked (any filter other than the
unit's own q below 1), the row numbers of the picked rows are found on the narrow
columns, by an aggregate for protein groups (few groups of many rows) and by a window
for base peptides (about one row per group), and only the picked rows are read whole.

The engine's `peptides.tsv` lists one precursor (the winner) per accepted base peptide, so
its row count, recorded as `n_precursors` in the experiment manifest, equals the peptide
count and not the precursor count. `overview.engine_report_numbers` returns it with that
label.

## The precursor detail

`detail.precursor_detail` assembles one identification. Only the scored row is required;
every other part is None (or empty) when its artifact is missing or refused, and the
detail's notes say why. After match-between-runs a run's `scored.parquet` holds q values
lowered to min(native q, transfer_q) on transferred rows; the detail shows the rescorer's
native values from `scored_combined.parquet` and lists the lowered values separately, as
the values quant and the report used. The exact decoy partner is looked up per candidate
(one query for the rows of the candidate's library `peptidoform_id`), and only when the
library row carries the scored row's peptidoform and charge, which rejects a library that
is not the one the run searched.

### Mobilograms (diaPASEF)

`mobilogram.centroid_mobilogram(rs, detail, centre_rt=None, frames=2)` returns intensity
against 1/K0 for every predicted fragment and MS1 isotope of a candidate, from the run's
own spectra tables: the frame nearest `centre_rt` (the apex by default) and up to `frames`
frames either side, over every slot that holds the precursor. A fragment's points are every
peak within the extraction's tolerance and mass offset, an MS1 isotope's every peak within
`extract.prec_tol_ppm`, inside the slots' 1/K0 range; convert keeps one 1/K0 per centroid,
so these are sticks, not a profile. `mobilogram.engine_mobilogram(rs, detail, ...)` runs
`mumdia mobilogram` (an engine newer than 0.9.0, `--engine` or `MUMDIA_BIN`) on the `.d` the
spectra report names, through `--remap`, and returns the raw events binned in 1/K0 over the
same frames; its answers are kept in the viewer cache, keyed by the `.d` and input stamps,
the candidate, the centre RT, the frames, the bin width and the engine version, and a call
times out after 120 s (`MUMDIA_VIEWER_MOBILOGRAM_TIMEOUT`). Both return a `Mobilogram`
whose `profile(ion, k)` sums (raw) or collects (centroids) the frames within `k` of the
centre.
### ZT Scan DIA runs

`Run.is_ztscan` is true for a run whose `spectra_ms2` records the Q1 sweep.
`ztscan.ztscan_run(rs, run)` returns its sweep (`ZtSweep`: the cycle and step of every MS2
row, the row of every (cycle, step) cell, each step's target, the matching tolerance) and its
calibrated Q1 model (`Q1Model`, from the `psms_extracted` report's `params.q1_fragments`,
else `cal.json`'s `q1_calibration`), loaded once per file version; it is None for any other
run. `ztscan.CandQ1` is the engine's model of a candidate (`q1frag::CandQ1`): the expected
Q1 centre and each fragment's expected profile, the trapezoid transmission summed over the
precursor's isotopes.

- `ztscan.cycle_traces(rs, detail)` collapses the candidate's chromatogram to one value per
  Q1 cycle: each sample is located in the sweep (the nearest covering spectrum within 0.45
  of the median scan spacing) and the samples of a cycle become `sum(w x) / sum(w^2)` at
  their mean RT (`q1quant::collapse_trace`). On a chromatogram the engine already wrote per
  cycle (versions 5 and 6) it returns those traces, `source = "engine"`.
- `ztscan.q1_patch(rs, detail, half_cycles=3, centre_cycle=None)` reads the candidate's
  RT x Q1-step patch from the spectra, the engine's `read_patch` rule, for the fragments and
  the unfragmented precursor's first three isotopes; only the row groups that hold its cells
  are read (in parallel), and the cell spectra are kept in `ztscan.CELL_CACHE`.
- `ztscan.q1_profile(patch, cycle=None, half=0)` sums the patch over cycles and gives each
  ion's expected profile and least-squares amplitude.
- `ztscan.q1f_row(rs, detail)` is the candidate's `psms_extracted` row with its `q1f_*` and
  `q1s_*` columns, read with DuckDB filtered on `candidate_id`.

## The fragments behind a quantity

`quant_fragments.quant_fragments(rs, run, cid)` says which fragments quant used for a
candidate's quantity, with their areas, ranks and weights. Three sources are tried in order:

1. the engine's `quant_fragments.parquet` (MuMDIA with `quant.fragment_detail`, which the
   desktop application sets): nothing is derived;
2. in a single run, the areas of `fragment_quant.parquet` with the configured top-N rule
   applied by the viewer (`observed_area` by area; every other selection falls back to
   `predicted` in a single run);
3. the chromatograms, integrated as quant integrates them (the engine's float32 trapezoid
   over the reported window), when quant subtracted no baseline, applied no envelope and,
   in an experiment, ran no cross-run step.

A viewer-derived selection is used only when its selected areas reproduce
`peptide_quant.quantity` (exactly, on every fixture), and it is labelled "viewer-derived,
verified". In an experiment the run directories' `fragment_quant.parquet` is the cross-run
step's first-pass export and is never used. Without a source that reproduces the quantity
the account says so and names `quant.fragment_detail`. Under the pattern rules the engine's
quantity is a fit, not a sum of areas: the table names the fitted fragments and the check is
their count. `fragment_areas_across` gives one precursor's run x fragment areas for the
across-runs page.
