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
from its manifest record, then from its report. The few files the engine writes with
neither (the per-run LFQ siblings, the multi-head library table, the MBR transfer table)
get a version inferred from their columns, labelled as inferred.

| schema | versions read | notes |
|---|---|---|
| `psms_scored` | 4; 3 (pre-release builds, no `selected_peak_rank`) | |
| `psms_competed` | 3, 4; 2 (pre-release) | |
| `features` | 1 (float64), 2 (float32) | |
| `psms_extracted` | 2; 1 (pre-release) | |
| `chromatograms` | 1, 2 | 3 and 4 (ion mobility, PR #140) need `allow_unreleased=True` |
| `peptide_quant`, `protein_group_quant` | 2 | |
| every other artifact | 1 | the ion-mobility branch raises spectra, windows, seeds and the library precursors; opt-in only |

## Cache

Derived data is stored under `MUMDIA_VIEWER_CACHE_DIR`, else `%LOCALAPPDATA%\mumdia-viewer`
(Windows), `~/Library/Caches/mumdia-viewer` (macOS), `$XDG_CACHE_HOME/mumdia-viewer` or
`~/.cache/mumdia-viewer`. Entries are keyed by the artifact's recorded blake3 content hash,
so an entry can never describe a different file. A file without a recorded hash is keyed
by a fingerprint of its size, modification time and footer. The cache refuses to write
inside an opened run directory. The engine's own cache (`MUMDIA_CACHE_DIR`,
`%LOCALAPPDATA%\mumdia\cache`) is a different directory.

Opening a directory never hashes a file. `hashing.verify` compares a file with its
recorded hash on request and caches the verdict by (path, size, modification time).

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

The engine's `peptides.tsv` lists one precursor (the winner) per accepted base peptide, so
its row count, recorded as `n_precursors` in the experiment manifest, equals the peptide
count and not the precursor count. The viewer labels it that way.
