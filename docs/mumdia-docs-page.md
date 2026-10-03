<!--
Draft for the MuMDIA repository: docs/34_results_viewer.md, linked from docs/README.md.
It is not pushed there; the maintainers decide whether and when to add it.
-->

# 34. Results viewer (mumdia-viewer)

`mumdia-viewer` is an interactive, read-only viewer for MuMDIA outputs. It lives in its
own repository (visDIA). It opens a run directory (`manifest.json`) or an experiment
directory (`experiment_manifest.json`) and never writes to it.

```bash
pip install mumdia-viewer
mumdia-viewer <run-or-experiment-dir> [--fasta proteins.fasta] [--compare <other-dir>]
```

On a server, start it with `--no-browser` and forward the port. Use
`ssh -L <port>:127.0.0.1:<port> <user>@<server>`, then open the printed address.

## What it shows

- **Overview:** version, configuration, inputs, stage timings, counts per unit with their
  q columns, identification curves, and score distributions. Experiments also show
  per-run counts on `run_psm_q`.
- **Identifications:** protein groups, peptides and precursors as linked panels, with
  search, filters and sequence coverage (with a FASTA).
- **Precursor detail:** fragment and MS1 XICs with the identification, integration and
  window markers, linked to the spectrum mirror. Also every q value with its unit, the
  evidence features with the run's percentiles, the base-peptide competition, the decoy
  partner and quantification.
- **Calibration, run QC, quant QC, protein view, condition ratios, spectrum browser,
  experiment views, compare, validation notes, export.**

## How it reads the outputs

- The viewer counts the engine's own columns. It never recomputes q values, scores or
  quantities. Any number it derives (a percentile, a coverage, a CV, a TIC) is labelled
  so.
- **Grouped q columns:** `precursor_q`, `peptide_q_value` and `pg_q_value` are read as
  set on each group's winning row, and are experiment-wide in an experiment. Per-run
  counts use `run_psm_q`.
- **Chromatograms:** schema v1 and v2 are decoded as `mumdia::chromatograms::Decoder`
  does (docs/15).
- **RT anchors** are rebuilt with the rule of `rt_im_train.rs`. The viewer checks them
  against `cal.json`.
- **Artifacts** are found through the manifest and the `.report.json` files. Versions
  come from the records; an unknown schema version is refused with a message.
- **Derived data** (indexes, cached signals) is kept in the viewer's own cache directory.
  Validation notes are kept in its notes directory.

The viewer's user guide (`docs/user-guide.md` in visDIA) describes every page and option.
