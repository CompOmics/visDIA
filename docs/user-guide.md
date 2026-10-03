# User guide

`mumdia-viewer` shows what MuMDIA found in a run or an experiment, and why. It reads the
output directory and never writes to it.

## Install

```bash
pip install mumdia-viewer            # from a wheel or the repository: pip install .
```

You need Python 3.11 or newer. The dependencies are pyarrow, DuckDB, pandas, numpy, Dash,
dash-mantine-components, dash-ag-grid, Plotly and blake3. All are MIT, BSD or Apache-2.0
licensed. Nothing is fetched from the internet at run time.

## Start

```bash
mumdia-viewer <run-or-experiment-dir> [--fasta proteins.fasta] [--compare <other-dir>] [--port N]
```

| Option | What it does |
|---|---|
| `--fasta PATH` | Protein sequences for the coverage views. The option can be repeated. Without it, the viewer uses the FASTA that a run searched from a FASTA recorded, if that file is still there. |
| `--compare DIR` | A second result set for the compare view. It is served as a full viewer at `<address>b/`. |
| `--port N` | The port; the default is the first free port from 8050. |
| `--no-browser` | Do not open a browser (use it on a server). |
| `--remap OLD=NEW` | Inputs recorded under `OLD` are now under `NEW`. |
| `--host` | The address to bind. The default is 127.0.0.1; any other value exposes the viewer to the network. |
| `--no-token` | Serve at `/` instead of a random path. Use it only on a single-user machine. |

**On a remote server**, start the viewer with `--no-browser`. Then forward the port:
`ssh -L <port>:127.0.0.1:<port> <user>@<server>`. Open the printed address, including
its token, on your own machine.

## Rules the viewer keeps

- **Engine columns.** Every number is one of MuMDIA's columns. A number the viewer computes
  is marked "derived" or "viewer". Examples: a percentile, a coverage, a CV, a TIC, or a
  spectrum match outside the engine. No q value is ever recomputed.
- **Counts.** Every count names its row unit and its q column, for example
  "81,310 peptides (unique base_peptide_id, peptide_q_value ≤ 0.01)". The q threshold in
  the header applies everywhere.
- **Grouped q columns.** `precursor_q`, `peptide_q_value` and `pg_q_value` are set on each
  group's winning row only. In an experiment they are experiment-wide. Per-run counts use
  `run_psm_q`.
- **Decoys and quantities.** Decoys are hidden in the identification tables, with a switch
  to show them. A missing quantity is "not quantifiable" or "not selected", never 0.
- **Retention times** are in seconds.

## Pages

| Page | What it shows |
|---|---|
| **Overview** | Version, configuration, inputs and stage timings. Counts per unit, with a threshold slider that shows exact counts at each stop. Identification curves, score distributions and per-run counts. The engine-report check. A **Report** button downloads a static, offline HTML file. |
| **Identifications** | Linked panels, as in PeptideShaker's Overview: protein groups, then the peptides of the selected group (with sequence coverage), then the precursors of the selected peptide, then a preview of the selected precursor. Click or use the arrow keys to select; Enter opens the precursor page. **TSV** downloads every row under the filters. |
| **Precursor detail** | Why one identification was accepted. Fragment and MS1 XICs linked to the spectrum mirror: click a scan, or use the slider or the ← → keys. The fragmentation diagram and the ion table. Every q value with its unit and winner, evidence cards with percentiles, the base-peptide competition, the decoy partner, quantification, coverage, and your verdict. |
| **Protein** | A protein group: its members, a verdict strip, linked peptides and precursors, sequence coverage, and the quantity per run (experiments). |
| **Calibration** | RT anchors rebuilt with the engine's rule and checked against `cal.json`. The fitted map and the window. In-sample residuals, which are fit diagnostics, not error estimates. The RT error of accepted identifications. Fragment mass calibration. |
| **Run QC** | TIC and base peak, identifications across RT, scan rate, isolation windows, peaks per MS2 spectrum (the `--top-peaks-ms2` check), and the charge, length, missed-cleavage and modification distributions. |
| **Quant QC** | Quantity distributions, missing values, quant states, CVs within conditions, the LFQ matrix and protein profiles. Conditions are suggested from the mzML names and can be edited. |
| **Condition ratios** | log2 A/B per species (by protein-name suffix), with expected ratios you enter. |
| **Spectrum browser** | Any scan by index or RT, stepping within its window, and the candidates near it with their fragment matches. |
| **Across runs** | One precursor in every run of an experiment: XICs, quantities with MBR transfers, and q values. Open it from the precursor page. |
| **Compare** | Two result sets: overlap per unit, score and quantity scatters, and the identifications unique to each side, with links into both viewers. It also shows configuration differences. |
| **Notes** | Your verdicts. Give one on the precursor page (keys A accepted, R rejected, U unsure, or the card). Export them as TSV or JSON. |

Screenshots: [screenshots/m2](screenshots/m2) and [screenshots/views](screenshots/views).

## Where the viewer keeps its files

| What | Where |
|---|---|
| Cache (derived data, safe to delete) | `MUMDIA_VIEWER_CACHE_DIR`, else `%LOCALAPPDATA%\mumdia-viewer` on Windows, `~/Library/Caches/mumdia-viewer` on macOS, `~/.cache/mumdia-viewer` on Linux |
| Validation notes (your data) | `MUMDIA_VIEWER_NOTES_DIR`, else `%APPDATA%\mumdia-viewer\notes` on Windows, `~/Library/Application Support/mumdia-viewer/notes` on macOS, `~/.local/share/mumdia-viewer/notes` on Linux. There is one JSON file per result set, keyed by the content hash of its scored table. |
| Browser settings (theme, conditions, expected ratios) | The browser's local storage |

## From Python

The data layer works without the UI, for example in a notebook. See the
[README](../README.md) and [data-layer.md](data-layer.md).

## Troubleshooting

- **"not a result directory"**: open the directory that holds `manifest.json` (a run) or
  `experiment_manifest.json` (an experiment).
- **Coverage says "start the viewer with --fasta"**: the run was searched with a spectral
  library. Pass the FASTA the library was built from.
- **An input shows "needs remap"**: the run directory or its inputs moved. Pass
  `--remap <old prefix>=<new prefix>`.
- **The first visit of a page is slower**: the viewer builds indexes and caches, such as
  the MS2 TIC or the candidate index, and keeps them in its cache directory.
