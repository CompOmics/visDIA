# M2: P0 checklist

Manual checklist of the P0 views on the Astral runs (MuMDIA 0.5.0, HYE library,
`configs/examples/diann-library.json`):

- single run `astral_A1_single` (683,297 scored rows);
- six-run experiment `astral_6run`;
- the HYE FASTA for coverage (`ProteoBenchFASTA_MixedSpecies_HYE.fasta`, 31,889 proteins).

Screenshots are in [screenshots/m2](screenshots/m2) (1440 x 900, Chromium).

Legend: **yes** works and was checked; **partly** works with the stated limit.

## 1. Run overview

| Spec item | State | Where |
|---|---|---|
| Version, git SHA, configuration and inputs from the manifest | yes | Identity chips (version, git SHA and date in the tooltip, rescorer, competition key, RT and fragment models, quant filter); tabs Inputs, Artifacts, Engine report, Configuration (`config_json`), Command line. [01](screenshots/m2/01_overview_light.png) |
| Stage timings from the reports | yes | "Stage timings" chart (one colour per run in an experiment) and the stages without a report named. [01](screenshots/m2/01_overview_light.png), [03](screenshots/m2/03_overview_experiment.png) |
| Counts at the threshold for every unit, with the q column named | yes | Four cards (PSMs, precursors, peptides, protein groups), each with its q column and its row unit. The threshold slider shows the exact counts at each stop while it is dragged. |
| Identification curves (count against q) per unit | yes | Exact counts of the engine's columns at 37 thresholds on a log axis; linear or log y. |
| Target and decoy score histograms | yes | "Score distribution" (targets, decoys, spike-ins when present); linear or log y. |
| For experiments, per-run counts on `run_psm_q` | yes | "Per run" chart and table; precursors, peptides and protein groups per run are labelled as derived from the accepted PSMs. [03](screenshots/m2/03_overview_experiment.png) |
| Engine agreement | yes | The viewer's counts at q ≤ 0.01 beside the rescore report's statistics: equal for all four units on both runs. |
| Light and dark mode | yes | [02](screenshots/m2/02_overview_dark.png) |

## 2. Identification browser

Built as PeptideShaker's Overview tab: linked panels on one screen.

| Spec item | State | Where |
|---|---|---|
| Precursor, peptide and protein tables | yes | Starting level "Protein groups" (protein groups, then the peptides of the selected group beside the precursors of the selected peptide, then the preview), "Peptides" or "Precursors". [04](screenshots/m2/04_identifications_proteins.png), [06](screenshots/m2/06_identifications_peptides.png), [07](screenshots/m2/07_identifications_experiment.png) |
| Search | yes | Peptidoform or protein substring; the header search lands here from every page. |
| Filters on q column and threshold, charge, label, protein, modification and quant status | yes | Filters popover (seven q columns; the header threshold), the decoy switch, removable chips for the active filters. |
| Server-side sorting and paging | yes | The first table uses AG Grid's infinite model on `identification_table`; child tables hold every row of their parent (up to 2,000 at once). |
| A row opens its detail page | yes | Enter, a double click or the row's open icon; a single click selects and fills the panels below. |
| PeptideShaker elements | yes | Validation column (check, cross, D, E), in-cell bars on fixed scales stated in the header tooltips, coloured modifications, b blue and y red, panel counts, keyboard selection, the selection in the address. |
| Sequence coverage (P1, done early) | yes | Coverage bar above the peptides of the selected group, with `--fasta`; a click selects the peptide; members of a group can be switched. [04](screenshots/m2/04_identifications_proteins.png), [05](screenshots/m2/05_identifications_dark.png) |

## 3. Precursor detail

| Spec item | State | Where |
|---|---|---|
| Every fragment trace over retention time | yes | Fragment XICs, b ions blue and y ions red, one colour per fragment. [08](screenshots/m2/08_precursor_light.png) |
| Markers: `apex_rt`, elution bounds, integration bounds, `rt_pred_cal`, RT window | yes | With a key; the RT window is marked at the plot's edge when it is wider than the plot. |
| MS1 isotope traces in a second panel | yes | On the same RT axis as the fragments. |
| Alternative peaks (`peak_rank` ≥ 1) marked | yes | Arrows above the plot; checked on the `topk` fixture (the Astral run has none). |
| Observed MS2 spectrum of the window nearest `apex_rt` against the predicted fragments | yes | Mirror plot; "Matched" or "Base peak" scaling. |
| Matched peaks with fragment name and ppm error; the tolerance used shown | yes | Labels with the raw ppm; the tolerance and offset with their source (the extraction report). |
| Step to the neighbouring scans | yes | Previous, apex and next buttons, the scan slider, the arrow keys, and a click on the XIC. |
| Every q value with its unit, the score, `prelim_score`, the rescorer identity | yes | Verdict tiles and the q table; a grouped column on a row that is not its group's winner says so and names the winner's value. |
| Matched fragments, co-elution, fragment correlation, spectral angle, mass error | yes | Evidence cards with the run's target and decoy percentiles. |
| RT error relative to the window; MS1 support; `contested_frac`; peak rank | yes | RT card with a gauge; MS1 and interference cards. |
| Quant status and quantity | yes | Quantification card: identification and integration bounds, fragments used, the reason. |
| Decoy partner | yes | Exact library partner with its score and q; "not scored" when it was not. [10](screenshots/m2/10_precursor_decoy.png) |
| Features as percentiles of the run's targets and decoys | yes | Feature chart with the population named. |
| PeptideShaker elements | yes | Sequence fragmentation diagram, ladder ion table, linked hover between the XIC, the spectrum and the tables, validation marks. |
| Experiments | yes | Run chip and the run's `run_psm_q`. [11](screenshots/m2/11_precursor_experiment.png) |
| Protein coverage card | yes | Bar, peptide lanes and sequence text; a click opens the peptide in the identifications. [12](screenshots/m2/12_coverage_card.png) |

## Performance

Measured with `benchmarks/m2_ui_performance.py`. Each scenario runs in a fresh process with the OS file cache of the run directory purged. Each page is posted to the router as the browser posts it, so building and serialising the page are included; drawing in the browser is not.

| Target | Single run | Six-run |
|---|---|---|
| Open in under 5 s (import, open, app, shell, overview) | 2.5 s | 2.2 s |
| Precursor detail, cold, under 3 s | median 0.84 s, max 1.14 s (10 candidates) | median 1.01 s, max 1.63 s |
| Precursor detail, warm, under 1 s | median 0.21 s, max 0.30 s | median 0.22 s, max 0.31 s |
| Memory under 2 GB | 519 MB after 10 precursor pages | 602 MB |
| Overview page | 0.57 s | 1.02 s |
| Identification page | 0.34 s | 0.49 s |

Browser timings from the builders' playwright measurements (headless Chromium, localhost):

- a protein-group click fills the child panels in 0.16 s (median, single run);
- the preview is complete in about 0.9 s;
- one group's coverage takes about 0.17 s on the server.

## Known limits

- On the six-run experiment, a click on a protein group the browser has not loaded yet fills the child panels in 0.6 to 1.0 s (target 0.5 s). Prefetched neighbours take 0.14 to 0.30 s.
- A cold first table appears 1.1 s after navigation on the single run, and 1.5 s on the six-run (target 1 s).
- Child tables with more than 2,000 rows load in blocks of 2,000 (titin: 2,572 peptides).
- In the infinite first table, a selection restored from an address is highlighted only once its block has loaded. The panels below show it either way.
- The competition figure's axis labels show the raw peptidoform; Plotly cannot draw the coloured tags there. The table under it uses the coloured style.
- Fragments hidden on the precursor page stay visible in the static preview.
