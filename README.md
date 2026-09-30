# visDIA: MuMDIA results viewer

`mumdia-viewer` is an interactive, read-only viewer for the outputs of
[MuMDIA](https://github.com/CompOmics/MuMDIA), a DIA proteomics search engine.
It opens a run or an experiment directory and shows the identifications, the
evidence behind each one (XICs, spectra, decoy competition), the calibrations,
the quantification, and the differences between two result sets.

Status: milestone 1 (the data layer) is done; the user interface follows in
milestone 2. The data layer reads MuMDIA v0.5.0 outputs (and the schema versions of
earlier releases) and never writes to a run directory.

## Install (development)

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; use `source .venv/bin/activate` elsewhere
pip install -e ".[dev]"
```

Python 3.11 or newer.

## The data layer

`mumdia_viewer.data` is a plain Python API that returns pandas DataFrames, numpy arrays
and dataclasses, for use in a notebook or another application:

```python
from mumdia_viewer.runtime import configure_environment
configure_environment()                      # before numpy/pyarrow: bounded threads and memory

from mumdia_viewer.data import open_results
from mumdia_viewer.data import counts, tables
from mumdia_viewer.data.detail import precursor_detail, mirror

rs = open_results("path/to/run-or-experiment")
for c in counts.unit_counts(rs, 0.01):
    print(c.label)                           # e.g. "81,310 peptides (unique base_peptide_id, peptide_q_value <= 0.01)"

page = tables.identification_table(rs, tables.TableQuery(unit="precursor", limit=20))
detail = precursor_detail(rs, rs.runs[0], int(page.rows.candidate_id.iloc[0]))
spectrum = mirror(rs, detail)                # apex MS2 scan against the predicted fragments
```

See [docs/data-layer.md](docs/data-layer.md) for how artifacts are found, versioned,
cached and counted.

## Tests

```bash
pytest                                        # hermetic tests on tests/fixtures
MUMDIA_VIEWER_REAL_SINGLE=<run dir> MUMDIA_VIEWER_REAL_EXPERIMENT=<experiment dir> pytest -m real_data
python benchmarks/m1_performance.py <run dir> # open, precursor detail and memory, cold and warm
```

Licence: Apache-2.0.
