# Soiling Analysis

End-to-end diagnostics for utility / C&I solar plants from inverter-string
telemetry. Detects and quantifies **soiling**, **curtailment**, and
**degradation**, classifies per-string health, and produces an Excel report plus
interactive HTML dashboards.

## Project layout

```
.
├── pyproject.toml              # single source of truth for deps + commands
├── uv.lock
├── data/                       # local inputs (spec workbook, fetched CSVs) — git-ignored
│   └── RequireInputData.xlsx   # static spec workbook (Plant / Inverter / Panel sheets)
├── output/                     # generated reports + figures — git-ignored
├── src/
│   └── soiling_analysis/
│       ├── cli.py              # `soiling` entry point
│       ├── loader.py           # CSV + spec workbook → pipeline inputs
│       ├── inputs/             # spec-workbook parsing + preflight checks
│       ├── diagnostics/        # the analysis engine (was `pv_diag`)
│       └── mongo_export/       # MongoDB → CSV fetcher (`soiling-fetch`)
├── tests/                      # pytest suite (mirrors the package)
└── canBeDeleted/               # archived junk — safe to delete after review
```

## Setup

Uses [uv](https://docs.astral.sh/uv/) for everything.

```bash
uv sync          # create the venv and install all deps (incl. dev tools)
```

Copy `.env.example` to `.env` and set `MONGO_URI` if you need to fetch data.

## Commands (uv short commands)

| Command | What it does |
| --- | --- |
| `uv sync` | Install/refresh the virtual environment from `pyproject.toml` / `uv.lock`. |
| `uv run soiling --csv data/measured.csv` | Run the full diagnostics pipeline on a measured CSV (uses `data/RequireInputData.xlsx` by default). Writes the Excel report + HTML dashboards to `output/`. |
| `uv run soiling-fetch` | Interactive MongoDB → CSV fetcher. Pick a plant + date range; saves the long-format CSV into `data/`. |
| `uv run pytest` | Run the test suite. |
| `uv run pytest tests/diagnostics -q` | Run just the engine tests. |

### Common pipeline options

```bash
uv run soiling --csv data/measured.csv \
  --specs data/RequireInputData.xlsx \
  --out-dir output \
  --cluster-method combined \   # combined | mppt | orient
  --n-jobs -1 \                 # -1 = all cores
  --tariff 38 \                 # PKR/kWh override
  --wash-cost 5000 \            # PKR/string — enables cleaning-economics advice
  --no-figures --quiet          # optional
```

Run `uv run soiling --help` for the full list.

## Typical workflow

```bash
uv sync
uv run soiling-fetch                       # MongoDB → data/<plant>_<range>.csv
uv run soiling --csv data/<that file>.csv  # → output/<plant>_soiling_results.xlsx + dashboards
```

## Inputs

1. **`data/RequireInputData.xlsx`** — the static spec workbook (plant location,
   per-string panel specs, orientation, commissioning dates, tariff). Authored by
   hand in Excel, parsed once by `soiling_analysis.inputs.specs`.
2. **Measured CSV** — long-format telemetry exported from MongoDB by
   `soiling-fetch` (one `level` column tags plant / inverter / mppt / string rows).

All physics and baselines come from the workbook — there are no hard-coded plant
defaults in the pipeline.

## Output

Written to `--out-dir` (default `output/`):

- `<plant>_soiling_results.xlsx` — multi-sheet diagnostics report.
- `soiling_dashboard__<string>.html` — per-string interactive dashboards.
- `plant_overview.html`, `data_quality__<plant>.html` — plant-level views.

## Testing

```bash
uv run pytest          # full suite
```

> Note: two `sdm_test.py` cases (`test_c_iv_metrics_at_stc_clean`,
> `test_e_voltage_degraded_voc_ratio`) are known pre-existing failures unrelated
> to packaging.
