"""Command-line entry point for the soiling-analysis pipeline.

Runs the end-to-end diagnostics on a measured CSV (exported from MongoDB via
``soiling-fetch``) plus the static spec workbook ``RequireInputData.xlsx``.

Invoke via the console script defined in ``pyproject.toml``::

    uv run soiling --csv data/measured.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

from soiling_analysis.loader import load_from_sources
from soiling_analysis.diagnostics.excel_export import export_results_to_excel
from soiling_analysis.diagnostics.pipeline import run_pipeline_from_frame
from soiling_analysis.diagnostics.plotting import make_all_figures

DEFAULT_SPECS = "data/RequireInputData.xlsx"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="soiling",
        description="Run the PV soiling/curtailment diagnostics pipeline on measured CSV data.",
    )
    parser.add_argument("--csv", required=True, help="Path to the measured CSV exported from MongoDB")
    parser.add_argument("--specs", default=DEFAULT_SPECS, help="Path to RequireInputData.xlsx spec workbook")
    parser.add_argument("--out-dir", default="output", help="Directory to write output files")
    parser.add_argument(
        "--cluster-method", default="combined", choices=["combined", "mppt", "orient"],
        help="Method for peer grouping",
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs (-1 = all cores)")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")
    parser.add_argument("--tariff", type=float, default=None, help="Electricity tariff PKR/kWh (overrides workbook default)")
    parser.add_argument(
        "--wash-cost", type=float, default=None, dest="wash_cost",
        help="Wash cost PKR per string (required for cleaning-economics recommendation)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    if verbose:
        print(f"Loading data from:\n  CSV:   {args.csv}\n  Specs: {args.specs}")

    # Step 1: Load sources
    long_df, plant_meta, cfg = load_from_sources(args.csv, args.specs)

    # Step 2: Apply CLI overrides
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs
    if args.tariff is not None:
        cfg.site.tariff = args.tariff
        plant_meta["plant_resolved"]["tariff"] = args.tariff
    if args.wash_cost is not None:
        cfg.wash_cost_per_string_pkr = args.wash_cost
        plant_meta["plant_resolved"]["wash_cost_per_string_pkr"] = args.wash_cost
        # Patch C6 in the already-computed preflight dict rather than re-running
        if "preflight" in plant_meta:
            plant_meta["preflight"]["C6"] = True
            plant_meta["preflight"]["details"]["C6"] = (
                f"wash cost={args.wash_cost:.0f} PKR/string supplied via --wash-cost"
            )

    # Step 3: Run pipeline
    results = run_pipeline_from_frame(
        long_df=long_df,
        plant_meta=plant_meta,
        cfg=cfg,
        cluster_method=args.cluster_method,
        verbose=verbose,
    )

    # Step 4: Export results
    out_xlsx = out_dir / f"{cfg.site.name}_soiling_results.xlsx"
    if verbose:
        print(f"Exporting results to {out_xlsx}")
    export_results_to_excel(results, str(out_xlsx), verbose=verbose)

    # Step 5: Generate figures
    if not args.no_figures:
        if verbose:
            print(f"Generating figures in {out_dir}")
        make_all_figures(results, str(out_dir), verbose=verbose)

    if verbose:
        print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
