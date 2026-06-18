import argparse
import sys
from pathlib import Path

# Add the src directories to the python path
repo_root = Path(__file__).resolve().parent
sys.path.insert(0, str(repo_root / "src"))
sys.path.insert(0, str(repo_root / "src" / "soiling_analysis" / "soiling_old_7_prompt"))

from soiling_analysis.loader import load_from_sources
from pv_diag.pipeline import run_pipeline_from_frame
from pv_diag.excel_export import export_results_to_excel
from pv_diag.plotting import make_all_figures

def main():
    parser = argparse.ArgumentParser(description="Run pv_diag pipeline on CSV data")
    parser.add_argument("--csv", required=True, help="Path to measured CSV from MongoDB")
    parser.add_argument("--specs", default=str(repo_root / "src/soiling_analysis/RequireInputData/RequireInputData.xlsx"), help="Path to RequireInputData.xlsx")
    parser.add_argument("--out-dir", default="output", help="Directory to write output files")
    parser.add_argument("--cluster-method", default="combined", choices=["combined", "mppt", "orient"], help="Method for peer grouping")
    parser.add_argument("--n-jobs", type=int, default=-1, help="Number of parallel jobs")
    parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    parser.add_argument("--quiet", action="store_true", help="Reduce logging output")

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    if verbose:
        print(f"Loading data from:\n  CSV:   {args.csv}\n  Specs: {args.specs}")

    # Step 1: Load sources
    long_df, plant_meta, cfg = load_from_sources(args.csv, args.specs)

    # Step 2: Apply config
    if args.n_jobs is not None:
        cfg.n_jobs = args.n_jobs

    # Step 3: Run pipeline
    results = run_pipeline_from_frame(
        long_df=long_df,
        plant_meta=plant_meta,
        cfg=cfg,
        cluster_method=args.cluster_method,
        verbose=verbose
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

if __name__ == "__main__":
    main()
