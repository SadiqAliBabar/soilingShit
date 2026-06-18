"""CLI entry — run full diagnostics pipeline on a customer xlsx.

Usage:
    python run.py /path/to/data.xlsx [--out-dir /path/to/output]
                                     [--cluster-method combined|mppt|orient]
                                     [--n-jobs N]
                                     [--no-figures]
                                     [--quiet]
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

# Ensure pv_diag package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from pv_diag import (PipelineConfig, run_pipeline,
                     export_results_to_excel, make_all_figures)


def main():
    p = argparse.ArgumentParser(description="Run PV diagnostics pipeline.")
    p.add_argument("xlsx_path", help="Input customer .xlsx file")
    p.add_argument("--out-dir", default="/mnt/user-data/outputs",
                   help="Output directory")
    p.add_argument("--cluster-method", default="combined",
                   choices=["combined","mppt","orient"])
    p.add_argument("--n-jobs", type=int, default=1)
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    cfg = PipelineConfig()
    cfg.n_jobs = args.n_jobs

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    verbose = not args.quiet

    print(f"\n=== PV diagnostics — pv_diag v2.0 ===\n")
    results = run_pipeline(args.xlsx_path, cfg,
                           cluster_method=args.cluster_method, verbose=verbose)
    xlsx_out = out_dir / f"{Path(args.xlsx_path).stem}_diagnostics.xlsx"
    export_results_to_excel(results, xlsx_out,
                            source_file=args.xlsx_path, verbose=verbose)
    if not args.no_figures:
        fig_dir = out_dir / "figures"
        make_all_figures(results, fig_dir, verbose=verbose)
    print(f"\nDone. Output -> {out_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
