"""Export DataFrames to a single CSV (all levels, with a "level" column)."""

from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()

SHEET_ORDER = ["plant", "inverter", "mppt", "string"]


def export_all(
    dfs: dict[str, pd.DataFrame],
    plant_name: str,
    start_str: str,
    end_str: str,
    output_dir: Path,
) -> dict[str, Path]:
    slug = f"{plant_name.replace(' ', '_')}_{start_str}_to_{end_str}"
    csv_dir = output_dir / "csv"
    csv_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # ── CSV (all levels in one file, with a "level" column) ──────
    labeled = []
    for level in SHEET_ORDER:
        df = dfs[level]
        if df.empty:
            continue
        tmp = df.copy()
        tmp.insert(0, "level", level)
        labeled.append(tmp)
    if labeled:
        combined_csv = pd.concat(labeled, axis=0, ignore_index=True, sort=False)
        csv_path = csv_dir / f"{slug}.csv"
        combined_csv.to_csv(csv_path, index=False)
        paths["csv"] = csv_path
        console.print(f"  [cyan]CSV (all levels, raw)[/cyan] → {csv_path}")

    return paths
