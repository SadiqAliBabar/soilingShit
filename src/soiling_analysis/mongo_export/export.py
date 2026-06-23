"""Export DataFrames to four separate CSVs and one Excel workbook (four sheets)."""

from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()

LEVELS = ["plant", "inverter", "mppt", "string"]


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

    # ── Four separate CSVs ────────────────────────────────────────
    for level in LEVELS:
        df = dfs.get(level)
        if df is None or df.empty:
            continue
        path = csv_dir / f"{slug}_{level}.csv"
        df.to_csv(path, index=False)
        paths[f"csv_{level}"] = path
        console.print(f"  [cyan]CSV ({level})[/cyan] → {path}")

    # ── One Excel workbook, four sheets ───────────────────────────
    excel_path = output_dir / f"{slug}.xlsx"
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        for level in LEVELS:
            df = dfs.get(level)
            if df is None or df.empty:
                continue
            df.to_excel(writer, sheet_name=level.capitalize(), index=False)
    paths["excel"] = excel_path
    console.print(f"  [green]Excel (all sheets)[/green] → {excel_path}")

    return paths
