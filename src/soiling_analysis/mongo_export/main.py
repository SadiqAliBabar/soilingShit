"""
Soiling Analysis — Data Fetcher
================================
Connects to MongoDB, lets you pick a plant and date range,
then saves the flattened data as a single CSV file.
"""

import sys
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.table import Table

from . import db
from . import flatten
from . import export

console = Console()
# Save CSV exports under the repository-level data/ directory.
OUTPUT_DIR = Path(__file__).resolve().parents[3] / "data"


# ── Helpers ───────────────────────────────────────────────────────────────────

def pick_plant(plants: list[dict]) -> dict:
    console.print()
    tbl = Table(title="Available Plants", show_lines=True, border_style="cyan")
    tbl.add_column("#",            style="bold cyan", justify="right")
    tbl.add_column("Plant Name",   style="bold white")
    tbl.add_column("Database",     style="dim")
    for i, p in enumerate(plants, 1):
        tbl.add_row(str(i), p["display_name"], p["db_name"])
    console.print(tbl)

    while True:
        choice = Prompt.ask("[bold cyan]Select plant number[/bold cyan]")
        if choice.isdigit() and 1 <= int(choice) <= len(plants):
            return plants[int(choice) - 1]
        console.print("[red]Invalid choice, try again.[/red]")


def parse_date(prompt_text: str) -> datetime:
    while True:
        raw = Prompt.ask(f"[bold cyan]{prompt_text}[/bold cyan] [dim](YYYY-MM-DD)[/dim]")
        try:
            return datetime.strptime(raw.strip(), "%Y-%m-%d")
        except ValueError:
            console.print("[red]Invalid date format. Use YYYY-MM-DD (e.g. 2025-01-01)[/red]")


def confirm_selection(plant: dict, start: datetime, end: datetime) -> bool:
    console.print()
    console.print(Panel(
        f"[bold]Plant:[/bold]  {plant['display_name']}\n"
        f"[bold]DB:[/bold]     {plant['db_name']}\n"
        f"[bold]Range:[/bold]  {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}",
        title="[yellow]Confirm Fetch[/yellow]",
        border_style="yellow",
    ))
    ans = Prompt.ask("Proceed? [Y/n]", default="Y")
    return ans.strip().lower() in ("y", "yes", "")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    console.print(Panel(
        "[bold cyan]Soiling Analysis — Data Fetcher[/bold cyan]\n"
        "[dim]MongoDB  →  CSV[/dim]",
        border_style="cyan",
    ))

    # 1. Connect
    console.print("\n[dim]Connecting to MongoDB...[/dim]")
    try:
        client = db.get_client()
        client.admin.command("ping")
        console.print("[green]Connected.[/green]")
    except Exception as e:
        console.print(f"[bold red]Connection failed:[/bold red] {e}")
        console.print("[yellow]Tip: set MONGO_URI in your .env file[/yellow]")
        sys.exit(1)

    # 2. Pick plant
    plants = db.list_plants(client)
    if not plants:
        console.print("[red]No 'shams_' databases found in MongoDB.[/red]")
        sys.exit(1)
    plant = pick_plant(plants)

    # 3. Date range
    console.print()
    start_dt = parse_date("Start date")
    end_dt   = parse_date("End   date")
    if end_dt < start_dt:
        console.print("[red]End date must be after start date.[/red]")
        sys.exit(1)

    if not confirm_selection(plant, start_dt, end_dt):
        console.print("[yellow]Aborted.[/yellow]")
        sys.exit(0)

    # 4. Fetch
    console.print("\n[dim]Fetching records...[/dim]")
    records = db.fetch_records(client, plant["db_name"], start_dt, end_dt)
    if not records:
        console.print("[red]No records found for the selected range.[/red]")
        sys.exit(1)
    console.print(f"[green]Fetched {len(records)} hourly documents.[/green]")

    # 5. Flatten
    console.print("[dim]Flattening data...[/dim]")
    dfs = flatten.flatten_all(records)
    console.print(
        f"  Plant rows: [cyan]{len(dfs['plant'])}[/cyan]  |  "
        f"Inverter rows: [cyan]{len(dfs['inverter'])}[/cyan]  |  "
        f"MPPT rows: [cyan]{len(dfs['mppt'])}[/cyan]  |  "
        f"String rows: [cyan]{len(dfs['string'])}[/cyan]"
    )

    # 6. Enrich pv_temperature from FM_OD_PRD (EMI device, matched by timestamp)
    console.print("\n[dim]Fetching pv_temperature from [bold]FM_OD_PRD[/bold]...[/dim]")
    try:
        temp_records = db.fetch_temperature_records(
            client, plant["db_name"], start_dt, end_dt
        )
        console.print(f"  EMI docs fetched: [cyan]{len(temp_records)}[/cyan]")
        lookup = flatten.build_temperature_lookup(temp_records)
        console.print(f"  Hourly temperature buckets: [cyan]{len(lookup)}[/cyan]")
        dfs["string"] = flatten.enrich_pv_temperature(dfs["string"], lookup)
        filled = dfs["string"]["pv_temperature"].notna().sum()
        total  = len(dfs["string"])
        console.print(
            f"  String rows enriched: [green]{filled}[/green] / {total}"
            + (f" [yellow](no match for {total - filled} rows)[/yellow]" if filled < total else "")
        )
    except Exception as e:
        console.print(f"  [yellow]Temperature enrichment skipped:[/yellow] {e}")

    # 7. Export (raw long CSV — all levels)
    console.print("\n[bold]Saving files...[/bold]")
    start_str = start_dt.strftime("%Y%m%d")
    end_str = end_dt.strftime("%Y%m%d")
    export.export_all(
        dfs,
        plant_name=plant["display_name"],
        start_str=start_str,
        end_str=end_str,
        output_dir=OUTPUT_DIR,
    )

    console.print(Panel(
        "[bold green]Done![/bold green] All files saved to [cyan]output/[/cyan]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
