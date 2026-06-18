"""Interactive plots saved as HTML via Plotly — hover for full detail."""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import PipelineConfig

VERDICT_COLORS = {
    "Clean": "#2BAE66", "Clean (post-wash)": "#1F8A4D",
    "Partial Recovery": "#79C26B",
    "Lt.Soiling": "#F5C95E", "Mod.Soiling": "#E89441", "Hvy.Soiling": "#C0392B",
    "Shading": "#3A6FB5", "Degradation": "#7E4FB5", "Mixed": "#8C7B6F",
    "Skipped": "#B0B0B0", "Insufficient": "#B0B0B0",
}


def _vc(v: str) -> str:
    if v in VERDICT_COLORS:
        return VERDICT_COLORS[v]
    for k, col in VERDICT_COLORS.items():
        if v.startswith(k):
            return col
    return "#5D6D7E"


def plot_soiling_dashboard(label, result, cfg, out_dir):
    daily = result.get("daily_df")
    if daily is None or daily.empty:
        return None
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])

    nci_col = (
        "NCI_corrected_noon"
        if "NCI_corrected_noon" in daily.columns
        and daily["NCI_corrected_noon"].notna().sum() >= 3
        else "NCI_noon"
    )

    wash = result.get("wash", {})
    events = wash.get("events_df", pd.DataFrame())
    soil_f = result.get("soiling_full", {})
    soil_c = result.get("soiling_current", {})
    verdict = result.get("classification", {}).get("verdict", "Unknown")
    cluster = result.get("cluster", {}).get("full_cluster", "")

    # --- Extract adaptive baseline value for this string ---
    adaptive_ref = None
    ab = result.get("adaptive_baseline")
    if ab is not None:
        try:
            adaptive_ref = float(getattr(ab, "value", None) or ab.get("value") if isinstance(ab, dict) else getattr(ab, "value", None))
        except Exception:
            adaptive_ref = None

    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.46, 0.14, 0.40],
        vertical_spacing=0.04,
        specs=[[{"type": "xy"}], [{"type": "xy"}], [{"type": "table"}]],
    )

    # --- NCI trace ---
    pr_vals = daily["PR"] if "PR" in daily.columns else pd.Series([float("nan")] * len(daily))
    nci_hover = []
    for d, n, p in zip(daily["date"], daily[nci_col], pr_vals):
        tip = (
            f"<b>{d.strftime('%Y-%m-%d')}</b><br>"
            f"NCI: <b>{n:.4f}</b><br>"
        )
        if pd.notna(p):
            tip += f"PR: {p:.4f}<br>"
        nci_hover.append(tip)

    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily[nci_col],
        mode="lines+markers",
        name=nci_col,
        line=dict(color="#2C3E50", width=1.6),
        marker=dict(size=5, color="#2C3E50"),
        hovertext=nci_hover,
        hoverinfo="text",
    ), row=1, col=1)

    # --- PR trace ---
    if "PR" in daily.columns:
        pr_hover = [
            f"<b>{d.strftime('%Y-%m-%d')}</b><br>PR: <b>{p:.4f}</b>"
            for d, p in zip(daily["date"], daily["PR"])
        ]
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["PR"],
            mode="lines+markers",
            name="PR",
            line=dict(color="#7F8C8D", width=1, dash="dash"),
            marker=dict(size=3, symbol="square"),
            hovertext=pr_hover,
            hoverinfo="text",
            opacity=0.6,
        ), row=1, col=1)

    # --- Soiling trend lines ---
    for i, seg in enumerate(soil_f.get("segments", [])):
        if not np.isfinite(seg.get("slope_per_day", np.nan)):
            continue
        s0 = pd.to_datetime(seg["start"])
        s1 = pd.to_datetime(seg["end"])
        sub = daily[(daily["date"] >= s0) & (daily["date"] <= s1)]
        if sub.empty:
            continue
        x_days = (sub["date"] - sub["date"].min()).dt.days.values
        slope = seg["slope_per_day"]
        intercept = sub[nci_col].mean() - slope * x_days.mean()
        y_trend = slope * x_days + intercept
        seg_hover = [
            f"<b>Soiling Trend</b><br>"
            f"Date: {d.strftime('%Y-%m-%d')}<br>"
            f"Trend NCI: {y:.4f}<br>"
            f"Slope: {slope*100:.4f} %/day<br>"
            f"Period: {str(seg['start'])[:10]} → {str(seg['end'])[:10]}"
            for d, y in zip(sub["date"], y_trend)
        ]
        fig.add_trace(go.Scatter(
            x=sub["date"], y=y_trend,
            mode="lines",
            name=f"Trend seg {i+1}",
            line=dict(color="#C0392B", width=2.4),
            hovertext=seg_hover,
            hoverinfo="text",
        ), row=1, col=1)

    # --- Baseline reference lines ---
    base = float(daily["NCI_baseline"].iloc[0]) if "NCI_baseline" in daily.columns else 1.0
    x0, x1 = daily["date"].min(), daily["date"].max()
    ref_lines = [
        (1.0,  "#7F8C8D", "dot",  "NCI=1.0"),
        (base, "#8E44AD", "dash", f"NCI baseline={base:.3f}"),
    ]
    # Add adaptive baseline line if available
    if adaptive_ref is not None and 0.5 < adaptive_ref < 1.15:
        ref_lines.append(
            (adaptive_ref, "#16A085", "dashdot",
             f"Adaptive clean ref={adaptive_ref:.3f}")
        )
    for y_val, color, dash, name in ref_lines:
        fig.add_shape(type="line", x0=x0, x1=x1, y0=y_val, y1=y_val,
                      line=dict(color=color, width=1.5, dash=dash), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="lines",
            name=name, line=dict(color=color, width=1.5, dash=dash),
            showlegend=True,
        ), row=1, col=1)

    # --- Wash event markers (scatter points + vertical shapes) ---
    if events is not None and not events.empty:
        ev_x, ev_y, ev_hover, ev_colors = [], [], [], []
        for _, ev in events.iterrows():
            ed = pd.to_datetime(ev["event_date"])
            colour = (
                "#2BAE66" if ev["recovery_class"] == "Full recovery"
                else ("#79C26B" if ev["recovery_class"] == "Partial recovery"
                      else "#E89441")
            )
            closest = daily.iloc[(daily["date"] - ed).abs().argsort()[:1]]
            y_pos = float(closest[nci_col].iloc[0]) if not closest.empty else 1.0
            ev_x.append(ed)
            ev_y.append(y_pos + 0.03)
            ev_colors.append(colour)
            tip = (
                f"<b>Wash / Rain Event</b><br>"
                f"Date: <b>{ed.strftime('%Y-%m-%d')}</b><br>"
                f"Cause: {ev['cause']}<br>"
                f"Recovery: {ev['recovery_class']}<br>"
                f"ΔNCI: +{ev['delta_nci']*100:.2f} pp<br>"
                f"Rain mm: {ev.get('rain_mm', 'n/a')}"
            )
            ev_hover.append(tip)
            fig.add_shape(
                type="line", x0=ed, x1=ed, y0=0.55, y1=1.10,
                line=dict(color=colour, width=2, dash="dot"),
                row=1, col=1,
            )
        fig.add_trace(go.Scatter(
            x=ev_x, y=ev_y,
            mode="markers",
            name="Wash events",
            marker=dict(symbol="triangle-down", size=14, color=ev_colors,
                        line=dict(color="white", width=1)),
            hovertext=ev_hover,
            hoverinfo="text",
        ), row=1, col=1)

    # --- Bottom panel: rain or valid sample count ---
    if "rain_mm" in daily.columns and daily["rain_mm"].sum() > 0:
        rain_hover = [
            f"<b>{d.strftime('%Y-%m-%d')}</b><br>Rain: <b>{r:.1f} mm</b>"
            for d, r in zip(daily["date"], daily["rain_mm"])
        ]
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["rain_mm"],
            name="Rain (mm)",
            marker_color="#3498DB", opacity=0.65,
            hovertext=rain_hover, hoverinfo="text",
        ), row=2, col=1)
        fig.update_yaxes(title_text="Rain (mm)", row=2, col=1)
    else:
        valid_hover = [
            f"<b>{d.strftime('%Y-%m-%d')}</b><br>Valid samples: <b>{n}</b>"
            for d, n in zip(daily["date"], daily["n_valid"])
        ]
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["n_valid"],
            name="Valid samples",
            marker_color="#95A5A6", opacity=0.5,
            hovertext=valid_hover, hoverinfo="text",
        ), row=2, col=1)
        fig.update_yaxes(title_text="# valid samples", row=2, col=1)

    # --- Daily data table ---
    tbl_daily = daily.sort_values("date").copy()
    tbl_cols_def = [
        ("Date",        "date",               lambda v: v.strftime("%Y-%m-%d") if pd.notna(v) else ""),
        ("NCI",         nci_col,              lambda v: f"{v:.4f}" if pd.notna(v) else "—"),
        ("NCI adaptive",  "NCI_adaptive_noon",lambda v: f"{v:.4f}" if pd.notna(v) else "—"),
        ("PR",          "PR",                 lambda v: f"{v:.4f}" if pd.notna(v) else "—"),
        ("E meas (kWh)","E_meas_kWh",         lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
        ("E exp (kWh)", "E_exp_kWh",          lambda v: f"{v:.2f}" if pd.notna(v) else "—"),
        ("Rain (mm)",   "rain_mm",            lambda v: f"{v:.1f}" if pd.notna(v) else "—"),
        ("Samples",     "n_valid",            lambda v: str(int(v)) if pd.notna(v) else "—"),
        ("Asymmetry",   "asym",               lambda v: f"{v:.4f}" if pd.notna(v) else "—"),
    ]
    tbl_headers, tbl_values, tbl_col_colors = [], [], []
    nci_vals_raw = tbl_daily[nci_col].values if nci_col in tbl_daily.columns else []

    def _nci_cell_color(nci_v):
        if not pd.notna(nci_v):
            return "#FFFFFF"
        if nci_v >= 0.95:
            return "#D5F5E3"
        if nci_v >= 0.90:
            return "#FCF3CF"
        if nci_v >= 0.80:
            return "#FAD7A0"
        return "#FADBD8"

    for col_label, col_key, fmt in tbl_cols_def:
        if col_key not in tbl_daily.columns:
            continue
        tbl_headers.append(col_label)
        tbl_values.append([fmt(v) for v in tbl_daily[col_key]])
        if col_key == nci_col or col_key == "NCI_adaptive_noon":
            tbl_col_colors.append([_nci_cell_color(v) for v in tbl_daily[col_key]])
        else:
            tbl_col_colors.append(["#FFFFFF"] * len(tbl_daily))

    n_rows_tbl = len(tbl_daily)
    row_h = max(20, min(28, int(900 / max(n_rows_tbl, 1))))
    fig.add_trace(go.Table(
        header=dict(
            values=[f"<b>{h}</b>" for h in tbl_headers],
            fill_color="#2C3E50",
            font=dict(color="white", size=11),
            align="center",
            height=28,
        ),
        cells=dict(
            values=tbl_values,
            fill_color=tbl_col_colors,
            font=dict(color="#2C3E50", size=10),
            align="center",
            height=row_h,
        ),
    ), row=3, col=1)

    srr = (
        f"SRR full = {soil_f.get('srr_pct_per_day', float('nan')):.3f} %/d  |  "
        f"Current-seg SRR = {soil_c.get('srr_pct_per_day', float('nan')):.3f} %/d  |  "
        f"Loss(window) = {soil_f.get('weighted_soiling_loss_pct', float('nan')):.1f}%"
    )
    tbl_height = max(300, min(600, n_rows_tbl * row_h + 60))
    fig.update_layout(
        title=dict(
            text=(
                f"<b>{label}</b>   |   Cluster: {cluster}   |   "
                f"<span style='color:{_vc(verdict)}'><b>{verdict}</b></span><br>"
                f"<sup style='color:#555'>{srr}</sup>"
            ),
            font_size=13,
        ),
        hovermode="x unified",
        height=700 + tbl_height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font_size=10),
        margin=dict(l=70, r=20, t=110, b=40),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
    )
    fig.update_yaxes(title_text="NCI", range=[0.50, 1.12], row=1, col=1,
                     gridcolor="#E8E8E8")
    fig.update_xaxes(title_text="Date", row=2, col=1, gridcolor="#E8E8E8")

    fp = Path(out_dir) / f"soiling_dashboard__{label}.html"
    fig.write_html(str(fp), include_plotlyjs="cdn")
    return str(fp)


def plot_data_quality(label, df, result, cfg, out_dir):
    if df is None or df.empty:
        return None
    from .constants import QUALITY_FLAGS
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    ts = d["ts"]
    d["date"] = (
        ts.dt.tz_convert(None).dt.date if getattr(ts.dt, "tz", None)
        else ts.dt.date
    )
    qf = d["qflag"].astype(int)
    cls = pd.Series("OK", index=d.index)
    cls[(qf & QUALITY_FLAGS["NIGHT"]) > 0] = "NIGHT"
    cls[(qf & QUALITY_FLAGS["STANDBY"]) > 0] = "STANDBY"
    cls[(qf & QUALITY_FLAGS["INVERTER_FAULT"]) > 0] = "FAULT"
    cls[(qf & (QUALITY_FLAGS["CURT_STATE"] | QUALITY_FLAGS["CURT_STATISTICAL"]
               | QUALITY_FLAGS["CURT_SUPPRESSED"])) > 0] = "CURTAILED"
    cls[(qf & QUALITY_FLAGS["IV_SCAN"]) > 0] = "IV_SCAN"
    cls[(qf & QUALITY_FLAGS["TRANSIENT"]) > 0] = "TRANSIENT"

    tbl = (d.assign(cls=cls).groupby(["date", "cls"]).size()
           .unstack("cls", fill_value=0))
    order = ["OK", "CURTAILED", "FAULT", "STANDBY", "NIGHT", "TRANSIENT", "IV_SCAN"]
    for c in order:
        if c not in tbl.columns:
            tbl[c] = 0
    tbl = tbl[order]
    totals = tbl.sum(axis=1).replace(0, np.nan)
    pct = tbl.div(totals, axis=0) * 100.0

    cmap = {
        "OK": "#2BAE66", "CURTAILED": "#C0392B", "FAULT": "#7F1D1D",
        "STANDBY": "#95A5A6", "NIGHT": "#34495E",
        "TRANSIENT": "#F5C95E", "IV_SCAN": "#7E4FB5",
    }
    cat_desc = {
        "OK": "Normal operating data",
        "CURTAILED": "Inverter output clamped / curtailed",
        "FAULT": "Inverter fault or alarm active",
        "STANDBY": "Inverter in standby / low-irradiance",
        "NIGHT": "Night-time interval (no generation)",
        "TRANSIENT": "Short-duration anomaly / transient dip",
        "IV_SCAN": "IV sweep scan in progress",
    }

    dates_str = [str(x) for x in pct.index]
    fig = go.Figure()

    for cat in order:
        hover = [
            f"<b>{dt}</b><br>"
            f"Category: <b>{cat}</b><br>"
            f"{cat_desc[cat]}<br>"
            f"Share: <b>{p:.1f}%</b><br>"
            f"Intervals: {int(c)}"
            for dt, p, c in zip(dates_str, pct[cat], tbl[cat])
        ]
        fig.add_trace(go.Bar(
            x=dates_str,
            y=pct[cat],
            name=cat,
            marker_color=cmap[cat],
            hovertext=hover,
            hoverinfo="text",
        ))

    fig.update_layout(
        barmode="stack",
        title=f"<b>Daily Data Quality — {label}</b>",
        xaxis_title="Date",
        yaxis=dict(title="% of intervals", range=[0, 100]),
        hovermode="x unified",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font_size=11),
        margin=dict(l=70, r=20, t=90, b=120),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-45, tickfont_size=9, gridcolor="#E8E8E8")
    fig.update_yaxes(gridcolor="#E8E8E8")

    fp = Path(out_dir) / f"data_quality__{label}.html"
    fig.write_html(str(fp), include_plotlyjs="cdn")
    return str(fp)


def plot_plant_overview(results, cfg, out_dir):
    ps = results.get("per_string", {})
    if not ps:
        return None
    rows = []
    for lbl, r in ps.items():
        v = r.get("classification", {}).get("verdict", "Unknown")
        lo = r.get("losses", {})
        clx = r.get("classification", {})
        rows.append(dict(
            label=lbl,
            verdict=v,
            soiling_kwh=float(lo.get("soiling_kwh", 0)),
            curt_kwh=float(lo.get("curtailment_kwh", 0)),
            total_pkr=float(lo.get("total_avoidable_pkr", 0)),
            soiling_pkr=float(lo.get("soiling_pkr", 0)),
            curt_pkr=float(lo.get("curtailment_pkr", 0)),
            annualised_kwh=float(lo.get("annualised_kwh", 0)),
            confidence=clx.get("confidence", ""),
            primary_driver=clx.get("primary_driver", ""),
        ))
    summary = pd.DataFrame(rows)
    if summary.empty:
        return None

    pl = results.get("plant_losses", {})
    cur = cfg.site.currency

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "pie"}, {"type": "table"}],
            [{"type": "bar", "colspan": 2}, None],
        ],
        subplot_titles=[
            "Verdict Distribution",
            f"Plant KPIs — {pl.get('period_days', 0)} day period",
            "Avoidable Losses per String",
        ],
        row_heights=[0.38, 0.62],
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
    )

    # --- Verdict pie ---
    vcounts = summary["verdict"].value_counts()
    total_strings = vcounts.sum()
    pie_hover = [
        f"<b>{v}</b><br>Count: {c}<br>Share: {c/total_strings*100:.1f}%"
        for v, c in zip(vcounts.index, vcounts.values)
    ]
    fig.add_trace(go.Pie(
        labels=vcounts.index,
        values=vcounts.values,
        marker_colors=[_vc(v) for v in vcounts.index],
        hovertext=pie_hover,
        hoverinfo="text",
        textinfo="label+percent",
        hole=0.35,
        textfont_size=10,
    ), row=1, col=1)

    # --- KPI table ---
    kpi_labels = ["Soiling loss", "Curtailment loss", "Total avoidable", "Annualised"]
    kpi_kwh = [
        f"{pl.get('soiling_kwh', 0):,.0f}",
        f"{pl.get('curtailment_kwh', 0):,.0f}",
        f"{pl.get('total_avoidable_kwh', 0):,.0f}",
        f"{pl.get('annualised_kwh', 0):,.0f}",
    ]
    kpi_rev = [
        f"{cur} {pl.get('soiling_pkr', 0):,.0f}",
        f"{cur} {pl.get('curtailment_pkr', 0):,.0f}",
        f"{cur} {pl.get('total_avoidable_pkr', 0):,.0f}",
        f"{cur} {pl.get('annualised_pkr', 0):,.0f}",
    ]
    fig.add_trace(go.Table(
        header=dict(
            values=["<b>Metric</b>", "<b>Energy (kWh)</b>", f"<b>Revenue ({cur})</b>"],
            fill_color="#2C3E50",
            font=dict(color="white", size=12),
            align="left",
            height=28,
        ),
        cells=dict(
            values=[kpi_labels, kpi_kwh, kpi_rev],
            fill_color=[["#F2F2F2", "white", "#F2F2F2", "white"]],
            align="left",
            font_size=12,
            height=26,
        ),
    ), row=1, col=2)

    # --- Per-string bar chart ---
    s = summary.sort_values("total_pkr", ascending=False)

    soil_hover = [
        f"<b>{lbl}</b><br>"
        f"Verdict: {vrd}  ({conf})<br>"
        f"Primary driver: {drv}<br>"
        f"Soiling loss: <b>{kwh:,.0f} kWh</b> ({cur} {pkr:,.0f})<br>"
        f"Annualised: {ann:,.0f} kWh/yr"
        for lbl, vrd, conf, drv, kwh, pkr, ann in zip(
            s["label"], s["verdict"], s["confidence"], s["primary_driver"],
            s["soiling_kwh"], s["soiling_pkr"], s["annualised_kwh"])
    ]
    curt_hover = [
        f"<b>{lbl}</b><br>"
        f"Curtailment loss: <b>{kwh:,.0f} kWh</b> ({cur} {pkr:,.0f})"
        for lbl, kwh, pkr in zip(s["label"], s["curt_kwh"], s["curt_pkr"])
    ]

    fig.add_trace(go.Bar(
        x=s["label"], y=s["soiling_kwh"],
        name="Soiling kWh",
        marker_color="#E89441",
        hovertext=soil_hover,
        hoverinfo="text",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=s["label"], y=s["curt_kwh"],
        name="Curtailment kWh",
        marker_color="#C0392B",
        hovertext=curt_hover,
        hoverinfo="text",
    ), row=2, col=1)

    fig.update_layout(
        barmode="stack",
        title=dict(
            text=(
                f"<b>{cfg.site.name}</b> — Plant Diagnostics Overview<br>"
                f"<sup>Tariff: {cfg.site.tariff:.1f} {cur}/kWh  |  "
                f"Period: {pl.get('period_days', 0)} days  |  "
                f"Strings: {len(s)}</sup>"
            ),
            font_size=14,
        ),
        hovermode="closest",
        height=780,
        legend=dict(orientation="h", yanchor="bottom", y=-0.18,
                    xanchor="center", x=0.5, font_size=11),
        margin=dict(l=70, r=20, t=110, b=140),
        plot_bgcolor="#FAFAFA",
        paper_bgcolor="white",
    )
    fig.update_xaxes(tickangle=-50, tickfont_size=8, row=2, col=1,
                     gridcolor="#E8E8E8")
    fig.update_yaxes(title_text="Lost energy (kWh)", row=2, col=1,
                     gridcolor="#E8E8E8")

    fp = Path(out_dir) / "plant_overview.html"
    fig.write_html(str(fp), include_plotlyjs="cdn")
    return str(fp)


def make_all_figures(results, out_dir, verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = results["cfg"]
    long_df = results["long_df"]
    saved = dict(soiling=[], iv=[], quality=[], plant=[])

    for label, r in results["per_string"].items():
        try:
            p1 = plot_soiling_dashboard(label, r, cfg, out_dir)
            if p1:
                saved["soiling"].append(p1)
        except Exception as e:
            warnings.warn(f"plot failure for {label}: {e}")

    try:
        p3 = plot_data_quality(cfg.site.name, long_df, None, cfg, out_dir)
        if p3:
            saved["quality"].append(p3)
    except Exception as e:
        warnings.warn(f"plant data quality plot failure: {e}")

    p4 = plot_plant_overview(results, cfg, out_dir)
    if p4:
        saved["plant"].append(p4)

    if verbose:
        n = sum(len(v) for v in saved.values())
        print(f"  Saved {n} interactive HTML figures -> {out_dir}")
    return saved
