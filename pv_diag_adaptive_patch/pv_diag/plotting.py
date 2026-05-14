"""Plots: soiling dashboard, IV diagnostics, data-quality, plant overview using Plotly."""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from .config import PipelineConfig

VERDICT_COLORS = {
    "Clean": "#A8E6CF", "Clean (post-wash)": "#81C784",
    "Partial Recovery": "#DCE775",
    "Lt.Soiling": "#FFF176", "Mod.Soiling": "#FFB74D", "Hvy.Soiling": "#E57373",
    "Shading": "#64B5F6", "Degradation": "#BA68C8", "Mixed": "#A1887F",
    "Skipped": "#BDBDBD", "Insufficient": "#BDBDBD",
}

QUALITY_COLORS = {
    "OK": "#A8E6CF", "CURTAILED": "#FF8A65", "FAULT": "#E57373",
    "STANDBY": "#90A4AE", "NIGHT": "#455A64",
    "TRANSIENT": "#FFF176", "IV_SCAN": "#CE93D8",
    "FAULT/ZERO": "#E57373", "MISSING": "#BDBDBD"
}

def _vc(v: str) -> str:
    if v in VERDICT_COLORS: return VERDICT_COLORS[v]
    for k, col in VERDICT_COLORS.items():
        if v.startswith(k): return col
    return "#90A4AE"


def plot_soiling_dashboard(label, result, cfg, out_dir):
    daily = result.get("daily_df")
    if daily is None or daily.empty: return None
    daily = daily.copy(); daily["date"] = pd.to_datetime(daily["date"])
    
    if ("NCI_corrected_noon" in daily.columns
            and daily["NCI_corrected_noon"].notna().sum() >= 3):
        nci_col = "NCI_corrected_noon"
    else:
        nci_col = "NCI_noon"
        
    wash = result.get("wash", {})
    events = wash.get("events_df", pd.DataFrame())
    soil_f = result.get("soiling_full", {})
    soil_c = result.get("soiling_current", {})
    verdict = result.get("classification", {}).get("verdict", "Unknown")
    cluster = result.get("cluster", {}).get("full_cluster", "")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.07,
                        row_heights=[0.7, 0.3])

    # NCI plot
    fig.add_trace(go.Scatter(
        x=daily["date"], y=daily[nci_col],
        mode="lines+markers",
        name="NCI",
        line=dict(color="#A8D1E7", width=3),
        marker=dict(size=8, symbol="circle", line=dict(width=1, color="white")),
        customdata=np.stack([daily["PR"], daily["n_valid"]], axis=-1) if "PR" in daily.columns else daily["n_valid"].values.reshape(-1, 1),
        hovertemplate="<b>Date: %{x|%Y-%m-%d}</b><br>NCI: %{y:.3f}<br>" + 
                      ("PR: %{customdata[0]:.3f}<br>" if "PR" in daily.columns else "") +
                      "Samples: %{customdata[-1]}<extra></extra>"
    ), row=1, col=1)

    if "PR" in daily.columns:
        fig.add_trace(go.Scatter(
            x=daily["date"], y=daily["PR"],
            mode="lines",
            name="PR (Reference)",
            line=dict(color="#BDBDBD", width=1, dash="dot"),
            hovertemplate="PR: %{y:.3f}<extra></extra>"
        ), row=1, col=1)

    # Wash events
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            ed = pd.to_datetime(ev["event_date"])
            colour = ("#A8E6CF" if ev["recovery_class"]=="Full recovery"
                      else ("#DCE775" if ev["recovery_class"]=="Partial recovery"
                            else "#FFB74D"))
            
            # Vertical line for the event
            fig.add_vline(x=ed, line_width=2.5, line_dash="dash", line_color=colour, row=1, col=1)
            
            # Marker at the top
            fig.add_trace(go.Scatter(
                x=[ed], y=[1.05],
                mode="markers",
                marker=dict(symbol="triangle-down", size=12, color=colour, line=dict(width=1, color="white")),
                name=f"Event: {ev['cause']}",
                hovertemplate=f"<b>Wash Event</b><br>Date: {ed.date()}<br>Cause: {ev['cause']}<br>Recovery: {ev['recovery_class']}<br>Delta NCI: +{ev['delta_nci']*100:.1f}pp<extra></extra>",
                showlegend=True
            ), row=1, col=1)

            # Permanent label (Visible without hover)
            fig.add_annotation(
                x=ed, y=1.03,
                text=f"<b>{ev['cause']}</b><br>{ev['recovery_class']}<br><b>+{ev['delta_nci']*100:.1f}pp</b>",
                showarrow=False,
                font=dict(size=10, color=colour),
                bgcolor="rgba(0,0,0,0.7)",
                bordercolor=colour,
                borderwidth=1,
                borderpad=4,
                yshift=20,
                row=1, col=1
            )

    # Soiling segments
    for i, seg in enumerate(soil_f.get("segments", [])):
        if not np.isfinite(seg.get("slope_per_day", np.nan)): continue
        s0 = pd.to_datetime(seg["start"]); s1 = pd.to_datetime(seg["end"])
        sub = daily[(daily["date"] >= s0) & (daily["date"] <= s1)]
        if sub.empty: continue
        x_days = (sub["date"] - sub["date"].min()).dt.days.values
        slope = seg["slope_per_day"]
        intercept = sub[nci_col].mean() - slope * x_days.mean()
        y = slope * x_days + intercept
        
        fig.add_trace(go.Scatter(
            x=sub["date"], y=y,
            mode="lines",
            name=f"Slope {i+1}",
            line=dict(color="#FF8A65", width=4),
            hovertemplate=f"<b>Soiling Segment {i+1}</b><br>Slope: {slope*100:.3f}%/day<br>Start: {s0.date()}<br>End: {s1.date()}<extra></extra>",
            showlegend=False
        ), row=1, col=1)

    # Baseline
    base = float(daily["NCI_baseline"].iloc[0]) if "NCI_baseline" in daily.columns else 1.0
    fig.add_hline(y=1.0, line_width=1, line_dash="dot", line_color="#BDBDBD", row=1, col=1)
    fig.add_hline(y=base, line_width=2, line_dash="dash", line_color="#BA68C8", 
                  annotation_text=f"Baseline: {base:.3f}", annotation_position="bottom right",
                  row=1, col=1)

    # Rain / Valid samples
    if "rain_mm" in daily.columns and daily["rain_mm"].sum() > 0:
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["rain_mm"],
            name="Rain (mm)",
            marker_color="#4FC3F7",
            hovertemplate="<b>Date: %{x}</b><br>Rain: %{y:.1f} mm<extra></extra>"
        ), row=2, col=1)
        fig.update_yaxes(title_text="Rain (mm)", row=2, col=1)
    else:
        fig.add_trace(go.Bar(
            x=daily["date"], y=daily["n_valid"],
            name="Valid Samples",
            marker_color="#90A4AE",
            hovertemplate="<b>Date: %{x}</b><br>Valid Samples: %{y}<extra></extra>"
        ), row=2, col=1)
        fig.update_yaxes(title_text="# valid samples", row=2, col=1)

    srr_text = (f"SRR full={soil_f.get('srr_pct_per_day', float('nan')):.3f} %/d | "
                f"current-seg SRR={soil_c.get('srr_pct_per_day', float('nan')):.3f} %/d | "
                f"loss(window)={soil_f.get('weighted_soiling_loss_pct', float('nan')):.1f}%")

    fig.update_layout(
        template="plotly_dark",
        title=dict(
            text=f"Soiling Dashboard: {label}<br><sup>Cluster: {cluster} | Verdict: <span style='color:{_vc(verdict)}'>{verdict}</span></sup>",
            font=dict(size=22, family="Arial")
        ),
        xaxis2_title="Date",
        yaxis_title="Normalized Cleaning Index (NCI)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=120, b=80, l=60, r=40),
        hovermode="x unified",
        height=900,
        annotations=[
            dict(
                text=srr_text,
                showarrow=False,
                xref="paper", yref="paper",
                x=0.5, y=-0.12,
                font=dict(size=14, color="#BDBDBD")
            )
        ]
    )
    
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#424242", tickformat="%b %d")
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#424242")

    fp = Path(out_dir) / f"soiling_dashboard__{label}.html"
    fig.write_html(fp, include_plotlyjs="cdn", full_html=True)
    return str(fp)


def plot_iv_diagnostics(label, df, result, cfg, out_dir):
    if df is None or df.empty: return None
    sub = df[(df["POA"] > 100) & df["I"].notna() & df["V"].notna()]
    if len(sub) < 50: return None
    
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.1,
                        subplot_titles=(f"Measured I-V cloud — {label}", "Reconstructed I-V at STC"))

    # Scatter plot
    fig.add_trace(go.Scatter(
        x=sub["V"], y=sub["I"],
        mode="markers",
        marker=dict(
            size=4,
            color=sub["POA"],
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title="POA W/m²", x=0.45)
        ),
        name="Measured",
        hovertemplate="V: %{x:.1f}V<br>I: %{y:.2f}A<br>POA: %{marker.color:.0f}W/m²<extra></extra>"
    ), row=1, col=1)

    sdm = result.get("sdm", {}); sdm_m = result.get("sdm_metrics", {}) or {}
    if sdm and sdm.get("success"):
        try:
            from .sdm import iv_curve_from_sdm
            iv = iv_curve_from_sdm(sdm, cfg.module)
            fig.add_trace(go.Scatter(
                x=iv["V"], y=iv["I"],
                mode="lines",
                line=dict(color="#E57373", width=3),
                name="SDM@STC",
                hovertemplate="V: %{x:.1f}V<br>I: %{y:.2f}A<extra></extra>"
            ), row=1, col=2)
            
            t = (f"Isc x{sdm_m.get('isc_stc_ratio',float('nan')):.3f}<br>"
                 f"Voc x{sdm_m.get('voc_stc_ratio',float('nan')):.3f}<br>"
                 f"FF  x{sdm_m.get('ff_stc_ratio', float('nan')):.3f}")
            
            fig.add_annotation(
                text=t,
                xref="x2", yref="y2",
                x=0.05, y=0.05,
                showarrow=False,
                align="left",
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="#BDBDBD",
                borderwidth=1,
                row=1, col=2
            )
        except Exception as e:
            fig.add_annotation(text=f"SDM render failed: {e}", x=0.5, y=0.5, showarrow=False, row=1, col=2)
    else:
        fig.add_annotation(text=f"SDM unavailable<br>({sdm.get('reason','no fit')})", 
                           x=0.5, y=0.5, showarrow=False, row=1, col=2)

    fig.update_xaxes(title_text="V_dc (V)", row=1, col=1)
    fig.update_yaxes(title_text="I_dc (A)", row=1, col=1)
    fig.update_xaxes(title_text="V_dc (V)", row=1, col=2)
    fig.update_yaxes(title_text="I_dc (A)", row=1, col=2)

    fig.update_layout(
        template="plotly_dark",
        height=600,
        margin=dict(t=80, b=50, l=50, r=50),
        showlegend=False
    )

    fp = Path(out_dir) / f"iv_diagnostics__{label}.html"
    fig.write_html(fp, include_plotlyjs="cdn")
    return str(fp)


def plot_data_quality(label, df, result, cfg, out_dir):
    if df is None or df.empty: return None
    from .constants import QUALITY_FLAGS
    d = df.copy()
    d["ts"] = pd.to_datetime(d["ts"])
    ts = d["ts"]
    d["date"] = (ts.dt.tz_convert(None).dt.date if getattr(ts.dt, "tz", None)
                 else ts.dt.date)
    qf = d["qflag"].astype(int)
    cls = pd.Series("OK", index=d.index)
    cls[(qf & QUALITY_FLAGS["NIGHT"])              > 0] = "NIGHT"
    cls[(qf & QUALITY_FLAGS["STANDBY"])            > 0] = "STANDBY"
    cls[(qf & QUALITY_FLAGS["INVERTER_FAULT"])     > 0] = "FAULT"
    cls[(qf & (QUALITY_FLAGS["CURT_STATE"] | QUALITY_FLAGS["CURT_STATISTICAL"])) > 0] = "CURTAILED"
    cls[(qf & QUALITY_FLAGS["IV_SCAN"])            > 0] = "IV_SCAN"
    cls[(qf & QUALITY_FLAGS["TRANSIENT"])          > 0] = "TRANSIENT"
    
    tbl = (d.assign(cls=cls).groupby(["date","cls"]).size()
            .unstack("cls", fill_value=0))
    order = ["OK","CURTAILED","FAULT","STANDBY","NIGHT","TRANSIENT","IV_SCAN"]
    for c in order:
        if c not in tbl.columns: tbl[c] = 0
    tbl = tbl[order]
    
    pct = tbl.div(tbl.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    pct = pct.reset_index().melt(id_vars="date", var_name="State", value_name="Percentage")

    fig = px.bar(pct, x="date", y="Percentage", color="State",
                 title=f"Daily data quality — {label}",
                 color_discrete_map=QUALITY_COLORS,
                 template="plotly_dark",
                 category_orders={"State": order})

    fig.update_layout(
        yaxis_title="% of intervals",
        xaxis_title="Date",
        legend_title="Quality State",
        yaxis_range=[0, 100],
        height=500
    )
    
    fig.update_traces(hovertemplate="<b>Date: %{x}</b><br>State: %{fullData.name}<br>Share: %{y:.1f}%<extra></extra>")

    fp = Path(out_dir) / f"data_quality__{label}.html"
    fig.write_html(fp, include_plotlyjs="cdn")
    return str(fp)


def plot_aggregated_data_quality(label, df, target_col, out_dir):
    if df is None or df.empty or target_col not in df.columns:
        return None
        
    d = df.drop_duplicates(subset=["ts"]).copy()
    d["date"] = (d["ts"].dt.tz_convert(None).dt.date if getattr(d["ts"].dt, "tz", None)
                 else d["ts"].dt.date)
                 
    irr_col = "irradiance KW/m2" if "irradiance KW/m2" in d.columns else "POA"
    cls = pd.Series("OK", index=d.index)
    cls[d[target_col].isna()] = "MISSING"
    if irr_col in d.columns:
        cls[d[irr_col] < 0.05] = "NIGHT"
    cls[(d[target_col] <= 0) & (cls != "NIGHT") & (cls != "MISSING")] = "FAULT/ZERO"
    
    if "inverter_state" in d.columns and "Inverter" in label:
        curt_states = {513, 514, 1284, 1285}
        cls[(d["inverter_state"].isin(curt_states)) & (cls != "NIGHT")] = "CURTAILED"

    tbl = (d.assign(cls=cls).groupby(["date","cls"]).size()
            .unstack("cls", fill_value=0))
            
    order = ["OK", "CURTAILED", "FAULT/ZERO", "MISSING", "NIGHT"]
    for c in order:
        if c not in tbl.columns: tbl[c] = 0
    tbl = tbl[order]
    
    pct = tbl.div(tbl.sum(axis=1).replace(0, np.nan), axis=0) * 100.0
    pct = pct.reset_index().melt(id_vars="date", var_name="State", value_name="Percentage")

    fig = px.bar(pct, x="date", y="Percentage", color="State",
                 title=f"Daily data quality ({target_col}) — {label}",
                 color_discrete_map=QUALITY_COLORS,
                 template="plotly_dark",
                 category_orders={"State": order})

    fig.update_layout(
        yaxis_title="% of intervals",
        xaxis_title="Date",
        legend_title="Quality State",
        yaxis_range=[0, 100],
        height=500
    )
    
    fig.update_traces(hovertemplate="<b>Date: %{x}</b><br>State: %{fullData.name}<br>Share: %{y:.1f}%<extra></extra>")
    
    fp = Path(out_dir) / f"data_quality__{label}.html"
    fig.write_html(fp, include_plotlyjs="cdn")
    return str(fp)


def plot_plant_overview(results, cfg, out_dir):
    ps = results.get("per_string", {})
    if not ps: return None
    rows = []
    for label, r in ps.items():
        v = r.get("classification", {}).get("verdict", "Unknown")
        l = r.get("losses", {})
        rows.append(dict(label=label, verdict=v,
            soiling_kwh=float(l.get("soiling_kwh", 0)),
            curt_kwh=float(l.get("curtailment_kwh", 0)),
            total_pkr=float(l.get("total_avoidable_pkr", 0))))
    summary = pd.DataFrame(rows)
    if summary.empty: return None

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type": "domain"}, {"type": "table"}],
               [{"colspan": 2}, None]],
        subplot_titles=("Verdict Distribution", "Plant Loss Summary", "Avoidable Losses per String"),
        vertical_spacing=0.15,
        row_heights=[0.35, 0.65]
    )

    # Pie chart
    vcounts = summary["verdict"].value_counts()
    fig.add_trace(go.Pie(
        labels=vcounts.index, values=vcounts.values,
        marker=dict(colors=[_vc(v) for v in vcounts.index], line=dict(color="white", width=1)),
        textinfo="percent+label",
        hole=0.4,
        name="Verdicts"
    ), row=1, col=1)

    # KPI Table
    pl = results.get("plant_losses", {})
    cur = cfg.site.currency
    kpi_data = [
        ["Soiling loss", f"{pl.get('soiling_kwh',0):,.0f} kWh", f"{cur} {pl.get('soiling_pkr',0):,.0f}"],
        ["Curtailment loss", f"{pl.get('curtailment_kwh',0):,.0f} kWh", f"{cur} {pl.get('curtailment_pkr',0):,.0f}"],
        ["Total avoidable", f"{pl.get('total_avoidable_kwh',0):,.0f} kWh", f"{cur} {pl.get('total_avoidable_pkr',0):,.0f}"],
        ["Annualised", f"{pl.get('annualised_kwh',0):,.0f} kWh", f"{cur} {pl.get('annualised_pkr',0):,.0f}"]
    ]
    
    fig.add_trace(go.Table(
        header=dict(values=["Loss Type", "Energy (kWh)", f"Value ({cur})"],
                    fill_color="#263238", align="left", font=dict(color="white", size=14, family="Arial Bold")),
        cells=dict(values=list(zip(*kpi_data)),
                   fill_color="#37474F", align="left", font=dict(color="white", size=13),
                   height=30)
    ), row=1, col=2)

    # Bar chart
    s = summary.sort_values("total_pkr", ascending=False)
    fig.add_trace(go.Bar(
        x=s["label"], y=s["soiling_kwh"],
        name="Soiling kWh", marker_color="#FFB74D",
        hovertemplate="<b>%{x}</b><br>Soiling: %{y:.1f} kWh<extra></extra>"
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        x=s["label"], y=s["curt_kwh"],
        name="Curtailment kWh", marker_color="#E57373",
        hovertemplate="<b>%{x}</b><br>Curtailment: %{y:.1f} kWh<extra></extra>"
    ), row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        title=dict(text=f"{cfg.site.name} — Diagnostics Overview", font=dict(size=28, family="Arial")),
        barmode="stack",
        height=1000,
        margin=dict(t=120, b=50, l=60, r=40),
        legend=dict(orientation="h", yanchor="bottom", y=0.62, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Lost energy (kWh)", row=2, col=1)
    fig.update_xaxes(tickangle=45, row=2, col=1)

    fp = Path(out_dir) / "plant_overview.html"
    fig.write_html(fp, include_plotlyjs="cdn", full_html=True)
    return str(fp)


def make_all_figures(results, out_dir, verbose=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = results["cfg"]
    long_df = results["long_df"]
    saved = dict(soiling=[], iv=[], quality=[], plant=[])
    
    # 1. Per-String Dashboards
    for label, r in results["per_string"].items():
        try:
            p1 = plot_soiling_dashboard(label, r, cfg, out_dir)
            if p1: saved["soiling"].append(p1)
        except Exception as e:
            warnings.warn(f"plot failure for {label}: {e}")
            
    # 2. Plant-Level and Inverter-Level Data Quality
    try:
        if long_df is not None and not long_df.empty:
            plant_name = cfg.site.name.replace(" ", "_") if cfg and cfg.site else "Plant"
            
            # --- Plant Level Data Quality ---
            p_col = "Plant_P_abd"
            if p_col not in long_df.columns:
                # Fallback: aggregate all strings
                temp_df = long_df.groupby("ts")["P"].sum().reset_index().rename(columns={"P": "Plant_P_total"})
                p_col = "Plant_P_total"
                plant_df = long_df.drop_duplicates("ts").merge(temp_df, on="ts", how="left")
            else:
                plant_df = long_df
                
            p_plant_dq = plot_aggregated_data_quality(f"Plant__{plant_name}", plant_df, p_col, out_dir)
            if p_plant_dq: saved["quality"].append(p_plant_dq)
            
            # --- Inverter Level Data Quality (Skipped per user request) ---
            # if "inverter_id" in long_df.columns:
            #     inv_p_col = "Inverter_P_abd"
            #     if inv_p_col not in long_df.columns:
            #         # Fallback: aggregate by inverter
            #         inv_totals = long_df.groupby(["ts", "inverter_id"])["P"].sum().reset_index().rename(columns={"P": "Inverter_P_total"})
            #         inv_p_col = "Inverter_P_total"
            #         # For each inverter, we need a df
            #         for inv_id, inv_grp in inv_totals.groupby("inverter_id"):
            #             # Merge back some metadata if needed, but plot_aggregated_data_quality mostly needs ts and power
            #             # We'll also need irradiance
            #             meta_cols = ["ts", "irradiance KW/m2", "POA", "inverter_state"]
            #             existing_meta = [c for c in meta_cols if c in long_df.columns]
            #             inv_df = long_df[long_df["inverter_id"] == inv_id].drop_duplicates("ts")[existing_meta].merge(inv_grp, on="ts")
            #             
            #             p_inv_dq = plot_aggregated_data_quality(f"Inverter__{inv_id}", inv_df, inv_p_col, out_dir)
            #             if p_inv_dq: saved["quality"].append(p_inv_dq)
            #     else:
            #         for inv_id, inv_df in long_df.groupby("inverter_id"):
            #             p_inv_dq = plot_aggregated_data_quality(f"Inverter__{inv_id}", inv_df, inv_p_col, out_dir)
            #             if p_inv_dq: saved["quality"].append(p_inv_dq)
    except Exception as e:
        warnings.warn(f"plant/inverter data quality plot failure: {e}")

    # 3. Plant Overview Graph
    p4 = plot_plant_overview(results, cfg, out_dir)
    if p4: saved["plant"].append(p4)
    
    if verbose:
        n = sum(len(v) for v in saved.values())
        print(f"  Saved {n} figures (HTML) -> {out_dir}")
    return saved
