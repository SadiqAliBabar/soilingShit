"""Plots: soiling dashboard, IV diagnostics, data-quality, plant overview."""
from __future__ import annotations
from pathlib import Path
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from .config import PipelineConfig

VERDICT_COLORS = {
    "Clean":"#2BAE66", "Clean (post-wash)":"#1F8A4D",
    "Partial Recovery":"#79C26B",
    "Lt.Soiling":"#F5C95E", "Mod.Soiling":"#E89441", "Hvy.Soiling":"#C0392B",
    "Shading":"#3A6FB5", "Degradation":"#7E4FB5", "Mixed":"#8C7B6F",
    "Skipped":"#B0B0B0", "Insufficient":"#B0B0B0",
}


def _vc(v: str) -> str:
    if v in VERDICT_COLORS: return VERDICT_COLORS[v]
    for k, col in VERDICT_COLORS.items():
        if v.startswith(k): return col
    return "#5D6D7E"


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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                   gridspec_kw={"height_ratios":[3,1]})
    ax1.plot(daily["date"], daily[nci_col], "o-", lw=1.4, ms=4,
             color="#2C3E50", alpha=0.85, label=nci_col)
    if "PR" in daily.columns:
        ax1.plot(daily["date"], daily["PR"], "s--", lw=1, ms=3,
                 color="#7F8C8D", alpha=0.55, label="PR")
    if events is not None and not events.empty:
        for _, ev in events.iterrows():
            ed = pd.to_datetime(ev["event_date"])
            colour = ("#2BAE66" if ev["recovery_class"]=="Full recovery"
                      else ("#79C26B" if ev["recovery_class"]=="Partial recovery"
                            else "#E89441"))
            ax1.axvline(ed, color=colour, lw=1.8, alpha=0.7)
            ax1.text(ed, ax1.get_ylim()[1] * 0.97,
                     f" {ev['cause']}\n {ev['recovery_class']}\n +{ev['delta_nci']*100:.1f}pp",
                     fontsize=7, color=colour, va="top",
                     bbox=dict(boxstyle="round,pad=0.2", fc="white",
                               ec=colour, alpha=0.85))
    for seg in soil_f.get("segments", []):
        if not np.isfinite(seg.get("slope_per_day", np.nan)): continue
        s0 = pd.to_datetime(seg["start"]); s1 = pd.to_datetime(seg["end"])
        sub = daily[(daily["date"] >= s0) & (daily["date"] <= s1)]
        if sub.empty: continue
        x = (sub["date"] - sub["date"].min()).dt.days.values
        slope = seg["slope_per_day"]
        intercept = sub[nci_col].mean() - slope * x.mean()
        y = slope * x + intercept
        ax1.plot(sub["date"], y, "-", color="#C0392B", lw=2.2, alpha=0.7)
    base = float(daily["NCI_baseline"].iloc[0]) if "NCI_baseline" in daily.columns else 1.0
    ax1.axhline(1.0, color="#7F8C8D", lw=0.8, ls=":")
    ax1.axhline(base, color="#8E44AD", lw=1.0, ls="--",
                label=f"NCI baseline = {base:.3f}", alpha=0.7)
    ax1.set_ylim(min(0.55, ax1.get_ylim()[0]), 1.10)
    ax1.set_ylabel("NCI")
    ax1.set_title(f"{label}   |   Cluster: {cluster}   |   Verdict: {verdict}",
                  fontsize=11, color=_vc(verdict), weight="bold")
    ax1.legend(loc="lower right", fontsize=8); ax1.grid(True, alpha=0.3)

    if "rain_mm" in daily.columns and daily["rain_mm"].sum() > 0:
        ax2.bar(daily["date"], daily["rain_mm"], color="#3498DB", alpha=0.65)
        ax2.set_ylabel("Rain (mm)")
    else:
        ax2.bar(daily["date"], daily["n_valid"], color="#95A5A6", alpha=0.5)
        ax2.set_ylabel("# valid samples")
    ax2.set_xlabel("Date"); ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b-%d"))
    srr_text = (f"SRR full={soil_f.get('srr_pct_per_day', float('nan')):.3f} %/d | "
                f"current-seg SRR={soil_c.get('srr_pct_per_day', float('nan')):.3f} %/d | "
                f"loss(window)={soil_f.get('weighted_soiling_loss_pct', float('nan')):.1f}%")
    fig.text(0.5, 0.005, srr_text, ha="center", fontsize=8, color="#34495E")
    fig.tight_layout(rect=[0, 0.02, 1, 1])
    fp = Path(out_dir) / f"soiling_dashboard__{label}.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
    return str(fp)


def plot_iv_diagnostics(label, df, result, cfg, out_dir):
    if df is None or df.empty: return None
    sub = df[(df["POA"] > 100) & df["I"].notna() & df["V"].notna()]
    if len(sub) < 50: return None
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    sc = axes[0].scatter(sub["V"], sub["I"], c=sub["POA"], s=4, alpha=0.35, cmap="viridis")
    plt.colorbar(sc, ax=axes[0], label="POA W/m²")
    axes[0].set_xlabel("V_dc (V)"); axes[0].set_ylabel("I_dc (A)")
    axes[0].set_title(f"Measured I-V cloud — {label}", fontsize=10)
    axes[0].grid(True, alpha=0.3)

    sdm = result.get("sdm", {}); sdm_m = result.get("sdm_metrics", {}) or {}
    if sdm and sdm.get("success"):
        try:
            from .sdm import iv_curve_from_sdm
            iv = iv_curve_from_sdm(sdm, cfg.module)
            axes[1].plot(iv["V"], iv["I"], "-", lw=2, color="#C0392B", label="SDM@STC")
            t = (f"Isc x{sdm_m.get('isc_stc_ratio',float('nan')):.3f}\n"
                 f"Voc x{sdm_m.get('voc_stc_ratio',float('nan')):.3f}\n"
                 f"FF  x{sdm_m.get('ff_stc_ratio', float('nan')):.3f}")
            axes[1].text(0.05, 0.05, t, transform=axes[1].transAxes,
                         fontsize=9, va="bottom",
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))
        except Exception as e:
            axes[1].text(0.5, 0.5, f"SDM render failed: {e}",
                         transform=axes[1].transAxes, ha="center")
    else:
        axes[1].text(0.5, 0.5, f"SDM unavailable\n({sdm.get('reason','no fit')})",
                     transform=axes[1].transAxes, ha="center",
                     fontsize=10, color="#7F8C8D")
    axes[1].set_xlabel("V_dc (V)"); axes[1].set_ylabel("I_dc (A)")
    axes[1].set_title("Reconstructed I-V at STC", fontsize=10)
    axes[1].grid(True, alpha=0.3); fig.tight_layout()
    fp = Path(out_dir) / f"iv_diagnostics__{label}.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
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
    fig, ax = plt.subplots(figsize=(12, 4.5))
    cmap = {"OK":"#2BAE66","CURTAILED":"#C0392B","FAULT":"#7F1D1D",
            "STANDBY":"#95A5A6","NIGHT":"#34495E",
            "TRANSIENT":"#F5C95E","IV_SCAN":"#7E4FB5"}
    pct.plot(kind="bar", stacked=True, ax=ax,
             color=[cmap[c] for c in order], width=0.95)
    ax.set_ylabel("% of intervals")
    ax.set_title(f"Daily data quality — {label}", fontsize=10)
    ax.set_xticklabels([str(x)[:10] for x in pct.index], rotation=45,
                       ha="right", fontsize=7)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    ax.set_ylim(0, 100); fig.tight_layout()
    fp = Path(out_dir) / f"data_quality__{label}.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
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

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.4], hspace=0.4, wspace=0.3)
    ax_pie = fig.add_subplot(gs[0, 0])
    vcounts = summary["verdict"].value_counts()
    cols = [_vc(v) for v in vcounts.index]
    ax_pie.pie(vcounts.values, labels=vcounts.index, colors=cols,
               autopct="%1.0f%%", startangle=90, textprops={"fontsize":9})
    ax_pie.set_title("Verdict distribution", fontsize=10)

    pl = results.get("plant_losses", {})
    cur = cfg.site.currency
    ax_kpi = fig.add_subplot(gs[0, 1]); ax_kpi.axis("off")
    txt = (f"Plant: {cfg.site.name}\n"
           f"Period: {pl.get('period_days',0)} d   tariff: {cfg.site.tariff:.1f} {cur}/kWh\n\n"
           f"Soiling loss    : {pl.get('soiling_kwh',0):>10,.0f} kWh   "
           f"({cur} {pl.get('soiling_pkr',0):>12,.0f})\n"
           f"Curtailment loss: {pl.get('curtailment_kwh',0):>10,.0f} kWh   "
           f"({cur} {pl.get('curtailment_pkr',0):>12,.0f})\n"
           f"Total avoidable : {pl.get('total_avoidable_kwh',0):>10,.0f} kWh   "
           f"({cur} {pl.get('total_avoidable_pkr',0):>12,.0f})\n\n"
           f"Annualised      : {pl.get('annualised_kwh',0):>10,.0f} kWh   "
           f"({cur} {pl.get('annualised_pkr',0):>12,.0f})")
    ax_kpi.text(0, 0.95, txt, transform=ax_kpi.transAxes,
                fontsize=11, family="monospace", va="top")

    ax_bar = fig.add_subplot(gs[1, :])
    s = summary.sort_values("total_pkr", ascending=False)
    x = np.arange(len(s))
    ax_bar.bar(x, s["soiling_kwh"], color="#E89441", label="Soiling kWh")
    ax_bar.bar(x, s["curt_kwh"], bottom=s["soiling_kwh"],
               color="#C0392B", label="Curtailment kWh")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(s["label"], rotation=45, ha="right", fontsize=7)
    ax_bar.set_ylabel("Lost energy (kWh)")
    ax_bar.set_title("Avoidable losses per string", fontsize=10)
    ax_bar.grid(True, axis="y", alpha=0.3)
    ax_bar.legend(loc="upper right", fontsize=9)

    fig.suptitle(f"{cfg.site.name} — Diagnostics Overview",
                 fontsize=13, weight="bold")
    fp = Path(out_dir) / "plant_overview.png"
    fig.savefig(fp, dpi=120, bbox_inches="tight"); plt.close(fig)
    return str(fp)


def make_all_figures(results, out_dir, verbose=True):
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    cfg = results["cfg"]
    long_df = results["long_df"]
    saved = dict(soiling=[], iv=[], quality=[], plant=[])
    for label, r in results["per_string"].items():
        try:
            df_str = long_df[long_df["string_label"] == label]
            p1 = plot_soiling_dashboard(label, r, cfg, out_dir)
            if p1: saved["soiling"].append(p1)
            p2 = plot_iv_diagnostics(label, df_str, r, cfg, out_dir)
            if p2: saved["iv"].append(p2)
            p3 = plot_data_quality(label, df_str, r, cfg, out_dir)
            if p3: saved["quality"].append(p3)
        except Exception as e:
            warnings.warn(f"plot failure for {label}: {e}")
    p4 = plot_plant_overview(results, cfg, out_dir)
    if p4: saved["plant"].append(p4)
    if verbose:
        n = sum(len(v) for v in saved.values())
        print(f"  Saved {n} figures -> {out_dir}")
    return saved
