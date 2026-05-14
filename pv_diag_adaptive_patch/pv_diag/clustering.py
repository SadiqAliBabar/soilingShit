"""Combined MPPT + orientation clustering."""
from __future__ import annotations
import pandas as pd
from .orientation import cluster_by_azimuth_tilt, cluster_by_mppt


def assign_clusters(string_dfs, string_meta, cluster_method="combined"):
    mppt_map = cluster_by_mppt(string_dfs)
    orient_map = cluster_by_azimuth_tilt(string_meta)
    out = {}
    for label in string_meta:
        m = mppt_map.get(label, "MPPT_unknown")
        o = orient_map.get(label, "az180_t25")
        if cluster_method == "mppt": full = m
        elif cluster_method == "orient": full = o
        else: full = f"{m}__{o}"
        out[label] = dict(mppt_cluster=m, orient_cluster=o, full_cluster=full)
    return out


def cluster_summary(cluster_map, string_meta):
    rows = []
    for label, c in cluster_map.items():
        m = string_meta.get(label, {})
        rows.append(dict(
            string_label=label,
            inverter=m.get("inverter_id", ""),
            mppt=m.get("mppt_id", ""),
            azimuth=m.get("azimuth", float("nan")),
            tilt=m.get("tilt", float("nan")),
            mppt_cluster=c["mppt_cluster"],
            orient_cluster=c["orient_cluster"],
            full_cluster=c["full_cluster"],
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["full_cluster","string_label"]).reset_index(drop=True)
    return df
