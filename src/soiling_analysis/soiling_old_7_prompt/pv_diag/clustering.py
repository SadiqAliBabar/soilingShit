"""Combined MPPT + orientation clustering."""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PipelineConfig
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


def build_peer_groups(
    string_meta: dict,
    string_dfs: dict,
    cfg: PipelineConfig,
) -> dict:
    """Build per-string peer groups via a 4-level structural ladder.

    On plants where every MPPT port has a single string, each full_cluster is
    unique, so estimate_cluster_clean_baseline always returns None and Layer 2
    is dead.  This function overcomes that by grouping strings on progressively
    relaxed criteria and returning the first level that yields a group large
    enough (>= cfg.peer_min_members members, self included) for a valid median.

    Ladder levels
    -------------
    1 – Same orientation bin  AND  DC capacity within cfg.peer_capacity_tolerance
        AND  same commissioning year (when available in string_meta).
    2 – Same orientation bin only (inverter/MPPT/capacity/age dropped).
    3 – Whole-plant pool.  Caller should use physics-corrected relative_NCI;
        NCI_noon is the current fallback until that column is added.
    4 – No valid peer group.  Cluster baseline will be None for this string.

    Parameters
    ----------
    string_meta : {label: meta_dict}
        Keys used: ``azimuth``, ``tilt``, ``commissioning_year`` (optional).
    string_dfs : {label: DataFrame}
        Keys used: ``pv_capacity`` column (optional).
    cfg : PipelineConfig

    Returns
    -------
    {string_label: {"level": int (1–4), "peers": [list of other labels]}}
    """
    orient_map = cluster_by_azimuth_tilt(string_meta)
    labels = list(string_meta.keys())

    # DC capacity: first non-null value per string.
    cap_map: dict = {}
    for label in labels:
        df = string_dfs.get(label)
        cap = None
        if df is not None and "pv_capacity" in df.columns:
            cap_vals = pd.to_numeric(df["pv_capacity"], errors="coerce").dropna()
            if not cap_vals.empty:
                cap = float(cap_vals.iloc[0])
        cap_map[label] = cap

    # Commissioning year per string (optional).
    year_map: dict = {}
    for label, meta in string_meta.items():
        yr = meta.get("commissioning_year")
        year_map[label] = int(yr) if yr is not None else None

    # Group is valid when it has at least peer_min_members members (self + peers).
    min_peers = cfg.peer_min_members - 1

    result: dict = {}
    for label in labels:
        orient_bin = orient_map.get(label)
        cap = cap_map.get(label)
        yr = year_map.get(label)

        # ---- Level 1: orient + capacity tolerance + age bracket ----
        peers_l1: list = []
        if orient_bin is not None:
            for other in labels:
                if other == label:
                    continue
                if orient_map.get(other) != orient_bin:
                    continue
                other_cap = cap_map.get(other)
                if cap is not None and other_cap is not None:
                    max_cap = max(cap, other_cap)
                    if max_cap > 0 and abs(cap - other_cap) / max_cap > cfg.peer_capacity_tolerance:
                        continue
                other_yr = year_map.get(other)
                if yr is not None and other_yr is not None and yr != other_yr:
                    continue
                peers_l1.append(other)

        if len(peers_l1) >= min_peers:
            result[label] = {"level": 1, "peers": peers_l1}
            continue

        # ---- Level 2: orientation bin only ----
        peers_l2: list = (
            [o for o in labels if o != label and orient_map.get(o) == orient_bin]
            if orient_bin is not None else []
        )
        if len(peers_l2) >= min_peers:
            result[label] = {"level": 2, "peers": peers_l2}
            continue

        # ---- Level 3: whole plant ----
        peers_l3 = [o for o in labels if o != label]
        if len(peers_l3) >= min_peers:
            result[label] = {"level": 3, "peers": peers_l3}
            continue

        # ---- Level 4: no valid peer group ----
        result[label] = {"level": 4, "peers": []}

    return result


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
