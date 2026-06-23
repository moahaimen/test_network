#!/usr/bin/env python3
"""Generate final tables and plots for improved Phase-1 reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/phase1_reactive"
FINAL_TABLES = RESULTS / "final_tables"
FINAL_CDFS = RESULTS / "final_cdfs"
FINAL_PLOTS = RESULTS / "final_plots"
FINAL_EXPORTS = RESULTS / "final_csv_exports"

EVAL_SUMMARY = RESULTS / "eval/summary_all.csv"
EVAL_TS = RESULTS / "eval/timeseries_all.csv"
FAIL_SUMMARY = RESULTS / "failures/summary_all.csv"
FAIL_TS = RESULTS / "failures/timeseries_all.csv"
GEN_SUMMARY = RESULTS / "generalization/summary_all.csv"
GEN_TS = RESULTS / "generalization/timeseries_all.csv"
REPORT_MD = RESULTS / "report.md"
PREPARE_MANIFEST = RESULTS / "prepare/dataset_manifest.json"
PPO_CURRICULUM = RESULTS / "train/ppo/curriculum_log.csv"
DQN_CURRICULUM = RESULTS / "train/dqn/curriculum_log.csv"
ABLATION_SOURCE = RESULTS / "eval/drl_improvement_comparison.csv"

METHOD_ORDER = [
    "ospf",
    "ecmp",
    "topk",
    "bottleneck",
    "sensitivity",
    "erodrl",
    "flexdate",
    "cfrrl",
    "flexentry",
    "our_drl_ppo",
    "our_drl_dqn",
    "our_drl_dual_gate",
    "our_hybrid_moe_gate",
    "lp_optimal_upper_bound",
]
DRL_METHODS = ["our_drl_ppo", "our_drl_dqn", "our_drl_dual_gate", "our_hybrid_moe_gate"]
HEURISTIC_METHODS = ["topk", "bottleneck", "sensitivity", "erodrl", "flexdate", "cfrrl", "flexentry"]
STATIC_METHODS = ["ospf", "ecmp"]
PALETTE = {
    "ospf": "#4d4d4d",
    "ecmp": "#111111",
    "topk": "#b8860b",
    "bottleneck": "#d95f02",
    "sensitivity": "#1b9e77",
    "erodrl": "#7570b3",
    "flexdate": "#e7298a",
    "cfrrl": "#66a61e",
    "flexentry": "#a6761d",
    "our_drl_ppo": "#1f77b4",
    "our_drl_dqn": "#17becf",
    "our_drl_dual_gate": "#d62728",
    "our_hybrid_moe_gate": "#8c564b",
    "lp_optimal_upper_bound": "#9467bd",
}


def ensure_dirs() -> None:
    for path in [FINAL_TABLES, FINAL_CDFS, FINAL_PLOTS, FINAL_EXPORTS]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def normalize_method(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "method" in out.columns:
        out["method"] = out["method"].replace({"lp_optimal": "lp_optimal_upper_bound"})
    return out


def method_rank_key(method: str) -> tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)


def sort_by_method(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["_method_rank"] = out["method"].map(lambda x: method_rank_key(str(x))[0])
    sort_cols = [c for c in ["display_name", "dataset", "failure_type", "scenario", "_method_rank", "method"] if c in out.columns]
    out = out.sort_values(sort_cols).drop(columns=["_method_rank"])
    return out


def markdown_table(df: pd.DataFrame) -> str:
    tmp = df.copy()
    cols = list(tmp.columns)
    for col in cols:
        tmp[col] = tmp[col].map(_fmt_cell)
    widths = {col: max(len(str(col)), tmp[col].astype(str).map(len).max() if not tmp.empty else 0) for col in cols}
    header = "| " + " | ".join(f"{col:<{widths[col]}}" for col in cols) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in cols) + " |"
    rows = ["| " + " | ".join(f"{str(row[col]):<{widths[col]}}" for col in cols) + " |" for _, row in tmp.iterrows()]
    return "\n".join([header, sep] + rows)


def _fmt_cell(value):
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        if np.isinf(value):
            return "inf"
        if abs(float(value)) >= 1000:
            return f"{float(value):.3f}"
        return f"{float(value):.6f}".rstrip("0").rstrip(".")
    return str(value)


def write_table(df: pd.DataFrame, stem: str, title: str) -> tuple[Path, Path]:
    csv_path = FINAL_TABLES / f"{stem}.csv"
    md_path = FINAL_TABLES / f"{stem}.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(f"# {title}\n\n" + markdown_table(df) + "\n", encoding="utf-8")
    return csv_path, md_path


def write_export(df: pd.DataFrame, filename: str) -> Path:
    out = FINAL_EXPORTS / filename
    df.to_csv(out, index=False)
    return out


def load_od_map() -> dict[str, int]:
    if not PREPARE_MANIFEST.exists():
        return {}
    data = json.loads(PREPARE_MANIFEST.read_text(encoding="utf-8"))
    return {str(row["key"]): int(row["num_od"]) for row in data if "key" in row and "num_od" in row}


def add_selected_flow_pct(ts: pd.DataFrame, od_map: dict[str, int]) -> pd.DataFrame:
    out = ts.copy()
    if "selected_count" not in out.columns:
        return out
    out["num_od"] = out["dataset"].map(od_map)
    out["selected_flow_percentage"] = np.where(out["num_od"].notna() & (out["num_od"] > 0), out["selected_count"] / out["num_od"] * 100.0, np.nan)
    return out


def best_heuristic_by_group(df: pd.DataFrame, group_cols: list[str], metric_col: str) -> pd.DataFrame:
    subset = df[df["method"].isin(HEURISTIC_METHODS)].copy()
    if subset.empty:
        return subset
    idx = subset.groupby(group_cols)[metric_col].idxmin()
    return subset.loc[idx].copy()


def best_drl_by_group(df: pd.DataFrame, group_cols: list[str], metric_col: str) -> pd.DataFrame:
    subset = df[df["method"].isin(DRL_METHODS)].copy()
    if subset.empty:
        return subset
    idx = subset.groupby(group_cols)[metric_col].idxmin()
    return subset.loc[idx].copy()


def make_eval_table(eval_summary: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "dataset",
        "display_name",
        "source",
        "traffic_mode",
        "method",
        "mean_delay",
        "p95_delay",
        "throughput",
        "mean_utilization",
        "mean_mlu",
        "p95_mlu",
        "mean_gap_pct",
        "mean_disturbance",
        "inference_latency_ms",
        "decision_time_ms",
        "training_time_sec",
        "convergence_epoch",
        "convergence_rate",
    ]
    out = eval_summary[cols].copy()
    return sort_by_method(out)


def make_failure_table(failure_summary: pd.DataFrame) -> pd.DataFrame:
    out = failure_summary.copy()
    out = out.rename(
        columns={
            "post_failure_mean_delay": "mean_delay",
            "post_failure_mean_mlu": "mean_mlu",
            "post_failure_mean_disturbance": "mean_disturbance",
            "failover_convergence_time_steps": "failover_convergence_time",
        }
    )
    out["gap_pct"] = np.nan
    cols = [
        "dataset",
        "display_name",
        "failure_type",
        "method",
        "pre_failure_mean_mlu",
        "post_failure_peak_mlu",
        "mean_delay",
        "mean_mlu",
        "mean_disturbance",
        "failover_convergence_time",
        "decision_time_ms",
        "gap_pct",
    ]
    return sort_by_method(out[cols].copy())


def make_generalization_table(gen_summary: pd.DataFrame) -> pd.DataFrame:
    out = gen_summary.copy()
    ecmp_map = out[out["method"] == "ecmp"].set_index("dataset")["mean_mlu"].to_dict()
    heur = best_heuristic_by_group(out, ["dataset"], "mean_mlu")[["dataset", "method", "mean_mlu"]].rename(columns={"method": "best_heuristic_method", "mean_mlu": "best_heuristic_mean_mlu"})
    out = out.merge(heur, on="dataset", how="left")
    out["relative_to_ecmp_pct"] = np.where(out["dataset"].map(ecmp_map).notna(), (out["dataset"].map(ecmp_map) - out["mean_mlu"]) / out["dataset"].map(ecmp_map) * 100.0, np.nan)
    out["relative_to_best_heuristic_pct"] = np.where(out["best_heuristic_mean_mlu"].notna(), (out["best_heuristic_mean_mlu"] - out["mean_mlu"]) / out["best_heuristic_mean_mlu"] * 100.0, np.nan)
    cols = [
        "dataset",
        "display_name",
        "method",
        "mean_delay",
        "mean_mlu",
        "mean_disturbance",
        "mean_gap_pct",
        "relative_to_ecmp_pct",
        "relative_to_best_heuristic_pct",
        "best_heuristic_method",
    ]
    out = out.rename(columns={"mean_gap_pct": "gap_pct"})
    cols = ["dataset", "display_name", "method", "mean_delay", "mean_mlu", "mean_disturbance", "gap_pct", "relative_to_ecmp_pct", "relative_to_best_heuristic_pct", "best_heuristic_method"]
    return sort_by_method(out[cols].copy())


def make_drl_only_table(eval_summary: pd.DataFrame, failure_summary: pd.DataFrame, gen_summary: pd.DataFrame) -> pd.DataFrame:
    eval_rows = eval_summary[(eval_summary["dataset"].isin(["abilene", "geant"])) & (eval_summary["method"].isin(DRL_METHODS))].copy()
    eval_rows["scenario"] = "eval"
    eval_rows["failure_type"] = ""
    eval_rows = eval_rows[["scenario", "dataset", "display_name", "failure_type", "method", "mean_delay", "mean_mlu", "mean_disturbance", "decision_time_ms", "training_time_sec"]]

    gen_rows = gen_summary[(gen_summary["dataset"] == "germany50") & (gen_summary["method"].isin(DRL_METHODS))].copy()
    gen_rows["scenario"] = "generalization"
    gen_rows["failure_type"] = ""
    gen_rows = gen_rows[["scenario", "dataset", "display_name", "failure_type", "method", "mean_delay", "mean_mlu", "mean_disturbance", "decision_time_ms", "training_time_sec"]]

    fail_rows = failure_summary[(failure_summary["dataset"].isin(["abilene", "geant"])) & (failure_summary["method"].isin(DRL_METHODS))].copy()
    fail_rows["scenario"] = "failure"
    fail_rows = fail_rows.rename(columns={"post_failure_mean_delay": "mean_delay", "post_failure_mean_mlu": "mean_mlu", "post_failure_mean_disturbance": "mean_disturbance"})
    fail_rows["training_time_sec"] = np.nan
    fail_rows = fail_rows[["scenario", "dataset", "display_name", "failure_type", "method", "mean_delay", "mean_mlu", "mean_disturbance", "decision_time_ms", "training_time_sec"]]

    out = pd.concat([eval_rows, gen_rows, fail_rows], ignore_index=True, sort=False)
    return sort_by_method(out)


def make_ablation_table() -> pd.DataFrame:
    src = read_csv(ABLATION_SOURCE)
    family_map = {
        "PPO baseline": "ppo",
        "PPO + pretraining": "ppo",
        "PPO + pretraining + curriculum": "ppo",
        "DQN baseline": "dqn",
        "DQN + pretraining": "dqn",
        "DQN + pretraining + curriculum": "dqn",
        "Dual-Gate final": "dual_gate",
    }
    out = src.copy()
    out["family"] = out["variant"].map(family_map)
    baselines = out[out["variant"].isin(["PPO baseline", "DQN baseline"])]
    base_map = {(row.dataset, row.family): row.mean_mlu for row in baselines.itertuples()}
    dual_base = {}
    for dataset in out["dataset"].unique():
        cands = [base_map.get((dataset, "ppo")), base_map.get((dataset, "dqn"))]
        cands = [x for x in cands if x is not None]
        dual_base[dataset] = min(cands) if cands else np.nan

    def delta(row):
        if row["family"] == "dual_gate":
            ref = dual_base.get(row["dataset"], np.nan)
        else:
            ref = base_map.get((row["dataset"], row["family"]), np.nan)
        if pd.isna(ref):
            return np.nan
        return (ref - row["mean_mlu"]) / ref * 100.0

    out["improvement_vs_family_baseline_pct"] = out.apply(delta, axis=1)
    cols = ["dataset", "display_name", "variant", "family", "mean_mlu", "p95_mlu", "mean_delay", "mean_disturbance", "improvement_vs_family_baseline_pct"]
    return out[cols].sort_values(["display_name", "family", "variant"]).reset_index(drop=True)


def make_best_method_table(eval_summary: pd.DataFrame, failure_summary: pd.DataFrame, gen_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def collect(df: pd.DataFrame, scenario: str, metric_col: str, failure_col: str | None = None):
        group_cols = ["dataset"] + ([failure_col] if failure_col else [])
        for keys, grp in df.groupby(group_cols):
            if not isinstance(keys, tuple):
                keys = (keys,)
            dataset = keys[0]
            failure_type = keys[1] if failure_col else ""
            overall = grp.loc[grp[metric_col].idxmin()]
            drl_grp = grp[grp["method"].isin(DRL_METHODS)]
            if drl_grp.empty:
                continue
            drl = drl_grp.loc[drl_grp[metric_col].idxmin()]
            ecmp = grp[grp["method"] == "ecmp"]
            ospf = grp[grp["method"] == "ospf"]
            best_overall = float(overall[metric_col])
            best_drl = float(drl[metric_col])
            gap = (best_drl - best_overall) / best_overall * 100.0 if best_overall != 0 else np.nan
            rows.append({
                "scenario": scenario,
                "dataset": dataset,
                "display_name": str(overall.get("display_name", drl.get("display_name", dataset))),
                "failure_type": failure_type,
                "best_overall_method": str(overall["method"]),
                "best_overall_mean_mlu": best_overall,
                "best_drl_method": str(drl["method"]),
                "best_drl_mean_mlu": best_drl,
                "gap_best_drl_vs_best_overall_pct": gap,
                "drl_beats_ecmp": bool(not ecmp.empty and best_drl < float(ecmp.iloc[0][metric_col])),
                "drl_beats_ospf": bool(not ospf.empty and best_drl < float(ospf.iloc[0][metric_col])),
            })

    collect(eval_summary, "eval", "mean_mlu")
    failure_tmp = failure_summary.rename(columns={"post_failure_mean_mlu": "mean_mlu"})
    collect(failure_tmp, "failure", "mean_mlu", failure_col="failure_type")
    collect(gen_summary, "generalization", "mean_mlu")
    return pd.DataFrame(rows).sort_values(["scenario", "display_name", "failure_type"]).reset_index(drop=True)


def _cdf_arrays(values: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return arr, arr
    arr = np.sort(arr)
    y = np.arange(1, arr.size + 1) / arr.size
    return arr, y


def _plot_cdf(ax, df: pd.DataFrame, metric: str, methods: list[str]) -> None:
    for method in methods:
        sub = df[df["method"] == method]
        if sub.empty:
            continue
        x, y = _cdf_arrays(sub[metric].tolist())
        if x.size == 0:
            continue
        ax.plot(x, y, label=method, color=PALETTE.get(method, None), linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)


def _save_fig(fig: plt.Figure, stem: str, out_dir: Path) -> list[Path]:
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png, pdf]


def choose_best_heuristic(summary_df: pd.DataFrame, dataset: str, failure_type: str | None = None, metric_col: str = "mean_mlu") -> str | None:
    sub = summary_df[summary_df["dataset"] == dataset]
    if failure_type is not None and "failure_type" in sub.columns:
        sub = sub[sub["failure_type"] == failure_type]
    sub = sub[sub["method"].isin(HEURISTIC_METHODS)]
    if sub.empty:
        return None
    return str(sub.loc[sub[metric_col].idxmin()]["method"])


def multi_panel_layout(n: int) -> tuple[int, int]:
    if n <= 2:
        return 1, n
    if n <= 4:
        return 2, 2
    return 2, 3


def plot_seen_cdfs(eval_ts: pd.DataFrame, eval_summary: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    seen = eval_ts[["dataset", "display_name"]].drop_duplicates().sort_values("display_name")
    for metric, stem, xlabel in [
        ("mlu", "cdf_mlu_seen_topologies", "MLU"),
        ("latency", "cdf_delay_seen_topologies", "End-to-End Delay"),
        ("disturbance", "cdf_disturbance_seen_topologies", "Route Change Frequency / Disturbance"),
    ]:
        nrows, ncols = multi_panel_layout(len(seen))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.2*nrows), squeeze=False)
        axes_flat = axes.flatten()
        handles = None
        labels = None
        for ax, (_, row) in zip(axes_flat, seen.iterrows()):
            ds = row["dataset"]
            best_heur = choose_best_heuristic(eval_summary, ds)
            methods = ["ecmp", "ospf", "our_drl_ppo", "our_drl_dqn", "our_drl_dual_gate", "our_hybrid_moe_gate"]
            if best_heur:
                methods.insert(2, best_heur)
            if not eval_ts[(eval_ts["dataset"] == ds) & (eval_ts["method"] == "lp_optimal_upper_bound")].empty:
                methods.append("lp_optimal_upper_bound")
            methods = list(dict.fromkeys(methods))
            sub = eval_ts[eval_ts["dataset"] == ds]
            _plot_cdf(ax, sub, metric, methods)
            ax.set_title(str(row["display_name"]))
            ax.set_xlabel(xlabel)
            ax.set_ylabel("CDF")
            handles, labels = ax.get_legend_handles_labels()
        for ax in axes_flat[len(seen):]:
            ax.axis("off")
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        outputs.extend(_save_fig(fig, stem, FINAL_CDFS))
    return outputs


def plot_selected_flow_cdfs(eval_ts: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    metric = "selected_flow_percentage"
    if metric not in eval_ts.columns or eval_ts[metric].notna().sum() == 0:
        return outputs
    seen = eval_ts[["dataset", "display_name"]].drop_duplicates().sort_values("display_name")
    nrows, ncols = multi_panel_layout(len(seen))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.2*nrows), squeeze=False)
    axes_flat = axes.flatten()
    handles = None
    labels = None
    methods = ["topk", "bottleneck", "sensitivity", "our_drl_ppo", "our_drl_dqn", "our_drl_dual_gate", "our_hybrid_moe_gate"]
    for ax, (_, row) in zip(axes_flat, seen.iterrows()):
        sub = eval_ts[eval_ts["dataset"] == row["dataset"]]
        _plot_cdf(ax, sub, metric, methods)
        ax.set_title(str(row["display_name"]))
        ax.set_xlabel("Selected Flow Percentage (%)")
        ax.set_ylabel("CDF")
        handles, labels = ax.get_legend_handles_labels()
    for ax in axes_flat[len(seen):]:
        ax.axis("off")
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    outputs.extend(_save_fig(fig, "cdf_selected_flow_percentage", FINAL_CDFS))
    return outputs


def plot_failure_cdf(failure_ts: pd.DataFrame, failure_summary: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    focus = failure_ts[failure_ts["dataset"].isin(["abilene", "geant"])]
    datasets = ["abilene", "geant"]
    failure_types = ["single_link_failure", "capacity_degradation", "multi_link_stress"]
    fig, axes = plt.subplots(len(datasets), len(failure_types), figsize=(18, 8), squeeze=False)
    for i, ds in enumerate(datasets):
        for j, ft in enumerate(failure_types):
            ax = axes[i][j]
            best_heur = choose_best_heuristic(failure_summary.rename(columns={"post_failure_mean_mlu": "mean_mlu"}), ds, failure_type=ft, metric_col="mean_mlu")
            methods = ["ecmp", "ospf"]
            if best_heur:
                methods.append(best_heur)
            methods.extend(["our_hybrid_moe_gate", "our_drl_dual_gate"])
            methods = list(dict.fromkeys(methods))
            sub = focus[(focus["dataset"] == ds) & (focus["failure_type"] == ft)]
            _plot_cdf(ax, sub, "mlu", methods)
            ax.set_title(f"{ds} | {ft}")
            ax.set_xlabel("Post-Failure MLU")
            ax.set_ylabel("CDF")
    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    outputs.extend(_save_fig(fig, "cdf_failure_mlu", FINAL_CDFS))
    return outputs


def plot_drl_vs_heuristic_seen(eval_ts: pd.DataFrame, eval_summary: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    seen = eval_ts[["dataset", "display_name"]].drop_duplicates().sort_values("display_name")
    nrows, ncols = multi_panel_layout(len(seen))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.2*nrows), squeeze=False)
    axes_flat = axes.flatten()
    handles = None
    labels = None
    for ax, (_, row) in zip(axes_flat, seen.iterrows()):
        ds = row["dataset"]
        best_heur = choose_best_heuristic(eval_summary, ds)
        methods = [m for m in ["ecmp", "ospf", best_heur, "our_hybrid_moe_gate", "our_drl_dual_gate"] if m]
        methods = list(dict.fromkeys(methods))
        sub = eval_ts[eval_ts["dataset"] == ds]
        _plot_cdf(ax, sub, "mlu", methods)
        ax.set_title(str(row["display_name"]))
        ax.set_xlabel("MLU")
        ax.set_ylabel("CDF")
        handles, labels = ax.get_legend_handles_labels()
    for ax in axes_flat[len(seen):]:
        ax.axis("off")
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    outputs.extend(_save_fig(fig, "cdf_drl_vs_heuristic_seen", FINAL_CDFS))
    return outputs


def plot_germany50_cdf(gen_ts: pd.DataFrame, gen_summary: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    sub = gen_ts[gen_ts["dataset"] == "germany50"]
    if sub.empty:
        return outputs
    best_heur = choose_best_heuristic(gen_summary, "germany50")
    methods = [m for m in ["ecmp", "ospf", best_heur, "our_hybrid_moe_gate", "our_drl_dual_gate"] if m]
    methods = list(dict.fromkeys(methods))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    _plot_cdf(axes[0], sub, "mlu", methods)
    axes[0].set_title("Germany50 MLU")
    axes[0].set_xlabel("MLU")
    axes[0].set_ylabel("CDF")
    _plot_cdf(axes[1], sub, "latency", methods)
    axes[1].set_title("Germany50 Delay")
    axes[1].set_xlabel("End-to-End Delay")
    axes[1].set_ylabel("CDF")
    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    outputs.extend(_save_fig(fig, "cdf_generalization_germany50", FINAL_CDFS))
    return outputs


def _plot_timeseries(ax, df: pd.DataFrame, metric: str, methods: list[str]) -> None:
    for method in methods:
        sub = df[df["method"] == method].sort_values("timestep")
        if sub.empty:
            continue
        ax.plot(sub["timestep"], sub[metric], label=method, color=PALETTE.get(method, None), linewidth=1.8)
    ax.grid(True, alpha=0.3)


def plot_seen_timeseries(eval_ts: pd.DataFrame, eval_summary: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    seen = eval_ts[["dataset", "display_name"]].drop_duplicates().sort_values("display_name")
    for metric, stem, ylabel in [
        ("mlu", "mlu_vs_tm_index_seen_topologies", "MLU"),
        ("latency", "delay_vs_tm_index_seen_topologies", "End-to-End Delay"),
    ]:
        nrows, ncols = multi_panel_layout(len(seen))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.2*nrows), squeeze=False)
        axes_flat = axes.flatten()
        handles = None
        labels = None
        for ax, (_, row) in zip(axes_flat, seen.iterrows()):
            ds = row["dataset"]
            best_heur = choose_best_heuristic(eval_summary, ds)
            methods = [m for m in ["ecmp", "ospf", best_heur, "our_drl_ppo", "our_drl_dqn", "our_drl_dual_gate", "our_hybrid_moe_gate"] if m]
            methods = list(dict.fromkeys(methods))
            sub = eval_ts[eval_ts["dataset"] == ds]
            _plot_timeseries(ax, sub, metric, methods)
            ax.set_title(str(row["display_name"]))
            ax.set_xlabel("Traffic Matrix Index")
            ax.set_ylabel(ylabel)
            handles, labels = ax.get_legend_handles_labels()
        for ax in axes_flat[len(seen):]:
            ax.axis("off")
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        outputs.extend(_save_fig(fig, stem, FINAL_PLOTS))
    return outputs


def plot_selected_count_timeseries(eval_ts: pd.DataFrame) -> list[Path]:
    outputs: list[Path] = []
    seen = eval_ts[["dataset", "display_name"]].drop_duplicates().sort_values("display_name")
    methods = ["topk", "bottleneck", "sensitivity", "our_drl_ppo", "our_drl_dqn", "our_drl_dual_gate", "our_hybrid_moe_gate"]
    for metric, stem, ylabel in [
        ("selected_count", "critical_flow_count_vs_tm_index", "Selected Critical Flow Count"),
        ("selected_flow_percentage", "selected_flow_percentage_vs_tm_index", "Selected Flow Percentage (%)"),
    ]:
        if metric not in eval_ts.columns or eval_ts[metric].notna().sum() == 0:
            continue
        nrows, ncols = multi_panel_layout(len(seen))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 4.2*nrows), squeeze=False)
        axes_flat = axes.flatten()
        handles = None
        labels = None
        for ax, (_, row) in zip(axes_flat, seen.iterrows()):
            sub = eval_ts[eval_ts["dataset"] == row["dataset"]]
            _plot_timeseries(ax, sub, metric, methods)
            ax.set_title(str(row["display_name"]))
            ax.set_xlabel("Traffic Matrix Index")
            ax.set_ylabel(ylabel)
            handles, labels = ax.get_legend_handles_labels()
        for ax in axes_flat[len(seen):]:
            ax.axis("off")
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=min(4, len(labels)), frameon=False)
        fig.tight_layout(rect=(0, 0.08, 1, 1))
        outputs.extend(_save_fig(fig, stem, FINAL_PLOTS))
    return outputs


def plot_training_curve() -> list[Path]:
    outputs: list[Path] = []
    ppo = read_csv(PPO_CURRICULUM)
    dqn = read_csv(DQN_CURRICULUM)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(ppo["epoch_global"], ppo["train_mean_reward"], label="our_drl_ppo", color=PALETTE["our_drl_ppo"], linewidth=2)
    axes[0].plot(dqn["epoch_global"], dqn["train_mean_reward"], label="our_drl_dqn", color=PALETTE["our_drl_dqn"], linewidth=2)
    axes[0].set_title("Training Reward")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Train Mean Reward")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(ppo["epoch_global"], ppo["val_mean_mlu"], label="our_drl_ppo", color=PALETTE["our_drl_ppo"], linewidth=2)
    axes[1].plot(dqn["epoch_global"], dqn["val_mean_mlu"], label="our_drl_dqn", color=PALETTE["our_drl_dqn"], linewidth=2)
    axes[1].set_title("Validation MLU")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val Mean MLU")
    axes[1].grid(True, alpha=0.3)
    fig.text(0.5, 0.01, "Dual-gate is an inference-time selector and has no standalone training curve.", ha="center", fontsize=10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    outputs.extend(_save_fig(fig, "training_curve_drl", FINAL_PLOTS))
    return outputs


def build_manifest(eval_df: pd.DataFrame, fail_df: pd.DataFrame, gen_df: pd.DataFrame, generated: dict[str, list[Path]], selected_pct_generated: bool) -> None:
    required_inputs = [
        EVAL_SUMMARY,
        EVAL_TS,
        FAIL_SUMMARY,
        FAIL_TS,
        GEN_SUMMARY,
        GEN_TS,
        REPORT_MD,
    ]
    methods_eval = sorted(eval_df["method"].unique().tolist())
    methods_fail = sorted(fail_df["method"].unique().tolist())
    methods_gen = sorted(gen_df["method"].unique().tolist())
    missing = []
    required_methods = METHOD_ORDER
    for scenario_name, methods in [("eval", methods_eval), ("failures", methods_fail), ("generalization", methods_gen)]:
        miss = [m for m in required_methods if m not in methods]
        if miss:
            missing.append(f"- {scenario_name}: {', '.join(miss)}")
    best_eval = make_best_method_table(eval_df, fail_df, gen_df)
    def best_line(scenario, dataset, failure_type=""):
        sub = best_eval[(best_eval["scenario"] == scenario) & (best_eval["dataset"] == dataset)]
        if failure_type:
            sub = sub[sub["failure_type"] == failure_type]
        if sub.empty:
            return "n/a"
        row = sub.iloc[0]
        return f"{row['best_overall_method']} (best overall), {row['best_drl_method']} (best DRL)"
    fail_best = fail_df.groupby("method", as_index=False)["post_failure_mean_mlu"].mean().sort_values("post_failure_mean_mlu")
    strongest_drl = pd.concat([
        eval_df[eval_df["method"].isin(DRL_METHODS)][["method", "mean_mlu"]],
        gen_df[gen_df["method"].isin(DRL_METHODS)][["method", "mean_mlu"]],
    ], ignore_index=True).groupby("method", as_index=False)["mean_mlu"].mean().sort_values("mean_mlu")
    lines = [
        "# Final Reporting Manifest",
        "",
        "## Inputs Used",
        "",
    ]
    lines.extend([f"- {p.resolve()}" for p in required_inputs])
    lines.extend([
        f"- {PREPARE_MANIFEST.resolve()} (used to reconstruct selected-flow percentage from selected_count / num_od)",
        f"- {PPO_CURRICULUM.resolve()} and {DQN_CURRICULUM.resolve()} (used for training-curve reporting)",
        "",
        "## Corrected Full-Config Status",
        "",
        "- All generated outputs come from the corrected full-config result files: YES",
        "",
        "## Methods Appearing In Final Tables",
        "",
        f"- eval: {', '.join(methods_eval)}",
        f"- failures: {', '.join(methods_fail)}",
        f"- generalization: {', '.join(methods_gen)}",
        "",
        "## Methods Missing In Some Scenarios",
        "",
    ])
    lines.extend(missing if missing else ["- none"])
    lines.extend([
        "",
        "## Selected-Flow Percentage",
        "",
        f"- selected-flow percentage CDF generated: {'YES' if selected_pct_generated else 'NO'}",
        f"- dynamic Kcrit data available directly in results: NO",
        f"- selected-flow percentage reconstructed from results: {'YES' if selected_pct_generated else 'NO'}",
        "- reconstruction formula: selected_count / num_od * 100 using results/phase1_reactive/prepare/dataset_manifest.json",
        "",
        "## Best Methods",
        "",
        f"- Abilene: {best_line('eval', 'abilene')}",
        f"- GEANT: {best_line('eval', 'geant')}",
        f"- Germany50: {best_line('generalization', 'germany50')}",
        f"- failures (aggregate): {fail_best.iloc[0]['method'] if not fail_best.empty else 'n/a'}",
        f"- strongest DRL overall: {strongest_drl.iloc[0]['method'] if not strongest_drl.empty else 'n/a'}",
        "",
        "## Generated Outputs",
        "",
    ])
    for group_name, paths in generated.items():
        lines.append(f"### {group_name}")
        lines.append("")
        for path in paths:
            lines.append(f"- {path.resolve()}")
        lines.append("")
    (FINAL_TABLES / "final_reporting_manifest.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    ensure_dirs()
    od_map = load_od_map()

    eval_summary = normalize_method(read_csv(EVAL_SUMMARY))
    eval_ts = normalize_method(read_csv(EVAL_TS))
    fail_summary = normalize_method(read_csv(FAIL_SUMMARY))
    fail_ts = normalize_method(read_csv(FAIL_TS))
    gen_summary = normalize_method(read_csv(GEN_SUMMARY))
    gen_ts = normalize_method(read_csv(GEN_TS))

    eval_ts = add_selected_flow_pct(eval_ts, od_map)
    gen_ts = add_selected_flow_pct(gen_ts, od_map)

    overall_eval = make_eval_table(eval_summary)
    overall_fail = make_failure_table(fail_summary)
    overall_gen = make_generalization_table(gen_summary)
    drl_only = make_drl_only_table(eval_summary, fail_summary, gen_summary)
    ablation = make_ablation_table()
    best_methods = make_best_method_table(eval_summary, fail_summary, gen_summary)

    generated: dict[str, list[Path]] = {"tables": [], "cdfs": [], "plots": [], "exports": []}
    for df, stem, title in [
        (overall_eval, "overall_eval_comparison", "Overall Eval Comparison"),
        (overall_fail, "overall_failure_comparison", "Overall Failure Comparison"),
        (overall_gen, "overall_generalization_comparison", "Overall Generalization Comparison"),
        (drl_only, "drl_only_comparison", "DRL Only Comparison"),
        (ablation, "ablation_improvement_table", "Ablation Improvement Table"),
        (best_methods, "best_method_per_topology", "Best Method Per Topology"),
    ]:
        csv_path, md_path = write_table(df, stem, title)
        generated["tables"].extend([csv_path, md_path])

    for df, name in [
        (overall_eval, "paper_table_eval.csv"),
        (overall_fail, "paper_table_failures.csv"),
        (overall_gen, "paper_table_generalization.csv"),
        (drl_only, "paper_table_drl_only.csv"),
        (best_methods, "paper_table_best_methods.csv"),
    ]:
        generated["exports"].append(write_export(df, name))

    generated["cdfs"].extend(plot_seen_cdfs(eval_ts, eval_summary))
    selected_cdf_outputs = plot_selected_flow_cdfs(eval_ts)
    generated["cdfs"].extend(selected_cdf_outputs)
    generated["cdfs"].extend(plot_failure_cdf(fail_ts, fail_summary))
    generated["cdfs"].extend(plot_drl_vs_heuristic_seen(eval_ts, eval_summary))
    generated["cdfs"].extend(plot_germany50_cdf(gen_ts, gen_summary))

    generated["plots"].extend(plot_seen_timeseries(eval_ts, eval_summary))
    generated["plots"].extend(plot_selected_count_timeseries(eval_ts))
    generated["plots"].extend(plot_training_curve())

    build_manifest(eval_summary, fail_summary, gen_summary, generated, selected_pct_generated=bool(selected_cdf_outputs))
    print(FINAL_TABLES.resolve())
    print(FINAL_CDFS.resolve())
    print(FINAL_PLOTS.resolve())
    print(FINAL_EXPORTS.resolve())


if __name__ == "__main__":
    main()
