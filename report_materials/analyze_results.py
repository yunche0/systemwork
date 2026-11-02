from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

SRC = Path("report_materials/results.csv")
OUT = Path("report_materials")
FIG = OUT / "figures"
FIG.mkdir(exist_ok=True)

df = pd.read_csv(SRC)

# 数值列转换；空字段保留为缺失值，不当作 0。
numeric_cols = [
    "repeat", "selectivity_min", "selectivity_median", "selectivity_max",
    "L", "k", "threads", "beamwidth", "index_bytes", "elapsed_seconds",
    "peak_rss_kb", "qps", "mean_latency_us", "recall_percent", "mean_ios",
    "pre_filter_queries", "in_filter_queries", "post_filter_queries",
    "filter_false_positives", "filter_skips", "io_time_us",
    "tunnel_time_us", "process_time_us"
]
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# 每个 system + bucket + L 必须恰好有 3 次重复。
group_cols = ["system", "bucket", "L"]
counts = df.groupby(group_cols).size().reset_index(name="n_repeats")
assert (counts["n_repeats"] == 3).all(), "存在不是 3 次重复的实验组，请检查 results.csv。"

metrics = [
    "selectivity_median", "recall_percent", "mean_latency_us",
    "qps", "mean_ios", "peak_rss_kb", "index_bytes",
    "pre_filter_queries", "in_filter_queries", "post_filter_queries",
    "filter_false_positives", "filter_skips",
    "io_time_us", "tunnel_time_us", "process_time_us"
]

summary = df.groupby(group_cols)[metrics].agg(["median", "min", "max"])
summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
summary = summary.reset_index()
summary.to_csv(OUT / "summary_median_range.csv", index=False)

# 表 1：High workload 的 Recall–Latency 数据。
high = summary[summary["bucket"] == "high"].copy()
high.to_csv(OUT / "high_recall_latency_table.csv", index=False)

# 图 1：High workload Recall@10–Mean Latency。
colors = {"pipeann": "#1f77b4", "gateann": "#d62728"}
labels = {"pipeann": "PipeANN-Filter", "gateann": "GateANN"}

plt.figure(figsize=(7.2, 5.2))
for system in ["pipeann", "gateann"]:
    part = high[high["system"] == system].sort_values("L")
    x = part["mean_latency_us_median"] / 1000
    y = part["recall_percent_median"]
    xerr = [
        (part["mean_latency_us_median"] - part["mean_latency_us_min"]) / 1000,
        (part["mean_latency_us_max"] - part["mean_latency_us_median"]) / 1000,
    ]
    yerr = [
        part["recall_percent_median"] - part["recall_percent_min"],
        part["recall_percent_max"] - part["recall_percent_median"],
    ]
    plt.errorbar(
        x, y, xerr=xerr, yerr=yerr, marker="o", linewidth=2,
        capsize=3, color=colors[system], label=labels[system]
    )
    for _, row in part.iterrows():
        plt.annotate(
            f"L={int(row['L'])}",
            (row["mean_latency_us_median"] / 1000, row["recall_percent_median"]),
            xytext=(5, 5), textcoords="offset points", fontsize=9
        )

plt.xlabel("Mean query latency (ms, median of 3 runs)")
plt.ylabel("Recall@10 (%)")
plt.title("High workload: Recall–Latency")
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig(FIG / "fig1_recall_latency_high.png", dpi=220)
plt.close()

# 固定代表性预算的 Selectivity 图数据。
representative = pd.concat([
    summary[(summary["system"] == "pipeann") & (summary["L"] == 40)],
    summary[(summary["system"] == "gateann") & (summary["L"] == 320)],
]).copy()

bucket_order = {"low": 0, "medium": 1, "high": 2}
representative["bucket_order"] = representative["bucket"].map(bucket_order)
representative = representative.sort_values(["system", "bucket_order"])
representative.to_csv(OUT / "selectivity_performance_table.csv", index=False)

# 图 2：Recall、延迟、I/O 三联图。
fig, axes = plt.subplots(1, 3, figsize=(14, 4.3), sharex=True)
plot_specs = [
    ("recall_percent", "Recall@10 (%)"),
    ("mean_latency_us", "Mean latency (ms)"),
    ("mean_ios", "Mean logical SSD I/O / query"),
]

for ax, (metric, ylabel) in zip(axes, plot_specs):
    for system in ["pipeann", "gateann"]:
        part = representative[representative["system"] == system].sort_values("bucket_order")
        x = part["selectivity_median_median"] * 100
        y = part[f"{metric}_median"]
        if metric == "mean_latency_us":
            y = y / 1000
        ax.plot(x, y, marker="o", linewidth=2, color=colors[system], label=labels[system])

    ax.set_xlabel("Selectivity median (%)")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)

axes[0].legend()
fig.suptitle("Representative budgets: PipeANN L=40, GateANN L=320", y=1.03)
fig.tight_layout()
fig.savefig(FIG / "fig2_selectivity_performance.png", dpi=220, bbox_inches="tight")
plt.close(fig)

# 图 3：系统内部行为，供报告机制分析使用。
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
pipe = representative[representative["system"] == "pipeann"].sort_values("bucket_order")
gate = representative[representative["system"] == "gateann"].sort_values("bucket_order")
x_labels = ["Low", "Medium", "High"]

axes[0].plot(x_labels, pipe["pre_filter_queries_median"], marker="o", label="Pre-filter")
axes[0].plot(x_labels, pipe["in_filter_queries_median"], marker="o", label="In-filter")
axes[0].plot(x_labels, pipe["post_filter_queries_median"], marker="o", label="Post-filter")
axes[0].set_title("PipeANN routing (queries / 200)")
axes[0].set_ylabel("Queries")
axes[0].grid(alpha=0.25)
axes[0].legend()

axes[1].plot(x_labels, gate["filter_skips_median"], marker="o", color=colors["gateann"])
axes[1].set_title("GateANN FilterSkips")
axes[1].set_ylabel("Skips per query")
axes[1].grid(alpha=0.25)

fig.tight_layout()
fig.savefig(FIG / "fig3_internal_behavior.png", dpi=220)
plt.close(fig)

# 资源背景表：索引大小以 GiB、峰值 RSS 以 MiB 表示。
resource = representative[[
    "system", "index_bytes_median", "peak_rss_kb_median",
    "peak_rss_kb_min", "peak_rss_kb_max"
]].drop_duplicates("system").copy()
resource["index_size_GiB"] = resource["index_bytes_median"] / 1024**3
resource["peak_rss_MiB"] = resource["peak_rss_kb_median"] / 1024
resource["peak_rss_min_MiB"] = resource["peak_rss_kb_min"] / 1024
resource["peak_rss_max_MiB"] = resource["peak_rss_kb_max"] / 1024
resource.to_csv(OUT / "resource_tradeoff_table.csv", index=False)

print("分析完成，已生成：")
for path in sorted(OUT.iterdir()):
    print(" -", path)
for path in sorted(FIG.iterdir()):
    print(" -", path)
