#!/usr/bin/env python3
"""Reproduce the public ATM-Hard price/performance chart with GraphMemix."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-graphmemix-atm-hard")

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "release/atm_hard_price_performance.json"
DEFAULT_RESULTS = ROOT / "release/benchmark_results.json"
DEFAULT_OUTPUT = ROOT / "assets/atm_hard_price_performance.png"


def nested_value(document: dict, dotted_key: str) -> float:
    value = document
    for part in dotted_key.split("."):
        value = value[part]
    return float(value)


def priced_cost(method: dict) -> float:
    prices = method["pricing_snapshot"]
    graph = method["graph_builder"]
    reader = method["reader"]
    deepseek = prices["deepseek_v4_flash"]
    sol = prices["gpt_5_6_sol"]
    graph_cost = (
        graph["uncached_input_tokens"] * deepseek["input"]
        + graph["cache_read_input_tokens"] * deepseek["cache_read"]
        + graph["output_tokens"] * deepseek["output"]
    ) / 1_000_000
    uncached_reader = reader["input_tokens"] - reader["cache_read_input_tokens"]
    reader_cost = (
        uncached_reader * sol["input"]
        + reader["cache_read_input_tokens"] * sol["cache_read"]
        + reader["output_tokens"] * sol["output"]
    ) / 1_000_000
    expected = float(method["method_inference"]["api_list_price_equivalent_usd"])
    calculated = graph_cost + reader_cost
    if abs(calculated - expected) > 1e-9:
        raise ValueError(f"cost ledger mismatch: calculated={calculated}, recorded={expected}")
    return calculated


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    data = json.loads(args.data.read_text(encoding="utf-8"))
    results = json.loads(args.results.read_text(encoding="utf-8"))
    official = data["official_snapshot"]
    method = data["graphmemix_forest_v3"]
    graphmemix_cost = priced_cost(method)
    graphmemix_qs = nested_value(results, method["score_key"])

    plt.rcParams.update({"font.size": 11, "axes.titleweight": "bold"})
    fig, ax = plt.subplots(figsize=(13.2, 8.0))
    ax.set_xscale("log")
    ax.set_xlim(0.2, 52)
    ax.set_ylim(25, 64)
    ax.grid(True, which="both", color="#dbe3ec", linewidth=0.8, alpha=0.85)
    ax.set_axisbelow(True)

    for series in official["series"]:
        xs = [point[0] for point in series["points"]]
        ys = [point[1] for point in series["points"]]
        ax.plot(
            xs, ys, "o-", color=series["color"], linewidth=2.2,
            markersize=7.5, alpha=0.8, label=series["label"],
        )

    singles = official["single_points"]
    ax.scatter(
        [point["cost_usd"] for point in singles],
        [point["qs_percent"] for point in singles],
        marker="D", s=72, color="#657384", edgecolor="#384656",
        label="Official single configurations",
    )
    for point in singles:
        if point["label"] in {"DeepSeek V4 Flash 0731", "GPT-5.5 (xhigh)"}:
            dx = -7 if point["label"] == "GPT-5.5 (xhigh)" else 7
            ax.annotate(
                point["label"], (point["cost_usd"], point["qs_percent"]),
                xytext=(dx, 7), textcoords="offset points", fontsize=9,
                ha="right" if dx < 0 else "left", color="#44515e",
            )

    ax.scatter(
        [graphmemix_cost], [graphmemix_qs], marker="*", s=330,
        color="#0f8b5f", edgecolor="#07583d", linewidth=1.2, zorder=8,
        label="GraphMemix forest v3",
    )
    ax.annotate(
        f"GraphMemix forest v3\n${graphmemix_cost:.2f} · {graphmemix_qs:.2f} QS",
        (graphmemix_cost, graphmemix_qs), xytext=(14, -7),
        textcoords="offset points", fontsize=11, fontweight="bold",
        color="#086c49", va="center",
    )

    ax.set_title("ATM-Bench-Hard: Official Price–Performance Runs + GraphMemix", fontsize=18, pad=14)
    ax.set_xlabel("API list-price equivalent cost for 31 questions (USD, log scale)", fontsize=13)
    ax.set_ylabel("GPT-5-mini QS (%)", fontsize=13)
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.95, fontsize=9.5)
    fig.tight_layout(rect=(0.02, 0.02, 0.99, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=190, bbox_inches="tight", facecolor="white")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(json.dumps({"output": str(args.output), "cost_usd": graphmemix_cost, "qs": graphmemix_qs}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
