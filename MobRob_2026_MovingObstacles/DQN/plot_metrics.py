"""Plot training reward and collision curves, aggregated per algorithm.

Scans a `log/` directory of run folders (as produced by DDQN.loop / PPO_SB3.loop),
groups them by algorithm (all DDQN runs together, all PPO_SB3 runs together, ...),
and for each group plots the mean Avg_Reward / Avg_Collision across runs with a
semi-transparent +-1 std band showing the spread between runs.

Usage:
    python plot_metrics.py [--log_dir log] [--output log/training_metrics.png]
"""
import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def group_key(run_dir_name):
    """Groups by algorithm + tag, matching
    f"{args.alg}__{tag}__{args.seed}__{timestamp}" from DDQN/PPO_SB3/SAC_SB3.loop.
    Untagged runs (tag='') group by algorithm alone, same as before -- so old runs
    stay comparable. Give a sweep of seeds a shared --tag (e.g. "ppo-continuous-v1")
    to keep it grouped as one band, separate from unrelated runs of the same
    algorithm (e.g. an earlier discrete-action run)."""
    parts = run_dir_name.split("__")
    alg = parts[0]
    tag = parts[1] if len(parts) > 1 else ""
    return f"{alg}_{tag}" if tag else alg


def load_runs(log_dir):
    """Returns {alg_name: [DataFrame, ...]} for every run folder found under log_dir."""
    groups = {}
    for run_dir in sorted(os.listdir(log_dir)):
        metrics_path = os.path.join(log_dir, run_dir, "metric", "metrics.csv")
        if not os.path.isfile(metrics_path):
            continue
        df = pd.read_csv(metrics_path)
        groups.setdefault(group_key(run_dir), []).append(df)
    return groups


def plot_metric(ax, groups, column, colors, ylabel, title):
    plotted_any = False
    for alg, runs in groups.items():
        series = [df[column].to_numpy() for df in runs if column in df.columns]
        skipped = len(runs) - len(series)
        if not series:
            print(f"  [{title}] {alg}: no '{column}' data in any of its {len(runs)} run(s), skipping")
            continue
        if skipped:
            print(f"  [{title}] {alg}: {skipped}/{len(runs)} run(s) missing '{column}', excluded from the band")

        min_len = min(len(s) for s in series)
        stacked = np.stack([s[:min_len] for s in series])
        mean = stacked.mean(axis=0)
        std = stacked.std(axis=0)
        episodes = np.arange(min_len)

        color = colors[alg]
        ax.plot(episodes, mean, color=color, label=f"{alg} (n={len(series)})")
        ax.fill_between(episodes, mean - std, mean + std, color=color, alpha=0.2)
        plotted_any = True

    ax.set_xlabel("Episode")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if plotted_any:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="log", help="Directory of run folders")
    parser.add_argument("--output", type=str, default=None, help="Path to save the figure (default: <log_dir>/training_metrics.png)")
    args = parser.parse_args()

    output = args.output or os.path.join(args.log_dir, "training_metrics.png")

    groups = load_runs(args.log_dir)
    if not groups:
        raise SystemExit(f"No run folders with metric/metrics.csv found under '{args.log_dir}'")

    colors = dict(zip(groups.keys(), plt.rcParams["axes.prop_cycle"].by_key()["color"]))

    fig, (ax_reward, ax_collision) = plt.subplots(1, 2, figsize=(14, 5))

    print("Reward:")
    plot_metric(ax_reward, groups, "Avg_Reward", colors,
                ylabel="Avg reward (last-100 rolling mean)", title="Reward")

    print("Collisions:")
    plot_metric(ax_collision, groups, "Avg_Collision", colors,
                ylabel="Avg collision rate % (last-100 rolling mean)", title="Collisions")

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    print(f"Saved to {output}")
    plt.show()


if __name__ == "__main__":
    main()
