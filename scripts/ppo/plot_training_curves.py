from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot PPO training curves from metrics.csv.")
    parser.add_argument("--log-file", type=Path, default=Path("logs_ppo/metrics.csv"))
    parser.add_argument("--output", type=Path, default=Path("logs_ppo/training_curves.png"))
    parser.add_argument("--smooth", type=int, default=5, help="Moving average window. Use 1 to disable smoothing.")
    return parser.parse_args()


def load_metrics(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        raise FileNotFoundError(f"Log file not found: {path}")

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        metrics: dict[str, list[float]] = {field: [] for field in reader.fieldnames or []}
        for row in reader:
            for key, value in row.items():
                try:
                    metrics[key].append(float(value))
                except (TypeError, ValueError):
                    metrics[key].append(float("nan"))
    return metrics


def moving_average(values: list[float], window: int) -> list[float]:
    if window <= 1:
        return values

    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = [value for value in values[start : index + 1] if math.isfinite(value)]
        smoothed.append(sum(chunk) / len(chunk) if chunk else float("nan"))
    return smoothed


def plot_group(ax, x_values: list[float], metrics: dict[str, list[float]], names: list[str], smooth: int) -> None:
    for name in names:
        values = metrics.get(name)
        if not values:
            continue
        ax.plot(x_values, moving_average(values, smooth), label=name)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)


def main() -> None:
    args = parse_args()
    metrics = load_metrics(args.log_file)
    if not metrics.get("steps"):
        raise ValueError(f"No metrics found in {args.log_file}")

    x_values = metrics["steps"]
    figure, axes = plt.subplots(4, 2, figsize=(14, 13), constrained_layout=True)
    axes = axes.flatten()

    plot_group(axes[0], x_values, metrics, ["mean_return"], args.smooth)
    axes[0].set_title("Episode Return")
    axes[0].set_xlabel("steps")

    plot_group(axes[1], x_values, metrics, ["success_rate", "collision_rate"], args.smooth)
    axes[1].set_title("Task Rates")
    axes[1].set_xlabel("steps")

    plot_group(
        axes[2],
        x_values,
        metrics,
        ["ground_collision_rate", "self_collision_rate", "obstacle_collision_rate"],
        args.smooth,
    )
    axes[2].set_title("Collision Breakdown")
    axes[2].set_xlabel("steps")

    plot_group(axes[3], x_values, metrics, ["policy_loss", "value_loss"], args.smooth)
    axes[3].set_title("Losses")
    axes[3].set_xlabel("steps")

    plot_group(axes[4], x_values, metrics, ["entropy", "approx_kl", "clip_fraction"], args.smooth)
    axes[4].set_title("PPO Diagnostics")
    axes[4].set_xlabel("steps")

    plot_group(axes[5], x_values, metrics, ["target_distance"], args.smooth)
    axes[5].set_title("Target Distance")
    axes[5].set_xlabel("steps")

    plot_group(axes[6], x_values, metrics, ["min_obstacle_distance"], args.smooth)
    axes[6].set_title("Min Obstacle Distance")
    axes[6].set_xlabel("steps")

    plot_group(axes[7], x_values, metrics, ["mean_episode_len"], args.smooth)
    axes[7].set_title("Mean Episode Length")
    axes[7].set_xlabel("steps")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=160)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
