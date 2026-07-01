from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import DDPGAgent, DDPGConfig
from envs import PandaObstacleEnv


REWARD_FIELDS = ["reward_target", "reward_obstacle", "reward_time"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record reward terms from a DDPG best.pt rollout.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ddpg/best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("logs_ddpg/reward_test"))
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-obstacles", type=int, default=1)
    parser.add_argument("--obstacle-radius", type=float, default=0.04)
    parser.add_argument("--joint-step-deg", type=float, default=5.0)
    parser.add_argument("--render", action="store_true")
    return parser.parse_args()


def write_reward_txt(path: Path, rows: list[dict[str, float]]) -> None:
    fieldnames = [
        "step",
        *REWARD_FIELDS,
        "reward_sum",
        "env_reward",
        "target_distance",
        "min_obstacle_distance",
        "success",
        "collision",
        "timeout",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_rewards(path: Path, rows: list[dict[str, float]]) -> Path:
    if plt is None:
        svg_path = path.with_suffix(".svg")
        write_reward_svg(svg_path, rows)
        return svg_path

    steps = [row["step"] for row in rows]
    plt.figure(figsize=(10, 6))
    for field in REWARD_FIELDS:
        plt.plot(steps, [row[field] for row in rows], label=field, linewidth=1.8)
    plt.plot(steps, [row["reward_sum"] for row in rows], label="reward_sum", linewidth=2.2, color="black")
    plt.xlabel("step")
    plt.ylabel("reward")
    plt.title("Reward terms over rollout steps")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return path


def write_reward_svg(path: Path, rows: list[dict[str, float]]) -> None:
    width = 1000
    height = 600
    left = 70
    right = 170
    top = 40
    bottom = 60
    plot_width = width - left - right
    plot_height = height - top - bottom
    series = [*REWARD_FIELDS, "reward_sum"]
    colors = {
        "reward_target": "#2563eb",
        "reward_obstacle": "#16a34a",
        "reward_time": "#dc2626",
        "reward_sum": "#111827",
    }
    steps = [row["step"] for row in rows]
    values = [row[field] for row in rows for field in series]
    min_step = min(steps)
    max_step = max(steps)
    min_value = min(values)
    max_value = max(values)
    if min_step == max_step:
        max_step += 1
    if min_value == max_value:
        min_value -= 1.0
        max_value += 1.0

    def x_pos(step: float) -> float:
        return left + (step - min_step) / (max_step - min_step) * plot_width

    def y_pos(value: float) -> float:
        return top + (max_value - value) / (max_value - min_value) * plot_height

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{left}" y="25" font-family="sans-serif" font-size="18">Reward terms over rollout steps</text>',
        f'<line x1="{left}" y1="{height - bottom}" x2="{width - right}" y2="{height - bottom}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height - bottom}" stroke="#374151"/>',
        f'<text x="{width / 2}" y="{height - 18}" text-anchor="middle" font-family="sans-serif" font-size="13">step</text>',
        f'<text x="18" y="{height / 2}" transform="rotate(-90 18 {height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13">reward</text>',
    ]
    for frac in np.linspace(0.0, 1.0, 6):
        y = top + frac * plot_height
        value = max_value - frac * (max_value - min_value)
        lines.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        lines.append(
            f'<text x="{left - 8}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" '
            f'font-size="11" fill="#4b5563">{value:.3f}</text>'
        )
    for index, field in enumerate(series):
        points = " ".join(f'{x_pos(row["step"]):.2f},{y_pos(row[field]):.2f}' for row in rows)
        stroke_width = 2.4 if field == "reward_sum" else 1.8
        lines.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[field]}" stroke-width="{stroke_width}"/>'
        )
        legend_y = top + 20 + index * 24
        lines.append(
            f'<line x1="{width - right + 25}" y1="{legend_y}" x2="{width - right + 55}" '
            f'y2="{legend_y}" stroke="{colors[field]}" stroke-width="{stroke_width}"/>'
        )
        lines.append(
            f'<text x="{width - right + 62}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{field}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    env = PandaObstacleEnv(
        num_obstacles=args.num_obstacles,
        obstacle_radius=args.obstacle_radius,
        randomize_reset=False,
        max_joint_step=np.deg2rad(args.joint_step_deg),
    )
    agent = DDPGAgent(
        env.obs_dim,
        env.action_dim,
        DDPGConfig(),
        device=args.device,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    agent.load(str(args.checkpoint))

    rows: list[dict[str, float]] = []
    obs = env.reset()
    done = False
    step = 0
    episode_return = 0.0
    try:
        while not done:
            action = agent.act(obs, explore=False)
            obs, reward, done, info = env.step(action)
            step += 1
            episode_return += reward
            reward_sum = sum(float(info[field]) for field in REWARD_FIELDS)
            rows.append(
                {
                    "step": step,
                    "reward_target": float(info["reward_target"]),
                    "reward_obstacle": float(info["reward_obstacle"]),
                    "reward_time": float(info["reward_time"]),
                    "reward_sum": reward_sum,
                    "env_reward": float(reward),
                    "target_distance": float(info["target_distance"]),
                    "min_obstacle_distance": float(info["min_obstacle_distance"]),
                    "success": float(info["success"]),
                    "collision": float(info["collision"]),
                    "timeout": float(info["timeout"]),
                }
            )
            if args.render:
                env.render()
    finally:
        env.close()

    txt_path = args.output_dir / "reward_terms_best.txt"
    plot_path = plot_rewards(args.output_dir / "reward_terms_best.png", rows)
    write_reward_txt(txt_path, rows)

    last = rows[-1]
    print(
        f"steps={step} return={episode_return:.3f} success={bool(last['success'])} "
        f"collision={bool(last['collision'])} timeout={bool(last['timeout'])} "
        f"txt={txt_path} plot={plot_path}"
    )


if __name__ == "__main__":
    main()
