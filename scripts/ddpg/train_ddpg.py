from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

try:
    from tqdm import trange
except ImportError:
    def trange(start, stop, step=1, desc=None, dynamic_ncols=None):
        return range(start, stop, step)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import DDPGAgent, DDPGConfig, ReplayBuffer
from envs import PandaObstacleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DDPG for six-joint Panda reaching and obstacle avoidance.")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_ddpg"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs_ddpg"))
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-hist-every", type=int, default=5_000)
    parser.add_argument("--log-every", type=int, default=2_048)
    parser.add_argument("--save-every", type=int, default=25_000)
    parser.add_argument("--num-obstacles", type=int, default=1)
    parser.add_argument("--obstacle-radius", type=float, default=0.04)
    parser.add_argument("--randomize-reset", action="store_true")
    parser.add_argument("--joint-step-deg", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--warmup-steps", type=int, default=5_000)
    parser.add_argument("--replay-size", type=int, default=10_000)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=1e-3)
    parser.add_argument("--exploration-noise", type=float, default=0.25)
    parser.add_argument("--min-noise", type=float, default=0.03)
    parser.add_argument("--noise-decay", type=float, default=0.99995)
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> PandaObstacleEnv:
    return PandaObstacleEnv(
        num_obstacles=args.num_obstacles,
        obstacle_radius=args.obstacle_radius,
        randomize_reset=args.randomize_reset,
        max_joint_step=np.deg2rad(args.joint_step_deg),
        seed=args.seed,
    )


def make_config(args: argparse.Namespace) -> DDPGConfig:
    return DDPGConfig(
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        batch_size=args.batch_size,
        replay_size=args.replay_size,
        warmup_steps=args.warmup_steps,
        exploration_noise=args.exploration_noise,
        min_noise=args.min_noise,
        noise_decay=args.noise_decay,
    )


def make_tensorboard_writer(args: argparse.Namespace):
    if args.no_tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard is not available. Install it with: python3 -m pip install tensorboard")
        return None
    tensorboard_dir = args.tensorboard_dir if args.tensorboard_dir is not None else args.log_dir / "tensorboard"
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tensorboard_dir))


def write_tensorboard(
    writer,
    agent: DDPGAgent,
    log: dict[str, float],
    action_batch: list[np.ndarray],
    step: int,
    hist_every: int,
) -> None:
    if writer is None:
        return

    scalar_groups = {
        "rollout": [
            "mean_return",
            "mean_episode_len",
            "success_rate",
            "episode_collision_rate",
            "collision_rate",
            "ground_collision_rate",
            "self_collision_rate",
            "obstacle_collision_rate",
            "target_distance",
            "min_obstacle_distance",
        ],
        "ddpg": ["actor_loss", "critic_loss", "noise_std", "replay_size"],
    }
    for group, fields in scalar_groups.items():
        for field in fields:
            value = log.get(field)
            if value is not None and np.isfinite(value):
                writer.add_scalar(f"{group}/{field}", float(value), step)

    if action_batch:
        actions = np.asarray(action_batch, dtype=np.float32)
        for index in range(actions.shape[1]):
            writer.add_scalar(f"actions/mean_{index:02d}", float(np.mean(actions[:, index])), step)
            writer.add_scalar(f"actions/std_{index:02d}", float(np.std(actions[:, index])), step)
            writer.add_scalar(f"actions/abs_mean_{index:02d}", float(np.mean(np.abs(actions[:, index]))), step)

    if hist_every > 0 and step % hist_every == 0:
        for module_name, module in (("actor", agent.actor), ("critic", agent.critic)):
            for name, parameter in module.named_parameters():
                writer.add_histogram(f"parameters/{module_name}.{name}", parameter.detach().cpu(), step)
                if parameter.grad is not None:
                    writer.add_histogram(f"gradients/{module_name}.{name}", parameter.grad.detach().cpu(), step)
    writer.flush()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args)
    cfg = make_config(args)
    agent = DDPGAgent(
        env.obs_dim,
        env.action_dim,
        cfg,
        device=args.device,
        action_low=env.action_space.low,
        action_high=env.action_space.high,
    )
    replay = ReplayBuffer(cfg.replay_size)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.log_dir / "metrics.csv"
    jsonl_path = args.log_dir / "metrics.jsonl"
    log_fields = [
        "steps",
        "episodes",
        "mean_return",
        "mean_episode_len",
        "success_rate",
        "episode_collision_rate",
        "collision_rate",
        "ground_collision_rate",
        "self_collision_rate",
        "obstacle_collision_rate",
        "target_distance",
        "min_obstacle_distance",
        "actor_loss",
        "critic_loss",
        "noise_std",
        "replay_size",
    ]
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    jsonl_file = jsonl_path.open("w", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=log_fields)
    writer.writeheader()

    tb_writer = make_tensorboard_writer(args)
    if tb_writer is not None:
        tb_writer.add_text("config/args", json.dumps(vars(args), default=str, indent=2), 0)
        tb_writer.add_text("config/ddpg", json.dumps(cfg.__dict__, indent=2), 0)

    obs = env.reset()
    episode_return = 0.0
    episode_len = 0
    completed_episodes = 0
    best_return = -float("inf")
    last_info = {
        "success": False,
        "collision": False,
        "target_distance": float("nan"),
        "min_obstacle_distance": float("nan"),
    }

    window_returns: list[float] = []
    window_lengths: list[int] = []
    window_successes: list[float] = []
    window_episode_collisions: list[float] = []
    window_step_collisions: list[float] = []
    window_ground_collisions: list[float] = []
    window_self_collisions: list[float] = []
    window_obstacle_collisions: list[float] = []
    window_actions: list[np.ndarray] = []
    metrics = {"actor_loss": 0.0, "critic_loss": 0.0, "noise_std": agent.noise_std}
    rng = np.random.default_rng(args.seed)

    try:
        progress = trange(1, args.total_steps + 1, desc="six-joint-ddpg", dynamic_ncols=True)
        for step in progress:
            if step <= cfg.warmup_steps:
                action = env.action_space.sample(rng)
            else:
                action = agent.act(obs, explore=True)

            next_obs, reward, done, info = env.step(action)
            replay.add(obs, action, reward, next_obs, done)
            metrics = agent.update(replay)

            obs = next_obs
            episode_return += reward
            episode_len += 1
            last_info = info
            window_actions.append(action.copy())
            window_step_collisions.append(float(info["collision"]))
            window_ground_collisions.append(float(info.get("ground_collision", False)))
            window_self_collisions.append(float(info.get("self_collision", False)))
            window_obstacle_collisions.append(float(info.get("obstacle_collision", False)))

            if done:
                window_returns.append(episode_return)
                window_lengths.append(episode_len)
                window_successes.append(float(info["success"]))
                window_episode_collisions.append(float(info["collision"]))
                completed_episodes += 1
                obs = env.reset()
                episode_return = 0.0
                episode_len = 0

            if step % args.log_every == 0 or step == args.total_steps:
                mean_return = float(np.mean(window_returns)) if window_returns else float("nan")
                mean_episode_len = float(np.mean(window_lengths)) if window_lengths else float("nan")
                mean_success = float(np.mean(window_successes)) if window_successes else 0.0
                mean_episode_collision = float(np.mean(window_episode_collisions)) if window_episode_collisions else 0.0
                mean_collision = float(np.mean(window_step_collisions)) if window_step_collisions else 0.0
                mean_ground_collision = float(np.mean(window_ground_collisions)) if window_ground_collisions else 0.0
                mean_self_collision = float(np.mean(window_self_collisions)) if window_self_collisions else 0.0
                mean_obstacle_collision = float(np.mean(window_obstacle_collisions)) if window_obstacle_collisions else 0.0

                log = {
                    "steps": step,
                    "episodes": completed_episodes,
                    "mean_return": mean_return,
                    "mean_episode_len": mean_episode_len,
                    "success_rate": mean_success,
                    "episode_collision_rate": mean_episode_collision,
                    "collision_rate": mean_collision,
                    "ground_collision_rate": mean_ground_collision,
                    "self_collision_rate": mean_self_collision,
                    "obstacle_collision_rate": mean_obstacle_collision,
                    "target_distance": float(last_info["target_distance"]),
                    "min_obstacle_distance": float(last_info["min_obstacle_distance"]),
                    "actor_loss": float(metrics["actor_loss"]),
                    "critic_loss": float(metrics["critic_loss"]),
                    "noise_std": float(metrics["noise_std"]),
                    "replay_size": len(replay),
                }
                writer.writerow(log)
                csv_file.flush()
                jsonl_file.write(json.dumps(log, ensure_ascii=False) + "\n")
                jsonl_file.flush()
                write_tensorboard(tb_writer, agent, log, window_actions, step, args.tensorboard_hist_every)
                print(json.dumps(log, ensure_ascii=False))

                if hasattr(progress, "set_postfix"):
                    progress.set_postfix(
                        ret=f"{mean_return:.2f}",
                        succ=f"{mean_success:.2f}",
                        coll=f"{mean_episode_collision:.2f}",
                        noise=f"{agent.noise_std:.3f}",
                    )

                if window_returns and mean_return > best_return:
                    best_return = mean_return
                    agent.save(str(args.checkpoint_dir / "best.pt"))

                window_returns.clear()
                window_lengths.clear()
                window_successes.clear()
                window_episode_collisions.clear()
                window_step_collisions.clear()
                window_ground_collisions.clear()
                window_self_collisions.clear()
                window_obstacle_collisions.clear()
                window_actions.clear()

            if args.save_every > 0 and step % args.save_every == 0:
                agent.save(str(args.checkpoint_dir / "last.pt"))
    finally:
        csv_file.close()
        jsonl_file.close()
        if tb_writer is not None:
            tb_writer.close()
        env.close()

    agent.save(str(args.checkpoint_dir / "last.pt"))


if __name__ == "__main__":
    main()
