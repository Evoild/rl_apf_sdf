from __future__ import annotations

import argparse
import csv
import json
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
    def trange(start, stop, step, desc=None, dynamic_ncols=None):
        return range(start, stop, step)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import PPOAgent, PPOConfig, RolloutBuffer
from envs import PandaObstacleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PPO for six-joint Panda reaching and obstacle avoidance.")
    parser.add_argument("--total-steps", type=int, default=300_000)
    parser.add_argument("--rollout-steps", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints_ppo"))
    parser.add_argument("--log-dir", type=Path, default=Path("logs_ppo"))
    parser.add_argument("--tensorboard-dir", type=Path, default=None)
    parser.add_argument("--no-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-hist-every", type=int, default=10)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--num-obstacles", type=int, default=1)
    parser.add_argument("--obstacle-radius", type=float, default=0.04)
    parser.add_argument("--randomize-reset", action="store_true")
    return parser.parse_args()


def make_env(args: argparse.Namespace) -> PandaObstacleEnv:
    return PandaObstacleEnv(
        num_obstacles=args.num_obstacles,
        obstacle_radius=args.obstacle_radius,
        randomize_reset=args.randomize_reset,
        seed=args.seed,
    )


def make_tensorboard_writer(args: argparse.Namespace):
    if args.no_tensorboard:
        return None
    if SummaryWriter is None:
        print("TensorBoard is not available. Install it with: python3 -m pip install tensorboard")
        return None

    tensorboard_root = args.tensorboard_dir if args.tensorboard_dir is not None else args.log_dir / "tensorboard"
    run_name = (
        f"total-steps_{args.total_steps}"
        f"_rollout-steps_{args.rollout_steps}"
        f"_seed_{args.seed}"
    )
    tensorboard_dir = tensorboard_root / run_name
    run_index = 2
    while tensorboard_dir.exists() and any(tensorboard_dir.iterdir()):
        tensorboard_dir = tensorboard_root / f"{run_name}_run-{run_index:03d}"
        run_index += 1
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    print(f"TensorBoard log_dir: {tensorboard_dir}")
    return SummaryWriter(log_dir=str(tensorboard_dir))


def write_tensorboard(
    writer,
    agent: PPOAgent,
    log: dict[str, float],
    action_batch: list[np.ndarray],
    obs: np.ndarray,
    update_index: int,
    hist_every: int,
) -> None:
    if writer is None:
        return

    step = int(log["steps"])
    scalar_groups = {
        "rollout": [
            "mean_return",
            "mean_episode_len",
            "success_rate",
            "collision_rate",
            "ground_collision_rate",
            "self_collision_rate",
            "obstacle_collision_rate",
            "target_distance",
            "min_obstacle_distance",
        ],
        "ppo": ["policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"],
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

    with torch.no_grad():
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).unsqueeze(0)
        mean = agent.model.action_mean(obs_tensor).squeeze(0).detach().cpu().numpy()
        std = torch.exp(agent.model.log_std).detach().cpu().numpy()
    for index, value in enumerate(mean):
        writer.add_scalar(f"policy/mean_{index:02d}", float(value), step)
    for index, value in enumerate(std):
        writer.add_scalar(f"policy/std_{index:02d}", float(value), step)

    if hist_every > 0 and update_index % hist_every == 0:
        for name, parameter in agent.model.named_parameters():
            writer.add_histogram(f"parameters/{name}", parameter.detach().cpu(), step)
            if parameter.grad is not None:
                writer.add_histogram(f"gradients/{name}", parameter.grad.detach().cpu(), step)
    writer.flush()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    env = make_env(args)
    cfg = PPOConfig(policy_lr=1e-4, value_lr=3e-4, max_grad_norm=0.3)
    agent = PPOAgent(
        env.obs_dim,
        env.action_dim,
        cfg,
        device=args.device,
    )
    buffer = RolloutBuffer()
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.log_dir / "metrics.csv"
    jsonl_path = args.log_dir / "metrics.jsonl"
    log_fields = [
        "update",
        "steps",
        "episodes",
        "mean_return",
        "mean_episode_len",
        "success_rate",
        "collision_rate",
        "ground_collision_rate",
        "self_collision_rate",
        "obstacle_collision_rate",
        "target_distance",
        "min_obstacle_distance",
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_fraction",
    ]
    csv_file = csv_path.open("w", newline="", encoding="utf-8")
    jsonl_file = jsonl_path.open("w", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=log_fields)
    writer.writeheader()
    tb_writer = make_tensorboard_writer(args)
    if tb_writer is not None:
        tb_writer.add_text("config/args", json.dumps(vars(args), default=str, indent=2), 0)
        tb_writer.add_text("config/ppo", json.dumps(cfg.__dict__, indent=2), 0)

    obs = env.reset()
    episode_return = 0.0
    episode_len = 0
    completed_episodes = 0
    best_success_rate = -float("inf")
    best_return = -float("inf")
    best_success_return = -float("inf")

    try:
        progress = trange(0, args.total_steps, args.rollout_steps, desc="six-joint-ppo", dynamic_ncols=True)
        for global_step in progress:
            rollout_returns: list[float] = []
            rollout_lengths: list[int] = []
            rollout_successes: list[float] = []
            rollout_collisions: list[float] = []
            rollout_ground_collisions: list[float] = []
            rollout_self_collisions: list[float] = []
            rollout_obstacle_collisions: list[float] = []
            rollout_actions: list[np.ndarray] = []

            for _ in range(args.rollout_steps):
                action, log_prob, value = agent.act(obs)
                next_obs, reward, done, info = env.step_joint_position_action(action)
                buffer.add(obs, action, log_prob, reward, done, value)
                obs = next_obs
                episode_return += reward
                episode_len += 1
                rollout_actions.append(action.copy())
                rollout_collisions.append(float(info["collision"]))
                rollout_ground_collisions.append(float(info.get("ground_collision", False)))
                rollout_self_collisions.append(float(info.get("self_collision", False)))
                rollout_obstacle_collisions.append(float(info.get("obstacle_collision", False)))

                if done:
                    rollout_returns.append(episode_return)
                    rollout_lengths.append(episode_len)
                    rollout_successes.append(float(info["success"]))
                    completed_episodes += 1
                    obs = env.reset()
                    episode_return = 0.0
                    episode_len = 0

            _, _, last_value = agent.act(obs, deterministic=True)
            metrics = agent.update(buffer, last_value)
            buffer.clear()

            mean_return = float(np.mean(rollout_returns)) if rollout_returns else float("nan")
            mean_episode_len = float(np.mean(rollout_lengths)) if rollout_lengths else float("nan")
            mean_success = float(np.mean(rollout_successes)) if rollout_successes else 0.0
            mean_collision = float(np.mean(rollout_collisions)) if rollout_collisions else 0.0
            mean_ground_collision = float(np.mean(rollout_ground_collisions)) if rollout_ground_collisions else 0.0
            mean_self_collision = float(np.mean(rollout_self_collisions)) if rollout_self_collisions else 0.0
            mean_obstacle_collision = float(np.mean(rollout_obstacle_collisions)) if rollout_obstacle_collisions else 0.0
            if hasattr(progress, "set_postfix"):
                progress.set_postfix(
                    ret=f"{mean_return:.2f}",
                    succ=f"{mean_success:.2f}",
                    coll=f"{mean_collision:.2f}",
                    dist=f"{info['target_distance']:.3f}",
                )

            update_index = global_step // args.rollout_steps
            log = {
                "update": update_index,
                "steps": global_step + args.rollout_steps,
                "episodes": completed_episodes,
                "mean_return": mean_return,
                "mean_episode_len": mean_episode_len,
                "success_rate": mean_success,
                "collision_rate": mean_collision,
                "ground_collision_rate": mean_ground_collision,
                "self_collision_rate": mean_self_collision,
                "obstacle_collision_rate": mean_obstacle_collision,
                "target_distance": float(info["target_distance"]),
                "min_obstacle_distance": float(info["min_obstacle_distance"]),
                **metrics,
            }
            writer.writerow(log)
            csv_file.flush()
            jsonl_file.write(json.dumps(log, ensure_ascii=False) + "\n")
            jsonl_file.flush()
            write_tensorboard(
                tb_writer,
                agent,
                log,
                rollout_actions,
                obs,
                update_index,
                args.tensorboard_hist_every,
            )

            if update_index % args.log_every == 0:
                print(json.dumps(log, ensure_ascii=False))

            if rollout_returns and (
                mean_success > best_success_rate
                or (mean_success == best_success_rate and mean_return > best_success_return)
            ):
                best_success_rate = mean_success
                best_success_return = mean_return
                agent.save(str(args.checkpoint_dir / "best_success.pt"))

            if rollout_returns and mean_return > best_return:
                best_return = mean_return
                agent.save(str(args.checkpoint_dir / "best_return.pt"))
    finally:
        csv_file.close()
        jsonl_file.close()
        if tb_writer is not None:
            tb_writer.close()
        env.close()

    agent.save(str(args.checkpoint_dir / "last.pt"))


if __name__ == "__main__":
    main()
