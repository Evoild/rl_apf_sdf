from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import DDPGAgent, DDPGConfig
from envs import PandaObstacleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate six-joint obstacle-avoidance DDPG policy.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ddpg/best.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-obstacles", type=int, default=1)
    parser.add_argument("--obstacle-radius", type=float, default=0.04)
    parser.add_argument("--joint-step-deg", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

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

    returns = []
    successes = []
    episode_collisions = []
    step_collisions = []
    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            episode_return = 0.0
            step = 0
            while not done:
                action = agent.act(obs, explore=False)
                obs, reward, done, info = env.step(action)
                episode_return += reward
                step_collisions.append(float(info["collision"]))
                step += 1
                if args.render:
                    env.render()

            returns.append(episode_return)
            successes.append(float(info["success"]))
            episode_collisions.append(float(info["collision"]))
            print(
                f"episode={episode} return={episode_return:.2f} success={info['success']} steps={step} "
                f"target_distance={info['target_distance']:.4f} clearance={info['obstacle_clearance']:.4f} "
                f"collision={info['collision']} timeout={info['timeout']}"
            )

        print(
            f"summary return={np.mean(returns):.2f} success_rate={np.mean(successes):.2f} "
            f"episode_collision_rate={np.mean(episode_collisions):.2f} "
            f"step_collision_rate={np.mean(step_collisions):.3f}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
