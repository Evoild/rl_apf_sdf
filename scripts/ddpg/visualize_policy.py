from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import DDPGAgent, DDPGConfig
from envs import PandaObstacleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize six-joint obstacle-avoidance DDPG policy.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ddpg/best.pt"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
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
        max_steps=12_000,
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
    frame_dt = 1.0 / max(args.fps, 1e-6)

    try:
        for episode in range(args.episodes):
            obs = env.reset()
            done = False
            episode_return = 0.0
            step = 0
            print(f"\nEpisode {episode + 1}/{args.episodes}")
            while not done:
                started = time.time()
                action = agent.act(obs, explore=False) * args.action_scale
                action = np.clip(action, env.action_space.low, env.action_space.high)
                obs, reward, done, info = env.step(action)
                episode_return += reward
                env.render()
                if step % 10 == 0 or done:
                    print(
                        "step={:03d} reward={:8.3f} return={:9.3f} target_dist={:.4f} "
                        "clearance={:.4f} collision={} success={} action={}".format(
                            step,
                            reward,
                            episode_return,
                            info["target_distance"],
                            info["obstacle_clearance"],
                            info["collision"],
                            info["success"],
                            np.round(action, 3),
                        )
                    )
                elapsed = time.time() - started
                time.sleep(max(0.0, frame_dt - elapsed))
                step += 1
            print(f"episode_return={episode_return:.3f} success={info['success']} timeout={info['timeout']}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
