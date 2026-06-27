from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import PPOAgent, PPOConfig
from envs import PandaObstacleEnv
from discrete_actions import DISCRETE_ACTION_DIM, JOINT_STEP_DEG, discrete_action_to_env, make_joint_direction_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize six-joint obstacle-avoidance PPO policy.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ppo/best.pt"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--fps", type=float, default=15.0)
    parser.add_argument("--action-scale", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stochastic", action="store_true")
    parser.add_argument("--joint-step-deg", type=float, default=JOINT_STEP_DEG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    env = PandaObstacleEnv(
        randomize_reset=False,
        max_steps=12000,
        max_joint_step=np.deg2rad(args.joint_step_deg),
    )
    agent = PPOAgent(
        env.obs_dim,
        DISCRETE_ACTION_DIM,
        PPOConfig(),
        device=args.device,
    )
    agent.load(str(args.checkpoint))
    action_table = make_joint_direction_actions(env.action_dim)
    frame_dt = 1.0 / max(args.fps, 1e-6)

    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        step = 0
        print(f"\nEpisode {episode + 1}/{args.episodes}")
        while not done:
            started = time.time()
            action_index, _, _ = agent.act(obs, deterministic=not args.stochastic)
            action = discrete_action_to_env(action_index, action_table) * args.action_scale
            obs, reward, done, info = env.step(action)
            episode_return += reward
            env.render()
            if step % 10 == 0 or done:
                print(
                    "step={:03d} reward={:8.3f} return={:9.3f} target_dist={:.4f} "
                    "clearance={:.4f} collision={} success={}".format(
                        step,
                        reward,
                        episode_return,
                        info["target_distance"],
                        info["obstacle_clearance"],
                        info["collision"],
                        info["success"],
                    )
                )
            elapsed = time.time() - started
            time.sleep(max(0.0, frame_dt - elapsed))
            step += 1
        print(f"episode_return={episode_return:.3f} success={info['success']} timeout={info['timeout']}")

    env.close()


if __name__ == "__main__":
    main()
