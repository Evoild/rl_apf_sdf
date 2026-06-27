from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from algorithms import PPOAgent, PPOConfig
from envs import PandaObstacleEnv
from discrete_actions import DISCRETE_ACTION_DIM, JOINT_STEP_DEG, discrete_action_to_env, make_joint_direction_actions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate six-joint obstacle-avoidance PPO policy.")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints_ppo/best.pt"))
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--joint-step-deg", type=float, default=JOINT_STEP_DEG)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = PandaObstacleEnv(randomize_reset=False, max_joint_step=np.deg2rad(args.joint_step_deg))
    agent = PPOAgent(
        env.obs_dim,
        DISCRETE_ACTION_DIM,
        PPOConfig(),
        device=args.device,
    )
    agent.load(str(args.checkpoint))
    action_table = make_joint_direction_actions(env.action_dim)

    returns = []
    successes = []
    collisions = []
    for episode in range(args.episodes):
        obs = env.reset()
        done = False
        episode_return = 0.0
        step = 0
        while not done:
            action_index, _, _ = agent.act(obs, deterministic=True)
            obs, reward, done, info = env.step(discrete_action_to_env(action_index, action_table))
            episode_return += reward
            collisions.append(float(info["collision"]))
            step += 1
            if args.render:
                env.render()
        returns.append(episode_return)
        successes.append(float(info["success"]))
        print(
            f"episode={episode} return={episode_return:.2f} success={info['success']} steps={step} "
            f"target_distance={info['target_distance']:.4f} clearance={info['obstacle_clearance']:.4f} "
            f"collision={info['collision']}"
        )

    print(
        f"summary return={np.mean(returns):.2f} success_rate={np.mean(successes):.2f} "
        f"collision_rate={np.mean(collisions):.3f}"
    )
    env.close()


if __name__ == "__main__":
    main()
