from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs import PandaObstacleEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Random six-joint rollout for the Panda obstacle-avoidance environment.")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--render", dest="render", action="store_true", default=True)
    parser.add_argument("--no-render", dest="render", action="store_false")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--print-every", type=int, default=25)
    parser.add_argument("--num-obstacles", type=int, default=1)
    parser.add_argument("--seed", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = PandaObstacleEnv(seed=args.seed, num_obstacles=args.num_obstacles)
    rng = np.random.default_rng(args.seed)
    frame_dt = 1.0 / max(args.fps, 1e-6)
    obs = env.reset()
    total_reward = 0.0

    try:
        print(
            "scene "
            f"start={np.round(env.initial_ee_pos, 3)} "
            f"goal={np.round(env.goal, 3)} "
            f"obstacles={np.round(env.obstacles, 3)}"
        )
        if args.render:
            env.render()

        for step in range(args.steps):
            started = time.time()
            action = env.action_space.sample(rng)
            obs, reward, done, info = env.step(action)
            total_reward += reward

            if args.render:
                env.render()
                time.sleep(max(0.0, frame_dt - (time.time() - started)))

            if step % max(args.print_every, 1) == 0 or done:
                print(
                    f"step={step:04d} reward={reward:8.3f} return={total_reward:9.3f} "
                    f"target_distance={info['target_distance']:.4f} "
                    f"clearance={info['obstacle_clearance']:.4f} "
                    f"collision={info['collision']} success={info['success']}"
                )

            if done:
                obs = env.reset()
                total_reward = 0.0
                print(
                    "scene "
                    f"start={np.round(env.initial_ee_pos, 3)} "
                    f"goal={np.round(env.goal, 3)} "
                    f"obstacles={np.round(env.obstacles, 3)}"
                )
                if args.render:
                    env.render()
    finally:
        env.close()


if __name__ == "__main__":
    main()
