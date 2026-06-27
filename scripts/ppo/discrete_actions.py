from __future__ import annotations

import numpy as np


DISCRETE_ACTION_DIM = 12
JOINT_STEP_DEG = 5.0


def make_joint_direction_actions(controlled_joints: int = 6) -> np.ndarray:
    """Return actions for +/- one joint at a time.

    Rows are ordered as joint0 positive, joint0 negative, joint1 positive,
    joint1 negative, and so on. The environment converts +/-1 into
    +/-max_joint_step radians.
    """
    actions = np.zeros((2 * controlled_joints, controlled_joints), dtype=np.float32)
    for joint in range(controlled_joints):
        actions[2 * joint, joint] = 1.0
        actions[2 * joint + 1, joint] = -1.0
    return actions


def discrete_action_to_env(action_index: int, action_table: np.ndarray) -> np.ndarray:
    if action_index < 0 or action_index >= len(action_table):
        raise ValueError(f"Discrete action index {action_index} out of range [0, {len(action_table)})")
    return action_table[action_index].copy()
