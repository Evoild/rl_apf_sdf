from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np


ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 8))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rotate the seven Panda arm joints in sequence.")
    parser.add_argument("--angle", type=float, default=15.0, help="Rotation of each joint in degrees.")
    parser.add_argument("--move-time", type=float, default=1.5, help="Seconds used for each joint motion.")
    parser.add_argument("--hold-time", type=float, default=0.5, help="Seconds to hold after each motion.")
    parser.add_argument("--fps", type=float, default=60.0)
    parser.add_argument("--no-render", action="store_true", help="Run without opening the MuJoCo viewer.")
    return parser.parse_args()


def joint_ids(model: mujoco.MjModel) -> np.ndarray:
    ids = np.asarray(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in ARM_JOINTS],
        dtype=np.int32,
    )
    if np.any(ids < 0):
        missing = [name for name, joint_id in zip(ARM_JOINTS, ids) if joint_id < 0]
        raise RuntimeError(f"Missing Panda joints: {missing}")
    return ids


def step_for_duration(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    duration: float,
    viewer,
    fps: float,
) -> None:
    end_time = data.time + max(duration, 0.0)
    frame_dt = 1.0 / max(fps, 1e-6)
    while data.time < end_time:
        started = time.monotonic()
        next_frame = min(data.time + frame_dt, end_time)
        while data.time < next_frame:
            mujoco.mj_step(model, data)
        if viewer is not None:
            viewer.sync()
            time.sleep(max(0.0, frame_dt - (time.monotonic() - started)))


def main() -> None:
    args = parse_args()
    model_path = Path(__file__).resolve().parents[1] / "models" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ids = joint_ids(model)

    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    data.ctrl[:7] = data.qpos[:7]
    if model.nu > 7:
        data.ctrl[7:] = 255.0

    viewer = None
    if not args.no_render:
        from mujoco import viewer as mujoco_viewer

        viewer = mujoco_viewer.launch_passive(model, data)
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -25.0
        viewer.cam.lookat[:] = (0.25, 0.0, 0.45)

    angle_rad = np.deg2rad(args.angle)
    try:
        for actuator_index, (name, joint_id) in enumerate(zip(ARM_JOINTS, ids)):
            qpos_index = int(model.jnt_qposadr[joint_id])
            start = float(data.qpos[qpos_index])
            lower, upper = model.jnt_range[joint_id]
            target = float(np.clip(start + angle_rad, lower, upper))
            actual_angle = np.rad2deg(target - start)

            print(f"{name}: {np.rad2deg(start):.2f}° -> {np.rad2deg(target):.2f}° "
                  f"(rotation {actual_angle:.2f}°)")

            motion_start_time = data.time
            motion_end_time = motion_start_time + max(args.move_time, 0.0)
            while data.time < motion_end_time:
                elapsed = data.time - motion_start_time
                ratio = min(elapsed / max(args.move_time, 1e-9), 1.0)
                # Smoothstep interpolation avoids an abrupt position target jump.
                ratio = ratio * ratio * (3.0 - 2.0 * ratio)
                data.ctrl[actuator_index] = start + ratio * (target - start)
                step_for_duration(model, data, 1.0 / max(args.fps, 1e-6), viewer, args.fps)

            data.ctrl[actuator_index] = target
            step_for_duration(model, data, args.hold_time, viewer, args.fps)
    finally:
        if viewer is not None:
            viewer.close()


if __name__ == "__main__":
    main()
