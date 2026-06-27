from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install MuJoCo first: python3 -m pip install -r requirements.txt") from exc


@dataclass(frozen=True)
class ContinuousBoxSpace:
    """Minimal continuous-space descriptor for algorithm-agnostic env use."""

    low: np.ndarray
    high: np.ndarray
    shape: tuple[int, ...]
    dtype: type[np.floating]

    # 从连续空间中按均匀分布采样一个动作或观测样本，用于随机探索和 warmup。
    def sample(self, rng: np.random.Generator | None = None) -> np.ndarray:
        generator = rng if rng is not None else np.random.default_rng()
        return generator.uniform(self.low, self.high).astype(self.dtype)


class PandaObstacleEnv:
    """Six-joint Panda reaching with spherical obstacle avoidance.

    The policy controls joint1 through joint6 with normalized joint increments.
    Joint7 is held at its home angle and the gripper stays open.
    """

    # 初始化 MuJoCo 模型、动作/观测空间、目标和障碍物生成参数。
    def __init__(
        self,
        model_path: str | Path | None = None,
        num_obstacles: int = 1,
        obstacle_radius: float = 0.04,
        goal_radius: float = 0.03,
        success_threshold: float = 0.05,
        reward_huber_delta: float = 0.1,
        reward_obstacle_ref: float = 0.1,
        reward_obstacle_decay: float = 2.0,
        reward_time_ref: float = 0.2,
        collision_margin: float = 0.01,
        proximity_margin: float = 0.05,
        path_offset_range: float = 0.08,
        min_safety_dist: float = 0.3,
        obstacle_disturb_prob: float = 0.0,
        obstacle_disturb_step: float = 0.02,
        max_steps: int = 350,
        frame_skip: int = 10,
        max_joint_step: float = 0.12,
        seed: int | None = None,
        randomize_reset: bool = False,
    ) -> None:
        if model_path is None:
            model_path = Path(__file__).resolve().parents[1] / "models" / "scene.xml"

        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self.rng = np.random.default_rng(seed)

        self.num_obstacles = int(num_obstacles)
        if self.num_obstacles < 1:
            raise ValueError("num_obstacles must be at least 1")
        self.obstacle_radius = float(obstacle_radius)
        self.goal_radius = float(goal_radius)
        self.success_threshold = float(success_threshold)
        self.reward_huber_delta = float(reward_huber_delta)
        self.reward_obstacle_ref = float(reward_obstacle_ref)
        self.reward_obstacle_decay = float(reward_obstacle_decay)
        self.reward_time_ref = float(reward_time_ref)
        self.collision_margin = float(collision_margin)
        self.proximity_margin = float(proximity_margin)
        self.path_offset_range = float(path_offset_range)
        self.min_safety_dist = float(min_safety_dist)
        self.obstacle_disturb_prob = float(obstacle_disturb_prob)
        self.obstacle_disturb_step = float(obstacle_disturb_step)
        self.max_steps = int(max_steps)
        self.frame_skip = int(frame_skip)
        self.max_joint_step = float(max_joint_step)
        self.randomize_reset = bool(randomize_reset)

        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee_center_site")
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link0")
        if self.base_body_id < 0:
            raise RuntimeError("Missing Panda base body id for link0")
        self.target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.obstacle_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "obstacle")
        self.obstacle_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle_geom")
        self.floor_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.robot_body_ids = self._body_ids(
            [
                "link0",
                "link1",
                "link2",
                "link3",
                "link4",
                "link5",
                "link6",
                "link7",
                "hand",
                "left_finger",
                "right_finger",
            ]
        )

        self.arm_joints = 7
        self.controlled_joints = 6
        self.controlled_joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
            for index in range(1, self.controlled_joints + 1)
        ]
        if any(joint_id < 0 for joint_id in self.controlled_joint_ids):
            raise RuntimeError("Missing Panda joint ids for joint1 through joint6")
        self.floor_height = float(self.model.geom_pos[self.floor_geom_id, 2]) if self.floor_geom_id >= 0 else 0.0
        self.platform_geom_ids = [
            geom_id
            for name in ("platform", "work_platform", "table", "workbench")
            if (geom_id := mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)) >= 0
        ]
        # Each component commands a normalized increment for joint1--joint6.
        self.action_space = ContinuousBoxSpace(
            low=np.full(self.controlled_joints, -1.0, dtype=np.float32),
            high=np.full(self.controlled_joints, 1.0, dtype=np.float32),
            shape=(self.controlled_joints,),
            dtype=np.float32,
        )
        # State: joint_pos(6), ee_pos(3), distance_to_goal(1), joint-obstacle distances(6).
        obs_dim = self.controlled_joints + 3 + 1 + self.controlled_joints
        self.observation_space = ContinuousBoxSpace(
            low=np.full(obs_dim, -np.inf, dtype=np.float32),
            high=np.full(obs_dim, np.inf, dtype=np.float32),
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_dim = self.action_space.shape[0]
        self.obs_dim = self.observation_space.shape[0]

        self.workspace = {
            "x": np.array([0.5, 0.8], dtype=np.float64),
            "y": np.array([-0.5, 0.5], dtype=np.float64),
            "z": np.array([0.05, 0.3], dtype=np.float64),
        }
        self.home_joint_pos = np.array(
            [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4],
            dtype=np.float64,
        )
        self.goal = np.zeros(3, dtype=np.float64)
        self.obstacles = np.zeros((self.num_obstacles, 3), dtype=np.float64)
        self.initial_ee_pos = np.zeros(3, dtype=np.float64)
        self.base_to_goal_distance = 1.0
        self.prev_action = np.zeros(self.action_dim, dtype=np.float64)
        self.step_count = 0
        self.viewer = None

        self.reset()

    # 重置仿真到 home 位姿，重新生成目标和路径附近障碍物，并返回初始观测。
    def reset(self) -> np.ndarray:
        self.step_count = 0
        mujoco.mj_resetData(self.model, self.data)

        qpos = self.home_joint_pos.copy()
        if self.randomize_reset:
            qpos[: self.controlled_joints] += self.rng.uniform(-0.08, 0.08, size=self.controlled_joints)
        self.data.qpos[: self.arm_joints] = np.clip(
            qpos,
            self.model.jnt_range[: self.arm_joints, 0],
            self.model.jnt_range[: self.arm_joints, 1],
        )
        if self.model.nq >= 9:
            self.data.qpos[7:9] = 0.04
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.initial_ee_pos = self._ee_pos()
        self.goal = self._sample_goal()
        self.obstacles = self._generate_path_obstacles()
        self.prev_action.fill(0.0)
        self._sync_task_bodies()
        mujoco.mj_forward(self.model, self.data)
        self.base_to_goal_distance = max(float(np.linalg.norm(self._base_pos() - self.goal)), 1e-6)
        self.data.ctrl[: self.controlled_joints] = self.data.qpos[: self.controlled_joints]
        self.data.ctrl[6] = self.home_joint_pos[6]
        if self.model.nu > self.arm_joints:
            self.data.ctrl[self.arm_joints :] = 255.0
        return self._get_obs()

    # 接收 joint1--joint6 的归一化增量；joint7 固定在 home 角度。
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.shape != self.action_space.shape:
            raise ValueError(f"Expected action shape {self.action_space.shape}, got {action.shape}")

        action = np.clip(action, self.action_space.low, self.action_space.high)
        joint_target = self._joint_action_to_target(action)
        self.data.ctrl[: self.controlled_joints] = joint_target
        self.data.ctrl[6] = self.home_joint_pos[6]
        if self.model.nu > self.arm_joints:
            self.data.ctrl[self.arm_joints :] = 255.0
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        self._disturb_obstacles()
        ee_pos = self._ee_pos()
        reward, distance_to_goal, min_obstacle_distance, collision_info, reward_terms = self._reward(ee_pos)

        collision = collision_info["collision"]
        success = distance_to_goal < self.success_threshold and not collision
        timeout = self.step_count >= self.max_steps
        done = bool(success or collision or timeout)
        obs = self._get_obs()
        self.prev_action = action.copy()
        info = {
            "success": success,
            "is_success": success,
            "timeout": timeout,
            "collision": collision,
            "distance_to_goal": distance_to_goal,
            "target_distance": distance_to_goal,
            "min_obstacle_distance": min_obstacle_distance,
            "obstacle_clearance": min_obstacle_distance,
            "reward_target": reward_terms["target"],
            "reward_obstacle": reward_terms["obstacle"],
            "reward_energy": reward_terms["energy"],
            "reward_time": reward_terms["time"],
            "normalized_distance_to_goal": reward_terms["normalized_distance_to_goal"],
            "normalized_min_obstacle_distance": reward_terms["normalized_min_obstacle_distance"],
            "base_to_goal_distance": self.base_to_goal_distance,
            "ground_collision": collision_info["ground_collision"],
            "self_collision": collision_info["self_collision"],
            "obstacle_collision": collision_info["obstacle_collision"],
        }
        return obs, float(reward), done, info

    # 懒加载 MuJoCo viewer，并同步显示当前仿真状态和额外障碍物可视化几何。
    def render(self) -> None:
        if self.viewer is None:
            from mujoco import viewer

            self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 0.0
            self.viewer.cam.elevation = -30.0
            self.viewer.cam.lookat = np.array([0.2, 0.0, 0.4])
        self._render_user_scene()
        self.viewer.sync()

    # 关闭 viewer，释放可视化资源。
    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    # 在工作空间中随机采样目标点，并保证目标与初始末端位置有足够距离。
    def _sample_goal(self) -> np.ndarray:
        low = np.array([self.workspace["x"][0], self.workspace["y"][0], self.workspace["z"][0]])
        high = np.array([self.workspace["x"][1], self.workspace["y"][1], self.workspace["z"][1]])
        for _ in range(1_000):
            goal = self.rng.uniform(low=low, high=high)
            if np.linalg.norm(goal - self.initial_ee_pos) > self.min_safety_dist:
                return goal.astype(np.float64)
        return self.rng.uniform(low=low, high=high).astype(np.float64)

    # 在起点到目标点的线段中间区域采样一个路径点，作为障碍物基准位置。
    def _sample_path_point(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        t = self.rng.uniform(0.2, 0.8)
        return start + t * (end - start)

    # 给路径基准点添加垂直于起点-目标连线的小偏移，保证障碍物仍位于两点之间的路径管道内。
    def _add_path_offset(self, point: np.ndarray) -> np.ndarray:
        if self.path_offset_range <= 0.0:
            return point

        path_vector = self.goal - self.initial_ee_pos
        path_norm = float(np.linalg.norm(path_vector))
        if path_norm < 1e-8:
            return point

        direction = path_vector / path_norm
        raw_offset = self.rng.normal(0.0, self.path_offset_range / 3.0, size=3)
        perpendicular_offset = raw_offset - np.dot(raw_offset, direction) * direction
        offset_norm = float(np.linalg.norm(perpendicular_offset))
        if offset_norm > self.path_offset_range:
            perpendicular_offset *= self.path_offset_range / offset_norm
        return point + perpendicular_offset

    # 按“起点到目标路径附近”规则生成障碍物列表，并过滤离起点/目标过近的位置。
    def _generate_path_obstacles(self) -> np.ndarray:
        obstacles: list[np.ndarray] = []
        for _ in range(self.num_obstacles):
            for _attempt in range(1_000):
                path_base = self._sample_path_point(self.initial_ee_pos, self.goal)
                obstacle = self._add_path_offset(path_base)
                if self._is_valid_obstacle(obstacle):
                    obstacles.append(obstacle.astype(np.float64))
                    break
            else:
                obstacles.append(self._sample_path_point(self.initial_ee_pos, self.goal).astype(np.float64))
        return np.asarray(obstacles, dtype=np.float64)

    # 检查候选障碍物是否与起点和目标保持最小安全距离。
    def _is_valid_obstacle(self, obstacle: np.ndarray) -> bool:
        dist_to_start = float(np.linalg.norm(obstacle - self.initial_ee_pos))
        dist_to_goal = float(np.linalg.norm(obstacle - self.goal))
        return dist_to_start > self.min_safety_dist and dist_to_goal > self.min_safety_dist

    # 按给定概率小幅扰动障碍物位置，用于增加训练时的场景随机性。
    def _disturb_obstacles(self) -> None:
        if self.obstacle_disturb_prob <= 0.0 or self.rng.random() >= self.obstacle_disturb_prob:
            return
        for index in range(self.num_obstacles):
            disturb = self.rng.normal(0.0, self.obstacle_disturb_step / 3.0, size=3)
            disturb = np.clip(disturb, -self.obstacle_disturb_step, self.obstacle_disturb_step)
            candidate = self.obstacles[index] + disturb
            if self._is_valid_obstacle(candidate):
                self.obstacles[index] = candidate
        self._sync_task_bodies()

    # 将目标和第一个障碍物位置写入 MuJoCo 模型，使其参与真实仿真和碰撞显示。
    def _sync_task_bodies(self) -> None:
        self.model.body_pos[self.target_body_id] = self.goal
        self.model.body_pos[self.obstacle_body_id] = self.obstacles[0]
        self.model.geom_size[self.obstacle_geom_id, 0] = self.obstacle_radius

    # 在 viewer 的 user scene 中绘制额外障碍物；第一个障碍物和目标已由 XML 几何体显示。
    def _render_user_scene(self) -> None:
        if self.viewer is None:
            return
        max_geoms = len(self.viewer.user_scn.geoms)
        extra_obstacles = max(0, self.num_obstacles - 1)
        self.viewer.user_scn.ngeom = min(extra_obstacles, max_geoms)
        for index in range(self.viewer.user_scn.ngeom):
            obstacle_index = index + 1
            rgba = np.array([0.9, 0.12, 0.08, 0.8], dtype=np.float32)
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[index],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                size=np.array([self.obstacle_radius, 0.0, 0.0], dtype=np.float64),
                pos=self.obstacles[obstacle_index],
                mat=np.eye(3).reshape(-1),
                rgba=rgba,
            )

    # 网络输入包含前六个关节角、末端位置、末端到目标距离、各关节到静态障碍物的最小距离。
    def _get_obs(self) -> np.ndarray:
        ee_pos = self._ee_pos()
        distance_to_goal = np.array([np.linalg.norm(ee_pos - self.goal)], dtype=np.float64)
        min_joint_obstacle_distances = self._min_joint_static_obstacle_distances()
        obs = np.concatenate(
            [
                self.data.qpos[: self.controlled_joints],
                ee_pos,
                distance_to_goal,
                min_joint_obstacle_distances,
            ]
        )
        if obs.shape != self.observation_space.shape:
            raise RuntimeError(f"Observation shape drifted to {obs.shape}, expected {self.observation_space.shape}")
        return obs.astype(np.float32)

    # 论文组合奖励：R = Rtarget + Robstacle + Renergy + Rtime。
    def _reward(self, ee_pos: np.ndarray) -> tuple[float, float, float, dict[str, bool], dict[str, float]]:
        distance_to_goal = float(np.linalg.norm(ee_pos - self.goal))
        joint_obstacle_distances = self._min_joint_static_obstacle_distances()
        min_obstacle_distance = float(np.min(joint_obstacle_distances))
        collision_info = self._collision_info(ee_pos, min_obstacle_distance)
        action_norm = float(np.linalg.norm(self.data.qvel[: self.controlled_joints]))

        distance_scale = self.base_to_goal_distance
        normalized_distance_to_goal = distance_to_goal / distance_scale
        normalized_min_obstacle_distance = min_obstacle_distance / distance_scale
        normalized_huber_delta = self.reward_huber_delta / distance_scale
        normalized_obstacle_ref = self.reward_obstacle_ref / distance_scale
        normalized_time_ref = self.reward_time_ref / distance_scale
    
        delta = normalized_huber_delta
        if normalized_distance_to_goal < delta:
            reward_target = 0.5 * normalized_distance_to_goal**2
        else:
            reward_target = delta * (normalized_distance_to_goal - 0.5 * delta)

        reward_target = 1 / max(reward_target, 1e-6)

        dref1 = normalized_obstacle_ref
        dobstacle = max(normalized_min_obstacle_distance, 1e-6)
        reward_obstacle = -((dref1 / (dobstacle + dref1)) ** self.reward_obstacle_decay)

        reward_energy = -float(np.tanh(action_norm))
        exp_action_norm = float(np.exp(np.clip(action_norm, -50.0, 50.0)))
        if normalized_distance_to_goal < normalized_time_ref:
            reward_time = float((1.0 / exp_action_norm) - 1.0)
        else:
            reward_time = float(1.0 / (1.0 + exp_action_norm))

        terms = {
            "target": reward_target,
            "obstacle": reward_obstacle,
            "energy": reward_energy,
            "time": reward_time,
            "normalized_distance_to_goal": normalized_distance_to_goal,
            "normalized_min_obstacle_distance": normalized_min_obstacle_distance,
        }
        reward = sum(terms.values())
        print(terms)
        return reward, distance_to_goal, min_obstacle_distance, collision_info, terms

    # 将归一化动作映射成受关节限位约束的位置目标。
    def _joint_action_to_target(self, action: np.ndarray) -> np.ndarray:
        joint_delta = action * self.max_joint_step
        joint_target = self.data.qpos[: self.controlled_joints] + joint_delta
        joint_low = self.model.jnt_range[: self.controlled_joints, 0]
        joint_high = self.model.jnt_range[: self.controlled_joints, 1]
        return np.clip(joint_target, joint_low, joint_high)

    # 读取末端执行器 site 的世界坐标。
    def _ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.ee_site_id].copy()

    def _base_pos(self) -> np.ndarray:
        return self.data.xpos[self.base_body_id].copy()

    def _min_joint_static_obstacle_distances(self) -> np.ndarray:
        joint_positions = self.data.xanchor[np.asarray(self.controlled_joint_ids, dtype=np.int32)]
        sphere_distances = np.linalg.norm(joint_positions[:, None, :] - self.obstacles[None, :, :], axis=2)
        sphere_clearances = np.maximum(sphere_distances - self.obstacle_radius, 0.0)
        floor_clearances = np.maximum(joint_positions[:, 2] - self.floor_height, 0.0)

        clearances = [np.min(sphere_clearances, axis=1), floor_clearances]
        for geom_id in self.platform_geom_ids:
            geom_pos = self.data.geom_xpos[geom_id]
            geom_radius = float(np.linalg.norm(self.model.geom_size[geom_id]))
            platform_clearance = np.maximum(np.linalg.norm(joint_positions - geom_pos, axis=1) - geom_radius, 0.0)
            clearances.append(platform_clearance)
        return np.min(np.vstack(clearances), axis=0)

    def _collision_info(self, ee_pos: np.ndarray, min_obstacle_distance: float) -> dict[str, bool]:
        ground_collision = bool(ee_pos[2] < 0.05)
        obstacle_collision = bool(min_obstacle_distance < self.collision_margin + self.obstacle_radius)
        self_collision = False

        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            body1 = int(self.model.geom_bodyid[geom1])
            body2 = int(self.model.geom_bodyid[geom2])
            robot1 = body1 in self.robot_body_ids
            robot2 = body2 in self.robot_body_ids

            if (geom1 == self.floor_geom_id and robot2) or (geom2 == self.floor_geom_id and robot1):
                ground_collision = True
            if (geom1 == self.obstacle_geom_id and robot2) or (geom2 == self.obstacle_geom_id and robot1):
                obstacle_collision = True
            if robot1 and robot2 and body1 != body2 and not self._are_directly_connected(body1, body2):
                self_collision = True

        collision = ground_collision or obstacle_collision or self_collision
        return {
            "collision": bool(collision),
            "ground_collision": bool(ground_collision),
            "self_collision": bool(self_collision),
            "obstacle_collision": bool(obstacle_collision),
        }

    def _body_ids(self, names: list[str]) -> set[int]:
        body_ids = set()
        for name in names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                body_ids.add(int(body_id))
        return body_ids

    def _are_directly_connected(self, body1: int, body2: int) -> bool:
        return int(self.model.body_parentid[body1]) == body2 or int(self.model.body_parentid[body2]) == body1
