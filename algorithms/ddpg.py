from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import random

import numpy as np
import torch
from torch import nn


@dataclass
class DDPGConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 1e-4
    critic_lr: float = 1e-3
    batch_size: int = 256
    replay_size: int = 500_000
    warmup_steps: int = 5_000
    exploration_noise: float = 0.25
    noise_decay: float = 0.99995
    min_noise: float = 0.03
    max_grad_norm: float = 1.0
    hidden_size: int = 256


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class Critic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_value = self.net(torch.cat([obs, action], dim=-1)).squeeze(-1)
        return q_value


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.storage: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(maxlen=capacity)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, next_obs: np.ndarray, done: bool) -> None:
        self.storage.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.storage, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
        obs_np = np.nan_to_num(np.asarray(obs), nan=0.0, posinf=1e3, neginf=-1e3)
        actions_np = np.nan_to_num(np.asarray(actions), nan=0.0, posinf=1.0, neginf=-1.0)
        rewards_np = np.clip(np.nan_to_num(np.asarray(rewards, dtype=np.float32), nan=0.0), -25.0, 25.0)
        next_obs_np = np.nan_to_num(np.asarray(next_obs), nan=0.0, posinf=1e3, neginf=-1e3)
        dones_np = np.asarray(dones, dtype=np.float32)
        return (
            torch.as_tensor(obs_np, dtype=torch.float32, device=device),
            torch.as_tensor(actions_np, dtype=torch.float32, device=device),
            torch.as_tensor(rewards_np, dtype=torch.float32, device=device),
            torch.as_tensor(next_obs_np, dtype=torch.float32, device=device),
            torch.as_tensor(dones_np, dtype=torch.float32, device=device),
        )

    def __len__(self) -> int:
        return len(self.storage)


class DDPGAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: DDPGConfig,
        device: str = "cpu",
        action_low: np.ndarray | None = None,
        action_high: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.action_low_np = self._as_action_bound(action_low, action_dim, -1.0)
        self.action_high_np = self._as_action_bound(action_high, action_dim, 1.0)
        self.action_scale_np = (self.action_high_np - self.action_low_np) * 0.5
        self.action_offset_np = (self.action_high_np + self.action_low_np) * 0.5
        self.action_low = torch.as_tensor(self.action_low_np, dtype=torch.float32, device=self.device)
        self.action_high = torch.as_tensor(self.action_high_np, dtype=torch.float32, device=self.device)
        self.action_scale = torch.as_tensor(self.action_scale_np, dtype=torch.float32, device=self.device)
        self.action_offset = torch.as_tensor(self.action_offset_np, dtype=torch.float32, device=self.device)
        self.actor = Actor(obs_dim, action_dim, config.hidden_size).to(self.device)
        self.actor_target = Actor(obs_dim, action_dim, config.hidden_size).to(self.device)
        self.critic = Critic(obs_dim, action_dim, config.hidden_size).to(self.device)
        self.critic_target = Critic(obs_dim, action_dim, config.hidden_size).to(self.device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)
        self.noise_std = config.exploration_noise

    @torch.no_grad()
    def act(self, obs: np.ndarray, explore: bool = True) -> np.ndarray:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        normalized_action = self.actor(obs_tensor).squeeze(0).cpu().numpy()
        if explore:
            normalized_action += np.random.normal(0.0, self.noise_std, size=normalized_action.shape)
            self.noise_std = max(self.config.min_noise, self.noise_std * self.config.noise_decay)
        normalized_action = np.clip(normalized_action, -1.0, 1.0)
        action = normalized_action * self.action_scale_np + self.action_offset_np
        return np.clip(action, self.action_low_np, self.action_high_np).astype(np.float32)

    def update(self, replay: ReplayBuffer) -> dict[str, float]:
        cfg = self.config
        if len(replay) < cfg.batch_size:
            return {"actor_loss": 0.0, "critic_loss": 0.0, "noise_std": self.noise_std}

        obs, actions, rewards, next_obs, dones = replay.sample(cfg.batch_size, self.device)
        with torch.no_grad():
            next_actions = self._scale_normalized_action(self.actor_target(next_obs))
            next_q = self.critic_target(next_obs, next_actions)
            target_q = rewards + cfg.gamma * (1.0 - dones) * next_q
            target_q = torch.clamp(target_q, -100.0, 100.0)

        current_q = self.critic(obs, actions)
        critic_loss = nn.functional.mse_loss(current_q, target_q)
        if torch.isfinite(critic_loss):
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self.critic.parameters(), cfg.max_grad_norm)
            self.critic_optimizer.step()

        actor_actions = self._scale_normalized_action(self.actor(obs))
        actor_loss = -self.critic(obs, actor_actions).mean()
        if torch.isfinite(actor_loss):
            self.actor_optimizer.zero_grad(set_to_none=True)
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), cfg.max_grad_norm)
            self.actor_optimizer.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        return {
            "actor_loss": float(actor_loss.item()) if torch.isfinite(actor_loss) else 0.0,
            "critic_loss": float(critic_loss.item()) if torch.isfinite(critic_loss) else 0.0,
            "noise_std": float(self.noise_std),
        }

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        with torch.no_grad():
            for target_param, source_param in zip(target.parameters(), source.parameters()):
                target_param.mul_(1.0 - self.config.tau)
                target_param.add_(self.config.tau * source_param)

    @staticmethod
    def _as_action_bound(bound: np.ndarray | None, action_dim: int, default: float) -> np.ndarray:
        if bound is None:
            return np.full(action_dim, default, dtype=np.float32)
        array = np.asarray(bound, dtype=np.float32)
        if array.shape != (action_dim,):
            raise ValueError(f"Expected action bound shape {(action_dim,)}, got {array.shape}")
        return array

    def _scale_normalized_action(self, normalized_action: torch.Tensor) -> torch.Tensor:
        action = normalized_action * self.action_scale + self.action_offset
        return torch.clamp(action, self.action_low, self.action_high)

    def save(self, path: str) -> None:
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_target": self.actor_target.state_dict(),
                "critic_target": self.critic_target.state_dict(),
                "config": self.config.__dict__,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.actor_target.load_state_dict(checkpoint.get("actor_target", checkpoint["actor"]))
        self.critic_target.load_state_dict(checkpoint.get("critic_target", checkpoint["critic"]))
