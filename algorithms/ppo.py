from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    policy_lr: float = 3e-4
    value_lr: float = 1e-3
    train_epochs: int = 5
    minibatch_size: int = 512
    entropy_coef: float = 1e-4
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_size: int = 256
    log_std_init: float = -0.5


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, log_std_init: float) -> None:
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh(),
        )
        final_actor_layer = self.actor_mean[-2]
        nn.init.zeros_(final_actor_layer.weight)
        nn.init.zeros_(final_actor_layer.bias)
        self.log_std = torch.nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def action_mean(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor_mean(obs)

    def distribution(self, obs: torch.Tensor) -> Normal:
        mean = self.action_mean(obs)
        std = torch.exp(self.log_std).expand_as(mean)
        return Normal(mean, std)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs).squeeze(-1)


class RolloutBuffer:
    def __init__(self) -> None:
        self.observations: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.values: list[float] = []

    def add(self, obs, action: np.ndarray, log_prob, reward, done, value) -> None:
        self.observations.append(obs)
        self.actions.append(np.asarray(action, dtype=np.float32))
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def clear(self) -> None:
        self.__init__()

    def __len__(self) -> int:
        return len(self.rewards)


class PPOAgent:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig,
        device: str = "cpu",
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.model = ActorCritic(obs_dim, action_dim, config.hidden_size, config.log_std_init).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.model.actor_mean.parameters(), "lr": config.policy_lr},
                {"params": [self.model.log_std], "lr": config.policy_lr},
                {"params": self.model.critic.parameters(), "lr": config.value_lr},
            ]
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple[np.ndarray, float, float]:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        dist = self.model.distribution(obs_tensor)
        raw_action = dist.mean if deterministic else dist.sample()
        action = torch.clamp(raw_action, -1.0, 1.0)
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.model.value(obs_tensor)
        return (
            action.squeeze(0).cpu().numpy().astype(np.float32),
            float(log_prob.item()),
            float(value.item()),
        )

    def update(self, buffer: RolloutBuffer, last_value: float) -> dict[str, float]:
        cfg = self.config
        rewards = np.asarray(buffer.rewards, dtype=np.float32)
        dones = np.asarray(buffer.dones, dtype=np.float32)
        values = np.asarray(buffer.values + [last_value], dtype=np.float32)

        advantages = np.zeros_like(rewards)
        gae = 0.0
        for step in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[step]
            delta = rewards[step] + cfg.gamma * values[step + 1] * nonterminal - values[step]
            gae = delta + cfg.gamma * cfg.gae_lambda * nonterminal * gae
            advantages[step] = gae
        returns = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        obs_np = np.nan_to_num(np.asarray(buffer.observations), nan=0.0, posinf=1e3, neginf=-1e3)
        actions_np = np.clip(np.nan_to_num(np.asarray(buffer.actions), nan=0.0), -1.0, 1.0).astype(np.float32)
        log_probs_np = np.nan_to_num(np.asarray(buffer.log_probs), nan=0.0, posinf=0.0, neginf=0.0)
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions_np, dtype=torch.float32, device=self.device)
        old_log_probs = torch.as_tensor(log_probs_np, dtype=torch.float32, device=self.device)
        returns_t = torch.as_tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)

        indices = np.arange(len(buffer))
        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_fraction": 0.0}
        updates = 0
        for _ in range(cfg.train_epochs):
            np.random.shuffle(indices)
            for start in range(0, len(indices), cfg.minibatch_size):
                batch = indices[start : start + cfg.minibatch_size]
                dist = self.model.distribution(obs[batch])
                log_probs = dist.log_prob(actions[batch]).sum(dim=-1)
                log_ratio = log_probs - old_log_probs[batch]
                ratio = torch.exp(log_ratio)
                unclipped = ratio * advantages_t[batch]
                clipped = torch.clamp(ratio, 1.0 - cfg.clip_ratio, 1.0 + cfg.clip_ratio) * advantages_t[batch]
                policy_loss = -torch.min(unclipped, clipped).mean()

                values_pred = self.model.value(obs[batch])
                value_loss = nn.functional.mse_loss(values_pred, returns_t[batch])
                entropy = dist.entropy().sum(dim=-1).mean()
                loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * entropy
                if not torch.isfinite(loss):
                    continue

                with torch.no_grad():
                    approx_kl = ((ratio - 1.0) - log_ratio).mean()
                    clip_fraction = (torch.abs(ratio - 1.0) > cfg.clip_ratio).float().mean()

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), cfg.max_grad_norm)
                self.optimizer.step()

                metrics["policy_loss"] += float(policy_loss.item())
                metrics["value_loss"] += float(value_loss.item())
                metrics["entropy"] += float(entropy.item())
                metrics["approx_kl"] += float(approx_kl.item())
                metrics["clip_fraction"] += float(clip_fraction.item())
                updates += 1

        for key in metrics:
            metrics[key] /= max(updates, 1)
        return metrics

    def save(self, path: str) -> None:
        torch.save({"model": self.model.state_dict(), "config": self.config.__dict__}, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
