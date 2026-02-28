from copy import deepcopy
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from xuance.common import get_configs
from xuance.environment import make_envs
from xuance.torch.agents import OffPolicyAgent
from xuance.torch.learners import Learner, REGISTRY_Learners

from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA

class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1)  
        )

    def forward(self, state):
        return self.model(state)

class DiffusionPPOPolicy(nn.Module):

    def __init__(
        self,
        representation: nn.Module,
        action_dim: int,
        max_action: float,
        device: torch.device,
        beta_schedule: str = "linear",
        n_timesteps: int = 10,
        ema_decay: float = 0.99,
        step_start_ema: int = 500,
        update_ema_every: int = 3,
    ):
        super(DiffusionPPOPolicy, self).__init__()
        self.representation = representation
        self.feature_dim = int(self.representation.output_shapes["state"][0])
        self.action_dim = action_dim
        self.max_action = max_action
        self.device = device

        self.actor_backbone = MLP(
            state_dim=self.feature_dim,
            action_dim=action_dim,
            device=device,
        )
        self.actor = Diffusion(
            state_dim=self.feature_dim,
            action_dim=action_dim,
            model=self.actor_backbone,
            max_action=max_action,
            beta_schedule=beta_schedule,
            n_timesteps=n_timesteps,
        ).to(device)
        self.actor_old = deepcopy(self.actor).to(device)

        self.critic = Critic(self.feature_dim).to(device)

        self.ema = EMA(ema_decay)
        self.ema_model = deepcopy(self.actor).to(device)
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.step = 0

    def forward(self, observation):
        if isinstance(observation, np.ndarray):
            observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        elif not torch.is_tensor(observation):
            observation = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        else:
            observation = observation.to(self.device)

        output_rep = self.representation(observation)
        state_feat = output_rep["state"]
        actions = self.actor.sample(state_feat)
        # OffPolicyAgent.action() in Xuance expects a tuple of 3 elements.
        return output_rep, actions, None

    def sync_old_actor(self):
        self.actor_old.load_state_dict(self.actor.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)

    @torch.no_grad()
    def action(self, observation):
        output_rep = self.representation(observation)
        state_feat = output_rep["state"]
        if self.step > self.step_start_ema:
            if np.random.random() < 0.8:
                action = self.ema_model.sample(state_feat)
            else:
                action = self.actor.sample(state_feat)
        else:
            action = self.actor.sample(state_feat)
        return action


class DiffusionPPOLearner(Learner):

    def __init__(self, config, policy, callback):
        super(DiffusionPPOLearner, self).__init__(config, policy, callback)

        actor_lr = float(getattr(config, "actor_learning_rate", getattr(config, "learning_rate", 2e-4)))
        critic_lr = float(getattr(config, "critic_learning_rate", actor_lr * 3.0))
        self.grad_norm = float(getattr(config, "grad_norm", 1.0))
        self.tau = float(getattr(config, "tau", 0.95))
        self.clip_param = float(getattr(config, "clip_param", 0.2))
        self.beta_diffusion = float(getattr(config, "beta_diffusion", 0.5))
        self.value_coef = float(getattr(config, "value_loss_coef", 0.25))
        self.old_policy_sync = int(getattr(config, "old_policy_sync", 10))
        self.critic_updates = int(getattr(config, "critic_updates", 3))

        self.actor_optimizer = torch.optim.Adam(
            self.policy.actor.parameters(),
            lr=actor_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.policy.critic.parameters(),
            lr=critic_lr,
            betas=(0.9, 0.999),
        )

        self.optimizer = {
            "actor": self.actor_optimizer,
            "critic": self.critic_optimizer,
        }

    def compute_gae(self, rewards, values, next_values, not_dones):
        advantages = torch.zeros_like(rewards)
        last_advantage = 0.0
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values[t] * not_dones[t] - values[t]
            delta = torch.clamp(delta, -10.0, 10.0)
            last_advantage = delta + self.gamma * self.tau * not_dones[t] * last_advantage
            advantages[t] = last_advantage
        returns = advantages + values
        return advantages, returns

    def compute_ppo_loss(self, state_feat, actions, advantages):
        current_loss = self.policy.actor.loss(actions, state_feat)
        with torch.no_grad():
            old_loss = self.policy.actor_old.loss(actions, state_feat)
        log_ratio = torch.clamp(old_loss - current_loss, -20.0, 20.0)
        ratio = torch.exp(log_ratio)
        ratio = torch.where(torch.isnan(ratio), torch.ones_like(ratio), ratio)
        ratio = torch.clamp(ratio, 0.1, 10.0)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
        ppo_loss = -torch.min(surr1, surr2).mean()
        return ppo_loss, ratio

    def update(self, **samples):
        self.iterations += 1
        info = {}

        obs_batch = torch.as_tensor(samples["obs"], dtype=torch.float32, device=self.device)
        act_batch = torch.as_tensor(samples["actions"], dtype=torch.float32, device=self.device)
        next_batch = torch.as_tensor(samples["obs_next"], dtype=torch.float32, device=self.device)
        rew_batch = torch.as_tensor(samples["rewards"], dtype=torch.float32, device=self.device).reshape(-1)
        done_batch = torch.as_tensor(samples["terminals"], dtype=torch.float32, device=self.device).reshape(-1)
        not_done = 1.0 - done_batch

        with torch.no_grad():
            output_rep = self.policy.representation(obs_batch)
            next_rep = self.policy.representation(next_batch)
            state_feat = output_rep["state"].detach()
            next_feat = next_rep["state"].detach()

        with torch.no_grad():
            next_values = self.policy.critic(next_feat).squeeze(-1)
            td_target = rew_batch + self.gamma * next_values * not_done

        value_loss = torch.tensor(0.0, device=self.device)
        for _ in range(self.critic_updates):
            value_pred = self.policy.critic(state_feat).squeeze(-1)
            value_loss = F.mse_loss(value_pred, td_target.detach())
            if torch.isnan(value_loss) or torch.isinf(value_loss):
                break
            self.critic_optimizer.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.grad_norm)
            self.critic_optimizer.step()

        with torch.no_grad():
            values = self.policy.critic(state_feat).squeeze(-1)
            next_values = self.policy.critic(next_feat).squeeze(-1)
            advantages, _ = self.compute_gae(rew_batch, values, next_values, not_done)
            if advantages.std() > 1e-8:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                advantages = advantages - advantages.mean()

        diffusion_loss = self.policy.actor.loss(act_batch, state_feat)
        ppo_loss, ratio = self.compute_ppo_loss(state_feat, act_batch, advantages.detach())

        if torch.isnan(ppo_loss) or torch.isinf(ppo_loss):
            total_actor_loss = diffusion_loss.mean()
            ppo_loss = torch.tensor(0.0, device=self.device)
        else:
            total_actor_loss = ppo_loss + self.beta_diffusion * diffusion_loss.mean()

        self.actor_optimizer.zero_grad()
        total_actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.grad_norm)
        self.actor_optimizer.step()

        if self.iterations % self.old_policy_sync == 0:
            self.policy.sync_old_actor()

        if self.policy.step % self.policy.update_ema_every == 0:
            self.policy.step_ema()
        self.policy.step += 1

        info.update(
            {
                "iterations": self.iterations,
                "ppo_loss": float(ppo_loss.detach().item()),
                "diffusion_loss": float(diffusion_loss.mean().detach().item()),
                "value_loss": float(value_loss.detach().item()),
                "actor_loss": float(total_actor_loss.detach().item()),
                "ratio_mean": float(ratio.mean().detach().item()),
            }
        )
        return info


class DiffusionPPOAgent(OffPolicyAgent):
    def __init__(self, config, envs, callback=None):
        super(DiffusionPPOAgent, self).__init__(config, envs, callback)
        self.policy = self._build_policy()
        self.memory = self._build_memory()
        REGISTRY_Learners["DiffusionPPOLearner"] = DiffusionPPOLearner
        self.learner = self._build_learner(self.config, self.policy, self.callback)

    def _build_policy(self):
        representation = self._build_representation("Basic_MLP", self.observation_space, self.config)

        if not hasattr(self.action_space, "shape") or self.action_space.shape is None:
            raise ValueError("DiffusionPPOAgent expects continuous action space (Box).")

        action_dim = int(np.prod(self.action_space.shape))
        max_action = float(np.max(np.abs(self.action_space.high)))

        policy = DiffusionPPOPolicy(
            representation=representation,
            action_dim=action_dim,
            max_action=max_action,
            device=self.config.device,
            beta_schedule=getattr(self.config, "beta_schedule", "linear"),
            n_timesteps=int(getattr(self.config, "n_timesteps", 10)),
            ema_decay=float(getattr(self.config, "ema_decay", 0.99)),
            step_start_ema=int(getattr(self.config, "step_start_ema", 500)),
            update_ema_every=int(getattr(self.config, "update_ema_every", 3)),
        )
        return policy


if __name__ == '__main__':
    config = get_configs(file_dir="new_rl.yaml")
    config = Namespace(**config)

    if getattr(config, "learner", None) is None:
        config.learner = "DiffusionPPOLearner"

    envs = make_envs(config)
    agent = DiffusionPPOAgent(config, envs)

    if not config.test_mode:
        agent.train(config.running_steps // envs.num_envs)
        agent.save_model("final_train_model.pth")
    else:
        config.parallels = 1
        env_fn = lambda: make_envs(config)
        agent.load_model(agent.model_dir_load)
        _ = agent.test(env_fn, config.test_episode)

    agent.finish()
    envs.close()