import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal, kl_divergence
import numpy as np
import random
from collections import deque
import gymnasium as gym
from typing import List, Tuple, Dict, Deque, NamedTuple, Optional, Any


# --- Replay Buffer ---
class Transition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    policy_index: int


class ReplayBuffer:
    memory: Deque[Transition]  # Type hint for memory attribute

    def __init__(self, capacity: int):
        self.memory = deque([], maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        policy_index: int,
    ) -> None:
        """Save a transition"""
        self.memory.append(
            Transition(state, action, reward, next_state, done, policy_index)
        )

    def sample(self, batch_size: int) -> List[Transition]:  # Corrected type hint
        return random.sample(self.memory, batch_size)

    def __len__(self) -> int:
        return len(self.memory)


# --- Networks ---
class ValueNetwork(nn.Module):
    linear1: nn.Linear
    linear2: nn.Linear
    linear3: nn.Linear

    def __init__(self, num_inputs: int, num_actions: int, hidden_dim: int):
        super(ValueNetwork, self).__init__()
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    DEVICE: torch.device
    linear1: nn.Linear
    linear2: nn.Linear
    mean_linear: nn.Linear
    log_std_linear: nn.Linear
    log_std_min: float
    log_std_max: float
    action_scale: torch.Tensor
    action_bias: torch.Tensor

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_dim: int,
        action_space: Optional[gym.spaces.Box] = None,
        log_std_min: float = -20,
        log_std_max: float = 2,
        computation_device: torch.device = torch.device("cpu"),
    ):
        super(PolicyNetwork, self).__init__()
        self.DEVICE = computation_device
        self.linear1 = nn.Linear(num_inputs, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

        self.mean_linear = nn.Linear(hidden_dim, num_actions)
        self.log_std_linear = nn.Linear(hidden_dim, num_actions)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Action scaling to make sure that actions are in the correct range
        if action_space is not None:
            self.action_scale = torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
                device=self.DEVICE,
            )
            self.action_bias = torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
                device=self.DEVICE,
            )
        else:
            print("WARNING: action_space is None, using default scaling.")
            self.action_scale = torch.tensor(
                1.0, dtype=torch.float32, device=self.DEVICE
            )
            self.action_bias = torch.tensor(
                0.0, dtype=torch.float32, device=self.DEVICE
            )

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_min, max=self.log_std_max)
        return mean, log_std

    def sample(
        self, state: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Sample an action from the policy network.
        Args:
            state: Current state tensor (batch size 1).
        Returns:
            action: Sampled action tensor.
            log_prob: Log probability of the sampled action.
            mean_action: Mean action after tanh and scaling.
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (2 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

    def get_distribution(self, state: torch.Tensor) -> Normal:
        mean, log_std = self.forward(state)
        std = log_std.exp()
        return Normal(mean, std)


# --- Aggregation Function ---
def aggregate_actions_max_q(
    state: torch.Tensor, actors: List[PolicyNetwork], critics: List[ValueNetwork]
) -> Tuple[torch.Tensor, int, List[torch.Tensor]]:
    """
    Selects action based on the highest minimum Q-value estimate.
    Args:
        state: Current state tensor (batch size 1).
        actors: List of PolicyNetwork instances.
        critics: List of two ValueNetwork instances [critic1, critic2].
    Returns:
        action: The selected action tensor.
        policy_index: The index of the actor whose action was selected.
        all_actions: List of actions proposed by each actor.
    """
    with torch.no_grad():
        all_actions: List[torch.Tensor] = []
        min_q_values: List[torch.Tensor] = []
        for actor in actors:
            action, _, _ = actor.sample(state)
            q1 = critics[0](state, action)
            q2 = critics[1](state, action)
            min_q = torch.min(q1, q2)
            all_actions.append(action)
            min_q_values.append(min_q)

        min_q_tensor = torch.cat(min_q_values, dim=1)  # Shape: (1, num_actors)
        best_actor_idx: int = int(torch.argmax(min_q_tensor, dim=1).item())
        selected_action = all_actions[best_actor_idx]

    return selected_action, best_actor_idx, all_actions


# --- SnAC Agent ---
class SnAC:
    DEVICE: torch.device
    gamma: float
    tau: float
    alpha: float
    alpha_div: float
    target_update_interval: int
    update_every: int
    batch_size: int
    num_actors: int
    action_space: gym.spaces.Box  # Assuming Box space
    state_dim: int
    action_dim: int
    critic1: ValueNetwork
    critic2: ValueNetwork
    critic1_target: ValueNetwork
    critic2_target: ValueNetwork
    q_optimizer: optim.Optimizer
    actors: List[PolicyNetwork]
    policy_optimizers: List[optim.Optimizer]
    auto_entropy: bool
    target_entropy: float
    log_alpha: torch.Tensor
    alpha_optimizer: Optional[optim.Optimizer]  # Can be None if auto_entropy is False
    memory: ReplayBuffer
    updates: int

    def __init__(
        self,
        env: gym.Env,
        num_actors: int = 3,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr_q: float = 3e-4,
        lr_pi: float = 3e-4,
        lr_alpha: float = 3e-4,
        alpha: float = 0.2,
        alpha_div: float = 0.1,
        target_update_interval: int = 1,
        update_every: int = 1,
        batch_size: int = 256,
        memory_size: int = 1000000,
        hidden_dim: int = 256,
        auto_entropy: bool = True,
        target_entropy: Optional[float] = None,
        computation_device: torch.device = torch.device("cpu"),
    ):
        self.DEVICE = computation_device
        self.gamma = gamma
        self.tau = tau
        self.alpha_div = alpha_div
        self.target_update_interval = target_update_interval
        self.update_every = update_every
        self.batch_size = batch_size
        self.num_actors = num_actors

        # Type assertion for action_space if needed, or handle different space types
        assert isinstance(env.action_space, gym.spaces.Box), (
            "Action space must be gym.spaces.Box"
        )
        assert isinstance(env.observation_space, gym.spaces.Box), (
            "Observation space must be gym.spaces.Box"
        )
        self.action_space = env.action_space
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = self.action_space.shape[0]

        # Critics
        self.critic1 = ValueNetwork(self.state_dim, self.action_dim, hidden_dim).to(
            self.DEVICE
        )
        self.critic2 = ValueNetwork(self.state_dim, self.action_dim, hidden_dim).to(
            self.DEVICE
        )
        self.critic1_target = ValueNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.DEVICE)
        self.critic2_target = ValueNetwork(
            self.state_dim, self.action_dim, hidden_dim
        ).to(self.DEVICE)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        self.q_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), lr=lr_q
        )

        # Actors
        self.actors = [
            PolicyNetwork(
                self.state_dim,
                self.action_dim,
                hidden_dim,
                self.action_space,
                computation_device=self.DEVICE,
            ).to(self.DEVICE)
            for _ in range(num_actors)
        ]
        self.policy_optimizers = [
            optim.Adam(actor.parameters(), lr=lr_pi) for actor in self.actors
        ]

        # Entropy tuning
        self.auto_entropy = auto_entropy
        if self.auto_entropy:
            if target_entropy is None:
                self.target_entropy = -float(np.prod(self.action_space.shape).item())
            else:
                self.target_entropy = target_entropy
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.DEVICE)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.target_entropy = 0.0
            self.log_alpha = torch.log(torch.tensor(alpha, device=self.DEVICE))
            self.alpha_optimizer = None
            self.alpha = alpha

        self.memory = ReplayBuffer(memory_size)
        self.updates = 0

    def select_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[np.ndarray, int]:
        state_tensor = torch.FloatTensor(state).to(self.DEVICE).unsqueeze(0)

        # TODO: handle multiple aggregation methods
        action, policy_index, _ = aggregate_actions_max_q(
            state_tensor, self.actors, [self.critic1, self.critic2]
        )

        return action.detach().cpu().numpy()[0], policy_index

    def update_parameters(self, updates_per_step: int) -> None:
        if len(self.memory) < self.batch_size:
            return

        for _ in range(updates_per_step):
            transitions = self.memory.sample(self.batch_size)

            # Convert tuples to numpy arrays before creating tensors
            state_np = np.array([t.state for t in transitions])
            action_np = np.array([t.action for t in transitions])
            reward_np = np.array([t.reward for t in transitions], dtype=np.float32)
            next_state_np = np.array([t.next_state for t in transitions])
            done_np = np.array([t.done for t in transitions], dtype=bool)
            policy_indices_np = np.array([t.policy_index for t in transitions], dtype=int)

            state_batch = torch.FloatTensor(state_np).to(self.DEVICE)
            action_batch = torch.FloatTensor(action_np).to(self.DEVICE)
            reward_batch = torch.FloatTensor(reward_np).to(self.DEVICE).unsqueeze(1)
            next_state_batch = torch.FloatTensor(next_state_np).to(self.DEVICE)
            done_batch = torch.FloatTensor(done_np).to(self.DEVICE).unsqueeze(1)
            policy_indices = torch.tensor(policy_indices_np, dtype=torch.long).to(
                self.DEVICE
            )

            # --- Critic Update ---
            with torch.no_grad():
                next_actions_list: List[torch.Tensor] = []
                next_log_probs_list: List[torch.Tensor] = []
                for i in range(self.batch_size):
                    idx: int = int(policy_indices[i].item())
                    if idx < 0 or idx >= self.num_actors:
                        # Handle cases where policy_index might be invalid (e.g., random actions)
                        # Option 1: Skip this sample (might introduce bias)
                        # Option 2: Use a default actor (e.g., actor 0)
                        # Option 3: Re-sample action (complex)
                        # Option 4: Randomly select an actor from the available ones
                        # Using actor 0 as a fallback for now:
                        idx = 0
                    next_state_i = next_state_batch[i].unsqueeze(0)
                    next_action_i, next_log_prob_i, _ = self.actors[idx].sample(
                        next_state_i
                    )
                    next_actions_list.append(next_action_i)
                    next_log_probs_list.append(next_log_prob_i)

                if not next_actions_list:
                    continue  # Skip update if no valid samples from indexing problems

                next_state_action = torch.cat(next_actions_list)
                next_state_log_pi = torch.cat(next_log_probs_list)

                qf1_next_target = self.critic1_target(
                    next_state_batch, next_state_action
                )
                qf2_next_target = self.critic2_target(
                    next_state_batch, next_state_action
                )
                min_qf_next_target = (
                    torch.min(qf1_next_target, qf2_next_target)
                    - self.alpha * next_state_log_pi
                )
                next_q_value = (
                    reward_batch + (1 - done_batch) * self.gamma * min_qf_next_target
                )

            qf1 = self.critic1(state_batch, action_batch)
            qf2 = self.critic2(state_batch, action_batch)
            qf1_loss = F.mse_loss(qf1, next_q_value)
            qf2_loss = F.mse_loss(qf2, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            self.q_optimizer.zero_grad()
            qf_loss.backward()
            self.q_optimizer.step()

            # --- Actor and Alpha Update ---
            for p in self.critic1.parameters():
                p.requires_grad = False
            for p in self.critic2.parameters():
                p.requires_grad = False

            alpha_loss_total = torch.tensor(0.0, device=self.DEVICE)
            for i, (actor, optimizer) in enumerate(
                zip(self.actors, self.policy_optimizers)
            ):
                pi_actions, pi_log_pi, _ = actor.sample(state_batch)

                qf1_pi = self.critic1(state_batch, pi_actions)
                qf2_pi = self.critic2(state_batch, pi_actions)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                policy_loss_q_term = (self.alpha * pi_log_pi - min_qf_pi).mean()

                kl_div_term = torch.tensor(0.0, device=self.DEVICE)
                if self.num_actors > 1 and self.alpha_div > 0:
                    current_dist = actor.get_distribution(state_batch)
                    kl_sum = torch.tensor(0.0, device=self.DEVICE)
                    for k, other_actor in enumerate(self.actors):
                        if i == k:
                            continue
                        with torch.no_grad():
                            other_dist = other_actor.get_distribution(state_batch)
                        kl = kl_divergence(current_dist, other_dist)
                        kl_sum += kl.mean()
                    kl_div_term = kl_sum

                policy_loss = policy_loss_q_term + self.alpha_div * kl_div_term

                optimizer.zero_grad()
                policy_loss.backward()
                optimizer.step()

                if self.auto_entropy:
                    alpha_loss = -(
                        self.log_alpha * (pi_log_pi + self.target_entropy).detach()
                    ).mean()
                    alpha_loss_total += alpha_loss

            for p in self.critic1.parameters():
                p.requires_grad = True
            for p in self.critic2.parameters():
                p.requires_grad = True

            if self.auto_entropy and self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                avg_alpha_loss = alpha_loss_total / self.num_actors
                avg_alpha_loss.backward()
                self.alpha_optimizer.step()
                self.alpha = self.log_alpha.exp().item()

            self.updates += 1
            if (self.updates % self.target_update_interval) == 0:
                self._soft_update(self.critic1_target, self.critic1, self.tau)
                self._soft_update(self.critic2_target, self.critic2, self.tau)

    def _soft_update(self, target: nn.Module, source: nn.Module, tau: float) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)

    def save_model(self, path: str) -> None:
        save_dict: Dict[str, Any] = {  # Use Dict from typing
            "critic1_state_dict": self.critic1.state_dict(),
            "critic2_state_dict": self.critic2.state_dict(),
            "critic1_target_state_dict": self.critic1_target.state_dict(),
            "critic2_target_state_dict": self.critic2_target.state_dict(),
            "q_optimizer_state_dict": self.q_optimizer.state_dict(),
            "actors_state_dict": [actor.state_dict() for actor in self.actors],
            "policy_optimizers_state_dict": [
                opt.state_dict() for opt in self.policy_optimizers
            ],
        }
        if self.auto_entropy and self.alpha_optimizer is not None:
            save_dict["log_alpha"] = self.log_alpha
            save_dict["alpha_optimizer_state_dict"] = self.alpha_optimizer.state_dict()

        torch.save(save_dict, f"{path}_snac.pt")

    def load_model(self, path: str) -> None:
        # Specify map_location type
        checkpoint = torch.load(f"{path}_snac.pt", map_location=self.DEVICE)
        self.critic1.load_state_dict(checkpoint["critic1_state_dict"])
        self.critic2.load_state_dict(checkpoint["critic2_state_dict"])
        self.critic1_target.load_state_dict(checkpoint["critic1_target_state_dict"])
        self.critic2_target.load_state_dict(checkpoint["critic2_target_state_dict"])
        self.q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])

        # Ensure the number of loaded actors matches
        num_loaded_actors = len(checkpoint["actors_state_dict"])
        if num_loaded_actors != self.num_actors:
            print(
                f"Warning: Loaded model has {num_loaded_actors} actors, but agent is configured for {self.num_actors}. Loading first {min(num_loaded_actors, self.num_actors)} actors."
            )
            load_count = min(num_loaded_actors, self.num_actors)
        else:
            load_count = self.num_actors

        for i in range(load_count):
            self.actors[i].load_state_dict(checkpoint["actors_state_dict"][i])
            self.policy_optimizers[i].load_state_dict(
                checkpoint["policy_optimizers_state_dict"][i]
            )

        if (
            self.auto_entropy
            and "log_alpha" in checkpoint
            and self.alpha_optimizer is not None
        ):
            self.log_alpha = (
                checkpoint["log_alpha"].to(self.DEVICE).requires_grad_(True)
            )  # Ensure correct device and grad
            # Re-wrap log_alpha in optimizer if necessary, or load state dict carefully
            # It might be safer to re-initialize the optimizer with the loaded log_alpha
            self.alpha_optimizer = optim.Adam(
                [self.log_alpha], lr=self.alpha_optimizer.defaults["lr"]
            )  # Re-init optimizer
            try:
                self.alpha_optimizer.load_state_dict(
                    checkpoint["alpha_optimizer_state_dict"]
                )
            except Exception as e:
                print(
                    f"Warning: Could not load alpha_optimizer state dict: {e}. Optimizer state reset."
                )
            self.alpha = self.log_alpha.exp().item()
