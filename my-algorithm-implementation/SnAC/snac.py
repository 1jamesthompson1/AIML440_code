import time
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
class Network(nn.Module):
    """
    Very simply super class used to abstract out the hidden layer management functions
    """
    hidden_layers: List[nn.Linear]
    computation_device: torch.device

    def __init__(self, computation_device):
        super().__init__()
        self.computation_device = computation_device

    def init_hidden_layers(self, input, arch):
        """
        It will generate a list of hidden layers which will have input of size 'input' and output of size arch[-1]
        The hidden layers are stored in the hidden_layers attribute
        """
        self.hidden_layers = [nn.Linear(input, arch[0], device=self.computation_device)]
        for i,layer_size in enumerate(arch[1:]):
            self.hidden_layers.append(nn.Linear(self.hidden_layers[i].out_features, layer_size, device=self.computation_device))

    def hidden_layer_fp(self, input):
        """
        Run through all of the hidden layers with the Relu activiation function applied to each one.
        """
        x = input
        for layer in self.hidden_layers:
            x = F.relu(layer(x))

        return x

    
class CriticNetwork(Network):
    output: nn.Linear

    def __init__(self, computation_device: torch.device, num_inputs: int, num_actions: int, arch: List[int]):
        super(CriticNetwork, self).__init__(computation_device)
        self.init_hidden_layers(num_inputs + num_actions, arch)

        self.output = nn.Linear(self.hidden_layers[-1].out_features, 1, device=self.computation_device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer_fp(torch.cat([state, action], 1))
        x = self.output(x)
        return x


class PolicyNetwork(Network):
    hidden_layers: List[nn.Linear]
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
        arch: List[int],
        action_space: Optional[gym.spaces.Box],
        computation_device: torch.device,
        log_std_min: float = -20,
        log_std_max: float = 2,
    ):
        super(PolicyNetwork, self).__init__(computation_device)
        self.computation_device = computation_device
        self.init_hidden_layers(num_inputs, arch)
        self.mean_linear = nn.Linear(self.hidden_layers[-1].out_features, num_actions, device=self.computation_device)
        self.log_std_linear = nn.Linear(self.hidden_layers[-1].out_features, num_actions, device=self.computation_device)

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # Action scaling to make sure that actions are in the correct range
        if action_space is not None:
            self.action_scale = torch.tensor(
                (action_space.high - action_space.low) / 2.0,
                dtype=torch.float32,
                device=self.computation_device,
            )
            self.action_bias = torch.tensor(
                (action_space.high + action_space.low) / 2.0,
                dtype=torch.float32,
                device=self.computation_device,
            )
        else:
            raise ValueError("action_space cannot be none")

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.hidden_layer_fp(state)
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


# --- SnAC Agent ---
class SnAC:
    computation_device: torch.device
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
    critic1: CriticNetwork
    critic2: CriticNetwork
    critic1_target: CriticNetwork
    critic2_target: CriticNetwork
    q_optimizer: optim.Optimizer
    actors: List[PolicyNetwork]
    policy_optimizers: List[optim.Optimizer]
    auto_entropy: bool
    target_entropy: float
    log_alpha: torch.Tensor
    alpha_optimizer: Optional[optim.Optimizer]  # Can be None if auto_entropy is False
    memory: ReplayBuffer
    updates: int
    aggregation_method: str

    def __init__(
        self,
        env: gym.Env,
        num_actors: int = 3,
        aggregation_method: str = "max_q",
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
        critic_arch: List[int] = [256, 256],
        actor_arch: List[int] = [256,256],
        auto_entropy: bool = True,
        target_entropy: Optional[float] = None,
        computation_device: torch.device = torch.device("cpu"),
    ):
        self.computation_device = computation_device
        self.gamma = gamma
        self.tau = tau
        self.alpha_div = alpha_div
        self.target_update_interval = target_update_interval
        self.update_every = update_every
        self.batch_size = batch_size
        self.num_actors = num_actors

        self.supported_aggregation_methods = [
            "elementwise",
            "weighted_elementwise",
            "max_q",
        ]
        assert aggregation_method in self.supported_aggregation_methods, (
            f"Aggregation method must be one of {self.supported_aggregation_methods}"
        )
        self.aggregation_method = aggregation_method

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
        self.critic1 = CriticNetwork(self.computation_device, self.state_dim, self.action_dim, critic_arch).to(
            self.computation_device
        )
        self.critic2 = CriticNetwork(self.computation_device, self.state_dim, self.action_dim, critic_arch).to(
            self.computation_device
        )
        self.critic1_target = CriticNetwork(
            self.computation_device,
            self.state_dim, self.action_dim, critic_arch
        ).to(self.computation_device)
        self.critic2_target = CriticNetwork(
            self.computation_device,
            self.state_dim, self.action_dim, critic_arch
        ).to(self.computation_device)
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
                actor_arch,
                self.action_space,
                computation_device=self.computation_device,
            ).to(self.computation_device)
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
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.computation_device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=lr_alpha)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.target_entropy = 0.0
            self.log_alpha = torch.log(torch.tensor(alpha, device=self.computation_device))
            self.alpha_optimizer = None
            self.alpha = alpha

        self.memory = ReplayBuffer(memory_size)
        self.updates = 0

        print("==SnAC agent made==")
        print(f"Parameters: num_actors={num_actors}, gamma={gamma}, tau={tau}, lr_q={lr_q}, lr_pi={lr_pi}, lr_alpha={lr_alpha}, alpha_div={alpha_div}, target_update_interval={target_update_interval}, update_every={update_every}, memory_size={memory_size}, batch_size={batch_size}, actor_arch={actor_arch}, critic_arch={critic_arch}, auto_entropy={auto_entropy}, target_entropy={target_entropy}, aggregation_method={aggregation_method}")

    def select_action(
        self, state: np.ndarray, evaluate: bool = False
    ) -> Tuple[np.ndarray, int]:
        state_tensor = torch.FloatTensor(state).to(self.computation_device).unsqueeze(0)

        with torch.no_grad():
            all_actions: List[torch.Tensor] = []
            min_q_values: List[torch.Tensor] = []
            for actor in self.actors:
                action, _, _ = actor.sample(state_tensor)
                q1 = self.critic1(state_tensor, action)
                q2 = self.critic2(state_tensor, action)
                min_q = torch.min(q1, q2)
                all_actions.append(action)
                min_q_values.append(min_q)
        
        selected_action, policy_index = self.aggregate_actions(
            all_actions, min_q_values
        ) 

        return selected_action.detach().cpu().numpy()[0], policy_index

    def aggregate_actions(
            self, actions: List[torch.Tensor], action_values: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, int]:
        """
        Selects a naction based on the aggregation method of the class.

        Args:
            actions: List of actions proposed by each actor.
            values: List of Q-values for each action.
        Returns:
            action: The selected action tensor.
            policy_index: The index of the actor whose action was selected. Note that for averaging methods it uses the actor that was closest to the average.
        """
        best_actor_idx: int = -1 # Initialize with a default

        if self.aggregation_method in ["elementwise", "weighted_elementwise"]:
            selected_action: torch.Tensor # Declare type hint
            if self.aggregation_method == "elementwise":
                # Element-wise average of actions
                selected_action = torch.mean(torch.stack(actions), dim=0)
            elif self.aggregation_method == "weighted_elementwise":
                # Weighted average of actions based on Q-values
                # Ensure values are positive for softmax weighting, add small epsilon for stability
                value_stack = torch.stack(action_values)
                shifted_values = value_stack - value_stack.min() + 1e-6
                weights = F.softmax(shifted_values, dim=0)
                selected_action = torch.sum(
                    torch.stack(actions) * weights, dim=0
                )


            normalized_actions = [F.normalize(action, p=2, dim=-1) for action in actions]
            normalized_selection = F.normalize(selected_action, p=2, dim=-1)
            similarities = [torch.sum(action * normalized_selection, dim=-1) for action in normalized_actions]
            # Find the index of the maximum similarity
            best_actor_idx = int(torch.argmax(torch.stack(similarities)).item())


        elif self.aggregation_method == "max_q":
            best_actor_idx = int(torch.argmax(
                torch.stack(action_values, dim=0)
            ).item())
            selected_action = actions[best_actor_idx]
        else:
            raise ValueError(
                f"Unknown aggregation method: {self.aggregation_method}. Use 'elementwise', 'weighted_elementwise', or 'max_q'."
            )


        return selected_action, best_actor_idx

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

            state_batch = torch.FloatTensor(state_np).to(self.computation_device)
            action_batch = torch.FloatTensor(action_np).to(self.computation_device)
            reward_batch = torch.FloatTensor(reward_np).to(self.computation_device).unsqueeze(1)
            next_state_batch = torch.FloatTensor(next_state_np).to(self.computation_device)
            done_batch = torch.FloatTensor(done_np).to(self.computation_device).unsqueeze(1)
            policy_indices = torch.tensor(policy_indices_np, dtype=torch.long).to(
                self.computation_device
            )

            # --- Critic Update ---
            with torch.no_grad():
                all_next_actions = []
                all_next_log_probs = []
                for actor in self.actors:
                    next_action_batch, next_log_prob_batch, _ = actor.sample(next_state_batch)
                    all_next_actions.append(next_action_batch)
                    all_next_log_probs.append(next_log_prob_batch)

                stacked_next_actions = torch.stack(all_next_actions)
                stacked_next_log_probs = torch.stack(all_next_log_probs)

                # Handle cases where policy_index is -1 (random actions during start_steps)
                # Currently handles it by assigning random action index for the actions
                random_action_mask = (policy_indices == -1)
                policy_indices_for_gather = policy_indices.clone()
                policy_indices_for_gather[random_action_mask] = torch.randint(
                    0, self.num_actors, (random_action_mask.sum(),), device=self.computation_device
                )

                idx_expanded_log_prob = policy_indices_for_gather.view(1, self.batch_size, 1).expand(-1, -1, stacked_next_log_probs.shape[-1])
                idx_expanded_action = policy_indices_for_gather.view(1, self.batch_size, 1).expand(-1, -1, stacked_next_actions.shape[-1])

                next_state_action = torch.gather(stacked_next_actions, 0, idx_expanded_action).squeeze(0)
                next_state_log_pi = torch.gather(stacked_next_log_probs, 0, idx_expanded_log_prob).squeeze(0)

                # Set log_pi to 0 for transitions where the original action was random
                # This effectively removes the entropy bonus for these steps in the target Q calculation
                next_state_log_pi[random_action_mask.unsqueeze(1)] = 0.0

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

            alpha_loss_total = torch.tensor(0.0, device=self.computation_device)
            for i, (actor, optimizer) in enumerate(
                zip(self.actors, self.policy_optimizers)
            ):
                pi_actions, pi_log_pi, _ = actor.sample(state_batch)

                qf1_pi = self.critic1(state_batch, pi_actions)
                qf2_pi = self.critic2(state_batch, pi_actions)
                min_qf_pi = torch.min(qf1_pi, qf2_pi)
                policy_loss_q_term = (self.alpha * pi_log_pi - min_qf_pi).mean()

                kl_div_term = torch.tensor(0.0, device=self.computation_device)
                if self.num_actors > 1 and self.alpha_div > 0:
                    current_dist = actor.get_distribution(state_batch)
                    kl_sum = torch.tensor(0.0, device=self.computation_device)
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

        torch.save(save_dict, path)

    def load_model(self, path: str) -> None:
        # Specify map_location type
        checkpoint = torch.load(f"{path}_snac.pt", map_location=self.computation_device)
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
                checkpoint["log_alpha"].to(self.computation_device).requires_grad_(True)
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
