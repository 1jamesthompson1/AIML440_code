import torch
import torch.nn as nn
import numpy as np

from typing import List, Tuple, Dict, Deque, NamedTuple, Optional, Any, Union
import random
from collections import deque
from torch.distributions import Normal, kl_divergence
import torch.nn.functional as F

# from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic

import gymnasium as gym
from rich.table import Table
from rich.console import Console

# --- Replay Buffer ---
class Transition(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    mask: np.ndarray


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
        mask: np.ndarray,
    ) -> None:
        """Save a transition"""
        # Check the format of the inputs

        if not isinstance(state, np.ndarray):
            raise ValueError("State must be a numpy array")
        elif state.ndim != 1:
            raise ValueError(f"State must be a 1D numpy array, instead recieved {state.shape}")

        if not isinstance(action, np.ndarray):
            raise ValueError("Action must be a numpy array")
        elif action.ndim != 1:
            print(action)
            raise ValueError(f"Action must be a 1D numpy array, instead recieved {action.shape}")

        self.memory.append(
            Transition(state, action, reward, next_state, done, mask)
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
        self.init_hidden_layers(num_inputs + num_actions, arch) # TODO: Check to see if the intilization of layers is significantly differnet from original
        self.actor_aware_critic = False

        self.output = nn.Linear(self.hidden_layers[-1].out_features, 1, device=self.computation_device)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = self.hidden_layer_fp(torch.cat([state, action], 1))
        x = self.output(x)
        return x
    

    
class ActorNetwork(Network):
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
        super(ActorNetwork, self).__init__(computation_device)
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
        # TODO: Reconcile this with the sunrise version
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean_action

class SAC_agent:
    def __init__(self, obs_dim, action_space, critic_arch, actor_arch, computation_device, entropy_lr, actor_lr, critic_lr, discount, soft_target_tau, reward_scale, expl_gamma, feedback_type):
        # TODO: implement feedback_type

        self.discount = discount
        self.soft_target_tau = soft_target_tau
        self.expl_gamma = expl_gamma
        self.reward_scale = reward_scale
        self.target_entropy = -np.prod(action_space.shape).item()
        self.log_alpha = torch.zeros(1, requires_grad=True, device=computation_device)
        self.alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=entropy_lr
        )
        self.feedback_type = feedback_type
        if feedback_type not in [1,3]:
            raise ValueError("feedback_type must be 1 or 3 as that is all that is supported for the SAC agent.")
        
        self.computation_device = computation_device

        action_dim = action_space.shape[0]

        self.qf1 = CriticNetwork(
            computation_device=self.computation_device,
            num_inputs=obs_dim,
            num_actions=action_dim,
            arch=critic_arch,
        )
        self.qf2 = CriticNetwork(
            computation_device=self.computation_device,
            num_inputs=obs_dim,
            num_actions=action_dim,
            arch=critic_arch,
        )
        self.target_qf1 = CriticNetwork(
            computation_device=self.computation_device,
            num_inputs=obs_dim,
            num_actions=action_dim,
            arch=critic_arch,
        )
        self.target_qf2 = CriticNetwork(
            computation_device=self.computation_device,
            num_inputs=obs_dim,
            num_actions=action_dim,
            arch=critic_arch,
        )
        self.policy = ActorNetwork(
            num_inputs=obs_dim,
            num_actions=action_dim,
            arch=actor_arch,
            action_space=action_space,
            computation_device=self.computation_device,
        )

        self.qf1_optimizer = torch.optim.Adam(
            self.qf1.parameters(), lr=critic_lr
        )
        self.qf2_optimizer = torch.optim.Adam(
            self.qf2.parameters(), lr=critic_lr
        )
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=actor_lr
        )



    # TODO: Figure out what this does
    def corrective_feedback(self, obs, update_type):
        """
        Compute the standard deviation of Q-values for a single SAC agent.
        Args:
            obs: Observation tensor.
            update_type: 0 for actor, 1 for critic.
        Returns:
            std_Q: Standard deviation of Q-values.
        """

        with torch.no_grad():
            policy_action, _, _ = self.policy.sample(obs)
            if update_type == 0:  # actor
                target_Q1 = self.qf1(obs, policy_action)
                target_Q2 = self.qf2(obs, policy_action)
            else:  # critic
                target_Q1 = self.target_qf1(obs, policy_action)
                target_Q2 = self.target_qf2(obs, policy_action)

            mean_Q = 0.5 * (target_Q1 + target_Q2)
            var_Q = 0.5 * ((target_Q1 - mean_Q) ** 2 + (target_Q2 - mean_Q) ** 2)
            std_Q = torch.sqrt(var_Q).detach()

        return std_Q

    def update_parameters(
        self,
        obs: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_obs: torch.Tensor,
        dones: torch.Tensor,
        mask: torch.Tensor,
        weight_actor_Q: torch.Tensor,
        weight_target_Q: torch.Tensor,
        std_Q: torch.Tensor,
        target_update: bool,
    ) -> None:
        


        """
        Policy and Alpha Loss
        """
        new_obs_actions, log_prob, _ = self.policy.sample(
            obs
        )

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()) * mask
        alpha_loss = alpha_loss.sum() / (mask.sum() + 1)
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        alpha = self.log_alpha.exp()

        q_new_actions = torch.min(
            self.qf1(obs, new_obs_actions),
            self.qf2(obs, new_obs_actions),
        )


        policy_loss = (alpha*log_prob - q_new_actions - self.expl_gamma * std_Q) * mask * weight_actor_Q.detach()
        policy_loss = policy_loss.sum() / (mask.sum() + 1)

        """
        QF Loss
        """
        q1_pred = self.qf1(obs, actions)
        q2_pred = self.qf2(obs, actions)
        
        # Make sure policy accounts for squashing functions like tanh correctly!
        new_next_actions, new_log_pi, _ = self.policy.sample(
            next_obs
        )
        target_q_values = torch.min(
            self.target_qf1(next_obs, new_next_actions),
            self.target_qf2(next_obs, new_next_actions),
        ) - alpha * new_log_pi
        
        q_target = self.reward_scale * rewards + (1. - dones) * self.discount * target_q_values
        qf1_loss = nn.MSELoss(reduce=False)(q1_pred, q_target.detach()) * mask * (weight_target_Q.detach())
        qf2_loss = nn.MSELoss(reduce=False)(q2_pred, q_target.detach()) * mask * (weight_target_Q.detach())
        qf1_loss = qf1_loss.sum() / (mask.sum() + 1)
        qf2_loss = qf2_loss.sum() / (mask.sum() + 1)
        
        """
        Update networks
        """
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        """
        Soft Updates
        """
        if target_update:
            self._soft_update(
                self.qf1, self.target_qf1
            )
            self._soft_update(
                self.qf2, self.target_qf2
            )

    def _soft_update(self, target: nn.Module, source: nn.Module) -> None:
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.soft_target_tau * param.data + (1.0 - self.soft_target_tau) * target_param.data)

    def get_action(self, obs, eval=False):
        sample = self.policy.sample(obs)

        if eval:
            return sample[2] # Mean of normal distribution
        else:
            return sample[0] # Sample from Normal distrbution

class DSUNRISE:
    def __init__(
        self,
        env: gym.Env,
        num_ensemble: int,
        critic_arch: List[int],
        actor_arch: List[int],
        actor_lr: float,
        critic_lr: float,
        entropy_lr: float,
        reward_scale: float,
        expl_gamma: float,
        temperature: float,
        temperature_act: float,
        discount: float,
        soft_target_tau: float,
        target_update_period: int,
        auto_entropy_tuning: bool,
        feedback_type: int,
        max_replay_buffer_size: int,
        batch_size: int,
        computation_device: torch.device
    ):
        self.env = env
        self.num_ensemble = num_ensemble
        self.temperature=temperature
        self.temperature_act=temperature_act
        self.target_update_period = target_update_period
        self.max_replay_buffer_size = max_replay_buffer_size
        self.batch_size = batch_size
        self.auto_entropy_tuning = auto_entropy_tuning
        self.computation_device = computation_device
        self.feedback_type = feedback_type
    
        self.learners = [
            SAC_agent(env.observation_space.shape[0],
                      env.action_space,
                      critic_arch,
                      actor_arch,
                      computation_device,
                      entropy_lr=entropy_lr,
                      actor_lr=actor_lr,
                      critic_lr=critic_lr,
                      discount=discount,
                      soft_target_tau=soft_target_tau,
                      reward_scale=reward_scale,
                      feedback_type=feedback_type,
                      expl_gamma=expl_gamma
            )
            for _ in range(num_ensemble)
        ]

        self.memory = ReplayBuffer(capacity=max_replay_buffer_size)

        self.updates = 0

        table = Table(title="DSUNRISE Initialization Parameters")
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        params = {
            "num_ensemble": self.num_ensemble,
            "critic_arch": critic_arch,
            "actor_arch": actor_arch,
            "actor_lr": actor_lr,
            "critic_lr": critic_lr,
            "entropy_lr": entropy_lr,
            "reward_scale": reward_scale,
            "expl_gamma": expl_gamma,
            "temperature": temperature,
            "temperature_act": temperature_act,
            "discount": discount,
            "soft_target_tau": soft_target_tau,
            "target_update_period": target_update_period,
            "auto_entropy_tuning": auto_entropy_tuning,
            "feedback_type": feedback_type,
            "max_replay_buffer_size": max_replay_buffer_size,
            "batch_size": batch_size,
            "computation_device": computation_device,
        }

        for k, v in params.items():
            table.add_row(str(k), str(v))

        console = Console()
        console.print(table)
    
    def update_parameters(self):

        if len(self.memory) < self.batch_size:
            print("WARNING: Not enough samples to update the parameters")
            return
        
        batch = self.memory.sample(self.batch_size)
        obs = torch.tensor(np.array([t.state for t in batch]), dtype=torch.float32, device=self.computation_device)
        actions = torch.tensor(np.array([t.action for t in batch]), dtype=torch.float32, device=self.computation_device).squeeze(1)
        rewards = torch.tensor(np.array([t.reward for t in batch]), dtype=torch.float32, device=self.computation_device).unsqueeze(1)
        next_obs = torch.tensor(np.array([t.next_state for t in batch]), dtype=torch.float32, device=self.computation_device)
        dones = torch.tensor(np.array([t.done for t in batch]), dtype=torch.float32, device=self.computation_device).unsqueeze(1)
        masks = torch.tensor(np.array([t.mask for t in batch]), dtype=torch.float32, device=self.computation_device)

        std_Q_actor_list = [agent.corrective_feedback(obs, 0) for agent in self.learners]
        std_Q_critic_list = [agent.corrective_feedback(obs, 1) for agent in self.learners]

        for i, learner in enumerate(self.learners):
            mask = masks[:,i].reshape(-1, 1)

            if self.feedback_type == 0 or self.feedback_type == 2:
                std_Q = std_Q_actor_list[i]
            else:
                std_Q = std_Q_actor_list[0]

            if self.feedback_type == 1 or self.feedback_type == 0:
                weight_actor_Q = torch.sigmoid(-std_Q*self.temperature_act) + 0.5
            else:
                weight_actor_Q = 2*torch.sigmoid(-std_Q*self.temperature_act)

            if self.feedback_type == 0 or self.feedback_type == 2:
                if self.feedback_type == 0:
                    weight_target_Q = torch.sigmoid(-std_Q_critic_list[i]*self.temperature) + 0.5
                else:
                    weight_target_Q = 2*torch.sigmoid(-std_Q_critic_list[i]*self.temperature)
            else:
                if self.feedback_type == 1:
                    weight_target_Q = torch.sigmoid(-std_Q_critic_list[0]*self.temperature) + 0.5
                else:
                    weight_target_Q = 2*torch.sigmoid(-std_Q_critic_list[0]*self.temperature)

            learner.update_parameters(
                obs=obs,
                actions=actions,
                rewards=rewards,
                next_obs=next_obs,
                dones = dones,
                target_update=self.updates % self.target_update_period == 0,
                mask=mask,
                weight_target_Q=weight_target_Q,
                weight_actor_Q=weight_actor_Q,
                std_Q=std_Q
            )
                
            """
            Statistics for log
            """
            # tot_qf1_loss += qf1_loss * (1/self.num_ensemble)
            # tot_qf2_loss += qf2_loss * (1/self.num_ensemble)
            # tot_q1_pred += q1_pred * (1/self.num_ensemble)
            # tot_q2_pred += q2_pred * (1/self.num_ensemble)
            # tot_q_target += q_target * (1/self.num_ensemble)
            # tot_log_pi += log_pi * (1/self.num_ensemble)
            # tot_policy_mean += policy_mean * (1/self.num_ensemble)
            # tot_policy_log_std += policy_log_std * (1/self.num_ensemble)
            # tot_alpha += alpha.item() * (1/self.num_ensemble)
            # tot_alpha_loss += alpha_loss.item()
            # tot_policy_loss = (log_pi - q_new_actions).mean() * (1/self.num_ensemble)

        """
        Save some statistics for eval
        """
        # if self._need_to_update_eval_statistics:
        #     self._need_to_update_eval_statistics = False
        #     """
        #     Eval should set this to None.
        #     This way, these statistics are only computed for one batch.
        #     """
        #     self.eval_statistics['QF1 Loss'] = np.mean(ptu.get_numpy(tot_qf1_loss))
        #     self.eval_statistics['QF2 Loss'] = np.mean(ptu.get_numpy(tot_qf2_loss))
        #     self.eval_statistics['Policy Loss'] = np.mean(ptu.get_numpy(
        #         tot_policy_loss
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Q1 Predictions',
        #         ptu.get_numpy(tot_q1_pred),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Q2 Predictions',
        #         ptu.get_numpy(tot_q2_pred),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Q Targets',
        #         ptu.get_numpy(tot_q_target),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Log Pis',
        #         ptu.get_numpy(tot_log_pi),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Policy mu',
        #         ptu.get_numpy(tot_policy_mean),
        #     ))
        #     self.eval_statistics.update(create_stats_ordered_dict(
        #         'Policy log std',
        #         ptu.get_numpy(tot_policy_log_std),
        #     ))
        #     if self.use_automatic_entropy_tuning:
        #         self.eval_statistics['Alpha'] = tot_alpha
        #         self.eval_statistics['Alpha Loss'] = tot_alpha_loss
                
        self.updates += 1


    def get_ucb_std(self, obs, policy_action, inference_type, learner) -> torch.Tensor:
        obs = obs.reshape(1,-1)
        policy_action = policy_action.reshape(1,-1)
        
        if self.feedback_type == 0 or self.feedback_type==2:
            with torch.no_grad():
                target_Q1 = learner.qf1(obs, policy_action)
                target_Q2 = learner.qf2(obs, policy_action)
            mean_Q = 0.5*(target_Q1.detach() + target_Q2.detach())
            var_Q = 0.5*((target_Q1.detach() - mean_Q)**2 + (target_Q2.detach() - mean_Q)**2)
            ucb_score = mean_Q + inference_type * torch.sqrt(var_Q).detach()

        elif self.feedback_type == 1 or self.feedback_type==3:
            mean_Q, var_Q = 0.0, None
            L_target_Q = []
            for learner in self.learners:
                with torch.no_grad():
                    target_Q1 = learner.qf1(obs, policy_action)
                    target_Q2 = learner.qf2(obs, policy_action)
                    L_target_Q.append(target_Q1)
                    L_target_Q.append(target_Q2)
                    mean_Q += 0.5*(target_Q1 + target_Q2) / self.num_ensemble

            temp_count = 0
            for target_Q in L_target_Q:
                if temp_count == 0:
                    var_Q = (target_Q.detach() - mean_Q)**2
                else:
                    var_Q += (target_Q.detach() - mean_Q)**2
                temp_count += 1
            var_Q = var_Q / temp_count
            ucb_score = mean_Q + inference_type * torch.sqrt(var_Q).detach()
            
        return ucb_score

    def select_action(self, obs, learner_id=None, evaluate=False):
        """
        Get action from the policy network.
        Args:
            obs: Current observation tensor.
            learner_id: the ID of the learner to use. If None (default) then it will use UCB exploration
            eval: Whether to use exploration or not.
        Returns:
            action: Action tensor.
        """
        obs_tensor = torch.FloatTensor(obs).to(self.computation_device).unsqueeze(0)
        
        if learner_id is not None:
            learner = self.learners[learner_id]
            _a, _ = learner.get_action(obs_tensor)
            return _a

        actions, ucb_scores = [], []
        for i, learner in enumerate(self.learners):
            _a = learner.get_action(obs_tensor)
            ucb_score = self.get_ucb_std(obs_tensor, _a, 0, learner)

            actions.append(_a)
            ucb_scores.append(ucb_score)

        stacked_actions = torch.stack(actions, dim=0).detach().cpu()
        ucb_scores = torch.stack(ucb_scores, dim=0).detach().cpu()
        
        if evaluate:
            a = stacked_actions.mean(dim=0).squeeze(0)
        else:
            a = stacked_actions[np.argmax(ucb_scores)].squeeze(0)

        # Set which SAC agents will learn with this transition
        mask = torch.bernoulli(torch.Tensor([0.5]*self.num_ensemble))
        if mask.sum() == 0:
            rand_index = np.random.randint(self.num_ensemble, size=1)
            mask[rand_index] = 1
        mask = mask.numpy()

        return a.numpy(), mask
        _

