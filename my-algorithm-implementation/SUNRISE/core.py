"""Core functions of the SUNRISE algorithm."""

import gymnasium as gym
import numpy as np
import tensorflow as tf


EPS = 1e-8

LOG_STD_MAX = 2
LOG_STD_MIN = -20


def heuristic_target_entropy(action_space):
    # pylint: disable=line-too-long
    """Copied from https://github.com/rail-berkeley/softlearning/blob/master/softlearning/algorithms/sac.py"""
    if isinstance(action_space, gym.spaces.Box):  # continuous space
        heuristic_target_entropy = -np.prod(action_space.shape)
    else:
        raise NotImplementedError((type(action_space), action_space))

    return heuristic_target_entropy


class ExpLayer(tf.keras.layers.Layer):
    """Custom Keras layer to apply tf.exp."""
    def call(self, inputs):
        return tf.exp(inputs)


class ClipByValueLayer(tf.keras.layers.Layer):
    """Custom Keras layer to apply tf.clip_by_value."""
    def __init__(self, min_value, max_value, **kwargs):
        super().__init__(**kwargs)
        self.min_value = min_value
        self.max_value = max_value

    def call(self, inputs):
        return tf.clip_by_value(inputs, self.min_value, self.max_value)


class RandomNormalLayer(tf.keras.layers.Layer):
    """Custom Keras layer to apply tf.random.normal."""
    def call(self, inputs):
        # Ensure inputs are cast to int32 and used as shape
        shape = tf.cast(inputs, tf.int32)
        return tf.random.normal(shape)

    def compute_output_shape(self, input_shape):
        return input_shape


class ReduceSumLayer(tf.keras.layers.Layer):
    def __init__(self, axis=1, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        if input_shape[self.axis] is None:
            return input_shape[:self.axis] + input_shape[self.axis+1:]
        return tuple([d for i, d in enumerate(input_shape) if i != self.axis])


class SoftplusLayer(tf.keras.layers.Layer):
    """Custom Keras layer to apply tf.nn.softplus."""
    def call(self, inputs):
        return tf.nn.softplus(inputs)

    def compute_output_shape(self, input_shape):
        return input_shape


def gaussian_likelihood(value, mu, log_std):
    """Calculates value's likelihood under Gaussian pdf."""
    pre_sum = -0.5 * (
        ((value - mu) / (ExpLayer()(log_std) + EPS)) ** 2 +
        2 * log_std + np.log(2 * np.pi)
    )
    return ReduceSumLayer(axis=1)(pre_sum)


def apply_squashing_func(mu, pi, logp_pi):
    """Applies adjustment to mean, pi and log prob.

    This formula is a little bit magic. To get an understanding of where it
    comes from, check out the original SAC paper (arXiv 1801.01290) and look
    in appendix C. This is a more numerically-stable equivalent to Eq 21.
    Try deriving it yourself as a (very difficult) exercise. :)
    """
    logp_pi -= ReduceSumLayer(axis=1)(
        2 * (tf.math.log(2.0) - pi - SoftplusLayer()(-2 * pi))
    )

    # Squash those unbounded actions!
    mu = tf.keras.layers.Lambda(lambda x: tf.tanh(x))(mu)
    pi = tf.keras.layers.Lambda(lambda x: tf.tanh(x))(pi)
    return mu, pi, logp_pi


def mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(size, activation=activation)
        for size in hidden_sizes
    ], name)


def layer_norm_mlp(hidden_sizes, activation, name=None):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(hidden_sizes[0]),
        tf.keras.layers.LayerNormalization(),
        tf.keras.layers.Activation(tf.nn.tanh),
        mlp(hidden_sizes[1:], activation)
    ], name)


class StackLayer(tf.keras.layers.Layer):
    """Custom Keras layer to apply tf.stack."""
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.stack(inputs, axis=self.axis)

class UnstackLayer(tf.keras.layers.Layer):
    """Custom Keras layer to apply tf.unstack."""
    def __init__(self, axis, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis

    def call(self, inputs):
        return tf.unstack(inputs, axis=self.axis)

    def compute_output_shape(self, input_shape):
        return [input_shape[:self.axis] + input_shape[self.axis+1:]]


class MLPActorCriticFactory:
    """Factory of MLP stochastic actors and critics.

    Args:
        observation_space (gym.spaces.Box): A continuous observation space
          specification.
        action_space (gym.spaces.Box): A continuous action space
          specification.
        hidden_sizes (list): A hidden layers shape specification.
        activation (tf.function): A hidden layers activations specification.
        ac_number (int): Number of the actor-critic models in the ensemble.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        hidden_sizes,
        activation,
        ac_number,
    ):
        self._obs_dim = observation_space.shape[0]
        self._act_dim = action_space.shape[0]
        self._act_scale = action_space.high[0]
        self._hidden_sizes = hidden_sizes
        self._activation = activation
        self._ac_number = ac_number

    def _make_actor(self):
        """Constructs and returns the actor model (tf.keras.Model)."""
        obs_input = tf.keras.Input(shape=(self._obs_dim,))
        body = mlp(self._hidden_sizes, self._activation)(obs_input)
        mu = tf.keras.layers.Dense(self._act_dim)(body)
        log_std = tf.keras.layers.Dense(self._act_dim)(body)

        # Wrap tf.clip_by_value and tf.exp in custom layers
        log_std = ClipByValueLayer(LOG_STD_MIN, LOG_STD_MAX)(log_std)
        std = ExpLayer()(log_std)

        # Compute shape using batch size and action dimension
        random_noise = tf.keras.layers.Lambda(
            lambda x: tf.random.normal(shape=(tf.shape(x)[0], self._act_dim))
        )(mu)

        print(f"Shapes of the items in the computations: mu={mu.shape}, noise={random_noise.shape}, std={std.shape}")

        pi = mu + random_noise * std

        logp_pi = gaussian_likelihood(pi, mu, log_std)

        mu, pi, logp_pi = apply_squashing_func(mu, pi, logp_pi)

        # Put the actions in the limit.
        mu = mu * self._act_scale
        pi = pi * self._act_scale

        return tf.keras.Model(inputs=obs_input, outputs=[mu, pi, logp_pi])

    def make_actor(self):
        """Constructs and returns the ensemble of actor models."""
        obs_inputs = tf.keras.Input(shape=(None, self._obs_dim),
                                    batch_size=self._ac_number)
        outputs = []
        for i in range(self._ac_number):
            obs_input = tf.keras.layers.Lambda(lambda x: x[i])(obs_inputs)
            model = self._make_actor()
            outputs.append(model(obs_input))
        mus, pis, logp_pis = zip(*outputs)
        return tf.keras.Model(inputs=obs_inputs, outputs=[
            StackLayer(axis=0)(mus),
            StackLayer(axis=0)(pis),
            StackLayer(axis=0)(logp_pis),
        ])

    def _make_critic(self):
        """Constructs and returns the critic model (tf.keras.Model)."""
        obs_input = tf.keras.Input(shape=(self._obs_dim,))
        act_input = tf.keras.Input(shape=(self._act_dim,))

        concat_input = tf.keras.layers.Concatenate(
            axis=-1)([obs_input, act_input])

        q = tf.keras.Sequential([
            mlp(self._hidden_sizes, self._activation),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Reshape([])  # Very important to squeeze values!
        ])(concat_input)

        return tf.keras.Model(inputs=[obs_input, act_input], outputs=q)

    def make_critic(self):
        """Constructs and returns the ensemble of critic models."""
        obs_inputs = tf.keras.Input(shape=(None, self._obs_dim),
                                    batch_size=self._ac_number)
        act_inputs = tf.keras.Input(shape=(None, self._act_dim),
                                    batch_size=self._ac_number)
        qs = []
        for obs_input, act_input in zip(UnstackLayer(axis=0)(obs_inputs),
                                        UnstackLayer(axis=0)(act_inputs)):
            model = self._make_critic()
            q = model([obs_input, act_input])
            qs.append(q)
        return tf.keras.Model(inputs=[obs_inputs, act_inputs],
                              outputs=StackLayer(axis=0)(qs))
