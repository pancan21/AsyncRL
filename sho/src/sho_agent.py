from dataclasses import replace
from functools import partial
from typing import cast

import flashbax as fbx
import jax
import optax as opt

from flashbax.buffers.flat_buffer import (
    ExperiencePair,
    TrajectoryBuffer,
    TrajectoryBufferState,
)
from flashbax.buffers.trajectory_buffer import TrajectoryBufferSample
from jax import numpy as np
from jaximal.core import Jaximal, Static
from jaximal.nn import Activation, Linear, Sequential, WeightInitialization
from jaxtyping import Array, Float, PRNGKeyArray


class SHOAgentParameters(Jaximal):
    latent_dimension: Static[int]
    control_dimension: Static[int]
    q: Sequential
    action: Sequential
    target_q: Sequential
    target_action: Sequential

    @staticmethod
    def init_state(
        key: PRNGKeyArray, latent_dimension: int, control_dimension: int
    ) -> 'SHOAgentParameters':
        q_key, action_key = jax.random.split(key)

        q = Sequential.init_state(
            [
                Linear.init_state(
                    latent_dimension + control_dimension,
                    32,
                    bias_initialization=WeightInitialization.GlorotUniform,
                ),
                Activation.init_state(jax.nn.swish),
                Linear.init_state(
                    32, 32, bias_initialization=WeightInitialization.GlorotUniform
                ),
                Activation.init_state(jax.nn.swish),
                Linear.init_state(
                    32, 32, bias_initialization=WeightInitialization.GlorotUniform
                ),
                Activation.init_state(jax.nn.swish),
                Linear.init_state(
                    32, 1, bias_initialization=WeightInitialization.GlorotUniform
                ),
            ]
        )(q_key)

        action = Sequential.init_state(
            [
                Linear.init_state(
                    latent_dimension,
                    32,
                    bias_initialization=WeightInitialization.GlorotUniform,
                ),
                Activation.init_state(jax.nn.swish),
                Linear.init_state(
                    32, 32, bias_initialization=WeightInitialization.GlorotUniform
                ),
                Activation.init_state(jax.nn.swish),
                Linear.init_state(
                    32, 32, bias_initialization=WeightInitialization.GlorotUniform
                ),
                Activation.init_state(jax.nn.swish),
                Linear.init_state(
                    32,
                    control_dimension,
                    bias_initialization=WeightInitialization.GlorotUniform,
                ),
            ]
        )(action_key)

        return SHOAgentParameters(
            latent_dimension=latent_dimension,
            control_dimension=control_dimension,
            q=q,
            action=action,
            target_q=q,
            target_action=action,
        )

    @staticmethod
    def freeze_q(state: 'SHOAgentParameters') -> 'SHOAgentParameters':
        return SHOAgentParameters(
            state.latent_dimension,
            state.control_dimension,
            jax.lax.stop_gradient(state.q),
            state.action,
            state.target_q,
            state.target_action,
        )

    @staticmethod
    def freeze_action(state: 'SHOAgentParameters') -> 'SHOAgentParameters':
        return SHOAgentParameters(
            state.latent_dimension,
            state.control_dimension,
            state.q,
            jax.lax.stop_gradient(state.action),
            state.target_q,
            state.target_action,
        )


class Timestep(Jaximal):
    latent_dimension: Static[int]
    control_dimension: Static[int]
    latent_state: Float[Array, '{control_dimension}']
    dynamics_match: float

    @staticmethod
    def empty(latent_dimension: int, control_dimension: int) -> 'Timestep':
        latent_state = np.zeros((latent_dimension,))
        dynamics_match = 0.0

        return Timestep(
            latent_dimension,
            control_dimension,
            latent_state,
            dynamics_match,
        )


class SHOAgent(Jaximal):
    buffer: Static[
        TrajectoryBuffer[
            Timestep,  # pyright: ignore[reportInvalidTypeArguments]
            TrajectoryBufferState[Timestep],  # pyright: ignore[reportInvalidTypeArguments]
            TrajectoryBufferSample[ExperiencePair[Timestep]],  # pyright: ignore[reportInvalidTypeArguments]
        ]
    ]
    optimizer: Static[opt.GradientTransformation]
    buffer_state: TrajectoryBufferState[Timestep]  # pyright: ignore[reportInvalidTypeArguments]
    opt_state: opt.OptState
    agent_params: SHOAgentParameters
    gamma: float
    step_count: int
    key: PRNGKeyArray

    @staticmethod
    def init_state(
        key: PRNGKeyArray,
        latent_dimension: int,
        control_dimension: int,
        gamma: float,
    ) -> 'SHOAgent':
        key, agent_key = jax.random.split(key)
        agent_params = SHOAgentParameters.init_state(
            agent_key,
            latent_dimension,
            control_dimension,
        )

        buffer = fbx.make_flat_buffer(
            1024,
            16,
            64,
            add_sequences=False,
            add_batch_size=None,
        )

        empty_timestep = Timestep.empty(latent_dimension, control_dimension)
        buffer_state = buffer.init(empty_timestep)

        optimizer = opt.adam(1e-3)
        opt_state = optimizer.init(cast(opt.Params, agent_params))

        return SHOAgent(
            buffer, optimizer, buffer_state, opt_state, agent_params, gamma, 0, key
        )

    @staticmethod
    @jax.jit
    def update(state: 'SHOAgent', sample_key: PRNGKeyArray):
        sample = state.buffer.sample(state.buffer_state, sample_key).experience

        grad: SHOAgentParameters = jax.lax.cond(
            state.step_count % 3 == 0,
            jax.grad(
                lambda agent_params: bellman_loss(
                    SHOAgentParameters.freeze_action(agent_params),
                    sample,
                    state.gamma,
                ).mean()
            ),
            jax.grad(
                lambda agent_params: bellman_loss(
                    SHOAgentParameters.freeze_q(agent_params),
                    sample,
                    state.gamma,
                ).mean()
            ),
            state.agent_params,
        )

        updates, opt_state = state.optimizer.update(
            cast(opt.Updates, grad),
            state.opt_state,
            cast(opt.Params, state.agent_params),
        )
        state = replace(state, opt_state=opt_state)
        state = replace(
            state,
            agent_params=cast(
                SHOAgentParameters,
                opt.apply_updates(cast(opt.Params, state.agent_params), updates),
            ),
        )

        return state

    @staticmethod
    @jax.jit
    def step(
        state: 'SHOAgent',
        curr_system_state: Float[Array, '{agent_params.latent_dimension}'],
        dynamics_match: float,
    ) -> tuple['SHOAgent', Float[Array, '{agent_params.control_dimension}']]:
        state_key, key = jax.random.split(state.key)
        state = replace(state, key=state_key)
        sample_key, key = jax.random.split(key)

        state = jax.lax.cond(
            state.buffer.can_sample(state.buffer_state),
            SHOAgent.update,
            lambda state, _: state,
            # *args
            state,
            sample_key,
        )

        state = replace(
            state,
            buffer_state=state.buffer.add(
                state.buffer_state,
                Timestep(
                    state.agent_params.latent_dimension,
                    state.agent_params.control_dimension,
                    curr_system_state,
                    dynamics_match,
                ),
            ),
        )

        state = replace(state, step_count=state.step_count + 1)
        return state, state.agent_params.target_action(curr_system_state)


@partial(jax.vmap, in_axes=(None, 0, None))
def bellman_loss(
    agent_params: SHOAgentParameters,
    experience: ExperiencePair[Timestep],  # pyright: ignore[reportInvalidTypeArguments]
    gamma: float,
):
    q = agent_params.q
    target_q = jax.lax.stop_gradient(agent_params.target_q)

    action = agent_params.action
    target_action = jax.lax.stop_gradient(agent_params.target_action)

    curr, next = experience

    reward = (curr.dynamics_match + next.dynamics_match) / 2.0

    curr_action = action(curr.latent_state)
    curr_value = q(np.concat([curr.latent_state, curr_action]))
    next_action = target_action(next.latent_state)
    pred_next_value = target_q(np.concat([next.latent_state, next_action]))

    return np.square(curr_value - (reward + gamma * pred_next_value)).mean()
