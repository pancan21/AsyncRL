import jax

from jax import Array
from jaximal.core import Jaximal, Static
from jaximal.nn import Activation, Linear, Sequential, WeightInitialization
from jaxtyping import Float, PRNGKeyArray


class SHOPredictorParameters(Jaximal):
    delay_depth: Static[int]
    observation_dimension: Static[int]
    latent_dimension: Static[int]

    model: Sequential

    @staticmethod
    def init_state(
        key: PRNGKeyArray,
        delay_depth: int,
        observation_dimension: int,
        latent_dimension: int,
    ) -> 'SHOPredictorParameters':
        model = Sequential.init_state(
            [
                Linear.init_state(
                    delay_depth * observation_dimension,
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
                    latent_dimension,
                    bias_initialization=WeightInitialization.GlorotUniform,
                ),
            ]
        )(key)

        return SHOPredictorParameters(
            delay_depth, observation_dimension, latent_dimension, model
        )


class SHOPredictor(Jaximal):
    predictor_params: SHOPredictorParameters

    @staticmethod
    def init_state(
        key: PRNGKeyArray,
        delay_depth: int,
        observation_dimension: int,
        latent_dimension: int,
    ) -> 'SHOPredictor':
        params = SHOPredictorParameters.init_state(
            key,
            delay_depth,
            observation_dimension,
            latent_dimension,
        )

        return SHOPredictor(params)

    @staticmethod
    @jax.jit
    def step(
        state: 'SHOPredictor',
        curr_system_state: Float[
            Array,
            '{predictor_params.delay_depth} {predictor_params.observation_dimension}',
        ],
    ) -> tuple['SHOPredictor', Float[Array, '{predictor_params.latent_dimension}']]:
        return state, state.predictor_params.model(curr_system_state.ravel())
