import jax

from jax import numpy as np
from sho_agent import SHOAgent

key = jax.random.key(0)
agent = SHOAgent.init_state(key, 2, 1, 0.5)


SHOAgent.step(agent, np.array([1.0, 1.0]), 0.0)
