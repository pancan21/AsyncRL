import jax
from jax import numpy as np

from jaximal.core import Static, Jaximal
from jaxtyping import Float, Array

class Magic(Jaximal):
    f: Static[int]
    g: Float[Array, "f"]


@jax.jit
def model() -> Magic:
    return Magic(5, np.array([1., 2., 3., 4., 5.]))
