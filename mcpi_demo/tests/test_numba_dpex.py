from math import fabs
import dpnp as np

from mcpi_demo.impl.impl_numba_dpex import monte_carlo_pi_batch


def test_numpy():
    batch_size = 1000
    np.random.seed(7777777)
    pi = monte_carlo_pi_batch(batch_size)
    assert fabs(pi-3.14) <= 0.1
