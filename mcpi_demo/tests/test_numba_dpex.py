from math import fabs
import dpnp as np

from mcpi_demo.impl.impl_numba_dpex import monte_carlo_pi_batch


def test_numpy():
    batch_size = 1000
    np.random.seed(7777777)
    a = np.random.random(size=BATCH_SIZE)
    b = np.random.random(size=BATCH_SIZE)
    pi = monte_carlo_pi_batch(a, b, batch_size)
    assert fabs(pi-3.14) <= 0.1
