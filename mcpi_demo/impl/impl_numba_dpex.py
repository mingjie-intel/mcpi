import dpnp as np
import numba
from numba_dpex import dpjit


@dpjit(parallel=True)
def monte_carlo_pi_batch(a, b, batch_size):
   # x = np.random.random(batch_size)
   # y = np.random.random(batch_size)
    acc = 0.0
    for i in numba.prange(batch_size):
        if a[i] * a[i] + b[i] * b[i] <= 1.0:
            acc += 1.0
    return 4.0 * acc / batch_size
