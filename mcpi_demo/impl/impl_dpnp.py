import dpnp as np


def monte_carlo_pi_batch(x, y, batch_size):
#    x = np.random.random(batch_size)
#    y = np.random.random(batch_size)
    acc = np.count_nonzero(x * x + y * y <= np.asarray(1.0))
    return 4.0 * acc / batch_size
