import numpy as np

from collections import deque


def mackey_glass(sample_len=1000, tau=17, seed=None, n_samples=1):
    """
    mackey_glass(sample_len=1000, tau=17, seed = None, n_samples = 1) -> input
    Generate the Mackey Glass time-series.
    :param sample_len: length of the time-series in timesteps. Default is 1000.
    :param tau: delay of the MG - system. Commonly used values are tau=17 (mild
          chaos) and tau=30 (moderate chaos). Default is 17.
    :param seed: to seed the random generator, can be used to generate the same
          timeseries at each invocation.
    :param n_samples : number of samples to generate
    """
    delta_t = 10
    history_len = tau * delta_t
    # Initial conditions for the history of the system
    timeseries = 1.2

    if seed is not None:
        np.random.seed(seed)

    samples = []

    for _ in range(n_samples):
        history = deque(1.2 * np.ones(history_len) + 0.2 * \
                                    (np.random.rand(history_len) - 0.5))
        # Preallocate the array for the time-series
        inp = np.zeros((sample_len,1))

        for timestep in range(sample_len):
            for _ in range(delta_t):
                xtau = history.popleft()
                history.append(timeseries)
                timeseries = history[-1] + (0.2 * xtau / (1.0 + xtau ** 10) -
                                            0.1 * history[-1]) / delta_t
            inp[timestep] = timeseries

        # Squash timeseries through tanh
        inp = np.tanh(inp - 1)
        samples.append(inp)

    return np.array(samples)