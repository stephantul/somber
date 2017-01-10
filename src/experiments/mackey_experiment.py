import numpy as np
import logging

# from matplotlib import pyplot as plt
from merging import Merging, MergingGas
from recurrent import Recurrent
from som import Som
from utils import MultiPlexer


from src.experiments.mackey import mackey_glass

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    glass = mackey_glass(n_samples=1, sample_len=10000, seed=44)[0]

    g_plex = MultiPlexer(glass, 100)

    # s = Som((10, 10), 1, 1.0)
    # s.train(g_plex, num_effective_epochs=100)
    # err_1 = s.quant_error(glass)

    m = Merging((10, 10), 1, 0.03, alpha=0.001, beta=0.75)
    m.train(g_plex, num_effective_epochs=1000, use_entropy=True)
    err_2 = m.quant_error(glass)

    '''plt.plot(np.arange(len(glass[0][:1000])), glass[0][:1000])

    p_2 = s.predict(glass[0][:1000])
    plt.plot(np.arange(len(glass[0][:1000])), s.weights[p_2], color='red')

    p = m.predict(glass[0][:1000])
    plt.plot(np.arange(len(glass[0][:1000])), m.weights[p], color='green')

    plt.show()'''