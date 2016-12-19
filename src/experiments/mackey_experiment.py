import numpy as np
# from matplotlib import pyplot as plt
from rsom import RSom
from som import Som
from utils import MultiPlexer

from src.experiments.mackey import mackey_glass

if __name__ == "__main__":

    glass = mackey_glass(n_samples=1, sample_len=10000, seed=44)

    g_plex = MultiPlexer(glass[0], 100)

    s = Som(10, 10, 1, [1.0])
    s.train(g_plex, num_effective_epochs=100)

    m = RSom(10, 10, 1, [1.0], alpha=0.5)
    m.train(g_plex, num_effective_epochs=100)

    '''plt.plot(np.arange(len(glass[0][:1000])), glass[0][:1000])

    p_2 = s.predict(glass[0][:1000])
    plt.plot(np.arange(len(glass[0][:1000])), s.weights[p_2], color='red')

    p = m.predict(glass[0][:1000])
    plt.plot(np.arange(len(glass[0][:1000])), m.weights[p], color='green')

    plt.show()'''