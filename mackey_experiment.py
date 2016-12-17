import numpy as np
import matplotlib

from mackey import mackey_glass
from msom_no_batch import MSom
from som import Som
from matplotlib import pyplot as plt

if __name__ == "__main__":

    glass = mackey_glass(n_samples=1, sample_len=10000)

    s = Som(10, 10, 1, [1.0])
    s.train(glass.reshape(len(glass[0]), 1), num_epochs=1000)

    m = MSom(10, 10, 1, [0.03], 0.0, 0.75)
    m.train(np.squeeze(glass), batch_size=1, num_epochs=1000)

    plt.scatter(np.arange(len(glass[0])), glass[0])

    p = m.predict(np.squeeze(glass))
    plt.scatter(np.arange(len(glass[0])), m.weights[p], color='green')

    p_2 = s.predict(glass.reshape(len(glass[0]), 1))
    plt.scatter(np.arange(len(glass[0])), s.weights[p_2], color='red')

    plt.show()