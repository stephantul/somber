import imageio
import numpy as np
from sklearn.decomposition import PCA


def write_movie(history):

    pca = PCA(n_components=3)
    pca.fit(history[-1])

    with imageio.get_writer('evol.gif', mode='I') as writer:
        for hist in history:

            x = pca.transform(hist)

            print(x.shape)

            min_value = x.min()
            x -= min_value
            max_value = x.max()
            min_value = 0.0

            step = (max_value - min_value) / 255

            hist = np.array([[c // step for c in x] * 3 for x in x.reshape(x.shape[0] * x.shape[1], 3)]).reshape((x.shape[0] * 3, x.shape[1] * 3, 3))

            writer.append_data(hist)

