import imageio
import numpy as np


def write_movie(history):

    with imageio.get_writer('evol.gif', mode='I') as writer:
        for hist in history:

            min_value = hist.min()
            hist -= min_value
            max_value = hist.max()
            min_value = 0.0

            step = (max_value - min_value) / 255

            hist = np.array([[c // step for c in x] * 3 for x in hist.reshape(hist.shape[0] * hist.shape[1], 3)]).reshape((hist.shape[0] * 3, hist.shape[1] * 3, 3))

            writer.append_data(hist)