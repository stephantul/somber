import numpy as np
import logging

from somber.pytorch.sequential import Recursive, Recurrent, Merging
from somber.pytorch.som import Som

def test():

    X = np.random.binomial(1, 0.5, 10000)[:, np.newaxis]
    # initialize
    s = Som((10, 10), data_dim=1, learning_rate=0.3)
    s.train(X, num_epochs=10, total_updates=1000, show_progressbar=True, batch_size=100)
    s.predict(X)
    s.quant_error(X)
    s.map_weights()
    print("Passed SOM")
    # s.invert_projection(X, list(X))

    s = Recursive((10, 10), data_dim=1, learning_rate=0.3, beta=1.0, alpha=3.0)
    s.train(X, num_epochs=10, total_updates=1000, show_progressbar=True, batch_size=100)
    s.predict(X)
    s.quant_error(X)
    s.map_weights()
    print("Passed recursive")
    # s.invert_projection(X, list(X))

    s = Merging((10, 10), data_dim=1, learning_rate=0.3, alpha=0.02, beta=0.5)
    s.train(X, num_epochs=10, total_updates=1000, show_progressbar=True, batch_size=100)
    s.predict(X)
    s.quant_error(X)
    s.map_weights()
    print("Passed merging")
    # s.invert_projection(X, list(X))

    s = Recurrent((10, 10), data_dim=1, learning_rate=0.3, alpha=0.5)
    s.train(X, num_epochs=10, total_updates=1000, show_progressbar=True, batch_size=100)
    s.predict(X)
    s.quant_error(X)
    s.map_weights()
    print("Passed recurrent")
    # s.invert_projection(X, list(X))

logging.basicConfig(level=logging.INFO)

test()

from somber.batch.sequential import Recursive, Recurrent, Merging
from somber.batch.som import Som

test()



# predict: get the index of each best matching unit.
# predictions = s.predict(X)
# quantization error: how well do the best matching units fit?
# quantization_error = s.quant_error(X)
# inversion: associate each node with the exemplar that fits best.
# inverted = s.invert_projection(X, color_names)
# Mapping: get weights, mapped to the grid points of the SOM
# mapped = s.map_weights()

# import matplotlib.pyplot as plt

# plt.imshow(mapped)
