import numpy as np

from somber.pytorch.sequential import Recursive

X = np.array(
     [[0., 0., 0.],
      [0., 0., 1.],
      [0., 0., 0.5],
      [0.125, 0.529, 1.0],
      [0.33, 0.4, 0.67],
      [0.6, 0.5, 1.0],
      [0., 1., 0.],
      [1., 0., 0.],
      [0., 1., 1.],
      [1., 0., 1.],
      [1., 1., 0.],
      [1., 1., 1.],
      [.33, .33, .33],
      [.5, .5, .5],
      [.66, .66, .66]])

X = np.asarray([X] * 100).reshape(X.shape[0] * 100, 3)
print(X.shape)

color_names = \
    ['black', 'blue', 'darkblue', 'skyblue',
     'greyblue', 'lilac', 'green', 'red',
     'cyan', 'violet', 'yellow', 'white',
     'darkgrey', 'mediumgrey', 'lightgrey']

# initialize
s = Recursive((10, 10), data_dim=1, alpha=3.0, beta=1.0, learning_rate=0.3)

# train
X = np.random.binomial(1, 0.5, 300000)[:, np.newaxis]
s.train(X, num_epochs=100, total_updates=1000, show_progressbar=True, batch_size=1)

s_2 = Recursive((10, 10), data_dim=1, alpha=3.0, beta=1.0, learning_rate=0.3)
s_2.train(X, num_epochs=100, total_updates=1000, show_progressbar=True, batch_size=10)

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
