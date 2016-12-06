import numpy as np

from som import Som


class Pyramid(object):

    def __init__(self, layers):

        self.layers = layers

        if not self._is_consistent():
            raise ValueError("Layers not consistent, the map_dim of each layer should match the data_dim of the next layer")

        self.learning_rates = [l.learning_rates for l in layers]
        self.scaling_factors = [l.scaling_factor for l in layers]

    def propagate(self, X, num_epochs, batch_size):
        """
        Propagate an input through all layers.

        :param X: An input.
        :return: A vector which resembles the activation of the final layer.
        """

        if not num_epochs:
            return []

        learning_rates = self.learning_rates

        for epoch in range(num_epochs):

            scalers = num_epochs / np.log(self.scaling_factors)
            map_radius = self.scaling_factors * np.exp(-epoch / scalers)

            out = X
            for idx, layer in enumerate(self.layers):

                print(out.shape)
                out = layer.epoch_step(out, learning_rate=learning_rates[idx], map_radius=map_radius[idx], batch_size=batch_size, return_activations=True)

            learning_rates = [x * np.exp(-epoch / num_epochs) for x in self.learning_rates]

        return out

    def _is_consistent(self):
        """
        Checks whether the layers are compatible.

        :return:
        """

        dim = self.layers[0].map_dim

        for x in self.layers[1:]:

            if dim != x.data_dim:
                return False

        return True

    def predict(self, X):

        out = X

        for l in self.layers:
            out = np.array([l.predict_distance(x) for x in X])

        return out

if __name__ == "__main__":

    colors = np.array(
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

    colors = np.array(colors)

    layers = [Som(10, 10, 3, [1.0]), Som(5, 5, 100, [1.0])]
    p = Pyramid(layers)
    p.propagate(colors, 10, 10)