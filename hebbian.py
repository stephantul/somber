import numpy as np

from progressbar import progressbar


class Hebbian(object):

    def __init__(self, som1, som2, learning_rate):
        """
        A Hebbian learner for 2 Self Organizing Maps.

        The idea behind this is that soms can learn be fit to a single dataset
        in two different modalities, so that the system learns that there us a
        connection between similar inputs across the maps.

        :param som1: The first SOM, already initialized
        :param som2: The second SOM, already initialized
        """

        self.som1 = som1
        self.som2 = som2

        # The matrix of hebbian weights.
        self.hebbian_matrix = np.zeros((len(self.som1.weights), len(self.som2.weights)))
        self.learning_rate = learning_rate

    def run_samples(self, X, Y, samples, num_epochs=10):
        """
        Assumes online learning, no concept of epochs.

        Tracks learning rate globally, so there is no concept of
        individual learning rates across the soms. Can be augmented.

        X and Y are the same observations in a different modality.

        :param X: The first modality
        :param Y: The second modality
        :param samples: the number of samples to take from X and Y
        :return:
        """

        # X and Y must be same size
        assert len(X) == len(Y)

        # Local copy of learning rate.
        learning_rate = self.learning_rate

        epoch_equiv = samples // num_epochs

        # Scaling factors and scalers must be separate because the maps
        # need not be the same size.
        scaling_factor_1 = np.log(max(self.som1.width, self.som1.height))
        scaling_factor_2 = np.log(max(self.som2.width, self.som2.height))

        scaler_1 = num_epochs / np.log(scaling_factor_1)
        scaler_2 = num_epochs / np.log(scaling_factor_2)

        sample_range = np.arange(len(X))
        epoch = 0

        radius_1 = scaling_factor_1 * np.exp(-epoch / scaler_1)
        radius_2 = scaling_factor_2 * np.exp(-epoch / scaler_2)

        for sample in progressbar(range(samples)):

            is_epoch_step = sample and sample % epoch_equiv == 0

            if is_epoch_step:
                epoch += 1
                radius_1 = scaling_factor_1 * np.exp(-epoch / scaler_1)
                radius_2 = scaling_factor_2 * np.exp(-epoch / scaler_2)

            # The index of the chosen item.
            chosen = np.random.choice(sample_range)
            x = X[chosen]
            y = Y[chosen]

            # Recompute radius at every sample
            # could be done every n epochs to increase speed.

            # Do a single cycle
            bmu_x = self.som1.single_cycle(x, radius_1, learning_rate, is_epoch_step or not sample)
            bmu_y = self.som2.single_cycle(y, radius_2, learning_rate, is_epoch_step or not sample)

            # Update the Hebbian weights using a simple update rule
            # Could be replaced by a more complex update rule.
            self.hebbian_matrix[bmu_x, bmu_y] += learning_rate * (1 - learning_rate)

            # Update learning rate.
            if is_epoch_step:
                learning_rate = self.learning_rate * np.exp(-epoch / num_epochs)

    def predict(self, X, Y):
        """
        Predict the Best Matching Units for a dataset in both modalities.

        :param X: The first modality
        :param Y: The second modality
        :return:
        """

        assert len(X) == len(Y)

        X = self.som1.predict(X)
        Y = self.som2.predict(Y)

        return X, Y
