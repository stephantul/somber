import numpy as np
import time

from progressbar import progressbar


def softmax(x):

    e_x = np.exp(x - np.max(x))
    out = e_x / e_x.sum()
    return out


class Hebbian(object):

    def __init__(self, phon_som, orth_som, learning_rate, hebbian_offset):
        """
        A Hebbian learner for 2 Self Organizing Maps.

        The idea behind this is that soms can learn be fit to a single dataset
        in two different modalities, so that the system learns that there us a
        connection between similar inputs across the maps.

        :param phon_som: The first SOM, already initialized
        :param orth_som: The second SOM, already initialized
        """

        self.phon_som = phon_som
        self.orth_som = orth_som

        # The matrix of hebbian weights.
        self.hebbian_matrix = np.zeros((len(self.phon_som.weights), len(self.orth_som.weights)))
        self.transition_probas = None
        self.learning_rate = learning_rate
        self.hebbian_offset = hebbian_offset

    def run_samples(self, X, Y, num_epochs=10, batch_size=100):
        """
        X and Y are vectorial representations of the same input
        in a different modality.

        :param X: The first modality
        :param Y: The second modality
        :param num_epochs: The number of epochs for which to run.
        :return:
        """

        # X and Y must be same size
        assert len(X) == len(Y)

        # Local copy of learning rate.
        learning_rate = self.learning_rate

        # Scaling factors and scalers must be separate because the maps
        # need not be the same size.
        scaling_factor_1 = np.log(max(self.phon_som.width, self.phon_som.height))
        scaling_factor_2 = np.log(max(self.orth_som.width, self.orth_som.height))

        scaler_1 = num_epochs / np.log(scaling_factor_1)
        scaler_2 = num_epochs / np.log(scaling_factor_2)

        for epoch in range(num_epochs):

            radius_1 = scaling_factor_1 * np.exp(-epoch / scaler_1)
            radius_2 = scaling_factor_2 * np.exp(-epoch / scaler_2)

            start = time.time()

            activation_x = self.phon_som._example(X, radius_1, [learning_rate], batch_size=batch_size)
            activation_y = self.orth_som._example(Y, radius_2, [learning_rate], batch_size=batch_size)

            p = self.hebbian_offset * np.kron(activation_x, activation_y).reshape((self.hebbian_matrix.shape[0], self.hebbian_matrix.shape[1]))
            print(p.mean())

            self.hebbian_matrix += p

            # Update learning rate.
            print("Epoch {0}/{1} took {2:.2f} seconds".format(epoch, num_epochs, time.time() - start))
            learning_rate = self.learning_rate * np.exp(-epoch / num_epochs)

        self._mtr_to_proba()

    def _mtr_to_proba(self):

        self.transition_probas = np.array(np.square([x / np.linalg.norm(x) for x in self.hebbian_matrix + 0.01]))

    def predict(self, orthography):
        """
        Predict the Best Matching Units for a dataset in both modalities.

        :param orthography: Vectorized orthographic representations
        :param num_to_return: The number of bmus to return
        :return: A prediction for both X and Y
        """

        if len(orthography.shape) == 2:
            orthography = orthography.reshape(1, orthography.shape[0], orthography.shape[1])

        bmus = self.orth_som._predict_base(orthography)

        phon_bmus = []

        for bmu_seq in bmus:

            mtr = self.transition_probas[bmu_seq]
            phon_bmus.append(mtr)

        return np.array(phon_bmus), bmus

    def viterbi_decode(self, orthography):

        if orthography.shape == 2:
            orthography = orthography.reshape(1, orthography.shape[0], orthography.shape[1])

        bmus = self.orth_som._predict_base(orthography)
        mtr = self.transition_probas[bmus]

