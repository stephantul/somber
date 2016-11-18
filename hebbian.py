import numpy as np


class Hebbian(object):

    def __init__(self, som1, som2):
        """


        :param som1:
        :param som2:
        :return:
        """

        self.som1 = som1
        self.som2 = som2

        self.hebbian_matrix = np.zeros((len(self.som1.weights), len(self.som2.weights)))

    def proceed(self, X, Y):
        """
        Assumes online learning, no concept of epochs.
        So, how does the radius diminish?

        :param x:
        :param y:
        :return:
        """

        for x, y in zip(X, Y):

            bmu_x = self.som1.train(x)
            bmu_y = self.som2.train(y)

            self.hebbian_matrix[bmu_x, bmu_y] += 0.01
