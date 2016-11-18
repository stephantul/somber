import matplotlib
matplotlib.use("Agg")

from .view import MatplotView
from matplotlib import pyplot as plt
from pylab import imshow, contour
from math import sqrt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

from scipy.spatial import distance_matrix


def calculate_map_dist(width, height):
    """
    Calculates the grid distance, which will be used during the training
    steps. It supports only planar grids for the moment
    """

    nnodes = width * height
    distance_matrix = np.zeros((nnodes, nnodes))

    for i in range(nnodes):
        distance_matrix[i] = calculate_map_dists(width, height, i).reshape(1, nnodes)

    return distance_matrix


def calculate_map_dists(width, height, index):

    # bmu should be an integer between 0 to no_nodes
    if 0 <= index <= (width*height):
        node_col = int(index % height)
        node_row = int(index / height)
    else:
        raise ValueError(
            "Node index '%s' is invalid" % index)

    if width > 0 and height > 0:
        r = np.arange(0, width, 1)[:, np.newaxis]
        c = np.arange(0, height, 1)
        dist2 = (r-node_row)**2 + (c-node_col)**2

        dist = dist2.ravel()
    else:
        raise ValueError(
            "One or both of the map dimensions are invalid. "
            "Cols '%s', Rows '%s'".format(cols=height, rows=width))

    return dist


class UMatrixView(MatplotView):

    def build_u_matrix(self, weights, width, height, distance=1):

        ud2 = calculate_map_dist(width, height)
        u_matrix = np.zeros((weights.shape[0], 1))
        vector = weights

        for i in range(weights.shape[0]):
            v = vector[i][np.newaxis, :]
            neighborbor_ind = ud2[i][0:] <= distance
            neighbors = vector[neighborbor_ind]
            u_matrix[i] = distance_matrix(
                v, neighbors).mean()

        return u_matrix.reshape((width, height))

    def create(self, weights, data, w, h, pred, distance2=1, row_normalized=False, show_data=True,
               use_contour=True, labels=False):
        umat = self.build_u_matrix(weights, w, h, distance=distance2)
        msz = (w, h)

        coord = np.array([(idx // h, idx % h) for idx in pred])

        fig, ax = plt.subplots(1, 1)

        '''x_min, y_min = np.min(data, axis=0)
        new_data = data[:, 0] - x_min, data[:, 1] - y_min
        new_data = np.array([new_data[0] * (w / new_data[0].max()), new_data[1] * (h / new_data[1].max())])

        plt.scatter(new_data[0, :], new_data[1,:], cmap=plt.cm.get_cmap('coolwarm'), alpha=1)'''
        plt.imshow(umat, cmap=plt.cm.get_cmap('RdYlBu_r'), alpha=1)

        if use_contour:
            mn = np.min(umat.flatten())
            mx = np.max(umat.flatten())
            std = np.std(umat.flatten())
            md = np.median(umat.flatten())
            mx = md + 0*std
            contour(umat, np.linspace(mn, mx, 15), linewidths=0.7,
                    cmap=plt.cm.get_cmap('Blues'))

        if show_data:
            plt.scatter(coord[:, 1], coord[:, 0], s=2, alpha=1., c='Gray',
                        marker='o', cmap='jet', linewidths=3, edgecolor='Gray')
            plt.axis('off')

        ratio = float(msz[0])/(msz[0]+msz[1])
        fig.set_size_inches((1-ratio)*15, ratio*15)
        plt.tight_layout()
        plt.subplots_adjust(hspace=.00, wspace=.000)
        sel_points = list()

        self._fig = fig

        return sel_points, umat
