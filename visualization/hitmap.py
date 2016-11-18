from .view import MatplotView
from matplotlib import pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


class HitMapView(MatplotView):

    def _set_labels(self, cents, ax, labels):
        for i, txt in enumerate(labels):
            ax.annotate(txt, (cents[i, 1], cents[i, 0]), size=10, va="center")

    def create(self, weights, h, w, pred):

        clusters = KMeans.fit(weights, 8)
        # codebook = getattr(som, 'cluster_labels', som.cluster())

        self.prepare()
        ax = self._fig.add_subplot(111)

        coord = np.array([(idx // h, idx % h) for idx in pred])
        self._set_labels(coord, ax, clusters[pred])

        plt.imshow(weights.reshape(h, w), alpha=.5)