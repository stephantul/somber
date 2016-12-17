import numpy as np
import logging
import time
import cProfile
import tensorflow as tf

from progressbar import progressbar
from som_tf import Som

logging.basicConfig(level=logging.INFO)


class THSom(Som):

    def __init__(self, width, height, dim, alpha, zeta, beta):

        self.beta = beta
        self.zeta = zeta
        self.const_dim = np.sqrt(dim)
        super().__init__(width, height, dim, alpha)

    def _initialize_graph(self, datalength, num_epochs):

        with self._graph.as_default():

            self.weights = tf.Variable(tf.random_uniform([self.map_dim, self.data_dim], minval=0.0, maxval=1.0))
            self.temporal_weights = tf.Variable(tf.zeros([self.map_dim, self.map_dim]))

            self.vect_input = tf.placeholder("float", [self.data_dim])
            self.prev_activation = tf.placeholder("float", [self.map_dim])
            self.prev_bmu = tf.placeholder("int64")
            self.epoch = tf.placeholder("float")
            self.distance_grid = tf.constant(self._calculate_distance_grid())

            self._sess = tf.Session()

    def _get_neighborhood(self, num_epochs):

        with self._graph.as_default():

            with tf.variable_scope("learning_rates"):

                learning_rate = tf.exp(tf.div(-self.epoch, num_epochs))
                alpha = tf.mul(self.alpha, learning_rate, "alpha")
                zeta = tf.mul(self.zeta, learning_rate, "zeta")
                sigma = tf.mul(self.sigma, learning_rate, "sigma")

                neighborhood_func = tf.exp(tf.div(tf.cast(
                    self.distance_grid, "float32"), tf.square(tf.mul(2.0, zeta))))

                tf.mul(alpha, neighborhood_func, name="neighborhood")

    def _get_activation(self):

        with self._graph.as_default():

            with tf.variable_scope("learning_rates", reuse=True):

                spatial_activation = tf.sub(self.vect_input, self.weights)

                temporal_sum = tf.reduce_sum(self.temporal_weights, 0)
                temporal_activation = tf.mul(self.prev_activation, temporal_sum)

                euclidean = tf.sqrt(tf.reduce_sum(tf.square(spatial_activation), 1))
                total_activation = tf.add(tf.sub(tf.cast(self.const_dim, "float32"), euclidean), temporal_activation)

                self._total_activation = tf.transpose(tf.div(tf.transpose(total_activation), tf.reduce_max(total_activation)))
                self._bmu = tf.cast(tf.argmax(self._total_activation, 0), "int64")

                zeta = tf.get_variable("zeta")

                # CALCULATE NEIGHBORHOOD FUNCTIONS
                influences = tf.pack([tf.gather(tf.get_variable("neighborhood"), self._bmu)] * self.data_dim)
                influences = tf.reshape(influences, (self.map_dim, self.data_dim))

                # CALCULATE DELTA FOR SPATIAL
                spatial_delta = tf.reduce_mean(tf.mul(influences, spatial_activation), 0)

                # DEFINE TEMPORAL BASIC
                temporal_delta = tf.neg(tf.mul(zeta, tf.add(self.temporal_weights, self.beta)))

                g = tf.gather(tf.gather(self.temporal_weights, self._bmu), self.prev_bmu)

                delta_plus = tf.SparseTensor([[self._bmu, self.prev_bmu]], tf.mul(zeta, (tf.sub(1.0, tf.add(g, self.beta)))), (self.map_dim, self.map_dim))

                temporal_delta = tf.add(temporal_delta, tf.sparse_tensor_to_dense(delta_plus))

                new_weights = tf.add(self.weights,
                                     spatial_delta)

                new_temporal_weights = tf.add(self.temporal_weights, temporal_delta)

                self._training_op_temporal = tf.assign(self.temporal_weights,
                                                       tf.clip_by_value(new_temporal_weights, 0.0, 1.0))

                self._training_op_spatial = tf.assign(self.weights,
                                                      tf.clip_by_value(new_weights, 0.0, 1.0))

    def epoch_step(self, X, epoch, num_epochs):

        with self._graph.as_default():

            self._get_neighborhood(num_epochs)

        prev_activation = np.zeros(self.map_dim, dtype=np.float)
        bmu = 0

        for x in progressbar(X):

            for value in x:

                bmu, prev_activation, _, _ = self._sess.run([self._bmu, self._total_activation, self._training_op_spatial, self._training_op_temporal],
                                                            feed_dict={self.vect_input: value,
                                                                       self.epoch: epoch,
                                                                       self.prev_activation: prev_activation,
                                                                       self.prev_bmu: bmu})

    def assign_exemplar(self, exemplars, names=()):

        exemplars = np.array(exemplars)
        distances = self._pseudo_distance(exemplars, self.weights)
        distances = np.sum(np.square(distances), axis=2)

        if not names:
            return distances.argmax(axis=0)
        else:
            return [names[x] for x in distances.argmax(axis=0)]

    def get_weights(self):

        with self._sess.as_default():

            w = self.weights.eval()
            t = self.temporal_weights.eval()

        return w, t

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    colors = np.array(
         [[1., 0., 1.],
          [0., 0., 0.],
          [0., 0., 1.],
          [0., 0., 0.5],
          [0.125, 0.529, 1.0],
          [0.33, 0.4, 0.67],
          [0.6, 0.5, 1.0],
          [0., 1., 0.],
          [1., 0., 0.],
          [0., 1., 1.],
          [1., 1., 0.],
          [1., 1., 1.],
          [.33, .33, .33],
          [.5, .5, .5],
          [.66, .66, .66]])

    data = np.tile(colors, (100, 1, 1))

    colorpicker = np.arange(len(colors))

    data = colors[np.random.choice(colorpicker, size=15)]
    data = np.array([data] * 1000)
    print(data.shape)

    s = THSom(30, 30, 3, 1.0, 1.0, 0.01)
    start = time.time()
    s.train(data, num_epochs=10, batch_size=100)

    # bmu_history = np.array(bmu_history).T
    print("Took {0} seconds".format(time.time() - start))