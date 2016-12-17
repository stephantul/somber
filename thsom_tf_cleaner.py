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
        self.alpha = alpha

        self.const_dim = np.sqrt(dim)

        super().__init__(width, height, dim, alpha)

    def _initialize_graph(self, batch_size, num_epochs):

        with self._graph.as_default():

            self._sess = tf.Session()

            self._input = tf.placeholder(tf.float32, shape=[None, self.data_dim], name='input_data')
            self._bmu = tf.placeholder(tf.float32, name='bmu')
            self._prev_activation = tf.placeholder(tf.float32, shape=self.map_dim, name='prev')
            self._neighborhood = tf.placeholder(tf.float32, shape=[self.map_dim, self.map_dim])
            # self._concat_input = self._prev_activation

            self.distance_grid = tf.constant(self._calculate_distance_grid())
            self.learning_rate = self.alpha

            self.zeta = tf.get_variable("zeta", 1, dtype=np.float32)
            self.alpha = tf.get_variable("alpha", 1, dtype=np.float32)
            self.sigma = tf.get_variable("sigma", 1, dtype=np.float32)
            self.beta = tf.constant(self.beta)

            self.alpha = tf.cast(tf.multiply(self.alpha, self.learning_rate), "float64")
            self.zeta = tf.multiply(self.zeta, self.learning_rate)
            self.sigma = tf.multiply(self.sigma, self.learning_rate)

            tf.while_loop()

            self.states = tf.scan(self._som_step, self._input, initializer=tf.zeros(self.map_dim), back_prop=False, infer_shape=False)

            self._sess.run(tf.global_variables_initializer())

    def _som_step(self, prev_activation, current_input):

        with tf.variable_scope("som_step"):

            # input_, prev_activation = tf.split(0, 2, prev_state)

            prev_bmu = tf.cast(tf.argmax(prev_activation, 0), "int64")

            self.spatial = tf.get_variable("spatial", shape=[self.map_dim, self.data_dim])
            self.temporal = tf.get_variable("temporal", shape=[self.map_dim, self.map_dim])

            spatial_activation = tf.sub(current_input, self.spatial)

            temporal_sum = tf.reduce_sum(self.temporal, 0)
            temporal_activation = tf.mul(prev_activation, temporal_sum)

            euclidean = tf.sqrt(tf.reduce_sum(tf.square(spatial_activation), 1))
            total_activation = tf.add(tf.sub(tf.cast(self.const_dim, "float32"), euclidean), temporal_activation)

            total_activation = tf.transpose(tf.div(tf.transpose(total_activation), tf.reduce_max(total_activation)))
            bmu = tf.cast(tf.argmax(total_activation, 0), "int64")

            # CALCULATE NEIGHBORHOOD FUNCTIONS
            influences = tf.pack([tf.gather(self._neighborhood, bmu)] * self.data_dim)
            influences = tf.reshape(influences, (self.map_dim, self.data_dim))

            # CALCULATE DELTA FOR SPATIAL
            spatial_delta = tf.reduce_mean(tf.mul(influences, spatial_activation), 0)

            # DEFINE TEMPORAL BASIC
            temporal_delta = tf.neg(tf.mul(self.zeta, tf.add(self.temporal, self.beta)))

            g = tf.expand_dims(tf.gather(tf.gather(self.temporal, bmu), prev_bmu), 0)
            val = tf.mul(self.zeta, (tf.sub(1.0, tf.add(g, self.beta))))

            delta_plus = tf.SparseTensor([[bmu, prev_bmu]], val, (self.map_dim, self.map_dim))

            temporal_delta = tf.add(temporal_delta, tf.sparse_tensor_to_dense(delta_plus))

            self.spatial = tf.add(self.spatial,
                                 spatial_delta)

            self.temporal = tf.add(self.temporal, temporal_delta)

        return total_activation

    def _update_neighborhood(self):

        with tf.variable_scope("neighborhood"):

            neighborhood_func = tf.exp(tf.div(
                self.distance_grid, tf.cast(tf.square(tf.mul(2.0, self.sigma)), "float64")))

            self.neighborhood = tf.mul(self.alpha, neighborhood_func)

    def epoch_step(self, X, epoch, batch_size=0):

        with self._graph.as_default():

            for x in progressbar(X):

                self._update_neighborhood()
                self._sess.run([self.states], feed_dict={self._input: x})

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