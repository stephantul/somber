SOMBER
======

**somber** (Somber Organizes Maps By Enabling Recurrence) is a collection of numpy/python implementations of various kinds of *Self-Organizing Maps* (SOMS), with a focus on SOMs for sequence data.

To the best of my knowledge, the sequential SOM algorithms implemented in this package haven't been open-sourced yet. If you do find examples, please let me know, so I can compare and link to them.

The package currently contains implementations of:

  * Regular Som (SOM) (Kohonen, various publications)
  * Recursive Som (RecSOM) (`Voegtlin, 2002 <http://www.sciencedirect.com/science/article/pii/S0893608002000722>`_)
  * Merge Som (MSOM) (`Hammer and Strickert, 2005 <http://www.sciencedirect.com/science/article/pii/S0925231204005107>`_)

Because these various sequential SOMs rely on internal dynamics for convergence, i.e. they do not fixate on some external label like a regular Recurrent Neural Network, processing in a sequential SOM is currently strictly online. This means that every example is processed separately, and weight updates happen after every example. Research into the development of batching and/or multi-threading is currently underway.

If you need a fast regular SOM, check out `SOMPY <https://github.com/sevamoo/SOMPY>`_, which is a direct port of the MATLAB Som toolbox.

If you need Neural Gas or Growing Neural Gas, check out `Kohonen <https://github.com/lmjohns3/kohonen>`_.

Usage
-----

Unlike most sequence-based neural network packages, SOMBER doesn't use batches of sentences. The input to a SOMBER is therefore simply a single long sequence.

This makes switching and comparing between non-sequential SOM and the sequential SOMs easy, as the SOM itself will decide whether to pay attention to what came before, and how it will pay attention to what came before.

A consequence of representing everything as a single sequence is that the sequential SOMs assume everything in the sequence depends on what comes before, which might be unsuitable for your problem.

Examples
--------

Colors
------

The color dataset comes from this nice `blog, <https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow>`_

.. code-block:: python

  import numpy as np

  from somber.batch.som import Som

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

  color_names = \
      ['black', 'blue', 'darkblue', 'skyblue',
       'greyblue', 'lilac', 'green', 'red',
       'cyan', 'violet', 'yellow', 'white',
       'darkgrey', 'mediumgrey', 'lightgrey']

  # initialize
  s = Som((10, 10), dim=3, learning_rate=0.3)

  # train
  s.fit(X, num_epochs=100, total_updates=50)

  # predict: get the index of each best matching unit.
  predictions = s.predict(X)
  # quantization error: how well do the best matching units fit?
  quantization_error = s.quant_error(X)
  # inversion: associate each node with the exemplar that fits best.
  inverted = s.invert_projection(X, color_names)
  # Mapping: get weights, mapped to the grid points of the SOM
  mapped = s.map_weights()

  import matplotlib.pyplot as plt

  plt.imshow(mapped)

TODO
----

See issues for TODOs/enhancements. If you use SOMBER, feel free to send me suggestions!

Contributors
------------

* Stéphan Tulkens

LICENSE
-------

MIT
