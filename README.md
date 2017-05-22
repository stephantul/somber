# **SOMBER**

**somber** (**S**omber **O**rganizes **M**aps **B**y **E**nabling **R**ecurrence) is a collection of numpy/python implementations of various kinds of _Self-Organizing Maps_ (SOMS), with a focus on SOMs for sequence data.

To the best of my knowledge, the sequential SOM algorithms implemented in this package haven't been open-sourced yet. If you do find examples, please let me know, so I can compare and link to them.

The package currently contains implementations of:

  * Regular Som (SOM) (Kohonen, various publications)
  * Recurrent Som (RSOM) ([Koskela et al., 1998](http://ieeexplore.ieee.org/document/725861/), among others)
  * Recursive Som (RecSOM) ([Voegtlin, 2002](http://www.sciencedirect.com/science/article/pii/S0893608002000722))
  * Merge Som (MSOM) ([Hammer and Strickert, 2005](http://www.sciencedirect.com/science/article/pii/S0925231204005107))

Because these various sequential SOMs rely on internal dynamics for convergence, i.e. they do not fixate on some external label like a regular Recurrent Neural Network, processing in a sequential SOM is currently strictly online. This means that every example is processed separately, and weight updates happen after every example. Research into the development of batching and/or multi-threading is currently underway.

If you need a fast regular SOM, check out [SOMPY](https://github.com/sevamoo/SOMPY), which is a direct port of the MATLAB Som toolbox.

If you need Neural Gas or Growing Neural Gas, check out [Kohonen](https://github.com/lmjohns3/kohonen).

### Usage

Unlike most sequence-based neural network packages, SOMBER doesn't use batches of sentences. The input to a SOMBER is therefore simply a single long sequence.

This makes switching and comparing between non-sequential SOM and the sequential SOMs easy, as the SOM itself will decide whether to pay attention to what came before, and how it will pay attention to what came before.

A consequence of representing everything as a single sequence is that the sequential SOMs assume everything in the sequence depends on what comes before, which might be unsuitable for your problem. This can be remedied using the `reset_context_symbol` function, which allows you to reset the context for given symbols (e.g. spaces between words, periods, newlines between sentences). Example below.

### Examples

#### Colors

The color dataset comes from this nice [blog]( https://codesachin.wordpress.com/2015/11/28/self-organizing-maps-with-googles-tensorflow/
)

```python3
import numpy as np

from somber.som import Som

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
s.train(X, num_epochs=100, total_updates=1000)

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

```

#### Using context reset

In this example, I assume you have some text, encoded as one-hot vectors
of alphabetic characters + space (hence dim == 27)

```python3
from somber.recurrent import Recursive
from somber.utils import reset_context_symbol

# Some array of characters
text = "i am a nice dog and like to play in the yard"

# Reset the context on space symbol
reset = reset_context_symbol(text, [" "])

# This is a dummy vectorize operation, not implemented!
X = vectorize(text)

r = Recursive((10, 10), 27, 0.3, alpha=1.0, beta=1.0)

# Now, context will be reset at the indices corresponding to the spaces.
# Therefore, the characters after the space are conditionally independent
# from the characters before the space.
r.train(X, context_mask=reset)

```

### TODO

See issues for TODOs/enhancements. If you use SOMBER, feel free to send me suggestions!

### Contributors

* Stéphan Tulkens

## LICENSE

MIT
