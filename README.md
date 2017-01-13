# **SOMBER**

**somber** (**S**elf **O**rganizing **M**aps **B**y **E**nabling **R**ecurrence) is a collection of numpy/python implementations of various kinds of _Self-Organizing Maps_ (SOMS), with a focus on SOMs for sequence data.

To the best of my knowledge, the sequential SOM algorithms haven't been open-sourced yet. If you do find examples, please let me know, so I can compare and link to them.

It currently contains implementations of:

  * Regular Som (SOM) (Kohonen, various publications)
  * Recurrent Som (RSOM) ([Koskela, 1998](http://ieeexplore.ieee.org/document/725861/), among others)
  * Recursive Som (RecSOM) ([Voegtlin, 2002](http://www.sciencedirect.com/science/article/pii/S0893608002000722))
  * Merge Som (MSOM) ([Hammer and Strickert, 2005](http://www.sciencedirect.com/science/article/pii/S0925231204005107))

Because these various sequential SOMs rely on internal dynamics for convergence, i.e. they do not fixate on some external label like a regular Recurrent Neural Network, processing in a sequential SOM is currently strictly online. This means that every example is processed separately, and weight updates happen after every example. Research into the development of batching and/or multi-threading is currently underway.

If you need a fast regular SOM, check out [SOMPY](https://github.com/sevamoo/SOMPY), which is a direct port of the MATLAB Som toolbox.

If you need Neural Gas or Growing Neural Gas, check out [Kohonen](https://github.com/lmjohns3/kohonen).

### Contributors

* Stéphan Tulkens

### Examples

TODO

## LICENSE

MIT
