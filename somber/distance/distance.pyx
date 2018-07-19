cimport cython
cimport numpy as np
import numpy as np
from libc.math cimport fabs, sqrt, exp, cos, pow

DTYPE = np.int64
DTYPE_F = np.float64
ctypedef np.int64_t DTYPE_t
ctypedef np.float64_t DTYPE_F_t


@cython.boundscheck(False)
def euclidean(np.ndarray[DTYPE_F_t, ndim=2] data,
              np.ndarray[DTYPE_F_t, ndim=2] nodes):
    """Fast euclidean distance

    Parameters
    ----------
    a : np.ndarray - int64 - dim 2
        The first array, dim (M * N)
    b : np.ndarray - int64 - dim 2
        The second array, dim (P * N)

    Returns
    -------
    dot product : np.ndarray - float64 - dim2
        The cosine distance  product between each vector in a and each
        vector in b, dim (M * P)

    """
    cdef np.intp_t j = 0
    cdef np.intp_t i = 0

    cdef int n_items = data.shape[0]
    cdef int n_nodes = nodes.shape[0]
    cdef int length = data.shape[1]

    cdef np.ndarray[DTYPE_F_t, ndim=3] diff = np.zeros([n_items, n_nodes, length], dtype=DTYPE_F)

    for i in range(n_items):
        for j in range(n_nodes):
            for x in range(length):
                diff[i, j, x] = data[i, x] - nodes[j, x]

    return np.sqrt(np.sum(np.square(diff), -1)), diff
