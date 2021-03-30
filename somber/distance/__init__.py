import pyximport
import numpy as np

pyximport.install(setup_args={"include_dirs": np.get_include()})
from .distance import euclidean  # noqa

__all__ = ["euclidean"]
