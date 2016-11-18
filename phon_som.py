import logging

from som import Som
from preprocessing.pron_to_dictionary import pron_to_dictionary
from preprocessing.pythonic_patpho import PatPho
from collections import OrderedDict
from itertools import chain
from visualization import umatrix


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    dutch = OrderedDict(pron_to_dictionary("data/dpl.cd"))
    english = OrderedDict(pron_to_dictionary("data/epl.cd", phon_indices=(11, 7)))

    p = PatPho(max_length=3)

    english = {k: v for k, v in english.items() if not set(v).difference(p.phonemes.keys())}

    X = p.transform(list(english.values()))

    s = Som(50, 50, X.shape[1], 0.3)
    history = s.fit(50, X, return_history=5)

    from visualization.umatrix import UMatrixView

    for idx, x_w in enumerate(history):

        x, weight = x_w

        view = UMatrixView(500, 500, 'dom')
        view.create(weight, X, s.width, s.height, x)
        view.save("junk_viz/_{0}.svg".format(idx))

        print("Made {0}".format(idx))