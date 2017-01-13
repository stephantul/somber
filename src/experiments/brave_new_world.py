import re
import numpy as np
import logging

from recursive import Recursive
from som import Som
from experiments.preprocessing.ortho import Orthographizer
from string import ascii_lowercase
from utils import MultiPlexer


removenumbers = re.compile(r"\d")

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    brv = " ".join(open("../data/brvnwworld.txt").readlines())

    brv = removenumbers.sub(" ", brv)

    r = re.compile("\n")
    brv = r.sub(" ", brv)

    print("{0}".format(len(brv)))

    # words = brv.split()

    test = 'the cat sat on the mat mask master'

    # maxlen = 10
    # words = filter(lambda x: len(x) <= maxlen, words)

    o = Orthographizer()
    X = np.array(o.transform(brv))

    # X = np.hstack([X[:-2], X[1:-1], X[2:]])

    X_test = o.transform(test)

    r = Recursive((20, 20), X.shape[1], learning_rate=0.1, alpha=3, beta=0.7)
    r.train(MultiPlexer(X, 2), total_epochs=1000, rough_epochs=0.5)

    X_letters = o.transform(ascii_lowercase)
    print(r.predict(X_letters))
    print(r.predict(X_test))