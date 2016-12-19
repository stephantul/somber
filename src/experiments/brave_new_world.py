import re
import numpy as np

from msom import MSom
from experiments.preprocessing.ortho import Orthographizer
from string import ascii_lowercase
from utils import MultiPlexer


removenumbers = re.compile(r"\d")


def orth_help(sentence):

    return Orthographizer(max_length=len(sentence)).vectorize_single(sentence)

if __name__ == "__main__":

    brv = " ".join(open("../data/brvnwworld.txt").readlines())

    brv = removenumbers.sub(" ", brv)

    r = re.compile("\n")
    brv = r.sub(" ", brv)

    print(len(brv))

    # words = brv.split()

    test = 'the cat sat on the mat mask master'

    # maxlen = 10
    # words = filter(lambda x: len(x) <= maxlen, words)

    o = Orthographizer(max_length=len(brv))
    X = np.array(o.vectorize_single(brv))

    print(X.shape)

    X_test = Orthographizer(max_length=len(test)).vectorize_single(test)

    r = MSom(20, 20, X.shape[1], learning_rate=0.03, alpha=0.0, beta=0.5)
    r.train(MultiPlexer(X, 100), num_effective_epochs=1000)

    X_letters = Orthographizer(max_length=26).vectorize_single(ascii_lowercase)
    print(r.predict(X_letters))
    print(r.predict(X_test))
