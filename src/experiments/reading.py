import re
import numpy as np
import logging

from experiments.preprocessing.ortho import Orthographizer
from experiments.preprocessing.gecco import GeccoReader
from merging import Merging
from utils import MultiPlexer
from unidecode import unidecode

removenumbers = re.compile(r"\W")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    o = Orthographizer()

    g_nl = GeccoReader("../data/L1ReadingData.csv")
    g_en = GeccoReader("../data/L2ReadingData.csv")

    nl_corpus = " ".join(g_nl.corpus[0])
    #en_corpus = " ".join(g_en.corpus[0])
    #corpus = " ".join([nl_corpus, en_corpus])

    #del nl_corpus
    #del en_corpus

    corpus = removenumbers.sub(" ", nl_corpus)
    corpus = unidecode(corpus)
    X = o.transform(corpus)

    m = Merging((20, 20), 17, 0.3, alpha=0.001, beta=0.5)
    m.train(MultiPlexer(X, 100), num_effective_epochs=1000)

    whole = lambda x: m.predict(o.transform(x))