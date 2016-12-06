import numpy as np
import logging
import cProfile

from som import Som
from hebbian import Hebbian
from preprocessing.sample_from_corpus import sample_sentences_from_corpus, create_orth_phon_dictionary
from preprocessing.corpus import read_carmel
from preprocessing.patpho import PatPho
from preprocessing.ortho import Orthographizer
from itertools import chain


if __name__ == "__main__":

    english = read_carmel("data/dpc_en_carmel.txt", "data/dpc_en_carmelled.txt")
    dutch = read_carmel("data/dpc_nl_carmel.txt", "data/dpc_nl_carmelled.txt")

    max_o = 10
    max_phon = 3

    phones = set()
    for x in chain(english.values(), dutch.values()):
        phones.update(x)

    o = Orthographizer(max_length=max_o)
    p = PatPho(phones)

    english = create_orth_phon_dictionary(english, orthography=o, phonology=p)
    dutch = create_orth_phon_dictionary(dutch, orthography=o, phonology=p)

    Xeng, Yeng, weng = sample_sentences_from_corpus(["data/dpc_en.txt"],
                                                    english,
                                                    30000)

    print("Loaded English")

    Xdut, Ydut, wdut = sample_sentences_from_corpus(["data/dpc_nl.txt"],
                                                    dutch,
                                                    30000)

    print("Loaded Dutch")

    X = np.concatenate([Xeng, Xdut])
    Y = np.concatenate([Yeng, Ydut])

    s_phon = Som(100, 100, X.shape[1], 0.3)
    s_orth = Som(100, 100, Y.shape[1], 0.3)

    h = Hebbian(s_phon, s_orth, 0.3)
    h.run_samples(X, Y, samples_per_epoch=100000, batch_size=10000, num_epochs=10)
