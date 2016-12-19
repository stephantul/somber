import numpy as np
from src.preprocessing.ortho import Orthographizer

from experiments.preprocessing.phono import Phonologizer
from hebbian import Hebbian


def unison_shuffled_copies(a, b):
    """
    from http://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    :param a:
    :param b:
    :return:
    """

    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def wordlist_from_dpalign(filename, maxlen):

    wordlist = []
    phonelist = []

    for line in open(filename):

        word, pron = line.strip().split()

        if len(word) < maxlen and "-" not in word and len(pron) < maxlen:
            wordlist.append("{0}".format(word))
            phonelist.append("{0}".format(pron))

    return wordlist, phonelist

if __name__ == "__main__":

    np.random.seed(44)

    words, phones = wordlist_from_dpalign("data/dpalign-txt-dutch.txt.dpalign", 15)

    phoneset = set("".join(phones))

    p = Phonologizer(list(phoneset), maxlen=15)
    o = Orthographizer(15)

    X = np.array([p.vectorize_single(phon) for phon in phones])
    Y = np.array([o.vectorize_single(orth) for orth in words])

    X, Y = unison_shuffled_copies(X, Y)

    dev_split = int(0.9 * len(X))

    X_train, X_dev = X[:dev_split], X[dev_split:]
    Y_train, Y_dev = Y[:dev_split], Y[dev_split:]

    r_phon = Rsom(30, 30, X.shape[2], 1.0, 0.5)
    r_orth = Rsom(30, 30, Y.shape[2], 1.0, 0.5)

    orth_to_phon = {k: v for k, v in zip(words, phones)}

    test_orth = ['kan', 'haan', 'maan', 'pan', 'span', 'man']
    test_phon = [orth_to_phon[k] for k in test_orth]

    X_test = np.array([p.vectorize_single(phon) for phon in test_phon])
    Y_test = np.array([o.vectorize_single(orth) for orth in test_orth])

    h = Hebbian(orth_som=r_orth, phon_som=r_phon, learning_rate=1.0, hebbian_offset=0.01)
    h.run_samples(X_train[:1000], Y_train[:1000], num_epochs=100, batch_size=100)

