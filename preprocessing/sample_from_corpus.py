import numpy as np

from preprocessing.corpus import CorpusWordIter
from preprocessing.patpho import PatPho
from preprocessing.ortho import Orthographizer


def sample_sentences_from_corpus(paths, o, p, num_words):
    """
    Samples a number of sentences from a corpus iterator.
    Uses a dictionary to look up the phonological forms of words.
    Any words which are not in the dictionary, or which contain
    letters that are not in the feature space of the orthographic
    featurizer, are skipped.

    :param paths: A list of paths to files.
    :param dictionary: A dictionary from orthography to phonology strings.
    :param num_sentences: The number of sentences to sample. If this number is
        larger than the number of sentences in the corpus, the iterator stops.
    :return: An array of phonological forms, an array of orthograpic forms, and a list of words
    all have the same length.
    """

    w = []
    orthography = []
    phonology = []

    for word in CorpusWordIter(paths, num_words):

        try:
            loc_o = o.vectorize_single(word)
            loc_p = p.vectorize_single(word)
        except (ValueError, KeyError):
            continue

        orthography.append(loc_o)
        phonology.append(loc_p)
        w.append(word)

    return np.array(phonology), np.array(orthography), w
