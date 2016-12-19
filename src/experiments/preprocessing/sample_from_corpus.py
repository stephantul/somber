import numpy as np

from experiments.preprocessing.corpus import CorpusWordIter


def create_orth_phon_dictionary(dictionary, orthography, phonology):
    """
    Create a mapping from strings to their vectorized orthography
    and phonology.

    :param dictionary: A dictionary from orthography to phonology
    :param orthography: An initialized orthographic featurizer
    :param phonology: An initialized phonological featurizer
    :return:
    """

    result = {}
    for k, v in dictionary.items():
        try:
            result[k] = (orthography.vectorize_single(k), phonology.vectorize_single(v))
        except (KeyError, ValueError):
            continue

    return result


def sample_sentences_from_corpus(paths, dictionary, num_words):
    """
    Samples a number of sentences from a corpus iterator.
    Uses a dictionary to look up the phonological forms of words.
    Any words which are not in the dictionary, or which contain
    letters that are not in the feature space of the orthographic
    featurizer, are skipped.

    :param paths: A list of paths to files.
    :param dictionary: A dictionary from orthography to vectorized representations.
    :param num_words: The number of words to sample. If this number is
        larger than the number of words in the corpus, the iterator stops.
    :return: An array of phonological forms, an array of orthograpic forms, and a list of words
    all have the same length.
    """

    orthography = []
    phonology = []
    w = []

    for word in CorpusWordIter(paths, num_words):

        try:
            phon, orth = dictionary[word]
            phonology.append(phon)
            orthography.append(orth)
            w.append(word)
        except KeyError:
            pass

    return np.array(phonology), np.array(orthography), w
