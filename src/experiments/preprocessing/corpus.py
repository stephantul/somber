import re

from io import open
from glob import glob
from itertools import chain

remover = re.compile(r"\d")
strip_remove = lambda x: remover.sub(" ", x.strip())
matcher = re.compile(r"\w")


class CorpusWordIter(object):

    def __init__(self, filenames, maxwords, lower=False):

        self.filenames = filenames
        self.maxwords = maxwords
        self.lower = lower

    def __iter__(self):

        words = 0

        for f in self.filenames:
            for line in open(f):

                line = strip_remove(line.lower() if self.lower else line)

                if matcher.match(line):

                    for word in line.split():

                        yield word
                        words += 1

                        if words > self.maxwords:
                            raise StopIteration
                else:
                    continue


class CorpusSentenceIter(object):

    def __init__(self, filenames, maxlines, lower=True):
        """
        An efficient corpus iterator, which can read multiple files.

        :param filenames: A list of filenames
        :param maxlines: The maximum number of lines to read from
        the corpora
        :param lower: Whether to lowercase the corpus.
        """

        self.filenames = filenames
        self.maxlines = maxlines
        self.lower = lower

    def __iter__(self):
        """
        Iterates over the corpus
        """

        lines = 0

        for f in self.filenames:
            for line in open(f):

                line = strip_remove(line.lower() if self.lower else line)

                lines += 1

                if matcher.match(line):
                    yield line.split()
                else:
                    continue

                if lines > self.maxlines:
                    raise StopIteration


def get_paths(rootpaths, extensions):
    """
    Generate a list of paths from a list of rootpaths and  a list of extensions.
    Extensions must be the same length of rootpaths or empty.

    Each rootpath is assumed to directly correspond to the extension with the same
    index. [X, Y] [.txt, .xml] will thus look for .txt files in X, and .xml files in Y

    :param rootpaths: A list of rootpaths.
    :param extensions: A list of extensions.
    :return:
    """

    if not extensions:
        extensions = ["*.txt"] * len(rootpaths)

    return chain.from_iterable([glob("{0}/{1}".format(r, ext)) for r, ext in zip(rootpaths, extensions)])


def unigram_list(filenames, maxlines):
    """
    Get a unigram list for processing in Carmel.

    :param filenames: A list of filenames.
    :param maxlines: The maximum number of lines to iterate.
    :return: A set of unigrams.
    """

    unigrams = set()

    c = CorpusSentenceIter(filenames, maxlines=maxlines)
    for line in c:
        unigrams.update(line)

    return unigrams


def write_carmel(unigrams, filename):
    """
    Writes all the unigrams in a corpus to
    a file which can be read by Carmel.

    Used for phonological coding.

    :param unigrams: A list of unigrams
    :param filename: The filename to which to write.
    :return: None
    """

    unigrams = {x.lower() for x in unigrams}

    with open(filename, 'w') as f:

        for uni in unigrams:
            f.write("{0}\n".format(" ".join(['"{0}"'.format(x) for x in uni])))


def read_carmel(unigramfilename, phonologicalfilename):
    """
    Reads carmel files and the vocabulary files they were generated from
    to a dictionary.

    :param unigramfilename: The path to the file containing the unigrams
    :param phonologicalfilename: The path to the file containing the phonological forms
    :return:
    """

    phon_dict = {}

    for line1, line2 in zip(open(unigramfilename), open(phonologicalfilename)):

        line1 = line1.strip()
        line2 = line2.strip()

        line1 = "".join(line1.replace('"', "").split())

        if line2:
            phon_dict[line1] = "".join(line2.split())
        else:
            continue

    return phon_dict