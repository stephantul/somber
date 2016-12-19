import numpy as np

class Phonologizer(object):

    def __init__(self, phoneset, maxlen, empty=" ", pause="-"):

        phoneset.append(empty)
        phoneset.append(pause)

        self.maxlen = maxlen
        self.empty = empty
        self.pause = pause
        self.phoneset = phoneset
        self.data = dict(zip(phoneset, np.eye(len(phoneset))))

    def vectorize_single(self, word):
        """
        Vectorize a single word.

        Raises a ValueError if the word is too long.

        :param word: A string of characters
        :return: A numpy array, representing the concatenated letter representations.
        """

        if len(word) > self.maxlen:
            raise ValueError("Too long")

        x = np.zeros((self.maxlen, len(self.data[self.empty])))
        for idx, c in enumerate(word):
            x[idx] += self.data[c]

        return x