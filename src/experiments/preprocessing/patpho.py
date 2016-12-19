import numpy as np

from sklearn.feature_extraction import DictVectorizer
from ipapy.ipastring import IPAString
from ipapy.ipachar import IPAVowel


class PatPho(object):

    def __init__(self, phoneset, max_length=3, left=True):
        """
        Python re-implementation of of PatPho -- a system for converting sequences of phonemes to vector representations
        that capture phonological similarity of words.

        The system is described in:

            Li, P., & MacWhinney, B. (2002). PatPho: A phonological pattern generator for neural networks.
                Behavior Research Methods, Instruments, & Computers, 34(3), 408-415.

        The original C implementation can be found here (June 2015): http://www.personal.psu.edu/pul8/patpho_e.shtml
        """
        super().__init__()
        self.syllabic_grid = None
        self.idx = None
        self.max_length = max_length
        self.init_syllabic_grid()
        self.left = left

        self.phonemes, self.vowels, self.phoneset = self._featurize_phoneset(phoneset)

    def _prune(self, string):

        string = IPAString(unicode_string=string, ignore=True, single_char_parsing=True)
        return "".join([p for p in [str(p) for p in string] if p in self.phoneset])

    def _featurize_phoneset(self, phoneset):
        """
        Featurize the phonetic features into a given binary representation.

        :param phoneset: A list of phonemes
        :return: A dictionary of featurized phonemes and a list of vowels.
        """

        phonemes = IPAString(unicode_string="".join(phoneset), ignore=True, single_char_parsing=True)
        phonemes = [p for p in phonemes if not p.is_suprasegmental and not p.is_diacritic]

        # Convert to feature-based representation.
        data = {str(p): self._convert(p) for idx, p in enumerate(phonemes)}

        # Separate vowels and consonants and remove vowel feature
        vowels = {k: {c: p for c, p in v.items() if c != "vowel"} for k, v in data.items() if v['vowel']}
        consonants = {k: {c: p for c, p in v.items() if c != "vowel"} for k, v in data.items() if not v['vowel']}

        # Vectorize into binary array
        temp_v = DictVectorizer(sparse=False).fit_transform(vowels.values())
        temp_c = DictVectorizer(sparse=False).fit_transform(consonants.values())

        # Finalize into dict
        vowels = {k: list(temp_v[idx]) for idx, k in enumerate(vowels)}
        consonants = {k: list(temp_c[idx]) for idx, k in enumerate(consonants)}
        phonemes = vowels.copy()
        phonemes.update(consonants)

        phonemes.update({"VO": np.zeros(len(phonemes['a']), dtype=float),
                         "CO": np.zeros(len(phonemes['d']), dtype=float)})

        return phonemes, vowels, list(phonemes.keys())

    @staticmethod
    def _convert(phoneme):
        """
        Converts a single phoneme to the features found in Ipapy.

        :param phoneme: A phoneme as ipapy IPAChar.
        :return: A dictionary of features.
        """

        out = {"vowel": phoneme.is_vowel}

        if type(phoneme) is IPAVowel:
            out["height"] = phoneme.height
            out["backness"] = phoneme.backness
            out["roundness"] = phoneme.roundness
        else:
            out["manner"] = phoneme.manner
            out["voicing"] = phoneme.voicing
            out["place"] = phoneme.place

        return out

    def init_syllabic_grid(self):
        """ initialize trisyllabic consonant-vowel (CO-VO) grid """
        return ["CO", "CO", "CO", "VO", "VO"] * self.max_length + ["CO", "CO", "CO"]

    @staticmethod
    def index_to_next_vowel(grid):
        """ increment self.idx to next empty vowel postion in syllabic grid """
        return grid.index("VO")

    @staticmethod
    def index_to_next_consonant(grid):
        """ increment self.idx to next empty consonant postion in syllabic grid """
        return grid.index("CO")

    def check(self, x):

        try:
            return not set(x).difference(self.phonemes.keys())
        except KeyError:
            return False

    def vectorize_single(self, x):
        """
        Convert the phoneme sequence to a vector representation.
        """

        x = self._prune(x)

        if not self.left:
            x = x[::-1]

        grid = self.init_syllabic_grid()
        index = 0

        # go through the phonemes and insert them into the metrical grid
        for p in x:
            if p in self.vowels:
                try:
                    index += self.index_to_next_vowel(grid[index:])
                    grid[index] = p
                except ValueError:
                    raise ValueError('Word is too long: {0}'.format(x))

            elif p in self.phonemes:
                try:
                    index += self.index_to_next_consonant(grid[index:])
                    grid[index] = p
                except ValueError:
                    raise ValueError('Word is too long: {0}'.format(x))

            else:
                print('Unknown phoneme in {0} ({1} chars long): {2}'.format(x,
                                                                            len(x),
                                                                            p))

        if not self.left:
            grid = grid[::-1]

        # convert syllabic grid to vector
        phon_vector = []
        for phon in grid:
            phon_vector.extend(self.phonemes[phon])

        return phon_vector

    def vectorize(self, X):

        results = []

        for x in X:
            try:
                results.append(self.vectorize_single(x))
            except ValueError:
                pass

        return np.array(results)

if __name__ == "__main__":
    # some test cases
    pass