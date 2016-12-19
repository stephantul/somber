import os


def replace_all(text, replace_dict):
    """
    For every key, value pair in 'replace_dict', if there is a sub-string in 'text' that is equal to the key,
    replace it with the value. Return 'text' with all sub-strings replaced.
    """
    for i, j in replace_dict.items():
        text = text.replace(i, j)
    return text


def get_primary_stress_vector(phonemes_stress, syllable_limit):
    """
    Return a dictionary mapping word strings to lexical stress vectors.

    :type phonemes_stress:     str
    :param phonemes_stress:    word strings for which you want to obtain phonological vectors.
    """
    phonemes_by_syllable = phonemes_stress.split('-')
    stress_markers_by_syllable = [0] * syllable_limit

    for idx, s in enumerate(phonemes_by_syllable):
        if "'" in s:
            stress_markers_by_syllable[idx] = 1

    # function words may be unstressed; else, for all other words, there can only be one primary stress marker
    assert sum(stress_markers_by_syllable) == 0 or sum(stress_markers_by_syllable) == 1

    return stress_markers_by_syllable


def pron_to_dictionary(path, phrases=False, phon_indices=(12, 8)):
    """
    Extract sequences of phonemes and primary stress vectors for each word from the CELEX database.
    Store this information in a dictionary, and return it when you have gone through the entire database.
    """

    # replace keys with values in sequences of phonemes
    replace_to_get_phonemes = {'tS': 'C',   # replace 'tS' with the single-char variant 'C' (used by PatPho)
                               'dZ': 'D',   # replace 'dZ' with the single-char variant 'D' (used by PatPho)
                               '[': '',     # remove square brackets (used to delimit syllables)
                               ']': '',     # see above
                               ':': '',     # colon marks long vowels -- we don't distinguish between long and short
                               ',': '',
                               'r*': '',
                               "'": '',
                               '"': '',}

    # populate and return this with phoneme sequences and primary stress vectors
    words_to_phon = {}

    # keep track of number of words for which we extract a sequence of phonemes
    n_words = 0

    # path to phonology part of the CELEX database
    for line in open(path):

        line = line.strip()
        columns = line.split('\\')

        word = columns[1].lower()
        n_words += 1

        if not phrases:
            if ' ' in word:
                continue

        # try to use the reduced, probably more frequent stylistic variant (13th column of the record)
        try:
            phonemes = columns[phon_indices[0]]
        # but use standard variant if there isn't a reduced variant (9th column of the record)
        except IndexError:
            try:
                phonemes = columns[phon_indices[1]]
            except IndexError:
                print("skipped {0}".format(columns))
                continue

        if not phonemes:
            continue

        # replace certain substrings
        phonemes = replace_all(phonemes, replace_to_get_phonemes)

        # don't store extracted material if we already have an entry for the word
        # this means we may end up extracting material for e.g. a verb when we really have a noun
        # this is not a problem for the small CDI vocabulary
        if word in words_to_phon:
            continue
        else:
            words_to_phon[word] = phonemes

    print('Nr of words: %s' % n_words)
    print('But I only kept: %s (the rest had multiple entries)' % len(words_to_phon))

    return words_to_phon


if __name__ == '__main__':
    # extract information from database and pickle result to dictionary
    epw_dictionary = pron_to_dictionary("../data/dpl.cd")
