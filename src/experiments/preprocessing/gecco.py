import csv
import numpy as np

from collections import defaultdict

# TODO: look up what demberg and keller threw out. Throw the same stuff out.
# TODO: do the same type of modeling as demberg and keller in terms of variables.
# TODO: Q: in what variables does the difference between L1 and L2 show?


class GeccoReader(object):
    """
    Contains a minimal implementation for reading the file format in which Gecko is presented
    Does not assume L1, L2 etc.

    Class assumes that the data has been converted to csv, instead of XLSX.

    LEGEND:

    PP_NR,
    GROUP,
    LANGUAGE_RANK,
    LANGUAGE,
    PART,
    TRIAL,
    TRIAL_FIXATION_COUNT,
    TRIAL_TOTAL_READING_TIME,
    WORD_ID_WITHIN_TRIAL,
    WORD_ID,
    WORD,
    WORD_AVERAGE_FIX_PUPIL_SIZE,
    WORD_FIXATION_COUNT,
    WORD_FIXATION_%,
    WORD_RUN_COUNT,
    WORD_FIRST_RUN_START_TIME,
    WORD_FIRST_RUN_END_TIME,
    WORD_FIRST_RUN_FIXATION_COUNT,
    WORD_FIRST_RUN_FIXATION_%,
    WORD_GAZE_DURATION,
    WORD_SECOND_RUN_START_TIME,
    WORD_SECOND_RUN_END_TIME,
    WORD_SECOND_RUN_FIXATION_COUNT,
    WORD_SECOND_RUN_FIXATION_%,
    WORD_THIRD_RUN_START_TIME,
    WORD_THIRD_RUN_END_TIME,
    WORD_THIRD_RUN_FIXATION_COUNT,
    WORD_THIRD_RUN_FIXATION_%,
    WORD_FIRST_FIXATION_DURATION,
    WORD_FIRST_FIXATION_INDEX,
    WORD_FIRST_FIXATION_RUN_INDEX,
    WORD_FIRST_FIXATION_TIME,
    WORD_FIRST_FIXATION_VISITED_WORD_COUNT,
    WORD_FIRST_FIXATION_X,
    WORD_FIRST_FIXATION_Y,
    WORD_FIRST_FIX_PROGRESSIVE,
    WORD_SECOND_FIXATION_DURATION,
    WORD_SECOND_FIXATION_RUN,
    WORD_SECOND_FIXATION_TIME,
    WORD_SECOND_FIXATION_X,
    WORD_SECOND_FIXATION_Y,
    WORD_THIRD_FIXATION_DURATION,
    WORD_THIRD_FIXATION_RUN,
    WORD_THIRD_FIXATION_TIME,
    WORD_THIRD_FIXATION_X,
    WORD_THIRD_FIXATION_Y,
    WORD_LAST_FIXATION_DURATION,
    WORD_LAST_FIXATION_RUN,
    WORD_LAST_FIXATION_TIME,
    WORD_LAST_FIXATION_X,
    WORD_LAST_FIXATION_Y,
    WORD_GO_PAST_TIME,
    WORD_SELECTIVE_GO_PAST_TIME,
    WORD_TOTAL_READING_TIME,
    WORD_TOTAL_READING_TIME_%,
    WORD_SPILLOVER,
    WORD_SKIP
    """

    def __init__(self, filename):

        reader = csv.reader(open(filename, encoding='latin-1'), dialect='excel')
        subjects = defaultdict(list)

        legend = {k: idx-1 for idx, k in enumerate(next(reader))}

        for x in reader:
            subjects[x[0]].append(x[1:])

        self.subjects = [Subject(k, v, legend) for k, v in sorted(subjects.items(), key=lambda x: x[0])]
        self.reading_matrix = np.array([s.reading_sequence() for s in self.subjects])
        self.corpus = [[d[legend['WORD']] for d in s.data] for s in self.subjects]

class Subject(object):
    """
    A class which facilitates storage of data on a subject basis.
    """

    def __init__(self, key, subject_dict, legend):

        self.id = key
        self.data = subject_dict
        self.aggregated_words = defaultdict(list)
        for d in self.data:

            try:
                r_time = int(d[legend['WORD_TOTAL_READING_TIME']])
            except ValueError:
                continue

            self.aggregated_words[d[legend['WORD']]].append(r_time)

        self.legend = legend

    def reading_sequence(self):

        return [t if t != '.' else 0 for t in [d[self.legend['WORD_TOTAL_READING_TIME']] for d in self.data]]

    def reading_sequence_with_words(self):

        return [t if t[0] != '.' else (0, t[1]) for t in [(d[self.legend['WORD_TOTAL_READING_TIME']], d[self.legend['WORD']]) for d in self.data]]

    def reading_time(self, word):

        return self.aggregated_words[word]

    def average_reading_time(self, word):

        times = self.reading_time(word)

        if not times:
            return 0

        return sum(times) / len(times)

if __name__ == "__main__":

    g = GeccoReader("../data/L1ReadingData.csv")
    g_2 = GeccoReader("../data/L2ReadingData.csv")
    print('done')
