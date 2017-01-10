import numpy as np


class MarkovGenerator(object):

    def __init__(self, representations, transitions, start_probabilities, seed=44):

        np.random.seed(seed)

        check = transitions.sum(axis=1)
        if not np.all([check == 1.0]):
            raise ValueError("Probabilities do not add up to 1")

        assert len(representations) == len(transitions)

        self.representations = np.array(representations)
        self.transitions = transitions
        self.choices = np.arange(len(transitions))
        self.start = start_probabilities

    def generate_sequences(self, numsequences, seqlen):

        output = []

        for x in range(numsequences):

            pos = np.random.choice(a=self.choices, p=self.start)
            seq = []

            for y in range(seqlen):

                chosen = np.random.choice(a=self.choices, p=self.transitions[pos])
                seq.append(self.representations[chosen])
                pos = chosen

            output.append(np.array(seq))

        return np.array(output)

if __name__ == "__main__":

    transition = np.array([[0.2, 0.3, 0.0, 0.5], [0.0, 0.0, 0.3, 0.7], [0.5, 0.25, 0.125, 0.125], [1.0, 0.0, 0.0, 0.0]])

    m = MarkovGenerator(transition)
