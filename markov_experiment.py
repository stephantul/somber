import numpy as np
import cProfile

from thsom import THSom
from thsom_slow import THSom as THSom_slo
from markov_chain import MarkovGenerator

if __name__ == "__main__":

    np.random.seed(22)

    transition = np.array([[0.2, 0.8, 0.0, 0.0], [0.0, 0.2, 0.8, 0.0], [0.0, 0.0, 0.2, 0.8], [0.8, 0.0, 0.0, 0.2]])
    representations = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    start_probabilities = np.array([1.0, 0.0, 0.0, 0.0])

    m = MarkovGenerator(representations, transition, start_probabilities)
    p = m.generate_sequences(10, 100)

    w = np.random.uniform(0.0, 1.0, size=(4, 2))

    s = THSom(20, 20, 2, [0.1, 0.3], 0.01)
    #s.weights = np.copy(w)
    s.train(p, num_epochs=100, batch_size=10)

    s_2 = THSom_slo(2, 2, 2, [0.1, 0.3], 0.01)
    #s_2.weights = np.copy(w)
    s_2.train(p, num_epochs=100, batch_size=1)

    # p, f = s.get_weights()

    print(s.temporal_weights / s.temporal_weights.sum(axis=0))
    print(transition)

    print(s_2.temporal_weights / s_2.temporal_weights.sum(axis=0))
    print(transition)
