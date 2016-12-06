import re
import numpy as np

from preprocessing.ortho import Orthographizer
from thsom import THSom
from som import static

removenumbers = re.compile(r"\d")

if __name__ == "__main__":

    brv = " ".join(open("data/brvnwworld.txt").readlines())

    brv = removenumbers.sub(" ", brv)
    words = brv.split()

    test = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'mask', 'master']

    maxlen = 10
    words = filter(lambda x: len(x) <= maxlen, words)

    o = Orthographizer(max_length=maxlen)
    X = np.array(o.fit_transform(words))

    # X_test = b.transform(test, maxlen=7)

    r = THSom(20, 20, X.shape[2], learning_rates=[1.0], beta=0.01)
    r.train(X, batch_size=100, num_epochs=1000)
