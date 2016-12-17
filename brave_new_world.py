import re
import numpy as np

from preprocessing.ortho import Orthographizer
from msom_no_batch import MSom
from string import ascii_lowercase

removenumbers = re.compile(r"\d")

if __name__ == "__main__":

    brv = " ".join(open("data/brvnwworld.txt").readlines())

    brv = removenumbers.sub(" ", brv)[:1000]

    r = re.compile("\n")
    brv = r.sub(" ", brv)

    print(len(brv))

    # words = brv.split()

    test = ['the', 'cat', 'sat', 'on', 'the', 'mat', 'mask', 'master']

    # maxlen = 10
    # words = filter(lambda x: len(x) <= maxlen, words)

    o = Orthographizer(max_length=len(brv))
    X = np.array(o.fit_transform([brv]))

    # X_test = o.transform(words)

    r = MSom(20, 20, X.shape[2], learning_rate=[0.03], alpha=0.0, beta=0.5)
    r.train(np.squeeze(X), batch_size=1, num_epochs=1000)

    X_letters = np.squeeze(Orthographizer(max_length=26).transform([ascii_lowercase]))
    print(r.predict(X_letters))
