import numpy as np
import pandas as pd

from som import Som
from utils import MultiPlexer


def create_square_data():

    dlen = 200
    data1 = pd.DataFrame(data= 1 * np.random.rand(dlen, 2))
    data1.values[:, 1] = (data1.values[:, 0][:, np.newaxis] + .42 * np.random.rand(dlen, 1))[:, 0]

    data2 = pd.DataFrame(data= 1 * np.random.rand(dlen, 2) + 1)
    data2.values[:, 1] = (-1 * data2.values[:, 0][:, np.newaxis] + .62 * np.random.rand(dlen, 1))[:, 0]

    data3 = pd.DataFrame(data=1 * np.random.rand(dlen, 2) + 2)
    data3.values[:, 1] = (.5 * data3.values[:, 0][:, np.newaxis] + 1 * np.random.rand(dlen, 1))[:, 0]

    data4 = pd.DataFrame(data=1 * np.random.rand(dlen, 2)+3.5)
    data4.values[:, 1] = (-.1 * data4.values[:, 0][:, np.newaxis] + .5 * np.random.rand(dlen, 1))[:, 0]

    return data4


def create_data():

    dlen = 700
    tetha = np.random.uniform(low=0,high= 2 * np.pi, size=dlen)[:, np.newaxis]
    X1 = 3 * np.cos(tetha) + .22 * np.random.rand(dlen, 1)
    Y1 = 3 * np.sin(tetha) + .22 * np.random.rand(dlen, 1)
    data1 = np.concatenate((X1, Y1), axis=1)

    X2 = 1 * np.cos(tetha) + .22 * np.random.rand(dlen, 1)
    Y2 = 1 * np.sin(tetha) + .22 * np.random.rand(dlen, 1)
    data2 = np.concatenate((X2, Y2), axis=1)

    X3 = 5 * np.cos(tetha) + .22 * np.random.rand(dlen, 1)
    Y3 = 5 * np.sin(tetha) + .22 * np.random.rand(dlen, 1)
    data3 = np.concatenate((X3, Y3), axis=1)

    X4 = 8 * np.cos(tetha) + .22 * np.random.rand(dlen, 1)
    Y4 = 8* np.sin(tetha) + .22 * np.random.rand(dlen, 1)
    data4 = np.concatenate((X4, Y4), axis=1)

    return np.concatenate((data1, data2, data3, data4),axis=0)

if __name__ == "__main__":

    X = create_square_data()
    d = MultiPlexer(X, 1000)

    s = Som((30, 30), 2, 1.0)

    s.train(d, 100)
    error = s.quant_error(X)
    print(np.mean(error))