import numpy as np

from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from pylab import rcParams
rcParams['figure.figsize'] = 10, 10

v = np.array([[0.267004,  0.004874,  0.329415],
              [0.270595,  0.214069,  0.507052],
              [0.19943,  0.387607,  0.554642],
              [0.13777,  0.537492,  0.554906],
              [0.157851,  0.683765,  0.501686],
              [0.440137,  0.811138,  0.340967]])


l = ListedColormap(v)


def show_labelmap(labels, width, lang_dict={}):
    """
    Shows a map with labels

    :param labels: The labels
    :param width: The width of the map
    :return:
    """

    plt.axis([0, width, 0, width])

    loc_dict = {k: 0 for k in range(width * width)}

    for k, v in labels.items():

        for word in v:

            try:
                if lang_dict[word] == 'both':
                    color = 'teal'
                elif lang_dict[word] == 'eng':
                    color = 'white'
                else:
                    color = 'magenta'
            except KeyError:
                color = 'white'

            plt.annotate(word, ((k // width) + 0.1, (k % width) + 0.05 + (0.2 * loc_dict[k])), size=10, color=color)
            loc_dict[k] += 1

    # plt.show()


def show_label_activation_map(labels, width, activation, interpolation='sinc', cmap=None, langdict={}):

    plt.axis([0, width, 0, width])
    show_labelmap(labels, width, langdict)

    if cmap is None:
        cmap = l

    a = np.zeros((width, width))
    for idx, x in enumerate(activation):
        a[idx % width, idx // width] = x

    plt.imshow(a, extent=[0, width, width, 0], interpolation=interpolation, cmap=cmap)


def show_label_scatter_map(labels, width, arrow_vectors):

    plt.axis([0, width, 0, width])
    show_labelmap(labels, width)

    colors = ['pink']

    for idx in range(len(arrow_vectors)):

        a = arrow_vectors[idx].ravel()
        jitter = (np.random.random(len(a) * 2).reshape(len(a), 2) - 0.5) / 2

        x, y = zip(*[((x // width) + jitter[idx][0], (x % width) + jitter[idx][1]) for idx, x in enumerate(a)])
        x = np.array(x) + 0.5
        y = np.array(y) + 0.5
        plt.scatter(x, y, color=colors[0], alpha=0.8 - ((0.5 / len(arrow_vectors)) * idx), s=30.0 - ((10.0 / len(arrow_vectors)) * idx))

    plt.show()


def show_label_arrow_map(labels, width, arrow_vectors, colors=('magenta',)):

    plt.axis([0, width, 0, width])

    show_labelmap(labels, width)

    curr_vectors = np.array([(x // width, x % width) for x in arrow_vectors[0].ravel()])

    for idx in range(len(arrow_vectors)-1):

        size = len(curr_vectors)

        a = arrow_vectors[idx+1]
        curr_vectors = np.array([[curr_vectors[i]] * size for i in range(len(curr_vectors))])
        curr_vectors = curr_vectors.reshape(len(curr_vectors.ravel()) // 2, 2)

        vec = np.array([((x // width) + ((np.random.random() - 0.5) * 0.1), (x % width) + ((np.random.random() - 0.5) * 0.1)) for idx, x in enumerate(a.ravel())])

        x = np.array([curr_vectors[:, 0], vec[:, 0]]).T + 0.5
        y = np.array([curr_vectors[:, 1], vec[:, 1]]).T + 0.5

        for i in range(len(x)):
            x_, y_ = x[i, 0], y[i, 0]
            dx, dy = x[i, 1] - x_, y[i, 1] - y_

            plt.arrow(x_, y_, dx, dy,
                      color=colors[idx%len(colors)],
                      alpha=(0.5 / (len(arrow_vectors) * (idx+1))) + 0.5,
                      linewidth=1.0)
        curr_vectors = vec

    plt.show()

def context_map(contexts, width, height):
    """
    To be used with a recursive SOM.
    In a recursive SOM, each context is a full copy of the map.
    Hence, we need lots of subplots to show the effect of context.

    :param contexts:
    :param width:
    :param height:
    :return:
    """

    plt.close()

    for idx, map in enumerate(contexts):

        f = plt.subplot(width, height, idx+1)
        f.axis("off")
        plt.imshow(map.reshape(width, height).transpose(), vmax=1.0, vmin=0.0)

    plt.show()