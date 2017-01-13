from matplotlib import pyplot as plt


def show_labelmap(labels, width):
    """
    Shows a map with labels

    :param labels: The labels
    :param width: The width of the map
    :return:
    """

    plt.close()
    plt.axis([0, width-1, 0, width-1])

    for idx, l in enumerate(labels):

        plt.annotate(l, ((idx // width), (idx % width)), size=5)

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
        f.axes("off")
        plt.imshow(map.reshape(width, height))

    plt.show()