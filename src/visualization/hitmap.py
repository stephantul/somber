from matplotlib import pyplot as plt


def show_labelmap(labels, width):

    plt.axis([0, width-1, 0, width-1])

    for idx, l in enumerate(labels):

        plt.annotate(l, ((idx // width), (idx % width)), size=5)

    plt.show()