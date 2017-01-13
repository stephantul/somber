from matplotlib import pyplot as plt


def show_labelmap(labels, width):

    plt.close()
    plt.axis([0, width-1, 0, width-1])

    for idx, l in enumerate(labels):

        plt.annotate(l, ((idx // width), (idx % width)), size=5)

    plt.show()


def context_map(contexts, width, height):

    plt.close()
    for idx, map in enumerate(contexts):
        f = plt.subplot(width, height, idx+1)
        f.axes("off")
        plt.imshow(map.reshape(width, height))

    plt.show()