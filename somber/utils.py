import time
import sys
import numpy as np


def expo(value, current_step, total_steps):
    """
    Decreases a value X_0
    according to an exponential function with lambda equal to (-2.5 * (current_step / total_steps))

    :param value: The original value.
    :param current_step: The current timestep
    :return:
    """

    return value * np.exp(-2.5 * (current_step / total_steps))


def static(value, current_step, total_steps):
    """
    Static function: nothing changes.

    :param value: the value
    :return:
    """

    return value


def linear(value, current_step, total_steps):
    """
    Decrease a value X_0
    According to a linear function.

    :param value:
    :param current_step:
    :param total_steps:
    :return:
    """

    return (value * (total_steps - current_step) / total_steps) + 0.01


def progressbar(target, width=30, interval=0.2, idx_interval=10, use=True, mult=1):
    """
    Progressbar, partially borrowed from Keras:
    https://github.com/fchollet/keras/blob/088dbe6866fd51f4e0e64866e442968c17abfa10/keras/utils/generic_utils.py

    :param target: The target of the progressbar, must be some kind of iterable
    :param width: The width of the progressbar in characters
    :param interval: The time interval with which to update the bar.
    :param idx_interval: The index interval with which to update the bar. Raise this if the bar is going too fast.
    :param use: Boolean whetehr to actually use the progressbar (for debugging etc.)
    :param mult: The multiplier to multiply the progressbar with, useful for accurately displaying batches.
    e.g. instead of displaying 1/1000 batches of size 100, you can display 100/100000 items.
    :return: None
    """

    start = time.time()
    last_update = 0
    interval = interval
    total_width = 0

    target = list(target)
    iter_length = len(target)

    prev_total_width = 0

    for current, p in enumerate(target):

        if not use or current % idx_interval:
            yield p
            continue

        now = time.time()

        if (now - last_update) < interval:
            yield p
            continue

        prev_total_width = total_width
        sys.stdout.write("\b" * prev_total_width)
        sys.stdout.write("\r")

        numdigits = int(np.floor(np.log10(iter_length))) + 1
        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
        bar = barstr % (current * mult, iter_length * mult)
        prog = float(current) / iter_length
        prog_width = int(width * prog)
        if prog_width > 0:
            bar += ('=' * (prog_width-1))
            if current < iter_length:
                bar += '>'
            else:
                bar += '='
        bar += ('.' * (width - prog_width))
        bar += ']'
        sys.stdout.write(bar)
        total_width = len(bar)

        if current:
            time_per_unit = (now - start) / current
        else:
            time_per_unit = 0
        eta = time_per_unit * (iter_length - current)
        info = ''
        if current < iter_length:
            info += ' - ETA: %ds' % eta
        else:
            info += ' - %ds' % (now - start)

        total_width += len(info)
        if prev_total_width > total_width:
            info += ((prev_total_width - total_width) * " ")

        sys.stdout.write(info)
        sys.stdout.flush()

        if current >= iter_length:
            sys.stdout.write("\n")

        last_update = now

        yield p

    else:

        if not use:
            pass
        else:
            sys.stdout.write("\b" * prev_total_width)
            sys.stdout.write("\r")

            numdigits = int(np.floor(np.log10(iter_length))) + 1
            barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
            bar = barstr % (iter_length * mult, iter_length * mult)
            prog = float(iter_length) / iter_length
            prog_width = int(width * prog)
            if prog_width > 0:
                bar += ('=' * (prog_width-1))
                bar += '='
            bar += ('.' * (width - prog_width))
            bar += ']'
            sys.stdout.write(bar)


class MultiPlexer(object):
    """
    Multiplies a given array n times.
    This is useful in simulating epochs.

    """

    def __init__(self, array, times):
        """
        Multiplies a given np array n times.

        :param array: The array to multiply.
        :param times: The number of times to multiply this array.
        """

        self.array = np.asarray(array)
        self.times = times

        self.new_shape = list(self.array.shape)
        self.new_shape[0] *= times
        self.new_shape = tuple(self.new_shape)

    def __iter__(self):

        for i in range(self.times):

            for item in self.array:

                yield item

    @property
    def shape(self):
        return self.new_shape

    def mean(self, axis=None):

        if axis is None:
            return self.array.mean()
        return self.array.mean(axis=axis)

    def __len__(self):

        return self.new_shape[0]


def reset_context_symbol(X, symbols):
    """
    This function can be used to create a kind of context mask.
    In all SOM models in this package, all sequences are assumed to be
    conditionally dependent on all preceding items. This is usually
    not the case. Therefore, it can be useful to automatically set the
    context to 0 at certain points.

    Given some input, this function generates a 1 if the symbol is not
    in symbols, and a 0 if it is.

    :param X: The input sequence
    :param symbols: A list of symbols which cause the context to be reset to 0
    :return: A list, the size of X, with 1 in places in which the context should
    continue, and 0 in places it should be reset.
    """

    indices = [idx for idx, x in enumerate(X) if x in symbols]
    mask = np.ones((len(X), 1))
    mask[indices] = 0

    return mask