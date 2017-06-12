

class _Flags(dict):

    def __init__(self, defaults):
        super().__init__()
        super().update(defaults)
        # just for the sake of information


_flags = _Flags({"gpu": False})


def Flags(): return _flags


if __name__ == "__main__":

    s1 = Flags()
    s2 = Flags()
