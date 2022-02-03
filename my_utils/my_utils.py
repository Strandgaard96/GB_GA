from collections import UserDict


class DotDict(UserDict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = UserDict.get
    __setattr__ = UserDict.__setitem__
    __delattr__ = UserDict.__delitem__