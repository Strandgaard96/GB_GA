from collections import UserDict
import os

class DotDict(UserDict):
    """dot.notation access to dictionary attributes
    Currently not in use as it clashed with Multiprocessing-Pool pickling"""

    __getattr__ = UserDict.get
    __setattr__ = UserDict.__setitem__
    __delattr__ = UserDict.__delitem__


class cd:
    """Context manager for changing the current working directory dynamically.
    # See: https://book.pythontips.com/en/latest/context_managers.html"""

    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)
