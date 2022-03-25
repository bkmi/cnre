try:
    from cnre.__version__ import version as __version__
except ModuleNotFoundError:
    __version__ = ""

from cnre.experiments import *
