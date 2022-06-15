try:
    from cnre.__version__ import version as __version__
except ModuleNotFoundError:
    __version__ = ""

import cnre.algorithms
from cnre.loss import *
from cnre.metrics import *
from cnre.posterior import *
from cnre.simulators import *
