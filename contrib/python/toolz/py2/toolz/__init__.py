from .itertoolz import *

from .functoolz import *

from .dicttoolz import *

from .recipes import *

from .compatibility import map, filter

from functools import partial, reduce

sorted = sorted

# Aliases
comp = compose

from . import curried, sandbox

functoolz._sigs.create_signature_registry()

__version__ = '0.10.0'
