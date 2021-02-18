from warnings import warn

from . import traitlets
from .traitlets import *
from .utils.importstring import import_item
from .utils.decorators import signature_has_traits
from .utils.bunch import Bunch
from ._version import version_info, __version__


class Sentinel(traitlets.Sentinel):
    def __init__(self, *args, **kwargs):
        super(Sentinel, self).__init__(*args, **kwargs)
        warn(
            """
            Sentinel is not a public part of the traitlets API.
            It was published by mistake, and may be removed in the future.
            """,
            DeprecationWarning,
            stacklevel=2,
        )
