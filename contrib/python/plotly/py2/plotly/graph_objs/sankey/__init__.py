import sys

if sys.version_info < (3, 7):
    from ._domain import Domain
    from ._hoverlabel import Hoverlabel
    from ._link import Link
    from ._node import Node
    from ._stream import Stream
    from ._textfont import Textfont
    from . import hoverlabel
    from . import link
    from . import node
else:
    from _plotly_utils.importers import relative_import

    __all__, __getattr__, __dir__ = relative_import(
        __name__,
        [".hoverlabel", ".link", ".node"],
        [
            "._domain.Domain",
            "._hoverlabel.Hoverlabel",
            "._link.Link",
            "._node.Node",
            "._stream.Stream",
            "._textfont.Textfont",
        ],
    )
