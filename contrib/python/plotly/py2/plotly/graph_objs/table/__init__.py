import sys

if sys.version_info < (3, 7):
    from ._cells import Cells
    from ._domain import Domain
    from ._header import Header
    from ._hoverlabel import Hoverlabel
    from ._stream import Stream
    from . import cells
    from . import header
    from . import hoverlabel
else:
    from _plotly_utils.importers import relative_import

    __all__, __getattr__, __dir__ = relative_import(
        __name__,
        [".cells", ".header", ".hoverlabel"],
        [
            "._cells.Cells",
            "._domain.Domain",
            "._header.Header",
            "._hoverlabel.Hoverlabel",
            "._stream.Stream",
        ],
    )
