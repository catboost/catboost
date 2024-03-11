import sys

if sys.version_info < (3, 7):
    from ._error_x import ErrorX
    from ._error_y import ErrorY
    from ._error_z import ErrorZ
    from ._hoverlabel import Hoverlabel
    from ._line import Line
    from ._marker import Marker
    from ._projection import Projection
    from ._stream import Stream
    from ._textfont import Textfont
    from . import hoverlabel
    from . import line
    from . import marker
    from . import projection
else:
    from _plotly_utils.importers import relative_import

    __all__, __getattr__, __dir__ = relative_import(
        __name__,
        [".hoverlabel", ".line", ".marker", ".projection"],
        [
            "._error_x.ErrorX",
            "._error_y.ErrorY",
            "._error_z.ErrorZ",
            "._hoverlabel.Hoverlabel",
            "._line.Line",
            "._marker.Marker",
            "._projection.Projection",
            "._stream.Stream",
            "._textfont.Textfont",
        ],
    )
