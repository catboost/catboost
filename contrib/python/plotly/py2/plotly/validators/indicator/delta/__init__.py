import sys

if sys.version_info < (3, 7):
    from ._valueformat import ValueformatValidator
    from ._relative import RelativeValidator
    from ._reference import ReferenceValidator
    from ._position import PositionValidator
    from ._increasing import IncreasingValidator
    from ._font import FontValidator
    from ._decreasing import DecreasingValidator
else:
    from _plotly_utils.importers import relative_import

    __all__, __getattr__, __dir__ = relative_import(
        __name__,
        [],
        [
            "._valueformat.ValueformatValidator",
            "._relative.RelativeValidator",
            "._reference.ReferenceValidator",
            "._position.PositionValidator",
            "._increasing.IncreasingValidator",
            "._font.FontValidator",
            "._decreasing.DecreasingValidator",
        ],
    )
