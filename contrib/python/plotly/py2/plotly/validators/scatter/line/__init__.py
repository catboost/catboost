import sys

if sys.version_info < (3, 7):
    from ._width import WidthValidator
    from ._smoothing import SmoothingValidator
    from ._simplify import SimplifyValidator
    from ._shape import ShapeValidator
    from ._dash import DashValidator
    from ._color import ColorValidator
else:
    from _plotly_utils.importers import relative_import

    __all__, __getattr__, __dir__ = relative_import(
        __name__,
        [],
        [
            "._width.WidthValidator",
            "._smoothing.SmoothingValidator",
            "._simplify.SimplifyValidator",
            "._shape.ShapeValidator",
            "._dash.DashValidator",
            "._color.ColorValidator",
        ],
    )
