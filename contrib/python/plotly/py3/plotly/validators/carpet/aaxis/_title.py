import _plotly_utils.basevalidators


class TitleValidator(_plotly_utils.basevalidators.TitleValidator):
    def __init__(self, plotly_name="title", parent_name="carpet.aaxis", **kwargs):
        super(TitleValidator, self).__init__(
            plotly_name=plotly_name,
            parent_name=parent_name,
            data_class_str=kwargs.pop("data_class_str", "Title"),
            data_docs=kwargs.pop(
                "data_docs",
                """
            font
                Sets this axis' title font.
            offset
                An additional amount by which to offset the
                title from the tick labels, given in pixels.
            text
                Sets the title of this axis.
""",
            ),
            **kwargs,
        )
