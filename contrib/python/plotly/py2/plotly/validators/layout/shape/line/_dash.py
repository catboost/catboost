import _plotly_utils.basevalidators


class DashValidator(_plotly_utils.basevalidators.StringValidator):
    def __init__(self, plotly_name="dash", parent_name="layout.shape.line", **kwargs):
        super(DashValidator, self).__init__(
            plotly_name=plotly_name,
            parent_name=parent_name,
            edit_type=kwargs.pop("edit_type", "arraydraw"),
            role=kwargs.pop("role", "style"),
            values=kwargs.pop(
                "values", ["solid", "dot", "dash", "longdash", "dashdot", "longdashdot"]
            ),
            **kwargs
        )
