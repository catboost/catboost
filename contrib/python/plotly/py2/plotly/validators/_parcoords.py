import _plotly_utils.basevalidators


class ParcoordsValidator(_plotly_utils.basevalidators.CompoundValidator):
    def __init__(self, plotly_name="parcoords", parent_name="", **kwargs):
        super(ParcoordsValidator, self).__init__(
            plotly_name=plotly_name,
            parent_name=parent_name,
            data_class_str=kwargs.pop("data_class_str", "Parcoords"),
            data_docs=kwargs.pop(
                "data_docs",
                """
            customdata
                Assigns extra data each datum. This may be
                useful when listening to hover, click and
                selection events. Note that, "scatter" traces
                also appends customdata items in the markers
                DOM elements
            customdatasrc
                Sets the source reference on Chart Studio Cloud
                for  customdata .
            dimensions
                The dimensions (variables) of the parallel
                coordinates chart. 2..60 dimensions are
                supported.
            dimensiondefaults
                When used in a template (as layout.template.dat
                a.parcoords.dimensiondefaults), sets the
                default property values to use for elements of
                parcoords.dimensions
            domain
                :class:`plotly.graph_objects.parcoords.Domain`
                instance or dict with compatible properties
            ids
                Assigns id labels to each datum. These ids for
                object constancy of data points during
                animation. Should be an array of strings, not
                numbers or any other type.
            idssrc
                Sets the source reference on Chart Studio Cloud
                for  ids .
            labelangle
                Sets the angle of the labels with respect to
                the horizontal. For example, a `tickangle` of
                -90 draws the labels vertically. Tilted labels
                with "labelangle" may be positioned better
                inside margins when `labelposition` is set to
                "bottom".
            labelfont
                Sets the font for the `dimension` labels.
            labelside
                Specifies the location of the `label`. "top"
                positions labels above, next to the title
                "bottom" positions labels below the graph
                Tilted labels with "labelangle" may be
                positioned better inside margins when
                `labelposition` is set to "bottom".
            line
                :class:`plotly.graph_objects.parcoords.Line`
                instance or dict with compatible properties
            meta
                Assigns extra meta information associated with
                this trace that can be used in various text
                attributes. Attributes such as trace `name`,
                graph, axis and colorbar `title.text`,
                annotation `text` `rangeselector`,
                `updatemenues` and `sliders` `label` text all
                support `meta`. To access the trace `meta`
                values in an attribute in the same trace,
                simply use `%{meta[i]}` where `i` is the index
                or key of the `meta` item in question. To
                access trace `meta` in layout attributes, use
                `%{data[n[.meta[i]}` where `i` is the index or
                key of the `meta` and `n` is the trace index.
            metasrc
                Sets the source reference on Chart Studio Cloud
                for  meta .
            name
                Sets the trace name. The trace name appear as
                the legend item and on hover.
            rangefont
                Sets the font for the `dimension` range values.
            stream
                :class:`plotly.graph_objects.parcoords.Stream`
                instance or dict with compatible properties
            tickfont
                Sets the font for the `dimension` tick values.
            uid
                Assign an id to this trace, Use this to provide
                object constancy between traces during
                animations and transitions.
            uirevision
                Controls persistence of some user-driven
                changes to the trace: `constraintrange` in
                `parcoords` traces, as well as some `editable:
                true` modifications such as `name` and
                `colorbar.title`. Defaults to
                `layout.uirevision`. Note that other user-
                driven trace attribute changes are controlled
                by `layout` attributes: `trace.visible` is
                controlled by `layout.legend.uirevision`,
                `selectedpoints` is controlled by
                `layout.selectionrevision`, and
                `colorbar.(x|y)` (accessible with `config:
                {editable: true}`) is controlled by
                `layout.editrevision`. Trace changes are
                tracked by `uid`, which only falls back on
                trace index if no `uid` is provided. So if your
                app can add/remove traces before the end of the
                `data` array, such that the same trace has a
                different index, you can still preserve user-
                driven changes if you give each trace a `uid`
                that stays with it as it moves.
            visible
                Determines whether or not this trace is
                visible. If "legendonly", the trace is not
                drawn, but can appear as a legend item
                (provided that the legend itself is visible).
""",
            ),
            **kwargs
        )
