import _plotly_utils.basevalidators


class ViolinValidator(_plotly_utils.basevalidators.CompoundValidator):
    def __init__(self, plotly_name="violin", parent_name="", **kwargs):
        super(ViolinValidator, self).__init__(
            plotly_name=plotly_name,
            parent_name=parent_name,
            data_class_str=kwargs.pop("data_class_str", "Violin"),
            data_docs=kwargs.pop(
                "data_docs",
                """
            alignmentgroup
                Set several traces linked to the same position
                axis or matching axes to the same
                alignmentgroup. This controls whether bars
                compute their positional range dependently or
                independently.
            bandwidth
                Sets the bandwidth used to compute the kernel
                density estimate. By default, the bandwidth is
                determined by Silverman's rule of thumb.
            box
                :class:`plotly.graph_objects.violin.Box`
                instance or dict with compatible properties
            customdata
                Assigns extra data each datum. This may be
                useful when listening to hover, click and
                selection events. Note that, "scatter" traces
                also appends customdata items in the markers
                DOM elements
            customdatasrc
                Sets the source reference on Chart Studio Cloud
                for  customdata .
            fillcolor
                Sets the fill color. Defaults to a half-
                transparent variant of the line color, marker
                color, or marker line color, whichever is
                available.
            hoverinfo
                Determines which trace information appear on
                hover. If `none` or `skip` are set, no
                information is displayed upon hovering. But, if
                `none` is set, click and hover events are still
                fired.
            hoverinfosrc
                Sets the source reference on Chart Studio Cloud
                for  hoverinfo .
            hoverlabel
                :class:`plotly.graph_objects.violin.Hoverlabel`
                instance or dict with compatible properties
            hoveron
                Do the hover effects highlight individual
                violins or sample points or the kernel density
                estimate or any combination of them?
            hovertemplate
                Template string used for rendering the
                information that appear on hover box. Note that
                this will override `hoverinfo`. Variables are
                inserted using %{variable}, for example "y:
                %{y}". Numbers are formatted using d3-format's
                syntax %{variable:d3-format}, for example
                "Price: %{y:$.2f}".
                https://github.com/d3/d3-3.x-api-
                reference/blob/master/Formatting.md#d3_format
                for details on the formatting syntax. Dates are
                formatted using d3-time-format's syntax
                %{variable|d3-time-format}, for example "Day:
                %{2019-01-01|%A}".
                https://github.com/d3/d3-time-
                format#locale_format for details on the date
                formatting syntax. The variables available in
                `hovertemplate` are the ones emitted as event
                data described at this link
                https://plotly.com/javascript/plotlyjs-
                events/#event-data. Additionally, every
                attributes that can be specified per-point (the
                ones that are `arrayOk: true`) are available.
                Anything contained in tag `<extra>` is
                displayed in the secondary box, for example
                "<extra>{fullData.name}</extra>". To hide the
                secondary box completely, use an empty tag
                `<extra></extra>`.
            hovertemplatesrc
                Sets the source reference on Chart Studio Cloud
                for  hovertemplate .
            hovertext
                Same as `text`.
            hovertextsrc
                Sets the source reference on Chart Studio Cloud
                for  hovertext .
            ids
                Assigns id labels to each datum. These ids for
                object constancy of data points during
                animation. Should be an array of strings, not
                numbers or any other type.
            idssrc
                Sets the source reference on Chart Studio Cloud
                for  ids .
            jitter
                Sets the amount of jitter in the sample points
                drawn. If 0, the sample points align along the
                distribution axis. If 1, the sample points are
                drawn in a random jitter of width equal to the
                width of the violins.
            legendgroup
                Sets the legend group for this trace. Traces
                part of the same legend group hide/show at the
                same time when toggling legend items.
            line
                :class:`plotly.graph_objects.violin.Line`
                instance or dict with compatible properties
            marker
                :class:`plotly.graph_objects.violin.Marker`
                instance or dict with compatible properties
            meanline
                :class:`plotly.graph_objects.violin.Meanline`
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
                the legend item and on hover. For violin
                traces, the name will also be used for the
                position coordinate, if `x` and `x0` (`y` and
                `y0` if horizontal) are missing and the
                position axis is categorical. Note that the
                trace name is also used as a default value for
                attribute `scalegroup` (please see its
                description for details).
            offsetgroup
                Set several traces linked to the same position
                axis or matching axes to the same offsetgroup
                where bars of the same position coordinate will
                line up.
            opacity
                Sets the opacity of the trace.
            orientation
                Sets the orientation of the violin(s). If "v"
                ("h"), the distribution is visualized along the
                vertical (horizontal).
            pointpos
                Sets the position of the sample points in
                relation to the violins. If 0, the sample
                points are places over the center of the
                violins. Positive (negative) values correspond
                to positions to the right (left) for vertical
                violins and above (below) for horizontal
                violins.
            points
                If "outliers", only the sample points lying
                outside the whiskers are shown If
                "suspectedoutliers", the outlier points are
                shown and points either less than 4*Q1-3*Q3 or
                greater than 4*Q3-3*Q1 are highlighted (see
                `outliercolor`) If "all", all sample points are
                shown If False, only the violins are shown with
                no sample points. Defaults to
                "suspectedoutliers" when `marker.outliercolor`
                or `marker.line.outliercolor` is set, otherwise
                defaults to "outliers".
            scalegroup
                If there are multiple violins that should be
                sized according to to some metric (see
                `scalemode`), link them by providing a non-
                empty group id here shared by every trace in
                the same group. If a violin's `width` is
                undefined, `scalegroup` will default to the
                trace's name. In this case, violins with the
                same names will be linked together
            scalemode
                Sets the metric by which the width of each
                violin is determined."width" means each violin
                has the same (max) width*count* means the
                violins are scaled by the number of sample
                points makingup each violin.
            selected
                :class:`plotly.graph_objects.violin.Selected`
                instance or dict with compatible properties
            selectedpoints
                Array containing integer indices of selected
                points. Has an effect only for traces that
                support selections. Note that an empty array
                means an empty selection where the `unselected`
                are turned on for all points, whereas, any
                other non-array values means no selection all
                where the `selected` and `unselected` styles
                have no effect.
            showlegend
                Determines whether or not an item corresponding
                to this trace is shown in the legend.
            side
                Determines on which side of the position value
                the density function making up one half of a
                violin is plotted. Useful when comparing two
                violin traces under "overlay" mode, where one
                trace has `side` set to "positive" and the
                other to "negative".
            span
                Sets the span in data space for which the
                density function will be computed. Has an
                effect only when `spanmode` is set to "manual".
            spanmode
                Sets the method by which the span in data space
                where the density function will be computed.
                "soft" means the span goes from the sample's
                minimum value minus two bandwidths to the
                sample's maximum value plus two bandwidths.
                "hard" means the span goes from the sample's
                minimum to its maximum value. For custom span
                settings, use mode "manual" and fill in the
                `span` attribute.
            stream
                :class:`plotly.graph_objects.violin.Stream`
                instance or dict with compatible properties
            text
                Sets the text elements associated with each
                sample value. If a single string, the same
                string appears over all the data points. If an
                array of string, the items are mapped in order
                to the this trace's (x,y) coordinates. To be
                seen, trace `hoverinfo` must contain a "text"
                flag.
            textsrc
                Sets the source reference on Chart Studio Cloud
                for  text .
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
            unselected
                :class:`plotly.graph_objects.violin.Unselected`
                instance or dict with compatible properties
            visible
                Determines whether or not this trace is
                visible. If "legendonly", the trace is not
                drawn, but can appear as a legend item
                (provided that the legend itself is visible).
            width
                Sets the width of the violin in data
                coordinates. If 0 (default value) the width is
                automatically selected based on the positions
                of other violin traces in the same subplot.
            x
                Sets the x sample data or coordinates. See
                overview for more info.
            x0
                Sets the x coordinate for single-box traces or
                the starting coordinate for multi-box traces
                set using q1/median/q3. See overview for more
                info.
            xaxis
                Sets a reference between this trace's x
                coordinates and a 2D cartesian x axis. If "x"
                (the default value), the x coordinates refer to
                `layout.xaxis`. If "x2", the x coordinates
                refer to `layout.xaxis2`, and so on.
            xsrc
                Sets the source reference on Chart Studio Cloud
                for  x .
            y
                Sets the y sample data or coordinates. See
                overview for more info.
            y0
                Sets the y coordinate for single-box traces or
                the starting coordinate for multi-box traces
                set using q1/median/q3. See overview for more
                info.
            yaxis
                Sets a reference between this trace's y
                coordinates and a 2D cartesian y axis. If "y"
                (the default value), the y coordinates refer to
                `layout.yaxis`. If "y2", the y coordinates
                refer to `layout.yaxis2`, and so on.
            ysrc
                Sets the source reference on Chart Studio Cloud
                for  y .
""",
            ),
            **kwargs
        )
