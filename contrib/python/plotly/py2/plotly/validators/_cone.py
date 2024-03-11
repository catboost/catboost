import _plotly_utils.basevalidators


class ConeValidator(_plotly_utils.basevalidators.CompoundValidator):
    def __init__(self, plotly_name="cone", parent_name="", **kwargs):
        super(ConeValidator, self).__init__(
            plotly_name=plotly_name,
            parent_name=parent_name,
            data_class_str=kwargs.pop("data_class_str", "Cone"),
            data_docs=kwargs.pop(
                "data_docs",
                """
            anchor
                Sets the cones' anchor with respect to their
                x/y/z positions. Note that "cm" denote the
                cone's center of mass which corresponds to 1/4
                from the tail to tip.
            autocolorscale
                Determines whether the colorscale is a default
                palette (`autocolorscale: true`) or the palette
                determined by `colorscale`. In case
                `colorscale` is unspecified or `autocolorscale`
                is true, the default  palette will be chosen
                according to whether numbers in the `color`
                array are all positive, all negative or mixed.
            cauto
                Determines whether or not the color domain is
                computed with respect to the input data (here
                u/v/w norm) or the bounds set in `cmin` and
                `cmax`  Defaults to `false` when `cmin` and
                `cmax` are set by the user.
            cmax
                Sets the upper bound of the color domain. Value
                should have the same units as u/v/w norm and if
                set, `cmin` must be set as well.
            cmid
                Sets the mid-point of the color domain by
                scaling `cmin` and/or `cmax` to be equidistant
                to this point. Value should have the same units
                as u/v/w norm. Has no effect when `cauto` is
                `false`.
            cmin
                Sets the lower bound of the color domain. Value
                should have the same units as u/v/w norm and if
                set, `cmax` must be set as well.
            coloraxis
                Sets a reference to a shared color axis.
                References to these shared color axes are
                "coloraxis", "coloraxis2", "coloraxis3", etc.
                Settings for these shared color axes are set in
                the layout, under `layout.coloraxis`,
                `layout.coloraxis2`, etc. Note that multiple
                color scales can be linked to the same color
                axis.
            colorbar
                :class:`plotly.graph_objects.cone.ColorBar`
                instance or dict with compatible properties
            colorscale
                Sets the colorscale. The colorscale must be an
                array containing arrays mapping a normalized
                value to an rgb, rgba, hex, hsl, hsv, or named
                color string. At minimum, a mapping for the
                lowest (0) and highest (1) values are required.
                For example, `[[0, 'rgb(0,0,255)'], [1,
                'rgb(255,0,0)']]`. To control the bounds of the
                colorscale in color space, use`cmin` and
                `cmax`. Alternatively, `colorscale` may be a
                palette name string of the following list: Grey
                s,YlGnBu,Greens,YlOrRd,Bluered,RdBu,Reds,Blues,
                Picnic,Rainbow,Portland,Jet,Hot,Blackbody,Earth
                ,Electric,Viridis,Cividis.
            customdata
                Assigns extra data each datum. This may be
                useful when listening to hover, click and
                selection events. Note that, "scatter" traces
                also appends customdata items in the markers
                DOM elements
            customdatasrc
                Sets the source reference on Chart Studio Cloud
                for  customdata .
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
                :class:`plotly.graph_objects.cone.Hoverlabel`
                instance or dict with compatible properties
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
                variable `norm` Anything contained in tag
                `<extra>` is displayed in the secondary box,
                for example "<extra>{fullData.name}</extra>".
                To hide the secondary box completely, use an
                empty tag `<extra></extra>`.
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
            legendgroup
                Sets the legend group for this trace. Traces
                part of the same legend group hide/show at the
                same time when toggling legend items.
            lighting
                :class:`plotly.graph_objects.cone.Lighting`
                instance or dict with compatible properties
            lightposition
                :class:`plotly.graph_objects.cone.Lightposition
                ` instance or dict with compatible properties
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
            opacity
                Sets the opacity of the surface. Please note
                that in the case of using high `opacity` values
                for example a value greater than or equal to
                0.5 on two surfaces (and 0.25 with four
                surfaces), an overlay of multiple transparent
                surfaces may not perfectly be sorted in depth
                by the webgl API. This behavior may be improved
                in the near future and is subject to change.
            reversescale
                Reverses the color mapping if true. If true,
                `cmin` will correspond to the last color in the
                array and `cmax` will correspond to the first
                color.
            scene
                Sets a reference between this trace's 3D
                coordinate system and a 3D scene. If "scene"
                (the default value), the (x,y,z) coordinates
                refer to `layout.scene`. If "scene2", the
                (x,y,z) coordinates refer to `layout.scene2`,
                and so on.
            showlegend
                Determines whether or not an item corresponding
                to this trace is shown in the legend.
            showscale
                Determines whether or not a colorbar is
                displayed for this trace.
            sizemode
                Determines whether `sizeref` is set as a
                "scaled" (i.e unitless) scalar (normalized by
                the max u/v/w norm in the vector field) or as
                "absolute" value (in the same units as the
                vector field).
            sizeref
                Adjusts the cone size scaling. The size of the
                cones is determined by their u/v/w norm
                multiplied a factor and `sizeref`. This factor
                (computed internally) corresponds to the
                minimum "time" to travel across two successive
                x/y/z positions at the average velocity of
                those two successive positions. All cones in a
                given trace use the same factor. With
                `sizemode` set to "scaled", `sizeref` is
                unitless, its default value is 0.5 With
                `sizemode` set to "absolute", `sizeref` has the
                same units as the u/v/w vector field, its the
                default value is half the sample's maximum
                vector norm.
            stream
                :class:`plotly.graph_objects.cone.Stream`
                instance or dict with compatible properties
            text
                Sets the text elements associated with the
                cones. If trace `hoverinfo` contains a "text"
                flag and "hovertext" is not set, these elements
                will be seen in the hover labels.
            textsrc
                Sets the source reference on Chart Studio Cloud
                for  text .
            u
                Sets the x components of the vector field.
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
            usrc
                Sets the source reference on Chart Studio Cloud
                for  u .
            v
                Sets the y components of the vector field.
            visible
                Determines whether or not this trace is
                visible. If "legendonly", the trace is not
                drawn, but can appear as a legend item
                (provided that the legend itself is visible).
            vsrc
                Sets the source reference on Chart Studio Cloud
                for  v .
            w
                Sets the z components of the vector field.
            wsrc
                Sets the source reference on Chart Studio Cloud
                for  w .
            x
                Sets the x coordinates of the vector field and
                of the displayed cones.
            xsrc
                Sets the source reference on Chart Studio Cloud
                for  x .
            y
                Sets the y coordinates of the vector field and
                of the displayed cones.
            ysrc
                Sets the source reference on Chart Studio Cloud
                for  y .
            z
                Sets the z coordinates of the vector field and
                of the displayed cones.
            zsrc
                Sets the source reference on Chart Studio Cloud
                for  z .
""",
            ),
            **kwargs
        )
