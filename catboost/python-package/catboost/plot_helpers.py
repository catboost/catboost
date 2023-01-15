import warnings

from . import _catboost
fspath = _catboost.fspath


def try_plot_offline(figs):
    try:
        from plotly.offline import iplot
        from plotly.offline import init_notebook_mode
        init_notebook_mode(connected=True)
    except ImportError as e:
        warn_msg = "To draw plots you should install plotly."
        warnings.warn(warn_msg)
        raise ImportError(str(e))
    if not isinstance(figs, list):
        figs = [figs]
    for fig in figs:
        iplot(fig)


def save_plot_file(plot_file, plot_name, figs):
    warn_msg = "To draw plots you should install plotly."
    try:
        from plotly.offline import plot as plotly_plot
    except ImportError as e:
        warnings.warn(warn_msg)
        raise ImportError(str(e))

    with open(fspath(plot_file), 'w') as html_plot_file:
        html_plot_file.write('\n'.join((
            '<html>',
            '<head>',
            '<meta charset="utf-8" />',
            '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>',
            '<title>{}</title>'.format(plot_name),
            '</head>',
            '<body>'
        )))
        for fig in figs:
            graph_div = plotly_plot(
                fig,
                output_type='div',
                show_link=False,
                include_plotlyjs=False
            )
            html_plot_file.write('\n{}\n'.format(graph_div))
        html_plot_file.write('</body>\n</html>')
