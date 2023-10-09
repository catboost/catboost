import json
import os
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

    def write_plot_file(plot_file_stream):
        plot_file_stream.write('\n'.join((
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
            plot_file_stream.write('\n{}\n'.format(graph_div))
        plot_file_stream.write('</body>\n</html>')

    if isinstance(plot_file, str):
        with open(fspath(plot_file), 'w') as plot_file_stream:
            write_plot_file(plot_file_stream)
    else:
        write_plot_file(plot_file)


class OfflineMetricVisualizer(object):
    def __init__(self, train_dirs):
        if isinstance(train_dirs, str):
            train_dirs = [train_dirs]
        self._load_data(train_dirs)

    def _update_data_from_dir(self, path):
        data = {
            'iterations': [],
            'meta': {}
        }

        training_json = os.path.join(path, 'catboost_training.json')

        if os.path.isfile(training_json):
            try:
                with open(training_json, 'r') as json_data:
                    training_data = json.load(json_data)
                    data['meta'] = training_data['meta']
                    data['iterations'] = training_data['iterations']
            except ValueError:
                pass

        return {
            'passed_iterations': data['iterations'][-1]['iteration'] if data['iterations'] else 0,
            'total_iterations': data['meta']['iteration_count'] if data['meta'] else 0,
            'data': data
        }

    def _load_data(self, train_dirs):
        names = []
        curdir = os.path.abspath(os.path.curdir)
        for train_dir in train_dirs:
            abspath = os.path.abspath(train_dir)
            names.append(os.path.basename(abspath) if abspath != curdir else 'current')

        data = {}
        dirs = [{'name': name, 'path': path} for name, path in zip(names, train_dirs)]

        for dir_info in dirs:
            path = dir_info.get('path')
            content = self._update_data_from_dir(path)

            if not content:
                continue

            data[path] = {
                'path': path,
                'name': dir_info.get('name'),
                'content': content
            }
        self.data = data

    def _get_plotly_figs(self, title):
        try:
            import plotly.graph_objs as go
        except ImportError as err:
            warnings.warn("To save plots to files you should install plotly.")
            raise ImportError(str(err))

        figs = []
        for path, dir_data in self.data.items():
            meta = dir_data['content']['data']['meta']

            has_parameters = meta['parameters'] == 'parameters'

            # name -> {'learn_sets': [(name, metric_idx)], 'test_sets': [(name, metric_idx)]}
            metrics = {}
            for i, learn_metric in enumerate(meta['learn_metrics']):
                metric_name = learn_metric['name']
                metrics.setdefault(metric_name, {})
                metrics[metric_name]['learn_sets'] = [(set_name, i) for set_name in meta['learn_sets']]
            for i, test_metric in enumerate(meta['test_metrics']):
                metric_name = test_metric['name']
                metrics.setdefault(metric_name, {})
                metrics[metric_name]['test_sets'] = [(set_name, i) for set_name in meta['test_sets']]

            iterations = dir_data['content']['data']['iterations']

            for metric_name, subsets in metrics.items():
                fig = go.Figure()

                figure_title = (title if (dir_data['name'] == 'catboost_info') else dir_data['name']) + ' : ' + metric_name
                fig['layout']['title'] = go.layout.Title(text=figure_title)

                learn_graph_color = 'rgb(160,0,0)'

                for learn_set_name, metric_idx in subsets.get('learn_sets', []):
                    fig.add_trace(go.Scatter(
                        x=[e['iteration'] for e in iterations],
                        y=[e[learn_set_name][metric_idx] for e in iterations],
                        line=go.scatter.Line(color=learn_graph_color),
                        mode='lines',
                        name=learn_set_name
                    ))

                test_graph_color = 'rgb(0,160,0)'

                for test_set_name, metric_idx in subsets.get('test_sets', []):

                    def generate_params_hover(e):
                        result = []
                        for param_name, param_value in e["parameters"][0].items():
                            result.append(param_name + ' : ' + str(param_value))
                        return '<br>'.join(result)

                    fig.add_trace(go.Scatter(
                        x=[e['iteration'] for e in iterations],
                        y=[e[test_set_name][metric_idx] for e in iterations],
                        line=go.scatter.Line(color=test_graph_color),
                        mode='lines',
                        name=test_set_name,
                        hovertext=[generate_params_hover(e) for e in iterations] if has_parameters else None
                    ))

                fig.update_layout(
                    xaxis=dict(title='iterations'),
                    yaxis=dict(title=metric_name)
                )

                figs.append(fig)

        return figs

    def save_to_file(self, title, file_output):
        save_plot_file(file_output, title, self._get_plotly_figs(title))
