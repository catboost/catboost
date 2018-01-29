import os
import csv
import time
from threading import Thread
from IPython.core.display import display, HTML
from traitlets import Unicode, Dict, default
from ipywidgets import DOMWidget, Layout, widget_serialization


class CatboostIpythonWidget(DOMWidget):
    _view_name = Unicode('CatboostIpythonWidgetView').tag(sync=True)
    _view_module = Unicode('catboost_module').tag(sync=True)

    data = Dict({}).tag(sync=True, **widget_serialization)

    def __init__(self, train_dir):
        super(self.__class__, self).__init__()
        self._train_dir = train_dir

    @default('layout')
    def _default_layout(self):
        return Layout(height='500px', align_self='stretch')

    def update_widget(self, subdirs=False):
        # wait for start train (meta.tsv)
        self._init_static()
        time.sleep(1.0)
        self._update_data(subdirs=subdirs)

        display(self)

        while self._need_update:
            self._update_data(subdirs=subdirs)
            time.sleep(2.0)

    def _run_update(self):
        thread = Thread(target=self.update_widget, args=())
        thread.start()

    def _get_subdirectories(self, a_dir):
        return [{'name': name, 'path': os.path.join(a_dir, name)}
                    for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

    def _update_data(self, subdirs=False):
        data = {}
        dirs = [{'name': 'current', 'path': self._train_dir}]
        need_update = False

        if subdirs:
            dirs = self._get_subdirectories(self._train_dir)

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

            if not need_update:
                need_update = data[path]['content']['passed_iterations'] < data[path]['content']['total_iterations']

        self.data = data
        self._need_update = need_update

    def _update_data_from_dir(self, path):
        data = {
            'learn_error': [],
            'test_error': [],
            'time_left': '',
            'meta': []
        }

        meta_tsv = os.path.join(path, 'meta.tsv')
        if os.path.isfile(meta_tsv):
            with open(meta_tsv, 'r') as meta_in:
                data['meta'] = {}
                for row in list(csv.reader(meta_in, delimiter='\t')):
                    if not len(row):
                        continue

                    if row[0] != 'loss':
                        data['meta'][row[0]] = row[1]
                    else:
                        data['meta'][row[0] + '_' + row[1]] = row[2]

        logs = {
            'test_error': data['meta']['testErrorLog'] if 'testErrorLog' in data['meta'] else 'test_error.tsv',
            'learn_error': data['meta']['learnErrorLog'] if 'learnErrorLog' in data['meta'] else 'learn_error.tsv',
            'time_left': data['meta']['timeLeft'] if 'timeLeft' in data['meta'] else 'time_left.tsv'
        }

        for error_type in logs:
            file_path = os.path.join(path, logs[error_type])
            if os.path.isfile(file_path):
                with open(file_path, 'r') as f:
                    data[error_type] = list(csv.reader(f, delimiter='\t'))

        passed_test_iterations = len(data['test_error']) - 1
        passed_learn_iterations = len(data['learn_error']) - 1
        passed_iterations = 0

        if (passed_test_iterations > 0 and passed_learn_iterations > 0):
            passed_iterations = min(passed_test_iterations, passed_learn_iterations)
        elif passed_test_iterations > 0:
            passed_iterations = passed_test_iterations
        elif passed_learn_iterations > 0:
            passed_iterations = passed_learn_iterations

        if data['meta'] and data['meta']['iterCount']:
            return {
                'passed_iterations': passed_iterations,
                'total_iterations': int(data['meta']['iterCount']),
                'rows': data
            }
        else:
            return None

    @staticmethod
    def _get_static_path(file_name):
        return os.path.join(os.path.dirname(__file__), file_name)

    def _init_static(self):
        with open(self._get_static_path('CatboostIpython.css')) as f:
            css = f.read()
        js = ''

        # never use require in your projects
        js += 'window.__define = window.define;window.__require = window.require;window.define = undefined;window.require = undefined;'
        with open(self._get_static_path('plotly-basic.min.js')) as f:
            js += f.read()
        js += 'window.define = window.__define;window.require = window.__require;window.__define = undefined; window.__require = undefined;'

        with open(self._get_static_path('CatboostIpythonPlotly.js')) as f:
            js += f.read()
        with open(self._get_static_path('CatboostIpythonInit.js')) as f:
            js += f.read()
        html = """
            <style>
                {}
            </style>
            <script>
                {}
            </script>
        """.format(css, js)

        display(HTML(html))
