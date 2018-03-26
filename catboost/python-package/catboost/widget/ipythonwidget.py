import os
import time
import json
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
                need_update = data[path]['content']['passed_iterations'] + 1 < data[path]['content']['total_iterations']

        self.data = data
        self._need_update = need_update

    def _update_data_from_dir(self, path):
        data = {
            'iterations': {},
            'meta': {}
        }

        training_json = os.path.join(path, 'catboost_training.json')

        if os.path.isfile(training_json):
            with open(training_json, 'r') as json_data:
                training_data = json.load(json_data)
                data['meta'] = training_data['meta']
                data['iterations'] = training_data['iterations']
        else:
            return None

        return {
            'passed_iterations': data['iterations'][-1]['iteration'],
            'total_iterations': data['meta']['iteration_count'],
            'data': data
        }

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
