import os
import json
from threading import Thread, Event
from traitlets import Unicode, Dict, default
from IPython.display import display
from ipywidgets import DOMWidget, Layout, widget_serialization


class MetricVisualizer(DOMWidget):
    _view_name = Unicode('CatboostWidgetView').tag(sync=True)
    _model_name = Unicode('CatboostWidgetModel').tag(sync=True)
    _view_module = Unicode('catboost-widget').tag(sync=True)
    _model_module = Unicode('catboost-widget').tag(sync=True)
    _view_module_version = Unicode('^1.0.0').tag(sync=True)
    _model_module_version = Unicode('^1.0.0').tag(sync=True)

    data = Dict({}).tag(sync=True, **widget_serialization)

    def __init__(self, train_dirs, subdirs=False):
        super(self.__class__, self).__init__()
        if isinstance(train_dirs, str):
            train_dirs = [train_dirs]
        if subdirs:
            train_subdirs = []
            for train_dir in train_dirs:
                train_subdirs.extend(self._get_subdirectories(train_dir))
            train_dirs = train_subdirs
        self._train_dirs = train_dirs[:]
        self._names = []
        curdir = os.path.abspath(os.path.curdir)
        for train_dir in train_dirs:
            abspath = os.path.abspath(train_dir)
            self._names.append(os.path.basename(abspath) if abspath != curdir else 'current')
        self._need_to_stop = Event()
        self._update_after_stop_signal = False

    @default('layout')
    def _default_layout(self):
        return Layout(height='500px', align_self='stretch')

    def start(self):
        display(self)
        self._update_data()
        while not self._need_to_stop.wait(1.0):
            self._update_data()

        if self._update_after_stop_signal:
            self._update_data()

    def _run_update(self):
        self.thread = Thread(target=self.start, args=())
        self.thread.start()

    def _stop_update(self):
        self._update_after_stop_signal = True
        self._need_to_stop.set()
        self.thread.join()

    def _get_subdirectories(self, a_dir):
        return [os.path.join(a_dir, name) for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

    def _update_data(self):
        data = {}
        dirs = [{'name': name, 'path': path} for name, path in zip(self._names, self._train_dirs)]

        all_completed = True
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

            passed_iterations = data[path]['content']['passed_iterations']
            total_iterations = data[path]['content']['total_iterations']
            all_completed &= (passed_iterations + 1 >= total_iterations and total_iterations != 0)

        if all_completed:
            self._need_to_stop.set()

        self.data = data

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

    @staticmethod
    def _get_static_path(file_name):
        return os.path.join(os.path.dirname(__file__), file_name)
