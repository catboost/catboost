from .ipythonwidget import MetricVisualizer


def _jupyter_labextension_paths():
    return [{
        'src': 'labextension',
        'dest': 'catboost_widget',
    }]


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'nbextension',
        'dest': 'catboost_widget',
        'require': 'catboost_widget/extension'
    }]
