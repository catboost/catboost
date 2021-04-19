from .ipythonwidget import MetricVisualizer


def _jupyter_labextension_paths():
    return [{
        'src': 'labextension',
        'dest': 'catboost-widget',
    }]


def _jupyter_nbextension_paths():
    return [{
        'section': 'notebook',
        'src': 'nbextension',
        'dest': 'catboost-widget',
        'require': 'catboost-widget/extension'
    }]
