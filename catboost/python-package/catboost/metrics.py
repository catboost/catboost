from copy import copy
from functools import partial

from . import _catboost

_dummy_metrics = _catboost.DummyMetrics

class BuiltinMetric(object):
    pass

class _MetricGenerator(type):
    def __new__(mcls, name, parents, attrs):
        '''
        Construct a new metric class.
        Params:
            mcls -- metaclass object,
            name -- string name of the class we are constructing,
            parents -- tuple of base classes for this new class,
            attrs -- dict of attributes that have to be set.
        Returns: metric class object.
        '''
        for k in attrs["_valid_params"]:
            attrs[k] = property(
                partial(_get_param, name=k),
                partial(_set_param, name=k),
                partial(_set_param, name=k, value=None),
                "Parameter {} of metric {}".format(k, name),
            )

        # Name for this function -- up to discussion (parameters, valid_parameters, etc)
        attrs["params_with_defaults"] = staticmethod(lambda: copy(attrs["_valid_params"]))

        # Set the serialization function.
        attrs["to_string"] = _to_string
        fmt = ["Builtin metric: '{}'".format(name)]
        fmt.append("Parameters:")
        if not attrs["_valid_params"]:
            fmt[-1] += " none"
        for k, v in attrs["_valid_params"].items():
            if v is not None:
                fmt.append(" " * 4 + "{} = {} (default)".format(k, v))
            else:
                fmt.append(" " * 4 + "{} (no default)".format(k))
        attrs["__doc__"] = "\n".join(fmt)
        attrs["__repr__"] = attrs["__str__"] = _to_repr
        cls = super(_MetricGenerator, mcls).__new__(mcls, name, parents, attrs)
        return cls

    def __call__(cls, **kwargs):
        '''
        Construct a new Metric object with validated parameters.
        Parameters:
            kwargs -- dict of metric parameters.
        Returns: a new metric object.
        '''
        metric_obj = cls.__new__(cls)
        params = {k: v for k, v in cls._valid_params.items()}

        # Overwrite default parameters and check that all passed parameters are valid.
        for k, v in kwargs.items():
            if k not in cls._valid_params:
                raise ValueError("Unexpected parameter {}".format(k))
            params[k] = v

        # Check that no parameters are left unset.
        for param, value in params.items():
            if value is None:
                raise ValueError("Parameter {} is mandatory and must be specified.".format(param))

        for k, v in params.items():
            setattr(metric_obj, "_" + k, v)
        metric_obj._params = list(params.keys())

        return metric_obj

    def __setattr__(cls, name, value):
        # Protect property fields from being mutated.
        if name == "_valid_params":
            raise ValueError("Metric's `{}` shouldn't be modified or deleted.".format(name))
        type.__setattr__(cls, name, value)

    def __delattr__(cls, name):
        # Protect property fields from being mutated.
        if name == "_valid_params":
            raise ValueError("Metric's `{}` shouldn't be modified or deleted.".format(name))
        type.__delattr__(cls, name)

def _get_param(metric_obj, name):
    return getattr(metric_obj, "_"+name)

def _set_param(metric_obj, value, name):
    """Validates a new parameter value in a created metric object."""
    if value is None:
        value = metric_obj._valid_params[name]
        if value is None:
            raise ValueError("Parameter {} is mandatory, cannot reset.".format(name))
    setattr(metric_obj, "_" + name, value)

def _to_string(metric_obj):
    s = type(metric_obj).__name__
    if len(metric_obj._params) == 0:
        return s
    ps = []
    for param in metric_obj._params:
        val = getattr(metric_obj, param)
        ps.append(param + "=" + str(val))
    return s + ":" + ",".join(ps)

def _to_repr(metric_obj):
    return _to_string(metric_obj)


for metric_name, metric_params in _dummy_metrics().items():
    globals()[metric_name] = _MetricGenerator(str(metric_name), (BuiltinMetric,), {
        "_valid_params": metric_params,
    })
