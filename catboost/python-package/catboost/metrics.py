from copy import copy
from functools import partial

from . import _catboost

__all__ = []

class BuiltinMetric(object):
    @staticmethod
    def params_with_defaults():
        '''
        Get valid metric parameters with defaults, if any.
        Implemented in child classes.
        Returns
        ----------
        valid_params: dict: param_name -> default value or None.
        '''
        raise Exception("Should be overridden by the child class.")

    def to_string():
        '''
        Get the representation of the metric object with overridden parameters.
        Implemented in child classes.
        Returns
        ----------
        metric_string: str representing the metric object.
        '''
        raise Exception("Should be overridden by the child class.")

class _MetricGenerator(type):
    def __new__(mcls, name, parents, attrs):
        for k in attrs["_valid_params"]:
            attrs[k] = property(
                partial(_get_param, name=k),
                partial(_set_param, name=k),
                partial(_set_param, name=k, value=None),
                "Parameter {} of metric {}".format(k, name),
            )

        attrs["params_with_defaults"] = staticmethod(lambda: copy(attrs["_valid_params"]))

        # Set the serialization function.
        attrs["to_string"] = lambda self: _to_string(self, False)
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
        attrs["__repr__"] = attrs["__str__"] = lambda self: _to_string(self, True)
        cls = super(_MetricGenerator, mcls).__new__(mcls, name, parents, attrs)
        return cls

    def __call__(cls, **kwargs):
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
    """Validate a new parameter value in a created metric object."""
    if value is None:
        value = metric_obj._valid_params[name]
        if value is None:
            raise ValueError("Parameter {} is mandatory, cannot reset.".format(name))
    setattr(metric_obj, "_" + name, value)

def _to_string(metric_obj, with_defaults):
    s = metric_obj._underlying_metric_name
    valid_params = metric_obj.params_with_defaults()
    ps = []
    for param in metric_obj._params:
        val = getattr(metric_obj, param)
        if not with_defaults:
            # Skip reporting parameters which are set to their default values.
            default_val = valid_params[param]
            if default_val == val:
                continue
        ps.append(param + "=" + str(val))
    if len(ps) == 0:
        return s
    return s + ":" + ";".join(ps)

for metric_name, metric_params in _catboost.AllMetricsParams().items():
    derived_name = metric_name + metric_params["_name_suffix"]
    del metric_params["_name_suffix"]
    globals()[derived_name] = _MetricGenerator(str(derived_name), (BuiltinMetric,), {
        "_valid_params": {param: param_value["default_value"] if not param_value["is_mandatory"] else None
                         for param, param_value in metric_params.items()},
        "_underlying_metric_name": metric_name,
    })
    globals()["__all__"].append(derived_name)
