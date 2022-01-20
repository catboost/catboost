from functools import partial

import numpy as np
try:
    from pandas import DataFrame, Series
except ImportError:
    class DataFrame(object):
        pass

    class Series(object):
        pass
# copied from core.py to avoid circular import
_ARRAY_TYPES = (list, np.ndarray, DataFrame, Series)

from . import _catboost


__all__ = []


class BuiltinMetric(object):
    @staticmethod
    def params_with_defaults():
        """
        For each valid metric parameter, returns its default value and if this parameter is mandatory.
        Implemented in child classes.

        Returns
        ----------
        valid_params: dict: param_name -> {'default_value': default value or None, 'is_mandatory': bool}
        """
        raise NotImplementedError('Should be overridden by the child class.')

    def __str__(self):
        """
        Gets the representation of the metric object with overridden parameters.
        Implemented in child classes.

        Returns
        ----------
        metric_string: str representing the metric object.
        """
        raise NotImplementedError('Should be overridden by the child class.')

    def set_hints(self, **hints):
        """
        Sets hints for the metric. Hints are not validated.
        Implemented in child classes.

        Returns
        ----------
        self: for chained calls.
        """
        raise NotImplementedError('Should be overridden by the child class.')

    def eval(self,
            label,
            approx,
            weight=None,
            group_id=None,
            group_weight=None,
            subgroup_id=None,
            pairs=None,
            thread_count=-1):
        """
        Evaluate the metric with raw approxes and labels.

        Parameters
        ----------
        label : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
            Object labels.

        approx : list or numpy.ndarrays or pandas.DataFrame or pandas.Series
            Object approxes.

        weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Object weights.

        group_id : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Object group ids.

        group_weight : list or numpy.ndarray or pandas.DataFrame or pandas.Series, optional (default=None)
            Group weights.

        subgroup_id : list or numpy.ndarray, optional (default=None)
            subgroup id for each instance.
            If not None, giving 1 dimensional array like data.

        pairs : list or numpy.ndarray or pandas.DataFrame or string or pathlib.Path
            The pairs description.
            If list or numpy.ndarrays or pandas.DataFrame, giving 2 dimensional.
            The shape should be Nx2, where N is the pairs' count. The first element of the pair is
            the index of winner object in the training set. The second element of the pair is
            the index of loser object in the training set.
            If string or pathlib.Path, giving the path to the file with pairs description.

        thread_count : int, optional (default=-1)
            Number of threads to work with.
            If -1, then the number of threads is set to the number of CPU cores.

        Returns
        -------
        metric results : list with metric values.
        """
        if len(label) > 0 and not isinstance(label[0], _ARRAY_TYPES):
            label = [label]
        if len(approx) == 0:
            approx = [[]]
        if not isinstance(approx[0], _ARRAY_TYPES):
            approx = [approx]
        return _catboost._eval_metric_util(
            label, approx, str(self), weight, group_id, group_weight, subgroup_id, pairs, thread_count
        )

    def is_max_optimal(self):
        """
        Returns
        ----------
        bool : True if metric is maximizable, False otherwise
        """
        return _catboost.is_maximizable_metric(str(self))

    def is_min_optimal(self):
        """
        Returns
        ----------
        bool :  True if metric is minimizable, False otherwise
        """
        return _catboost.is_minimizable_metric(str(self))


class _MetricGenerator(type):
    def __new__(mcs, name, parents, attrs):
        for k in attrs['_valid_params']:
            attrs[k] = property(
                partial(_get_param, name=k),
                partial(_set_param, name=k),
                partial(_del_param, name=k),
                'Parameter {} of metric {}'.format(k, name),
            )

        attrs['params_with_defaults'] = staticmethod(lambda: {param: {'default_value': default_value,
                                                                      'is_mandatory': attrs['_is_mandatory_param'][param]}
                                                              for param, default_value in attrs['_valid_params'].items()})
        # Set the serialization function.
        docstring = ['Builtin metric: \'{}\''.format(name), 'Parameters:']
        if not attrs['_valid_params']:
            docstring[-1] += ' none'
        for param, value in attrs['_valid_params'].items():
            if not attrs['_is_mandatory_param'][param]:
                docstring.append(' ' * 4 + '{} = {} (default value)'.format(param, repr(value)))
            else:
                docstring.append(' ' * 4 + '{} (mandatory)'.format(param))
        attrs['__doc__'] = '\n'.join(docstring)
        attrs['__repr__'] = lambda self: '{}({})'.format(
            self._underlying_metric_name,
            ", ".join(['{}={} [mandatory={}]'.format(param, repr(value), self._is_mandatory_param[param])
                      for param, value in _current_params(self, False).items()]),
        )
        attrs['__str__'] = _to_string

        def set_hints(self, **hints):
            for hint_key, hint_value in hints.items():
                if isinstance(hint_value, bool):
                    hints[hint_key] = str(hint_value).lower()
            setattr(self, 'hints',  '|'.join(['{}~{}'.format(hint_key, hint_value) for hint_key, hint_value in hints.items()]))
            if 'hints' not in self._params:
                self._params.append('hints')
            return self
        attrs['set_hints'] = set_hints

        cls = super(_MetricGenerator, mcs).__new__(mcs, name, parents, attrs)
        return cls

    def __call__(cls, **kwargs):
        metric_obj = cls.__new__(cls)
        params = {k: v for k, v in cls._valid_params.items()}
        param_is_set = {param: not mandatory for param, mandatory in cls._is_mandatory_param.items()}

        # Overwrite default parameters and check that all passed parameters are valid.
        for param, value in kwargs.items():
            if param not in cls._valid_params:
                raise ValueError('Unexpected parameter {}'.format(param))
            params[param] = value
            param_is_set[param] = True

        # Check that no parameters are left unset.
        for param, is_set in param_is_set.items():
            if not is_set:
                raise ValueError('Parameter {} is mandatory and must be specified.'.format(param))

        for param, value in params.items():
            _set_param(metric_obj, value, param)
        metric_obj._params = list(params.keys())

        return metric_obj

    def __repr__(cls):
        return cls.__doc__

    def __setattr__(cls, name, value):
        # Protect property fields from being mutated.
        if name in ('_valid_params', '_is_mandatory_param'):
            raise ValueError('Metric\'s `{}` shouldn\'t be modified or deleted.'.format(name))
        type.__setattr__(cls, name, value)

    def __delattr__(cls, name):
        # Protect property fields from being mutated.
        if name in ('_valid_params', '_is_mandatory_param'):
            raise ValueError('Metric\'s `{}` shouldn\'t be modified or deleted.'.format(name))
        type.__delattr__(cls, name)


def _get_param(metric_obj, name):
    if name not in metric_obj._valid_params:
        raise ValueError('Metric {} doesn\'t have a parameter {}.'.format(metric_obj.__name__, name))
    return getattr(metric_obj, '_'+name)


def _set_param(metric_obj, value, name):
    """Validate a new parameter value in a created metric object."""
    if name not in metric_obj._valid_params:
        raise ValueError('Metric {} doesn\'t have a parameter {}.'.format(metric_obj.__name__, name))
    setattr(metric_obj, '_' + name, value)


def _del_param(metric_obj, name):
    """Validate a new parameter value in a created metric object."""
    if name not in metric_obj._valid_params:
        raise ValueError('Metric {} doesn\'t have a parameter {}.'.format(metric_obj.__name__, name))
    if metric_obj._is_mandatory_param[name]:
        raise ValueError('Parameter {} is mandatory, cannot reset.'.format(name))
    value = metric_obj._valid_params[name]
    setattr(metric_obj, '_' + name, value)


def _current_params(metric_obj, override_only):
    params_with_defaults = metric_obj.params_with_defaults()
    param_info = {}
    for param in sorted(metric_obj._params):
        value = getattr(metric_obj, param)  # current value
        if param == 'hints' and value == '':
            # Skip unset hints.
            continue
        if override_only:
            # Skip reporting parameters which are set to their default values.
            default_value = params_with_defaults[param]['default_value']
            if (default_value is None and value is None) or (default_value is not None and default_value == value):
                continue
        param_info[param] = value
    return param_info


def _to_string(metric_obj):
    def _param_to_string(dict_item):
        param, value = dict_item
        if param == 'misclass_cost_matrix':
            str_value = '/'.join(map(str, value.flatten()))
        else:
            str_value = str(value)
        return '{}={}'.format(param, str_value)

    param_info = _current_params(metric_obj, True)
    # E.g. AUC for both AUC and AUCMulticlass:
    underlying_name = metric_obj._underlying_metric_name
    if len(param_info) == 0:
        return underlying_name
    return '{}:{}'.format(underlying_name, ';'.join(map(_param_to_string, param_info.items())))


def _generate_metric_classes():
    for metric_name, metric_param_sets in _catboost.AllMetricsParams().items():
        for param_set in metric_param_sets:
            derived_name = metric_name + param_set['_name_suffix']
            del param_set['_name_suffix']
            valid_params = {param: param_value['default_value'] if not param_value['is_mandatory'] else None
                            for param, param_value in param_set.items()}
            is_mandatory_param = {param: param_value['is_mandatory'] for param, param_value in param_set.items()}
            if 'hints' not in valid_params:
                valid_params.update({'hints': ''})
                is_mandatory_param.update({'hints': False})
            globals()[derived_name] = _MetricGenerator(str(derived_name), (BuiltinMetric,), {
                '_valid_params': valid_params,
                '_is_mandatory_param': is_mandatory_param,
                '_underlying_metric_name': metric_name,
            })
            globals()['__all__'].append(derived_name)

_generate_metric_classes()
