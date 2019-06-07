# being a bit too dynamic
# pylint: disable=E1101
from __future__ import division

from contextlib import contextmanager
import warnings

import numpy as np

import pandas.compat as compat
from pandas.compat import lmap, lrange

from pandas.core.dtypes.common import is_list_like


def _get_standard_colors(num_colors=None, colormap=None, color_type='default',
                         color=None):
    import matplotlib.pyplot as plt

    if color is None and colormap is not None:
        if isinstance(colormap, compat.string_types):
            import matplotlib.cm as cm
            cmap = colormap
            colormap = cm.get_cmap(colormap)
            if colormap is None:
                raise ValueError("Colormap {0} is not recognized".format(cmap))
        colors = lmap(colormap, np.linspace(0, 1, num=num_colors))
    elif color is not None:
        if colormap is not None:
            warnings.warn("'color' and 'colormap' cannot be used "
                          "simultaneously. Using 'color'")
        colors = list(color) if is_list_like(color) else color
    else:
        if color_type == 'default':
            # need to call list() on the result to copy so we don't
            # modify the global rcParams below
            try:
                colors = [c['color']
                          for c in list(plt.rcParams['axes.prop_cycle'])]
            except KeyError:
                colors = list(plt.rcParams.get('axes.color_cycle',
                                               list('bgrcmyk')))
            if isinstance(colors, compat.string_types):
                colors = list(colors)

            colors = colors[0:num_colors]
        elif color_type == 'random':
            import pandas.core.common as com

            def random_color(column):
                """ Returns a random color represented as a list of length 3"""
                # GH17525 use common._random_state to avoid resetting the seed
                rs = com.random_state(column)
                return rs.rand(3).tolist()

            colors = lmap(random_color, lrange(num_colors))
        else:
            raise ValueError("color_type must be either 'default' or 'random'")

    if isinstance(colors, compat.string_types):
        import matplotlib.colors
        conv = matplotlib.colors.ColorConverter()

        def _maybe_valid_colors(colors):
            try:
                [conv.to_rgba(c) for c in colors]
                return True
            except ValueError:
                return False

        # check whether the string can be convertible to single color
        maybe_single_color = _maybe_valid_colors([colors])
        # check whether each character can be convertible to colors
        maybe_color_cycle = _maybe_valid_colors(list(colors))
        if maybe_single_color and maybe_color_cycle and len(colors) > 1:
            hex_color = [c['color']
                         for c in list(plt.rcParams['axes.prop_cycle'])]
            colors = [hex_color[int(colors[1])]]
        elif maybe_single_color:
            colors = [colors]
        else:
            # ``colors`` is regarded as color cycle.
            # mpl will raise error any of them is invalid
            pass

    # Append more colors by cycling if there is not enough color.
    # Extra colors will be ignored by matplotlib if there are more colors
    # than needed and nothing needs to be done here.
    if len(colors) < num_colors:
        try:
            multiple = num_colors // len(colors) - 1
        except ZeroDivisionError:
            raise ValueError("Invalid color argument: ''")
        mod = num_colors % len(colors)

        colors += multiple * colors
        colors += colors[:mod]

    return colors


class _Options(dict):
    """
    Stores pandas plotting options.
    Allows for parameter aliasing so you can just use parameter names that are
    the same as the plot function parameters, but is stored in a canonical
    format that makes it easy to breakdown into groups later
    """

    # alias so the names are same as plotting method parameter names
    _ALIASES = {'x_compat': 'xaxis.compat'}
    _DEFAULT_KEYS = ['xaxis.compat']

    def __init__(self, deprecated=False):
        self._deprecated = deprecated
        # self['xaxis.compat'] = False
        super(_Options, self).__setitem__('xaxis.compat', False)

    def __getitem__(self, key):
        key = self._get_canonical_key(key)
        if key not in self:
            raise ValueError(
                '{key} is not a valid pandas plotting option'.format(key=key))
        return super(_Options, self).__getitem__(key)

    def __setitem__(self, key, value):
        key = self._get_canonical_key(key)
        return super(_Options, self).__setitem__(key, value)

    def __delitem__(self, key):
        key = self._get_canonical_key(key)
        if key in self._DEFAULT_KEYS:
            raise ValueError(
                'Cannot remove default parameter {key}'.format(key=key))
        return super(_Options, self).__delitem__(key)

    def __contains__(self, key):
        key = self._get_canonical_key(key)
        return super(_Options, self).__contains__(key)

    def reset(self):
        """
        Reset the option store to its initial state

        Returns
        -------
        None
        """
        self.__init__()

    def _get_canonical_key(self, key):
        return self._ALIASES.get(key, key)

    @contextmanager
    def use(self, key, value):
        """
        Temporarily set a parameter value using the with statement.
        Aliasing allowed.
        """
        old_value = self[key]
        try:
            self[key] = value
            yield self
        finally:
            self[key] = old_value


plot_params = _Options()
