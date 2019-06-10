"""
Contains data structures designed for manipulating panel (3-dimensional) data
"""
# pylint: disable=E1103,W0231,W0212,W0621
from __future__ import division

import warnings

import numpy as np

import pandas.compat as compat
from pandas.compat import OrderedDict, map, range, u, zip
from pandas.compat.numpy import function as nv
from pandas.util._decorators import Appender, Substitution, deprecate_kwarg
from pandas.util._validators import validate_axis_style_args

from pandas.core.dtypes.cast import (
    cast_scalar_to_array, infer_dtype_from_scalar, maybe_cast_item)
from pandas.core.dtypes.common import (
    is_integer, is_list_like, is_scalar, is_string_like)
from pandas.core.dtypes.missing import notna

import pandas.core.common as com
from pandas.core.frame import DataFrame
from pandas.core.generic import NDFrame, _shared_docs
from pandas.core.index import (
    Index, MultiIndex, _get_objs_combined_axis, ensure_index)
import pandas.core.indexes.base as ibase
from pandas.core.indexing import maybe_droplevels
from pandas.core.internals import (
    BlockManager, create_block_manager_from_arrays,
    create_block_manager_from_blocks)
import pandas.core.ops as ops
from pandas.core.reshape.util import cartesian_product
from pandas.core.series import Series

from pandas.io.formats.printing import pprint_thing

_shared_doc_kwargs = dict(
    axes='items, major_axis, minor_axis',
    klass="Panel",
    axes_single_arg="{0, 1, 2, 'items', 'major_axis', 'minor_axis'}",
    optional_mapper='', optional_axis='', optional_labels='')
_shared_doc_kwargs['args_transpose'] = (
    "three positional arguments: each one of\n{ax_single}".format(
        ax_single=_shared_doc_kwargs['axes_single_arg']))


def _ensure_like_indices(time, panels):
    """
    Makes sure that time and panels are conformable.
    """
    n_time = len(time)
    n_panel = len(panels)
    u_panels = np.unique(panels)  # this sorts!
    u_time = np.unique(time)
    if len(u_time) == n_time:
        time = np.tile(u_time, len(u_panels))
    if len(u_panels) == n_panel:
        panels = np.repeat(u_panels, len(u_time))
    return time, panels


def panel_index(time, panels, names=None):
    """
    Returns a multi-index suitable for a panel-like DataFrame.

    Parameters
    ----------
    time : array-like
        Time index, does not have to repeat
    panels : array-like
        Panel index, does not have to repeat
    names : list, optional
        List containing the names of the indices

    Returns
    -------
    multi_index : MultiIndex
        Time index is the first level, the panels are the second level.

    Examples
    --------
    >>> years = range(1960,1963)
    >>> panels = ['A', 'B', 'C']
    >>> panel_idx = panel_index(years, panels)
    >>> panel_idx
    MultiIndex([(1960, 'A'), (1961, 'A'), (1962, 'A'), (1960, 'B'),
                (1961, 'B'), (1962, 'B'), (1960, 'C'), (1961, 'C'),
                (1962, 'C')], dtype=object)

    or

    >>> years = np.repeat(range(1960,1963), 3)
    >>> panels = np.tile(['A', 'B', 'C'], 3)
    >>> panel_idx = panel_index(years, panels)
    >>> panel_idx
    MultiIndex([(1960, 'A'), (1960, 'B'), (1960, 'C'), (1961, 'A'),
                (1961, 'B'), (1961, 'C'), (1962, 'A'), (1962, 'B'),
                (1962, 'C')], dtype=object)
    """
    if names is None:
        names = ['time', 'panel']
    time, panels = _ensure_like_indices(time, panels)
    return MultiIndex.from_arrays([time, panels], sortorder=None, names=names)


class Panel(NDFrame):
    """
    Represents wide format panel data, stored as 3-dimensional array.

    .. deprecated:: 0.20.0
        The recommended way to represent 3-D data are with a MultiIndex on a
        DataFrame via the :attr:`~Panel.to_frame()` method or with the
        `xarray package <http://xarray.pydata.org/en/stable/>`__.
        Pandas provides a :attr:`~Panel.to_xarray()` method to automate this
        conversion.

    Parameters
    ----------
    data : ndarray (items x major x minor), or dict of DataFrames
    items : Index or array-like
        axis=0
    major_axis : Index or array-like
        axis=1
    minor_axis : Index or array-like
        axis=2
    copy : boolean, default False
        Copy data from inputs. Only affects DataFrame / 2d ndarray input
    dtype : dtype, default None
        Data type to force, otherwise infer
    """

    @property
    def _constructor(self):
        return type(self)

    _constructor_sliced = DataFrame

    def __init__(self, data=None, items=None, major_axis=None, minor_axis=None,
                 copy=False, dtype=None):
        # deprecation GH13563
        warnings.warn("\nPanel is deprecated and will be removed in a "
                      "future version.\nThe recommended way to represent "
                      "these types of 3-dimensional data are with a "
                      "MultiIndex on a DataFrame, via the "
                      "Panel.to_frame() method\n"
                      "Alternatively, you can use the xarray package "
                      "http://xarray.pydata.org/en/stable/.\n"
                      "Pandas provides a `.to_xarray()` method to help "
                      "automate this conversion.\n",
                      FutureWarning, stacklevel=3)

        self._init_data(data=data, items=items, major_axis=major_axis,
                        minor_axis=minor_axis, copy=copy, dtype=dtype)

    def _init_data(self, data, copy, dtype, **kwargs):
        """
        Generate ND initialization; axes are passed
        as required objects to __init__.
        """
        if data is None:
            data = {}
        if dtype is not None:
            dtype = self._validate_dtype(dtype)

        passed_axes = [kwargs.pop(a, None) for a in self._AXIS_ORDERS]

        if kwargs:
            raise TypeError('_init_data() got an unexpected keyword '
                            'argument "{0}"'.format(list(kwargs.keys())[0]))

        axes = None
        if isinstance(data, BlockManager):
            if com._any_not_none(*passed_axes):
                axes = [x if x is not None else y
                        for x, y in zip(passed_axes, data.axes)]
            mgr = data
        elif isinstance(data, dict):
            mgr = self._init_dict(data, passed_axes, dtype=dtype)
            copy = False
            dtype = None
        elif isinstance(data, (np.ndarray, list)):
            mgr = self._init_matrix(data, passed_axes, dtype=dtype, copy=copy)
            copy = False
            dtype = None
        elif is_scalar(data) and com._all_not_none(*passed_axes):
            values = cast_scalar_to_array([len(x) for x in passed_axes],
                                          data, dtype=dtype)
            mgr = self._init_matrix(values, passed_axes, dtype=values.dtype,
                                    copy=False)
            copy = False
        else:  # pragma: no cover
            raise ValueError('Panel constructor not properly called!')

        NDFrame.__init__(self, mgr, axes=axes, copy=copy, dtype=dtype)

    def _init_dict(self, data, axes, dtype=None):
        haxis = axes.pop(self._info_axis_number)

        # prefilter if haxis passed
        if haxis is not None:
            haxis = ensure_index(haxis)
            data = OrderedDict((k, v)
                               for k, v in compat.iteritems(data)
                               if k in haxis)
        else:
            keys = com.dict_keys_to_ordered_list(data)
            haxis = Index(keys)

        for k, v in compat.iteritems(data):
            if isinstance(v, dict):
                data[k] = self._constructor_sliced(v)

        # extract axis for remaining axes & create the slicemap
        raxes = [self._extract_axis(self, data, axis=i) if a is None else a
                 for i, a in enumerate(axes)]
        raxes_sm = self._extract_axes_for_slice(self, raxes)

        # shallow copy
        arrays = []
        haxis_shape = [len(a) for a in raxes]
        for h in haxis:
            v = values = data.get(h)
            if v is None:
                values = np.empty(haxis_shape, dtype=dtype)
                values.fill(np.nan)
            elif isinstance(v, self._constructor_sliced):
                d = raxes_sm.copy()
                d['copy'] = False
                v = v.reindex(**d)
                if dtype is not None:
                    v = v.astype(dtype)
                values = v.values
            arrays.append(values)

        return self._init_arrays(arrays, haxis, [haxis] + raxes)

    def _init_arrays(self, arrays, arr_names, axes):
        return create_block_manager_from_arrays(arrays, arr_names, axes)

    @classmethod
    def from_dict(cls, data, intersect=False, orient='items', dtype=None):
        """
        Construct Panel from dict of DataFrame objects.

        Parameters
        ----------
        data : dict
            {field : DataFrame}
        intersect : boolean
            Intersect indexes of input DataFrames
        orient : {'items', 'minor'}, default 'items'
            The "orientation" of the data. If the keys of the passed dict
            should be the items of the result panel, pass 'items'
            (default). Otherwise if the columns of the values of the passed
            DataFrame objects should be the items (which in the case of
            mixed-dtype data you should do), instead pass 'minor'
        dtype : dtype, default None
            Data type to force, otherwise infer

        Returns
        -------
        Panel
        """
        from collections import defaultdict

        orient = orient.lower()
        if orient == 'minor':
            new_data = defaultdict(OrderedDict)
            for col, df in compat.iteritems(data):
                for item, s in compat.iteritems(df):
                    new_data[item][col] = s
            data = new_data
        elif orient != 'items':  # pragma: no cover
            raise ValueError('Orientation must be one of {items, minor}.')

        d = cls._homogenize_dict(cls, data, intersect=intersect, dtype=dtype)
        ks = list(d['data'].keys())
        if not isinstance(d['data'], OrderedDict):
            ks = list(sorted(ks))
        d[cls._info_axis_name] = Index(ks)
        return cls(**d)

    def __getitem__(self, key):
        key = com.apply_if_callable(key, self)

        if isinstance(self._info_axis, MultiIndex):
            return self._getitem_multilevel(key)
        if not (is_list_like(key) or isinstance(key, slice)):
            return super(Panel, self).__getitem__(key)
        return self.loc[key]

    def _getitem_multilevel(self, key):
        info = self._info_axis
        loc = info.get_loc(key)
        if isinstance(loc, (slice, np.ndarray)):
            new_index = info[loc]
            result_index = maybe_droplevels(new_index, key)
            slices = [loc] + [slice(None)] * (self._AXIS_LEN - 1)
            new_values = self.values[slices]

            d = self._construct_axes_dict(self._AXIS_ORDERS[1:])
            d[self._info_axis_name] = result_index
            result = self._constructor(new_values, **d)
            return result
        else:
            return self._get_item_cache(key)

    def _init_matrix(self, data, axes, dtype=None, copy=False):
        values = self._prep_ndarray(self, data, copy=copy)

        if dtype is not None:
            try:
                values = values.astype(dtype)
            except Exception:
                raise ValueError('failed to cast to '
                                 '{datatype}'.format(datatype=dtype))

        shape = values.shape
        fixed_axes = []
        for i, ax in enumerate(axes):
            if ax is None:
                ax = ibase.default_index(shape[i])
            else:
                ax = ensure_index(ax)
            fixed_axes.append(ax)

        return create_block_manager_from_blocks([values], fixed_axes)

    # ----------------------------------------------------------------------
    # Comparison methods

    def _compare_constructor(self, other, func):
        if not self._indexed_same(other):
            raise Exception('Can only compare identically-labeled '
                            'same type objects')

        new_data = {col: func(self[col], other[col])
                    for col in self._info_axis}

        d = self._construct_axes_dict(copy=False)
        return self._constructor(data=new_data, **d)

    # ----------------------------------------------------------------------
    # Magic methods

    def __unicode__(self):
        """
        Return a string representation for a particular Panel.

        Invoked by unicode(df) in py2 only.
        Yields a Unicode String in both py2/py3.
        """

        class_name = str(self.__class__)

        dims = u('Dimensions: {dimensions}'.format(dimensions=' x '.join(
            ["{shape} ({axis})".format(shape=shape, axis=axis) for axis, shape
             in zip(self._AXIS_ORDERS, self.shape)])))

        def axis_pretty(a):
            v = getattr(self, a)
            if len(v) > 0:
                return u('{ax} axis: {x} to {y}'.format(ax=a.capitalize(),
                                                        x=pprint_thing(v[0]),
                                                        y=pprint_thing(v[-1])))
            else:
                return u('{ax} axis: None'.format(ax=a.capitalize()))

        output = '\n'.join(
            [class_name, dims] + [axis_pretty(a) for a in self._AXIS_ORDERS])
        return output

    def _get_plane_axes_index(self, axis):
        """
        Get my plane axes indexes: these are already
        (as compared with higher level planes),
        as we are returning a DataFrame axes indexes.
        """
        axis_name = self._get_axis_name(axis)

        if axis_name == 'major_axis':
            index = 'minor_axis'
            columns = 'items'
        if axis_name == 'minor_axis':
            index = 'major_axis'
            columns = 'items'
        elif axis_name == 'items':
            index = 'major_axis'
            columns = 'minor_axis'

        return index, columns

    def _get_plane_axes(self, axis):
        """
        Get my plane axes indexes: these are already
        (as compared with higher level planes),
        as we are returning a DataFrame axes.
        """
        return [self._get_axis(axi)
                for axi in self._get_plane_axes_index(axis)]

    fromDict = from_dict

    def to_sparse(self, *args, **kwargs):
        """
        NOT IMPLEMENTED: do not call this method, as sparsifying is not
        supported for Panel objects and will raise an error.

        Convert to SparsePanel.
        """
        raise NotImplementedError("sparsifying is not supported "
                                  "for Panel objects")

    def to_excel(self, path, na_rep='', engine=None, **kwargs):
        """
        Write each DataFrame in Panel to a separate excel sheet.

        Parameters
        ----------
        path : string or ExcelWriter object
            File path or existing ExcelWriter
        na_rep : string, default ''
            Missing data representation
        engine : string, default None
            write engine to use - you can also set this via the options
            ``io.excel.xlsx.writer``, ``io.excel.xls.writer``, and
            ``io.excel.xlsm.writer``.

        Other Parameters
        ----------------
        float_format : string, default None
            Format string for floating point numbers
        cols : sequence, optional
            Columns to write
        header : boolean or list of string, default True
            Write out column names. If a list of string is given it is
            assumed to be aliases for the column names
        index : boolean, default True
            Write row names (index)
        index_label : string or sequence, default None
            Column label for index column(s) if desired. If None is given, and
            `header` and `index` are True, then the index names are used. A
            sequence should be given if the DataFrame uses MultiIndex.
        startrow : upper left cell row to dump data frame
        startcol : upper left cell column to dump data frame

        Notes
        -----
        Keyword arguments (and na_rep) are passed to the ``to_excel`` method
        for each DataFrame written.
        """
        from pandas.io.excel import ExcelWriter

        if isinstance(path, compat.string_types):
            writer = ExcelWriter(path, engine=engine)
        else:
            writer = path
        kwargs['na_rep'] = na_rep

        for item, df in self.iteritems():
            name = str(item)
            df.to_excel(writer, name, **kwargs)
        writer.save()

    def as_matrix(self):
        self._consolidate_inplace()
        return self._data.as_array()

    # ----------------------------------------------------------------------
    # Getting and setting elements

    def get_value(self, *args, **kwargs):
        """
        Quickly retrieve single value at (item, major, minor) location.

        .. deprecated:: 0.21.0

        Please use .at[] or .iat[] accessors.

        Parameters
        ----------
        item : item label (panel item)
        major : major axis label (panel item row)
        minor : minor axis label (panel item column)
        takeable : interpret the passed labels as indexers, default False

        Returns
        -------
        value : scalar value
        """
        warnings.warn("get_value is deprecated and will be removed "
                      "in a future release. Please use "
                      ".at[] or .iat[] accessors instead", FutureWarning,
                      stacklevel=2)
        return self._get_value(*args, **kwargs)

    def _get_value(self, *args, **kwargs):
        nargs = len(args)
        nreq = self._AXIS_LEN

        # require an arg for each axis
        if nargs != nreq:
            raise TypeError('There must be an argument for each axis, you gave'
                            ' {0} args, but {1} are required'.format(nargs,
                                                                     nreq))
        takeable = kwargs.pop('takeable', None)

        if kwargs:
            raise TypeError('get_value() got an unexpected keyword '
                            'argument "{0}"'.format(list(kwargs.keys())[0]))

        if takeable is True:
            lower = self._iget_item_cache(args[0])
        else:
            lower = self._get_item_cache(args[0])

        return lower._get_value(*args[1:], takeable=takeable)
    _get_value.__doc__ = get_value.__doc__

    def set_value(self, *args, **kwargs):
        """
        Quickly set single value at (item, major, minor) location.

        .. deprecated:: 0.21.0

        Please use .at[] or .iat[] accessors.

        Parameters
        ----------
        item : item label (panel item)
        major : major axis label (panel item row)
        minor : minor axis label (panel item column)
        value : scalar
        takeable : interpret the passed labels as indexers, default False

        Returns
        -------
        panel : Panel
            If label combo is contained, will be reference to calling Panel,
            otherwise a new object
        """
        warnings.warn("set_value is deprecated and will be removed "
                      "in a future release. Please use "
                      ".at[] or .iat[] accessors instead", FutureWarning,
                      stacklevel=2)
        return self._set_value(*args, **kwargs)

    def _set_value(self, *args, **kwargs):
        # require an arg for each axis and the value
        nargs = len(args)
        nreq = self._AXIS_LEN + 1

        if nargs != nreq:
            raise TypeError('There must be an argument for each axis plus the '
                            'value provided, you gave {0} args, but {1} are '
                            'required'.format(nargs, nreq))
        takeable = kwargs.pop('takeable', None)

        if kwargs:
            raise TypeError('set_value() got an unexpected keyword '
                            'argument "{0}"'.format(list(kwargs.keys())[0]))

        try:
            if takeable is True:
                lower = self._iget_item_cache(args[0])
            else:
                lower = self._get_item_cache(args[0])

            lower._set_value(*args[1:], takeable=takeable)
            return self
        except KeyError:
            axes = self._expand_axes(args)
            d = self._construct_axes_dict_from(self, axes, copy=False)
            result = self.reindex(**d)
            args = list(args)
            likely_dtype, args[-1] = infer_dtype_from_scalar(args[-1])
            made_bigger = not np.array_equal(axes[0], self._info_axis)
            # how to make this logic simpler?
            if made_bigger:
                maybe_cast_item(result, args[0], likely_dtype)

            return result._set_value(*args)
    _set_value.__doc__ = set_value.__doc__

    def _box_item_values(self, key, values):
        if self.ndim == values.ndim:
            result = self._constructor(values)

            # a dup selection will yield a full ndim
            if result._get_axis(0).is_unique:
                result = result[key]

            return result

        d = self._construct_axes_dict_for_slice(self._AXIS_ORDERS[1:])
        return self._constructor_sliced(values, **d)

    def __setitem__(self, key, value):
        key = com.apply_if_callable(key, self)
        shape = tuple(self.shape)
        if isinstance(value, self._constructor_sliced):
            value = value.reindex(
                **self._construct_axes_dict_for_slice(self._AXIS_ORDERS[1:]))
            mat = value.values
        elif isinstance(value, np.ndarray):
            if value.shape != shape[1:]:
                raise ValueError('shape of value must be {0}, shape of given '
                                 'object was {1}'.format(
                                     shape[1:], tuple(map(int, value.shape))))
            mat = np.asarray(value)
        elif is_scalar(value):
            mat = cast_scalar_to_array(shape[1:], value)
        else:
            raise TypeError('Cannot set item of '
                            'type: {dtype!s}'.format(dtype=type(value)))

        mat = mat.reshape(tuple([1]) + shape[1:])
        NDFrame._set_item(self, key, mat)

    def _unpickle_panel_compat(self, state):  # pragma: no cover
        """
        Unpickle the panel.
        """
        from pandas.io.pickle import _unpickle_array

        _unpickle = _unpickle_array
        vals, items, major, minor = state

        items = _unpickle(items)
        major = _unpickle(major)
        minor = _unpickle(minor)
        values = _unpickle(vals)
        wp = Panel(values, items, major, minor)
        self._data = wp._data

    def conform(self, frame, axis='items'):
        """
        Conform input DataFrame to align with chosen axis pair.

        Parameters
        ----------
        frame : DataFrame
        axis : {'items', 'major', 'minor'}

            Axis the input corresponds to. E.g., if axis='major', then
            the frame's columns would be items, and the index would be
            values of the minor axis

        Returns
        -------
        DataFrame
        """
        axes = self._get_plane_axes(axis)
        return frame.reindex(**self._extract_axes_for_slice(self, axes))

    def head(self, n=5):
        raise NotImplementedError

    def tail(self, n=5):
        raise NotImplementedError

    def round(self, decimals=0, *args, **kwargs):
        """
        Round each value in Panel to a specified number of decimal places.

        .. versionadded:: 0.18.0

        Parameters
        ----------
        decimals : int
            Number of decimal places to round to (default: 0).
            If decimals is negative, it specifies the number of
            positions to the left of the decimal point.

        Returns
        -------
        Panel object

        See Also
        --------
        numpy.around
        """
        nv.validate_round(args, kwargs)

        if is_integer(decimals):
            result = np.apply_along_axis(np.round, 0, self.values)
            return self._wrap_result(result, axis=0)
        raise TypeError("decimals must be an integer")

    def _needs_reindex_multi(self, axes, method, level):
        """
        Don't allow a multi reindex on Panel or above ndim.
        """
        return False

    def align(self, other, **kwargs):
        raise NotImplementedError

    def dropna(self, axis=0, how='any', inplace=False):
        """
        Drop 2D from panel, holding passed axis constant.

        Parameters
        ----------
        axis : int, default 0
            Axis to hold constant. E.g. axis=1 will drop major_axis entries
            having a certain amount of NA data
        how : {'all', 'any'}, default 'any'
            'any': one or more values are NA in the DataFrame along the
            axis. For 'all' they all must be.
        inplace : bool, default False
            If True, do operation inplace and return None.

        Returns
        -------
        dropped : Panel
        """
        axis = self._get_axis_number(axis)

        values = self.values
        mask = notna(values)

        for ax in reversed(sorted(set(range(self._AXIS_LEN)) - {axis})):
            mask = mask.sum(ax)

        per_slice = np.prod(values.shape[:axis] + values.shape[axis + 1:])

        if how == 'all':
            cond = mask > 0
        else:
            cond = mask == per_slice

        new_ax = self._get_axis(axis)[cond]
        result = self.reindex_axis(new_ax, axis=axis)
        if inplace:
            self._update_inplace(result)
        else:
            return result

    def _combine(self, other, func, axis=0):
        if isinstance(other, Panel):
            return self._combine_panel(other, func)
        elif isinstance(other, DataFrame):
            return self._combine_frame(other, func, axis=axis)
        elif is_scalar(other):
            return self._combine_const(other, func)
        else:
            raise NotImplementedError(
                "{otype!s} is not supported in combine operation with "
                "{selftype!s}".format(otype=type(other), selftype=type(self)))

    def _combine_const(self, other, func):
        with np.errstate(all='ignore'):
            new_values = func(self.values, other)
        d = self._construct_axes_dict()
        return self._constructor(new_values, **d)

    def _combine_frame(self, other, func, axis=0):
        index, columns = self._get_plane_axes(axis)
        axis = self._get_axis_number(axis)

        other = other.reindex(index=index, columns=columns)

        with np.errstate(all='ignore'):
            if axis == 0:
                new_values = func(self.values, other.values)
            elif axis == 1:
                new_values = func(self.values.swapaxes(0, 1), other.values.T)
                new_values = new_values.swapaxes(0, 1)
            elif axis == 2:
                new_values = func(self.values.swapaxes(0, 2), other.values)
                new_values = new_values.swapaxes(0, 2)

        return self._constructor(new_values, self.items, self.major_axis,
                                 self.minor_axis)

    def _combine_panel(self, other, func):
        items = self.items.union(other.items)
        major = self.major_axis.union(other.major_axis)
        minor = self.minor_axis.union(other.minor_axis)

        # could check that everything's the same size, but forget it
        this = self.reindex(items=items, major=major, minor=minor)
        other = other.reindex(items=items, major=major, minor=minor)

        with np.errstate(all='ignore'):
            result_values = func(this.values, other.values)

        return self._constructor(result_values, items, major, minor)

    def major_xs(self, key):
        """
        Return slice of panel along major axis.

        Parameters
        ----------
        key : object
            Major axis label

        Returns
        -------
        y : DataFrame
            index -> minor axis, columns -> items

        Notes
        -----
        major_xs is only for getting, not setting values.

        MultiIndex Slicers is a generic way to get/set values on any level or
        levels and is a superset of major_xs functionality, see
        :ref:`MultiIndex Slicers <advanced.mi_slicers>`
        """
        return self.xs(key, axis=self._AXIS_LEN - 2)

    def minor_xs(self, key):
        """
        Return slice of panel along minor axis.

        Parameters
        ----------
        key : object
            Minor axis label

        Returns
        -------
        y : DataFrame
            index -> major axis, columns -> items

        Notes
        -----
        minor_xs is only for getting, not setting values.

        MultiIndex Slicers is a generic way to get/set values on any level or
        levels and is a superset of minor_xs functionality, see
        :ref:`MultiIndex Slicers <advanced.mi_slicers>`
        """
        return self.xs(key, axis=self._AXIS_LEN - 1)

    def xs(self, key, axis=1):
        """
        Return slice of panel along selected axis.

        Parameters
        ----------
        key : object
            Label
        axis : {'items', 'major', 'minor}, default 1/'major'

        Returns
        -------
        y : ndim(self)-1

        Notes
        -----
        xs is only for getting, not setting values.

        MultiIndex Slicers is a generic way to get/set values on any level or
        levels and is a superset of xs functionality, see
        :ref:`MultiIndex Slicers <advanced.mi_slicers>`
        """
        axis = self._get_axis_number(axis)
        if axis == 0:
            return self[key]

        self._consolidate_inplace()
        axis_number = self._get_axis_number(axis)
        new_data = self._data.xs(key, axis=axis_number, copy=False)
        result = self._construct_return_type(new_data)
        copy = new_data.is_mixed_type
        result._set_is_copy(self, copy=copy)
        return result

    _xs = xs

    def _ixs(self, i, axis=0):
        """
        Parameters
        ----------
        i : int, slice, or sequence of integers
        axis : int
        """

        ax = self._get_axis(axis)
        key = ax[i]

        # xs cannot handle a non-scalar key, so just reindex here
        # if we have a multi-index and a single tuple, then its a reduction
        # (GH 7516)
        if not (isinstance(ax, MultiIndex) and isinstance(key, tuple)):
            if is_list_like(key):
                indexer = {self._get_axis_name(axis): key}
                return self.reindex(**indexer)

        # a reduction
        if axis == 0:
            values = self._data.iget(i)
            return self._box_item_values(key, values)

        # xs by position
        self._consolidate_inplace()
        new_data = self._data.xs(i, axis=axis, copy=True, takeable=True)
        return self._construct_return_type(new_data)

    def groupby(self, function, axis='major'):
        """
        Group data on given axis, returning GroupBy object.

        Parameters
        ----------
        function : callable
            Mapping function for chosen access
        axis : {'major', 'minor', 'items'}, default 'major'

        Returns
        -------
        grouped : PanelGroupBy
        """
        from pandas.core.groupby import PanelGroupBy
        axis = self._get_axis_number(axis)
        return PanelGroupBy(self, function, axis=axis)

    def to_frame(self, filter_observations=True):
        """
        Transform wide format into long (stacked) format as DataFrame whose
        columns are the Panel's items and whose index is a MultiIndex formed
        of the Panel's major and minor axes.

        Parameters
        ----------
        filter_observations : boolean, default True
            Drop (major, minor) pairs without a complete set of observations
            across all the items

        Returns
        -------
        y : DataFrame
        """
        _, N, K = self.shape

        if filter_observations:
            # shaped like the return DataFrame
            mask = notna(self.values).all(axis=0)
            # size = mask.sum()
            selector = mask.ravel()
        else:
            # size = N * K
            selector = slice(None, None)

        data = {item: self[item].values.ravel()[selector]
                for item in self.items}

        def construct_multi_parts(idx, n_repeat, n_shuffle=1):
            # Replicates and shuffles MultiIndex, returns individual attributes
            codes = [np.repeat(x, n_repeat) for x in idx.codes]
            # Assumes that each label is divisible by n_shuffle
            codes = [x.reshape(n_shuffle, -1).ravel(order='F')
                     for x in codes]
            codes = [x[selector] for x in codes]
            levels = idx.levels
            names = idx.names
            return codes, levels, names

        def construct_index_parts(idx, major=True):
            levels = [idx]
            if major:
                codes = [np.arange(N).repeat(K)[selector]]
                names = idx.name or 'major'
            else:
                codes = np.arange(K).reshape(1, K)[np.zeros(N, dtype=int)]
                codes = [codes.ravel()[selector]]
                names = idx.name or 'minor'
            names = [names]
            return codes, levels, names

        if isinstance(self.major_axis, MultiIndex):
            major_codes, major_levels, major_names = construct_multi_parts(
                self.major_axis, n_repeat=K)
        else:
            major_codes, major_levels, major_names = construct_index_parts(
                self.major_axis)

        if isinstance(self.minor_axis, MultiIndex):
            minor_codes, minor_levels, minor_names = construct_multi_parts(
                self.minor_axis, n_repeat=N, n_shuffle=K)
        else:
            minor_codes, minor_levels, minor_names = construct_index_parts(
                self.minor_axis, major=False)

        levels = major_levels + minor_levels
        codes = major_codes + minor_codes
        names = major_names + minor_names

        index = MultiIndex(levels=levels, codes=codes, names=names,
                           verify_integrity=False)

        return DataFrame(data, index=index, columns=self.items)

    def apply(self, func, axis='major', **kwargs):
        """
        Applies function along axis (or axes) of the Panel.

        Parameters
        ----------
        func : function
            Function to apply to each combination of 'other' axes
            e.g. if axis = 'items', the combination of major_axis/minor_axis
            will each be passed as a Series; if axis = ('items', 'major'),
            DataFrames of items & major axis will be passed
        axis : {'items', 'minor', 'major'}, or {0, 1, 2}, or a tuple with two
            axes
        Additional keyword arguments will be passed as keywords to the function

        Returns
        -------
        result : Panel, DataFrame, or Series

        Examples
        --------

        Returns a Panel with the square root of each element

        >>> p = pd.Panel(np.random.rand(4, 3, 2))  # doctest: +SKIP
        >>> p.apply(np.sqrt)

        Equivalent to p.sum(1), returning a DataFrame

        >>> p.apply(lambda x: x.sum(), axis=1)  # doctest: +SKIP

        Equivalent to previous:

        >>> p.apply(lambda x: x.sum(), axis='major')  # doctest: +SKIP

        Return the shapes of each DataFrame over axis 2 (i.e the shapes of
        items x major), as a Series

        >>> p.apply(lambda x: x.shape, axis=(0,1))  # doctest: +SKIP
        """

        if kwargs and not isinstance(func, np.ufunc):
            f = lambda x: func(x, **kwargs)
        else:
            f = func

        # 2d-slabs
        if isinstance(axis, (tuple, list)) and len(axis) == 2:
            return self._apply_2d(f, axis=axis)

        axis = self._get_axis_number(axis)

        # try ufunc like
        if isinstance(f, np.ufunc):
            try:
                with np.errstate(all='ignore'):
                    result = np.apply_along_axis(func, axis, self.values)
                return self._wrap_result(result, axis=axis)
            except (AttributeError):
                pass

        # 1d
        return self._apply_1d(f, axis=axis)

    def _apply_1d(self, func, axis):

        axis_name = self._get_axis_name(axis)
        ndim = self.ndim
        values = self.values

        # iter thru the axes
        slice_axis = self._get_axis(axis)
        slice_indexer = [0] * (ndim - 1)
        indexer = np.zeros(ndim, 'O')
        indlist = list(range(ndim))
        indlist.remove(axis)
        indexer[axis] = slice(None, None)
        indexer.put(indlist, slice_indexer)
        planes = [self._get_axis(axi) for axi in indlist]
        shape = np.array(self.shape).take(indlist)

        # all the iteration points
        points = cartesian_product(planes)

        results = []
        for i in range(np.prod(shape)):

            # construct the object
            pts = tuple(p[i] for p in points)
            indexer.put(indlist, slice_indexer)

            obj = Series(values[tuple(indexer)], index=slice_axis, name=pts)
            result = func(obj)

            results.append(result)

            # increment the indexer
            slice_indexer[-1] += 1
            n = -1
            while (slice_indexer[n] >= shape[n]) and (n > (1 - ndim)):
                slice_indexer[n - 1] += 1
                slice_indexer[n] = 0
                n -= 1

        # empty object
        if not len(results):
            return self._constructor(**self._construct_axes_dict())

        # same ndim as current
        if isinstance(results[0], Series):
            arr = np.vstack([r.values for r in results])
            arr = arr.T.reshape(tuple([len(slice_axis)] + list(shape)))
            tranp = np.array([axis] + indlist).argsort()
            arr = arr.transpose(tuple(list(tranp)))
            return self._constructor(arr, **self._construct_axes_dict())

        # ndim-1 shape
        results = np.array(results).reshape(shape)
        if results.ndim == 2 and axis_name != self._info_axis_name:
            results = results.T
            planes = planes[::-1]
        return self._construct_return_type(results, planes)

    def _apply_2d(self, func, axis):
        """
        Handle 2-d slices, equiv to iterating over the other axis.
        """
        ndim = self.ndim
        axis = [self._get_axis_number(a) for a in axis]

        # construct slabs, in 2-d this is a DataFrame result
        indexer_axis = list(range(ndim))
        for a in axis:
            indexer_axis.remove(a)
        indexer_axis = indexer_axis[0]

        slicer = [slice(None, None)] * ndim
        ax = self._get_axis(indexer_axis)

        results = []
        for i, e in enumerate(ax):
            slicer[indexer_axis] = i
            sliced = self.iloc[tuple(slicer)]

            obj = func(sliced)
            results.append((e, obj))

        return self._construct_return_type(dict(results))

    def _reduce(self, op, name, axis=0, skipna=True, numeric_only=None,
                filter_type=None, **kwds):
        if numeric_only:
            raise NotImplementedError('Panel.{0} does not implement '
                                      'numeric_only.'.format(name))

        if axis is None and filter_type == 'bool':
            # labels = None
            # constructor = None
            axis_number = None
            axis_name = None
        else:
            # TODO: Make other agg func handle axis=None properly
            axis = self._get_axis_number(axis)
            # labels = self._get_agg_axis(axis)
            # constructor = self._constructor
            axis_name = self._get_axis_name(axis)
            axis_number = self._get_axis_number(axis_name)

        f = lambda x: op(x, axis=axis_number, skipna=skipna, **kwds)

        with np.errstate(all='ignore'):
            result = f(self.values)

        if axis is None and filter_type == 'bool':
            return np.bool_(result)
        axes = self._get_plane_axes(axis_name)
        if result.ndim == 2 and axis_name != self._info_axis_name:
            result = result.T

        return self._construct_return_type(result, axes)

    def _construct_return_type(self, result, axes=None):
        """
        Return the type for the ndim of the result.
        """
        ndim = getattr(result, 'ndim', None)

        # need to assume they are the same
        if ndim is None:
            if isinstance(result, dict):
                ndim = getattr(list(compat.itervalues(result))[0], 'ndim', 0)

                # have a dict, so top-level is +1 dim
                if ndim != 0:
                    ndim += 1

        # scalar
        if ndim == 0:
            return Series(result)

        # same as self
        elif self.ndim == ndim:
            # return the construction dictionary for these axes
            if axes is None:
                return self._constructor(result)
            return self._constructor(result, **self._construct_axes_dict())

        # sliced
        elif self.ndim == ndim + 1:
            if axes is None:
                return self._constructor_sliced(result)
            return self._constructor_sliced(
                result, **self._extract_axes_for_slice(self, axes))

        raise ValueError('invalid _construct_return_type [self->{self}] '
                         '[result->{result}]'.format(self=self, result=result))

    def _wrap_result(self, result, axis):
        axis = self._get_axis_name(axis)
        axes = self._get_plane_axes(axis)
        if result.ndim == 2 and axis != self._info_axis_name:
            result = result.T

        return self._construct_return_type(result, axes)

    @Substitution(**_shared_doc_kwargs)
    @Appender(NDFrame.reindex.__doc__)
    def reindex(self, *args, **kwargs):
        major = kwargs.pop("major", None)
        minor = kwargs.pop('minor', None)

        if major is not None:
            if kwargs.get("major_axis"):
                raise TypeError("Cannot specify both 'major' and 'major_axis'")
            kwargs['major_axis'] = major
        if minor is not None:
            if kwargs.get("minor_axis"):
                raise TypeError("Cannot specify both 'minor' and 'minor_axis'")

            kwargs['minor_axis'] = minor
        axes = validate_axis_style_args(self, args, kwargs, 'labels',
                                        'reindex')
        kwargs.update(axes)
        kwargs.pop('axis', None)
        kwargs.pop('labels', None)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            # do not warn about constructing Panel when reindexing
            result = super(Panel, self).reindex(**kwargs)
        return result

    @Substitution(**_shared_doc_kwargs)
    @Appender(NDFrame.rename.__doc__)
    def rename(self, items=None, major_axis=None, minor_axis=None, **kwargs):
        major_axis = (major_axis if major_axis is not None else
                      kwargs.pop('major', None))
        minor_axis = (minor_axis if minor_axis is not None else
                      kwargs.pop('minor', None))
        return super(Panel, self).rename(items=items, major_axis=major_axis,
                                         minor_axis=minor_axis, **kwargs)

    @Appender(_shared_docs['reindex_axis'] % _shared_doc_kwargs)
    def reindex_axis(self, labels, axis=0, method=None, level=None, copy=True,
                     limit=None, fill_value=np.nan):
        return super(Panel, self).reindex_axis(labels=labels, axis=axis,
                                               method=method, level=level,
                                               copy=copy, limit=limit,
                                               fill_value=fill_value)

    @Substitution(**_shared_doc_kwargs)
    @Appender(NDFrame.transpose.__doc__)
    def transpose(self, *args, **kwargs):
        # check if a list of axes was passed in instead as a
        # single *args element
        if (len(args) == 1 and hasattr(args[0], '__iter__') and
                not is_string_like(args[0])):
            axes = args[0]
        else:
            axes = args

        if 'axes' in kwargs and axes:
            raise TypeError("transpose() got multiple values for "
                            "keyword argument 'axes'")
        elif not axes:
            axes = kwargs.pop('axes', ())

        return super(Panel, self).transpose(*axes, **kwargs)

    @Substitution(**_shared_doc_kwargs)
    @Appender(NDFrame.fillna.__doc__)
    def fillna(self, value=None, method=None, axis=None, inplace=False,
               limit=None, downcast=None, **kwargs):
        return super(Panel, self).fillna(value=value, method=method, axis=axis,
                                         inplace=inplace, limit=limit,
                                         downcast=downcast, **kwargs)

    def count(self, axis='major'):
        """
        Return number of observations over requested axis.

        Parameters
        ----------
        axis : {'items', 'major', 'minor'} or {0, 1, 2}

        Returns
        -------
        count : DataFrame
        """
        i = self._get_axis_number(axis)

        values = self.values
        mask = np.isfinite(values)
        result = mask.sum(axis=i, dtype='int64')

        return self._wrap_result(result, axis)

    def shift(self, periods=1, freq=None, axis='major'):
        """
        Shift index by desired number of periods with an optional time freq.

        The shifted data will not include the dropped periods and the
        shifted axis will be smaller than the original. This is different
        from the behavior of DataFrame.shift()

        Parameters
        ----------
        periods : int
            Number of periods to move, can be positive or negative
        freq : DateOffset, timedelta, or time rule string, optional
        axis : {'items', 'major', 'minor'} or {0, 1, 2}

        Returns
        -------
        shifted : Panel
        """
        if freq:
            return self.tshift(periods, freq, axis=axis)

        return super(Panel, self).slice_shift(periods, axis=axis)

    def tshift(self, periods=1, freq=None, axis='major'):
        return super(Panel, self).tshift(periods, freq, axis)

    def join(self, other, how='left', lsuffix='', rsuffix=''):
        """
        Join items with other Panel either on major and minor axes column.

        Parameters
        ----------
        other : Panel or list of Panels
            Index should be similar to one of the columns in this one
        how : {'left', 'right', 'outer', 'inner'}
            How to handle indexes of the two objects. Default: 'left'
            for joining on index, None otherwise
            * left: use calling frame's index
            * right: use input frame's index
            * outer: form union of indexes
            * inner: use intersection of indexes
        lsuffix : string
            Suffix to use from left frame's overlapping columns
        rsuffix : string
            Suffix to use from right frame's overlapping columns

        Returns
        -------
        joined : Panel
        """
        from pandas.core.reshape.concat import concat

        if isinstance(other, Panel):
            join_major, join_minor = self._get_join_index(other, how)
            this = self.reindex(major=join_major, minor=join_minor)
            other = other.reindex(major=join_major, minor=join_minor)
            merged_data = this._data.merge(other._data, lsuffix, rsuffix)
            return self._constructor(merged_data)
        else:
            if lsuffix or rsuffix:
                raise ValueError('Suffixes not supported when passing '
                                 'multiple panels')

            if how == 'left':
                how = 'outer'
                join_axes = [self.major_axis, self.minor_axis]
            elif how == 'right':
                raise ValueError('Right join not supported with multiple '
                                 'panels')
            else:
                join_axes = None

            return concat([self] + list(other), axis=0, join=how,
                          join_axes=join_axes, verify_integrity=True)

    @deprecate_kwarg(old_arg_name='raise_conflict', new_arg_name='errors',
                     mapping={False: 'ignore', True: 'raise'})
    def update(self, other, join='left', overwrite=True, filter_func=None,
               errors='ignore'):
        """
        Modify Panel in place using non-NA values from other Panel.

        May also use object coercible to Panel. Will align on items.

        Parameters
        ----------
        other : Panel, or object coercible to Panel
            The object from which the caller will be udpated.
        join : {'left', 'right', 'outer', 'inner'}, default 'left'
            How individual DataFrames are joined.
        overwrite : bool, default True
            If True then overwrite values for common keys in the calling Panel.
        filter_func : callable(1d-array) -> 1d-array<bool>, default None
            Can choose to replace values other than NA. Return True for values
            that should be updated.
        errors : {'raise', 'ignore'}, default 'ignore'
            If 'raise', will raise an error if a DataFrame and other both.

            .. versionchanged :: 0.24.0
               Changed from `raise_conflict=False|True`
               to `errors='ignore'|'raise'`.

        See Also
        --------
        DataFrame.update : Similar method for DataFrames.
        dict.update : Similar method for dictionaries.
        """

        if not isinstance(other, self._constructor):
            other = self._constructor(other)

        axis_name = self._info_axis_name
        axis_values = self._info_axis
        other = other.reindex(**{axis_name: axis_values})

        for frame in axis_values:
            self[frame].update(other[frame], join=join, overwrite=overwrite,
                               filter_func=filter_func, errors=errors)

    def _get_join_index(self, other, how):
        if how == 'left':
            join_major, join_minor = self.major_axis, self.minor_axis
        elif how == 'right':
            join_major, join_minor = other.major_axis, other.minor_axis
        elif how == 'inner':
            join_major = self.major_axis.intersection(other.major_axis)
            join_minor = self.minor_axis.intersection(other.minor_axis)
        elif how == 'outer':
            join_major = self.major_axis.union(other.major_axis)
            join_minor = self.minor_axis.union(other.minor_axis)
        return join_major, join_minor

    # miscellaneous data creation
    @staticmethod
    def _extract_axes(self, data, axes, **kwargs):
        """
        Return a list of the axis indices.
        """
        return [self._extract_axis(self, data, axis=i, **kwargs)
                for i, a in enumerate(axes)]

    @staticmethod
    def _extract_axes_for_slice(self, axes):
        """
        Return the slice dictionary for these axes.
        """
        return {self._AXIS_SLICEMAP[i]: a for i, a in
                zip(self._AXIS_ORDERS[self._AXIS_LEN - len(axes):], axes)}

    @staticmethod
    def _prep_ndarray(self, values, copy=True):
        if not isinstance(values, np.ndarray):
            values = np.asarray(values)
            # NumPy strings are a pain, convert to object
            if issubclass(values.dtype.type, compat.string_types):
                values = np.array(values, dtype=object, copy=True)
        else:
            if copy:
                values = values.copy()
        if values.ndim != self._AXIS_LEN:
            raise ValueError("The number of dimensions required is {0}, "
                             "but the number of dimensions of the "
                             "ndarray given was {1}".format(self._AXIS_LEN,
                                                            values.ndim))
        return values

    @staticmethod
    def _homogenize_dict(self, frames, intersect=True, dtype=None):
        """
        Conform set of _constructor_sliced-like objects to either
        an intersection of indices / columns or a union.

        Parameters
        ----------
        frames : dict
        intersect : boolean, default True

        Returns
        -------
        dict of aligned results & indices
        """

        result = dict()
        # caller differs dict/ODict, preserved type
        if isinstance(frames, OrderedDict):
            result = OrderedDict()

        adj_frames = OrderedDict()
        for k, v in compat.iteritems(frames):
            if isinstance(v, dict):
                adj_frames[k] = self._constructor_sliced(v)
            else:
                adj_frames[k] = v

        axes = self._AXIS_ORDERS[1:]
        axes_dict = {a: ax for a, ax in zip(axes, self._extract_axes(
                     self, adj_frames, axes, intersect=intersect))}

        reindex_dict = {self._AXIS_SLICEMAP[a]: axes_dict[a] for a in axes}
        reindex_dict['copy'] = False
        for key, frame in compat.iteritems(adj_frames):
            if frame is not None:
                result[key] = frame.reindex(**reindex_dict)
            else:
                result[key] = None

        axes_dict['data'] = result
        axes_dict['dtype'] = dtype
        return axes_dict

    @staticmethod
    def _extract_axis(self, data, axis=0, intersect=False):

        index = None
        if len(data) == 0:
            index = Index([])
        elif len(data) > 0:
            raw_lengths = []

        have_raw_arrays = False
        have_frames = False

        for v in data.values():
            if isinstance(v, self._constructor_sliced):
                have_frames = True
            elif v is not None:
                have_raw_arrays = True
                raw_lengths.append(v.shape[axis])

        if have_frames:
            # we want the "old" behavior here, of sorting only
            # 1. we're doing a union (intersect=False)
            # 2. the indices are not aligned.
            index = _get_objs_combined_axis(data.values(), axis=axis,
                                            intersect=intersect, sort=None)

        if have_raw_arrays:
            lengths = list(set(raw_lengths))
            if len(lengths) > 1:
                raise ValueError('ndarrays must match shape on '
                                 'axis {ax}'.format(ax=axis))

            if have_frames:
                if lengths[0] != len(index):
                    raise AssertionError('Length of data and index must match')
            else:
                index = Index(np.arange(lengths[0]))

        if index is None:
            index = Index([])

        return ensure_index(index)

    def sort_values(self, *args, **kwargs):
        """
        NOT IMPLEMENTED: do not call this method, as sorting values is not
        supported for Panel objects and will raise an error.
        """
        super(Panel, self).sort_values(*args, **kwargs)


Panel._setup_axes(axes=['items', 'major_axis', 'minor_axis'], info_axis=0,
                  stat_axis=1, aliases={'major': 'major_axis',
                                        'minor': 'minor_axis'},
                  slicers={'major_axis': 'index',
                           'minor_axis': 'columns'},
                  docs={})

ops.add_special_arithmetic_methods(Panel)
ops.add_flex_arithmetic_methods(Panel)
Panel._add_numeric_operations()
