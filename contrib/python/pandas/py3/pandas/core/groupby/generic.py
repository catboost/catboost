"""
Define the SeriesGroupBy and DataFrameGroupBy
classes that hold the groupby interfaces (and some implementations).

These are user facing as the result of the ``df.groupby(...)`` operations,
which here returns a DataFrameGroupBy object.
"""
from __future__ import annotations

from collections import abc
from functools import partial
from textwrap import dedent
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Hashable,
    Iterable,
    Mapping,
    NamedTuple,
    Sequence,
    TypeVar,
    Union,
    cast,
)
import warnings

import numpy as np

from pandas._libs import (
    Interval,
    lib,
    reduction as libreduction,
)
from pandas._typing import (
    ArrayLike,
    Manager,
    Manager2D,
    SingleManager,
)
from pandas.errors import SpecificationError
from pandas.util._decorators import (
    Appender,
    Substitution,
    doc,
)
from pandas.util._exceptions import find_stack_level

from pandas.core.dtypes.common import (
    ensure_int64,
    is_bool,
    is_categorical_dtype,
    is_dict_like,
    is_integer_dtype,
    is_interval_dtype,
    is_scalar,
)
from pandas.core.dtypes.missing import (
    isna,
    notna,
)

from pandas.core import (
    algorithms,
    nanops,
)
from pandas.core.apply import (
    GroupByApply,
    maybe_mangle_lambdas,
    reconstruct_func,
    validate_func_kwargs,
)
from pandas.core.arrays.categorical import Categorical
import pandas.core.common as com
from pandas.core.construction import create_series_with_explicit_dtype
from pandas.core.frame import DataFrame
from pandas.core.groupby import base
from pandas.core.groupby.groupby import (
    GroupBy,
    _agg_template,
    _apply_docs,
    _transform_template,
    warn_dropping_nuisance_columns_deprecated,
)
from pandas.core.groupby.grouper import get_grouper
from pandas.core.indexes.api import (
    Index,
    MultiIndex,
    all_indexes_same,
)
from pandas.core.indexes.category import CategoricalIndex
from pandas.core.series import Series
from pandas.core.shared_docs import _shared_docs
from pandas.core.util.numba_ import maybe_use_numba

from pandas.plotting import boxplot_frame_groupby

if TYPE_CHECKING:
    from pandas.core.generic import NDFrame

# TODO(typing) the return value on this callable should be any *scalar*.
AggScalar = Union[str, Callable[..., Any]]
# TODO: validate types on ScalarResult and move to _typing
# Blocked from using by https://github.com/python/mypy/issues/1484
# See note at _mangle_lambda_list
ScalarResult = TypeVar("ScalarResult")


class NamedAgg(NamedTuple):
    column: Hashable
    aggfunc: AggScalar


def generate_property(name: str, klass: type[DataFrame | Series]):
    """
    Create a property for a GroupBy subclass to dispatch to DataFrame/Series.

    Parameters
    ----------
    name : str
    klass : {DataFrame, Series}

    Returns
    -------
    property
    """

    def prop(self):
        return self._make_wrapper(name)

    parent_method = getattr(klass, name)
    prop.__doc__ = parent_method.__doc__ or ""
    prop.__name__ = name
    return property(prop)


def pin_allowlisted_properties(
    klass: type[DataFrame | Series], allowlist: frozenset[str]
):
    """
    Create GroupBy member defs for DataFrame/Series names in a allowlist.

    Parameters
    ----------
    klass : DataFrame or Series class
        class where members are defined.
    allowlist : frozenset[str]
        Set of names of klass methods to be constructed

    Returns
    -------
    class decorator

    Notes
    -----
    Since we don't want to override methods explicitly defined in the
    base class, any such name is skipped.
    """

    def pinner(cls):
        for name in allowlist:
            if hasattr(cls, name):
                # don't override anything that was explicitly defined
                #  in the base class
                continue

            prop = generate_property(name, klass)
            setattr(cls, name, prop)

        return cls

    return pinner


@pin_allowlisted_properties(Series, base.series_apply_allowlist)
class SeriesGroupBy(GroupBy[Series]):
    _apply_allowlist = base.series_apply_allowlist

    def _wrap_agged_manager(self, mgr: Manager) -> Series:
        if mgr.ndim == 1:
            mgr = cast(SingleManager, mgr)
            single = mgr
        else:
            mgr = cast(Manager2D, mgr)
            single = mgr.iget(0)
        ser = self.obj._constructor(single, name=self.obj.name)
        # NB: caller is responsible for setting ser.index
        return ser

    def _get_data_to_aggregate(self) -> SingleManager:
        ser = self._obj_with_exclusions
        single = ser._mgr
        return single

    def _iterate_slices(self) -> Iterable[Series]:
        yield self._selected_obj

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> s = pd.Series([1, 2, 3, 4])

    >>> s
    0    1
    1    2
    2    3
    3    4
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).min()
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg('min')
    1    1
    2    3
    dtype: int64

    >>> s.groupby([1, 1, 2, 2]).agg(['min', 'max'])
       min  max
    1    1    2
    2    3    4

    The output column names can be controlled by passing
    the desired column names and aggregations as keyword arguments.

    >>> s.groupby([1, 1, 2, 2]).agg(
    ...     minimum='min',
    ...     maximum='max',
    ... )
       minimum  maximum
    1        1        2
    2        3        4

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> s.groupby([1, 1, 2, 2]).agg(lambda x: x.astype(float).min())
    1    1.0
    2    3.0
    dtype: float64
    """
    )

    @Appender(
        _apply_docs["template"].format(
            input="series", examples=_apply_docs["series_examples"]
        )
    )
    def apply(self, func, *args, **kwargs) -> Series:
        return super().apply(func, *args, **kwargs)

    @doc(_agg_template, examples=_agg_examples_doc, klass="Series")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):

        if maybe_use_numba(engine):
            with self._group_selection_context():
                data = self._selected_obj
            result = self._aggregate_with_numba(
                data.to_frame(), func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
            index = self.grouper.result_index
            return self.obj._constructor(result.ravel(), index=index, name=data.name)

        relabeling = func is None
        columns = None
        if relabeling:
            columns, func = validate_func_kwargs(kwargs)
            kwargs = {}

        if isinstance(func, str):
            return getattr(self, func)(*args, **kwargs)

        elif isinstance(func, abc.Iterable):
            # Catch instances of lists / tuples
            # but not the class list / tuple itself.
            func = maybe_mangle_lambdas(func)
            ret = self._aggregate_multiple_funcs(func)
            if relabeling:
                # columns is not narrowed by mypy from relabeling flag
                assert columns is not None  # for mypy
                ret.columns = columns
            return ret

        else:
            cyfunc = com.get_cython_func(func)
            if cyfunc and not args and not kwargs:
                return getattr(self, cyfunc)()

            if self.grouper.nkeys > 1:
                return self._python_agg_general(func, *args, **kwargs)

            try:
                return self._python_agg_general(func, *args, **kwargs)
            except KeyError:
                # TODO: KeyError is raised in _python_agg_general,
                #  see test_groupby.test_basic
                result = self._aggregate_named(func, *args, **kwargs)

                # result is a dict whose keys are the elements of result_index
                index = self.grouper.result_index
                return create_series_with_explicit_dtype(
                    result, index=index, dtype_if_empty=object
                )

    agg = aggregate

    def _aggregate_multiple_funcs(self, arg) -> DataFrame:
        if isinstance(arg, dict):

            # show the deprecation, but only if we
            # have not shown a higher level one
            # GH 15931
            raise SpecificationError("nested renamer is not supported")

        elif any(isinstance(x, (tuple, list)) for x in arg):
            arg = [(x, x) if not isinstance(x, (tuple, list)) else x for x in arg]

            # indicated column order
            columns = next(zip(*arg))
        else:
            # list of functions / function names
            columns = []
            for f in arg:
                columns.append(com.get_callable_name(f) or f)

            arg = zip(columns, arg)

        results: dict[base.OutputKey, DataFrame | Series] = {}
        for idx, (name, func) in enumerate(arg):

            key = base.OutputKey(label=name, position=idx)
            results[key] = self.aggregate(func)

        if any(isinstance(x, DataFrame) for x in results.values()):
            from pandas import concat

            res_df = concat(
                results.values(), axis=1, keys=[key.label for key in results.keys()]
            )
            return res_df

        indexed_output = {key.position: val for key, val in results.items()}
        output = self.obj._constructor_expanddim(indexed_output, index=None)
        output.columns = Index(key.label for key in results)

        output = self._reindex_output(output)
        return output

    def _indexed_output_to_ndframe(
        self, output: Mapping[base.OutputKey, ArrayLike]
    ) -> Series:
        """
        Wrap the dict result of a GroupBy aggregation into a Series.
        """
        assert len(output) == 1
        values = next(iter(output.values()))
        result = self.obj._constructor(values)
        result.name = self.obj.name
        return result

    def _wrap_applied_output(
        self,
        data: Series,
        values: list[Any],
        not_indexed_same: bool = False,
        override_group_keys: bool = False,
    ) -> DataFrame | Series:
        """
        Wrap the output of SeriesGroupBy.apply into the expected result.

        Parameters
        ----------
        data : Series
            Input data for groupby operation.
        values : List[Any]
            Applied output for each group.
        not_indexed_same : bool, default False
            Whether the applied outputs are not indexed the same as the group axes.

        Returns
        -------
        DataFrame or Series
        """
        if len(values) == 0:
            # GH #6265
            return self.obj._constructor(
                [],
                name=self.obj.name,
                index=self.grouper.result_index,
                dtype=data.dtype,
            )
        assert values is not None

        if isinstance(values[0], dict):
            # GH #823 #24880
            index = self.grouper.result_index
            res_df = self.obj._constructor_expanddim(values, index=index)
            res_df = self._reindex_output(res_df)
            # if self.observed is False,
            # keep all-NaN rows created while re-indexing
            res_ser = res_df.stack(dropna=self.observed)
            res_ser.name = self.obj.name
            return res_ser
        elif isinstance(values[0], (Series, DataFrame)):
            result = self._concat_objects(
                values,
                not_indexed_same=not_indexed_same,
                override_group_keys=override_group_keys,
            )
            if isinstance(result, Series):
                result.name = self.obj.name
            return result
        else:
            # GH #6265 #24880
            result = self.obj._constructor(
                data=values, index=self.grouper.result_index, name=self.obj.name
            )
            return self._reindex_output(result)

    def _aggregate_named(self, func, *args, **kwargs):
        # Note: this is very similar to _aggregate_series_pure_python,
        #  but that does not pin group.name
        result = {}
        initialized = False

        for name, group in self:
            object.__setattr__(group, "name", name)

            output = func(group, *args, **kwargs)
            output = libreduction.extract_result(output)
            if not initialized:
                # We only do this validation on the first iteration
                libreduction.check_result_array(output, group.dtype)
                initialized = True
            result[name] = output

        return result

    @Substitution(klass="Series")
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    def _cython_transform(
        self, how: str, numeric_only: bool = True, axis: int = 0, **kwargs
    ):
        assert axis == 0  # handled by caller

        obj = self._selected_obj

        try:
            result = self.grouper._cython_operation(
                "transform", obj._values, how, axis, **kwargs
            )
        except NotImplementedError as err:
            raise TypeError(f"{how} is not supported for {obj.dtype} dtype") from err

        return obj._constructor(result, index=self.obj.index, name=obj.name)

    def _transform_general(self, func: Callable, *args, **kwargs) -> Series:
        """
        Transform with a callable func`.
        """
        assert callable(func)
        klass = type(self.obj)

        results = []
        for name, group in self.grouper.get_iterator(
            self._selected_obj, axis=self.axis
        ):
            # this setattr is needed for test_transform_lambda_with_datetimetz
            object.__setattr__(group, "name", name)
            res = func(group, *args, **kwargs)

            results.append(klass(res, index=group.index))

        # check for empty "results" to avoid concat ValueError
        if results:
            from pandas.core.reshape.concat import concat

            concatenated = concat(results)
            result = self._set_result_index_ordered(concatenated)
        else:
            result = self.obj._constructor(dtype=np.float64)

        result.name = self.obj.name
        return result

    def filter(self, func, dropna: bool = True, *args, **kwargs):
        """
        Return a copy of a Series excluding elements from groups that
        do not satisfy the boolean criterion specified by func.

        Parameters
        ----------
        func : function
            To apply to each group. Should return True or False.
        dropna : Drop groups that do not pass the filter. True by default;
            if False, groups that evaluate False are filled with NaNs.

        Notes
        -----
        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> df.groupby('A').B.filter(lambda x: x.mean() > 3.)
        1    2
        3    4
        5    6
        Name: B, dtype: int64

        Returns
        -------
        filtered : Series
        """
        if isinstance(func, str):
            wrapper = lambda x: getattr(x, func)(*args, **kwargs)
        else:
            wrapper = lambda x: func(x, *args, **kwargs)

        # Interpret np.nan as False.
        def true_and_notna(x) -> bool:
            b = wrapper(x)
            return b and notna(b)

        try:
            indices = [
                self._get_index(name) for name, group in self if true_and_notna(group)
            ]
        except (ValueError, TypeError) as err:
            raise TypeError("the filter must return a boolean result") from err

        filtered = self._apply_filter(indices, dropna)
        return filtered

    def nunique(self, dropna: bool = True) -> Series:
        """
        Return number of unique elements in the group.

        Returns
        -------
        Series
            Number of unique values within each group.
        """
        ids, _, _ = self.grouper.group_info

        val = self.obj._values

        codes, _ = algorithms.factorize(val, sort=False)
        sorter = np.lexsort((codes, ids))
        codes = codes[sorter]
        ids = ids[sorter]

        # group boundaries are where group ids change
        # unique observations are where sorted values change
        idx = np.r_[0, 1 + np.nonzero(ids[1:] != ids[:-1])[0]]
        inc = np.r_[1, codes[1:] != codes[:-1]]

        # 1st item of each group is a new unique observation
        mask = codes == -1
        if dropna:
            inc[idx] = 1
            inc[mask] = 0
        else:
            inc[mask & np.r_[False, mask[:-1]]] = 0
            inc[idx] = 1

        out = np.add.reduceat(inc, idx).astype("int64", copy=False)
        if len(ids):
            # NaN/NaT group exists if the head of ids is -1,
            # so remove it from res and exclude its index from idx
            if ids[0] == -1:
                res = out[1:]
                idx = idx[np.flatnonzero(idx)]
            else:
                res = out
        else:
            res = out[1:]
        ri = self.grouper.result_index

        # we might have duplications among the bins
        if len(res) != len(ri):
            res, out = np.zeros(len(ri), dtype=out.dtype), res
            res[ids[idx]] = out

        result = self.obj._constructor(res, index=ri, name=self.obj.name)
        return self._reindex_output(result, fill_value=0)

    @doc(Series.describe)
    def describe(self, **kwargs):
        return super().describe(**kwargs)

    def value_counts(
        self,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        bins=None,
        dropna: bool = True,
    ) -> Series:

        from pandas.core.reshape.merge import get_join_indexers
        from pandas.core.reshape.tile import cut

        ids, _, _ = self.grouper.group_info
        val = self.obj._values

        names = self.grouper.names + [self.obj.name]

        if is_categorical_dtype(val.dtype) or (
            bins is not None and not np.iterable(bins)
        ):
            # scalar bins cannot be done at top level
            # in a backward compatible way
            # GH38672 relates to categorical dtype
            ser = self.apply(
                Series.value_counts,
                normalize=normalize,
                sort=sort,
                ascending=ascending,
                bins=bins,
            )
            ser.index.names = names
            return ser

        # groupby removes null keys from groupings
        mask = ids != -1
        ids, val = ids[mask], val[mask]

        if bins is None:
            lab, lev = algorithms.factorize(val, sort=True)
            llab = lambda lab, inc: lab[inc]
        else:

            # lab is a Categorical with categories an IntervalIndex
            lab = cut(Series(val), bins, include_lowest=True)
            # error: "ndarray" has no attribute "cat"
            lev = lab.cat.categories  # type: ignore[attr-defined]
            # error: No overload variant of "take" of "_ArrayOrScalarCommon" matches
            # argument types "Any", "bool", "Union[Any, float]"
            lab = lev.take(  # type: ignore[call-overload]
                # error: "ndarray" has no attribute "cat"
                lab.cat.codes,  # type: ignore[attr-defined]
                allow_fill=True,
                # error: Item "ndarray" of "Union[ndarray, Index]" has no attribute
                # "_na_value"
                fill_value=lev._na_value,  # type: ignore[union-attr]
            )
            llab = lambda lab, inc: lab[inc]._multiindex.codes[-1]

        if is_interval_dtype(lab.dtype):
            # TODO: should we do this inside II?
            lab_interval = cast(Interval, lab)

            sorter = np.lexsort((lab_interval.left, lab_interval.right, ids))
        else:
            sorter = np.lexsort((lab, ids))

        ids, lab = ids[sorter], lab[sorter]

        # group boundaries are where group ids change
        idchanges = 1 + np.nonzero(ids[1:] != ids[:-1])[0]
        idx = np.r_[0, idchanges]
        if not len(ids):
            idx = idchanges

        # new values are where sorted labels change
        lchanges = llab(lab, slice(1, None)) != llab(lab, slice(None, -1))
        inc = np.r_[True, lchanges]
        if not len(val):
            inc = lchanges
        inc[idx] = True  # group boundaries are also new values
        out = np.diff(np.nonzero(np.r_[inc, True])[0])  # value counts

        # num. of times each group should be repeated
        rep = partial(np.repeat, repeats=np.add.reduceat(inc, idx))

        # multi-index components
        codes = self.grouper.reconstructed_codes
        # error: Incompatible types in assignment (expression has type
        # "List[ndarray[Any, dtype[_SCT]]]",
        # variable has type "List[ndarray[Any, dtype[signedinteger[Any]]]]")
        codes = [  # type: ignore[assignment]
            rep(level_codes) for level_codes in codes
        ] + [llab(lab, inc)]
        # error: List item 0 has incompatible type "Union[ndarray[Any, Any], Index]";
        # expected "Index"
        levels = [ping.group_index for ping in self.grouper.groupings] + [
            lev  # type: ignore[list-item]
        ]

        if dropna:
            mask = codes[-1] != -1
            if mask.all():
                dropna = False
            else:
                out, codes = out[mask], [level_codes[mask] for level_codes in codes]

        if normalize:
            out = out.astype("float")
            d = np.diff(np.r_[idx, len(ids)])
            if dropna:
                m = ids[lab == -1]
                np.add.at(d, m, -1)
                acc = rep(d)[mask]
            else:
                acc = rep(d)
            out /= acc

        if sort and bins is None:
            cat = ids[inc][mask] if dropna else ids[inc]
            sorter = np.lexsort((out if ascending else -out, cat))
            out, codes[-1] = out[sorter], codes[-1][sorter]

        if bins is not None:
            # for compat. with libgroupby.value_counts need to ensure every
            # bin is present at every index level, null filled with zeros
            diff = np.zeros(len(out), dtype="bool")
            for level_codes in codes[:-1]:
                diff |= np.r_[True, level_codes[1:] != level_codes[:-1]]

            ncat, nbin = diff.sum(), len(levels[-1])

            left = [np.repeat(np.arange(ncat), nbin), np.tile(np.arange(nbin), ncat)]

            right = [diff.cumsum() - 1, codes[-1]]

            _, idx = get_join_indexers(left, right, sort=False, how="left")
            out = np.where(idx != -1, out[idx], 0)

            if sort:
                sorter = np.lexsort((out if ascending else -out, left[0]))
                out, left[-1] = out[sorter], left[-1][sorter]

            # build the multi-index w/ full levels
            def build_codes(lev_codes: np.ndarray) -> np.ndarray:
                return np.repeat(lev_codes[diff], nbin)

            codes = [build_codes(lev_codes) for lev_codes in codes[:-1]]
            codes.append(left[-1])

        mi = MultiIndex(levels=levels, codes=codes, names=names, verify_integrity=False)

        if is_integer_dtype(out.dtype):
            out = ensure_int64(out)
        return self.obj._constructor(out, index=mi, name=self.obj.name)

    @doc(Series.nlargest)
    def nlargest(self, n: int = 5, keep: str = "first") -> Series:
        f = partial(Series.nlargest, n=n, keep=keep)
        data = self._obj_with_exclusions
        # Don't change behavior if result index happens to be the same, i.e.
        # already ordered and n >= all group sizes.
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result

    @doc(Series.nsmallest)
    def nsmallest(self, n: int = 5, keep: str = "first") -> Series:
        f = partial(Series.nsmallest, n=n, keep=keep)
        data = self._obj_with_exclusions
        # Don't change behavior if result index happens to be the same, i.e.
        # already ordered and n >= all group sizes.
        result = self._python_apply_general(f, data, not_indexed_same=True)
        return result


@pin_allowlisted_properties(DataFrame, base.dataframe_apply_allowlist)
class DataFrameGroupBy(GroupBy[DataFrame]):

    _apply_allowlist = base.dataframe_apply_allowlist

    _agg_examples_doc = dedent(
        """
    Examples
    --------
    >>> df = pd.DataFrame(
    ...     {
    ...         "A": [1, 1, 2, 2],
    ...         "B": [1, 2, 3, 4],
    ...         "C": [0.362838, 0.227877, 1.267767, -0.562860],
    ...     }
    ... )

    >>> df
       A  B         C
    0  1  1  0.362838
    1  1  2  0.227877
    2  2  3  1.267767
    3  2  4 -0.562860

    The aggregation is for each column.

    >>> df.groupby('A').agg('min')
       B         C
    A
    1  1  0.227877
    2  3 -0.562860

    Multiple aggregations

    >>> df.groupby('A').agg(['min', 'max'])
        B             C
      min max       min       max
    A
    1   1   2  0.227877  0.362838
    2   3   4 -0.562860  1.267767

    Select a column for aggregation

    >>> df.groupby('A').B.agg(['min', 'max'])
       min  max
    A
    1    1    2
    2    3    4

    User-defined function for aggregation

    >>> df.groupby('A').agg(lambda x: sum(x) + 2)
        B	       C
    A
    1	5	2.590715
    2	9	2.704907

    Different aggregations per column

    >>> df.groupby('A').agg({'B': ['min', 'max'], 'C': 'sum'})
        B             C
      min max       sum
    A
    1   1   2  0.590715
    2   3   4  0.704907

    To control the output names with different aggregations per column,
    pandas supports "named aggregation"

    >>> df.groupby("A").agg(
    ...     b_min=pd.NamedAgg(column="B", aggfunc="min"),
    ...     c_sum=pd.NamedAgg(column="C", aggfunc="sum"))
       b_min     c_sum
    A
    1      1  0.590715
    2      3  0.704907

    - The keywords are the *output* column names
    - The values are tuples whose first element is the column to select
      and the second element is the aggregation to apply to that column.
      Pandas provides the ``pandas.NamedAgg`` namedtuple with the fields
      ``['column', 'aggfunc']`` to make it clearer what the arguments are.
      As usual, the aggregation can be a callable or a string alias.

    See :ref:`groupby.aggregate.named` for more.

    .. versionchanged:: 1.3.0

        The resulting dtype will reflect the return value of the aggregating function.

    >>> df.groupby("A")[["B"]].agg(lambda x: x.astype(float).min())
          B
    A
    1   1.0
    2   3.0
    """
    )

    @doc(_agg_template, examples=_agg_examples_doc, klass="DataFrame")
    def aggregate(self, func=None, *args, engine=None, engine_kwargs=None, **kwargs):

        if maybe_use_numba(engine):
            with self._group_selection_context():
                data = self._selected_obj
            result = self._aggregate_with_numba(
                data, func, *args, engine_kwargs=engine_kwargs, **kwargs
            )
            index = self.grouper.result_index
            return self.obj._constructor(result, index=index, columns=data.columns)

        relabeling, func, columns, order = reconstruct_func(func, **kwargs)
        func = maybe_mangle_lambdas(func)

        op = GroupByApply(self, func, args, kwargs)
        result = op.agg()
        if not is_dict_like(func) and result is not None:
            return result
        elif relabeling and result is not None:
            # this should be the only (non-raising) case with relabeling
            # used reordered index of columns
            result = result.iloc[:, order]
            result.columns = columns

        if result is None:

            # grouper specific aggregations
            if self.grouper.nkeys > 1:
                # test_groupby_as_index_series_scalar gets here with 'not self.as_index'
                return self._python_agg_general(func, *args, **kwargs)
            elif args or kwargs:
                # test_pass_args_kwargs gets here (with and without as_index)
                # can't return early
                result = self._aggregate_frame(func, *args, **kwargs)

            elif self.axis == 1:
                # _aggregate_multiple_funcs does not allow self.axis == 1
                # Note: axis == 1 precludes 'not self.as_index', see __init__
                result = self._aggregate_frame(func)
                return result

            else:

                # try to treat as if we are passing a list
                gba = GroupByApply(self, [func], args=(), kwargs={})
                try:
                    result = gba.agg()

                except ValueError as err:
                    if "no results" not in str(err):
                        # raised directly by _aggregate_multiple_funcs
                        raise
                    result = self._aggregate_frame(func)

                else:
                    sobj = self._selected_obj

                    if isinstance(sobj, Series):
                        # GH#35246 test_groupby_as_index_select_column_sum_empty_df
                        result.columns = self._obj_with_exclusions.columns.copy()
                    else:
                        # Retain our column names
                        result.columns._set_names(
                            sobj.columns.names, level=list(range(sobj.columns.nlevels))
                        )
                        # select everything except for the last level, which is the one
                        # containing the name of the function(s), see GH#32040
                        result.columns = result.columns.droplevel(-1)

        if not self.as_index:
            self._insert_inaxis_grouper_inplace(result)
            result.index = Index(range(len(result)))

        return result

    agg = aggregate

    def _iterate_slices(self) -> Iterable[Series]:
        obj = self._selected_obj
        if self.axis == 1:
            obj = obj.T

        if isinstance(obj, Series) and obj.name not in self.exclusions:
            # Occurs when doing DataFrameGroupBy(...)["X"]
            yield obj
        else:
            for label, values in obj.items():
                if label in self.exclusions:
                    continue

                yield values

    def _aggregate_frame(self, func, *args, **kwargs) -> DataFrame:
        if self.grouper.nkeys != 1:
            raise AssertionError("Number of keys must be 1")

        obj = self._obj_with_exclusions

        result: dict[Hashable, NDFrame | np.ndarray] = {}
        if self.axis == 0:
            # test_pass_args_kwargs_duplicate_columns gets here with non-unique columns
            for name, data in self.grouper.get_iterator(obj, self.axis):
                fres = func(data, *args, **kwargs)
                result[name] = fres
        else:
            # we get here in a number of test_multilevel tests
            for name in self.indices:
                grp_df = self.get_group(name, obj=obj)
                fres = func(grp_df, *args, **kwargs)
                result[name] = fres

        result_index = self.grouper.result_index
        other_ax = obj.axes[1 - self.axis]
        out = self.obj._constructor(result, index=other_ax, columns=result_index)
        if self.axis == 0:
            out = out.T

        return out

    def _aggregate_item_by_item(self, func, *args, **kwargs) -> DataFrame:
        # only for axis==0
        # tests that get here with non-unique cols:
        #  test_resample_with_timedelta_yields_no_empty_groups,
        #  test_resample_apply_product

        obj = self._obj_with_exclusions
        result: dict[int, NDFrame] = {}

        for i, (item, sgb) in enumerate(self._iterate_column_groupbys(obj)):
            result[i] = sgb.aggregate(func, *args, **kwargs)

        res_df = self.obj._constructor(result)
        res_df.columns = obj.columns
        return res_df

    def _wrap_applied_output(
        self,
        data: DataFrame,
        values: list,
        not_indexed_same: bool = False,
        override_group_keys: bool = False,
    ):

        if len(values) == 0:
            result = self.obj._constructor(
                index=self.grouper.result_index, columns=data.columns
            )
            result = result.astype(data.dtypes, copy=False)
            return result

        # GH12824
        first_not_none = next(com.not_none(*values), None)

        if first_not_none is None:
            # GH9684 - All values are None, return an empty frame.
            return self.obj._constructor()
        elif isinstance(first_not_none, DataFrame):
            return self._concat_objects(
                values,
                not_indexed_same=not_indexed_same,
                override_group_keys=override_group_keys,
            )

        key_index = self.grouper.result_index if self.as_index else None

        if isinstance(first_not_none, (np.ndarray, Index)):
            # GH#1738: values is list of arrays of unequal lengths
            #  fall through to the outer else clause
            # TODO: sure this is right?  we used to do this
            #  after raising AttributeError above
            return self.obj._constructor_sliced(
                values, index=key_index, name=self._selection
            )
        elif not isinstance(first_not_none, Series):
            # values are not series or array-like but scalars
            # self._selection not passed through to Series as the
            # result should not take the name of original selection
            # of columns
            if self.as_index:
                return self.obj._constructor_sliced(values, index=key_index)
            else:
                result = self.obj._constructor(values, columns=[self._selection])
                self._insert_inaxis_grouper_inplace(result)
                return result
        else:
            # values are Series
            return self._wrap_applied_output_series(
                values,
                not_indexed_same,
                first_not_none,
                key_index,
                override_group_keys,
            )

    def _wrap_applied_output_series(
        self,
        values: list[Series],
        not_indexed_same: bool,
        first_not_none,
        key_index,
        override_group_keys: bool,
    ) -> DataFrame | Series:
        # this is to silence a DeprecationWarning
        # TODO(2.0): Remove when default dtype of empty Series is object
        kwargs = first_not_none._construct_axes_dict()
        backup = create_series_with_explicit_dtype(dtype_if_empty=object, **kwargs)
        values = [x if (x is not None) else backup for x in values]

        all_indexed_same = all_indexes_same(x.index for x in values)

        # GH3596
        # provide a reduction (Frame -> Series) if groups are
        # unique
        if self.squeeze:
            applied_index = self._selected_obj._get_axis(self.axis)
            singular_series = len(values) == 1 and applied_index.nlevels == 1

            if singular_series:
                # GH2893
                # we have series in the values array, we want to
                # produce a series:
                # if any of the sub-series are not indexed the same
                # OR we don't have a multi-index and we have only a
                # single values
                return self._concat_objects(
                    values,
                    not_indexed_same=not_indexed_same,
                    override_group_keys=override_group_keys,
                )

            # still a series
            # path added as of GH 5545
            elif all_indexed_same:
                from pandas.core.reshape.concat import concat

                return concat(values)

        if not all_indexed_same:
            # GH 8467
            return self._concat_objects(
                values,
                not_indexed_same=True,
                override_group_keys=override_group_keys,
            )

        # Combine values
        # vstack+constructor is faster than concat and handles MI-columns
        stacked_values = np.vstack([np.asarray(v) for v in values])

        if self.axis == 0:
            index = key_index
            columns = first_not_none.index.copy()
            if columns.name is None:
                # GH6124 - propagate name of Series when it's consistent
                names = {v.name for v in values}
                if len(names) == 1:
                    columns.name = list(names)[0]
        else:
            index = first_not_none.index
            columns = key_index
            stacked_values = stacked_values.T

        if stacked_values.dtype == object:
            # We'll have the DataFrame constructor do inference
            stacked_values = stacked_values.tolist()
        result = self.obj._constructor(stacked_values, index=index, columns=columns)

        if not self.as_index:
            self._insert_inaxis_grouper_inplace(result)

        return self._reindex_output(result)

    def _cython_transform(
        self,
        how: str,
        numeric_only: bool | lib.NoDefault = lib.no_default,
        axis: int = 0,
        **kwargs,
    ) -> DataFrame:
        assert axis == 0  # handled by caller
        # TODO: no tests with self.ndim == 1 for DataFrameGroupBy
        numeric_only_bool = self._resolve_numeric_only(how, numeric_only, axis)

        # With self.axis == 0, we have multi-block tests
        #  e.g. test_rank_min_int, test_cython_transform_frame
        #  test_transform_numeric_ret
        # With self.axis == 1, _get_data_to_aggregate does a transpose
        #  so we always have a single block.
        mgr: Manager2D = self._get_data_to_aggregate()
        orig_mgr_len = len(mgr)
        if numeric_only_bool:
            mgr = mgr.get_numeric_data(copy=False)

        def arr_func(bvalues: ArrayLike) -> ArrayLike:
            return self.grouper._cython_operation(
                "transform", bvalues, how, 1, **kwargs
            )

        # We could use `mgr.apply` here and not have to set_axis, but
        #  we would have to do shape gymnastics for ArrayManager compat
        res_mgr = mgr.grouped_reduce(arr_func, ignore_failures=True)
        res_mgr.set_axis(1, mgr.axes[1])

        if len(res_mgr) < orig_mgr_len:
            warn_dropping_nuisance_columns_deprecated(type(self), how, numeric_only)

        res_df = self.obj._constructor(res_mgr)
        if self.axis == 1:
            res_df = res_df.T
        return res_df

    def _transform_general(self, func, *args, **kwargs):
        from pandas.core.reshape.concat import concat

        applied = []
        obj = self._obj_with_exclusions
        gen = self.grouper.get_iterator(obj, axis=self.axis)
        fast_path, slow_path = self._define_paths(func, *args, **kwargs)

        # Determine whether to use slow or fast path by evaluating on the first group.
        # Need to handle the case of an empty generator and process the result so that
        # it does not need to be computed again.
        try:
            name, group = next(gen)
        except StopIteration:
            pass
        else:
            object.__setattr__(group, "name", name)
            try:
                path, res = self._choose_path(fast_path, slow_path, group)
            except TypeError:
                return self._transform_item_by_item(obj, fast_path)
            except ValueError as err:
                msg = "transform must return a scalar value for each group"
                raise ValueError(msg) from err
            if group.size > 0:
                res = _wrap_transform_general_frame(self.obj, group, res)
                applied.append(res)

        # Compute and process with the remaining groups
        emit_alignment_warning = False
        for name, group in gen:
            if group.size == 0:
                continue
            object.__setattr__(group, "name", name)
            res = path(group)
            if (
                not emit_alignment_warning
                and res.ndim == 2
                and not res.index.equals(group.index)
            ):
                emit_alignment_warning = True

            res = _wrap_transform_general_frame(self.obj, group, res)
            applied.append(res)

        if emit_alignment_warning:
            # GH#45648
            warnings.warn(
                "In a future version of pandas, returning a DataFrame in "
                "groupby.transform will align with the input's index. Apply "
                "`.to_numpy()` to the result in the transform function to keep "
                "the current behavior and silence this warning.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )

        concat_index = obj.columns if self.axis == 0 else obj.index
        other_axis = 1 if self.axis == 0 else 0  # switches between 0 & 1
        concatenated = concat(applied, axis=self.axis, verify_integrity=False)
        concatenated = concatenated.reindex(concat_index, axis=other_axis, copy=False)
        return self._set_result_index_ordered(concatenated)

    @Substitution(klass="DataFrame")
    @Appender(_transform_template)
    def transform(self, func, *args, engine=None, engine_kwargs=None, **kwargs):
        return self._transform(
            func, *args, engine=engine, engine_kwargs=engine_kwargs, **kwargs
        )

    def _define_paths(self, func, *args, **kwargs):
        if isinstance(func, str):
            fast_path = lambda group: getattr(group, func)(*args, **kwargs)
            slow_path = lambda group: group.apply(
                lambda x: getattr(x, func)(*args, **kwargs), axis=self.axis
            )
        else:
            fast_path = lambda group: func(group, *args, **kwargs)
            slow_path = lambda group: group.apply(
                lambda x: func(x, *args, **kwargs), axis=self.axis
            )
        return fast_path, slow_path

    def _choose_path(self, fast_path: Callable, slow_path: Callable, group: DataFrame):
        path = slow_path
        res = slow_path(group)

        if self.ngroups == 1:
            # no need to evaluate multiple paths when only
            # a single group exists
            return path, res

        # if we make it here, test if we can use the fast path
        try:
            res_fast = fast_path(group)
        except AssertionError:
            raise  # pragma: no cover
        except Exception:
            # GH#29631 For user-defined function, we can't predict what may be
            #  raised; see test_transform.test_transform_fastpath_raises
            return path, res

        # verify fast path returns either:
        # a DataFrame with columns equal to group.columns
        # OR a Series with index equal to group.columns
        if isinstance(res_fast, DataFrame):
            if not res_fast.columns.equals(group.columns):
                return path, res
        elif isinstance(res_fast, Series):
            if not res_fast.index.equals(group.columns):
                return path, res
        else:
            return path, res

        if res_fast.equals(res):
            path = fast_path

        return path, res

    def _transform_item_by_item(self, obj: DataFrame, wrapper) -> DataFrame:
        # iterate through columns, see test_transform_exclude_nuisance
        #  gets here with non-unique columns
        output = {}
        inds = []
        for i, (colname, sgb) in enumerate(self._iterate_column_groupbys(obj)):
            try:
                output[i] = sgb.transform(wrapper)
            except TypeError:
                # e.g. trying to call nanmean with string values
                warn_dropping_nuisance_columns_deprecated(
                    type(self), "transform", numeric_only=False
                )
            else:
                inds.append(i)

        if not output:
            raise TypeError("Transform function invalid for data types")

        columns = obj.columns.take(inds)

        result = self.obj._constructor(output, index=obj.index)
        result.columns = columns
        return result

    def filter(self, func, dropna=True, *args, **kwargs):
        """
        Return a copy of a DataFrame excluding filtered elements.

        Elements from groups are filtered if they do not satisfy the
        boolean criterion specified by func.

        Parameters
        ----------
        func : function
            Function to apply to each subframe. Should return True or False.
        dropna : Drop groups that do not pass the filter. True by default;
            If False, groups that evaluate False are filled with NaNs.

        Returns
        -------
        filtered : DataFrame

        Notes
        -----
        Each subframe is endowed the attribute 'name' in case you need to know
        which group you are working on.

        Functions that mutate the passed object can produce unexpected
        behavior or errors and are not supported. See :ref:`gotchas.udf-mutation`
        for more details.

        Examples
        --------
        >>> df = pd.DataFrame({'A' : ['foo', 'bar', 'foo', 'bar',
        ...                           'foo', 'bar'],
        ...                    'B' : [1, 2, 3, 4, 5, 6],
        ...                    'C' : [2.0, 5., 8., 1., 2., 9.]})
        >>> grouped = df.groupby('A')
        >>> grouped.filter(lambda x: x['B'].mean() > 3.)
             A  B    C
        1  bar  2  5.0
        3  bar  4  1.0
        5  bar  6  9.0
        """
        indices = []

        obj = self._selected_obj
        gen = self.grouper.get_iterator(obj, axis=self.axis)

        for name, group in gen:
            object.__setattr__(group, "name", name)

            res = func(group, *args, **kwargs)

            try:
                res = res.squeeze()
            except AttributeError:  # allow e.g., scalars and frames to pass
                pass

            # interpret the result of the filter
            if is_bool(res) or (is_scalar(res) and isna(res)):
                if res and notna(res):
                    indices.append(self._get_index(name))
            else:
                # non scalars aren't allowed
                raise TypeError(
                    f"filter function returned a {type(res).__name__}, "
                    "but expected a scalar bool"
                )

        return self._apply_filter(indices, dropna)

    def __getitem__(self, key) -> DataFrameGroupBy | SeriesGroupBy:
        if self.axis == 1:
            # GH 37725
            raise ValueError("Cannot subset columns when using axis=1")
        # per GH 23566
        if isinstance(key, tuple) and len(key) > 1:
            # if len == 1, then it becomes a SeriesGroupBy and this is actually
            # valid syntax, so don't raise warning
            warnings.warn(
                "Indexing with multiple keys (implicitly converted to a tuple "
                "of keys) will be deprecated, use a list instead.",
                FutureWarning,
                stacklevel=find_stack_level(),
            )
        return super().__getitem__(key)

    def _gotitem(self, key, ndim: int, subset=None):
        """
        sub-classes to define
        return a sliced object

        Parameters
        ----------
        key : string / list of selections
        ndim : {1, 2}
            requested ndim of result
        subset : object, default None
            subset to act on
        """
        if ndim == 2:
            if subset is None:
                subset = self.obj
            return DataFrameGroupBy(
                subset,
                self.grouper,
                axis=self.axis,
                level=self.level,
                grouper=self.grouper,
                exclusions=self.exclusions,
                selection=key,
                as_index=self.as_index,
                sort=self.sort,
                group_keys=self.group_keys,
                squeeze=self.squeeze,
                observed=self.observed,
                mutated=self.mutated,
                dropna=self.dropna,
            )
        elif ndim == 1:
            if subset is None:
                subset = self.obj[key]
            return SeriesGroupBy(
                subset,
                level=self.level,
                grouper=self.grouper,
                selection=key,
                sort=self.sort,
                group_keys=self.group_keys,
                squeeze=self.squeeze,
                observed=self.observed,
                dropna=self.dropna,
            )

        raise AssertionError("invalid ndim for _gotitem")

    def _get_data_to_aggregate(self) -> Manager2D:
        obj = self._obj_with_exclusions
        if self.axis == 1:
            return obj.T._mgr
        else:
            return obj._mgr

    def _insert_inaxis_grouper_inplace(self, result: DataFrame) -> None:
        # zip in reverse so we can always insert at loc 0
        columns = result.columns
        for name, lev, in_axis in zip(
            reversed(self.grouper.names),
            reversed(self.grouper.get_group_levels()),
            reversed([grp.in_axis for grp in self.grouper.groupings]),
        ):
            # GH #28549
            # When using .apply(-), name will be in columns already
            if in_axis and name not in columns:
                result.insert(0, name, lev)

    def _indexed_output_to_ndframe(
        self, output: Mapping[base.OutputKey, ArrayLike]
    ) -> DataFrame:
        """
        Wrap the dict result of a GroupBy aggregation into a DataFrame.
        """
        indexed_output = {key.position: val for key, val in output.items()}
        columns = Index([key.label for key in output])
        columns._set_names(self._obj_with_exclusions._get_axis(1 - self.axis).names)

        result = self.obj._constructor(indexed_output)
        result.columns = columns
        return result

    def _wrap_agged_manager(self, mgr: Manager2D) -> DataFrame:
        if not self.as_index:
            # GH 41998 - empty mgr always gets index of length 0
            rows = mgr.shape[1] if mgr.shape[0] > 0 else 0
            index = Index(range(rows))
            mgr.set_axis(1, index)
            result = self.obj._constructor(mgr)

            self._insert_inaxis_grouper_inplace(result)
            result = result._consolidate()
        else:
            index = self.grouper.result_index
            mgr.set_axis(1, index)
            result = self.obj._constructor(mgr)

        if self.axis == 1:
            result = result.T

        # Note: we only need to pass datetime=True in order to get numeric
        #  values converted
        return self._reindex_output(result)._convert(datetime=True)

    def _iterate_column_groupbys(self, obj: DataFrame | Series):
        for i, colname in enumerate(obj.columns):
            yield colname, SeriesGroupBy(
                obj.iloc[:, i],
                selection=colname,
                grouper=self.grouper,
                exclusions=self.exclusions,
                observed=self.observed,
            )

    def _apply_to_column_groupbys(self, func, obj: DataFrame | Series) -> DataFrame:
        from pandas.core.reshape.concat import concat

        columns = obj.columns
        results = [
            func(col_groupby) for _, col_groupby in self._iterate_column_groupbys(obj)
        ]

        if not len(results):
            # concat would raise
            return DataFrame([], columns=columns, index=self.grouper.result_index)
        else:
            return concat(results, keys=columns, axis=1)

    def nunique(self, dropna: bool = True) -> DataFrame:
        """
        Return DataFrame with counts of unique elements in each position.

        Parameters
        ----------
        dropna : bool, default True
            Don't include NaN in the counts.

        Returns
        -------
        nunique: DataFrame

        Examples
        --------
        >>> df = pd.DataFrame({'id': ['spam', 'egg', 'egg', 'spam',
        ...                           'ham', 'ham'],
        ...                    'value1': [1, 5, 5, 2, 5, 5],
        ...                    'value2': list('abbaxy')})
        >>> df
             id  value1 value2
        0  spam       1      a
        1   egg       5      b
        2   egg       5      b
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y

        >>> df.groupby('id').nunique()
              value1  value2
        id
        egg        1       1
        ham        1       2
        spam       2       1

        Check for rows with the same id but conflicting values:

        >>> df.groupby('id').filter(lambda g: (g.nunique() > 1).any())
             id  value1 value2
        0  spam       1      a
        3  spam       2      a
        4   ham       5      x
        5   ham       5      y
        """

        if self.axis != 0:
            # see test_groupby_crash_on_nunique
            return self._python_agg_general(lambda sgb: sgb.nunique(dropna))

        obj = self._obj_with_exclusions
        results = self._apply_to_column_groupbys(
            lambda sgb: sgb.nunique(dropna), obj=obj
        )

        if not self.as_index:
            results.index = Index(range(len(results)))
            self._insert_inaxis_grouper_inplace(results)

        return results

    @doc(
        _shared_docs["idxmax"],
        numeric_only_default="True for axis=0, False for axis=1",
    )
    def idxmax(
        self,
        axis=0,
        skipna: bool = True,
        numeric_only: bool | lib.NoDefault = lib.no_default,
    ) -> DataFrame:
        axis = DataFrame._get_axis_number(axis)
        if numeric_only is lib.no_default:
            # Cannot use self._resolve_numeric_only; we must pass None to
            # DataFrame.idxmax for backwards compatibility
            numeric_only_arg = None if axis == 0 else False
        else:
            numeric_only_arg = numeric_only

        def func(df):
            with warnings.catch_warnings():
                # Suppress numeric_only warnings here, will warn below
                warnings.filterwarnings("ignore", ".*numeric_only in DataFrame.argmax")
                res = df._reduce(
                    nanops.nanargmax,
                    "argmax",
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only_arg,
                )
                indices = res._values
                index = df._get_axis(axis)
                result = [index[i] if i >= 0 else np.nan for i in indices]
                return df._constructor_sliced(result, index=res.index)

        func.__name__ = "idxmax"
        result = self._python_apply_general(
            func, self._obj_with_exclusions, not_indexed_same=True
        )
        self._maybe_warn_numeric_only_depr("idxmax", result, numeric_only)
        return result

    @doc(
        _shared_docs["idxmin"],
        numeric_only_default="True for axis=0, False for axis=1",
    )
    def idxmin(
        self,
        axis=0,
        skipna: bool = True,
        numeric_only: bool | lib.NoDefault = lib.no_default,
    ) -> DataFrame:
        axis = DataFrame._get_axis_number(axis)
        if numeric_only is lib.no_default:
            # Cannot use self._resolve_numeric_only; we must pass None to
            # DataFrame.idxmin for backwards compatibility
            numeric_only_arg = None if axis == 0 else False
        else:
            numeric_only_arg = numeric_only

        def func(df):
            with warnings.catch_warnings():
                # Suppress numeric_only warnings here, will warn below
                warnings.filterwarnings("ignore", ".*numeric_only in DataFrame.argmin")
                res = df._reduce(
                    nanops.nanargmin,
                    "argmin",
                    axis=axis,
                    skipna=skipna,
                    numeric_only=numeric_only_arg,
                )
                indices = res._values
                index = df._get_axis(axis)
                result = [index[i] if i >= 0 else np.nan for i in indices]
                return df._constructor_sliced(result, index=res.index)

        func.__name__ = "idxmin"
        result = self._python_apply_general(
            func, self._obj_with_exclusions, not_indexed_same=True
        )
        self._maybe_warn_numeric_only_depr("idxmin", result, numeric_only)
        return result

    boxplot = boxplot_frame_groupby

    def value_counts(
        self,
        subset: Sequence[Hashable] | None = None,
        normalize: bool = False,
        sort: bool = True,
        ascending: bool = False,
        dropna: bool = True,
    ) -> DataFrame | Series:
        """
        Return a Series or DataFrame containing counts of unique rows.

        .. versionadded:: 1.4.0

        Parameters
        ----------
        subset : list-like, optional
            Columns to use when counting unique combinations.
        normalize : bool, default False
            Return proportions rather than frequencies.
        sort : bool, default True
            Sort by frequencies.
        ascending : bool, default False
            Sort in ascending order.
        dropna : bool, default True
            Don’t include counts of rows that contain NA values.

        Returns
        -------
        Series or DataFrame
            Series if the groupby as_index is True, otherwise DataFrame.

        See Also
        --------
        Series.value_counts: Equivalent method on Series.
        DataFrame.value_counts: Equivalent method on DataFrame.
        SeriesGroupBy.value_counts: Equivalent method on SeriesGroupBy.

        Notes
        -----
        - If the groupby as_index is True then the returned Series will have a
          MultiIndex with one level per input column.
        - If the groupby as_index is False then the returned DataFrame will have an
          additional column with the value_counts. The column is labelled 'count' or
          'proportion', depending on the ``normalize`` parameter.

        By default, rows that contain any NA values are omitted from
        the result.

        By default, the result will be in descending order so that the
        first element of each group is the most frequently-occurring row.

        Examples
        --------
        >>> df = pd.DataFrame({
        ...    'gender': ['male', 'male', 'female', 'male', 'female', 'male'],
        ...    'education': ['low', 'medium', 'high', 'low', 'high', 'low'],
        ...    'country': ['US', 'FR', 'US', 'FR', 'FR', 'FR']
        ... })

        >>> df
                gender  education   country
        0       male    low         US
        1       male    medium      FR
        2       female  high        US
        3       male    low         FR
        4       female  high        FR
        5       male    low         FR

        >>> df.groupby('gender').value_counts()
        gender  education  country
        female  high       FR         1
                           US         1
        male    low        FR         2
                           US         1
                medium     FR         1
        dtype: int64

        >>> df.groupby('gender').value_counts(ascending=True)
        gender  education  country
        female  high       FR         1
                           US         1
        male    low        US         1
                medium     FR         1
                low        FR         2
        dtype: int64

        >>> df.groupby('gender').value_counts(normalize=True)
        gender  education  country
        female  high       FR         0.50
                           US         0.50
        male    low        FR         0.50
                           US         0.25
                medium     FR         0.25
        dtype: float64

        >>> df.groupby('gender', as_index=False).value_counts()
           gender education country  count
        0  female      high      FR      1
        1  female      high      US      1
        2    male       low      FR      2
        3    male       low      US      1
        4    male    medium      FR      1

        >>> df.groupby('gender', as_index=False).value_counts(normalize=True)
           gender education country  proportion
        0  female      high      FR        0.50
        1  female      high      US        0.50
        2    male       low      FR        0.50
        3    male       low      US        0.25
        4    male    medium      FR        0.25
        """
        if self.axis == 1:
            raise NotImplementedError(
                "DataFrameGroupBy.value_counts only handles axis=0"
            )

        with self._group_selection_context():
            df = self.obj

            in_axis_names = {
                grouping.name for grouping in self.grouper.groupings if grouping.in_axis
            }
            if isinstance(self._selected_obj, Series):
                name = self._selected_obj.name
                keys = [] if name in in_axis_names else [self._selected_obj]
            else:
                unique_cols = set(self._selected_obj.columns)
                if subset is not None:
                    subsetted = set(subset)
                    clashing = subsetted & set(in_axis_names)
                    if clashing:
                        raise ValueError(
                            f"Keys {clashing} in subset cannot be in "
                            "the groupby column keys."
                        )
                    doesnt_exist = subsetted - unique_cols
                    if doesnt_exist:
                        raise ValueError(
                            f"Keys {doesnt_exist} in subset do not "
                            f"exist in the DataFrame."
                        )
                else:
                    subsetted = unique_cols

                keys = [
                    # Can't use .values because the column label needs to be preserved
                    self._selected_obj.iloc[:, idx]
                    for idx, name in enumerate(self._selected_obj.columns)
                    if name not in in_axis_names and name in subsetted
                ]

            groupings = list(self.grouper.groupings)
            for key in keys:
                grouper, _, _ = get_grouper(
                    df,
                    key=key,
                    axis=self.axis,
                    sort=self.sort,
                    observed=False,
                    dropna=dropna,
                )
                groupings += list(grouper.groupings)

            # Take the size of the overall columns
            gb = df.groupby(
                groupings,
                sort=self.sort,
                observed=self.observed,
                dropna=self.dropna,
            )
            result_series = cast(Series, gb.size())

            # GH-46357 Include non-observed categories
            # of non-grouping columns regardless of `observed`
            if any(
                isinstance(grouping.grouping_vector, (Categorical, CategoricalIndex))
                and not grouping._observed
                for grouping in groupings
            ):
                levels_list = [ping.result_index for ping in groupings]
                multi_index, _ = MultiIndex.from_product(
                    levels_list, names=[ping.name for ping in groupings]
                ).sortlevel()
                result_series = result_series.reindex(multi_index, fill_value=0)

            if normalize:
                # Normalize the results by dividing by the original group sizes.
                # We are guaranteed to have the first N levels be the
                # user-requested grouping.
                levels = list(
                    range(len(self.grouper.groupings), result_series.index.nlevels)
                )
                indexed_group_size = result_series.groupby(
                    result_series.index.droplevel(levels),
                    sort=self.sort,
                    dropna=self.dropna,
                ).transform("sum")
                result_series /= indexed_group_size

                # Handle groups of non-observed categories
                result_series = result_series.fillna(0.0)

            if sort:
                # Sort the values and then resort by the main grouping
                index_level = range(len(self.grouper.groupings))
                result_series = result_series.sort_values(
                    ascending=ascending
                ).sort_index(level=index_level, sort_remaining=False)

            result: Series | DataFrame
            if self.as_index:
                result = result_series
            else:
                # Convert to frame
                name = "proportion" if normalize else "count"
                index = result_series.index
                columns = com.fill_missing_names(index.names)
                if name in columns:
                    raise ValueError(
                        f"Column label '{name}' is duplicate of result column"
                    )
                result_series.name = name
                result_series.index = index.set_names(range(len(columns)))
                result_frame = result_series.reset_index()
                result_frame.columns = columns + [name]
                result = result_frame
            return result.__finalize__(self.obj, method="value_counts")


def _wrap_transform_general_frame(
    obj: DataFrame, group: DataFrame, res: DataFrame | Series
) -> DataFrame:
    from pandas import concat

    if isinstance(res, Series):
        # we need to broadcast across the
        # other dimension; this will preserve dtypes
        # GH14457
        if res.index.is_(obj.index):
            res_frame = concat([res] * len(group.columns), axis=1)
            res_frame.columns = group.columns
            res_frame.index = group.index
        else:
            res_frame = obj._constructor(
                np.tile(res.values, (len(group.index), 1)),
                columns=group.columns,
                index=group.index,
            )
        assert isinstance(res_frame, DataFrame)
        return res_frame
    else:
        return res
