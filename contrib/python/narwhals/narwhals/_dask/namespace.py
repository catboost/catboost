from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Sequence
from typing import cast

import dask.dataframe as dd
import pandas as pd

from narwhals._dask.dataframe import DaskLazyFrame
from narwhals._dask.expr import DaskExpr
from narwhals._dask.selectors import DaskSelectorNamespace
from narwhals._dask.utils import name_preserving_div
from narwhals._dask.utils import name_preserving_sum
from narwhals._dask.utils import narwhals_to_native_dtype
from narwhals._dask.utils import validate_comparand
from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    try:
        import dask.dataframe.dask_expr as dx
    except ModuleNotFoundError:
        import dask_expr as dx

    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.utils import Version


class DaskNamespace(CompliantNamespace["dx.Series"]):
    @property
    def selectors(self: Self) -> DaskSelectorNamespace:
        return DaskSelectorNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._version = version

    def all(self: Self) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            return [df._native_frame[column_name] for column_name in df.columns]

        return DaskExpr(
            func,
            depth=0,
            function_name="all",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def col(self: Self, *column_names: str) -> DaskExpr:
        return DaskExpr.from_column_names(
            *column_names, backend_version=self._backend_version, version=self._version
        )

    def nth(self: Self, *column_indices: int) -> DaskExpr:
        return DaskExpr.from_column_indices(
            *column_indices, backend_version=self._backend_version, version=self._version
        )

    def lit(self: Self, value: Any, dtype: DType | None) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            return [
                dd.from_pandas(
                    pd.Series(
                        [value],
                        dtype=narwhals_to_native_dtype(dtype, self._version)
                        if dtype is not None
                        else None,
                        name="literal",
                    ),
                    npartitions=df._native_frame.npartitions,
                )
            ]

        return DaskExpr(
            func,
            depth=0,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def len(self: Self) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            if not df.columns:
                return [
                    dd.from_pandas(
                        pd.Series([0], name="len"),
                        npartitions=df._native_frame.npartitions,
                    )
                ]
            return [df._native_frame[df.columns[0]].size.to_series()]

        # coverage bug? this is definitely hit
        return DaskExpr(  # pragma: no cover
            func,
            depth=0,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            returns_scalar=True,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def all_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = [s for _expr in exprs for s in _expr(df)]
            return [reduce(operator.and_, series)]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def any_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = [s for _expr in exprs for s in _expr(df)]
            return [reduce(operator.or_, series)]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="any_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def sum_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = [s.fillna(0) for _expr in exprs for s in _expr(df)]
            return [reduce(operator.add, series)]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def concat(
        self: Self,
        items: Iterable[DaskLazyFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> DaskLazyFrame:
        if not items:
            msg = "No items to concatenate"  # pragma: no cover
            raise AssertionError(msg)
        dfs = [i._native_frame for i in items]
        cols_0 = dfs[0].columns
        if how == "vertical":
            for i, df in enumerate(dfs[1:], start=1):
                cols_current = df.columns
                if not (
                    (len(cols_current) == len(cols_0)) and (cols_current == cols_0).all()
                ):
                    msg = (
                        "unable to vstack, column names don't match:\n"
                        f"   - dataframe 0: {cols_0.to_list()}\n"
                        f"   - dataframe {i}: {cols_current.to_list()}\n"
                    )
                    raise TypeError(msg)
            return DaskLazyFrame(
                dd.concat(dfs, axis=0, join="inner"),
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )
        if how == "horizontal":
            all_column_names: list[str] = [
                column for frame in dfs for column in frame.columns
            ]
            if len(all_column_names) != len(set(all_column_names)):  # pragma: no cover
                duplicates = [
                    i for i in all_column_names if all_column_names.count(i) > 1
                ]
                msg = (
                    f"Columns with name(s): {', '.join(duplicates)} "
                    "have more than one occurrence"
                )
                raise AssertionError(msg)
            return DaskLazyFrame(
                dd.concat(dfs, axis=1, join="outer"),
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )
        if how == "diagonal":
            return DaskLazyFrame(
                dd.concat(dfs, axis=0, join="outer"),
                backend_version=self._backend_version,
                version=self._version,
                validate_column_names=True,
            )

        raise NotImplementedError

    def mean_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = (s.fillna(0) for _expr in exprs for s in _expr(df))
            non_na = (1 - s.isna() for _expr in exprs for s in _expr(df))
            return [
                name_preserving_div(
                    reduce(name_preserving_sum, series),
                    reduce(name_preserving_sum, non_na),
                )
            ]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def min_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = [s for _expr in exprs for s in _expr(df)]

            return [dd.concat(series, axis=1).min(axis=1)]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def max_horizontal(self: Self, *exprs: DaskExpr) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = [s for _expr in exprs for s in _expr(df)]

            return [dd.concat(series, axis=1).max(axis=1)]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"exprs": exprs},
        )

    def when(
        self: Self,
        *predicates: DaskExpr,
    ) -> DaskWhen:
        plx = self.__class__(backend_version=self._backend_version, version=self._version)
        condition = plx.all_horizontal(*predicates)
        return DaskWhen(
            condition, self._backend_version, returns_scalar=False, version=self._version
        )

    def concat_str(
        self: Self,
        *exprs: DaskExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> DaskExpr:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            series = (s.astype(str) for _expr in exprs for s in _expr(df))
            null_mask = [s for _expr in exprs for s in _expr.is_null()(df)]

            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, null_mask)
                result = reduce(lambda x, y: x + separator + y, series).where(
                    ~null_mask_result, None
                )
            else:
                init_value, *values = [
                    s.where(~nm, "") for s, nm in zip(series, null_mask)
                ]

                separators = (
                    nm.map({True: "", False: separator}, meta=str)
                    for nm in null_mask[:-1]
                )
                result = reduce(
                    operator.add,
                    (s + v for s, v in zip(separators, values)),
                    init_value,
                )

            return [result]

        return DaskExpr(
            call=func,
            depth=max(x._depth for x in exprs) + 1,
            function_name="concat_str",
            evaluate_output_names=getattr(
                exprs[0], "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(exprs[0], "_alias_output_names", None),
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={
                "exprs": exprs,
                "separator": separator,
                "ignore_nulls": ignore_nulls,
            },
        )


class DaskWhen:
    def __init__(
        self: Self,
        condition: DaskExpr,
        backend_version: tuple[int, ...],
        then_value: Any = None,
        otherwise_value: Any = None,
        *,
        returns_scalar: bool,
        version: Version,
    ) -> None:
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value
        self._returns_scalar = returns_scalar
        self._version = version

    def __call__(self: Self, df: DaskLazyFrame) -> Sequence[dx.Series]:
        condition = self._condition(df)[0]
        condition = cast("dx.Series", condition)

        if isinstance(self._then_value, DaskExpr):
            is_scalar = self._then_value._returns_scalar
            value_sequence: Sequence[Any] = self._then_value(df)[0]
        else:
            # `self._then_value` is a scalar
            value_sequence = [self._then_value]
            is_scalar = True

        if is_scalar:
            _df = condition.to_frame("a")
            _df["literal"] = value_sequence[0]
            value_series = _df["literal"]
        else:
            value_series = value_sequence

        value_series = cast("dx.Series", value_series)
        validate_comparand(condition, value_series)

        if self._otherwise_value is None:
            return [value_series.where(condition)]
        if not isinstance(self._otherwise_value, DaskExpr):
            # `self._otherwise_value` is a scalar
            return [value_series.where(condition, self._otherwise_value)]

        otherwise_expr = self._otherwise_value
        otherwise_series = otherwise_expr(df)[0]

        if otherwise_expr._returns_scalar:
            return [value_series.where(condition, otherwise_series[0])]

        validate_comparand(condition, otherwise_series)
        return [value_series.where(condition, otherwise_series)]

    def then(self: Self, value: DaskExpr | Any) -> DaskThen:
        self._then_value = value

        return DaskThen(
            self,
            depth=0,
            function_name="whenthen",
            evaluate_output_names=getattr(
                value, "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(value, "_alias_output_names", None),
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"value": value},
        )


class DaskThen(DaskExpr):
    def __init__(
        self: Self,
        call: DaskWhen,
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[DaskLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._returns_scalar = returns_scalar
        self._kwargs = kwargs

    def otherwise(self: Self, value: DaskExpr | Any) -> DaskExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `DaskWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
