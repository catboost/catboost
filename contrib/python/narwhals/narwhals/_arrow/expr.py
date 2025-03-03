from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from narwhals._arrow.expr_cat import ArrowExprCatNamespace
from narwhals._arrow.expr_dt import ArrowExprDateTimeNamespace
from narwhals._arrow.expr_list import ArrowExprListNamespace
from narwhals._arrow.expr_name import ArrowExprNameNamespace
from narwhals._arrow.expr_str import ArrowExprStringNamespace
from narwhals._arrow.series import ArrowSeries
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._expression_parsing import reuse_series_implementation
from narwhals.dependencies import get_numpy
from narwhals.dependencies import is_numpy_array
from narwhals.exceptions import ColumnNotFoundError
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.namespace import ArrowNamespace
    from narwhals._arrow.typing import IntoArrowExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class ArrowExpr(CompliantExpr[ArrowSeries]):
    _implementation: Implementation = Implementation.PYARROW

    def __init__(
        self: Self,
        call: Callable[[ArrowDataFrame], Sequence[ArrowSeries]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[ArrowDataFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._depth = depth
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._backend_version = backend_version
        self._version = version
        self._kwargs = kwargs

    def __repr__(self: Self) -> str:  # pragma: no cover
        return f"ArrowExpr(depth={self._depth}, function_name={self._function_name}, "

    def __call__(self: Self, df: ArrowDataFrame) -> Sequence[ArrowSeries]:
        return self._call(df)

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        from narwhals._arrow.series import ArrowSeries

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            try:
                return [
                    ArrowSeries(
                        df._native_frame[column_name],
                        name=column_name,
                        backend_version=df._backend_version,
                        version=df._version,
                    )
                    for column_name in column_names
                ]
            except KeyError as e:
                missing_columns = [x for x in column_names if x not in df.columns]
                raise ColumnNotFoundError.from_missing_and_available_column_names(
                    missing_columns=missing_columns, available_columns=df.columns
                ) from e

        return cls(
            func,
            depth=0,
            function_name="col",
            evaluate_output_names=lambda _df: list(column_names),
            alias_output_names=None,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        from narwhals._arrow.series import ArrowSeries

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            return [
                ArrowSeries(
                    df._native_frame[column_index],
                    name=df._native_frame.column_names[column_index],
                    backend_version=df._backend_version,
                    version=df._version,
                )
                for column_index in column_indices
            ]

        return cls(
            func,
            depth=0,
            function_name="nth",
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    def __narwhals_namespace__(self: Self) -> ArrowNamespace:
        from narwhals._arrow.namespace import ArrowNamespace

        return ArrowNamespace(
            backend_version=self._backend_version, version=self._version
        )

    def __narwhals_expr__(self: Self) -> None: ...

    def __eq__(self: Self, other: ArrowExpr | Any) -> Self:  # type: ignore[override]
        return reuse_series_implementation(self, "__eq__", other=other)

    def __ne__(self: Self, other: ArrowExpr | Any) -> Self:  # type: ignore[override]
        return reuse_series_implementation(self, "__ne__", other=other)

    def __ge__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__ge__", other=other)

    def __gt__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__gt__", other=other)

    def __le__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__le__", other=other)

    def __lt__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__lt__", other=other)

    def __and__(self: Self, other: ArrowExpr | bool | Any) -> Self:
        return reuse_series_implementation(self, "__and__", other=other)

    def __or__(self: Self, other: ArrowExpr | bool | Any) -> Self:
        return reuse_series_implementation(self, "__or__", other=other)

    def __add__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__add__", other=other)

    def __sub__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__sub__", other=other)

    def __mul__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__mul__", other=other)

    def __pow__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__pow__", other=other)

    def __floordiv__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__floordiv__", other=other)

    def __truediv__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__truediv__", other=other)

    def __mod__(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "__mod__", other=other)

    def __invert__(self: Self) -> Self:
        return reuse_series_implementation(self, "__invert__")

    def len(self: Self) -> Self:
        return reuse_series_implementation(self, "len", returns_scalar=True)

    def filter(self: Self, *predicates: IntoArrowExpr) -> Self:
        plx = self.__narwhals_namespace__()
        other = plx.all_horizontal(*predicates)
        return reuse_series_implementation(self, "filter", other=other)

    def mean(self: Self) -> Self:
        return reuse_series_implementation(self, "mean", returns_scalar=True)

    def median(self: Self) -> Self:
        return reuse_series_implementation(self, "median", returns_scalar=True)

    def count(self: Self) -> Self:
        return reuse_series_implementation(self, "count", returns_scalar=True)

    def n_unique(self: Self) -> Self:
        return reuse_series_implementation(self, "n_unique", returns_scalar=True)

    def std(self: Self, ddof: int) -> Self:
        return reuse_series_implementation(self, "std", ddof=ddof, returns_scalar=True)

    def var(self: Self, ddof: int) -> Self:
        return reuse_series_implementation(self, "var", ddof=ddof, returns_scalar=True)

    def skew(self: Self) -> Self:
        return reuse_series_implementation(self, "skew", returns_scalar=True)

    def cast(self: Self, dtype: DType) -> Self:
        return reuse_series_implementation(self, "cast", dtype=dtype)

    def abs(self: Self) -> Self:
        return reuse_series_implementation(self, "abs")

    def diff(self: Self) -> Self:
        return reuse_series_implementation(self, "diff")

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_sum", reverse=reverse)

    def round(self: Self, decimals: int) -> Self:
        return reuse_series_implementation(self, "round", decimals=decimals)

    def any(self: Self) -> Self:
        return reuse_series_implementation(self, "any", returns_scalar=True)

    def min(self: Self) -> Self:
        return reuse_series_implementation(self, "min", returns_scalar=True)

    def max(self: Self) -> Self:
        return reuse_series_implementation(self, "max", returns_scalar=True)

    def arg_min(self: Self) -> Self:
        return reuse_series_implementation(self, "arg_min", returns_scalar=True)

    def arg_max(self: Self) -> Self:
        return reuse_series_implementation(self, "arg_max", returns_scalar=True)

    def all(self: Self) -> Self:
        return reuse_series_implementation(self, "all", returns_scalar=True)

    def sum(self: Self) -> Self:
        return reuse_series_implementation(self, "sum", returns_scalar=True)

    def drop_nulls(self: Self) -> Self:
        return reuse_series_implementation(self, "drop_nulls")

    def shift(self: Self, n: int) -> Self:
        return reuse_series_implementation(self, "shift", n=n)

    def alias(self: Self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        # Define this one manually, so that we can
        # override `output_names` and not increase depth
        return self.__class__(
            lambda df: [series.alias(name) for series in self._call(df)],
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, "name": name},
        )

    def null_count(self: Self) -> Self:
        return reuse_series_implementation(self, "null_count", returns_scalar=True)

    def is_null(self: Self) -> Self:
        return reuse_series_implementation(self, "is_null")

    def is_nan(self: Self) -> Self:
        return reuse_series_implementation(self, "is_nan")

    def head(self: Self, n: int) -> Self:
        return reuse_series_implementation(self, "head", n=n)

    def tail(self: Self, n: int) -> Self:
        return reuse_series_implementation(self, "tail", n=n)

    def is_in(self: Self, other: ArrowExpr | Any) -> Self:
        return reuse_series_implementation(self, "is_in", other=other)

    def arg_true(self: Self) -> Self:
        return reuse_series_implementation(self, "arg_true")

    def sample(
        self: Self,
        n: int | None,
        *,
        fraction: float | None,
        with_replacement: bool,
        seed: int | None,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "sample",
            n=n,
            fraction=fraction,
            with_replacement=with_replacement,
            seed=seed,
        )

    def fill_null(
        self: Self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> Self:
        return reuse_series_implementation(
            self, "fill_null", value=value, strategy=strategy, limit=limit
        )

    def is_unique(self: Self) -> Self:
        return reuse_series_implementation(self, "is_unique")

    def is_first_distinct(self: Self) -> Self:
        return reuse_series_implementation(self, "is_first_distinct")

    def is_last_distinct(self: Self) -> Self:
        return reuse_series_implementation(self, "is_last_distinct")

    def unique(self: Self) -> Self:
        return reuse_series_implementation(self, "unique", maintain_order=False)

    def replace_strict(
        self: Self, old: Sequence[Any], new: Sequence[Any], *, return_dtype: DType | None
    ) -> Self:
        return reuse_series_implementation(
            self, "replace_strict", old=old, new=new, return_dtype=return_dtype
        )

    def sort(self: Self, *, descending: bool, nulls_last: bool) -> Self:
        return reuse_series_implementation(
            self, "sort", descending=descending, nulls_last=nulls_last
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        return reuse_series_implementation(
            self,
            "quantile",
            returns_scalar=True,
            quantile=quantile,
            interpolation=interpolation,
        )

    def gather_every(self: Self, n: int, offset: int) -> Self:
        return reuse_series_implementation(self, "gather_every", n=n, offset=offset)

    def clip(self: Self, lower_bound: Any | None, upper_bound: Any | None) -> Self:
        return reuse_series_implementation(
            self, "clip", lower_bound=lower_bound, upper_bound=upper_bound
        )

    def over(self: Self, keys: list[str]) -> Self:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            output_names, aliases = evaluate_output_names_and_aliases(self, df, [])
            if overlap := set(output_names).intersection(keys):
                # E.g. `df.select(nw.all().sum().over('a'))`. This is well-defined,
                # we just don't support it yet.
                msg = (
                    f"Column names {overlap} appear in both expression output names and in `over` keys.\n"
                    "This is not yet supported."
                )
                raise NotImplementedError(msg)

            tmp = df.group_by(*keys, drop_null_keys=False).agg(self)
            tmp = df.simple_select(*keys).join(
                tmp, how="left", left_on=keys, right_on=keys, suffix="_right"
            )
            return [tmp[alias] for alias in aliases]

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->over",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, "keys": keys},
        )

    def mode(self: Self) -> Self:
        return reuse_series_implementation(self, "mode")

    def map_batches(
        self: Self,
        function: Callable[[Any], Any],
        return_dtype: DType | None,
    ) -> Self:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            input_series_list = self._call(df)
            output_names = [input_series.name for input_series in input_series_list]
            result = [function(series) for series in input_series_list]

            if is_numpy_array(result[0]):
                result = [
                    df.__narwhals_namespace__()
                    ._create_compliant_series(array)
                    .alias(output_name)
                    for array, output_name in zip(result, output_names)
                ]
            elif (np := get_numpy()) is not None and np.isscalar(result[0]):
                result = [
                    df.__narwhals_namespace__()
                    ._create_compliant_series([array])
                    .alias(output_name)
                    for array, output_name in zip(result, output_names)
                ]
            if return_dtype is not None:
                result = [series.cast(return_dtype) for series in result]
            return result

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->map_batches",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, "function": function, "return_dtype": return_dtype},
        )

    def is_finite(self: Self) -> Self:
        return reuse_series_implementation(self, "is_finite")

    def cum_count(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_count", reverse=reverse)

    def cum_min(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_min", reverse=reverse)

    def cum_max(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_max", reverse=reverse)

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        return reuse_series_implementation(self, "cum_prod", reverse=reverse)

    def rolling_sum(
        self: Self,
        window_size: int,
        *,
        min_samples: int | None,
        center: bool,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "rolling_sum",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
        )

    def rolling_mean(
        self: Self,
        window_size: int,
        *,
        min_samples: int | None,
        center: bool,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "rolling_mean",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
        )

    def rolling_var(
        self: Self,
        window_size: int,
        *,
        min_samples: int | None,
        center: bool,
        ddof: int,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "rolling_var",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            ddof=ddof,
        )

    def rolling_std(
        self: Self,
        window_size: int,
        *,
        min_samples: int | None,
        center: bool,
        ddof: int,
    ) -> Self:
        return reuse_series_implementation(
            self,
            "rolling_std",
            window_size=window_size,
            min_samples=min_samples,
            center=center,
            ddof=ddof,
        )

    def rank(
        self: Self,
        method: Literal["average", "min", "max", "dense", "ordinal"],
        *,
        descending: bool,
    ) -> Self:
        return reuse_series_implementation(
            self, "rank", method=method, descending=descending
        )

    @property
    def dt(self: Self) -> ArrowExprDateTimeNamespace:
        return ArrowExprDateTimeNamespace(self)

    @property
    def str(self: Self) -> ArrowExprStringNamespace:
        return ArrowExprStringNamespace(self)

    @property
    def cat(self: Self) -> ArrowExprCatNamespace:
        return ArrowExprCatNamespace(self)

    @property
    def name(self: Self) -> ArrowExprNameNamespace:
        return ArrowExprNameNamespace(self)

    @property
    def list(self: Self) -> ArrowExprListNamespace:
        return ArrowExprListNamespace(self)
