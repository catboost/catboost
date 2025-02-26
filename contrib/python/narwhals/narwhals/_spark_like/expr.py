from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Sequence

from narwhals._spark_like.expr_dt import SparkLikeExprDateTimeNamespace
from narwhals._spark_like.expr_list import SparkLikeExprListNamespace
from narwhals._spark_like.expr_name import SparkLikeExprNameNamespace
from narwhals._spark_like.expr_str import SparkLikeExprStringNamespace
from narwhals._spark_like.utils import ExprKind
from narwhals._spark_like.utils import maybe_evaluate
from narwhals._spark_like.utils import n_ary_operation_expr_kind
from narwhals._spark_like.utils import narwhals_to_native_dtype
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation
from narwhals.utils import parse_version

if TYPE_CHECKING:
    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class SparkLikeExpr(CompliantExpr["Column"]):
    _depth = 0  # Unused, just for compatibility with CompliantExpr

    def __init__(
        self: Self,
        call: Callable[[SparkLikeLazyFrame], list[Column]],
        *,
        function_name: str,
        evaluate_output_names: Callable[[SparkLikeLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        expr_kind: ExprKind,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> None:
        self._call = call
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._expr_kind = expr_kind
        self._backend_version = backend_version
        self._version = version
        self._implementation = implementation

    def __call__(self: Self, df: SparkLikeLazyFrame) -> Sequence[Column]:
        return self._call(df)

    @property
    def _F(self) -> Any:  # noqa: N802
        if self._implementation is Implementation.SQLFRAME:
            from sqlframe.duckdb import functions

            return functions
        from pyspark.sql import functions

        return functions

    @property
    def _native_types(self) -> Any:
        if self._implementation is Implementation.SQLFRAME:
            from sqlframe.duckdb import types

            return types
        from pyspark.sql import types

        return types

    @property
    def _Window(self) -> Any:  # noqa: N802
        if self._implementation is Implementation.SQLFRAME:
            from sqlframe.duckdb import Window

            return Window
        from pyspark.sql import Window

        return Window

    def __narwhals_expr__(self: Self) -> None: ...

    def __narwhals_namespace__(self: Self) -> SparkLikeNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._spark_like.namespace import SparkLikeNamespace

        return SparkLikeNamespace(
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [df._F.col(col_name) for col_name in column_names]

        return cls(
            func,
            function_name="col",
            evaluate_output_names=lambda _df: list(column_names),
            alias_output_names=None,
            expr_kind=ExprKind.TRANSFORM,
            backend_version=backend_version,
            version=version,
            implementation=implementation,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            columns = df.columns
            return [df._F.col(columns[i]) for i in column_indices]

        return cls(
            func,
            function_name="nth",
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            expr_kind=ExprKind.TRANSFORM,
            backend_version=backend_version,
            version=version,
            implementation=implementation,
        )

    def _from_call(
        self: Self,
        call: Callable[..., Column],
        expr_name: str,
        *,
        expr_kind: ExprKind,
        **expressifiable_args: Self | Any,
    ) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            native_series_list = self._call(df)
            other_native_series = {
                key: maybe_evaluate(df, value, expr_kind=expr_kind)
                for key, value in expressifiable_args.items()
            }
            return [
                call(native_series, **other_native_series)
                for native_series in native_series_list
            ]

        return self.__class__(
            func,
            function_name=f"{self._function_name}->{expr_name}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            expr_kind=expr_kind,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def __eq__(self: Self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__eq__(other),
            "__eq__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __ne__(self: Self, other: SparkLikeExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__ne__(other),
            "__ne__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __add__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__add__(other),
            "__add__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __sub__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__sub__(other),
            "__sub__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __mul__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mul__(other),
            "__mul__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __truediv__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__truediv__(other),
            "__truediv__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __floordiv__(self: Self, other: SparkLikeExpr) -> Self:
        def _floordiv(_input: Column, other: Column) -> Column:
            return self._F.floor(_input / other)

        return self._from_call(
            _floordiv,
            "__floordiv__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __pow__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__pow__(other),
            "__pow__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __mod__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mod__(other),
            "__mod__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __ge__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__ge__(other),
            "__ge__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __gt__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input > other,
            "__gt__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __le__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__le__(other),
            "__le__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __lt__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__lt__(other),
            "__lt__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __and__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__and__(other),
            "__and__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __or__(self: Self, other: SparkLikeExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__or__(other),
            "__or__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __invert__(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.__invert__(),
            "__invert__",
            expr_kind=self._expr_kind,
        )

    def abs(self: Self) -> Self:
        return self._from_call(self._F.abs, "abs", expr_kind=self._expr_kind)

    def alias(self: Self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        return self.__class__(
            self._call,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            expr_kind=self._expr_kind,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def all(self: Self) -> Self:
        return self._from_call(self._F.bool_and, "all", expr_kind=ExprKind.AGGREGATION)

    def any(self: Self) -> Self:
        return self._from_call(self._F.bool_or, "any", expr_kind=ExprKind.AGGREGATION)

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def _cast(_input: Column) -> Column:
            spark_dtype = narwhals_to_native_dtype(
                dtype, self._version, self._native_types
            )
            return _input.cast(spark_dtype)

        return self._from_call(_cast, "cast", expr_kind=self._expr_kind)

    def count(self: Self) -> Self:
        return self._from_call(self._F.count, "count", expr_kind=ExprKind.AGGREGATION)

    def max(self: Self) -> Self:
        return self._from_call(self._F.max, "max", expr_kind=ExprKind.AGGREGATION)

    def mean(self: Self) -> Self:
        return self._from_call(self._F.mean, "mean", expr_kind=ExprKind.AGGREGATION)

    def median(self: Self) -> Self:
        def _median(_input: Column) -> Column:
            import pyspark  # ignore-banned-import

            if parse_version(pyspark) < (3, 4):
                # Use percentile_approx with default accuracy parameter (10000)
                return self._F.percentile_approx(_input.cast("double"), 0.5)

            return self._F.median(_input)

        return self._from_call(_median, "median", expr_kind=ExprKind.AGGREGATION)

    def min(self: Self) -> Self:
        return self._from_call(self._F.min, "min", expr_kind=ExprKind.AGGREGATION)

    def null_count(self: Self) -> Self:
        def _null_count(_input: Column) -> Column:
            return self._F.count_if(self._F.isnull(_input))

        return self._from_call(_null_count, "null_count", expr_kind=ExprKind.AGGREGATION)

    def sum(self: Self) -> Self:
        return self._from_call(self._F.sum, "sum", expr_kind=ExprKind.AGGREGATION)

    def std(self: Self, ddof: int) -> Self:
        from functools import partial

        import numpy as np  # ignore-banned-import

        from narwhals._spark_like.utils import _std

        func = partial(_std, ddof=ddof, np_version=parse_version(np), functions=self._F)

        return self._from_call(func, "std", expr_kind=ExprKind.AGGREGATION)

    def var(self: Self, ddof: int) -> Self:
        from functools import partial

        import numpy as np  # ignore-banned-import

        from narwhals._spark_like.utils import _var

        func = partial(_var, ddof=ddof, np_version=parse_version(np), functions=self._F)

        return self._from_call(func, "var", expr_kind=ExprKind.AGGREGATION)

    def clip(
        self: Self,
        lower_bound: Any | None = None,
        upper_bound: Any | None = None,
    ) -> Self:
        def _clip(_input: Column, lower_bound: Any, upper_bound: Any) -> Column:
            result = _input
            if lower_bound is not None:
                # Convert lower_bound to a literal Column
                result = self._F.when(
                    result < lower_bound, self._F.lit(lower_bound)
                ).otherwise(result)
            if upper_bound is not None:
                # Convert upper_bound to a literal Column
                result = self._F.when(
                    result > upper_bound, self._F.lit(upper_bound)
                ).otherwise(result)
            return result

        return self._from_call(
            _clip,
            "clip",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            expr_kind=self._expr_kind,
        )

    def is_finite(self: Self) -> Self:
        def _is_finite(_input: Column) -> Column:
            # A value is finite if it's not NaN, and not infinite, while NULLs should be
            # preserved
            is_finite_condition = (
                ~self._F.isnan(_input)
                & (_input != self._F.lit(float("inf")))
                & (_input != self._F.lit(float("-inf")))
            )
            return self._F.when(~self._F.isnull(_input), is_finite_condition).otherwise(
                None
            )

        return self._from_call(_is_finite, "is_finite", expr_kind=self._expr_kind)

    def is_in(self: Self, values: Sequence[Any]) -> Self:
        def _is_in(_input: Column) -> Column:
            return _input.isin(values)

        return self._from_call(
            _is_in,
            "is_in",
            expr_kind=self._expr_kind,
        )

    def is_unique(self: Self) -> Self:
        def _is_unique(_input: Column) -> Column:
            # Create a window spec that treats each value separately
            return self._F.count("*").over(self._Window.partitionBy(_input)) == 1

        return self._from_call(_is_unique, "is_unique", expr_kind=self._expr_kind)

    def len(self: Self) -> Self:
        def _len(_input: Column) -> Column:
            # Use count(*) to count all rows including nulls
            return self._F.count("*")

        return self._from_call(_len, "len", expr_kind=ExprKind.AGGREGATION)

    def round(self: Self, decimals: int) -> Self:
        def _round(_input: Column) -> Column:
            return self._F.round(_input, decimals)

        return self._from_call(
            _round,
            "round",
            expr_kind=self._expr_kind,
        )

    def skew(self: Self) -> Self:
        return self._from_call(self._F.skewness, "skew", expr_kind=ExprKind.AGGREGATION)

    def n_unique(self: Self) -> Self:
        def _n_unique(_input: Column) -> Column:
            return self._F.count_distinct(_input) + self._F.max(
                self._F.isnull(_input).cast(self._native_types.IntegerType())
            )

        return self._from_call(_n_unique, "n_unique", expr_kind=ExprKind.AGGREGATION)

    def over(self: Self, keys: list[str]) -> Self:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [expr.over(self._Window.partitionBy(*keys)) for expr in self._call(df)]

        return self.__class__(
            func,
            function_name=self._function_name + "->over",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
            expr_kind=ExprKind.TRANSFORM,
        )

    def is_null(self: Self) -> Self:
        return self._from_call(self._F.isnull, "is_null", expr_kind=self._expr_kind)

    def is_nan(self: Self) -> Self:
        def _is_nan(_input: Column) -> Column:
            return self._F.when(self._F.isnull(_input), None).otherwise(
                self._F.isnan(_input)
            )

        return self._from_call(_is_nan, "is_nan", expr_kind=self._expr_kind)

    @property
    def str(self: Self) -> SparkLikeExprStringNamespace:
        return SparkLikeExprStringNamespace(self)

    @property
    def name(self: Self) -> SparkLikeExprNameNamespace:
        return SparkLikeExprNameNamespace(self)

    @property
    def dt(self: Self) -> SparkLikeExprDateTimeNamespace:
        return SparkLikeExprDateTimeNamespace(self)

    @property
    def list(self: Self) -> SparkLikeExprListNamespace:
        return SparkLikeExprListNamespace(self)
