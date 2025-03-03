from __future__ import annotations

import operator
from functools import reduce
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Sequence

from narwhals._expression_parsing import combine_alias_output_names
from narwhals._expression_parsing import combine_evaluate_output_names
from narwhals._spark_like.dataframe import SparkLikeLazyFrame
from narwhals._spark_like.expr import SparkLikeExpr
from narwhals._spark_like.selectors import SparkLikeSelectorNamespace
from narwhals._spark_like.utils import ExprKind
from narwhals._spark_like.utils import n_ary_operation_expr_kind
from narwhals._spark_like.utils import narwhals_to_native_dtype
from narwhals.typing import CompliantNamespace

if TYPE_CHECKING:
    from pyspark.sql import Column
    from pyspark.sql import DataFrame
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.utils import Implementation
    from narwhals.utils import Version


class SparkLikeNamespace(CompliantNamespace["Column"]):
    def __init__(
        self: Self,
        *,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._implementation = implementation

    @property
    def selectors(self: Self) -> SparkLikeSelectorNamespace:
        return SparkLikeSelectorNamespace(
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def all(self: Self) -> SparkLikeExpr:
        def _all(df: SparkLikeLazyFrame) -> list[Column]:
            return [df._F.col(col_name) for col_name in df.columns]

        return SparkLikeExpr(
            call=_all,
            function_name="all",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            expr_kind=ExprKind.TRANSFORM,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def col(self: Self, *column_names: str) -> SparkLikeExpr:
        return SparkLikeExpr.from_column_names(
            *column_names,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def nth(self: Self, *column_indices: int) -> SparkLikeExpr:
        return SparkLikeExpr.from_column_indices(
            *column_indices,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def lit(self: Self, value: object, dtype: DType | None) -> SparkLikeExpr:
        def _lit(df: SparkLikeLazyFrame) -> list[Column]:
            column = df._F.lit(value)
            if dtype:
                native_dtype = narwhals_to_native_dtype(
                    dtype, version=self._version, spark_types=df._native_dtypes
                )
                column = column.cast(native_dtype)

            return [column]

        return SparkLikeExpr(
            call=_lit,
            function_name="lit",
            evaluate_output_names=lambda _df: ["literal"],
            alias_output_names=None,
            expr_kind=ExprKind.LITERAL,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def len(self: Self) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [df._F.count("*")]

        return SparkLikeExpr(
            func,
            function_name="len",
            evaluate_output_names=lambda _df: ["len"],
            alias_output_names=None,
            expr_kind=ExprKind.AGGREGATION,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def all_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.and_, cols)]

        return SparkLikeExpr(
            call=func,
            function_name="all_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def any_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [reduce(operator.or_, cols)]

        return SparkLikeExpr(
            call=func,
            function_name="any_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def sum_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (
                df._F.coalesce(col, df._F.lit(0)) for _expr in exprs for col in _expr(df)
            )
            return [reduce(operator.add, cols)]

        return SparkLikeExpr(
            call=func,
            function_name="sum_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def mean_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [c for _expr in exprs for c in _expr(df)]
            return [
                (
                    reduce(
                        operator.add,
                        (df._F.coalesce(col, df._F.lit(0)) for col in cols),
                    )
                    / reduce(
                        operator.add,
                        (
                            col.isNotNull().cast(df._native_dtypes.IntegerType())
                            for col in cols
                        ),
                    )
                )
            ]

        return SparkLikeExpr(
            call=func,
            function_name="mean_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def max_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [df._F.greatest(*cols)]

        return SparkLikeExpr(
            call=func,
            function_name="max_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def min_horizontal(self: Self, *exprs: SparkLikeExpr) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = (c for _expr in exprs for c in _expr(df))
            return [df._F.least(*cols)]

        return SparkLikeExpr(
            call=func,
            function_name="min_horizontal",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def concat(
        self: Self,
        items: Iterable[SparkLikeLazyFrame],
        *,
        how: Literal["horizontal", "vertical", "diagonal"],
    ) -> SparkLikeLazyFrame:
        dfs: list[DataFrame] = [item._native_frame for item in items]
        if how == "horizontal":
            msg = (
                "Horizontal concatenation is not supported for LazyFrame backed by "
                "a PySpark DataFrame."
            )
            raise NotImplementedError(msg)

        if how == "vertical":
            cols_0 = dfs[0].columns
            for i, df in enumerate(dfs[1:], start=1):
                cols_current = df.columns
                if not ((len(cols_current) == len(cols_0)) and (cols_current == cols_0)):
                    msg = (
                        "unable to vstack, column names don't match:\n"
                        f"   - dataframe 0: {cols_0}\n"
                        f"   - dataframe {i}: {cols_current}\n"
                    )
                    raise TypeError(msg)

            return SparkLikeLazyFrame(
                native_dataframe=reduce(lambda x, y: x.union(y), dfs),
                backend_version=self._backend_version,
                version=self._version,
                implementation=self._implementation,
            )

        if how == "diagonal":
            return SparkLikeLazyFrame(
                native_dataframe=reduce(
                    lambda x, y: x.unionByName(y, allowMissingColumns=True), dfs
                ),
                backend_version=self._backend_version,
                version=self._version,
                implementation=self._implementation,
            )
        raise NotImplementedError

    def concat_str(
        self: Self,
        *exprs: SparkLikeExpr,
        separator: str,
        ignore_nulls: bool,
    ) -> SparkLikeExpr:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            cols = [s for _expr in exprs for s in _expr(df)]
            cols_casted = [s.cast(df._native_dtypes.StringType()) for s in cols]
            null_mask = [df._F.isnull(s) for _expr in exprs for s in _expr(df)]

            if not ignore_nulls:
                null_mask_result = reduce(operator.or_, null_mask)
                result = df._F.when(
                    ~null_mask_result,
                    reduce(
                        lambda x, y: df._F.format_string(f"%s{separator}%s", x, y),
                        cols_casted,
                    ),
                ).otherwise(df._F.lit(None))
            else:
                init_value, *values = [
                    df._F.when(~nm, col).otherwise(df._F.lit(""))
                    for col, nm in zip(cols_casted, null_mask)
                ]

                separators = (
                    df._F.when(nm, df._F.lit("")).otherwise(df._F.lit(separator))
                    for nm in null_mask[:-1]
                )
                result = reduce(
                    lambda x, y: df._F.format_string("%s%s", x, y),
                    (
                        df._F.format_string("%s%s", s, v)
                        for s, v in zip(separators, values)
                    ),
                    init_value,
                )

            return [result]

        return SparkLikeExpr(
            call=func,
            function_name="concat_str",
            evaluate_output_names=combine_evaluate_output_names(*exprs),
            alias_output_names=combine_alias_output_names(*exprs),
            expr_kind=n_ary_operation_expr_kind(*exprs),
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def when(self: Self, *predicates: SparkLikeExpr) -> SparkLikeWhen:
        plx = self.__class__(
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )
        condition = plx.all_horizontal(*predicates)
        return SparkLikeWhen(
            condition,
            self._backend_version,
            expr_kind=ExprKind.TRANSFORM,
            version=self._version,
            implementation=self._implementation,
        )


class SparkLikeWhen:
    def __init__(
        self: Self,
        condition: SparkLikeExpr,
        backend_version: tuple[int, ...],
        then_value: Any | None = None,
        otherwise_value: Any | None = None,
        *,
        expr_kind: ExprKind,
        version: Version,
        implementation: Implementation,
    ) -> None:
        self._backend_version = backend_version
        self._condition = condition
        self._then_value = then_value
        self._otherwise_value = otherwise_value
        self._expr_kind = expr_kind
        self._version = version
        self._implementation = implementation

    def __call__(self: Self, df: SparkLikeLazyFrame) -> list[Column]:
        condition = self._condition(df)[0]

        if isinstance(self._then_value, SparkLikeExpr):
            value_ = self._then_value(df)[0]
        else:
            # `self._then_value` is a scalar
            value_ = df._F.lit(self._then_value)

        if isinstance(self._otherwise_value, SparkLikeExpr):
            other_ = self._otherwise_value(df)[0]
        else:
            # `self._otherwise_value` is a scalar
            other_ = df._F.lit(self._otherwise_value)

        return [df._F.when(condition=condition, value=value_).otherwise(value=other_)]

    def then(self: Self, value: SparkLikeExpr | Any) -> SparkLikeThen:
        self._then_value = value

        return SparkLikeThen(
            self,
            function_name="whenthen",
            evaluate_output_names=getattr(
                value, "_evaluate_output_names", lambda _df: ["literal"]
            ),
            alias_output_names=getattr(value, "_alias_output_names", None),
            expr_kind=self._expr_kind,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )


class SparkLikeThen(SparkLikeExpr):
    def __init__(
        self: Self,
        call: SparkLikeWhen,
        *,
        function_name: str,
        evaluate_output_names: Callable[[SparkLikeLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        expr_kind: ExprKind,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> None:
        self._backend_version = backend_version
        self._version = version
        self._call = call
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._expr_kind = expr_kind
        self._implementation = implementation

    def otherwise(self: Self, value: SparkLikeExpr | Any) -> SparkLikeExpr:
        # type ignore because we are setting the `_call` attribute to a
        # callable object of type `SparkLikeWhen`, base class has the attribute as
        # only a `Callable`
        self._call._otherwise_value = value  # type: ignore[attr-defined]
        self._function_name = "whenotherwise"
        return self
