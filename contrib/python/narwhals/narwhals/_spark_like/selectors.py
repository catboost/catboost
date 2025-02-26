from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

from narwhals._spark_like.expr import SparkLikeExpr
from narwhals._spark_like.utils import ExprKind
from narwhals.utils import Implementation
from narwhals.utils import _parse_time_unit_and_time_zone
from narwhals.utils import dtype_matches_time_unit_and_time_zone
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from datetime import timezone

    from pyspark.sql import Column
    from typing_extensions import Self

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit
    from narwhals.utils import Version


class SparkLikeSelectorNamespace:
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

    def by_dtype(self: Self, dtypes: Iterable[DType | type[DType]]) -> SparkLikeSelector:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [df._F.col(col) for col in df.columns if df.schema[col] in dtypes]

        def evaluate_output_names(df: SparkLikeLazyFrame) -> Sequence[str]:
            return [col for col in df.columns if df.schema[col] in dtypes]

        return SparkLikeSelector(
            func,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            expr_kind=ExprKind.TRANSFORM,
            version=self._version,
            implementation=self._implementation,
        )

    def matches(self: Self, pattern: str) -> SparkLikeSelector:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [df._F.col(col) for col in df.columns if re.search(pattern, col)]

        def evaluate_output_names(df: SparkLikeLazyFrame) -> Sequence[str]:
            return [col for col in df.columns if re.search(pattern, col)]

        return SparkLikeSelector(
            func,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            expr_kind=ExprKind.TRANSFORM,
            version=self._version,
            implementation=self._implementation,
        )

    def numeric(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype(
            {
                dtypes.Int128,
                dtypes.Int64,
                dtypes.Int32,
                dtypes.Int16,
                dtypes.Int8,
                dtypes.UInt128,
                dtypes.UInt64,
                dtypes.UInt32,
                dtypes.UInt16,
                dtypes.UInt8,
                dtypes.Float64,
                dtypes.Float32,
            },
        )

    def categorical(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype({dtypes.Categorical})

    def string(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype({dtypes.String})

    def boolean(self: Self) -> SparkLikeSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype({dtypes.Boolean})

    def all(self: Self) -> SparkLikeSelector:
        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [df._F.col(col) for col in df.columns]

        return SparkLikeSelector(
            func,
            function_name="selector",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            backend_version=self._backend_version,
            expr_kind=ExprKind.TRANSFORM,
            version=self._version,
            implementation=self._implementation,
        )

    def datetime(
        self: Self,
        time_unit: TimeUnit | Iterable[TimeUnit] | None,
        time_zone: str | timezone | Iterable[str | timezone | None] | None,
    ) -> SparkLikeSelector:
        dtypes = import_dtypes_module(version=self._version)
        time_units, time_zones = _parse_time_unit_and_time_zone(
            time_unit=time_unit, time_zone=time_zone
        )

        def func(df: SparkLikeLazyFrame) -> list[Column]:
            return [
                df._F.col(col)
                for col in df.columns
                if dtype_matches_time_unit_and_time_zone(
                    dtype=df.schema[col],
                    dtypes=dtypes,
                    time_units=time_units,
                    time_zones=time_zones,
                )
            ]

        def evalute_output_names(df: SparkLikeLazyFrame) -> Sequence[str]:
            return [
                col
                for col in df.columns
                if dtype_matches_time_unit_and_time_zone(
                    dtype=df.schema[col],
                    dtypes=dtypes,
                    time_units=time_units,
                    time_zones=time_zones,
                )
            ]

        return SparkLikeSelector(
            func,
            function_name="selector",
            evaluate_output_names=evalute_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            expr_kind=ExprKind.TRANSFORM,
            version=self._version,
            implementation=self._implementation,
        )


class SparkLikeSelector(SparkLikeExpr):
    def __repr__(self: Self) -> str:  # pragma: no cover
        return f"SparkLikeSelector(function_name={self._function_name})"

    def _to_expr(self: Self) -> SparkLikeExpr:
        return SparkLikeExpr(
            self._call,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            expr_kind=self._expr_kind,
            version=self._version,
            implementation=self._implementation,
        )

    def __sub__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Column]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name not in rhs_names]

            def evaluate_output_names(df: SparkLikeLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x not in rhs_names]

            return SparkLikeSelector(
                call,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                expr_kind=self._expr_kind,
                version=self._version,
                implementation=self._implementation,
            )
        else:
            return self._to_expr() - other

    def __or__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Column]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                rhs = other._call(df)
                return [
                    *(x for x, name in zip(lhs, lhs_names) if name not in rhs_names),
                    *rhs,
                ]

            def evaluate_output_names(df: SparkLikeLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [*(x for x in lhs_names if x not in rhs_names), *rhs_names]

            return SparkLikeSelector(
                call,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                expr_kind=self._expr_kind,
                version=self._version,
                implementation=self._implementation,
            )
        else:
            return self._to_expr() | other

    def __and__(self: Self, other: SparkLikeSelector | Any) -> SparkLikeSelector | Any:
        if isinstance(other, SparkLikeSelector):

            def call(df: SparkLikeLazyFrame) -> list[Column]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name in rhs_names]

            def evaluate_output_names(df: SparkLikeLazyFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x in rhs_names]

            return SparkLikeSelector(
                call,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                expr_kind=self._expr_kind,
                version=self._version,
                implementation=self._implementation,
            )
        else:
            return self._to_expr() & other

    def __invert__(self: Self) -> SparkLikeSelector:
        return (
            SparkLikeSelectorNamespace(
                backend_version=self._backend_version,
                version=self._version,
                implementation=self._implementation,
            ).all()
            - self
        )
