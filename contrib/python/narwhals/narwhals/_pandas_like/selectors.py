from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Sequence

from narwhals._pandas_like.expr import PandasLikeExpr
from narwhals.utils import _parse_time_unit_and_time_zone
from narwhals.utils import dtype_matches_time_unit_and_time_zone
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from datetime import timezone

    from typing_extensions import Self

    from narwhals._pandas_like.dataframe import PandasLikeDataFrame
    from narwhals._pandas_like.series import PandasLikeSeries
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit
    from narwhals.utils import _FullContext


class PandasSelectorNamespace:
    def __init__(self: Self, context: _FullContext, /) -> None:
        self._implementation = context._implementation
        self._backend_version = context._backend_version
        self._version = context._version

    def by_dtype(self: Self, dtypes: Iterable[DType | type[DType]]) -> PandasSelector:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            return [df[col] for col in df.columns if df.schema[col] in dtypes]

        def evaluate_output_names(df: PandasLikeDataFrame) -> Sequence[str]:
            return [col for col in df.columns if df.schema[col] in dtypes]

        return selector(self, func, evaluate_output_names)

    def matches(self: Self, pattern: str) -> PandasSelector:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            return [df[col] for col in df.columns if re.search(pattern, col)]

        def evaluate_output_names(df: PandasLikeDataFrame) -> Sequence[str]:
            return [col for col in df.columns if re.search(pattern, col)]

        return selector(self, func, evaluate_output_names)

    def numeric(self: Self) -> PandasSelector:
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
            }
        )

    def categorical(self: Self) -> PandasSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype({dtypes.Categorical})

    def string(self: Self) -> PandasSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype({dtypes.String})

    def boolean(self: Self) -> PandasSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype({dtypes.Boolean})

    def all(self: Self) -> PandasSelector:
        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            return [df[col] for col in df.columns]

        return selector(self, func, lambda df: df.columns)

    def datetime(
        self: Self,
        time_unit: TimeUnit | Iterable[TimeUnit] | None,
        time_zone: str | timezone | Iterable[str | timezone | None] | None,
    ) -> PandasSelector:
        dtypes = import_dtypes_module(version=self._version)
        time_units, time_zones = _parse_time_unit_and_time_zone(
            time_unit=time_unit, time_zone=time_zone
        )

        def func(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
            return [
                df[col]
                for col in df.columns
                if dtype_matches_time_unit_and_time_zone(
                    dtype=df.schema[col],
                    dtypes=dtypes,
                    time_units=time_units,
                    time_zones=time_zones,
                )
            ]

        def evaluate_output_names(df: PandasLikeDataFrame) -> Sequence[str]:
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

        return selector(self, func, evaluate_output_names)


class PandasSelector(PandasLikeExpr):
    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"PandasSelector(depth={self._depth}, function_name={self._function_name}, "
        )

    def _to_expr(self: Self) -> PandasLikeExpr:
        return PandasLikeExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            implementation=self._implementation,
            backend_version=self._backend_version,
            version=self._version,
            call_kwargs=self._call_kwargs,
        )

    def __sub__(self: Self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name not in rhs_names]

            def evaluate_output_names(df: PandasLikeDataFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x not in rhs_names]

            return selector(self, call, evaluate_output_names)
        else:
            return self._to_expr() - other

    def __or__(self: Self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                rhs = other._call(df)
                return [
                    *(x for x, name in zip(lhs, lhs_names) if name not in rhs_names),
                    *rhs,
                ]

            def evaluate_output_names(df: PandasLikeDataFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [*(x for x in lhs_names if x not in rhs_names), *rhs_names]

            return selector(self, call, evaluate_output_names)
        else:
            return self._to_expr() | other

    def __and__(self: Self, other: PandasSelector | Any) -> PandasSelector | Any:
        if isinstance(other, PandasSelector):

            def call(df: PandasLikeDataFrame) -> list[PandasLikeSeries]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name in rhs_names]

            def evaluate_output_names(df: PandasLikeDataFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x in rhs_names]

            return selector(self, call, evaluate_output_names)
        else:
            return self._to_expr() & other

    def __invert__(self: Self) -> PandasSelector:
        return PandasSelectorNamespace(self).all() - self


def selector(
    context: _FullContext,
    call: Callable[[PandasLikeDataFrame], Sequence[PandasLikeSeries]],
    evaluate_output_names: Callable[[PandasLikeDataFrame], Sequence[str]],
    /,
) -> PandasSelector:
    return PandasSelector(
        call,
        depth=0,
        function_name="selector",
        evaluate_output_names=evaluate_output_names,
        alias_output_names=None,
        implementation=context._implementation,
        backend_version=context._backend_version,
        version=context._version,
    )
