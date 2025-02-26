from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing import Any
from typing import Iterable
from typing import Sequence

from narwhals._arrow.expr import ArrowExpr
from narwhals.utils import Implementation
from narwhals.utils import _parse_time_unit_and_time_zone
from narwhals.utils import dtype_matches_time_unit_and_time_zone
from narwhals.utils import import_dtypes_module

if TYPE_CHECKING:
    from datetime import timezone

    from typing_extensions import Self

    from narwhals._arrow.dataframe import ArrowDataFrame
    from narwhals._arrow.series import ArrowSeries
    from narwhals.dtypes import DType
    from narwhals.typing import TimeUnit
    from narwhals.utils import Version


class ArrowSelectorNamespace:
    def __init__(
        self: Self, *, backend_version: tuple[int, ...], version: Version
    ) -> None:
        self._backend_version = backend_version
        self._implementation = Implementation.PYARROW
        self._version = version

    def by_dtype(self: Self, dtypes: Iterable[DType | type[DType]]) -> ArrowSelector:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            return [df[col] for col in df.columns if df.schema[col] in dtypes]

        def evaluate_output_names(df: ArrowDataFrame) -> Sequence[str]:
            return [col for col in df.columns if df.schema[col] in dtypes]

        return ArrowSelector(
            func,
            depth=0,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"dtypes": dtypes},
        )

    def matches(self: Self, pattern: str) -> ArrowSelector:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            return [df[col] for col in df.columns if re.search(pattern, col)]

        def evaluate_output_names(df: ArrowDataFrame) -> Sequence[str]:
            return [col for col in df.columns if re.search(pattern, col)]

        return ArrowSelector(
            func,
            depth=0,
            function_name="selector",
            evaluate_output_names=evaluate_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={"pattern": pattern},
        )

    def numeric(self: Self) -> ArrowSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype(
            [
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
            ],
        )

    def categorical(self: Self) -> ArrowSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Categorical])

    def string(self: Self) -> ArrowSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.String])

    def boolean(self: Self) -> ArrowSelector:
        dtypes = import_dtypes_module(self._version)
        return self.by_dtype([dtypes.Boolean])

    def all(self: Self) -> ArrowSelector:
        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
            return [df[col] for col in df.columns]

        return ArrowSelector(
            func,
            depth=0,
            function_name="selector",
            evaluate_output_names=lambda df: df.columns,
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )

    def datetime(
        self: Self,
        time_unit: TimeUnit | Iterable[TimeUnit] | None,
        time_zone: str | timezone | Iterable[str | timezone | None] | None,
    ) -> ArrowSelector:
        dtypes = import_dtypes_module(version=self._version)
        time_units, time_zones = _parse_time_unit_and_time_zone(
            time_unit=time_unit, time_zone=time_zone
        )

        def func(df: ArrowDataFrame) -> list[ArrowSeries]:
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

        def evalute_output_names(df: ArrowDataFrame) -> Sequence[str]:
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

        return ArrowSelector(
            func,
            depth=0,
            function_name="selector",
            evaluate_output_names=evalute_output_names,
            alias_output_names=None,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={},
        )


class ArrowSelector(ArrowExpr):
    def __repr__(self: Self) -> str:  # pragma: no cover
        return f"ArrowSelector(depth={self._depth}, function_name={self._function_name})"

    def _to_expr(self: Self) -> ArrowExpr:
        return ArrowExpr(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            backend_version=self._backend_version,
            version=self._version,
            kwargs=self._kwargs,
        )

    def __sub__(self: Self, other: Self | Any) -> ArrowSelector | Any:
        if isinstance(other, ArrowSelector):

            def call(df: ArrowDataFrame) -> list[ArrowSeries]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name not in rhs_names]

            def evaluate_output_names(df: ArrowDataFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x not in rhs_names]

            return ArrowSelector(
                call,
                depth=0,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                version=self._version,
                kwargs={**self._kwargs, "other": other},
            )
        else:
            return self._to_expr() - other

    def __or__(self: Self, other: Self | Any) -> ArrowSelector | Any:
        if isinstance(other, ArrowSelector):

            def call(df: ArrowDataFrame) -> list[ArrowSeries]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                rhs = other._call(df)
                return [
                    *(x for x, name in zip(lhs, lhs_names) if name not in rhs_names),
                    *rhs,
                ]

            def evaluate_output_names(df: ArrowDataFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [*(x for x in lhs_names if x not in rhs_names), *rhs_names]

            return ArrowSelector(
                call,
                depth=0,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                version=self._version,
                kwargs={**self._kwargs, "other": other},
            )
        else:
            return self._to_expr() | other

    def __and__(self: Self, other: Self | Any) -> ArrowSelector | Any:
        if isinstance(other, ArrowSelector):

            def call(df: ArrowDataFrame) -> list[ArrowSeries]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                lhs = self._call(df)
                return [x for x, name in zip(lhs, lhs_names) if name in rhs_names]

            def evaluate_output_names(df: ArrowDataFrame) -> list[str]:
                lhs_names = self._evaluate_output_names(df)
                rhs_names = other._evaluate_output_names(df)
                return [x for x in lhs_names if x in rhs_names]

            return ArrowSelector(
                call,
                depth=0,
                function_name="selector",
                evaluate_output_names=evaluate_output_names,
                alias_output_names=None,
                backend_version=self._backend_version,
                version=self._version,
                kwargs={**self._kwargs, "other": other},
            )
        else:
            return self._to_expr() & other

    def __invert__(self: Self) -> ArrowSelector:
        return (
            ArrowSelectorNamespace(
                backend_version=self._backend_version, version=self._version
            ).all()
            - self
        )
