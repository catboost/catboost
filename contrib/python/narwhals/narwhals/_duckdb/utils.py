from __future__ import annotations

import re
from enum import Enum
from enum import auto
from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any

import duckdb

from narwhals.utils import import_dtypes_module
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.expr import DuckDBExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class ExprKind(Enum):
    """Describe which kind of expression we are dealing with.

    Composition rule is:
    - LITERAL vs LITERAL -> LITERAL
    - TRANSFORM vs anything -> TRANSFORM
    - anything vs TRANSFORM -> TRANSFORM
    - all remaining cases -> AGGREGATION
    """

    LITERAL = auto()  # e.g. nw.lit(1)
    AGGREGATION = auto()  # e.g. nw.col('a').mean()
    TRANSFORM = auto()  # e.g. nw.col('a').round()


def maybe_evaluate(df: DuckDBLazyFrame, obj: Any, *, expr_kind: ExprKind) -> Any:
    from narwhals._duckdb.expr import DuckDBExpr

    if isinstance(obj, DuckDBExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:  # pragma: no cover
            msg = "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) not supported in this context"
            raise NotImplementedError(msg)
        column_result = column_results[0]
        if obj._expr_kind is ExprKind.AGGREGATION and expr_kind is ExprKind.TRANSFORM:
            # Returns scalar, but overall expression doesn't.
            # Not yet supported.
            msg = (
                "Mixing expressions which aggregate and expressions which don't\n"
                "is not yet supported by the DuckDB backend. Once they introduce\n"
                "duckdb.WindowExpression to their Python API, we'll be able to\n"
                "support this."
            )
            raise NotImplementedError(msg)
        return column_result
    return duckdb.ConstantExpression(obj)


def parse_exprs_and_named_exprs(
    df: DuckDBLazyFrame, /, *exprs: DuckDBExpr, **named_exprs: DuckDBExpr
) -> dict[str, duckdb.Expression]:
    native_results: dict[str, duckdb.Expression] = {}
    for expr in exprs:
        native_series_list = expr._call(df)
        output_names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            output_names = expr._alias_output_names(output_names)
        if len(output_names) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.update(zip(output_names, native_series_list))
    for col_alias, expr in named_exprs.items():
        native_series_list = expr._call(df)
        if len(native_series_list) != 1:  # pragma: no cover
            msg = "Named expressions must return a single column"
            raise ValueError(msg)
        native_results[col_alias] = native_series_list[0]
    return native_results


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(duckdb_dtype: str, version: Version) -> DType:
    dtypes = import_dtypes_module(version)
    if duckdb_dtype == "HUGEINT":
        return dtypes.Int128()
    if duckdb_dtype == "BIGINT":
        return dtypes.Int64()
    if duckdb_dtype == "INTEGER":
        return dtypes.Int32()
    if duckdb_dtype == "SMALLINT":
        return dtypes.Int16()
    if duckdb_dtype == "TINYINT":
        return dtypes.Int8()
    if duckdb_dtype == "UHUGEINT":
        return dtypes.UInt128()
    if duckdb_dtype == "UBIGINT":
        return dtypes.UInt64()
    if duckdb_dtype == "UINTEGER":
        return dtypes.UInt32()
    if duckdb_dtype == "USMALLINT":
        return dtypes.UInt16()
    if duckdb_dtype == "UTINYINT":
        return dtypes.UInt8()
    if duckdb_dtype == "DOUBLE":
        return dtypes.Float64()
    if duckdb_dtype == "FLOAT":
        return dtypes.Float32()
    if duckdb_dtype == "VARCHAR":
        return dtypes.String()
    if duckdb_dtype == "DATE":
        return dtypes.Date()
    if duckdb_dtype == "TIMESTAMP":
        return dtypes.Datetime()
    if duckdb_dtype == "BOOLEAN":
        return dtypes.Boolean()
    if duckdb_dtype == "INTERVAL":
        return dtypes.Duration()
    if duckdb_dtype.startswith("STRUCT"):
        matchstruc_ = re.findall(r"(\w+)\s+(\w+)", duckdb_dtype)
        return dtypes.Struct(
            [
                dtypes.Field(
                    matchstruc_[i][0],
                    native_to_narwhals_dtype(matchstruc_[i][1], version),
                )
                for i in range(len(matchstruc_))
            ]
        )
    if match_ := re.match(r"(.*)\[\]$", duckdb_dtype):
        return dtypes.List(native_to_narwhals_dtype(match_.group(1), version))
    if match_ := re.match(r"(\w+)((?:\[\d+\])+)", duckdb_dtype):
        duckdb_inner_type = match_.group(1)
        duckdb_shape = match_.group(2)
        shape = tuple(int(value) for value in re.findall(r"\[(\d+)\]", duckdb_shape))
        return dtypes.Array(
            inner=native_to_narwhals_dtype(duckdb_inner_type, version),
            shape=shape,
        )
    if duckdb_dtype.startswith("DECIMAL("):
        return dtypes.Decimal()
    return dtypes.Unknown()  # pragma: no cover


def narwhals_to_native_dtype(dtype: DType | type[DType], version: Version) -> str:
    dtypes = import_dtypes_module(version)
    if isinstance_or_issubclass(dtype, dtypes.Decimal):
        msg = "Casting to Decimal is not supported yet."
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return "DOUBLE"
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return "FLOAT"
    if isinstance_or_issubclass(dtype, dtypes.Int128):
        return "INT128"
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return "BIGINT"
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return "INTEGER"
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return "SMALLINT"
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return "TINYINT"
    if isinstance_or_issubclass(dtype, dtypes.UInt128):
        return "UINT128"
    if isinstance_or_issubclass(dtype, dtypes.UInt64):
        return "UBIGINT"
    if isinstance_or_issubclass(dtype, dtypes.UInt32):
        return "UINTEGER"
    if isinstance_or_issubclass(dtype, dtypes.UInt16):  # pragma: no cover
        return "USMALLINT"
    if isinstance_or_issubclass(dtype, dtypes.UInt8):  # pragma: no cover
        return "UTINYINT"
    if isinstance_or_issubclass(dtype, dtypes.String):
        return "VARCHAR"
    if isinstance_or_issubclass(dtype, dtypes.Boolean):  # pragma: no cover
        return "BOOLEAN"
    if isinstance_or_issubclass(dtype, dtypes.Categorical):
        msg = "Categorical not supported by DuckDB"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        _time_unit = getattr(dtype, "time_unit", "us")
        _time_zone = getattr(dtype, "time_zone", None)
        msg = "todo"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Duration):  # pragma: no cover
        _time_unit = getattr(dtype, "time_unit", "us")
        msg = "todo"
        raise NotImplementedError(msg)
    if isinstance_or_issubclass(dtype, dtypes.Date):  # pragma: no cover
        return "DATE"
    if isinstance_or_issubclass(dtype, dtypes.List):
        inner = narwhals_to_native_dtype(dtype.inner, version)  # type: ignore[union-attr]
        return f"{inner}[]"
    if isinstance_or_issubclass(dtype, dtypes.Struct):  # pragma: no cover
        inner = ", ".join(
            f'"{field.name}" {narwhals_to_native_dtype(field.dtype, version)}'
            for field in dtype.fields  # type: ignore[union-attr]
        )
        return f"STRUCT({inner})"
    if isinstance_or_issubclass(dtype, dtypes.Array):  # pragma: no cover
        shape: tuple[int] = dtype.shape  # type: ignore[union-attr]
        duckdb_shape_fmt = "".join(f"[{item}]" for item in shape)
        inner_dtype = dtype
        for _ in shape:
            inner_dtype = inner_dtype.inner  # type: ignore[union-attr]
        duckdb_inner = narwhals_to_native_dtype(inner_dtype, version)
        return f"{duckdb_inner}{duckdb_shape_fmt}"
    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def n_ary_operation_expr_kind(*args: DuckDBExpr | Any) -> ExprKind:
    if all(
        getattr(arg, "_expr_kind", ExprKind.LITERAL) is ExprKind.LITERAL for arg in args
    ):
        return ExprKind.LITERAL
    if any(
        getattr(arg, "_expr_kind", ExprKind.LITERAL) is ExprKind.TRANSFORM for arg in args
    ):
        return ExprKind.TRANSFORM
    return ExprKind.AGGREGATION
