from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING
from typing import Any

from narwhals.exceptions import UnsupportedDTypeError
from narwhals.utils import import_dtypes_module
from narwhals.utils import isinstance_or_issubclass

if TYPE_CHECKING:
    from types import ModuleType

    import pyspark.sql.types as pyspark_types
    from pyspark.sql import Column

    from narwhals._spark_like.dataframe import SparkLikeLazyFrame
    from narwhals._spark_like.expr import SparkLikeExpr
    from narwhals.dtypes import DType
    from narwhals.utils import Version


@lru_cache(maxsize=16)
def native_to_narwhals_dtype(
    dtype: pyspark_types.DataType,
    version: Version,
    spark_types: ModuleType,
) -> DType:  # pragma: no cover
    dtypes = import_dtypes_module(version=version)

    if isinstance(dtype, spark_types.DoubleType):
        return dtypes.Float64()
    if isinstance(dtype, spark_types.FloatType):
        return dtypes.Float32()
    if isinstance(dtype, spark_types.LongType):
        return dtypes.Int64()
    if isinstance(dtype, spark_types.IntegerType):
        return dtypes.Int32()
    if isinstance(dtype, spark_types.ShortType):
        return dtypes.Int16()
    if isinstance(dtype, spark_types.ByteType):
        return dtypes.Int8()
    if isinstance(
        dtype, (spark_types.StringType, spark_types.VarcharType, spark_types.CharType)
    ):
        return dtypes.String()
    if isinstance(dtype, spark_types.BooleanType):
        return dtypes.Boolean()
    if isinstance(dtype, spark_types.DateType):
        return dtypes.Date()
    if isinstance(dtype, spark_types.TimestampNTZType):
        return dtypes.Datetime()
    if isinstance(dtype, spark_types.TimestampType):
        return dtypes.Datetime(time_zone="UTC")
    if isinstance(dtype, spark_types.DecimalType):
        return dtypes.Decimal()
    if isinstance(dtype, spark_types.ArrayType):
        return dtypes.List(
            inner=native_to_narwhals_dtype(
                dtype.elementType, version=version, spark_types=spark_types
            )
        )
    if isinstance(dtype, spark_types.StructType):
        return dtypes.Struct(
            fields=[
                dtypes.Field(
                    name=field.name,
                    dtype=native_to_narwhals_dtype(
                        field.dataType, version=version, spark_types=spark_types
                    ),
                )
                for field in dtype
            ]
        )
    return dtypes.Unknown()


def narwhals_to_native_dtype(
    dtype: DType | type[DType], version: Version, spark_types: ModuleType
) -> pyspark_types.DataType:
    dtypes = import_dtypes_module(version)

    if isinstance_or_issubclass(dtype, dtypes.Float64):
        return spark_types.DoubleType()
    if isinstance_or_issubclass(dtype, dtypes.Float32):
        return spark_types.FloatType()
    if isinstance_or_issubclass(dtype, dtypes.Int64):
        return spark_types.LongType()
    if isinstance_or_issubclass(dtype, dtypes.Int32):
        return spark_types.IntegerType()
    if isinstance_or_issubclass(dtype, dtypes.Int16):
        return spark_types.ShortType()
    if isinstance_or_issubclass(dtype, dtypes.Int8):
        return spark_types.ByteType()
    if isinstance_or_issubclass(dtype, dtypes.String):
        return spark_types.StringType()
    if isinstance_or_issubclass(dtype, dtypes.Boolean):
        return spark_types.BooleanType()
    if isinstance_or_issubclass(dtype, dtypes.Date):
        return spark_types.DateType()
    if isinstance_or_issubclass(dtype, dtypes.Datetime):
        dt_time_zone = dtype.time_zone
        if dt_time_zone is None:
            return spark_types.TimestampNTZType()
        if dt_time_zone != "UTC":  # pragma: no cover
            msg = f"Only UTC time zone is supported for PySpark, got: {dt_time_zone}"
            raise ValueError(msg)
        return spark_types.TimestampType()
    if isinstance_or_issubclass(dtype, (dtypes.List, dtypes.Array)):
        return spark_types.ArrayType(
            elementType=narwhals_to_native_dtype(
                dtype.inner, version=version, spark_types=spark_types
            )
        )
    if isinstance_or_issubclass(dtype, dtypes.Struct):  # pragma: no cover
        return spark_types.StructType(
            fields=[
                spark_types.StructField(
                    name=field.name,
                    dataType=narwhals_to_native_dtype(
                        field.dtype,
                        version=version,
                        spark_types=spark_types,
                    ),
                )
                for field in dtype.fields
            ]
        )

    if isinstance_or_issubclass(
        dtype,
        (
            dtypes.UInt64,
            dtypes.UInt32,
            dtypes.UInt16,
            dtypes.UInt8,
            dtypes.Enum,
            dtypes.Categorical,
        ),
    ):  # pragma: no cover
        msg = "Unsigned integer, Enum and Categorical types are not supported by spark-like backend"
        raise UnsupportedDTypeError(msg)

    msg = f"Unknown dtype: {dtype}"  # pragma: no cover
    raise AssertionError(msg)


def parse_exprs(df: SparkLikeLazyFrame, /, *exprs: SparkLikeExpr) -> dict[str, Column]:
    native_results: dict[str, list[Column]] = {}

    for expr in exprs:
        native_series_list = expr._call(df)
        output_names = expr._evaluate_output_names(df)
        if expr._alias_output_names is not None:
            output_names = expr._alias_output_names(output_names)
        if len(output_names) != len(native_series_list):  # pragma: no cover
            msg = f"Internal error: got output names {output_names}, but only got {len(native_series_list)} results"
            raise AssertionError(msg)
        native_results.update(zip(output_names, native_series_list))

    return native_results


def maybe_evaluate_expr(df: SparkLikeLazyFrame, obj: SparkLikeExpr | object) -> Column:
    from narwhals._spark_like.expr import SparkLikeExpr

    if isinstance(obj, SparkLikeExpr):
        column_results = obj._call(df)
        if len(column_results) != 1:
            msg = "Multi-output expressions (e.g. `nw.all()` or `nw.col('a', 'b')`) not supported in this context"
            raise ValueError(msg)
        return column_results[0]
    return df._F.lit(obj)


def _std(
    _input: Column | str, ddof: int, np_version: tuple[int, ...], functions: Any
) -> Column:
    if np_version > (2, 0):
        if ddof == 1:
            return functions.stddev_samp(_input)
        if ddof == 0:
            return functions.stddev_pop(_input)

        n_rows = functions.count(_input)
        return functions.stddev_samp(_input) * functions.sqrt(
            (n_rows - 1) / (n_rows - ddof)
        )

    from pyspark.pandas.spark.functions import stddev

    input_col = functions.col(_input) if isinstance(_input, str) else _input
    return stddev(input_col, ddof=ddof)


def _var(
    _input: Column | str, ddof: int, np_version: tuple[int, ...], functions: Any
) -> Column:
    if np_version > (2, 0):
        if ddof == 1:
            return functions.var_samp(_input)

        n_rows = functions.count(_input)
        return functions.var_samp(_input) * (n_rows - 1) / (n_rows - ddof)

    from pyspark.pandas.spark.functions import var

    input_col = functions.col(_input) if isinstance(_input, str) else _input
    return var(input_col, ddof=ddof)
