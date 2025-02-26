from __future__ import annotations

from itertools import chain
from typing import TYPE_CHECKING
from typing import Any
from typing import Literal
from typing import Sequence

from narwhals._spark_like.utils import ExprKind
from narwhals._spark_like.utils import native_to_narwhals_dtype
from narwhals._spark_like.utils import parse_exprs_and_named_exprs
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import CompliantDataFrame
from narwhals.typing import CompliantLazyFrame
from narwhals.utils import Implementation
from narwhals.utils import check_column_exists
from narwhals.utils import import_dtypes_module
from narwhals.utils import parse_columns_to_drop
from narwhals.utils import parse_version
from narwhals.utils import validate_backend_version

if TYPE_CHECKING:
    from types import ModuleType

    from pyspark.sql import DataFrame
    from typing_extensions import Self

    from narwhals._spark_like.expr import SparkLikeExpr
    from narwhals._spark_like.group_by import SparkLikeLazyGroupBy
    from narwhals._spark_like.namespace import SparkLikeNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class SparkLikeLazyFrame(CompliantLazyFrame):
    def __init__(
        self: Self,
        native_dataframe: DataFrame,
        *,
        backend_version: tuple[int, ...],
        version: Version,
        implementation: Implementation,
    ) -> None:
        self._native_frame = native_dataframe
        self._backend_version = backend_version
        self._implementation = implementation
        self._version = version
        validate_backend_version(self._implementation, self._backend_version)

    @property
    def _F(self: Self) -> Any:  # noqa: N802
        if self._implementation is Implementation.SQLFRAME:
            from sqlframe.duckdb import functions

            return functions
        from pyspark.sql import functions

        return functions

    @property
    def _native_dtypes(self: Self) -> Any:
        if self._implementation is Implementation.SQLFRAME:
            from sqlframe.duckdb import types

            return types
        from pyspark.sql import types

        return types

    @property
    def _Window(self: Self) -> Any:  # noqa: N802
        if self._implementation is Implementation.SQLFRAME:
            from sqlframe.duckdb import Window

            return Window
        from pyspark.sql import Window

        return Window

    def __native_namespace__(self: Self) -> ModuleType:  # pragma: no cover
        return self._implementation.to_native_namespace()

    def __narwhals_namespace__(self: Self) -> SparkLikeNamespace:
        from narwhals._spark_like.namespace import SparkLikeNamespace

        return SparkLikeNamespace(
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    def __narwhals_lazyframe__(self: Self) -> Self:
        return self

    def _change_version(self: Self, version: Version) -> Self:
        return self.__class__(
            self._native_frame,
            backend_version=self._backend_version,
            version=version,
            implementation=self._implementation,
        )

    def _from_native_frame(self: Self, df: DataFrame) -> Self:
        return self.__class__(
            df,
            backend_version=self._backend_version,
            version=self._version,
            implementation=self._implementation,
        )

    @property
    def columns(self: Self) -> list[str]:
        return self._native_frame.columns  # type: ignore[no-any-return]

    def collect(
        self: Self,
        backend: ModuleType | Implementation | str | None,
        **kwargs: Any,
    ) -> CompliantDataFrame:
        if backend is Implementation.PANDAS:
            import pandas as pd  # ignore-banned-import

            from narwhals._pandas_like.dataframe import PandasLikeDataFrame

            return PandasLikeDataFrame(
                native_dataframe=self._native_frame.toPandas(),
                implementation=Implementation.PANDAS,
                backend_version=parse_version(pd),
                version=self._version,
                validate_column_names=False,
            )

        elif backend is None or backend is Implementation.PYARROW:
            import pyarrow as pa  # ignore-banned-import

            from narwhals._arrow.dataframe import ArrowDataFrame

            try:
                native_pyarrow_frame = pa.Table.from_batches(
                    self._native_frame._collect_as_arrow()
                )
            except ValueError as exc:
                if "at least one RecordBatch" in str(exc):
                    # Empty dataframe
                    from narwhals._arrow.utils import narwhals_to_native_dtype

                    data: dict[str, list[Any]] = {}
                    schema = []
                    current_schema = self.collect_schema()
                    for key, value in current_schema.items():
                        data[key] = []
                        schema.append(
                            (key, narwhals_to_native_dtype(value, self._version))
                        )
                    native_pyarrow_frame = pa.Table.from_pydict(
                        data, schema=pa.schema(schema)
                    )
                else:  # pragma: no cover
                    raise
            return ArrowDataFrame(
                native_pyarrow_frame,
                backend_version=parse_version(pa),
                version=self._version,
                validate_column_names=False,
            )

        elif backend is Implementation.POLARS:
            import polars as pl  # ignore-banned-import
            import pyarrow as pa  # ignore-banned-import

            from narwhals._polars.dataframe import PolarsDataFrame

            return PolarsDataFrame(
                df=pl.from_arrow(  # type: ignore[arg-type]
                    pa.Table.from_batches(self._native_frame._collect_as_arrow())
                ),
                backend_version=parse_version(pl),
                version=self._version,
            )

        msg = f"Unsupported `backend` value: {backend}"  # pragma: no cover
        raise ValueError(msg)  # pragma: no cover

    def simple_select(self: Self, *column_names: str) -> Self:
        return self._from_native_frame(self._native_frame.select(*column_names))

    def select(
        self: Self,
        *exprs: SparkLikeExpr,
        **named_exprs: SparkLikeExpr,
    ) -> Self:
        new_columns, expr_kinds = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)

        if not new_columns:
            # return empty dataframe, like Polars does
            spark_session = self._native_frame.sparkSession
            spark_df = spark_session.createDataFrame(
                [], self._native_dtypes.StructType([])
            )

            return self._from_native_frame(spark_df)

        if not any(expr_kind is ExprKind.TRANSFORM for expr_kind in expr_kinds):
            new_columns_list = [
                col.alias(col_name) for col_name, col in new_columns.items()
            ]
            return self._from_native_frame(self._native_frame.agg(*new_columns_list))
        else:
            new_columns_list = [
                col.over(self._Window().partitionBy(self._F.lit(1))).alias(col_name)
                if expr_kind is ExprKind.AGGREGATION
                else col.alias(col_name)
                for (col_name, col), expr_kind in zip(new_columns.items(), expr_kinds)
            ]
            return self._from_native_frame(self._native_frame.select(*new_columns_list))

    def with_columns(
        self: Self,
        *exprs: SparkLikeExpr,
        **named_exprs: SparkLikeExpr,
    ) -> Self:
        new_columns, expr_kinds = parse_exprs_and_named_exprs(self, *exprs, **named_exprs)

        new_columns_map = {
            col_name: col.over(self._Window().partitionBy(self._F.lit(1)))
            if expr_kind is ExprKind.AGGREGATION
            else col
            for (col_name, col), expr_kind in zip(new_columns.items(), expr_kinds)
        }
        return self._from_native_frame(self._native_frame.withColumns(new_columns_map))

    def filter(self: Self, *predicates: SparkLikeExpr, **constraints: Any) -> Self:
        plx = self.__narwhals_namespace__()
        expr = plx.all_horizontal(
            *chain(predicates, (plx.col(name) == v for name, v in constraints.items()))
        )
        # `[0]` is safe as all_horizontal's expression only returns a single column
        condition = expr._call(self)[0]
        spark_df = self._native_frame.where(condition)
        return self._from_native_frame(spark_df)

    @property
    def schema(self: Self) -> dict[str, DType]:
        return {
            field.name: native_to_narwhals_dtype(
                dtype=field.dataType,
                version=self._version,
                spark_types=self._native_dtypes,
            )
            for field in self._native_frame.schema
        }

    def collect_schema(self: Self) -> dict[str, DType]:
        return self.schema

    def drop(self: Self, columns: list[str], strict: bool) -> Self:  # noqa: FBT001
        columns_to_drop = parse_columns_to_drop(
            compliant_frame=self, columns=columns, strict=strict
        )
        return self._from_native_frame(self._native_frame.drop(*columns_to_drop))

    def head(self: Self, n: int) -> Self:
        spark_session = self._native_frame.sparkSession

        return self._from_native_frame(
            spark_session.createDataFrame(self._native_frame.take(num=n))
        )

    def group_by(self: Self, *keys: str, drop_null_keys: bool) -> SparkLikeLazyGroupBy:
        from narwhals._spark_like.group_by import SparkLikeLazyGroupBy

        return SparkLikeLazyGroupBy(
            compliant_frame=self, keys=list(keys), drop_null_keys=drop_null_keys
        )

    def sort(
        self: Self,
        *by: str,
        descending: bool | Sequence[bool],
        nulls_last: bool,
    ) -> Self:
        if isinstance(descending, bool):
            descending = [descending] * len(by)

        if nulls_last:
            sort_funcs = (
                self._F.desc_nulls_last if d else self._F.asc_nulls_last
                for d in descending
            )
        else:
            sort_funcs = (
                self._F.desc_nulls_first if d else self._F.asc_nulls_first
                for d in descending
            )

        sort_cols = [sort_f(col) for col, sort_f in zip(by, sort_funcs)]
        return self._from_native_frame(self._native_frame.sort(*sort_cols))

    def drop_nulls(self: Self, subset: list[str] | None) -> Self:
        return self._from_native_frame(self._native_frame.dropna(subset=subset))

    def rename(self: Self, mapping: dict[str, str]) -> Self:
        rename_mapping = {
            colname: mapping.get(colname, colname) for colname in self.columns
        }
        return self._from_native_frame(
            self._native_frame.select(
                [self._F.col(old).alias(new) for old, new in rename_mapping.items()]
            )
        )

    def unique(
        self: Self,
        subset: list[str] | None,
        *,
        keep: Literal["any", "none"],
    ) -> Self:
        if keep != "any":
            msg = "`LazyFrame.unique` with PySpark backend only supports `keep='any'`."
            raise ValueError(msg)
        check_column_exists(self.columns, subset)
        return self._from_native_frame(self._native_frame.dropDuplicates(subset=subset))

    def join(
        self: Self,
        other: Self,
        how: Literal["inner", "left", "cross", "semi", "anti"],
        left_on: list[str] | None,
        right_on: list[str] | None,
        suffix: str,
    ) -> Self:
        self_native = self._native_frame
        other_native = other._native_frame

        left_columns = self.columns
        right_columns = other.columns

        if isinstance(left_on, str):
            left_on = [left_on]
        if isinstance(right_on, str):
            right_on = [right_on]

        # create a mapping for columns on other
        # `right_on` columns will be renamed as `left_on`
        # the remaining columns will be either added the suffix or left unchanged.
        rename_mapping = {
            **dict(zip(right_on or [], left_on or [])),
            **{
                colname: f"{colname}{suffix}" if colname in left_columns else colname
                for colname in list(set(right_columns).difference(set(right_on or [])))
            },
        }
        other = other_native.select(
            [self._F.col(old).alias(new) for old, new in rename_mapping.items()]
        )

        # If how in {"semi", "anti"}, then resulting columns are same as left columns
        # Otherwise, we add the right columns with the new mapping, while keeping the
        # original order of right_columns.
        col_order = left_columns

        if how in {"inner", "left", "cross"}:
            col_order.extend(
                [
                    rename_mapping[colname]
                    for colname in right_columns
                    if colname not in (right_on or [])
                ]
            )

        return self._from_native_frame(
            self_native.join(other, on=left_on, how=how).select(col_order)
        )

    def explode(self: Self, columns: list[str]) -> Self:
        dtypes = import_dtypes_module(self._version)

        schema = self.collect_schema()
        for col_to_explode in columns:
            dtype = schema[col_to_explode]

            if dtype != dtypes.List:
                msg = (
                    f"`explode` operation not supported for dtype `{dtype}`, "
                    "expected List type"
                )
                raise InvalidOperationError(msg)

        native_frame = self._native_frame
        column_names = self.columns

        if len(columns) != 1:
            msg = (
                "Exploding on multiple columns is not supported with SparkLike backend since "
                "we cannot guarantee that the exploded columns have matching element counts."
            )
            raise NotImplementedError(msg)

        return self._from_native_frame(
            native_frame.select(
                *[
                    self._F.col(col_name).alias(col_name)
                    if col_name != columns[0]
                    else self._F.explode_outer(col_name).alias(col_name)
                    for col_name in column_names
                ]
            )
        )

    def unpivot(
        self: Self,
        on: list[str] | None,
        index: list[str] | None,
        variable_name: str,
        value_name: str,
    ) -> Self:
        return self._from_native_frame(
            self._native_frame.unpivot(
                ids=index,
                values=on,
                variableColumnName=variable_name,
                valueColumnName=value_name,
            )
        )
