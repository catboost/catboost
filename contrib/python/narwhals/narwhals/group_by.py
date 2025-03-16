from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import TypeVar
from typing import cast

from narwhals._expression_parsing import all_exprs_are_aggs_or_literals
from narwhals.dataframe import DataFrame
from narwhals.dataframe import LazyFrame
from narwhals.exceptions import InvalidOperationError
from narwhals.utils import flatten
from narwhals.utils import tupleify

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals.expr import Expr

DataFrameT = TypeVar("DataFrameT")
LazyFrameT = TypeVar("LazyFrameT")


class GroupBy(Generic[DataFrameT]):
    def __init__(self: Self, df: DataFrameT, *keys: str, drop_null_keys: bool) -> None:
        self._df = cast(DataFrame[Any], df)
        self._keys = keys
        self._grouped = self._df._compliant_frame.group_by(
            *self._keys, drop_null_keys=drop_null_keys
        )

    def agg(self: Self, *aggs: Expr | Iterable[Expr], **named_aggs: Expr) -> DataFrameT:
        """Compute aggregations for each group of a group by operation.

        Arguments:
            aggs: Aggregations to compute for each group of the group by operation,
                specified as positional arguments.
            named_aggs: Additional aggregations, specified as keyword arguments.

        Returns:
            A new Dataframe.

        Examples:
            Group by one column or by multiple columns and call `agg` to compute
            the grouped sum of another column.

            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_pd = pd.DataFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "c"],
            ...         "b": [1, 2, 1, 3, 3],
            ...         "c": [5, 4, 3, 2, 1],
            ...     }
            ... )
            >>> df_pl = pl.DataFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "c"],
            ...         "b": [1, 2, 1, 3, 3],
            ...         "c": [5, 4, 3, 2, 1],
            ...     }
            ... )

            We define library agnostic functions:

            >>> @nw.narwhalify
            ... def func(df):
            ...     return df.group_by("a").agg(nw.col("b").sum()).sort("a")

            >>> @nw.narwhalify
            ... def func_mult_col(df):
            ...     return df.group_by("a", "b").agg(nw.sum("c")).sort("a", "b")

            We can then pass either pandas or Polars to `func` and `func_mult_col`:

            >>> func(df_pd)
               a  b
            0  a  2
            1  b  5
            2  c  3
            >>> func(df_pl)
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ str ┆ i64 │
            ╞═════╪═════╡
            │ a   ┆ 2   │
            │ b   ┆ 5   │
            │ c   ┆ 3   │
            └─────┴─────┘
            >>> func_mult_col(df_pd)
               a  b  c
            0  a  1  8
            1  b  2  4
            2  b  3  2
            3  c  3  1
            >>> func_mult_col(df_pl)
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 1   ┆ 8   │
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            └─────┴─────┴─────┘
        """
        flat_aggs = tuple(flatten(aggs))
        if not all_exprs_are_aggs_or_literals(*flat_aggs, **named_aggs):
            msg = (
                "Found expression which does not aggregate.\n\n"
                "All expressions passed to GroupBy.agg must aggregate.\n"
                "For example, `df.group_by('a').agg(nw.col('b').sum())` is valid,\n"
                "but `df.group_by('a').agg(nw.col('b'))` is not."
            )
            raise InvalidOperationError(msg)
        plx = self._df.__narwhals_namespace__()
        compliant_aggs = (
            *(x._to_compliant_expr(plx) for x in flat_aggs),
            *(
                value._to_compliant_expr(plx).alias(key)
                for key, value in named_aggs.items()
            ),
        )
        return self._df._from_compliant_dataframe(  # type: ignore[return-value]
            self._grouped.agg(*compliant_aggs),
        )

    def __iter__(self: Self) -> Iterator[tuple[Any, DataFrameT]]:
        yield from (  # type: ignore[misc]
            (tupleify(key), self._df._from_compliant_dataframe(df))
            for (key, df) in self._grouped.__iter__()
        )


class LazyGroupBy(Generic[LazyFrameT]):
    def __init__(self: Self, df: LazyFrameT, *keys: str, drop_null_keys: bool) -> None:
        self._df = cast(LazyFrame[Any], df)
        self._keys = keys
        self._grouped = self._df._compliant_frame.group_by(
            *self._keys, drop_null_keys=drop_null_keys
        )

    def agg(self: Self, *aggs: Expr | Iterable[Expr], **named_aggs: Expr) -> LazyFrameT:
        """Compute aggregations for each group of a group by operation.

        Arguments:
            aggs: Aggregations to compute for each group of the group by operation,
                specified as positional arguments.
            named_aggs: Additional aggregations, specified as keyword arguments.

        Returns:
            A new LazyFrame.

        Examples:
            Group by one column or by multiple columns and call `agg` to compute
            the grouped sum of another column.

            >>> import polars as pl
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>> lf_pl = pl.LazyFrame(
            ...     {
            ...         "a": ["a", "b", "a", "b", "c"],
            ...         "b": [1, 2, 1, 3, 3],
            ...         "c": [5, 4, 3, 2, 1],
            ...     }
            ... )

            We define library agnostic functions:

            >>> def agnostic_func_one_col(lf_native: IntoFrameT) -> IntoFrameT:
            ...     lf = nw.from_native(lf_native)
            ...     return nw.to_native(lf.group_by("a").agg(nw.col("b").sum()).sort("a"))

            >>> def agnostic_func_mult_col(lf_native: IntoFrameT) -> IntoFrameT:
            ...     lf = nw.from_native(lf_native)
            ...     return nw.to_native(
            ...         lf.group_by("a", "b").agg(nw.sum("c")).sort("a", "b")
            ...     )

            We can then pass a lazy frame and materialise it with `collect`:

            >>> agnostic_func_one_col(lf_pl).collect()
            shape: (3, 2)
            ┌─────┬─────┐
            │ a   ┆ b   │
            │ --- ┆ --- │
            │ str ┆ i64 │
            ╞═════╪═════╡
            │ a   ┆ 2   │
            │ b   ┆ 5   │
            │ c   ┆ 3   │
            └─────┴─────┘
            >>> agnostic_func_mult_col(lf_pl).collect()
            shape: (4, 3)
            ┌─────┬─────┬─────┐
            │ a   ┆ b   ┆ c   │
            │ --- ┆ --- ┆ --- │
            │ str ┆ i64 ┆ i64 │
            ╞═════╪═════╪═════╡
            │ a   ┆ 1   ┆ 8   │
            │ b   ┆ 2   ┆ 4   │
            │ b   ┆ 3   ┆ 2   │
            │ c   ┆ 3   ┆ 1   │
            └─────┴─────┴─────┘
        """
        flat_aggs = tuple(flatten(aggs))
        if not all_exprs_are_aggs_or_literals(*flat_aggs, **named_aggs):
            msg = (
                "Found expression which does not aggregate.\n\n"
                "All expressions passed to GroupBy.agg must aggregate.\n"
                "For example, `df.group_by('a').agg(nw.col('b').sum())` is valid,\n"
                "but `df.group_by('a').agg(nw.col('b'))` is not."
            )
            raise InvalidOperationError(msg)
        plx = self._df.__narwhals_namespace__()
        compliant_aggs = (
            *(x._to_compliant_expr(plx) for x in flat_aggs),
            *(
                value._to_compliant_expr(plx).alias(key)
                for key, value in named_aggs.items()
            ),
        )
        return self._df._from_compliant_dataframe(  # type: ignore[return-value]
            self._grouped.agg(*compliant_aggs),
        )
