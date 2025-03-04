from __future__ import annotations

from typing import TYPE_CHECKING

from duckdb import FunctionExpression

from narwhals._duckdb.utils import lit

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._duckdb.expr import DuckDBExpr


class DuckDBExprDateTimeNamespace:
    def __init__(self: Self, expr: DuckDBExpr) -> None:
        self._compliant_expr = expr

    def year(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("year", _input),
            "year",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def month(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("month", _input),
            "month",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def day(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("day", _input),
            "day",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def hour(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("hour", _input),
            "hour",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def minute(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("minute", _input),
            "minute",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def second(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("second", _input),
            "second",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def millisecond(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("millisecond", _input)
            - FunctionExpression("second", _input) * lit(1_000),
            "millisecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def microsecond(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("microsecond", _input)
            - FunctionExpression("second", _input) * lit(1_000_000),
            "microsecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def nanosecond(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("nanosecond", _input)
            - FunctionExpression("second", _input) * lit(1_000_000_000),
            "nanosecond",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def to_string(self: Self, format: str) -> DuckDBExpr:  # noqa: A002
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("strftime", _input, lit(format)),
            "to_string",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def weekday(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("isodow", _input),
            "weekday",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def ordinal_day(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("dayofyear", _input),
            "ordinal_day",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def date(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: _input.cast("date"),
            "date",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def total_minutes(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: FunctionExpression("datepart", lit("minute"), _input),
            "total_minutes",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def total_seconds(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: lit(60) * FunctionExpression("datepart", lit("minute"), _input)
            + FunctionExpression("datepart", lit("second"), _input),
            "total_seconds",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def total_milliseconds(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: lit(60_000)
            * FunctionExpression("datepart", lit("minute"), _input)
            + FunctionExpression("datepart", lit("millisecond"), _input),
            "total_milliseconds",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def total_microseconds(self: Self) -> DuckDBExpr:
        return self._compliant_expr._from_call(
            lambda _input: lit(60_000_000)
            * FunctionExpression("datepart", lit("minute"), _input)
            + FunctionExpression("datepart", lit("microsecond"), _input),
            "total_microseconds",
            expr_kind=self._compliant_expr._expr_kind,
        )

    def total_nanoseconds(self: Self) -> DuckDBExpr:
        msg = "`total_nanoseconds` is not implemented for DuckDB"
        raise NotImplementedError(msg)
