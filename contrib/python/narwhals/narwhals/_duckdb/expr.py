from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from duckdb import CaseExpression
from duckdb import CoalesceOperator
from duckdb import ColumnExpression
from duckdb import FunctionExpression
from duckdb.typing import DuckDBPyType

from narwhals._duckdb.expr_dt import DuckDBExprDateTimeNamespace
from narwhals._duckdb.expr_list import DuckDBExprListNamespace
from narwhals._duckdb.expr_name import DuckDBExprNameNamespace
from narwhals._duckdb.expr_str import DuckDBExprStringNamespace
from narwhals._duckdb.utils import ExprKind
from narwhals._duckdb.utils import lit
from narwhals._duckdb.utils import maybe_evaluate
from narwhals._duckdb.utils import n_ary_operation_expr_kind
from narwhals._duckdb.utils import narwhals_to_native_dtype
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation

if TYPE_CHECKING:
    import duckdb
    from typing_extensions import Self

    from narwhals._duckdb.dataframe import DuckDBLazyFrame
    from narwhals._duckdb.namespace import DuckDBNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class DuckDBExpr(CompliantExpr["duckdb.Expression"]):  # type: ignore[type-var]
    _implementation = Implementation.DUCKDB
    _depth = 0  # Unused, just for compatibility with CompliantExpr

    def __init__(
        self: Self,
        call: Callable[[DuckDBLazyFrame], Sequence[duckdb.Expression]],
        *,
        function_name: str,
        evaluate_output_names: Callable[[DuckDBLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        expr_kind: ExprKind,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> None:
        self._call = call
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._expr_kind = expr_kind
        self._backend_version = backend_version
        self._version = version

    def __call__(self: Self, df: DuckDBLazyFrame) -> Sequence[duckdb.Expression]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DuckDBNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._duckdb.namespace import DuckDBNamespace

        return DuckDBNamespace(
            backend_version=self._backend_version, version=self._version
        )

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(_: DuckDBLazyFrame) -> list[duckdb.Expression]:
            return [ColumnExpression(col_name) for col_name in column_names]

        return cls(
            func,
            function_name="col",
            evaluate_output_names=lambda _df: list(column_names),
            alias_output_names=None,
            expr_kind=ExprKind.TRANSFORM,
            backend_version=backend_version,
            version=version,
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
            columns = df.columns

            return [ColumnExpression(columns[i]) for i in column_indices]

        return cls(
            func,
            function_name="nth",
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            expr_kind=ExprKind.TRANSFORM,
            backend_version=backend_version,
            version=version,
        )

    def _from_call(
        self: Self,
        call: Callable[..., duckdb.Expression],
        expr_name: str,
        *,
        expr_kind: ExprKind,
        **expressifiable_args: Self | Any,
    ) -> Self:
        """Create expression from callable.

        Arguments:
            call: Callable from compliant DataFrame to native Expression
            expr_name: Expression name
            expr_kind: kind of output expression
            expressifiable_args: arguments pass to expression which should be parsed
                as expressions (e.g. in `nw.col('a').is_between('b', 'c')`)
        """

        def func(df: DuckDBLazyFrame) -> list[duckdb.Expression]:
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
        )

    def __and__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input & other,
            "__and__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __or__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input | other,
            "__or__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __add__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input + other,
            "__add__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __truediv__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input / other,
            "__truediv__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __floordiv__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__floordiv__(other),
            "__floordiv__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __mod__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mod__(other),
            "__mod__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __sub__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input - other,
            "__sub__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __mul__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input * other,
            "__mul__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __pow__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input**other,
            "__pow__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __lt__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input < other,
            "__lt__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __gt__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input > other,
            "__gt__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __le__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input <= other,
            "__le__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __ge__(self: Self, other: DuckDBExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input >= other,
            "__ge__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __eq__(self: Self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input == other,
            "__eq__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __ne__(self: Self, other: DuckDBExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input != other,
            "__ne__",
            other=other,
            expr_kind=n_ary_operation_expr_kind(self, other),
        )

    def __invert__(self: Self) -> Self:
        return self._from_call(
            lambda _input: ~_input,
            "__invert__",
            expr_kind=self._expr_kind,
        )

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
        )

    def abs(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("abs", _input),
            "abs",
            expr_kind=self._expr_kind,
        )

    def mean(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("mean", _input),
            "mean",
            expr_kind=ExprKind.AGGREGATION,
        )

    def skew(self: Self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            count = FunctionExpression("count", _input)
            return CaseExpression(condition=(count == lit(0)), value=lit(None)).otherwise(
                CaseExpression(
                    condition=(count == lit(1)), value=lit(float("nan"))
                ).otherwise(
                    CaseExpression(condition=(count == lit(2)), value=lit(0.0)).otherwise(
                        # Adjust population skewness by correction factor to get sample skewness
                        FunctionExpression("skewness", _input)
                        * (count - lit(2))
                        / FunctionExpression("sqrt", count * (count - lit(1)))
                    )
                )
            )

        return self._from_call(func, "skew", expr_kind=ExprKind.AGGREGATION)

    def median(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("median", _input),
            "median",
            expr_kind=ExprKind.AGGREGATION,
        )

    def all(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("bool_and", _input),
            "all",
            expr_kind=ExprKind.AGGREGATION,
        )

    def any(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("bool_or", _input),
            "any",
            expr_kind=ExprKind.AGGREGATION,
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            if interpolation == "linear":
                return FunctionExpression("quantile_cont", _input, lit(quantile))
            msg = "Only linear interpolation methods are supported for DuckDB quantile."
            raise NotImplementedError(msg)

        return self._from_call(
            func,
            "quantile",
            expr_kind=ExprKind.AGGREGATION,
        )

    def clip(self: Self, lower_bound: Any, upper_bound: Any) -> Self:
        def func(
            _input: duckdb.Expression, lower_bound: Any, upper_bound: Any
        ) -> duckdb.Expression:
            return FunctionExpression(
                "greatest", FunctionExpression("least", _input, upper_bound), lower_bound
            )

        return self._from_call(
            func,
            "clip",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            expr_kind=self._expr_kind,
        )

    def sum(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("sum", _input),
            "sum",
            expr_kind=ExprKind.AGGREGATION,
        )

    def n_unique(self: Self) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            # https://stackoverflow.com/a/79338887/4451315
            return FunctionExpression(
                "array_unique", FunctionExpression("array_agg", _input)
            ) + FunctionExpression(
                "max",
                CaseExpression(condition=_input.isnotnull(), value=lit(0)).otherwise(
                    lit(1)
                ),
            )

        return self._from_call(
            func,
            "n_unique",
            expr_kind=ExprKind.AGGREGATION,
        )

    def count(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("count", _input),
            "count",
            expr_kind=ExprKind.AGGREGATION,
        )

    def len(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("count"),
            "len",
            expr_kind=ExprKind.AGGREGATION,
        )

    def std(self: Self, ddof: int) -> Self:
        def _std(_input: duckdb.Expression, ddof: int) -> duckdb.Expression:
            n_samples = FunctionExpression("count", _input)
            # NOTE: Not implemented Error: Unable to transform python value of type '<class 'duckdb.duckdb.Expression'>' to DuckDB LogicalType
            return (
                FunctionExpression("stddev_pop", _input)
                * FunctionExpression("sqrt", n_samples)
                / (FunctionExpression("sqrt", (n_samples - ddof)))  # type: ignore[operator]
            )

        return self._from_call(
            _std,
            "std",
            ddof=ddof,
            expr_kind=ExprKind.AGGREGATION,
        )

    def var(self: Self, ddof: int) -> Self:
        def _var(_input: duckdb.Expression, ddof: int) -> duckdb.Expression:
            n_samples = FunctionExpression("count", _input)
            # NOTE: Not implemented Error: Unable to transform python value of type '<class 'duckdb.duckdb.Expression'>' to DuckDB LogicalType
            return FunctionExpression("var_pop", _input) * n_samples / (n_samples - ddof)  # type: ignore[operator]

        return self._from_call(
            _var,
            "var",
            ddof=ddof,
            expr_kind=ExprKind.AGGREGATION,
        )

    def max(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("max", _input),
            "max",
            expr_kind=ExprKind.AGGREGATION,
        )

    def min(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("min", _input),
            "min",
            expr_kind=ExprKind.AGGREGATION,
        )

    def null_count(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("sum", _input.isnull().cast("int")),
            "null_count",
            expr_kind=ExprKind.AGGREGATION,
        )

    def is_null(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isnull(),
            "is_null",
            expr_kind=self._expr_kind,
        )

    def is_nan(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("isnan", _input),
            "is_nan",
            expr_kind=self._expr_kind,
        )

    def is_finite(self: Self) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("isfinite", _input),
            "is_finite",
            expr_kind=self._expr_kind,
        )

    def is_in(self: Self, other: Sequence[Any]) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("contains", lit(other), _input),
            "is_in",
            expr_kind=self._expr_kind,
        )

    def round(self: Self, decimals: int) -> Self:
        return self._from_call(
            lambda _input: FunctionExpression("round", _input, lit(decimals)),
            "round",
            expr_kind=self._expr_kind,
        )

    def fill_null(self: Self, value: Any, strategy: Any, limit: int | None) -> Self:
        if strategy is not None:
            msg = "todo"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: CoalesceOperator(_input, lit(value)),
            "fill_null",
            expr_kind=self._expr_kind,
        )

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def func(_input: duckdb.Expression) -> duckdb.Expression:
            native_dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.cast(DuckDBPyType(native_dtype))

        return self._from_call(
            func,
            "cast",
            expr_kind=self._expr_kind,
        )

    @property
    def str(self: Self) -> DuckDBExprStringNamespace:
        return DuckDBExprStringNamespace(self)

    @property
    def dt(self: Self) -> DuckDBExprDateTimeNamespace:
        return DuckDBExprDateTimeNamespace(self)

    @property
    def name(self: Self) -> DuckDBExprNameNamespace:
        return DuckDBExprNameNamespace(self)

    @property
    def list(self: Self) -> DuckDBExprListNamespace:
        return DuckDBExprListNamespace(self)
