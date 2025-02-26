from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Literal
from typing import Sequence

from narwhals._dask.expr_dt import DaskExprDateTimeNamespace
from narwhals._dask.expr_name import DaskExprNameNamespace
from narwhals._dask.expr_str import DaskExprStringNamespace
from narwhals._dask.utils import add_row_index
from narwhals._dask.utils import binary_operation_returns_scalar
from narwhals._dask.utils import maybe_evaluate
from narwhals._dask.utils import narwhals_to_native_dtype
from narwhals._expression_parsing import evaluate_output_names_and_aliases
from narwhals._pandas_like.utils import native_to_narwhals_dtype
from narwhals.exceptions import ColumnNotFoundError
from narwhals.exceptions import InvalidOperationError
from narwhals.typing import CompliantExpr
from narwhals.utils import Implementation
from narwhals.utils import generate_temporary_column_name

if TYPE_CHECKING:
    try:
        import dask.dataframe.dask_expr as dx
    except ModuleNotFoundError:
        import dask_expr as dx

    from typing_extensions import Self

    from narwhals._dask.dataframe import DaskLazyFrame
    from narwhals._dask.namespace import DaskNamespace
    from narwhals.dtypes import DType
    from narwhals.utils import Version


class DaskExpr(CompliantExpr["dx.Series"]):
    _implementation: Implementation = Implementation.DASK

    def __init__(
        self: Self,
        call: Callable[[DaskLazyFrame], Sequence[dx.Series]],
        *,
        depth: int,
        function_name: str,
        evaluate_output_names: Callable[[DaskLazyFrame], Sequence[str]],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
        # Whether the expression is a length-1 Series resulting from
        # a reduction, such as `nw.col('a').sum()`
        returns_scalar: bool,
        backend_version: tuple[int, ...],
        version: Version,
        kwargs: dict[str, Any],
    ) -> None:
        self._call = call
        self._depth = depth
        self._function_name = function_name
        self._evaluate_output_names = evaluate_output_names
        self._alias_output_names = alias_output_names
        self._returns_scalar = returns_scalar
        self._backend_version = backend_version
        self._version = version
        self._kwargs = kwargs

    def __call__(self: Self, df: DaskLazyFrame) -> Sequence[dx.Series]:
        return self._call(df)

    def __narwhals_expr__(self) -> None: ...

    def __narwhals_namespace__(self) -> DaskNamespace:  # pragma: no cover
        # Unused, just for compatibility with PandasLikeExpr
        from narwhals._dask.namespace import DaskNamespace

        return DaskNamespace(backend_version=self._backend_version, version=self._version)

    @classmethod
    def from_column_names(
        cls: type[Self],
        *column_names: str,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            try:
                return [df._native_frame[column_name] for column_name in column_names]
            except KeyError as e:
                missing_columns = [x for x in column_names if x not in df.columns]
                raise ColumnNotFoundError.from_missing_and_available_column_names(
                    missing_columns=missing_columns,
                    available_columns=df.columns,
                ) from e

        return cls(
            func,
            depth=0,
            function_name="col",
            evaluate_output_names=lambda _df: list(column_names),
            alias_output_names=None,
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    @classmethod
    def from_column_indices(
        cls: type[Self],
        *column_indices: int,
        backend_version: tuple[int, ...],
        version: Version,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            return [
                df._native_frame.iloc[:, column_index] for column_index in column_indices
            ]

        return cls(
            func,
            depth=0,
            function_name="nth",
            evaluate_output_names=lambda df: [df.columns[i] for i in column_indices],
            alias_output_names=None,
            returns_scalar=False,
            backend_version=backend_version,
            version=version,
            kwargs={},
        )

    def _from_call(
        self: Self,
        # First argument to `call` should be `dx.Series`
        call: Callable[..., dx.Series],
        expr_name: str,
        *,
        returns_scalar: bool,
        **expressifiable_args: Self | Any,
    ) -> Self:
        def func(df: DaskLazyFrame) -> list[dx.Series]:
            native_results: list[dx.Series] = []
            native_series_list = self._call(df)
            other_native_series = {
                key: maybe_evaluate(df, value)
                for key, value in expressifiable_args.items()
            }
            for native_series in native_series_list:
                if self._returns_scalar:
                    result_native = call(native_series[0], **other_native_series)
                else:
                    result_native = call(native_series, **other_native_series)
                if returns_scalar:
                    native_results.append(result_native.to_series())
                else:
                    native_results.append(result_native)
            return native_results

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=f"{self._function_name}->{expr_name}",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            returns_scalar=returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, **expressifiable_args},
        )

    def alias(self: Self, name: str) -> Self:
        def alias_output_names(names: Sequence[str]) -> Sequence[str]:
            if len(names) != 1:
                msg = f"Expected function with single output, found output names: {names}"
                raise ValueError(msg)
            return [name]

        return self.__class__(
            self._call,
            depth=self._depth,
            function_name=self._function_name,
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=alias_output_names,
            returns_scalar=self._returns_scalar,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, "name": name},
        )

    def __add__(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__add__(other),
            "__add__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __sub__(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__sub__(other),
            "__sub__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __mul__(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mul__(other),
            "__mul__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __truediv__(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__truediv__(other),
            "__truediv__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __floordiv__(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__floordiv__(other),
            "__floordiv__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __pow__(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__pow__(other),
            "__pow__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __mod__(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.__mod__(other),
            "__mod__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __eq__(self: Self, other: DaskExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__eq__(other),
            "__eq__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __ne__(self: Self, other: DaskExpr) -> Self:  # type: ignore[override]
        return self._from_call(
            lambda _input, other: _input.__ne__(other),
            "__ne__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __ge__(self: Self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__ge__(other),
            "__ge__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __gt__(self: Self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__gt__(other),
            "__gt__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __le__(self: Self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__le__(other),
            "__le__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __lt__(self: Self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__lt__(other),
            "__lt__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __and__(self: Self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__and__(other),
            "__and__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __or__(self: Self, other: DaskExpr) -> Self:
        return self._from_call(
            lambda _input, other: _input.__or__(other),
            "__or__",
            other=other,
            returns_scalar=binary_operation_returns_scalar(self, other),
        )

    def __invert__(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.__invert__(),
            "__invert__",
            returns_scalar=self._returns_scalar,
        )

    def mean(self: Self) -> Self:
        return self._from_call(lambda _input: _input.mean(), "mean", returns_scalar=True)

    def median(self: Self) -> Self:
        from narwhals.exceptions import InvalidOperationError

        def func(s: dx.Series) -> dx.Series:
            dtype = native_to_narwhals_dtype(s.dtype, self._version, Implementation.DASK)
            if not dtype.is_numeric():
                msg = "`median` operation not supported for non-numeric input type."
                raise InvalidOperationError(msg)
            return s.median_approximate()

        return self._from_call(func, "median", returns_scalar=True)

    def min(self: Self) -> Self:
        return self._from_call(lambda _input: _input.min(), "min", returns_scalar=True)

    def max(self: Self) -> Self:
        return self._from_call(lambda _input: _input.max(), "max", returns_scalar=True)

    def std(self: Self, ddof: int) -> Self:
        return self._from_call(
            lambda _input, ddof: _input.std(ddof=ddof),
            "std",
            ddof=ddof,
            returns_scalar=True,
        )

    def var(self: Self, ddof: int) -> Self:
        return self._from_call(
            lambda _input, ddof: _input.var(ddof=ddof),
            "var",
            ddof=ddof,
            returns_scalar=True,
        )

    def skew(self: Self) -> Self:
        return self._from_call(lambda _input: _input.skew(), "skew", returns_scalar=True)

    def shift(self: Self, n: int) -> Self:
        return self._from_call(
            lambda _input, n: _input.shift(n),
            "shift",
            n=n,
            returns_scalar=self._returns_scalar,
        )

    def cum_sum(self: Self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_sum(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.cumsum(),
            "cum_sum",
            returns_scalar=self._returns_scalar,
        )

    def cum_count(self: Self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_count(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: (~_input.isna()).astype(int).cumsum(),
            "cum_count",
            returns_scalar=self._returns_scalar,
        )

    def cum_min(self: Self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_min(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.cummin(),
            "cum_min",
            returns_scalar=self._returns_scalar,
        )

    def cum_max(self: Self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_max(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.cummax(),
            "cum_max",
            returns_scalar=self._returns_scalar,
        )

    def cum_prod(self: Self, *, reverse: bool) -> Self:
        if reverse:  # pragma: no cover
            msg = "`cum_prod(reverse=True)` is not supported with Dask backend"
            raise NotImplementedError(msg)

        return self._from_call(
            lambda _input: _input.cumprod(),
            "cum_prod",
            returns_scalar=self._returns_scalar,
        )

    def sum(self: Self) -> Self:
        return self._from_call(lambda _input: _input.sum(), "sum", returns_scalar=True)

    def count(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.count(), "count", returns_scalar=True
        )

    def round(self: Self, decimals: int) -> Self:
        return self._from_call(
            lambda _input, decimals: _input.round(decimals),
            "round",
            decimals=decimals,
            returns_scalar=self._returns_scalar,
        )

    def unique(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.unique(),
            "unique",
            returns_scalar=self._returns_scalar,
        )

    def drop_nulls(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.dropna(),
            "drop_nulls",
            returns_scalar=self._returns_scalar,
        )

    def replace_strict(
        self: Self, old: Sequence[Any], new: Sequence[Any], *, return_dtype: DType | None
    ) -> Self:
        msg = "`replace_strict` is not yet supported for Dask expressions"
        raise NotImplementedError(msg)

    def abs(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.abs(), "abs", returns_scalar=self._returns_scalar
        )

    def all(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.all(
                axis=None, skipna=True, split_every=False, out=None
            ),
            "all",
            returns_scalar=True,
        )

    def any(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.any(axis=0, skipna=True, split_every=False),
            "any",
            returns_scalar=True,
        )

    def fill_null(
        self: Self,
        value: Any | None,
        strategy: Literal["forward", "backward"] | None,
        limit: int | None,
    ) -> DaskExpr:
        def func(
            _input: dx.Series,
            value: Any | None,
            strategy: str | None,
            limit: int | None,
        ) -> dx.Series:
            if value is not None:
                res_ser = _input.fillna(value)
            else:
                res_ser = (
                    _input.ffill(limit=limit)
                    if strategy == "forward"
                    else _input.bfill(limit=limit)
                )
            return res_ser

        return self._from_call(
            func,
            "fillna",
            value=value,
            strategy=strategy,
            limit=limit,
            returns_scalar=self._returns_scalar,
        )

    def clip(
        self: Self,
        lower_bound: Self | Any | None,
        upper_bound: Self | Any | None,
    ) -> Self:
        return self._from_call(
            lambda _input, lower_bound, upper_bound: _input.clip(
                lower=lower_bound, upper=upper_bound
            ),
            "clip",
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            returns_scalar=self._returns_scalar,
        )

    def diff(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.diff(), "diff", returns_scalar=self._returns_scalar
        )

    def n_unique(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.nunique(dropna=False), "n_unique", returns_scalar=True
        )

    def is_null(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isna(), "is_null", returns_scalar=self._returns_scalar
        )

    def is_nan(self: Self) -> Self:
        def func(_input: dx.Series) -> dx.Series:
            dtype = native_to_narwhals_dtype(
                _input.dtype, self._version, self._implementation
            )
            if dtype.is_numeric():
                return _input != _input  # noqa: PLR0124
            msg = f"`.is_nan` only supported for numeric dtypes and not {dtype}, did you mean `.is_null`?"
            raise InvalidOperationError(msg)

        return self._from_call(func, "is_null", returns_scalar=self._returns_scalar)

    def len(self: Self) -> Self:
        return self._from_call(lambda _input: _input.size, "len", returns_scalar=True)

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        if interpolation == "linear":

            def func(_input: dx.Series, quantile: float) -> dx.Series:
                if _input.npartitions > 1:
                    msg = "`Expr.quantile` is not supported for Dask backend with multiple partitions."
                    raise NotImplementedError(msg)
                return _input.quantile(q=quantile, method="dask")  # pragma: no cover

            return self._from_call(
                func, "quantile", quantile=quantile, returns_scalar=True
            )
        else:
            msg = "`higher`, `lower`, `midpoint`, `nearest` - interpolation methods are not supported by Dask. Please use `linear` instead."
            raise NotImplementedError(msg)

    def is_first_distinct(self: Self) -> Self:
        def func(_input: dx.Series) -> dx.Series:
            _name = _input.name
            col_token = generate_temporary_column_name(n_bytes=8, columns=[_name])
            _input = add_row_index(
                _input.to_frame(),
                col_token,
                backend_version=self._backend_version,
                implementation=self._implementation,
            )
            first_distinct_index = _input.groupby(_name).agg({col_token: "min"})[
                col_token
            ]
            return _input[col_token].isin(first_distinct_index)

        return self._from_call(
            func, "is_first_distinct", returns_scalar=self._returns_scalar
        )

    def is_last_distinct(self: Self) -> Self:
        def func(_input: dx.Series) -> dx.Series:
            _name = _input.name
            col_token = generate_temporary_column_name(n_bytes=8, columns=[_name])
            _input = add_row_index(
                _input.to_frame(),
                col_token,
                backend_version=self._backend_version,
                implementation=self._implementation,
            )
            last_distinct_index = _input.groupby(_name).agg({col_token: "max"})[col_token]
            return _input[col_token].isin(last_distinct_index)

        return self._from_call(
            func, "is_last_distinct", returns_scalar=self._returns_scalar
        )

    def is_unique(self: Self) -> Self:
        def func(_input: dx.Series) -> dx.Series:
            _name = _input.name
            return (
                _input.to_frame()
                .groupby(_name, dropna=False)
                .transform("size", meta=(_name, int))
                == 1
            )

        return self._from_call(func, "is_unique", returns_scalar=self._returns_scalar)

    def is_in(self: Self, other: Any) -> Self:
        return self._from_call(
            lambda _input, other: _input.isin(other),
            "is_in",
            other=other,
            returns_scalar=self._returns_scalar,
        )

    def null_count(self: Self) -> Self:
        return self._from_call(
            lambda _input: _input.isna().sum(),
            "null_count",
            returns_scalar=True,
        )

    def over(self: Self, keys: list[str]) -> Self:
        def func(df: DaskLazyFrame) -> list[Any]:
            output_names, aliases = evaluate_output_names_and_aliases(self, df, [])
            if overlap := set(output_names).intersection(keys):
                # E.g. `df.select(nw.all().sum().over('a'))`. This is well-defined,
                # we just don't support it yet.
                msg = (
                    f"Column names {overlap} appear in both expression output names and in `over` keys.\n"
                    "This is not yet supported."
                )
                raise NotImplementedError(msg)
            if df._native_frame.npartitions == 1:  # pragma: no cover
                tmp = df.group_by(*keys, drop_null_keys=False).agg(self)
                tmp_native = (
                    df.simple_select(*keys)
                    .join(tmp, how="left", left_on=keys, right_on=keys, suffix="_right")
                    ._native_frame
                )
                return [tmp_native[name] for name in aliases]
            # https://github.com/dask/dask/issues/6659
            msg = (
                "`Expr.over` is not supported for Dask backend with multiple partitions."
            )
            raise NotImplementedError(msg)

        return self.__class__(
            func,
            depth=self._depth + 1,
            function_name=self._function_name + "->over",
            evaluate_output_names=self._evaluate_output_names,
            alias_output_names=self._alias_output_names,
            returns_scalar=False,
            backend_version=self._backend_version,
            version=self._version,
            kwargs={**self._kwargs, "keys": keys},
        )

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        def func(_input: dx.Series, dtype: DType | type[DType]) -> dx.Series:
            dtype = narwhals_to_native_dtype(dtype, self._version)
            return _input.astype(dtype)

        return self._from_call(
            func, "cast", dtype=dtype, returns_scalar=self._returns_scalar
        )

    def is_finite(self: Self) -> Self:
        import dask.array as da

        return self._from_call(
            da.isfinite,
            "is_finite",
            returns_scalar=self._returns_scalar,
        )

    @property
    def str(self: Self) -> DaskExprStringNamespace:
        return DaskExprStringNamespace(self)

    @property
    def dt(self: Self) -> DaskExprDateTimeNamespace:
        return DaskExprDateTimeNamespace(self)

    @property
    def name(self: Self) -> DaskExprNameNamespace:
        return DaskExprNameNamespace(self)
