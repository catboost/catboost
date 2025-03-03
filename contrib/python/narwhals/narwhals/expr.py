from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import Iterable
from typing import Literal
from typing import Mapping
from typing import Sequence

from narwhals._expression_parsing import extract_compliant
from narwhals._expression_parsing import operation_aggregates
from narwhals._expression_parsing import operation_changes_length
from narwhals._expression_parsing import operation_is_order_dependent
from narwhals.dtypes import _validate_dtype
from narwhals.exceptions import LengthChangingExprError
from narwhals.expr_cat import ExprCatNamespace
from narwhals.expr_dt import ExprDateTimeNamespace
from narwhals.expr_list import ExprListNamespace
from narwhals.expr_name import ExprNameNamespace
from narwhals.expr_str import ExprStringNamespace
from narwhals.utils import _validate_rolling_arguments
from narwhals.utils import flatten
from narwhals.utils import issue_deprecation_warning

if TYPE_CHECKING:
    from typing import TypeVar

    from typing_extensions import Concatenate
    from typing_extensions import ParamSpec
    from typing_extensions import Self

    from narwhals.dtypes import DType
    from narwhals.typing import CompliantExpr
    from narwhals.typing import CompliantNamespace
    from narwhals.typing import IntoExpr

    PS = ParamSpec("PS")
    R = TypeVar("R")


class Expr:
    def __init__(
        self: Self,
        to_compliant_expr: Callable[[Any], Any],
        is_order_dependent: bool,  # noqa: FBT001
        changes_length: bool,  # noqa: FBT001
        aggregates: bool,  # noqa: FBT001
    ) -> None:
        # callable from CompliantNamespace to CompliantExpr
        self._to_compliant_expr = to_compliant_expr
        self._is_order_dependent = is_order_dependent
        self._changes_length = changes_length
        self._aggregates = aggregates

    def __repr__(self: Self) -> str:
        return (
            "Narwhals Expr\n"
            f"is_order_dependent: {self._is_order_dependent}\n"
            f"changes_length: {self._changes_length}\n"
            f"aggregates: {self._aggregates}"
        )

    def _taxicab_norm(self: Self) -> Self:
        # This is just used to test out the stable api feature in a realistic-ish way.
        # It's not intended to be used.
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).abs().sum(),
            self._is_order_dependent,
            self._changes_length,
            self._aggregates,
        )

    # --- convert ---
    def alias(self: Self, name: str) -> Self:
        """Rename the expression.

        Arguments:
            name: The new name.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2], "b": [4, 5]})
            >>> df = nw.from_native(df_native)
            >>> df.select((nw.col("b") + 10).alias("c"))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |          c       |
            |      0  14       |
            |      1  15       |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).alias(name),
            is_order_dependent=self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def pipe(
        self: Self,
        function: Callable[Concatenate[Self, PS], R],
        *args: PS.args,
        **kwargs: PS.kwargs,
    ) -> R:
        """Pipe function call.

        Arguments:
            function: Function to apply.
            args: Positional arguments to pass to function.
            kwargs: Keyword arguments to pass to function.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3, 4]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_piped=nw.col("a").pipe(lambda x: x + 1))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |     a  a_piped   |
            |  0  1        2   |
            |  1  2        3   |
            |  2  3        4   |
            |  3  4        5   |
            └──────────────────┘
        """
        return function(self, *args, **kwargs)

    def cast(self: Self, dtype: DType | type[DType]) -> Self:
        """Redefine an object's data type.

        Arguments:
            dtype: Data type that the object will be cast into.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"foo": [1, 2, 3], "bar": [6.0, 7.0, 8.0]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("foo").cast(nw.Float32), nw.col("bar").cast(nw.UInt8))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |      foo  bar    |
            |   0  1.0    6    |
            |   1  2.0    7    |
            |   2  3.0    8    |
            └──────────────────┘
        """
        _validate_dtype(dtype)
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cast(dtype),
            is_order_dependent=self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    # --- binary ---
    def __eq__(self: Self, other: object) -> Self:  # type: ignore[override]
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__eq__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __ne__(self: Self, other: object) -> Self:  # type: ignore[override]
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__ne__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __and__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__and__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __rand__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__and__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __or__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__or__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __ror__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__or__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __add__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__add__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __radd__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__add__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __sub__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__sub__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __rsub__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__sub__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __truediv__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__truediv__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __rtruediv__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__truediv__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __mul__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__mul__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __rmul__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__mul__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __le__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__le__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __lt__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__lt__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __gt__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__gt__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __ge__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__ge__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __pow__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__pow__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __rpow__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__pow__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __floordiv__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__floordiv__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __rfloordiv__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__floordiv__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __mod__(self: Self, other: Any) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__mod__(
                extract_compliant(plx, other, parse_column_name_as_expr=False)
            ),
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    def __rmod__(self: Self, other: Any) -> Self:
        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            return plx.lit(
                extract_compliant(plx, other, parse_column_name_as_expr=False), dtype=None
            ).__mod__(extract_compliant(plx, self, parse_column_name_as_expr=False))

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(self, other),
            changes_length=operation_changes_length(self, other),
            aggregates=operation_aggregates(self, other),
        )

    # --- unary ---
    def __invert__(self: Self) -> Self:
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).__invert__(),
            is_order_dependent=self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def any(self: Self) -> Self:
        """Return whether any of the values in the column are `True`.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [True, False], "b": [True, True]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").any())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a     b   |
            |  0  True  True   |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).any(),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def all(self: Self) -> Self:
        """Return whether all values in the column are `True`.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [True, False], "b": [True, True]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").all())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |         a     b  |
            |  0  False  True  |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).all(),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def ewm_mean(
        self: Self,
        *,
        com: float | None = None,
        span: float | None = None,
        half_life: float | None = None,
        alpha: float | None = None,
        adjust: bool = True,
        min_samples: int = 1,
        ignore_nulls: bool = False,
    ) -> Self:
        r"""Compute exponentially-weighted moving average.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        Arguments:
            com: Specify decay in terms of center of mass, $\gamma$, with <br> $\alpha = \frac{1}{1+\gamma}\forall\gamma\geq0$
            span: Specify decay in terms of span, $\theta$, with <br> $\alpha = \frac{2}{\theta + 1} \forall \theta \geq 1$
            half_life: Specify decay in terms of half-life, $\tau$, with <br> $\alpha = 1 - \exp \left\{ \frac{ -\ln(2) }{ \tau } \right\} \forall \tau > 0$
            alpha: Specify smoothing factor alpha directly, $0 < \alpha \leq 1$.
            adjust: Divide by decaying adjustment factor in beginning periods to account for imbalance in relative weightings

                - When `adjust=True` (the default) the EW function is calculated
                  using weights $w_i = (1 - \alpha)^i$
                - When `adjust=False` the EW function is calculated recursively by
                  $$
                  y_0=x_0
                  $$
                  $$
                  y_t = (1 - \alpha)y_{t - 1} + \alpha x_t
                  $$
            min_samples: Minimum number of observations in window required to have a value, (otherwise result is null).
            ignore_nulls: Ignore missing values when calculating weights.

                - When `ignore_nulls=False` (default), weights are based on absolute
                  positions.
                  For example, the weights of $x_0$ and $x_2$ used in
                  calculating the final weighted average of $[x_0, None, x_2]$ are
                  $(1-\alpha)^2$ and $1$ if `adjust=True`, and
                  $(1-\alpha)^2$ and $\alpha$ if `adjust=False`.
                - When `ignore_nulls=True`, weights are based
                  on relative positions. For example, the weights of
                  $x_0$ and $x_2$ used in calculating the final weighted
                  average of $[x_0, None, x_2]$ are
                  $1-\alpha$ and $1$ if `adjust=True`,
                  and $1-\alpha$ and $\alpha$ if `adjust=False`.

        Returns:
            Expr

        Examples:
            >>> import pandas as pd
            >>> import polars as pl
            >>> import narwhals as nw
            >>> from narwhals.typing import IntoFrameT
            >>>
            >>> data = {"a": [1, 2, 3]}
            >>> df_pd = pd.DataFrame(data)
            >>> df_pl = pl.DataFrame(data)

            We define a library agnostic function:

            >>> def agnostic_ewm_mean(df_native: IntoFrameT) -> IntoFrameT:
            ...     df = nw.from_native(df_native)
            ...     return df.select(
            ...         nw.col("a").ewm_mean(com=1, ignore_nulls=False)
            ...     ).to_native()

            We can then pass either pandas or Polars to `agnostic_ewm_mean`:

            >>> agnostic_ewm_mean(df_pd)
                      a
            0  1.000000
            1  1.666667
            2  2.428571

            >>> agnostic_ewm_mean(df_pl)  # doctest: +NORMALIZE_WHITESPACE
            shape: (3, 1)
            ┌──────────┐
            │ a        │
            │ ---      │
            │ f64      │
            ╞══════════╡
            │ 1.0      │
            │ 1.666667 │
            │ 2.428571 │
            └──────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).ewm_mean(
                com=com,
                span=span,
                half_life=half_life,
                alpha=alpha,
                adjust=adjust,
                min_samples=min_samples,
                ignore_nulls=ignore_nulls,
            ),
            is_order_dependent=self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def mean(self: Self) -> Self:
        """Get mean value.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [-1, 0, 1], "b": [2, 4, 6]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").mean())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a    b    |
            |   0  0.0  4.0    |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).mean(),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def median(self: Self) -> Self:
        """Get median value.

        Returns:
            A new expression.

        Notes:
            Results might slightly differ across backends due to differences in the underlying algorithms used to compute the median.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").median())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a    b    |
            |   0  3.0  4.0    |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).median(),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def std(self: Self, *, ddof: int = 1) -> Self:
        """Get standard deviation.

        Arguments:
            ddof: "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
                where N represents the number of elements. By default ddof is 1.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [20, 25, 60], "b": [1.5, 1, -1.4]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").std(ddof=0))
            ┌─────────────────────┐
            | Narwhals DataFrame  |
            |---------------------|
            |          a         b|
            |0  17.79513  1.265789|
            └─────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).std(ddof=ddof),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def var(self: Self, *, ddof: int = 1) -> Self:
        """Get variance.

        Arguments:
            ddof: "Delta Degrees of Freedom": the divisor used in the calculation is N - ddof,
                     where N represents the number of elements. By default ddof is 1.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [20, 25, 60], "b": [1.5, 1, -1.4]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").var(ddof=0))
            ┌───────────────────────┐
            |  Narwhals DataFrame   |
            |-----------------------|
            |            a         b|
            |0  316.666667  1.602222|
            └───────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).var(ddof=ddof),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def map_batches(
        self: Self,
        function: Callable[[Any], Self],
        return_dtype: DType | None = None,
    ) -> Self:
        """Apply a custom python function to a whole Series or sequence of Series.

        The output of this custom function is presumed to be either a Series,
        or a NumPy array (in which case it will be automatically converted into
        a Series).

        Arguments:
            function: Function to apply to Series.
            return_dtype: Dtype of the output Series.
                If not set, the dtype will be inferred based on the first non-null value
                that is returned by the function.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     nw.col("a", "b")
            ...     .map_batches(lambda s: s.to_numpy() + 1, return_dtype=nw.Float64)
            ...     .name.suffix("_mapped")
            ... )
            ┌───────────────────────────┐
            |    Narwhals DataFrame     |
            |---------------------------|
            |   a  b  a_mapped  b_mapped|
            |0  1  4       2.0       5.0|
            |1  2  5       3.0       6.0|
            |2  3  6       4.0       7.0|
            └───────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).map_batches(
                function=function, return_dtype=return_dtype
            ),
            # safest assumptions
            is_order_dependent=True,
            changes_length=True,
            aggregates=False,
        )

    def skew(self: Self) -> Self:
        """Calculate the sample skewness of a column.

        Returns:
            An expression representing the sample skewness of the column.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 2, 10, 100]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").skew())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |      a         b |
            | 0  0.0  1.472427 |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).skew(),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def sum(self: Self) -> Expr:
        """Return the sum value.

        Returns:
            A new expression.

        Examples:
            >>> import duckdb
            >>> import narwhals as nw
            >>> df_native = duckdb.sql("SELECT * FROM VALUES (5, 50), (10, 100) df(a, b)")
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").sum())
            ┌───────────────────┐
            |Narwhals LazyFrame |
            |-------------------|
            |┌────────┬────────┐|
            |│   a    │   b    │|
            |│ int128 │ int128 │|
            |├────────┼────────┤|
            |│     15 │    150 │|
            |└────────┴────────┘|
            └───────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).sum(),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def min(self: Self) -> Self:
        """Returns the minimum value(s) from a column(s).

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2], "b": [4, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.min("a", "b"))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a  b      |
            |     0  1  3      |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).min(),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def max(self: Self) -> Self:
        """Returns the maximum value(s) from a column(s).

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [10, 20], "b": [50, 100]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.max("a", "b"))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a    b    |
            |    0  20  100    |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).max(),
            is_order_dependent=self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def arg_min(self: Self) -> Self:
        """Returns the index of the minimum value.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [10, 20], "b": [150, 100]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").arg_min().name.suffix("_arg_min"))
            ┌───────────────────────┐
            |  Narwhals DataFrame   |
            |-----------------------|
            |   a_arg_min  b_arg_min|
            |0          0          1|
            └───────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).arg_min(),
            is_order_dependent=True,
            changes_length=False,
            aggregates=True,
        )

    def arg_max(self: Self) -> Self:
        """Returns the index of the maximum value.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [10, 20], "b": [150, 100]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").arg_max().name.suffix("_arg_max"))
            ┌───────────────────────┐
            |  Narwhals DataFrame   |
            |-----------------------|
            |   a_arg_max  b_arg_max|
            |0          1          0|
            └───────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).arg_max(),
            is_order_dependent=True,
            changes_length=False,
            aggregates=True,
        )

    def count(self: Self) -> Self:
        """Returns the number of non-null elements in the column.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3], "b": [None, 4, 4]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.all().count())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a  b      |
            |     0  3  2      |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).count(),
            self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def n_unique(self: Self) -> Self:
        """Returns count of unique values.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 1, 3, 3, 5]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").n_unique())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a  b      |
            |     0  5  3      |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).n_unique(),
            self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def unique(self: Self) -> Self:
        """Return unique values of this expression.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 1, 3, 5, 5], "b": [2, 4, 4, 6, 6]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").unique().sum())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a   b     |
            |     0  9  12     |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).unique(),
            self._is_order_dependent,
            changes_length=True,
            aggregates=self._aggregates,
        )

    def abs(self: Self) -> Self:
        """Return absolute value of each element.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, -2], "b": [-3, 4]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(nw.col("a", "b").abs().name.suffix("_abs"))
            ┌─────────────────────┐
            | Narwhals DataFrame  |
            |---------------------|
            |   a  b  a_abs  b_abs|
            |0  1 -3      1      3|
            |1 -2  4      2      4|
            └─────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).abs(),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def cum_sum(self: Self, *, reverse: bool = False) -> Self:
        """Return cumulative sum.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 1, 3, 5, 5], "b": [2, 4, 4, 6, 6]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_cum_sum=nw.col("a").cum_sum())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |   a  b  a_cum_sum|
            |0  1  2          1|
            |1  1  4          2|
            |2  3  4          5|
            |3  5  6         10|
            |4  5  6         15|
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_sum(reverse=reverse),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def diff(self: Self) -> Self:
        """Returns the difference between each element and the previous one.

        Returns:
            A new expression.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to calculate
            the diff and fill missing values with `0` in a Int64 column, you could
            do:

                nw.col("a").diff().fill_null(0).cast(nw.Int64)

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 1, 3, 5, 5]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_diff=nw.col("a").diff())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            | shape: (5, 2)    |
            | ┌─────┬────────┐ |
            | │ a   ┆ a_diff │ |
            | │ --- ┆ ---    │ |
            | │ i64 ┆ i64    │ |
            | ╞═════╪════════╡ |
            | │ 1   ┆ null   │ |
            | │ 1   ┆ 0      │ |
            | │ 3   ┆ 2      │ |
            | │ 5   ┆ 2      │ |
            | │ 5   ┆ 0      │ |
            | └─────┴────────┘ |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).diff(),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def shift(self: Self, n: int) -> Self:
        """Shift values by `n` positions.

        Arguments:
            n: Number of positions to shift values by.

        Returns:
            A new expression.

        Notes:
            pandas may change the dtype here, for example when introducing missing
            values in an integer column. To ensure, that the dtype doesn't change,
            you may want to use `fill_null` and `cast`. For example, to shift
            and fill missing values with `0` in a Int64 column, you could
            do:

                nw.col("a").shift(1).fill_null(0).cast(nw.Int64)

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [1, 1, 3, 5, 5]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_shift=nw.col("a").shift(n=1))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |shape: (5, 2)     |
            |┌─────┬─────────┐ |
            |│ a   ┆ a_shift │ |
            |│ --- ┆ ---     │ |
            |│ i64 ┆ i64     │ |
            |╞═════╪═════════╡ |
            |│ 1   ┆ null    │ |
            |│ 1   ┆ 1       │ |
            |│ 3   ┆ 1       │ |
            |│ 5   ┆ 3       │ |
            |│ 5   ┆ 5       │ |
            |└─────┴─────────┘ |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).shift(n),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def replace_strict(
        self: Self,
        old: Sequence[Any] | Mapping[Any, Any],
        new: Sequence[Any] | None = None,
        *,
        return_dtype: DType | type[DType] | None = None,
    ) -> Self:
        """Replace all values by different values.

        This function must replace all non-null input values (else it raises an error).

        Arguments:
            old: Sequence of values to replace. It also accepts a mapping of values to
                their replacement as syntactic sugar for
                `replace_all(old=list(mapping.keys()), new=list(mapping.values()))`.
            new: Sequence of values to replace by. Length must match the length of `old`.
            return_dtype: The data type of the resulting expression. If set to `None`
                (default), the data type is determined automatically based on the other
                inputs.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [3, 0, 1, 2]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     b=nw.col("a").replace_strict(
            ...         [0, 1, 2, 3],
            ...         ["zero", "one", "two", "three"],
            ...         return_dtype=nw.String,
            ...     )
            ... )
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |      a      b    |
            |   0  3  three    |
            |   1  0   zero    |
            |   2  1    one    |
            |   3  2    two    |
            └──────────────────┘
        """
        if new is None:
            if not isinstance(old, Mapping):
                msg = "`new` argument is required if `old` argument is not a Mapping type"
                raise TypeError(msg)

            new = list(old.values())
            old = list(old.keys())

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).replace_strict(
                old, new, return_dtype=return_dtype
            ),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def sort(self: Self, *, descending: bool = False, nulls_last: bool = False) -> Self:
        """Sort this column. Place null values first.

        !!! warning
            `Expr.sort` is deprecated and will be removed in a future version.
            Hint: instead of `df.select(nw.col('a').sort())`, use
            `df.select(nw.col('a')).sort()` instead.
            Note: this will remain available in `narwhals.stable.v1`.
            See [stable api](../backcompat.md/) for more information.

        Arguments:
            descending: Sort in descending order.
            nulls_last: Place null values last instead of first.

        Returns:
            A new expression.
        """
        msg = (
            "`Expr.sort` is deprecated and will be removed in a future version.\n\n"
            "Hint: instead of `df.select(nw.col('a').sort())`, use `df.select(nw.col('a')).sort()`.\n\n"
            "Note: this will remain available in `narwhals.stable.v1`.\n"
            "See https://narwhals-dev.github.io/narwhals/backcompat/ for more information.\n"
        )
        issue_deprecation_warning(msg, _version="1.22.0")
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).sort(
                descending=descending, nulls_last=nulls_last
            ),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    # --- transform ---
    def is_between(
        self: Self,
        lower_bound: Any | IntoExpr,
        upper_bound: Any | IntoExpr,
        closed: Literal["left", "right", "none", "both"] = "both",
    ) -> Self:
        """Check if this expression is between the given lower and upper bounds.

        Arguments:
            lower_bound: Lower bound value. String literals are interpreted as column names.
            upper_bound: Upper bound value. String literals are interpreted as column names.
            closed: Define which sides of the interval are closed (inclusive).

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3, 4, 5]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(b=nw.col("a").is_between(2, 4, "right"))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |      a      b    |
            |   0  1  False    |
            |   1  2  False    |
            |   2  3   True    |
            |   3  4   True    |
            |   4  5  False    |
            └──────────────────┘
        """

        def func(plx: CompliantNamespace[Any]) -> CompliantExpr[Any]:
            lb = extract_compliant(plx, lower_bound, parse_column_name_as_expr=True)
            ub = extract_compliant(plx, upper_bound, parse_column_name_as_expr=True)
            expr = self._to_compliant_expr(plx)
            if closed == "left":
                return (expr >= lb) & (expr < ub)  # type: ignore[no-any-return]
            elif closed == "right":
                return (expr > lb) & (expr <= ub)  # type: ignore[no-any-return]
            elif closed == "none":
                return (expr > lb) & (expr < ub)  # type: ignore[no-any-return]
            return (expr >= lb) & (expr <= ub)  # type: ignore[no-any-return]

        return self.__class__(
            func,
            is_order_dependent=operation_is_order_dependent(
                self, lower_bound, upper_bound
            ),
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def is_in(self: Self, other: Any) -> Self:
        """Check if elements of this expression are present in the other iterable.

        Arguments:
            other: iterable

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 9, 10]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(b=nw.col("a").is_in([1, 2]))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |       a      b   |
            |   0   1   True   |
            |   1   2   True   |
            |   2   9  False   |
            |   3  10  False   |
            └──────────────────┘
        """
        if isinstance(other, Iterable) and not isinstance(other, (str, bytes)):
            return self.__class__(
                lambda plx: self._to_compliant_expr(plx).is_in(
                    extract_compliant(plx, other, parse_column_name_as_expr=False)
                ),
                self._is_order_dependent,
                changes_length=self._changes_length,
                aggregates=self._aggregates,
            )
        else:
            msg = "Narwhals `is_in` doesn't accept expressions as an argument, as opposed to Polars. You should provide an iterable instead."
            raise NotImplementedError(msg)

    def filter(self: Self, *predicates: Any) -> Self:
        """Filters elements based on a condition, returning a new expression.

        Arguments:
            predicates: Conditions to filter by (which get ANDed together).

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame(
            ...     {"a": [2, 3, 4, 5, 6, 7], "b": [10, 11, 12, 13, 14, 15]}
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.select(
            ...     nw.col("a").filter(nw.col("a") > 4),
            ...     nw.col("b").filter(nw.col("b") < 13),
            ... )
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a   b     |
            |     3  5  10     |
            |     4  6  11     |
            |     5  7  12     |
            └──────────────────┘
        """
        flat_predicates = flatten(predicates)
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).filter(
                *[
                    extract_compliant(plx, pred, parse_column_name_as_expr=True)
                    for pred in flat_predicates
                ],
            ),
            is_order_dependent=operation_is_order_dependent(*flat_predicates),
            changes_length=True,
            aggregates=self._aggregates,
        )

    def is_null(self: Self) -> Self:
        """Returns a boolean Series indicating which values are null.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.

        Examples:
            >>> import duckdb
            >>> import narwhals as nw
            >>> df_native = duckdb.sql(
            ...     "SELECT * FROM VALUES (null, CAST('NaN' AS DOUBLE)), (2, 2.) df(a, b)"
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     a_is_null=nw.col("a").is_null(), b_is_null=nw.col("b").is_null()
            ... )
            ┌──────────────────────────────────────────┐
            |            Narwhals LazyFrame            |
            |------------------------------------------|
            |┌───────┬────────┬───────────┬───────────┐|
            |│   a   │   b    │ a_is_null │ b_is_null │|
            |│ int32 │ double │  boolean  │  boolean  │|
            |├───────┼────────┼───────────┼───────────┤|
            |│  NULL │    nan │ true      │ false     │|
            |│     2 │    2.0 │ false     │ false     │|
            |└───────┴────────┴───────────┴───────────┘|
            └──────────────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).is_null(),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def is_nan(self: Self) -> Self:
        """Indicate which values are NaN.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.

        Examples:
            >>> import duckdb
            >>> import narwhals as nw
            >>> df_native = duckdb.sql(
            ...     "SELECT * FROM VALUES (null, CAST('NaN' AS DOUBLE)), (2, 2.) df(a, b)"
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     a_is_nan=nw.col("a").is_nan(), b_is_nan=nw.col("b").is_nan()
            ... )
            ┌────────────────────────────────────────┐
            |           Narwhals LazyFrame           |
            |----------------------------------------|
            |┌───────┬────────┬──────────┬──────────┐|
            |│   a   │   b    │ a_is_nan │ b_is_nan │|
            |│ int32 │ double │ boolean  │ boolean  │|
            |├───────┼────────┼──────────┼──────────┤|
            |│  NULL │    nan │ NULL     │ true     │|
            |│     2 │    2.0 │ false    │ false    │|
            |└───────┴────────┴──────────┴──────────┘|
            └────────────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).is_nan(),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def arg_true(self: Self) -> Self:
        """Find elements where boolean expression is True.

        Returns:
            A new expression.
        """
        msg = (
            "`Expr.arg_true` is deprecated and will be removed in a future version.\n\n"
            "Note: this will remain available in `narwhals.stable.v1`.\n"
            "See https://narwhals-dev.github.io/narwhals/backcompat/ for more information.\n"
        )
        issue_deprecation_warning(msg, _version="1.23.0")
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).arg_true(),
            is_order_dependent=True,
            changes_length=True,
            aggregates=self._aggregates,
        )

    def fill_null(
        self: Self,
        value: Any | None = None,
        strategy: Literal["forward", "backward"] | None = None,
        limit: int | None = None,
    ) -> Self:
        """Fill null values with given value.

        Arguments:
            value: Value used to fill null values.
            strategy: Strategy used to fill null values.
            limit: Number of consecutive null values to fill when using the 'forward' or 'backward' strategy.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame(
            ...     {
            ...         "a": [2, None, None, 3],
            ...         "b": [2.0, float("nan"), float("nan"), 3.0],
            ...     }
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     nw.col("a", "b").fill_null(0).name.suffix("_nulls_filled")
            ... )
            ┌────────────────────────────────────────────────┐
            |               Narwhals DataFrame               |
            |------------------------------------------------|
            |shape: (4, 4)                                   |
            |┌──────┬─────┬────────────────┬────────────────┐|
            |│ a    ┆ b   ┆ a_nulls_filled ┆ b_nulls_filled │|
            |│ ---  ┆ --- ┆ ---            ┆ ---            │|
            |│ i64  ┆ f64 ┆ i64            ┆ f64            │|
            |╞══════╪═════╪════════════════╪════════════════╡|
            |│ 2    ┆ 2.0 ┆ 2              ┆ 2.0            │|
            |│ null ┆ NaN ┆ 0              ┆ NaN            │|
            |│ null ┆ NaN ┆ 0              ┆ NaN            │|
            |│ 3    ┆ 3.0 ┆ 3              ┆ 3.0            │|
            |└──────┴─────┴────────────────┴────────────────┘|
            └────────────────────────────────────────────────┘

            Using a strategy:

            >>> df.with_columns(
            ...     nw.col("a", "b")
            ...     .fill_null(strategy="forward", limit=1)
            ...     .name.suffix("_nulls_forward_filled")
            ... )
            ┌────────────────────────────────────────────────────────────────┐
            |                       Narwhals DataFrame                       |
            |----------------------------------------------------------------|
            |shape: (4, 4)                                                   |
            |┌──────┬─────┬────────────────────────┬────────────────────────┐|
            |│ a    ┆ b   ┆ a_nulls_forward_filled ┆ b_nulls_forward_filled │|
            |│ ---  ┆ --- ┆ ---                    ┆ ---                    │|
            |│ i64  ┆ f64 ┆ i64                    ┆ f64                    │|
            |╞══════╪═════╪════════════════════════╪════════════════════════╡|
            |│ 2    ┆ 2.0 ┆ 2                      ┆ 2.0                    │|
            |│ null ┆ NaN ┆ 2                      ┆ NaN                    │|
            |│ null ┆ NaN ┆ null                   ┆ NaN                    │|
            |│ 3    ┆ 3.0 ┆ 3                      ┆ 3.0                    │|
            |└──────┴─────┴────────────────────────┴────────────────────────┘|
            └────────────────────────────────────────────────────────────────┘
        """
        if value is not None and strategy is not None:
            msg = "cannot specify both `value` and `strategy`"
            raise ValueError(msg)
        if value is None and strategy is None:
            msg = "must specify either a fill `value` or `strategy`"
            raise ValueError(msg)
        if strategy is not None and strategy not in {"forward", "backward"}:
            msg = f"strategy not supported: {strategy}"
            raise ValueError(msg)
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).fill_null(
                value=value, strategy=strategy, limit=limit
            ),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    # --- partial reduction ---
    def drop_nulls(self: Self) -> Self:
        """Drop null values.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [2.0, 4.0, float("nan"), 3.0, None, 5.0]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a").drop_nulls())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |  shape: (5, 1)   |
            |  ┌─────┐         |
            |  │ a   │         |
            |  │ --- │         |
            |  │ f64 │         |
            |  ╞═════╡         |
            |  │ 2.0 │         |
            |  │ 4.0 │         |
            |  │ NaN │         |
            |  │ 3.0 │         |
            |  │ 5.0 │         |
            |  └─────┘         |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).drop_nulls(),
            self._is_order_dependent,
            changes_length=True,
            aggregates=self._aggregates,
        )

    def sample(
        self: Self,
        n: int | None = None,
        *,
        fraction: float | None = None,
        with_replacement: bool = False,
        seed: int | None = None,
    ) -> Self:
        """Sample randomly from this expression.

        !!! warning
            `Expr.sample` is deprecated and will be removed in a future version.
            Hint: instead of `df.select(nw.col('a').sample())`, use
            `df.select(nw.col('a')).sample()` instead.
            Note: this will remain available in `narwhals.stable.v1`.
            See [stable api](../backcompat.md/) for more information.

        Arguments:
            n: Number of items to return. Cannot be used with fraction.
            fraction: Fraction of items to return. Cannot be used with n.
            with_replacement: Allow values to be sampled more than once.
            seed: Seed for the random number generator. If set to None (default), a random
                seed is generated for each sample operation.

        Returns:
            A new expression.
        """
        msg = (
            "`Expr.sample` is deprecated and will be removed in a future version.\n\n"
            "Hint: instead of `df.select(nw.col('a').sample())`, use `df.select(nw.col('a')).sample()`.\n\n"
            "Note: this will remain available in `narwhals.stable.v1`.\n"
            "See https://narwhals-dev.github.io/narwhals/backcompat/ for more information.\n"
        )
        issue_deprecation_warning(msg, _version="1.22.0")
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).sample(
                n, fraction=fraction, with_replacement=with_replacement, seed=seed
            ),
            self._is_order_dependent,
            changes_length=True,
            aggregates=self._aggregates,
        )

    def over(self: Self, *keys: str | Iterable[str]) -> Self:
        """Compute expressions over the given groups.

        Arguments:
            keys: Names of columns to compute window expression over.
                  Must be names of columns, as opposed to expressions -
                  so, this is a bit less flexible than Polars' `Expr.over`.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 4], "b": ["x", "x", "y"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_min_per_group=nw.col("a").min().over("b"))
            ┌────────────────────────┐
            |   Narwhals DataFrame   |
            |------------------------|
            |   a  b  a_min_per_group|
            |0  1  x                1|
            |1  2  x                1|
            |2  4  y                4|
            └────────────────────────┘

            Cumulative operations are also supported, but (currently) only for
            pandas and Polars:

            >>> df.with_columns(a_cum_sum_per_group=nw.col("a").cum_sum().over("b"))
            ┌────────────────────────────┐
            |     Narwhals DataFrame     |
            |----------------------------|
            |   a  b  a_cum_sum_per_group|
            |0  1  x                    1|
            |1  2  x                    3|
            |2  4  y                    4|
            └────────────────────────────┘
        """
        if self._changes_length:
            msg = "`.over()` can not be used for expressions which change length."
            raise LengthChangingExprError(msg)
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).over(flatten(keys)),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def is_duplicated(self: Self) -> Self:
        r"""Return a boolean mask indicating duplicated values.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(nw.all().is_duplicated().name.suffix("_is_duplicated"))
            ┌─────────────────────────────────────────┐
            |           Narwhals DataFrame            |
            |-----------------------------------------|
            |   a  b  a_is_duplicated  b_is_duplicated|
            |0  1  a             True             True|
            |1  2  a            False             True|
            |2  3  b            False            False|
            |3  1  c             True            False|
            └─────────────────────────────────────────┘
        """
        return ~self.is_unique()

    def is_unique(self: Self) -> Self:
        r"""Return a boolean mask indicating unique values.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(nw.all().is_unique().name.suffix("_is_unique"))
            ┌─────────────────────────────────┐
            |       Narwhals DataFrame        |
            |---------------------------------|
            |   a  b  a_is_unique  b_is_unique|
            |0  1  a        False        False|
            |1  2  a         True        False|
            |2  3  b         True         True|
            |3  1  c        False         True|
            └─────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).is_unique(),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def null_count(self: Self) -> Self:
        r"""Count null values.

        Returns:
            A new expression.

        Notes:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame(
            ...     {"a": [1, 2, None, 1], "b": ["a", None, "b", None]}
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.all().null_count())
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a  b      |
            |     0  1  2      |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).null_count(),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def is_first_distinct(self: Self) -> Self:
        r"""Return a boolean mask indicating the first occurrence of each distinct value.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     nw.all().is_first_distinct().name.suffix("_is_first_distinct")
            ... )
            ┌─────────────────────────────────────────────────┐
            |               Narwhals DataFrame                |
            |-------------------------------------------------|
            |   a  b  a_is_first_distinct  b_is_first_distinct|
            |0  1  a                 True                 True|
            |1  2  a                 True                False|
            |2  3  b                 True                 True|
            |3  1  c                False                 True|
            └─────────────────────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).is_first_distinct(),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def is_last_distinct(self: Self) -> Self:
        r"""Return a boolean mask indicating the last occurrence of each distinct value.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3, 1], "b": ["a", "a", "b", "c"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     nw.all().is_last_distinct().name.suffix("_is_last_distinct")
            ... )
            ┌───────────────────────────────────────────────┐
            |              Narwhals DataFrame               |
            |-----------------------------------------------|
            |   a  b  a_is_last_distinct  b_is_last_distinct|
            |0  1  a               False               False|
            |1  2  a                True                True|
            |2  3  b                True                True|
            |3  1  c                True                True|
            └───────────────────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).is_last_distinct(),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def quantile(
        self: Self,
        quantile: float,
        interpolation: Literal["nearest", "higher", "lower", "midpoint", "linear"],
    ) -> Self:
        r"""Get quantile value.

        Arguments:
            quantile: Quantile between 0.0 and 1.0.
            interpolation: Interpolation method.

        Returns:
            A new expression.

        Note:
            - pandas and Polars may have implementation differences for a given interpolation method.
            - [dask](https://docs.dask.org/en/stable/generated/dask.dataframe.Series.quantile.html) has
                its own method to approximate quantile and it doesn't implement 'nearest', 'higher',
                'lower', 'midpoint' as interpolation method - use 'linear' which is closest to the
                native 'dask' - method.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame(
            ...     {"a": list(range(50)), "b": list(range(50, 100))}
            ... )
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a", "b").quantile(0.5, interpolation="linear"))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |        a     b   |
            |  0  24.5  74.5   |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).quantile(quantile, interpolation),
            self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def head(self: Self, n: int = 10) -> Self:
        r"""Get the first `n` rows.

        !!! warning
            `Expr.head` is deprecated and will be removed in a future version.
            Hint: instead of `df.select(nw.col('a').head())`, use
            `df.select(nw.col('a')).head()` instead.
            Note: this will remain available in `narwhals.stable.v1`.
            See [stable api](../backcompat.md/) for more information.

        Arguments:
            n: Number of rows to return.

        Returns:
            A new expression.
        """
        msg = (
            "`Expr.head` is deprecated and will be removed in a future version.\n\n"
            "Hint: instead of `df.select(nw.col('a').head())`, use `df.select(nw.col('a')).head()`.\n\n"
            "Note: this will remain available in `narwhals.stable.v1`.\n"
            "See https://narwhals-dev.github.io/narwhals/backcompat/ for more information.\n"
        )
        issue_deprecation_warning(msg, _version="1.22.0")
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).head(n),
            is_order_dependent=True,
            changes_length=True,
            aggregates=self._aggregates,
        )

    def tail(self: Self, n: int = 10) -> Self:
        r"""Get the last `n` rows.

        !!! warning
            `Expr.tail` is deprecated and will be removed in a future version.
            Hint: instead of `df.select(nw.col('a').tail())`, use
            `df.select(nw.col('a')).tail()` instead.
            Note: this will remain available in `narwhals.stable.v1`.
            See [stable api](../backcompat.md/) for more information.

        Arguments:
            n: Number of rows to return.

        Returns:
            A new expression.
        """
        msg = (
            "`Expr.tail` is deprecated and will be removed in a future version.\n\n"
            "Hint: instead of `df.select(nw.col('a').tail())`, use `df.select(nw.col('a')).tail()`.\n\n"
            "Note: this will remain available in `narwhals.stable.v1`.\n"
            "See https://narwhals-dev.github.io/narwhals/backcompat/ for more information.\n"
        )
        issue_deprecation_warning(msg, _version="1.22.0")
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).tail(n),
            is_order_dependent=True,
            changes_length=True,
            aggregates=self._aggregates,
        )

    def round(self: Self, decimals: int = 0) -> Self:
        r"""Round underlying floating point data by `decimals` digits.

        Arguments:
            decimals: Number of decimals to round by.

        Returns:
            A new expression.


        Notes:
            For values exactly halfway between rounded decimal values pandas behaves differently than Polars and Arrow.

            pandas rounds to the nearest even value (e.g. -0.5 and 0.5 round to 0.0, 1.5 and 2.5 round to 2.0, 3.5 and
            4.5 to 4.0, etc..).

            Polars and Arrow round away from 0 (e.g. -0.5 to -1.0, 0.5 to 1.0, 1.5 to 2.0, 2.5 to 3.0, etc..).

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1.12345, 2.56789, 3.901234]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_rounded=nw.col("a").round(1))
            ┌──────────────────────┐
            |  Narwhals DataFrame  |
            |----------------------|
            |          a  a_rounded|
            |0  1.123450        1.1|
            |1  2.567890        2.6|
            |2  3.901234        3.9|
            └──────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).round(decimals),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def len(self: Self) -> Self:
        r"""Return the number of elements in the column.

        Null values count towards the total.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": ["x", "y", "z"], "b": [1, 2, 1]})
            >>> df = nw.from_native(df_native)
            >>> df.select(
            ...     nw.col("a").filter(nw.col("b") == 1).len().alias("a1"),
            ...     nw.col("a").filter(nw.col("b") == 2).len().alias("a2"),
            ... )
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |       a1  a2     |
            |    0   2   1     |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).len(),
            self._is_order_dependent,
            changes_length=False,
            aggregates=True,
        )

    def gather_every(self: Self, n: int, offset: int = 0) -> Self:
        r"""Take every nth value in the Series and return as new Series.

        !!! warning
            `Expr.gather_every` is deprecated and will be removed in a future version.
            Hint: instead of `df.select(nw.col('a').gather_every())`, use
            `df.select(nw.col('a')).gather_every()` instead.
            Note: this will remain available in `narwhals.stable.v1`.
            See [stable api](../backcompat.md/) for more information.

        Arguments:
            n: Gather every *n*-th row.
            offset: Starting index.

        Returns:
            A new expression.
        """
        msg = (
            "`Expr.gather_every` is deprecated and will be removed in a future version.\n\n"
            "Hint: instead of `df.select(nw.col('a').gather_every())`, use `df.select(nw.col('a')).gather_every()`.\n\n"
            "Note: this will remain available in `narwhals.stable.v1`.\n"
            "See https://narwhals-dev.github.io/narwhals/backcompat/ for more information.\n"
        )
        issue_deprecation_warning(msg, _version="1.22.0")
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).gather_every(n=n, offset=offset),
            is_order_dependent=True,
            changes_length=True,
            aggregates=self._aggregates,
        )

    # need to allow numeric typing
    # TODO @aivanoved: make type alias for numeric type
    def clip(
        self: Self,
        lower_bound: IntoExpr | Any | None = None,
        upper_bound: IntoExpr | Any | None = None,
    ) -> Self:
        r"""Clip values in the Series.

        Arguments:
            lower_bound: Lower bound value. String literals are treated as column names.
            upper_bound: Upper bound value. String literals are treated as column names.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 2, 3]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_clipped=nw.col("a").clip(-1, 3))
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |    a  a_clipped  |
            | 0  1          1  |
            | 1  2          2  |
            | 2  3          3  |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).clip(
                extract_compliant(plx, lower_bound, parse_column_name_as_expr=True),
                extract_compliant(plx, upper_bound, parse_column_name_as_expr=True),
            ),
            is_order_dependent=operation_is_order_dependent(
                self, lower_bound, upper_bound
            ),
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def mode(self: Self) -> Self:
        r"""Compute the most occurring value(s).

        Can return multiple values.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 1, 2, 3], "b": [1, 1, 2, 2]})
            >>> df = nw.from_native(df_native)
            >>> df.select(nw.col("a").mode()).sort("a")
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |          a       |
            |       0  1       |
            └──────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).mode(),
            self._is_order_dependent,
            changes_length=True,
            aggregates=self._aggregates,
        )

    def is_finite(self: Self) -> Self:
        """Returns boolean values indicating which original values are finite.

        Warning:
            pandas handles null values differently from Polars and PyArrow.
            See [null_handling](../pandas_like_concepts/null_handling.md/)
            for reference.
            `is_finite` will return False for NaN and Null's in the Dask and
            pandas non-nullable backend, while for Polars, PyArrow and pandas
            nullable backends null values are kept as such.

        Returns:
            Expression of `Boolean` data type.

        Examples:
            >>> import polars as pl
            >>> import narwhals as nw
            >>> df_native = pl.DataFrame({"a": [float("nan"), float("inf"), 2.0, None]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(a_is_finite=nw.col("a").is_finite())
            ┌──────────────────────┐
            |  Narwhals DataFrame  |
            |----------------------|
            |shape: (4, 2)         |
            |┌──────┬─────────────┐|
            |│ a    ┆ a_is_finite │|
            |│ ---  ┆ ---         │|
            |│ f64  ┆ bool        │|
            |╞══════╪═════════════╡|
            |│ NaN  ┆ false       │|
            |│ inf  ┆ false       │|
            |│ 2.0  ┆ true        │|
            |│ null ┆ null        │|
            |└──────┴─────────────┘|
            └──────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).is_finite(),
            self._is_order_dependent,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def cum_count(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative count of the non-null values in the column.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": ["x", "k", None, "d"]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     nw.col("a").cum_count().alias("a_cum_count"),
            ...     nw.col("a").cum_count(reverse=True).alias("a_cum_count_reverse"),
            ... )
            ┌─────────────────────────────────────────┐
            |           Narwhals DataFrame            |
            |-----------------------------------------|
            |      a  a_cum_count  a_cum_count_reverse|
            |0     x            1                    3|
            |1     k            2                    2|
            |2  None            2                    1|
            |3     d            3                    1|
            └─────────────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_count(reverse=reverse),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def cum_min(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative min of the non-null values in the column.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [3, 1, None, 2]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     nw.col("a").cum_min().alias("a_cum_min"),
            ...     nw.col("a").cum_min(reverse=True).alias("a_cum_min_reverse"),
            ... )
            ┌────────────────────────────────────┐
            |         Narwhals DataFrame         |
            |------------------------------------|
            |     a  a_cum_min  a_cum_min_reverse|
            |0  3.0        3.0                1.0|
            |1  1.0        1.0                1.0|
            |2  NaN        NaN                NaN|
            |3  2.0        1.0                2.0|
            └────────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_min(reverse=reverse),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def cum_max(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative max of the non-null values in the column.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 3, None, 2]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     nw.col("a").cum_max().alias("a_cum_max"),
            ...     nw.col("a").cum_max(reverse=True).alias("a_cum_max_reverse"),
            ... )
            ┌────────────────────────────────────┐
            |         Narwhals DataFrame         |
            |------------------------------------|
            |     a  a_cum_max  a_cum_max_reverse|
            |0  1.0        1.0                3.0|
            |1  3.0        3.0                3.0|
            |2  NaN        NaN                NaN|
            |3  2.0        3.0                2.0|
            └────────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_max(reverse=reverse),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def cum_prod(self: Self, *, reverse: bool = False) -> Self:
        r"""Return the cumulative product of the non-null values in the column.

        Arguments:
            reverse: reverse the operation

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1, 3, None, 2]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     nw.col("a").cum_prod().alias("a_cum_prod"),
            ...     nw.col("a").cum_prod(reverse=True).alias("a_cum_prod_reverse"),
            ... )
            ┌──────────────────────────────────────┐
            |          Narwhals DataFrame          |
            |--------------------------------------|
            |     a  a_cum_prod  a_cum_prod_reverse|
            |0  1.0         1.0                 6.0|
            |1  3.0         3.0                 6.0|
            |2  NaN         NaN                 NaN|
            |3  2.0         6.0                 2.0|
            └──────────────────────────────────────┘
        """
        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).cum_prod(reverse=reverse),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def rolling_sum(
        self: Self,
        window_size: int,
        *,
        min_samples: int | None = None,
        center: bool = False,
    ) -> Self:
        """Apply a rolling sum (moving sum) over the values.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        A window of length `window_size` will traverse the values. The resulting values
        will be aggregated to their sum.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Arguments:
            window_size: The length of the window in number of elements. It must be a
                strictly positive integer.
            min_samples: The number of values in the window that should be non-null before
                computing a result. If set to `None` (default), it will be set equal to
                `window_size`. If provided, it must be a strictly positive integer, and
                less than or equal to `window_size`
            center: Set the labels at the center of the window.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     a_rolling_sum=nw.col("a").rolling_sum(window_size=3, min_samples=1)
            ... )
            ┌─────────────────────┐
            | Narwhals DataFrame  |
            |---------------------|
            |     a  a_rolling_sum|
            |0  1.0            1.0|
            |1  2.0            3.0|
            |2  NaN            3.0|
            |3  4.0            6.0|
            └─────────────────────┘
        """
        window_size, min_samples = _validate_rolling_arguments(
            window_size=window_size, min_samples=min_samples
        )

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rolling_sum(
                window_size=window_size,
                min_samples=min_samples,
                center=center,
            ),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def rolling_mean(
        self: Self,
        window_size: int,
        *,
        min_samples: int | None = None,
        center: bool = False,
    ) -> Self:
        """Apply a rolling mean (moving mean) over the values.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        A window of length `window_size` will traverse the values. The resulting values
        will be aggregated to their mean.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Arguments:
            window_size: The length of the window in number of elements. It must be a
                strictly positive integer.
            min_samples: The number of values in the window that should be non-null before
                computing a result. If set to `None` (default), it will be set equal to
                `window_size`. If provided, it must be a strictly positive integer, and
                less than or equal to `window_size`
            center: Set the labels at the center of the window.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     a_rolling_mean=nw.col("a").rolling_mean(window_size=3, min_samples=1)
            ... )
            ┌──────────────────────┐
            |  Narwhals DataFrame  |
            |----------------------|
            |     a  a_rolling_mean|
            |0  1.0             1.0|
            |1  2.0             1.5|
            |2  NaN             1.5|
            |3  4.0             3.0|
            └──────────────────────┘
        """
        window_size, min_samples = _validate_rolling_arguments(
            window_size=window_size, min_samples=min_samples
        )

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rolling_mean(
                window_size=window_size,
                min_samples=min_samples,
                center=center,
            ),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def rolling_var(
        self: Self,
        window_size: int,
        *,
        min_samples: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        """Apply a rolling variance (moving variance) over the values.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        A window of length `window_size` will traverse the values. The resulting values
        will be aggregated to their variance.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Arguments:
            window_size: The length of the window in number of elements. It must be a
                strictly positive integer.
            min_samples: The number of values in the window that should be non-null before
                computing a result. If set to `None` (default), it will be set equal to
                `window_size`. If provided, it must be a strictly positive integer, and
                less than or equal to `window_size`.
            center: Set the labels at the center of the window.
            ddof: Delta Degrees of Freedom; the divisor for a length N window is N - ddof.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     a_rolling_var=nw.col("a").rolling_var(window_size=3, min_samples=1)
            ... )
            ┌─────────────────────┐
            | Narwhals DataFrame  |
            |---------------------|
            |     a  a_rolling_var|
            |0  1.0            NaN|
            |1  2.0            0.5|
            |2  NaN            0.5|
            |3  4.0            2.0|
            └─────────────────────┘
        """
        window_size, min_samples = _validate_rolling_arguments(
            window_size=window_size, min_samples=min_samples
        )

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rolling_var(
                window_size=window_size, min_samples=min_samples, center=center, ddof=ddof
            ),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def rolling_std(
        self: Self,
        window_size: int,
        *,
        min_samples: int | None = None,
        center: bool = False,
        ddof: int = 1,
    ) -> Self:
        """Apply a rolling standard deviation (moving standard deviation) over the values.

        !!! warning
            This functionality is considered **unstable**. It may be changed at any point
            without it being considered a breaking change.

        A window of length `window_size` will traverse the values. The resulting values
        will be aggregated to their standard deviation.

        The window at a given row will include the row itself and the `window_size - 1`
        elements before it.

        Arguments:
            window_size: The length of the window in number of elements. It must be a
                strictly positive integer.
            min_samples: The number of values in the window that should be non-null before
                computing a result. If set to `None` (default), it will be set equal to
                `window_size`. If provided, it must be a strictly positive integer, and
                less than or equal to `window_size`.
            center: Set the labels at the center of the window.
            ddof: Delta Degrees of Freedom; the divisor for a length N window is N - ddof.

        Returns:
            A new expression.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [1.0, 2.0, None, 4.0]})
            >>> df = nw.from_native(df_native)
            >>> df.with_columns(
            ...     a_rolling_std=nw.col("a").rolling_std(window_size=3, min_samples=1)
            ... )
            ┌─────────────────────┐
            | Narwhals DataFrame  |
            |---------------------|
            |     a  a_rolling_std|
            |0  1.0            NaN|
            |1  2.0       0.707107|
            |2  NaN       0.707107|
            |3  4.0       1.414214|
            └─────────────────────┘
        """
        window_size, min_samples = _validate_rolling_arguments(
            window_size=window_size, min_samples=min_samples
        )

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rolling_std(
                window_size=window_size,
                min_samples=min_samples,
                center=center,
                ddof=ddof,
            ),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    def rank(
        self: Self,
        method: Literal["average", "min", "max", "dense", "ordinal"] = "average",
        *,
        descending: bool = False,
    ) -> Self:
        """Assign ranks to data, dealing with ties appropriately.

        Notes:
            The resulting dtype may differ between backends.

        Arguments:
            method: The method used to assign ranks to tied elements.
                The following methods are available (default is 'average'):

                - 'average' : The average of the ranks that would have been assigned to
                  all the tied values is assigned to each value.
                - 'min' : The minimum of the ranks that would have been assigned to all
                    the tied values is assigned to each value. (This is also referred to
                    as "competition" ranking.)
                - 'max' : The maximum of the ranks that would have been assigned to all
                    the tied values is assigned to each value.
                - 'dense' : Like 'min', but the rank of the next highest element is
                   assigned the rank immediately after those assigned to the tied
                   elements.
                - 'ordinal' : All values are given a distinct rank, corresponding to the
                    order that the values occur in the Series.

            descending: Rank in descending order.

        Returns:
            A new expression with rank data.

        Examples:
            >>> import pandas as pd
            >>> import narwhals as nw
            >>> df_native = pd.DataFrame({"a": [3, 6, 1, 1, 6]})
            >>> df = nw.from_native(df_native)
            >>> result = df.with_columns(rank=nw.col("a").rank(method="dense"))
            >>> result
            ┌──────────────────┐
            |Narwhals DataFrame|
            |------------------|
            |       a  rank    |
            |    0  3   2.0    |
            |    1  6   3.0    |
            |    2  1   1.0    |
            |    3  1   1.0    |
            |    4  6   3.0    |
            └──────────────────┘
        """
        supported_rank_methods = {"average", "min", "max", "dense", "ordinal"}
        if method not in supported_rank_methods:
            msg = (
                "Ranking method must be one of {'average', 'min', 'max', 'dense', 'ordinal'}. "
                f"Found '{method}'"
            )
            raise ValueError(msg)

        return self.__class__(
            lambda plx: self._to_compliant_expr(plx).rank(
                method=method, descending=descending
            ),
            is_order_dependent=True,
            changes_length=self._changes_length,
            aggregates=self._aggregates,
        )

    @property
    def str(self: Self) -> ExprStringNamespace[Self]:
        return ExprStringNamespace(self)

    @property
    def dt(self: Self) -> ExprDateTimeNamespace[Self]:
        return ExprDateTimeNamespace(self)

    @property
    def cat(self: Self) -> ExprCatNamespace[Self]:
        return ExprCatNamespace(self)

    @property
    def name(self: Self) -> ExprNameNamespace[Self]:
        return ExprNameNamespace(self)

    @property
    def list(self: Self) -> ExprListNamespace[Self]:
        return ExprListNamespace(self)


__all__ = [
    "Expr",
]
