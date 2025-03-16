from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import Sequence

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.expr import ArrowExpr


class ArrowExprNameNamespace:
    def __init__(self: Self, expr: ArrowExpr) -> None:
        self._compliant_expr = expr

    def keep(self: Self) -> ArrowExpr:
        return self._from_colname_func_and_alias_output_names(
            name_mapping_func=lambda name: name,
            alias_output_names=None,
        )

    def map(self: Self, function: Callable[[str], str]) -> ArrowExpr:
        return self._from_colname_func_and_alias_output_names(
            name_mapping_func=function,
            alias_output_names=lambda output_names: [
                function(name) for name in output_names
            ],
        )

    def prefix(self: Self, prefix: str) -> ArrowExpr:
        return self._from_colname_func_and_alias_output_names(
            name_mapping_func=lambda name: f"{prefix}{name}",
            alias_output_names=lambda output_names: [
                f"{prefix}{output_name}" for output_name in output_names
            ],
        )

    def suffix(self: Self, suffix: str) -> ArrowExpr:
        return self._from_colname_func_and_alias_output_names(
            name_mapping_func=lambda name: f"{name}{suffix}",
            alias_output_names=lambda output_names: [
                f"{output_name}{suffix}" for output_name in output_names
            ],
        )

    def to_lowercase(self: Self) -> ArrowExpr:
        return self._from_colname_func_and_alias_output_names(
            name_mapping_func=str.lower,
            alias_output_names=lambda output_names: [
                name.lower() for name in output_names
            ],
        )

    def to_uppercase(self: Self) -> ArrowExpr:
        return self._from_colname_func_and_alias_output_names(
            name_mapping_func=str.upper,
            alias_output_names=lambda output_names: [
                name.upper() for name in output_names
            ],
        )

    def _from_colname_func_and_alias_output_names(
        self: Self,
        name_mapping_func: Callable[[str], str],
        alias_output_names: Callable[[Sequence[str]], Sequence[str]] | None,
    ) -> ArrowExpr:
        return self._compliant_expr.__class__(
            call=lambda df: [
                series.alias(name_mapping_func(name))
                for series, name in zip(
                    self._compliant_expr._call(df),
                    self._compliant_expr._evaluate_output_names(df),
                )
            ],
            depth=self._compliant_expr._depth,
            function_name=self._compliant_expr._function_name,
            evaluate_output_names=self._compliant_expr._evaluate_output_names,
            alias_output_names=alias_output_names,
            backend_version=self._compliant_expr._backend_version,
            version=self._compliant_expr._version,
            call_kwargs=self._compliant_expr._call_kwargs,
        )
