from __future__ import annotations

from typing import TYPE_CHECKING

from narwhals._pandas_like.utils import to_datetime

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._pandas_like.series import PandasLikeSeries


class PandasLikeSeriesStringNamespace:
    def __init__(self: Self, series: PandasLikeSeries) -> None:
        self._compliant_series = series

    def len_chars(self: Self) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.len()
        )

    def replace(
        self: Self, pattern: str, value: str, *, literal: bool, n: int
    ) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.replace(
                pat=pattern, repl=value, n=n, regex=not literal
            ),
        )

    def replace_all(
        self: Self, pattern: str, value: str, *, literal: bool
    ) -> PandasLikeSeries:
        return self.replace(pattern, value, literal=literal, n=-1)

    def strip_chars(self: Self, characters: str | None) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.strip(characters),
        )

    def starts_with(self: Self, prefix: str) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.startswith(prefix),
        )

    def ends_with(self: Self, suffix: str) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.endswith(suffix),
        )

    def contains(self: Self, pattern: str, *, literal: bool) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.contains(
                pat=pattern, regex=not literal
            )
        )

    def slice(self: Self, offset: int, length: int | None) -> PandasLikeSeries:
        stop = offset + length if length else None
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.slice(start=offset, stop=stop),
        )

    def to_datetime(self: Self, format: str | None) -> PandasLikeSeries:  # noqa: A002
        return self._compliant_series._from_native_series(
            to_datetime(self._compliant_series._implementation)(
                self._compliant_series._native_series, format=format
            )
        )

    def to_uppercase(self: Self) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.upper(),
        )

    def to_lowercase(self: Self) -> PandasLikeSeries:
        return self._compliant_series._from_native_series(
            self._compliant_series._native_series.str.lower(),
        )
