from __future__ import annotations

from typing import TYPE_CHECKING

import pyarrow as pa

if TYPE_CHECKING:
    from typing_extensions import Self

    from narwhals._arrow.series import ArrowSeries
    from narwhals._arrow.typing import Incomplete


class ArrowSeriesCatNamespace:
    def __init__(self: Self, series: ArrowSeries) -> None:
        self._compliant_series: ArrowSeries = series

    def get_categories(self: Self) -> ArrowSeries:
        # NOTE: Should be `list[pa.DictionaryArray]`, but `DictionaryArray` has no attributes
        chunks: Incomplete = self._compliant_series._native_series.chunks
        return self._compliant_series._from_native_series(
            pa.concat_arrays(x.dictionary for x in chunks).unique()
        )
