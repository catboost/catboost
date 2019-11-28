#pragma once

#include <util/charset/unicode_table.h>

namespace NUnicode {
    namespace NPrivate {
        typedef NUnicodeTable::TTable<NUnicodeTable::TSubtable<
            NUnicodeTable::UNICODE_TABLE_SHIFT, NUnicodeTable::TValues<const wchar32*>>>
            TDecompositionTable;

        const TDecompositionTable& CannonDecompositionTable();
        const TDecompositionTable& CompatDecompositionTable();

        template <bool compat>
        inline const TDecompositionTable& DecompositionTable();

        template <>
        inline const TDecompositionTable& DecompositionTable<false>() {
            return CannonDecompositionTable();
        }

        template <>
        inline const TDecompositionTable& DecompositionTable<true>() {
            return CompatDecompositionTable();
        }

    }
};    // namespace NUnicode
