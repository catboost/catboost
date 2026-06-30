#pragma once

#include <library/cpp/containers/dense_hash/dense_hash.h>

namespace NHnsw {
    class TDefaultSearchContext {
    private:
        static constexpr ui32 EMPTY_MARKER = Max<ui32>();

        TDenseHashSet<ui32> VisitedSet_;

    public:
        TDefaultSearchContext()
            : VisitedSet_(EMPTY_MARKER)
        {}

        bool TryMarkVisited(ui32 id) {
            return VisitedSet_.Insert(id);
        }
    };
}
