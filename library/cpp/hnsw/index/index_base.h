#pragma once

#include "builtin_graph_accessor.h"
#include <library/cpp/hnsw/index/core/index_base.h>

namespace NHnsw {
    /** @brief HNSW index over the built-in ui32 and packed graph formats. */
    class THnswIndexBase: public THnswIndexBaseImpl<TBuiltinGraphAccessor> {
        using TBase = THnswIndexBaseImpl<TBuiltinGraphAccessor>;

    public:
        using TBase::TBase;
        using TBase::GetNearestNeighbors;
    };
} // namespace NHnsw
