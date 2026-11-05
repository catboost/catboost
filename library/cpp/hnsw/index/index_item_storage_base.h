#pragma once

#include "index_base.h"
#include <library/cpp/hnsw/index/core/index_item_storage_base.h>

namespace NHnsw {
    /** @brief Item-storage HNSW index using the built-in graph formats. */
    template <class TItemStorage>
    using THnswItemStorageIndexBase = THnswItemStorageIndexBaseImpl<TItemStorage, THnswIndexBase>;
} // namespace NHnsw
