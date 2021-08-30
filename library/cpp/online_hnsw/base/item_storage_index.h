#pragma once

#include "index_base.h"

namespace NOnlineHnsw {
    /**
     * @brief Base class for OnlineHnswIndex.
     * Inheritance is based on CRTP.
     * If item storage is outside your class, see index_base.h
     *
     * @code
     *    class TOnlineHnswIndex: public TOnlineHnswItemStorageIndexBase<TOnlineHnswIndex, TItem, TDistance> {
     *        using TBase = TOnlineHnswItemStorageIndexBase<TOnlineHnswIndex, TItem, TDistance>;
     *        public:
     *            // In order to create with options.
     *            using TBase::TBase;
     *            // In order for Index to use itself as a storage these methods must be implemented.
     *            // Items are enumerated from zero in order of their appearance.
     *            const TItem& GetItem(size_t id) const;
     *            size_t GetNumItems() const;
     *            void AddItem(const TItem& item) const;
     *    }
     */
    template<class TItemStorage,
             class TItem,
             class TDistance,
             class TDistanceResult = typename TDistance::TResult,
             class TDistanceLess = typename TDistance::TLess>
    class TOnlineHnswItemStorageIndexBase: public TOnlineHnswIndexBase<TDistance, TDistanceResult, TDistanceLess> {
        using TBase = TOnlineHnswIndexBase<TDistance, TDistanceResult, TDistanceLess>;
        using TDistanceTraits = NHnsw::TDistanceTraits<TDistance, TDistanceResult, TDistanceLess>;
        using TNeighbors = typename TDistanceTraits::TNeighbors;

    public:
        using TBase::TBase;

        TNeighbors GetNearestNeighbors(const TItem& item, const size_t topSize = Max<size_t>()) const {
            return TBase::template GetNearestNeighbors<TItem, TItemStorage>(item, static_cast<const TItemStorage&>(*this), topSize);
        }

        TNeighbors GetNearestNeighborsAndAddItem(const TItem& item) {
            return TBase::template GetNearestNeighborsAndAddItem<TItem, TItemStorage>(item, static_cast<TItemStorage*>(this));
        }
    };
} // NOnlineHnsw
