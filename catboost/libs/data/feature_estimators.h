#pragma once

#include <catboost/private/libs/feature_estimator/feature_estimator.h>

#include <util/digest/multi.h>
#include <util/generic/hash.h>
#include <util/generic/map.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/str_stl.h>
#include <util/system/types.h>
#include <util/ysaveload.h>

#include <tuple>


namespace NCB {
    struct TEstimatorId {
    public:
        ui32 Id = 0;
        bool IsOnline = false;

    public:
        TEstimatorId() = default;

        TEstimatorId(ui32 id, bool isOnline)
            : Id(id)
            , IsOnline(isOnline) {}

        bool operator<(const TEstimatorId& rhs) const {
            return std::tie(IsOnline, Id) < std::tie(rhs.IsOnline, rhs.Id);
        }

        bool operator>(const TEstimatorId& rhs) const {
            return rhs < *this;
        }

        bool operator<=(const TEstimatorId& rhs) const {
            return !(rhs < *this);
        }

        bool operator>=(const TEstimatorId& rhs) const {
            return !(*this < rhs);
        }

        bool operator==(const TEstimatorId& rhs) const {
            return std::tie(IsOnline, Id) == std::tie(rhs.IsOnline, rhs.Id);
        }

        bool operator!=(const TEstimatorId& rhs) const {
            return !(rhs == *this);
        }

        ui64 GetHash() const {
            return MultiHash(IsOnline, Id);
        }

        Y_SAVELOAD_DEFINE(IsOnline, Id);
    };
}

template <>
struct THash<NCB::TEstimatorId> {
    inline size_t operator()(const NCB::TEstimatorId& value) const {
        return value.GetHash();
    }
};


namespace NCB {
    struct TEstimatorSourceId {
        TEstimatorSourceId() = default;

        TEstimatorSourceId(ui32 textFeatureId, ui32 tokenizedFeatureId)
            : TextFeatureId(textFeatureId)
            , TokenizedFeatureId(tokenizedFeatureId)
        {}

        bool operator<(const TEstimatorSourceId& rhs) const {
            return std::tie(TextFeatureId, TokenizedFeatureId)
                < std::tie(rhs.TextFeatureId, rhs.TokenizedFeatureId);
        }

        bool operator>(const TEstimatorSourceId& rhs) const {
            return rhs < *this;
        }

        bool operator<=(const TEstimatorSourceId& rhs) const {
            return !(rhs < *this);
        }

        bool operator>=(const TEstimatorSourceId& rhs) const {
            return !(*this < rhs);
        }

        bool operator==(const TEstimatorSourceId& rhs) const {
            return std::tie(TextFeatureId, TokenizedFeatureId)
                == std::tie(rhs.TextFeatureId, rhs.TokenizedFeatureId);
        }

        bool operator!=(const TEstimatorSourceId& rhs) const {
            return !(rhs == *this);
        }

        ui64 GetHash() const {
            return MultiHash(TextFeatureId, TokenizedFeatureId);
        }

        Y_SAVELOAD_DEFINE(TextFeatureId, TokenizedFeatureId);

    public:
        ui32 TextFeatureId;
        ui32 TokenizedFeatureId;
    };
}

template <>
struct THash<NCB::TEstimatorSourceId> {
    inline size_t operator()(const NCB::TEstimatorSourceId& value) const {
        return value.GetHash();
    }
};

namespace NCB {
    class TFeatureEstimators : public TThrRefBase {
    public:
        TFeatureEstimators() // TODO(d-kruchinin) replace default constructor calls to instantiation from Builder
            : FeatureEstimators()
            , OnlineFeatureEstimators()
            , EstimatorToSourceFeatures()
        {}

        TFeatureEstimators(
            TVector<TFeatureEstimatorPtr>&& featureEstimators,
            TVector<TOnlineFeatureEstimatorPtr>&& onlineFeatureEstimators,
            TMap<TEstimatorId, TEstimatorSourceId>&& estimatorToSourceFeatures)
        : FeatureEstimators(std::move(featureEstimators))
        , OnlineFeatureEstimators(std::move(onlineFeatureEstimators))
        , EstimatorToSourceFeatures(std::move(estimatorToSourceFeatures))
        {
            RebuildInnerData();
        }

        bool Empty() const {
            return FeatureEstimators.empty() && OnlineFeatureEstimators.empty();
        }

        ui32 GetOfflineFeatureEstimatorsSize() const {
            return FeatureEstimators.size();
        }

        ui32 GetOnlineFeatureEstimatorsSize() const {
            return OnlineFeatureEstimators.size();
        }

        TFeatureEstimatorPtr GetFeatureEstimator(TEstimatorId estimatorId) const;
        TGuid GetEstimatorGuid(TEstimatorId estimatorId) const;

        TFeatureEstimatorPtr GetEstimatorByGuid(const TGuid& calcerId) const;

        TFeatureEstimatorPtr GetFeatureEstimator(ui32 estimatorId) const;

        TOnlineFeatureEstimatorPtr GetOnlineFeatureEstimator(ui32 estimatorId) const;

        template <class F>
        void ForEach(F&& functor) const {
            for (const auto& [estimatorId, sourceFeatureIdx]: EstimatorToSourceFeatures) {
                Y_UNUSED(sourceFeatureIdx);
                if (estimatorId.IsOnline) {
                    functor(estimatorId, OnlineFeatureEstimators[estimatorId.Id]);
                } else {
                    functor(estimatorId, FeatureEstimators[estimatorId.Id]);
                }
            }
        }

        TEstimatorSourceId GetEstimatorSourceFeatureIdx(const TGuid& guid) const;

    private:
        void RebuildInnerData();

        const TVector<TFeatureEstimatorPtr> FeatureEstimators;
        const TVector<TOnlineFeatureEstimatorPtr> OnlineFeatureEstimators;

        const TMap<TEstimatorId, TEstimatorSourceId> EstimatorToSourceFeatures;
        THashMap<TGuid, TEstimatorId> EstimatorGuidToFlatId;
    };

    using TFeatureEstimatorsPtr = TIntrusiveConstPtr<TFeatureEstimators>;

    class TFeatureEstimatorsBuilder {
    public:
        void AddFeatureEstimator(
            TFeatureEstimatorPtr&& estimator,
            const TEstimatorSourceId& estimatorSourceId
        );

        void AddFeatureEstimator(
            TOnlineFeatureEstimatorPtr&& estimator,
            const TEstimatorSourceId& estimatorSourceId
        );

        TFeatureEstimatorsPtr Build();

    private:
        bool WasBuilt = false;

        TVector<TEstimatorSourceId> FeatureEstimatorsSourceId;
        TVector<TEstimatorSourceId> OnlineFeatureEstimatorsSourceId;

        TVector<TFeatureEstimatorPtr> FeatureEstimators;
        TVector<TOnlineFeatureEstimatorPtr> OnlineFeatureEstimators;
    };
}
