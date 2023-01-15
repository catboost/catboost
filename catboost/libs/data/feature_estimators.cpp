#include "feature_estimators.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/xrange.h>
#include <util/stream/output.h>


template <>
void Out<NCB::TEstimatedFeatureId>(IOutputStream& out, const NCB::TEstimatedFeatureId& feature) {
    out << "estimatorId=" << feature.EstimatorId.Id;
    if (feature.EstimatorId.IsOnline) {
        out << "(online)";
    }
    out << ", id=" << feature.LocalFeatureId;
}


namespace NCB {

    TFeatureEstimatorPtr TFeatureEstimators::GetFeatureEstimator(TEstimatorId estimatorId) const {
        if (estimatorId.IsOnline) {
            return OnlineFeatureEstimators[estimatorId.Id];
        } else {
            return FeatureEstimators[estimatorId.Id];
        }
    }

    TGuid TFeatureEstimators::GetEstimatorGuid(TEstimatorId estimatorId) const {
        return GetFeatureEstimator(estimatorId)->Id();
    }

    TFeatureEstimatorPtr TFeatureEstimators::GetEstimatorByGuid(const TGuid& calcerId) const {
        CB_ENSURE(
            EstimatorGuidToFlatId.contains(calcerId),
            "There is no estimator with " << LabeledOutput(calcerId)
        );
        return GetFeatureEstimator(EstimatorGuidToFlatId.at(calcerId));
    }

    TFeatureEstimatorPtr TFeatureEstimators::GetFeatureEstimator(ui32 estimatorId) const {
        return FeatureEstimators[estimatorId];
    }

    TOnlineFeatureEstimatorPtr TFeatureEstimators::GetOnlineFeatureEstimator(ui32 estimatorId) const {
        return OnlineFeatureEstimators[estimatorId];
    }

    TEstimatorSourceId TFeatureEstimators::GetEstimatorSourceFeatureIdx(const TGuid& guid) const {
        CB_ENSURE(
            EstimatorGuidToFlatId.contains(guid),
            "There is no estimator with " << LabeledOutput(guid)
        );
        TEstimatorId estimatorId = EstimatorGuidToFlatId.at(guid);
        return EstimatorToSourceFeatures.at(estimatorId);
    }

    TEstimatorSourceId TFeatureEstimators::GetEstimatorSourceFeatureIdx(TEstimatorId estimatorId) const {
        return EstimatorToSourceFeatures.at(estimatorId);
    }

    EFeatureType TFeatureEstimators::GetEstimatorSourceType(TEstimatorId estimatorId) const {
        if (estimatorId.IsOnline) {
            return OnlineFeatureEstimators.at(estimatorId.Id)->GetSourceType();
        } else {
            return FeatureEstimators.at(estimatorId.Id)->GetSourceType();
        }
    }

    EFeatureType TFeatureEstimators::GetEstimatorSourceType(const TGuid& guid) const {
        CB_ENSURE(
            EstimatorGuidToFlatId.contains(guid),
            "There is no estimator with " << LabeledOutput(guid)
        );
        TEstimatorId estimatorId = EstimatorGuidToFlatId.at(guid);
        return GetEstimatorSourceType(estimatorId);
    }

    void TFeatureEstimators::RebuildInnerData() {
        for (const auto& [estimatorId, sourceId]: EstimatorToSourceFeatures) {
            TFeatureEstimatorPtr estimator;
            if (estimatorId.IsOnline) {
                estimator = OnlineFeatureEstimators[estimatorId.Id];
            } else {
                estimator = FeatureEstimators[estimatorId.Id];
            }
            const TGuid& guid = estimator->Id();
            EstimatorGuidToFlatId[guid] = estimatorId;
        }
    }

    void TFeatureEstimatorsBuilder::AddFeatureEstimator(
        TFeatureEstimatorPtr&& estimator,
        const TEstimatorSourceId& estimatorSourceId
    ) {
        FeatureEstimatorsSourceId.push_back(estimatorSourceId);
        FeatureEstimators.push_back(std::move(estimator));
    }

    void TFeatureEstimatorsBuilder::AddFeatureEstimator(
        TOnlineFeatureEstimatorPtr&& estimator,
        const TEstimatorSourceId& estimatorSourceId
    ) {
        OnlineFeatureEstimatorsSourceId.push_back(estimatorSourceId);
        OnlineFeatureEstimators.push_back(std::move(estimator));
    }

    TFeatureEstimatorsPtr TFeatureEstimatorsBuilder::Build() {
        CB_ENSURE(!WasBuilt, "TFeatureEstimatorsBuilder::Build can be done only once");
        WasBuilt = true;

        TMap<TEstimatorId, TEstimatorSourceId> estimatorToSourceFeatures;

        for (ui32 estimatorId: xrange(FeatureEstimatorsSourceId.size())) {
            TEstimatorId id{estimatorId, false};
            estimatorToSourceFeatures[id] = FeatureEstimatorsSourceId[estimatorId];
        }

        for (ui32 estimatorId: xrange(OnlineFeatureEstimatorsSourceId.size())) {
            TEstimatorId id{estimatorId, true};
            estimatorToSourceFeatures[id] = OnlineFeatureEstimatorsSourceId[estimatorId];
        }

        return MakeIntrusive<TFeatureEstimators>(
            std::move(FeatureEstimators),
            std::move(OnlineFeatureEstimators),
            std::move(estimatorToSourceFeatures)
        );
    }

}
