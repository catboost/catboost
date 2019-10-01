#include "feature_estimators.h"

#include <catboost/libs/helpers/exception.h>

#include <util/generic/xrange.h>

namespace NCB {

    TFeatureEstimatorPtr TFeatureEstimators::GetFeatureEstimator(TEstimatorId estimatorId) const {
        if (estimatorId.IsOnline) {
            return OnlineFeatureEstimators[estimatorId.Id];
        } else {
            return FeatureEstimators[estimatorId.Id];
        }
    }

    TEstimatorId TFeatureEstimators::GetFeatureEstimatorIdByCalcerId(ui32 calcerId) const {
        CB_ENSURE(
            CalcerToEstimatorId.contains(calcerId),
            "There is no estimator with " << LabeledOutput(calcerId)
        );
        return CalcerToEstimatorId.at(calcerId);
    }

    TFeatureEstimatorPtr TFeatureEstimators::GetFeatureEstimator(ui32 estimatorId) const {
        return FeatureEstimators[estimatorId];
    }

    TOnlineFeatureEstimatorPtr TFeatureEstimators::GetOnlineFeatureEstimator(ui32 estimatorId) const {
        return OnlineFeatureEstimators[estimatorId];
    }

    TEstimatorSourceId TFeatureEstimators::GetEstimatorSourceFeatureIdx(const TEstimatorId& estimatorId) const {
        CB_ENSURE(
            EstimatorToSourceFeatures.contains(estimatorId),
            "There is no estimator with " << LabeledOutput(estimatorId.Id, estimatorId.IsOnline)
        );
        return EstimatorToSourceFeatures.at(estimatorId);
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

        ui32 calcerId = 0;
        for (ui32 featureEstimatorId: xrange(FeatureEstimatorsSourceId.size())) {
            TEstimatorId id{featureEstimatorId, calcerId, false};
            estimatorToSourceFeatures[id] = FeatureEstimatorsSourceId[featureEstimatorId];
            calcerId++;
        }

        for (ui32 onlineEstimatorId: xrange(OnlineFeatureEstimatorsSourceId.size())) {
            TEstimatorId id{onlineEstimatorId, calcerId, true};
            estimatorToSourceFeatures[id] = OnlineFeatureEstimatorsSourceId[onlineEstimatorId];
            calcerId++;
        }

        return MakeIntrusive<TFeatureEstimators>(
            std::move(FeatureEstimators),
            std::move(OnlineFeatureEstimators),
            std::move(estimatorToSourceFeatures)
        );
    }

}
