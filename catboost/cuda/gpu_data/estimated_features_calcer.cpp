#include "estimated_features_calcer.h"
#include "gpu_binarization_helpers.h"

namespace NCatboostCuda {

    void TEstimatorsExecutor::ExecEstimators(
        TConstArrayRef<NCB::TEstimatorId> estimatorIds,
        TBinarizedFeatureVisitor learnBinarizedVisitor,
        TMaybe<TBinarizedFeatureVisitor> testBinarizedVisitor
    ) {
        TGpuBordersBuilder bordersBuilder(FeaturesManager);
        for (const auto& estimator : estimatorIds) {
            auto featureVisitor = [&](TBinarizedFeatureVisitor visitor, ui32 featureId, TConstArrayRef<float> values) {
                NCB::TEstimatedFeatureId feature{estimator, featureId};
                auto id = FeaturesManager.GetId(feature);
                auto borders = bordersBuilder.GetOrComputeBorders(id, FeaturesManager.GetBinarizationDescription(feature), values);
                auto binarized = NCB::BinarizeLine<ui8>(values,
                                                        ENanMode::Forbidden,
                                                        borders);
                CB_ENSURE(borders.size() <= 255, "Error: too many borders " << borders.size());
                const ui8 binCount = borders.size() + 1;
                visitor(binarized, feature, binCount);
            };

            NCB::TCalculatedFeatureVisitor learnVisitor{
                std::bind(
                    featureVisitor,
                    learnBinarizedVisitor,
                    std::placeholders::_1,
                    std::placeholders::_2
                )
            };

            TVector<NCB::TCalculatedFeatureVisitor> testVisitors;
            if (testBinarizedVisitor) {
                testVisitors.push_back(
                    NCB::TCalculatedFeatureVisitor{
                        std::bind(
                            featureVisitor,
                            *testBinarizedVisitor,
                            std::placeholders::_1,
                            std::placeholders::_2
                        )
                    }
                );
            }

            if (estimator.IsOnline) {
                Estimators.GetOnlineFeatureEstimator(estimator.Id)->ComputeOnlineFeatures(
                    PermutationIndices,
                    learnVisitor,
                    testVisitors,
                    LocalExecutor);
            } else {
                Estimators.GetFeatureEstimator(estimator.Id)->ComputeFeatures(learnVisitor, testVisitors, LocalExecutor);
            }
        }
    }

}
