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

    static ui8 ExtractFeatureFromPack(ui32 featurePack, ui32 featureIndex) {
        return static_cast<ui8>((featurePack & (1u << featureIndex)) > 0);
    }

    static void ExtractFeatureFromPack(
        TConstArrayRef<ui32> featuresPack,
        ui32 featureIndex,
        TArrayRef<ui8> binaryFeature
    ) {
        for (ui32 docId: xrange(featuresPack.size())) {
            binaryFeature[docId] = ExtractFeatureFromPack(featuresPack[docId], featureIndex);
        }
    }

    void TEstimatorsExecutor::ExecBinaryFeaturesEstimators(
        TConstArrayRef<NCB::TEstimatorId> estimatorIds,
        TBinarizedFeatureVisitor learnBinarizedVisitor,
        TMaybe<TBinarizedFeatureVisitor> testBinarizedVisitor
    ) {
        for (const auto& estimator : estimatorIds) {
            auto featureVisitor =
                [&](
                    TBinarizedFeatureVisitor visitor,
                    TConstArrayRef<ui32> featureIds,
                    TConstArrayRef<ui32> binFeatures
                ) {
                    TVector<ui8> binarized(binFeatures.size());
                    const ui8 binCount = 2;
                    for (ui32 i: xrange(featureIds.size())) {
                        NCB::TEstimatedFeatureId feature{estimator, featureIds[i]};

                        auto id = FeaturesManager.GetId(feature);
                        if (!FeaturesManager.HasBorders(id)) {
                            FeaturesManager.SetBorders(id, {0.5});
                        }

                        ExtractFeatureFromPack(
                            MakeConstArrayRef(binFeatures),
                            i,
                            TArrayRef<ui8>(binarized.data(), binarized.size())
                        );
                        visitor(binarized, feature, binCount);
                    }
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
                    LocalExecutor
                );
            } else {
                Estimators.GetFeatureEstimator(estimator.Id)->ComputeFeatures(
                    learnVisitor,
                    testVisitors,
                    LocalExecutor
                );
            }
        }
    }

}
