#include "embedding_feature_estimators.h"
#include "base_embedding_feature_estimator.h"

#include <catboost/private/libs/embedding_features/embedding_calcers.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/enum_helpers.h>

namespace NCB {

    class TLDAEstimator final : public TEmbeddingBaseEstimator<TLinearDACalcer, TLinearDACalcerVisitor>{
    public:
        TLDAEstimator(
            TEmbeddingClassificationTargetPtr target,
            TEmbeddingDataSetPtr learnEmbeddings,
            TArrayRef<TEmbeddingDataSetPtr> testEmbedding,
            const NJson::TJsonValue& options)
            : TEmbeddingBaseEstimator(target, learnEmbeddings, testEmbedding)
        {
            if (options.Has("ProjectionDimension")) {
                ProjectionDim = FromString<int>(options["ProjectionDimension"].GetString());
            } else {
                ProjectionDim = GetTarget().NumClasses - 1u;
            }
            if (options.Has("Regularization")) {
                RegParam = FromString<float>(options["Regularization"].GetString());
            } else {
                RegParam = 0.01;
            }
            CB_ENSURE(
                ProjectionDim > 0,
                "Dimension of the projection should be positive"
            );
            CB_ENSURE(
                ProjectionDim < GetLearnDatasetPtr()->GetDimension(),
                "Dimension of the projection should be less then total dimension of the embedding"
            );
            CB_ENSURE(
                RegParam >= 0,
                "Regularisation coefficient shoul be positive"
            );
        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = ProjectionDim;
            meta.Type.resize(meta.FeaturesCount, EFeatureCalcerType::LDA);
            return meta;
        }

        TLinearDACalcer CreateFeatureCalcer() const override {
            return TLinearDACalcer(GetLearnDataset().GetDimension(), GetTarget().NumClasses, ProjectionDim, RegParam);
        }

        TLinearDACalcerVisitor CreateCalcerVisitor() const override {
            return {};
        }

    private:
        ui32 ProjectionDim;
        float RegParam;
    };

    TVector<TOnlineFeatureEstimatorPtr> CreateEmbeddingEstimators(
        TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
        TEmbeddingClassificationTargetPtr target,
        TEmbeddingDataSetPtr learnEmbeddings,
        TArrayRef<TEmbeddingDataSetPtr> testEmbedding
    ) {
        TVector<TOnlineFeatureEstimatorPtr> estimators;
        for (auto& calcerDescription: featureCalcerDescription) {
            if (calcerDescription.CalcerType == EFeatureCalcerType::LDA) {
                estimators.emplace_back(MakeIntrusive<TLDAEstimator>(
                        target,
                        learnEmbeddings,
                        testEmbedding,
                        calcerDescription.CalcerOptions
                    )
                );
            }
        }

        return estimators;
    }
};
