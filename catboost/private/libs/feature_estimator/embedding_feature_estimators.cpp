#include "embedding_feature_estimators.h"
#include "base_embedding_feature_estimator.h"

#include <catboost/private/libs/embedding_features/embedding_calcers.h>
#include <catboost/private/libs/options/json_helper.h>
#include <catboost/private/libs/options/enum_helpers.h>

namespace NCB {

    class TLDAEstimator final : public TEmbeddingBaseEstimator<TLinearDACalcer, TLinearDACalcerVisitor>{
    public:
        TLDAEstimator(
            TClassificationTargetPtr target,
            TEmbeddingDataSetPtr learnEmbeddings,
            TArrayRef<TEmbeddingDataSetPtr> testEmbedding,
            const NJson::TJsonValue& options)
            : TEmbeddingBaseEstimator(target, learnEmbeddings, testEmbedding)
        {
            if (options.Has("components")) {
                ProjectionDim = FromString<int>(options["components"].GetString());
            } else {
                ProjectionDim = Min(GetTarget().NumClasses - 1u, static_cast<ui32>(GetLearnDatasetPtr()->GetDimension()) - 1u);
            }
            if (options.Has("reg")) {
                RegParam = FromString<float>(options["reg"].GetString());
            } else {
                RegParam = 0.00005;
            }
            if (options.Has("likelihood")) {
                Likehood = FromString<bool>(options["likelihood"].GetString());
            } else {
                Likehood = false;
            }
            FeaturesCount = ProjectionDim;
            if (Likehood) {
                FeaturesCount += GetTarget().NumClasses;
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
                "Regularisation coefficient should be positive"
            );
        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = FeaturesCount;
            meta.Type.resize(meta.FeaturesCount, EFeatureCalcerType::LDA);
            return meta;
        }

        TLinearDACalcer CreateFeatureCalcer() const override {
            return TLinearDACalcer(GetLearnDataset().GetDimension(), GetTarget().NumClasses,
                                   ProjectionDim, RegParam, Likehood);
        }

        TLinearDACalcerVisitor CreateCalcerVisitor() const override {
            return {};
        }

    private:
        ui32 ProjectionDim;
        ui32 FeaturesCount;
        float RegParam;
        bool Likehood;
    };

    class TKNNEstimator final : public TEmbeddingBaseEstimator<TKNNCalcer, TKNNCalcerVisitor>{
    public:
        TKNNEstimator(
            TClassificationTargetPtr target,
            TEmbeddingDataSetPtr learnEmbeddings,
            TArrayRef<TEmbeddingDataSetPtr> testEmbedding,
            const NJson::TJsonValue& options)
            : TEmbeddingBaseEstimator(target, learnEmbeddings, testEmbedding)
        {
            ClassNum = GetTarget().NumClasses;
            if (options.Has("k")) {
                kNum = FromString<int>(options["k"].GetString());
            } else {
                kNum = 5;
            }
        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = ClassNum;
            meta.Type.resize(meta.FeaturesCount, EFeatureCalcerType::KNN);
            return meta;
        }

        TKNNCalcer CreateFeatureCalcer() const override {
            return TKNNCalcer(GetLearnDataset().GetDimension(), GetTarget().NumClasses, kNum);
        }

        TKNNCalcerVisitor CreateCalcerVisitor() const override {
            return {};
        }

    private:
        ui32 ClassNum;
        ui32 kNum;
    };

    TVector<TOnlineFeatureEstimatorPtr> CreateEmbeddingEstimators(
        TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
        TClassificationTargetPtr target,
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
            if (calcerDescription.CalcerType == EFeatureCalcerType::KNN) {
                estimators.emplace_back(MakeIntrusive<TKNNEstimator>(
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
