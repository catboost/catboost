#include "text_feature_estimators.h"
#include "base_text_feature_estimator.h"

#include <catboost/private/libs/text_features/text_feature_calcers.h>
#include <catboost/private/libs/text_processing/embedding.h>

#include <catboost/private/libs/options/enum_helpers.h>
#include <catboost/private/libs/options/json_helper.h>

#include <util/generic/set.h>


using namespace NCB;

namespace {
    class TNaiveBayesEstimator final: public TBaseEstimator<TMultinomialNaiveBayes, TNaiveBayesVisitor> {
    public:
        TNaiveBayesEstimator(
            TTextClassificationTargetPtr target,
            TTextDataSetPtr learnTexts,
            TArrayRef<TTextDataSetPtr> testText)
            : TBaseEstimator(std::move(target), std::move(learnTexts), testText)
        {
        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = TMultinomialNaiveBayes::BaseFeatureCount(GetTarget().NumClasses);
            meta.Type.resize(meta.FeaturesCount, EFeatureCalcerType::NaiveBayes);
            return meta;
        }

        TMultinomialNaiveBayes CreateFeatureCalcer() const override {
            return TMultinomialNaiveBayes(Id(), GetTarget().NumClasses);
        }

        TNaiveBayesVisitor CreateCalcerVisitor() const override {
            return TNaiveBayesVisitor();
        };
    };

    class TBM25Estimator final: public TBaseEstimator<TBM25, TBM25Visitor> {
    public:
        TBM25Estimator(
            TTextClassificationTargetPtr target,
            TTextDataSetPtr learnTexts,
            TArrayRef<TTextDataSetPtr> testText)
            : TBaseEstimator(std::move(target), std::move(learnTexts), testText)
        {
        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = TBM25::BaseFeatureCount(GetTarget().NumClasses);
            meta.Type.resize(meta.FeaturesCount, EFeatureCalcerType::BM25);
            return meta;
        }

        TBM25 CreateFeatureCalcer() const override {
            return TBM25(Id(), GetTarget().NumClasses);
        }

        TBM25Visitor CreateCalcerVisitor() const override {
            return TBM25Visitor();
        };
    };

    class TEmbeddingOnlineFeaturesEstimator final:
        public TBaseEstimator<TEmbeddingOnlineFeatures, TEmbeddingFeaturesVisitor> {
    public:
        TEmbeddingOnlineFeaturesEstimator(
            TEmbeddingPtr embedding,
            TTextClassificationTargetPtr target,
            TTextDataSetPtr learnTexts,
            TArrayRef<TTextDataSetPtr> testText,
            const TSet<EFeatureCalcerType>& enabledTypes)
            : TBaseEstimator(std::move(target), std::move(learnTexts), std::move(testText))
            , Embedding(std::move(embedding))
            , ComputeCosDistance(enabledTypes.contains(EFeatureCalcerType::CosDistanceWithClassCenter))
            , ComputeGaussianHomoscedatic(enabledTypes.contains(EFeatureCalcerType::GaussianHomoscedasticModel))
            , ComputeGaussianHeteroscedatic(enabledTypes.contains(EFeatureCalcerType::GaussianHeteroscedasticModel))
        {}

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = TEmbeddingOnlineFeatures::BaseFeatureCount(
                GetTarget().NumClasses,
                ComputeCosDistance,
                ComputeGaussianHomoscedatic,
                ComputeGaussianHeteroscedatic
            );

            for (ui32 classIdx = 0; classIdx < GetTarget().NumClasses; ++classIdx) {
                if (ComputeCosDistance) {
                    meta.Type.push_back(EFeatureCalcerType::CosDistanceWithClassCenter);
                }
                if (ComputeGaussianHomoscedatic) {
                    meta.Type.push_back(EFeatureCalcerType::GaussianHomoscedasticModel);
                }
                if (ComputeGaussianHeteroscedatic) {
                    meta.Type.push_back(EFeatureCalcerType::GaussianHeteroscedasticModel);
                }
            }
            return meta;
        }

        TEmbeddingOnlineFeatures CreateFeatureCalcer() const override {
            return TEmbeddingOnlineFeatures(
                Id(),
                GetTarget().NumClasses,
                Embedding,
                ComputeCosDistance,
                ComputeGaussianHomoscedatic,
                ComputeGaussianHeteroscedatic
            );
        }

        TEmbeddingFeaturesVisitor CreateCalcerVisitor() const override {
            return TEmbeddingFeaturesVisitor(GetTarget().NumClasses, Embedding->Dim());
        }

    private:
        TEmbeddingPtr Embedding;
        bool ComputeCosDistance = false;
        bool ComputeGaussianHomoscedatic = false;
        bool ComputeGaussianHeteroscedatic = false;
    };

    class TBagOfWordsEstimator final : public IFeatureEstimator {
    public:
        TBagOfWordsEstimator(
            TTextDataSetPtr learnTexts,
            TArrayRef<TTextDataSetPtr> testTexts,
            const NJson::TJsonValue& options)
            : LearnTexts({learnTexts})
            , TestTexts(testTexts.begin(), testTexts.end())
            , Dictionary(learnTexts->GetDictionary())
            , TopTokensCount("top_tokens_count", 2000)
        {
            if (options.Has(TopTokensCount.GetName())) {
                TopTokensCount.Set(FromString<ui32>(options[TopTokensCount.GetName()].GetString()));
            }
            CB_ENSURE(
                TopTokensCount > 0,
                "Parameter top_tokens_count for BagOfWords should be greater than zero"
            );
            const ui32 dictionarySize = Dictionary.Size();
            CB_ENSURE(
                dictionarySize > 0,
                "Dictionary size is 0, check out data or try to decrease occurrence_lower_bound parameter"
            );
            if (TopTokensCount > dictionarySize) {
                TopTokensCount = dictionarySize;
            }
        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            const ui32 featureCount = TopTokensCount;
            TEstimatedFeaturesMeta meta;
            meta.Type = TVector<EFeatureCalcerType>(featureCount, EFeatureCalcerType::BoW);
            meta.FeaturesCount = featureCount;
            meta.UniqueValuesUpperBoundHint = TVector<ui32>(featureCount, 2);
            return meta;
        }

        void ComputeFeatures(TCalculatedFeatureVisitor learnVisitor,
                             TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
                             NPar::TLocalExecutor* executor) const override {
            Calc(*executor, MakeConstArrayRef(LearnTexts), {learnVisitor});
            Calc(*executor, MakeConstArrayRef(TestTexts), testVisitors);
        }

        TGuid Id() const override {
            return Guid;
        }

        THolder<IFeatureCalcer> MakeFinalFeatureCalcer(
            TConstArrayRef<ui32> featureIndices,
            NPar::TLocalExecutor* executor) const override {

            Y_UNUSED(executor);

            TBagOfWordsCalcer calcer(Id(), Dictionary.Size());

            TVector<TTokenId> topTokens = Dictionary.GetTopTokens(TopTokensCount);

            TVector<ui32> remappedFeatureIndices;
            remappedFeatureIndices.reserve(featureIndices.size());

            for (ui32 featureIdx: featureIndices) {
                remappedFeatureIndices.push_back(static_cast<ui32>(topTokens[featureIdx]));
            }

            calcer.TrimFeatures(MakeConstArrayRef(remappedFeatureIndices));
            return MakeHolder<TBagOfWordsCalcer>(std::move(calcer));
        }

    protected:

        void Calc(NPar::TLocalExecutor& executor,
                  TConstArrayRef<TTextDataSetPtr> dataSets,
                  TConstArrayRef<TCalculatedFeatureVisitor> visitors) const {

            // TODO(d-kruchinin, noxoomo) better implementation:
            // add MaxRam option + bit mask compression for block on m features
            // estimation of all features in one pass
            THashSet<TTokenId> topTokensSet;
            for (TTokenId tokenId: Dictionary.GetTopTokens(TopTokensCount)) {
                topTokensSet.insert(tokenId);
            }

            for (ui32 id = 0; id < dataSets.size(); ++id) {
                const auto& ds = *dataSets[id];
                const ui64 samplesCount = ds.SamplesCount();

                const ui32 blockSize = sizeof(ui32);
                const ui32 numFeatureBins = (TopTokensCount + blockSize - 1) / blockSize;
                TVector<ui32> features(samplesCount * numFeatureBins);
                NPar::ParallelFor(
                    executor, 0, samplesCount, [&](ui32 line) {
                        for (const auto& tokenToCount : ds.GetText(line)) {
                            const TTokenId& token = tokenToCount.Token();
                            if (topTokensSet.contains(token)) {
                                SetFeatureValue(token, line, samplesCount, MakeArrayRef(features));
                            }
                        }
                    }
                );
                for (ui32 featureBin: xrange(numFeatureBins)) {
                    TVector<ui32> featureIds = xrange(
                        featureBin * blockSize,
                        Min((featureBin + 1) * blockSize, TopTokensCount.Get())
                    );
                    visitors[id](
                        featureIds,
                        TConstArrayRef<ui32>(
                            features.data() + featureBin * samplesCount,
                            samplesCount
                        )
                    );
                }
            }
        }

    private:
        void SetFeatureValue(
            ui32 featureIndex,
            ui32 docIndex,
            ui32 samplesCount,
            TArrayRef<ui32> features
        ) const {
            const ui32 binIndex = featureIndex / sizeof(ui32);
            const ui32 offset = featureIndex % sizeof(ui32);
            ui32& featurePack = features[binIndex * samplesCount + docIndex];
            featurePack |= 1 << offset;
        }

    private:
        TVector<TTextDataSetPtr> LearnTexts;
        TVector<TTextDataSetPtr> TestTexts;
        const TDictionaryProxy& Dictionary;
        const TGuid Guid = CreateGuid();
        NCatboostOptions::TOption<ui32> TopTokensCount;
    };
}

TVector<TOnlineFeatureEstimatorPtr> NCB::CreateEstimators(
    TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
    TEmbeddingPtr embedding,
    TTextClassificationTargetPtr target,
    TTextDataSetPtr learnTexts,
    TArrayRef<TTextDataSetPtr> testText) {

    TSet<EFeatureCalcerType> typesSet;
    for (auto& calcerDescription: featureCalcerDescription) {
        typesSet.insert(calcerDescription.CalcerType);
    }

    TVector<TOnlineFeatureEstimatorPtr> estimators;

    if (typesSet.contains(EFeatureCalcerType::NaiveBayes)) {
        estimators.push_back(new TNaiveBayesEstimator(target, learnTexts, testText));
    }
    if (typesSet.contains(EFeatureCalcerType::BM25)) {
        estimators.push_back(new TBM25Estimator(target, learnTexts, testText));
    }
    TSet<EFeatureCalcerType> embeddingEstimators = {
        EFeatureCalcerType::GaussianHomoscedasticModel,
        EFeatureCalcerType::GaussianHeteroscedasticModel,
        EFeatureCalcerType::CosDistanceWithClassCenter
    };

    TSet<EFeatureCalcerType> enabledEmbeddingCalculators;
    SetIntersection(
        typesSet.begin(), typesSet.end(),
        embeddingEstimators.begin(), embeddingEstimators.end(),
        std::inserter(enabledEmbeddingCalculators, enabledEmbeddingCalculators.end()));

    if (!enabledEmbeddingCalculators.empty()) {
        estimators.push_back(new TEmbeddingOnlineFeaturesEstimator(embedding, target, learnTexts, testText, enabledEmbeddingCalculators));
    }
    return estimators;
}

TVector<TFeatureEstimatorPtr> NCB::CreateEstimators(
    TConstArrayRef<NCatboostOptions::TFeatureCalcerDescription> featureCalcerDescription,
    TEmbeddingPtr embedding,
    TTextDataSetPtr learnTexts,
    TArrayRef<TTextDataSetPtr> testText) {

    Y_UNUSED(embedding);

    TVector<TFeatureEstimatorPtr> estimators;
    for (auto& calcerDescription: featureCalcerDescription) {
        if (calcerDescription.CalcerType == EFeatureCalcerType::BoW) {
            estimators.push_back(
                new TBagOfWordsEstimator(
                    learnTexts,
                    testText,
                    calcerDescription.CalcerOptions
                )
            );
        }
    }

    return estimators;
}
