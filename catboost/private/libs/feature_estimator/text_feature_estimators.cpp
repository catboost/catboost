#include "text_feature_estimators.h"
#include "base_text_feature_estimator.h"

#include <catboost/private/libs/text_features/text_feature_calcers.h>
#include <catboost/private/libs/text_processing/embedding.h>

#include <catboost/private/libs/options/enum_helpers.h>

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
            TArrayRef<TTextDataSetPtr> testTexts)
            : LearnTexts({learnTexts})
            , TestTexts(testTexts.begin(), testTexts.end())
            , Dictionary(learnTexts->GetDictionary())
        {}

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            const ui32 featureCount = Dictionary.Size();
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
            calcer.TrimFeatures(featureIndices);
            return MakeHolder<TBagOfWordsCalcer>(std::move(calcer));
        }

    protected:

        void Calc(NPar::TLocalExecutor& executor,
                  TConstArrayRef<TTextDataSetPtr> dataSets,
                  TConstArrayRef<TCalculatedFeatureVisitor> visitors) const {
            const ui32 featuresCount = Dictionary.Size();

            // TODO(d-kruchinin, noxoomo) better implementation:
            // add MaxRam option + bit mask compression for block on m features
            // estimation of all features in one pass

            for (ui32 id = 0; id < dataSets.size(); ++id) {
                const auto& ds = *dataSets[id];
                const ui64 samplesCount = ds.SamplesCount();

                //one-by-one, we don't want to acquire unnecessary RAM for very sparse features
                TVector<float> singleFeature(samplesCount);
                for (ui32 tokenId = 0; tokenId < featuresCount; ++tokenId) {
                    NPar::ParallelFor(
                        executor, 0, samplesCount, [&](ui32 line) {
                            const bool hasToken = ds.GetText(line).Has(TTokenId(tokenId));
                            singleFeature[line] = static_cast<float>(hasToken);
                        }
                    );
                    visitors[id](tokenId, singleFeature);
                }
            }
        }

    private:
        TVector<TTextDataSetPtr> LearnTexts;
        TVector<TTextDataSetPtr> TestTexts;
        const TDictionaryProxy& Dictionary;
        const TGuid Guid = CreateGuid();
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
    TSet<EFeatureCalcerType> typesSet;
    for (auto& calcerDescription: featureCalcerDescription) {
        typesSet.insert(calcerDescription.CalcerType);
    }

    TVector<TFeatureEstimatorPtr> estimators;
    if (typesSet.contains(EFeatureCalcerType::BoW)) {
        estimators.push_back(new TBagOfWordsEstimator(learnTexts, testText));
    }
    return estimators;
}
