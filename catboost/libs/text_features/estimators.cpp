#include "estimators.h"
#include "naive_bayesian.h"
#include "bm25.h"
#include "embedding.h"
#include "embedding_online_features.h"
#include "embedding_loader.h"
#include <library/containers/dense_hash/dense_hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/set.h>

using namespace NCB;


namespace {
    //TODO(noxoomo): we could fuse estimation in one pass for naive bayes and bm25
    template <class TEstimatorImpl>
    class TBaseEstimator : public IOnlineFeatureEstimator {
    public:
        TBaseEstimator(
            TTextClassificationTargetPtr target,
            TTextDataSetPtr learnTexts,
            TVector<TTextDataSetPtr> testTexts)
            : Target(std::move(target))
              , LearnTexts(std::move(learnTexts))
              , TestTexts(std::move(testTexts)) {

        }


        void ComputeFeatures(
            TCalculatedFeatureVisitor learnVisitor,
            TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
            NPar::TLocalExecutor*) const override {
            auto estimator = CreateEstimator();
            {
                const auto& ds = GetLearn();
                auto& target = GetTarget();
                const ui64 samplesCount = ds.SamplesCount();
                for (ui64 line = 0; line < samplesCount; ++line) {
                    estimator.AddText(target.Classes[line], ds.GetText(line));
                }

                TVector<TTextDataSetPtr> learnDs{GetLearnPtr()};
                TVector<TCalculatedFeatureVisitor> learnVisitors{std::move(learnVisitor)};
                Calc(estimator, learnDs, learnVisitors);
            }
            if (!testVisitors.empty()) {
                CB_ENSURE(testVisitors.size() == NumberOfTests(),
                          "If specified, testVisitors should be the same number as test sets");
                Calc(estimator, GetTests(), testVisitors);
            }
        }

        void ComputeOnlineFeatures(
            TConstArrayRef<ui32> learnPermutation,
            TCalculatedFeatureVisitor learnVisitor,
            TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
            NPar::TLocalExecutor*) const override {
            const ui32 featuresCount = GetFeaturesCount();
            auto estimator = CreateEstimator();
            {
                const auto& ds = GetLearn();
                auto& target = GetTarget();
                const ui64 samplesCount = ds.SamplesCount();
                TVector<TVector<float>> learnFeatures(featuresCount, TVector<float>(samplesCount));

                for (ui64 line  : learnPermutation) {
                    auto textFeatures = estimator.CalcFeaturesAndAddText(target.Classes[line], ds.GetText(line));
                    for (ui32 f = 0; f < featuresCount; ++f) {
                        learnFeatures[f][line] = static_cast<float>(textFeatures[f]);
                    }
                }
                for (ui32 f = 0; f < featuresCount; ++f) {
                    learnVisitor(f, learnFeatures[f]);
                }
            }
            if (!testVisitors.empty()) {
                CB_ENSURE(testVisitors.size() == NumberOfTests(),
                          "If specified, testVisitors should be the same number as test sets");
                Calc(estimator, GetTests(), testVisitors);
            }
        }
    protected:

        void Calc(const TEstimatorImpl& estimator,
                  TConstArrayRef<TTextDataSetPtr> dataSets,
                  TConstArrayRef<TCalculatedFeatureVisitor> visitors) const {
            const ui32 featuresCount = static_cast<const ui32>(GetFeaturesCount());
            for (ui32 id = 0; id < dataSets.size(); ++id) {
                const auto& ds = *dataSets[id];
                const ui64 samplesCount = ds.SamplesCount();
                TVector<TVector<float>> features(featuresCount, TVector<float>(samplesCount));

                for (ui64 line = 0; line < samplesCount; ++line) {
                    auto textFeatures = estimator.CalcFeatures(ds.GetText(line));
                    for (ui32 f = 0; f < featuresCount; ++f) {
                        features[f][line] = static_cast<float>(textFeatures[f]);
                    }
                }
                for (ui32 f = 0; f < featuresCount; ++f) {
                    visitors[id](f, features[f]);
                }
            }
        }

        virtual TEstimatorImpl CreateEstimator() const = 0;

        virtual ui64 GetFeaturesCount() const = 0;


        const TTextClassificationTarget& GetTarget() const {
            return *Target;
        }

        const TTextDataSet& GetLearn() const {
            return *LearnTexts;
        }

        TTextDataSetPtr GetLearnPtr() const {
            return LearnTexts;
        }

        ui32 NumberOfTests() const {
            return TestTexts.size();
        }

        const TTextDataSet& GetTest(ui32 idx) const {
            CB_ENSURE(idx < TestTexts.size(),
                      "Test dataset idx is out of bounds " << idx << " (tests count " << TestTexts.size() << ")");
            return *TestTexts[idx];
        }

        TConstArrayRef<TTextDataSetPtr> GetTests() const {
            return TestTexts;
        }
    private:
        TTextClassificationTargetPtr Target;
        TTextDataSetPtr LearnTexts;
        TVector<TTextDataSetPtr> TestTexts;
    };

    class TNaiveBayesEstimator final : public TBaseEstimator<TMultinomialOnlineNaiveBayes> {
    public:
        TNaiveBayesEstimator(
            TTextClassificationTargetPtr target,
            TTextDataSetPtr learnTexts,
            TVector<TTextDataSetPtr> testText)
            : TBaseEstimator(target, learnTexts, std::move(testText)) {

        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = GetFeaturesCount();
            meta.Type.resize(meta.FeaturesCount, EFeatureCalculatorType::NaiveBayes);
            return meta;
        }

    protected:
        ui64 GetFeaturesCount() const override {
            return GetTarget().NumClasses > 2 ? GetTarget().NumClasses : 1;
        }

        TMultinomialOnlineNaiveBayes CreateEstimator() const {
            return TMultinomialOnlineNaiveBayes(GetTarget().NumClasses);
        }

    };

    class TBM25Estimator final : public TBaseEstimator<TOnlineBM25> {
    public:
        TBM25Estimator(
            TTextClassificationTargetPtr target,
            TTextDataSetPtr learnTexts,
            TVector<TTextDataSetPtr> testText)
            : TBaseEstimator(std::move(target), std::move(learnTexts), std::move(testText)) {

        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = GetFeaturesCount();
            meta.Type.resize(meta.FeaturesCount, EFeatureCalculatorType::BM25);
            return meta;
        }


    protected:
        TOnlineBM25 CreateEstimator() const {
            return TOnlineBM25(GetTarget().NumClasses, 1e-3);
        }
        ui64 GetFeaturesCount() const override {
            return GetTarget().NumClasses;
        }
    };


    class TEmbeddingOnlineFeaturesEstimator final : public TBaseEstimator<TEmbeddingOnlineFeatures> {
    public:
        TEmbeddingOnlineFeaturesEstimator(
            TEmbeddingPtr embedding,
            TTextClassificationTargetPtr target,
            TTextDataSetPtr learnTexts,
            TVector<TTextDataSetPtr> testText,
            TSet<EFeatureCalculatorType> enabledTypes)
            : TBaseEstimator(std::move(target), std::move(learnTexts), std::move(testText))
            , Embedding(embedding)
            , EnabledTypes(enabledTypes){

        }

        TEstimatedFeaturesMeta FeaturesMeta() const override {
            TEstimatedFeaturesMeta meta;
            meta.FeaturesCount = GetFeaturesCount();

            for (ui32 classIdx = 0; classIdx < GetTarget().NumClasses; ++classIdx) {
                if (EnabledTypes.contains(EFeatureCalculatorType::CosDistanceWithClassCenter)) {
                    meta.Type.push_back(EFeatureCalculatorType::CosDistanceWithClassCenter);
                }
                if (EnabledTypes.contains(EFeatureCalculatorType::GaussianHomoscedasticModel)) {
                    meta.Type.push_back(EFeatureCalculatorType::GaussianHomoscedasticModel);
                }
                if (EnabledTypes.contains(EFeatureCalculatorType::GaussianHeteroscedasticiModel)) {
                    meta.Type.push_back(EFeatureCalculatorType::GaussianHeteroscedasticiModel);
                }
            }
            return meta;
        }

    protected:
        TEmbeddingOnlineFeatures CreateEstimator() const {
            return TEmbeddingOnlineFeatures(GetTarget().NumClasses, Embedding);
        }
        ui64 GetFeaturesCount() const override {
            return GetTarget().NumClasses * EnabledTypes.size();
        }

    private:
        TEmbeddingPtr Embedding;
        TSet<EFeatureCalculatorType> EnabledTypes;
    };
}


TVector<TOnlineFeatureEstimatorPtr> NCB::CreateEstimators(
    TConstArrayRef<EFeatureCalculatorType> type,
    TEmbeddingPtr embedding,
    TTextClassificationTargetPtr target,
    TTextDataSetPtr learnTexts,
    TVector<TTextDataSetPtr> testText) {
    TSet<EFeatureCalculatorType> typesSet(type.begin(), type.end());

    TVector<TOnlineFeatureEstimatorPtr> estimators;

    if (typesSet.contains(EFeatureCalculatorType::NaiveBayes)) {
        estimators.push_back(new TNaiveBayesEstimator(target, learnTexts, testText));
    }
    if (typesSet.contains(EFeatureCalculatorType::BM25)) {
        estimators.push_back(new TBM25Estimator(target, learnTexts, testText));
    }
    TSet<EFeatureCalculatorType> embeddingCalculators = { EFeatureCalculatorType::GaussianHomoscedasticModel,
                                                          EFeatureCalculatorType::GaussianHomoscedasticModel,
                                                          EFeatureCalculatorType::CosDistanceWithClassCenter};

    TSet<EFeatureCalculatorType> enabledEmbeddingCalculators;
    SetIntersection(
        typesSet.begin(), typesSet.end(),
        embeddingCalculators.begin(), embeddingCalculators.end(),
        std::inserter(enabledEmbeddingCalculators,enabledEmbeddingCalculators.end()));

    if (!enabledEmbeddingCalculators.empty()) {
        estimators.push_back(new TEmbeddingOnlineFeaturesEstimator(embedding, target, learnTexts, testText, enabledEmbeddingCalculators));
    }
    return estimators;
}
