#pragma once

#include "classification_target.h"
#include "feature_estimator.h"

#include <catboost/private/libs/embeddings/embedding_dataset.h>


namespace NCB {

    template <class TFeatureCalcer, class TCalcerVisitor>
    class TEmbeddingBaseEstimator : public IOnlineFeatureEstimator {
    public:
        TEmbeddingBaseEstimator(
            TConstArrayRef<float> target,
            TClassificationTargetPtr classificationTarget, // can be nullptr
            TEmbeddingDataSetPtr learnArrays,
            TArrayRef<TEmbeddingDataSetPtr> testArrays)
            : Target(target)
            , ClassificationTarget(classificationTarget)
            , LearnArrays(learnArrays)
            , TestArrays(testArrays.begin(), testArrays.end())
            , Guid(CreateGuid()) {
        }

        void ComputeFeatures(
            TCalculatedFeatureVisitor learnVisitor,
            TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
            NPar::ILocalExecutor*) const override {

            THolder<TFeatureCalcer> featureCalcer = EstimateFeatureCalcer();

            TVector<TEmbeddingDataSetPtr> learnDataset{GetLearnDatasetPtr()};
            TVector<TCalculatedFeatureVisitor> learnVisitors{std::move(learnVisitor)};
            Calc(*featureCalcer, learnDataset, learnVisitors);

            if (!testVisitors.empty()) {
                CB_ENSURE(testVisitors.size() == NumberOfTestDatasets(),
                          "If specified, testVisitors should be the same number as test sets");
                Calc(*featureCalcer, GetTestDatasets(), testVisitors);
            }
        }

        void ComputeOnlineFeatures(
            TConstArrayRef<ui32> learnPermutation,
            TCalculatedFeatureVisitor learnVisitor,
            TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
            NPar::ILocalExecutor*) const override {

            TFeatureCalcer featureCalcer = CreateFeatureCalcer();
            TCalcerVisitor calcerVisitor = CreateCalcerVisitor();

            const ui32 featuresCount = featureCalcer.FeatureCount();

            {
                const auto& learnDataset = GetLearnDataset();
                const auto& target = GetTarget();
                const ui64 samplesCount = learnDataset.SamplesCount();
                TVector<float> learnFeatures(featuresCount * samplesCount);

                for (ui64 line : learnPermutation) {
                    const TEmbeddingsArray& vector = learnDataset.GetVector(line);

                    Compute(featureCalcer, vector, line, samplesCount, learnFeatures);
                    calcerVisitor.Update(target[line], vector, &featureCalcer);
                }
                for (ui32 f = 0; f < featuresCount; ++f) {
                    learnVisitor(
                        f,
                        TConstArrayRef<float>(
                            learnFeatures.data() + f * samplesCount,
                            learnFeatures.data() + (f + 1) * samplesCount
                        )
                    );
                }
            }
            if (!testVisitors.empty()) {
                CB_ENSURE(testVisitors.size() == NumberOfTestDatasets(),
                          "If specified, testVisitors should be the same number as test sets");
                Calc(featureCalcer, GetTestDatasets(), testVisitors);
            }
        }

        virtual EFeatureType GetSourceType() const override {
            return EFeatureType::Embedding;
        }

        void ComputeOnlineFeatures();

        TGuid Id() const override {
            return Guid;
        }

        THolder<IFeatureCalcer> MakeFinalFeatureCalcer(
            TConstArrayRef<ui32> featureIndices,
            NPar::ILocalExecutor* executor) const override {

            Y_UNUSED(executor);

            THolder<TFeatureCalcer> calcer = EstimateFeatureCalcer();
            calcer->SetId(Id());
            calcer->TrimFeatures(featureIndices);
            return calcer;
        }

    protected:
        void Calc(
            const TFeatureCalcer& featureCalcer,
            TConstArrayRef<TEmbeddingDataSetPtr> datasets,
            TConstArrayRef<TCalculatedFeatureVisitor> visitors) const {

            const ui32 featuresCount = featureCalcer.FeatureCount();
            for (ui32 id = 0; id < datasets.size(); ++id) {
                const auto& currentDataset = *datasets[id];
                const ui64 samplesCount = currentDataset.SamplesCount();
                TVector<float> features(featuresCount * samplesCount);

                for (ui64 line = 0; line < samplesCount; ++line) {
                    Compute(featureCalcer, currentDataset.GetVector(line), line, samplesCount, features);
                }

                for (ui32 f = 0; f < featuresCount; ++f) {
                    visitors[id](
                        f,
                        TConstArrayRef<float>(
                            features.data() + f * samplesCount,
                            features.data() + (f + 1) * samplesCount
                        )
                    );
                }
            }
        }

        virtual TFeatureCalcer CreateFeatureCalcer() const = 0;
        virtual TCalcerVisitor CreateCalcerVisitor() const = 0;

        THolder<TFeatureCalcer> EstimateFeatureCalcer() const {
            THolder<TFeatureCalcer> featureCalcer = MakeHolder<TFeatureCalcer>(CreateFeatureCalcer());
            TCalcerVisitor calcerVisitor = CreateCalcerVisitor();

            const auto& learnDataset = GetLearnDataset();
            const auto& target = GetTarget();

            const ui64 samplesCount = learnDataset.SamplesCount();
            for (ui64 line = 0; line < samplesCount; ++line) {
                const TEmbeddingsArray& vector = learnDataset.GetVector(line);
                calcerVisitor.Update(target[line], vector, featureCalcer.Get());
            }

            return featureCalcer;
        }

        void Compute(
            const TFeatureCalcer& featureCalcer,
            const TEmbeddingsArray& vector,
            ui64 docId,
            ui64 docCount,
            TArrayRef<float> features) const {

            auto outputFeaturesIterator = TOutputFloatIterator(
                features.data() + docId,
                docCount,
                features.size()
            );

            featureCalcer.Compute(vector, outputFeaturesIterator);
        }

        const TConstArrayRef<float> GetTarget() const {
            return Target;
        }

        const TClassificationTargetPtr GetClassificationTarget() const {
            return ClassificationTarget;
        }

        const TEmbeddingDataSet& GetLearnDataset() const {
            return *LearnArrays;
        }

        TEmbeddingDataSetPtr GetLearnDatasetPtr() const {
            return LearnArrays;
        }

        ui32 NumberOfTestDatasets() const {
            return TestArrays.size();
        }

        const TEmbeddingDataSet& GetTestDataset(ui32 idx) const {
            CB_ENSURE(idx < TestArrays.size(),
                      "Test dataset idx is out of bounds " << idx << " (tests count " << TestArrays.size()
                                                           << ")");
            return *TestArrays[idx];
        }

        TConstArrayRef<TEmbeddingDataSetPtr> GetTestDatasets() const {
            return TestArrays;
        }


    private:
        TConstArrayRef<float> Target;
        TClassificationTargetPtr ClassificationTarget; // can be nullptr
        TEmbeddingDataSetPtr LearnArrays;
        TVector<TEmbeddingDataSetPtr> TestArrays;
        const TGuid Guid;
    };
};
