#pragma once

#include "classification_target.h"
#include "feature_estimator.h"

#include <catboost/private/libs/text_processing/text_dataset.h>


namespace NCB {
    //TODO(noxoomo): we could fuse estimation in one pass for naive bayes and bm25
    template <class TFeatureCalcer, class TCalcerVisitor>
    class TTextBaseEstimator : public IOnlineFeatureEstimator {
    public:
        TTextBaseEstimator(
            TClassificationTargetPtr target,
            TTextDataSetPtr learnTexts,
            TArrayRef<TTextDataSetPtr> testTexts)
            : Target(std::move(target))
            , LearnTexts(std::move(learnTexts))
            , TestTexts(testTexts.begin(), testTexts.end())
            , Guid(CreateGuid()) {
        }

        void ComputeFeatures(
            TCalculatedFeatureVisitor learnVisitor,
            TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
            NPar::ILocalExecutor*) const override {

            THolder<TFeatureCalcer> featureCalcer = EstimateFeatureCalcer();

            TVector<TTextDataSetPtr> learnDs{GetLearnDataSetPtr()};
            TVector<TCalculatedFeatureVisitor> learnVisitors{std::move(learnVisitor)};
            Calc(*featureCalcer, learnDs, learnVisitors);

            if (!testVisitors.empty()) {
                CB_ENSURE(testVisitors.size() == NumberOfTestDataSets(),
                          "If specified, testVisitors should be the same number as test sets");
                Calc(*featureCalcer, GetTestDataSets(), testVisitors);
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
                const auto& ds = GetLearnDataSet();
                const auto& target = GetTarget();
                const ui64 samplesCount = ds.SamplesCount();
                TVector<float> learnFeatures(featuresCount * samplesCount);

                for (ui64 line : learnPermutation) {
                    const TText& text = ds.GetText(line);

                    Compute(featureCalcer, text, line, samplesCount, learnFeatures);
                    calcerVisitor.Update(target.Classes[line], text, &featureCalcer);
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
                CB_ENSURE(testVisitors.size() == NumberOfTestDataSets(),
                          "If specified, testVisitors should be the same number as test sets");
                Calc(featureCalcer, GetTestDataSets(), testVisitors);
            }
        }

        virtual EFeatureType GetSourceType() const override {
            return EFeatureType::Text;
        }

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
            TConstArrayRef<TTextDataSetPtr> dataSets,
            TConstArrayRef<TCalculatedFeatureVisitor> visitors) const {

            const ui32 featuresCount = featureCalcer.FeatureCount();
            for (ui32 id = 0; id < dataSets.size(); ++id) {
                const auto& ds = *dataSets[id];
                const ui64 samplesCount = ds.SamplesCount();
                TVector<float> features(featuresCount * samplesCount);

                for (ui64 line = 0; line < samplesCount; ++line) {
                    Compute(featureCalcer, ds.GetText(line), line, samplesCount, features);
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

            const auto& ds = GetLearnDataSet();
            const auto& target = GetTarget();

            const ui64 samplesCount = ds.SamplesCount();
            for (ui64 line = 0; line < samplesCount; ++line) {
                const TText& text = ds.GetText(line);
                calcerVisitor.Update(target.Classes[line], text, featureCalcer.Get());
            }

            return featureCalcer;
        }

        void Compute(
            const TFeatureCalcer& featureCalcer,
            const TText& text,
            ui64 docId,
            ui64 docCount,
            TArrayRef<float> features) const {

            TOutputFloatIterator outputFeaturesIterator(
                features.data() + docId,
                docCount,
                features.size()
            );

            featureCalcer.Compute(text, outputFeaturesIterator);
        }

        const TClassificationTarget& GetTarget() const {
            return *Target;
        }

        const TTextDataSet& GetLearnDataSet() const {
            return *LearnTexts;
        }

        TTextDataSetPtr GetLearnDataSetPtr() const {
            return LearnTexts;
        }

        ui32 NumberOfTestDataSets() const {
            return TestTexts.size();
        }

        const TTextDataSet& GetTestDataSet(ui32 idx) const {
            CB_ENSURE(idx < TestTexts.size(),
                      "Test dataset idx is out of bounds " << idx << " (tests count " << TestTexts.size()
                                                           << ")");
            return *TestTexts[idx];
        }

        TConstArrayRef<TTextDataSetPtr> GetTestDataSets() const {
            return TestTexts;
        }

    private:
        TClassificationTargetPtr Target;
        TTextDataSetPtr LearnTexts;
        TVector<TTextDataSetPtr> TestTexts;
        const TGuid Guid;
    };
}
