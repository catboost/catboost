#include "bow.h"

void NCB::TBagOfWordsEstimator::Calc(NPar::TLocalExecutor& executor,
                                     TConstArrayRef<TTextDataSetPtr> dataSets,
                                     TConstArrayRef<TCalculatedFeatureVisitor> visitors) const {

    const ui32 featuresCount = Dictionary.Size();

    for (ui32 id = 0; id < dataSets.size(); ++id) {
        const auto& ds = *dataSets[id];
        const ui64 samplesCount = ds.SamplesCount();

        //one-by-one, we don't want to acquire unnecessary RAM for very sparse features
        TVector<float> singleFeature(samplesCount);
        for (ui32 tokenId = 0; tokenId < featuresCount; ++tokenId) {
            NPar::ParallelFor(executor, 0, samplesCount, [&](ui32 line) {
                const bool hasToken = ds.GetText(line).Has(TTokenId(tokenId));
                if (hasToken) {
                    singleFeature[line] = 1.0;
                }
            });
            visitors[id](tokenId, singleFeature);
        }
    }
}

void NCB::TBagOfWordsEstimator::ComputeFeatures(TCalculatedFeatureVisitor learnVisitor,
                                                TConstArrayRef<TCalculatedFeatureVisitor> testVisitors,
                                                NPar::TLocalExecutor* executor) const {

    Calc(*executor, MakeConstArrayRef(LearnTexts), {learnVisitor});
    Calc(*executor, MakeConstArrayRef(TestTexts), testVisitors);
}
