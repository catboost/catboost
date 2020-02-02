#include "naive_bayesian.h"
#include "helpers.h"

#include <catboost/private/libs/text_features/flatbuffers/feature_calcers.fbs.h>

#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>

using namespace NCB;

TTextFeatureCalcerFactory::TRegistrator<TMultinomialNaiveBayes>
    NaiveBayesRegistrator(EFeatureCalcerType::NaiveBayes);

double TMultinomialNaiveBayes::LogProb(
    const TDenseHash<TTokenId, ui32>& freqTable,
    double classSamples,
    double classTokensCount,
    const TText& text) const {

    Y_UNUSED(text, freqTable);
    double value = log(classSamples + ClassPrior);

    classTokensCount += TokenPrior * (NumSeenTokens + SEEN_TOKENS_PRIOR);
    double textLen = 0;

    for (const auto& tokenToCount : text) {
        textLen += tokenToCount.Count();

        auto tokenCountPtr = freqTable.find(tokenToCount.Token());
        double num = TokenPrior;

        if (tokenCountPtr != freqTable.end()) {
            num += tokenCountPtr->second;
        } else {
            //unseen word, adjust prior
            classTokensCount += TokenPrior;
        }
        value += tokenToCount.Count() * log(num);
    }

    //denum
    value -= textLen * log(classTokensCount);

    return value;
}

void TMultinomialNaiveBayes::Compute(
    const TText& text,
    TOutputFloatIterator outputFeaturesIterator) const {

    TVector<double> logProbs(NumClasses);
    for (ui32 clazz = 0; clazz < NumClasses; ++clazz) {
        logProbs[clazz] = LogProb(Frequencies[clazz], ClassDocs[clazz], ClassTotalTokens[clazz], text);
    }
    Softmax(logProbs);

    ForEachActiveFeature(
        [&logProbs, &outputFeaturesIterator](ui32 featureId) {
            *outputFeaturesIterator = logProbs[featureId];
            ++outputFeaturesIterator;
        }
    );
}

TTextFeatureCalcer::TFeatureCalcerFbs TMultinomialNaiveBayes::SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const {
    using namespace NCatBoostFbs;

    // TODO(d-kruchinin, kirillovs) change types in flatbuffers to arcadian
    static_assert(sizeof(ui32) == sizeof(uint32_t));
    const auto& fbsClassDocs = builder.CreateVector(
        reinterpret_cast<const uint32_t*>(ClassDocs.data()),
        ClassDocs.size()
    );

    static_assert(sizeof(ui64) == sizeof(uint64_t));
    const auto& fbsClassTotalTokens = builder.CreateVector(
        reinterpret_cast<const uint64_t*>(ClassTotalTokens.data()),
        ClassTotalTokens.size()
    );

    const auto& fbsNaiveBayes = CreateTNaiveBayes(
        builder,
        NumClasses,
        ClassPrior,
        TokenPrior,
        NumSeenTokens,
        fbsClassDocs,
        fbsClassTotalTokens
    );

    return TFeatureCalcerFbs(TAnyFeatureCalcer_TNaiveBayes, fbsNaiveBayes.Union());
}

void TMultinomialNaiveBayes::LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer* calcer) {
    auto naiveBayes = calcer->FeatureCalcerImpl_as_TNaiveBayes();

    NumClasses = naiveBayes->NumClasses();
    ClassPrior = naiveBayes->ClassPrior();
    TokenPrior = naiveBayes->TokenPrior();
    NumSeenTokens = naiveBayes->NumSeenTokens();

    auto fbsClassDocs = naiveBayes->ClassDocs();
    ClassDocs.yresize(fbsClassDocs->size());
    Copy(fbsClassDocs->begin(), fbsClassDocs->end(), ClassDocs.begin());

    auto fbsClassTotalTokens = naiveBayes->ClassTotalTokens();
    ClassTotalTokens.yresize(fbsClassTotalTokens->size());
    Copy(fbsClassTotalTokens->begin(), fbsClassTotalTokens->end(), ClassTotalTokens.begin());
}

void TMultinomialNaiveBayes::SaveLargeParameters(IOutputStream* stream) const {
    ::Save(stream, Frequencies);
}

void TMultinomialNaiveBayes::LoadLargeParameters(IInputStream* stream) {
    ::Load(stream, Frequencies);
}

void TNaiveBayesVisitor::Update(ui32 classId, const TText& text, TTextFeatureCalcer* calcer) {
    auto naiveBayes = dynamic_cast<TMultinomialNaiveBayes*>(calcer);
    Y_ASSERT(naiveBayes);

    auto& classCounts = naiveBayes->Frequencies[classId];

    for (const auto& tokenToCount : text) {
        SeenTokens.Insert(tokenToCount.Token());
        classCounts[tokenToCount.Token()] += tokenToCount.Count();
        naiveBayes->ClassTotalTokens[classId] += tokenToCount.Count();
    }
    naiveBayes->ClassDocs[classId] += 1;
    naiveBayes->NumSeenTokens = SeenTokens.Size();
}
