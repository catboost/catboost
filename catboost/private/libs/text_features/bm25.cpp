#include "bm25.h"

#include <catboost/private/libs/text_features/flatbuffers/feature_calcers.fbs.h>

#include <util/generic/ymath.h>

using namespace NCB;

TTextFeatureCalcerFactory::TRegistrator<TBM25> BM25Registrator(EFeatureCalcerType::BM25);

template <class T>
static inline ui32 NonZeros(TConstArrayRef<T> arr) {
    ui32 result = 0;
    for (const auto& val : arr) {
        result += static_cast<T>(0) != val;
    }
    return result;
}

static inline double CalcTruncatedInvClassFreq(TConstArrayRef<ui32> inClassFreq, double eps) {
    double classesWithTerm = NonZeros(inClassFreq);
    const auto numClasses = inClassFreq.size();
    return Max<double>(log(numClasses - classesWithTerm + 0.5) - log(classesWithTerm + 0.5), eps);
}

static inline void ExtractTermFreq(TConstArrayRef<TDenseHash<TTokenId, ui32>> freq,
                                   TTokenId term,
                                   TArrayRef<ui32> termFreq) {
    for (ui32 clazz = 0; clazz < freq.size(); ++clazz) {
        const auto& classFreqTable = freq[clazz];
        auto termFreqIt = classFreqTable.find(term);
        termFreq[clazz] = termFreqIt != classFreqTable.end() ? termFreqIt->second : 0;
    }
}

static inline double Score(double termFreq, double k, double b, double meanLength, double classLength) {
    return termFreq * (k + 1) / (termFreq + k * (1.0 - b + b * meanLength / classLength));
}

void TBM25::Compute(const TText& text, TOutputFloatIterator iterator) const {
    TVector<ui32> termFreqInClass(NumClasses);
    TVector<double> scores(NumClasses);

    for (const auto& tokenToCount : text) {
        ExtractTermFreq(Frequencies, tokenToCount.Token(), termFreqInClass);
        double inverseClassFreq = CalcTruncatedInvClassFreq(termFreqInClass, TruncateBorder);
        double meanClassLength = TotalTokens * 1.0 / NumClasses;

        for (ui32 clazz = 0; clazz < NumClasses; ++clazz) {
            scores[clazz] += inverseClassFreq * Score(termFreqInClass[clazz], K, B, meanClassLength,  ClassTotalTokens[clazz]);
        }
    }

    ForEachActiveFeature(
        [&scores, &iterator](ui32 featureId){
            *iterator = scores[featureId];
            ++iterator;
        }
    );
}

TTextFeatureCalcer::TFeatureCalcerFbs TBM25::SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const {
    using namespace NCatBoostFbs;

    // TODO(d-kruchinin, kirillovs) change flatbuffer types to arcadian
    static_assert(sizeof(uint64_t) == sizeof(ui64));

    auto fbClassTotalTokens = builder.CreateVector(
        reinterpret_cast<const uint64_t*>(ClassTotalTokens.data()),
        ClassTotalTokens.size()
    );
    const auto& fbBm25 = CreateTBM25(
        builder,
        NumClasses,
        K,
        B,
        TruncateBorder,
        TotalTokens,
        fbClassTotalTokens
    );
    return TFeatureCalcerFbs(TAnyFeatureCalcer_TBM25, fbBm25.Union());
}

void TBM25::LoadParametersFromFB(const NCatBoostFbs::TFeatureCalcer* calcer) {
    auto fbBm25 = calcer->FeatureCalcerImpl_as_TBM25();
    NumClasses = fbBm25->NumClasses();
    K = fbBm25->ParamK();
    B = fbBm25->ParamB();
    TruncateBorder = fbBm25->TruncateBorder();
    TotalTokens = fbBm25->TotalTokens();

    auto classTotalTokens = fbBm25->ClassTotalTokens();
    ClassTotalTokens.yresize(classTotalTokens->size());

    static_assert(sizeof(uint64_t) == sizeof(ui64));
    Copy(classTotalTokens->begin(), classTotalTokens->end(), ClassTotalTokens.begin());
}

void TBM25::SaveLargeParameters(IOutputStream* stream) const {
    ::Save(stream, Frequencies);
}

void TBM25::LoadLargeParameters(IInputStream* stream) {
    ::Load(stream, Frequencies);
}

void TBM25Visitor::Update(ui32 classId, const TText& text, TTextFeatureCalcer* calcer) {
    auto bm25 = dynamic_cast<TBM25*>(calcer);
    Y_ASSERT(bm25);

    auto& classCounts = bm25->Frequencies[classId];

    for (const auto& tokenToCount : text) {
        const ui32 count = tokenToCount.Count();
        classCounts[tokenToCount.Token()] += count;
        bm25->ClassTotalTokens[classId] += count;
        bm25->TotalTokens += count;
    }
}
