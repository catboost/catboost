#include "bm25.h"

#include <catboost/private/libs/text_features/flatbuffers/feature_calcers.fbs.h>

#include <util/generic/xrange.h>
#include <util/generic/ymath.h>

using namespace NCB;

TTextFeatureCalcerFactory::TRegistrator<TBM25> BM25Registrator(EFeatureCalcerType::BM25);

static inline double CalcTruncatedInvClassFreq(ui32 numClasses, ui32 classesWithTerm, double eps) {
    return Max<double>(log((numClasses - classesWithTerm + 0.5) / (classesWithTerm + 0.5)), eps);
}

static inline ui32 ExtractTermFreq(TConstArrayRef<TDenseHash<TTokenId, ui32>> freq,
                                   TTokenId term,
                                   TArrayRef<ui32> termFreq) {
    ui32 nonZeroCount = 0;
    for (ui32 clazz = 0; clazz < freq.size(); ++clazz) {
        const auto& classFreqTable = freq[clazz];
        const auto termFreqIt = classFreqTable.find(term);
        if (termFreqIt != classFreqTable.end()) {
            termFreq[clazz] = termFreqIt->second;
            ++nonZeroCount;
        } else {
            termFreq[clazz] = 0;
        }
    }
    return nonZeroCount;
}

static inline double Score(double termFreq, double k, double b, double meanLength, double classLength) {
    if (termFreq == 0) {
        return 0.0;
    }
    return termFreq * (k + 1.0) / (termFreq + k * (1.0 - b + b * meanLength / classLength));
}

static void InitTruncatedInvClassFreq(ui32 numClasses, double truncateBorder, TArrayRef<double> truncatedInvClassFreq) {
    for (auto classFreq : xrange(numClasses + 1)) {
        truncatedInvClassFreq[classFreq] = CalcTruncatedInvClassFreq(numClasses, classFreq, truncateBorder);
    }
}

TBM25::TBM25(
    const TGuid& calcerId,
    ui32 numClasses,
    double truncateBorder,
    double k,
    double b
)
    : TTextFeatureCalcer(BaseFeatureCount(numClasses), calcerId)
    , NumClasses(numClasses)
    , K(k)
    , B(b)
    , TruncateBorder(truncateBorder)
    , TotalTokens(1)
    , ClassTotalTokens(numClasses)
    , Frequencies(numClasses)
{
    TruncatedInvClassFreq.resize(NumClasses + 1);
    InitTruncatedInvClassFreq(numClasses, truncateBorder, TruncatedInvClassFreq);
}

void TBM25::Compute(const TText& text, TOutputFloatIterator iterator) const {
    TVector<ui32> termFreqInClass(NumClasses);
    TVector<double> scores(NumClasses);
    const double meanClassLength = (double)TotalTokens / NumClasses;
    for (const auto& tokenToCount : text) {
        const ui32 nonZeroCount = ExtractTermFreq(Frequencies, tokenToCount.Token(), termFreqInClass);
        for (ui32 clazz = 0; clazz < NumClasses; ++clazz) {
            scores[clazz] += TruncatedInvClassFreq[nonZeroCount] * Score(termFreqInClass[clazz], K, B, meanClassLength,  ClassTotalTokens[clazz]);
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

    TruncatedInvClassFreq.resize(NumClasses + 1);
    InitTruncatedInvClassFreq(NumClasses, TruncateBorder, TruncatedInvClassFreq);

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
