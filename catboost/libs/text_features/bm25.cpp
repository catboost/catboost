#include "bm25.h"
#include <util/generic/ymath.h>

using namespace NCB;


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

static inline void ExtractTermFreq(TConstArrayRef<TDenseHash<ui32, ui32>> freq,
                                   ui32 term,
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

TVector<double> TOnlineBM25::CalcFeatures(const TText& text, double k, double b) const {
    TVector<double> scores(NumClasses);
    TVector<ui32> termFreqInClass(NumClasses);

    for (const auto& [term, textFreq] : text) {
        Y_UNUSED(textFreq);
        ExtractTermFreq(Freq, term, termFreqInClass);
        double inverseClassFreq = CalcTruncatedInvClassFreq(termFreqInClass, TruncateBorder);
        double meanClassLength = TotalTokens * 1.0 / NumClasses;

        for (ui32 clazz = 0; clazz < NumClasses; ++clazz) {
            scores[clazz] += inverseClassFreq * Score(termFreqInClass[clazz], k, b, meanClassLength,  ClassTotalTokens[clazz]);
        }
    }
    return scores;
}


TVector<double> TOnlineBM25::CalcFeaturesAndAddText(ui32 classId, const TText& text, double k, double b) {
    auto result = CalcFeatures(text, k, b);
    AddText(classId, text);
    return result;
}


void TOnlineBM25::AddText(ui32 classId, const TText& text) {
    auto& classCounts = Freq[classId];

    for (const auto& [term, termCount] : text) {
        classCounts[term] += termCount;
        ClassTotalTokens[classId] += termCount;
        TotalTokens += termCount;
    }
}
