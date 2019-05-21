#include "naive_bayesian.h"
#include "helpers.h"
#include "text_dataset.h"
#include <library/containers/dense_hash/dense_hash.h>
#include <util/generic/array_ref.h>
#include <util/generic/ymath.h>

using namespace NCB;

double TMultinomialOnlineNaiveBayes::LogProb(
    const TDenseHash<ui32, ui32>& freqTable,
    double classSamples,
    double classTokensCount,
    const TText& text,
    double classPrior,
    double tokenPrior) const {

    double value = log(classSamples + classPrior);

    classTokensCount += tokenPrior * (KnownTokens.Size() + 1);
    double textLen = 0;

    for (const auto& [token, count] : text) {
        textLen += count;

        auto tokenCountPtr = freqTable.find(token);
        double num = tokenPrior;

        if (tokenCountPtr != freqTable.end()) {
            num += tokenCountPtr->second;
        } else {
            //unseen word, adjust prior
            classTokensCount += tokenPrior;
        }
        value += log(num);
    }

    //denum
    value -= textLen * log(classTokensCount);

    return value;
}

void TMultinomialOnlineNaiveBayes::AddText(ui32 classId, const TText& text)  {
    auto& classCounts = Counts[classId];

    for (const auto& [term, termCount] : text) {
        KnownTokens.Insert(term);
        classCounts[term] += termCount;
        ClassTotalTokens[classId] += termCount;
    }
    ++ClassDocs[classId];
}

TVector<double> TMultinomialOnlineNaiveBayes::CalcFeatures(const TText& text, double classPrior, double tokenPrior) const  {
    TVector<double> logProbs(NumClasses);
    for (ui32 clazz = 0; clazz < NumClasses; ++clazz) {
        logProbs[clazz] = LogProb(Counts[clazz],
                                  ClassDocs[clazz],
                                  ClassTotalTokens[clazz],
                                  text,
                                  classPrior,
                                  tokenPrior
                                  );
    }
    Softmax(logProbs);
    return logProbs;
}

TVector<double> TMultinomialOnlineNaiveBayes::CalcFeaturesAndAddText(ui32 classId, const TText& text, double classPrior, double tokenPrior) {
    auto result = CalcFeatures(text, classPrior, tokenPrior);
    AddText(classId, text);
    return result;
}
