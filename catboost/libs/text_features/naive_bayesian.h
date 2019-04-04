#pragma once

#include "text_dataset.h"
#include <util/system/types.h>
#include <util/generic/fwd.h>

namespace NCB {

    /*
    * p(x | c_k) = \prod p_i,k^{x_i} * (1 - p_{i, k})(1 - x_i), where x i binary vector
    */
    class TMultinomialOnlineNaiveBayes {
    public:

        explicit TMultinomialOnlineNaiveBayes(ui32 numClasses)
            : NumClasses(numClasses)
            , ClassDocs(numClasses)
            , ClassTotalTokens(numClasses)
            , Counts(numClasses) {
        }

        TVector<double> CalcFeatures(const TText& text, double classPrior = 0.5, double tokenPrior = 0.5) const;

        TVector<double> CalcFeaturesAndAddText(ui32 classId, const TText& text, double classPrior = 0.5, double tokenPrior = 0.5);

        void AddText(ui32 classId, const TText& text);

    private:

        double LogProb(const TDenseHash<ui32, ui32>& freqTable,
                       double classSamples,
                       double classTokensCount,
                       const TText& text,
                       double classPrior,
                       double tokenPrior) const;

    private:
        ui32 NumClasses;
        TVector<ui32> ClassDocs;
        TVector<ui64> ClassTotalTokens;
        TVector<TDenseHash<ui32, ui32>> Counts;
        TDenseHashSet<ui32> KnownTokens;
    };
}
