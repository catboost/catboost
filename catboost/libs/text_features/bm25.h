#pragma once

#include "text_dataset.h"
#include <util/system/types.h>
#include <util/generic/fwd.h>

namespace NCB {

    /*
     *  BM25 between class and text documents
     *  Convert to classical: class = document (BoW)
     *  query = text
     *  BM25(class, text)
    */
    class TOnlineBM25 {
    public:

        explicit TOnlineBM25(ui32 numClasses, double truncateBorder)
            : NumClasses(numClasses)
            , TotalTokens(1)
            , ClassTotalTokens(numClasses)
            , Freq(numClasses)
            , TruncateBorder(truncateBorder) {
        }

        TVector<double> CalcFeatures(const TText& text, double k = 1.5, double b = 0.75) const;

        TVector<double> CalcFeaturesAndAddText(ui32 classId, const TText& text, double k = 1.5, double b = 0.75);

        void AddText(ui32 classId, const TText& text);


    private:
        ui32 NumClasses;
        ui64 TotalTokens;
        TVector<ui64> ClassTotalTokens;
        TVector<TDenseHash<ui32, ui32>> Freq;
        double TruncateBorder;
    };
}
