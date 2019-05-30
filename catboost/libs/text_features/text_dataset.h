#pragma once

#include "dictionary.h"

#include <catboost/libs/data_types/text.h>

#include <util/generic/ptr.h>


namespace NCB {
    class TTextDataSet : public TThrRefBase {
    public:
        TTextDataSet(TTextColumn texts, TDictionaryPtr dictionary)
        : Text(std::move(texts))
        , Dictionary(std::move(dictionary)) {}

        ui64 SamplesCount() const {
            return (*Text).size();
        }

        const TText& GetText(ui64 idx) const {
            const ui64 samplesCount = SamplesCount();
            CB_ENSURE(idx < samplesCount, "Error: text line " << idx << " is out of bound (" << samplesCount << ")");
            return Text[idx];
        }

        TConstArrayRef<TText> GetTexts() const {
            return *Text;
        }

        const IDictionary& GetDictionary() const {
            return *Dictionary;
        }
    private:
        const TTextColumn Text;
        const TDictionaryPtr Dictionary;
    };

    struct TTextClassificationTarget : public TThrRefBase {
        TTextClassificationTarget(TVector<ui32> classes, ui32 numClasses)
        : Classes(std::move(classes))
        , NumClasses(numClasses)
        {}

        TVector<ui32> Classes;
        ui32 NumClasses;
    };

    using TTextDataSetPtr = TIntrusivePtr<TTextDataSet>;
    using TTextClassificationTargetPtr = TIntrusivePtr<TTextClassificationTarget>;
}
