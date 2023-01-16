#pragma once

#include "dictionary.h"

#include <catboost/private/libs/data_types/text.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <util/generic/ptr.h>


namespace NCB {
    using TTextColumn = TMaybeOwningConstArrayHolder<TText>;

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

        const TDictionaryProxy& GetDictionary() const {
            return *Dictionary;
        }
    private:
        const TTextColumn Text;
        const TDictionaryPtr Dictionary;
    };

    using TTextDataSetPtr = TIntrusivePtr<TTextDataSet>;
}
