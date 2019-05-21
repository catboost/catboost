#pragma once

#include "bpe_dictionary.h"
#include "frequency_based_dictionary.h"

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>

namespace NTextProcessing::NDictionary {

    class TBpeDictionaryBuilder {
    public:
        TBpeDictionaryBuilder(ui32 numUnits, bool skipUnknown, THolder<TDictionary> alphabet)
            : NumUnits(numUnits)
            , SkipUnknown(skipUnknown)
            , Alphabet(std::move(alphabet))
        {
            Y_ENSURE(Alphabet->GetDictionaryOptionsRef().GramOrder == 1,
                "GramOrder should be equal to 1 for Bpe dictionary");
        }

        void Add(TConstArrayRef<TString> tokens, ui64 weight = 1);
        void Add(TConstArrayRef<TStringBuf> tokens, ui64 weight = 1);

        THolder<TBpeDictionary> FinishBuilding();

    private:
        void CalcMostFrequentUnits();

        ui32 NumUnits;
        bool SkipUnknown;
        THolder<TDictionary> Alphabet;

        TVector<TVector<ui32>> Lines;
        TVector<ui64> Counts;
        TVector<TBpeDictionary::TBpeUnit> ResultingBpeUnits;
        bool IsBuildingFinish = false;
    };
}
