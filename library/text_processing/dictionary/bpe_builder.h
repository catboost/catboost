#pragma once

#include "bpe_dictionary.h"
#include "bpe_helpers.h"
#include "frequency_based_dictionary.h"

#include <library/cpp/containers/heap_dict/heap_dict.h>

#include <util/generic/array_ref.h>
#include <util/generic/fwd.h>

namespace NTextProcessing::NDictionary {
    using TPairStats = THeapDict<TPair, TPairStat>;

    class TBpeDictionaryBuilder: public TMoveOnly {
    public:
        TBpeDictionaryBuilder(ui32 numUnits, bool skipUnknown, TIntrusivePtr<TDictionary> alphabet)
            : NumUnits(numUnits)
            , SkipUnknown(skipUnknown)
            , Alphabet(alphabet)
        {
            Y_ENSURE(Alphabet->GetDictionaryOptionsRef().GramOrder == 1,
                "GramOrder should be equal to 1 for Bpe dictionary");
        }

        void Add(TConstArrayRef<TString> tokens, ui64 weight = 1);
        void Add(TConstArrayRef<TStringBuf> tokens, ui64 weight = 1);

        TIntrusivePtr<TBpeDictionary> FinishBuilding();

    private:
        void CalcMostFrequentUnits();

        ui32 NumUnits;
        bool SkipUnknown;
        TIntrusivePtr<TDictionary> Alphabet;

        TVector<TEraseList<TTokenId>> TokenIdsLists;
        TPairStats PairStats;
        TVector<ui64> Counts;
        TVector<TBpeDictionary::TBpeUnit> ResultingBpeUnits;
        bool IsBuildingFinish = false;
    };
}
