#pragma once

#include "online_ctr.h"

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>


struct TFeatureCombination;


namespace NCB {
    struct TCompressedModelCtr {
        const TFeatureCombination* Projection;
        TVector<const TModelCtr*> ModelCtrs;
    };

    TVector<TCompressedModelCtr> CompressModelCtrs(const TConstArrayRef<TModelCtr> neededCtrs);

    struct TCtrTablesMergeStatus {
        int LastModelOffset = 0;
        int SizeInCurrentModel = 0;

    public:
        inline int GetResultIndex(int indexInCurrentModel) {
            if (indexInCurrentModel >= SizeInCurrentModel) {
                SizeInCurrentModel = indexInCurrentModel + 1;
            }
            return LastModelOffset + indexInCurrentModel;
        }

        inline void FinishModel() {
            LastModelOffset += SizeInCurrentModel;
            SizeInCurrentModel = 0;
        }
    };

} // namespace NCB
