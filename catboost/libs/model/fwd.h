#pragma once

#include "enums.h"

#include <util/generic/fwd.h>
#include <util/generic/ptr.h> // TODO(kirillovs): remove

// TODO(kirillovs): move to NCB::NModelEvaluation
class TCtrValueTable;
struct TModelTrees;
class TFullModel;

namespace NCB { // split due to CUDA-compiler inability to support nested namespace definitions
    namespace NModelEvaluation {
        using TCalcerIndexType = ui32;

        class IModelEvaluator;
        class IQuantizedData;
        class ILeafIndexCalcer;

        using TModelEvaluatorPtr = TAtomicSharedPtr<IModelEvaluator>;
        using TConstModelEvaluatorPtr = TAtomicSharedPtr<const IModelEvaluator>;

        class IQuantizedData : public TThrRefBase {
        public:
            virtual size_t GetObjectsCount() const = 0;
        };
    }
}

namespace NCatBoostFbs {
    //features.fbs
    struct TFloatFeature;
    struct TCatFeature;
    struct TTextFeature;
    struct TEmbeddingFeature;
    struct TOneHotFeature;
    struct TEstimatedFeature;
    struct TFloatSplit;
    struct TOneHotSplit;
    struct TFeatureCombination;

    //ctr_data.fbs
    struct TModelCtrBase;
    struct TModelCtr;
    struct TCtrFeature;
    struct TCtrValueTable;

    //model.fbs
    struct TKeyValue;
    struct TNonSymmetricTreeStepNode;
    struct TRepackedBin;
    struct TModelTrees;
    struct TModelCore;
}

namespace flatbuffers {
    template <typename T>
    struct Offset;
}
