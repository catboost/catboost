#pragma once

#include "enums.h"

#include <util/generic/fwd.h>
#include <util/generic/ptr.h> // TODO(kirillovs): remove

// TODO(kirillovs): move to NCB::NModelEvaluation
class TCtrValueTable;
struct TObliviousTrees;
class TFullModel;

namespace NCB { // split due to CUDA-compiler inability to support nested namespace definitions
    namespace NModelEvaluation {
        using TCalcerIndexType = ui32;

        class IModelEvaluator;
        class IQuantizedData;
        class ILeafIndexCalcer;

        using TModelEvaluatorPtr = TAtomicSharedPtr<IModelEvaluator>;
        using TConstModelEvaluatorPtr = TAtomicSharedPtr<const IModelEvaluator>;
    }
}

namespace NCatBoostFbs {
    //features.fbs
    struct TFloatFeature;
    struct TCatFeature;
    struct TOneHotFeature;
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
    struct TObliviousTrees;
    struct TModelCore;
}

namespace flatbuffers {
    template <typename T>
    struct Offset;

    class FlatBufferBuilder;
}
