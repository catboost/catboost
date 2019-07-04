#include "eval_processing.h"

#include <catboost/libs/helpers/exception.h>

NCB::NModelEvaluation::TEvalResultProcessor::TEvalResultProcessor(TArrayRef<double> result,
                                                                  NCB::NModelEvaluation::EPredictionType predictionType,
                                                                  ui32 approxDimension, ui32 blockSize,
                                                                  TMaybe<double> binclassProbabilityBorder)
    : Result(result), PredictionType(predictionType), ApproxDimension(approxDimension), BlockSize(blockSize) {
    if (approxDimension > 1 && predictionType == EPredictionType::Class) {
        IntermediateBlockResults.resize(blockSize * approxDimension);
    }
    if (binclassProbabilityBorder.Defined() && predictionType == EPredictionType::Class &&
        approxDimension == 1) {
        double probabilityBorder = *binclassProbabilityBorder;
        CB_ENSURE(probabilityBorder > 0 && probabilityBorder < 1, "probability border should be in (0;1)");
        BinclassRawValueBorder = -log((1 / probabilityBorder) - 1);
    }
}
