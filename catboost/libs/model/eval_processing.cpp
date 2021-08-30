#include "eval_processing.h"

#include <catboost/libs/helpers/exception.h>

NCB::NModelEvaluation::TEvalResultProcessor::TEvalResultProcessor(
    size_t docCount,
    TArrayRef<double> results,
    NCB::NModelEvaluation::EPredictionType predictionType,
    TScaleAndBias scaleAndBias,
    ui32 approxDimension,
    ui32 blockSize,
    TMaybe<double> binclassProbabilityBorder
)
    : Results(results)
    , PredictionType(predictionType)
    , ScaleAndBias(scaleAndBias)
    , ApproxDimension(approxDimension)
    , BlockSize(blockSize)
{
    const auto resultApproxDimension = predictionType == EPredictionType::Class ? 1 : ApproxDimension;
    CB_ENSURE(
        Results.size() == docCount * resultApproxDimension,
        "`results` size is insufficient: " << LabeledOutput(Results.size(), resultApproxDimension, docCount * resultApproxDimension)
    );
    if (approxDimension > 1 && predictionType == EPredictionType::Class) {
        IntermediateBlockResults.resize(blockSize * approxDimension);
    }
    if (binclassProbabilityBorder.Defined() && predictionType == EPredictionType::Class &&
        approxDimension == 1) {
        double probabilityBorder = *binclassProbabilityBorder;
        CB_ENSURE(probabilityBorder > 0 && probabilityBorder < 1, "probability border should be in (0;1)");
        BinclassRawValueBorder = -log((1 / probabilityBorder) - 1);
    }
    if (ApproxDimension > 1 && (predictionType == EPredictionType::Class || predictionType == EPredictionType::Probability) ) {
        CB_ENSURE_IDENTITY(ScaleAndBias, "normalizing a multiclass model");
    }
}
