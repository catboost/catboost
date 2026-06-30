#include "evaluation_interface.h"

#include <util/system/yassert.h>

namespace NCB::NModelEvaluation {
    TModelEvaluatorPtr CreateEvaluator(EFormulaEvaluatorType formulaEvaluatorType, const TFullModel& model) {
        CB_ENSURE(
            TEvaluationBackendFactory::Has(formulaEvaluatorType),
            "No implementation available for formulaEvaluatorType=\"" << formulaEvaluatorType << "\""
        );
        auto* evaluatorRawPtr = TEvaluationBackendFactory::Construct(formulaEvaluatorType, model);
        Y_ASSERT(evaluatorRawPtr);
        return TModelEvaluatorPtr(evaluatorRawPtr);
    }
}
