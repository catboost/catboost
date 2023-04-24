#include "evaluation_interface.h"


namespace NCB::NModelEvaluation {
    TModelEvaluatorPtr CreateEvaluator(EFormulaEvaluatorType formulaEvaluatorType, const TFullModel& model) {
        return TEvaluationBackendFactory::Construct(formulaEvaluatorType, model);
    }
}
