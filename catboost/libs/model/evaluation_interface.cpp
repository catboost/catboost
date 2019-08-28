#include "evaluation_interface.h"


namespace NCB::NModelEvaluation {
    TModelEvaluatorPtr CreateEvaluator(EFormulaEvaluatorType formualEvaluatorType, const TFullModel& model) {
        return TEvaluationBackendFactory::Construct(formualEvaluatorType, model);
    }
}
