#include "../model.h"

namespace NCB::NModelEvaluation {

    TModelEvaluatorPtr CreateGpuEvaluator(const TFullModel& model) {
        Y_UNUSED(model);
        ythrow yexception() << "Cuda evaluator unavailable";
    }
    bool CudaEvaluationPossible(const TFullModel& model) {
        Y_UNUSED(model);
        return false;
    }
}
