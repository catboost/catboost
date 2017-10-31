#include "train.h"

class TGPUModelTrainer : public IModelTrainer {
    void TrainModel(
        const NJson::TJsonValue& params,
        const TMaybe<TCustomObjectiveDescriptor>& objectiveDescriptor,
        const TMaybe<TCustomMetricDescriptor>& evalMetricDescriptor,
        TPool& learnPool,
        bool allowClearPool,
        const TPool& testPool,
        const TString& outputModelPath,
        TFullModel* model,
        yvector<yvector<double>>* testApprox) const override
    {
        Y_UNUSED(objectiveDescriptor);
        Y_UNUSED(evalMetricDescriptor);
        Y_UNUSED(allowClearPool);
        NCatboostCuda::TrainModel(params, learnPool, testPool, outputModelPath, model);
        testApprox->resize(model->ApproxDimension);
    }
};

TTrainerFactory::TRegistrator<TGPUModelTrainer> GPURegistrator(ECalcerType::GPU);
