#include "perftest_module.h"

class TBaseCatboostModule : public TBasePerftestModule {
public:
    TBaseCatboostModule() = default;

    int GetComparisonPriority(EPerftestModuleDataLayout layout) const override {
        if (layout == EPerftestModuleDataLayout::ObjectsFirst) {
            return Priority - 1;
        }
        return Priority;
    }

    bool SupportsLayout(EPerftestModuleDataLayout ) const final {
        return true;
    }

    double Do(EPerftestModuleDataLayout layout, TConstArrayRef<TConstArrayRef<float>> features) final {
        if (layout == EPerftestModuleDataLayout::ObjectsFirst) {
            ResultsHolder.resize(features.size());
            Timer.Reset();
            ModelEvaluator->CalcFlat(features, ResultsHolder);
            return Timer.Passed();
        } else {
            ResultsHolder.resize(features[0].size());
            Timer.Reset();
            ModelEvaluator->CalcFlatTransposed(features, ResultsHolder);
            return Timer.Passed();
        }
    }
    TString GetName(TMaybe<EPerftestModuleDataLayout> layout) const final {
        if (!layout.Defined()) {
            return BaseName;
        } else if (*layout == EPerftestModuleDataLayout::ObjectsFirst) {
            return BaseName +  " objects order";
        } else {
            return BaseName + " features order";
        }
    }
protected:
    NCB::NModelEvaluation::TModelEvaluatorPtr ModelEvaluator;
    int Priority = 0;
    TString BaseName;
    TVector<double> ResultsHolder;
};

class TCPUCatboostModule : public TBaseCatboostModule {
public:
    TCPUCatboostModule(const TFullModel& model) {
        Priority = 10;
        ModelEvaluator = NCB::NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::CPU, model);
        BaseName = "catboost cpu";
    }
};

TPerftestModuleFactory::TRegistrator<TCPUCatboostModule> CPUCatboostModuleRegistar("CPUCatboost");

class TCPUCatboostAsymmetryModule : public TBaseCatboostModule {
public:
    TCPUCatboostAsymmetryModule(const TFullModel& model) {
        CB_ENSURE(model.IsOblivious(), "model is already asymmetrical");
        TFullModel asymmetricalModel = model;
        asymmetricalModel.ModelTrees.GetMutable()->ConvertObliviousToAsymmetric();
        ModelEvaluator = NCB::NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::CPU, asymmetricalModel);
        BaseName = "catboost cpu asymmetrical";
    }
};

TPerftestModuleFactory::TRegistrator<TCPUCatboostAsymmetryModule> CPUCatboostAsymmetryModuleRegistar("CPUCatboostAsymmetry");

class TGPUCatboostModule : public TBaseCatboostModule {
public:
    TGPUCatboostModule(const TFullModel& model) {
        ModelEvaluator = NCB::NModelEvaluation::CreateEvaluator(EFormulaEvaluatorType::GPU, model);
        BaseName = "catboost gpu";
    }
};

TPerftestModuleFactory::TRegistrator<TGPUCatboostModule> GPUCatboostModuleRegistar("GPUCatboostModule");
