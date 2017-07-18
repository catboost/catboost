#pragma once

#include "model_calcer_wrapper.h"
#include <array>
#include <vector>
#include <functional>

class ModelCalcerWrapper {
public:
    ModelCalcerWrapper() {}

    ModelCalcerWrapper(const std::string& filename) {
        CalcerHolder = CalcerHolderType(LoadModelCalcerFromFile(filename.c_str()), ModelCalcerDelete);
    }

    template <int N_CLASSES>
    std::array<float, N_CLASSES> predict_scores_single_example(const std::vector<float>& example) const {
        std::array<float, N_CLASSES> result;
        PredictMultiFloatValue(CalcerHolder.get(), &example[0], &result[0], N_CLASSES);
    }
    float prdict_score_for_class_single_example(const unsigned int class_, const std::vector<float>& example) const {
        return PredictFloatValue(CalcerHolder.get(), &example[0], class_);
    }
    void init_from_file(const std::string&) {
        CalcerHolder = CalcerHolderType(LoadModelCalcerFromJsonFile(filename.c_str()), ModelCalcerDelete);
    }

private:
    using CalcerHolderType = std::unique_ptr<ModelCalcerHandle, std::function<void(ModelCalcerHandle*)>>;
    CalcerHolderType CalcerHolder;
};
