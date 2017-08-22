#pragma once

#include "model_calcer_wrapper.h"
#include <array>
#include <vector>
#include <functional>
#include <memory>

class ModelCalcerWrapper {
public:
    ModelCalcerWrapper() {
    }

    ModelCalcerWrapper(const std::string& filename) {
        CalcerHolder = CalcerHolderType(ModelCalcerCreate(), ModelCalcerDelete);
        LoadFullModelFromFile(CalcerHolder.get(), filename.c_str());
    }

    double Calc(const std::vector<float>& features) {
        double result;
        const float* ptr = &features[0];
        CalcModelPredition(CalcerHolder.get(), 1, &ptr, 1, &result, 1);
        return result;
    }

    bool init_from_file(const std::string& filename) {
        return LoadFullModelFromFile(CalcerHolder.get(), filename.c_str());
    }

private:
    using CalcerHolderType = std::unique_ptr<ModelCalcerHandle, std::function<void(ModelCalcerHandle*)>>;
    CalcerHolderType CalcerHolder;
};
