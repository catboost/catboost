#pragma once

enum class ETargetFunction {
    RMSE,
    CrossEntropy,
    Logloss,

};

class TTargetOptions {
public:
    ETargetFunction GetTargetType() const {
        return TargetType;
    }

    bool IsUseBorderForClassification() const {
        return UseBorderForClassification;
    }

    float GetBinClassBorder() const {
        return BinClassBorder;
    }

private:
    template <class TConfig>
    friend class TOptionsBinder;

    ETargetFunction TargetType = ETargetFunction::RMSE;
    bool UseBorderForClassification = false;
    float BinClassBorder = 0.5;
};
