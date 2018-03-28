#pragma once

#include "model.h"

class TCatboostModelToCppConverter {
private:
    TOFStream Out;
public:
    explicit TCatboostModelToCppConverter(const TString& modelFile)
        : Out(modelFile){};

    void Write(const TFullModel& model) {
        WriteHeader();
        WriteModel(model);
        WriteApplicator();
    }
private:
    void WriteApplicator();
    void WriteModel(const TFullModel& model);
    void WriteHeader();
};
