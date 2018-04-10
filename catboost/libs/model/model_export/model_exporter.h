#pragma once

#include <catboost/libs/model/model.h>
#include <util/generic/ptr.h>

namespace NCatboost {
    class ICatboostModelExporter: public TThrRefBase {
    public:
        virtual ~ICatboostModelExporter() = default;

        virtual void Write(const TFullModel& model) = 0;
    };

    ICatboostModelExporter* CreateCatboostModelExporter(const TString& modelFile, const EModelType format, const TString& userParametersJSON, bool addFileFormatExtension);
}
