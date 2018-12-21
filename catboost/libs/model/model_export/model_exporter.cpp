#include "model_exporter.h"
#include "cpp_exporter.h"
#include "python_exporter.h"

#include <util/string/builder.h>

namespace NCatboost {
    ICatboostModelExporter* CreateCatboostModelExporter(const TString& modelFile, const EModelType format, const TString& userParametersJson, bool addFileFormatExtension) {
        switch (format) {
            case EModelType::Cpp:
                return new TCatboostModelToCppConverter(modelFile, addFileFormatExtension, userParametersJson);
            case EModelType::Python:
                return new TCatboostModelToPythonConverter(modelFile, addFileFormatExtension, userParametersJson);
            default:
                TStringBuilder err;
                err << "CreateCatboostModelExporter doesn't support " << format << ".";
                CB_ENSURE(false, err);
        }
    }
}
