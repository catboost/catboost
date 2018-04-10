#include "model_exporter.h"
#include "cpp_exporter.h"
#include "python_exporter.h"

#include <util/string/builder.h>

namespace NCatboost {
    ICatboostModelExporter* CreateCatboostModelExporter(const TString& modelFile, const EModelType format, const TString& userParametersJSON, bool addFileFormatExtension) {
        switch (format) {
            case EModelType::CPP:
                return new TCatboostModelToCppConverter(modelFile, addFileFormatExtension, userParametersJSON);
            case EModelType::Python:
                return new TCatboostModelToPythonConverter(modelFile, addFileFormatExtension, userParametersJSON);
            default:
                TStringBuilder err;
                err << "CreateCatboostModelExporter doesn't support " << format << ".";
                CB_ENSURE(false, err);
        }
    }
}
