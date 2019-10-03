#pragma once

#include <catboost/private/libs/options/data_processing_options.h>
#include <catboost/private/libs/labels/label_converter.h>

namespace NCB {

    class TClassificationTargetHelper {
    public:
        TClassificationTargetHelper(
            const TLabelConverter& labelConverter,
            const NCatboostOptions::TDataProcessingOptions& options)
            : LabelConverter(labelConverter)
            , Options(options)
        {}

        bool IsMultiClass() const {
            return LabelConverter.IsInitialized();
        }

        int GetNumClasses() const {
            return LabelConverter.IsInitialized() ? LabelConverter.GetApproxDimension() : 2;
        }
        TString Serialize() const;

    private:
        const TLabelConverter& LabelConverter;
        const NCatboostOptions::TDataProcessingOptions& Options;
    };
}
