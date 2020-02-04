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

        bool IsInitialized() const {
            return LabelConverter.IsInitialized();
        }

        bool IsMultiClass() const {
            return LabelConverter.IsMultiClass();
        }

        int GetApproxDimension() const {
            return LabelConverter.GetApproxDimension();
        }
        TString Serialize() const;

    private:
        const TLabelConverter& LabelConverter;
        const NCatboostOptions::TDataProcessingOptions& Options;
    };
}
