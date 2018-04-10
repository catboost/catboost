#pragma once

#include <catboost/libs/model/model.h>

#include <util/string/builder.h>
#include <util/generic/string.h>

namespace NCatboostModelExportHelpers {
    template <class T>
    TString OutputArrayInitializer(const TVector<T>& array) {
        TStringBuilder str;
        for (const auto& value : array) {
            str << value << (&value != &array.back() ? "," : "");
        }
        return str;
    }

    int GetBinaryFeatureCount(const TFullModel& model);

    TString OutputBorderCounts(const TFullModel& model);

    TString OutputBorders(const TFullModel& model);

    TString OutputLeafValues(const TFullModel& model);
}
