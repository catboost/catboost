#pragma once

#include "option.h"

#include <catboost/libs/column_description/feature_tag.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>

#include <util/generic/hash.h>


namespace NJson {
    class TJsonValue;
}

namespace NCatboostOptions {
    struct TPoolMetaInfoOptions {
        explicit TPoolMetaInfoOptions();

        void Load(const NJson::TJsonValue& options);
        void Save(NJson::TJsonValue* options) const;

        bool operator==(const TPoolMetaInfoOptions& rhs) const;
        bool operator!=(const TPoolMetaInfoOptions& rhs) const;

        TOption<THashMap<TString, NCB::TTagDescription>> Tags;
    };

    // `path` can be not inited
    TPoolMetaInfoOptions LoadPoolMetaInfoOptions(const NCB::TPathWithScheme& path);

    // `path` can be not inited
    void LoadPoolMetaInfoOptions(const NCB::TPathWithScheme& path, NJson::TJsonValue* catBoostJsonOptions);
}
