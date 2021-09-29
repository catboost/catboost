#include "pool_metainfo_options.h"

#include "json_helper.h"

#include <library/cpp/json/json_reader.h>

#include <util/stream/file.h>


namespace NCatboostOptions {
    TPoolMetaInfoOptions::TPoolMetaInfoOptions()
        : Tags("tags", THashMap<TString, NCB::TTagDescription>())
    {}

    void TPoolMetaInfoOptions::Load(const NJson::TJsonValue& options) {
        CheckedLoad(options, &Tags);
    }

    void TPoolMetaInfoOptions::Save(NJson::TJsonValue* options) const {
        SaveFields(options, Tags);
    }

    bool TPoolMetaInfoOptions::operator==(const TPoolMetaInfoOptions& rhs) const {
        return std::tie(Tags) == std::tie(rhs.Tags);
    }

    bool TPoolMetaInfoOptions::operator!=(const TPoolMetaInfoOptions& rhs) const {
        return !(rhs == *this);
    }

    TPoolMetaInfoOptions LoadPoolMetaInfoOptions(const NCB::TPathWithScheme& path) {
        TPoolMetaInfoOptions poolMetaInfoOptions;
        if (path.Inited()) {
            CB_ENSURE(
                path.Scheme.empty() || path.Scheme == "file",
                "Pool metainfo doesn't support path with scheme yet.");
            TIFStream in(path.Path);
            NJson::TJsonValue json = NJson::ReadJsonTree(&in);
            poolMetaInfoOptions.Load(json);
        }
        return poolMetaInfoOptions;
    }

    void LoadPoolMetaInfoOptions(const NCB::TPathWithScheme& path, NJson::TJsonValue* catBoostJsonOptions) {
        if (path.Inited()) {
            CB_ENSURE(
                path.Scheme.empty() || path.Scheme == "file",
                "Pool metainfo doesn't support path with scheme yet.");
            TIFStream in(path.Path);
            (*catBoostJsonOptions)["pool_metainfo_options"] = NJson::ReadJsonTree(&in);
        }
    }
}
