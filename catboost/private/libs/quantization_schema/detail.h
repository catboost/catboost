#pragma once

#include <catboost/idl/pool/proto/quantization_schema.pb.h>
#include <catboost/idl/pool/proto/metainfo.pb.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/private/libs/options/enums.h>

#include <util/generic/yexception.h>
#include <util/system/types.h>


namespace NCB {
    namespace NQuantizationSchemaDetail {
        inline ENanMode NanModeFromProto(const NIdl::ENanMode proto) {
            switch (proto) {
                case NIdl::NM_MIN:
                    return ENanMode::Min;
                case NIdl::NM_MAX:
                    return ENanMode::Max;
                case NIdl::NM_FORBIDDEN:
                    return ENanMode::Forbidden;
                case NIdl::NM_UNKNOWN:
                    // Native `ENanMode` doesn't have `Unknown` member
                    break;
            }

            ythrow TCatBoostException() << "got unexpected enum " << static_cast<int>(proto);
        }

        inline NIdl::ENanMode NanModeToProto(const ENanMode native) {
            switch (native) {
                case ENanMode::Min:
                    return NIdl::NM_MIN;
                case ENanMode::Max:
                    return NIdl::NM_MAX;
                case ENanMode::Forbidden:
                    return NIdl::NM_FORBIDDEN;
            }

            ythrow TCatBoostException() << "got unexpected enum " << static_cast<int>(native);
        }

        bool IsFakeIndex(ui32 index, const NIdl::TPoolMetainfo& metaInfo);
    }
}
