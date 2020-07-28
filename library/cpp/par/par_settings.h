#pragma once

#include "par_log.h"
#include <util/generic/singleton.h>
#include <util/system/env.h>

namespace NPar {
    struct TParNetworkSettings {
        TParNetworkSettings() {
            if (GetEnv("USE_NEH") == "1") {
                DEBUG_LOG << "USE_NEH environment variable detected" << Endl;
                RequesterType = ERequesterType::NEH;
            }
        }

        enum class ERequesterType {
            AutoDetect,
            Netliba,
            NEH
        };

        ERequesterType RequesterType = ERequesterType::AutoDetect;
        static TParNetworkSettings& GetRef() {
            return *Singleton<TParNetworkSettings>();
        }
    };
}
