#pragma once

#include "options_binding.h"

#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <library/getopt/small/last_getopt_opts.h>
#include <util/system/types.h>

class TApplicationOptions {
public:
    ui32 GetNumThreads() const {
        return NumThreads;
    }

    ui64 GetSeed() const {
        return Seed;
    }

    bool IsProfile() const {
        return Profile;
    }

    const NCudaLib::TCudaApplicationConfig& GetCudaApplicationConfig() const {
        return ApplicationConfig;
    }

    template <class TConfig>
    friend class TOptionsBinder;

private:
    ui32 NumThreads = 16;
    ui64 Seed = 0;
    bool Profile = false;
    NCudaLib::TCudaApplicationConfig ApplicationConfig;
};

template <>
class TOptionsBinder<TApplicationOptions> {
public:
    static void Bind(TApplicationOptions& applicationOptions, NLastGetopt::TOpts& options) {
        options
            .AddLongOption('T', "thread-count")
            .RequiredArgument("int")
            .Help("Enable threads")
            .StoreResult(&applicationOptions.NumThreads);

        options
            .AddLongOption("gpu-ram-part")
            .RequiredArgument("double")
            .Help("Part of gpu ram to use")
            .DefaultValue("0.95")
            .StoreResult(&applicationOptions.ApplicationConfig.GpuMemoryPartByWorker);

        options
            .AddLongOption("pinned-memory-size")
            .RequiredArgument("int")
            .Help("Part of gpu ram to use")
            .DefaultValue("1073741824")
            .StoreResult(&applicationOptions.ApplicationConfig.PinnedMemorySize);

        if (options.HasLongOption("random-seed")) {
            options
                .GetLongOption("random-seed")
                .StoreResult(&applicationOptions.Seed);
        } else {
            options
                .AddLongOption('r', "random-seed")
                .RequiredArgument("INT")
                .Help("Sets random generators seed.")
                .DefaultValue(ToString<long>(GetTime()))
                .StoreResult(&applicationOptions.Seed);
        }

        options
            .AddLongOption("detailed-profile")
            .RequiredArgument("FLAG")
            .Help("Enables profiling")
            .SetFlag(&applicationOptions.Profile)
            .NoArgument();
    }
};
