#pragma once

#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <library/getopt/small/last_getopt_opts.h>
#include <util/system/types.h>

class TApplicationOptions {
public:
    int GetNumThreads() const {
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

    NCudaLib::TCudaApplicationConfig& GetCudaApplicationConfig() {
        return ApplicationConfig;
    }

    ui32 GetDeviceCount() const {
        return ApplicationConfig.GetDeviceCount();
    }

    template <class TConfig>
    friend class TOptionsBinder;

private:
    ui32 NumThreads = 16;
    ui64 Seed = 0;
    bool Profile = false;
    NCudaLib::TCudaApplicationConfig ApplicationConfig;
};
