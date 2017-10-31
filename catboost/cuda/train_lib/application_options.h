#pragma once

#include <catboost/libs/logging/logging_level.h>
#include <catboost/cuda/cuda_lib/devices_provider.h>
#include <util/system/types.h>

namespace NCatboostCuda
{
    class TApplicationOptions
    {
    public:
        int GetNumThreads() const
        {
            return NumThreads;
        }


        bool IsProfile() const
        {
            return Profile;
        }

        const NCudaLib::TCudaApplicationConfig& GetCudaApplicationConfig() const
        {
            return ApplicationConfig;
        }

        NCudaLib::TCudaApplicationConfig& GetCudaApplicationConfig()
        {
            return ApplicationConfig;
        }

        ui32 GetDeviceCount() const
        {
            return ApplicationConfig.GetDeviceCount();
        }

        ELoggingLevel GetLoggingLevel() const
        {
            return LoggingLevel;
        }

        template<class TConfig>
        friend
        class TOptionsBinder;

        template<class TConfig>
        friend
        class TOptionsJsonConverter;

    private:
        ui32 NumThreads = 16;
        bool Profile = false;
        ELoggingLevel LoggingLevel = ELoggingLevel::Silent;
        NCudaLib::TCudaApplicationConfig ApplicationConfig;
    };
}
