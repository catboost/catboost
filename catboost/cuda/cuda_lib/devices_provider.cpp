#include "devices_provider.h"

NCudaLib::TDeviceRequestConfig NCudaLib::CreateDeviceRequestConfig(const NCatboostOptions::TCatBoostOptions& options) {
    NCudaLib::TDeviceRequestConfig config;
    const auto& systemOptions = options.SystemOptions.Get();
    config.DeviceConfig = systemOptions.Devices;
    config.PinnedMemorySize = ParseMemorySizeDescription(systemOptions.PinnedMemorySize.Get());
    config.GpuMemoryPartByWorker = systemOptions.GpuRamPart;
    return config;
}
