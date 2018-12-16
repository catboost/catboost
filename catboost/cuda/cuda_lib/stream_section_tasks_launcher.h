#pragma once

#include "single_device.h"
#include "inter_device_stream_section.h"
#include "cuda_manager.h"
#include "devices_list.h"
#include <catboost/cuda/cuda_lib/tasks_impl/stream_section_task.h>

namespace NCudaLib {
    class TStreamSectionTaskLauncher {
    public:
        void Add(ui32 deviceId) {
            Add(GetCudaManager().GetDevice(deviceId));
        }

        void Group(ui32 leftDevice, ui32 rightDevice) {
            auto& manager = GetCudaManager();
            Group(manager.GetDevice(leftDevice), manager.GetDevice(rightDevice));
        }

        template <class TTaskProvider>
        void LaunchTask(TDevicesList&& devices,
                        TTaskProvider&& provider,
                        ui32 stream = 0) {
            auto& manager = GetCudaManager();
            auto section = Build();
            for (auto& dev : devices) {
                auto device = manager.GetDevice(dev);
                CB_ENSURE(section.contains(device));
                const auto& config = section[device];
                LaunchSingleTask(device, config, provider(dev), manager.StreamAt(stream, dev));
            }
        }

    protected:
        friend class TDataCopier;

        template <class TKernel>
        void LaunchSingleTask(TCudaSingleDevice* device,
                              const TStreamSectionConfig& sectionConfig,
                              TKernel&& kernel,
                              ui32 stream) {
            using TTask = TStreamSectionKernelTask<TKernel>;
            auto task = MakeHolder<TTask>(std::move(kernel), sectionConfig, stream);
            device->AddTask(std::move(task));
        }

        template <class TTaskProvider>
        void LaunchTaskByDevicePtr(TTaskProvider&& provider,
                                   ui32 stream = 0) {
            auto section = Build();
            auto& manager = GetCudaManager();

            for (auto& entry : section) {
                auto device = entry.first;
                const auto& config = entry.second;
                LaunchSingleTask(device, config, provider(device), manager.StreamAt(stream, device));
            }
        }

        void Add(TCudaSingleDevice* device) {
            ui32 key = GetKey(device);
            Y_UNUSED(key);
        }

        void Group(TCudaSingleDevice* left, TCudaSingleDevice* right) {
            auto leftKey = GetKey(left);
            auto rightKey = GetKey(right);

            if (left->GetHostId() != right->GetHostId()) {
                HasRemote[left] = true;
                HasRemote[right] = true;
            } else {
                MergeKeys(leftKey, rightKey);
            }
        }

        TMap<TCudaSingleDevice*, TStreamSectionConfig> Build() {
            auto& streamSectionProvider = GetStreamSectionProvider();
            TMap<TCudaSingleDevice*, TStreamSectionConfig> result;

            for (const auto& group : Groups) {
                TStreamSectionConfig section;
                section.StreamSectionSize = group.second.size();
                section.StreamSectionUid = streamSectionProvider.NextUid();

                for (const auto& device : group.second) {
                    Y_ASSERT(!result.contains(device));
                    result[device] = section;
                }
            }
            for (auto& entry : result) {
                if (HasRemote[entry.first]) {
                    entry.second.LocalOnly = false;
                }
            }
            return result;
        };

    private:
        using TKey = ui32;

        TKey GetKey(TCudaSingleDevice* device) {
            if (!Keys.contains(device)) {
                const auto key = NewKey();
                Keys[device] = key;
                Groups[key].clear();
                Groups[key].push_back(device);
            }
            return Keys[device];
        }

        void MergeKeys(TKey firstKey, TKey secondKey) {
            if (firstKey == secondKey) {
                return;
            }
            if (firstKey > secondKey) {
                using std::swap;
                swap(firstKey, secondKey);
            }
            for (auto& device : Groups[secondKey]) {
                Keys[device] = firstKey;
                Groups[firstKey].push_back(device);
            }
            Groups.erase(secondKey);
        }

        ui32 NewKey() {
            return Cursor++;
        }

    private:
        TMap<TCudaSingleDevice*, TKey> Keys;
        TMap<TCudaSingleDevice*, bool> HasRemote;
        TMap<TKey, TVector<TCudaSingleDevice*>> Groups;
        ui32 Cursor = 0;
    };
}
