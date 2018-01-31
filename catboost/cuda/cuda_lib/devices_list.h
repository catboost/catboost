#pragma once

#include <util/system/types.h>
#include <bitset>
#include <util/system/yassert.h>
#include <util/generic/utility.h>

namespace NCudaLib {
    class TDevicesList {
    private:
#ifdef USE_MPI
        static const ui64 MaxDeviceCount = 256;
#else
        static const ui64 MaxDeviceCount = 64;
#endif
        using TDevicesSet = std::bitset<MaxDeviceCount>;

        TDevicesSet DevicesSet;
        ui64 Begin = 0; //inclusive
        ui64 End = 0;   //exclusive

        friend class TDevicesListBuilder;

    public:
        TDevicesList(TDevicesList&& other) = default;
        TDevicesList(const TDevicesList& other) = default;
        TDevicesList& operator=(TDevicesList&& other) = default;
        TDevicesList& operator=(const TDevicesList& other) = default;

        inline bool HasDevice(ui64 devId) const {
            return DevicesSet.test(devId);
        }

    public:
        TDevicesList() {
        }

        class TDeviceListIterator {
        private:
            const TDevicesList* Owner;
            ui64 Dev;

        public:
            inline TDeviceListIterator()
                : Owner(nullptr)
                , Dev(0)
            {
            }

            inline TDeviceListIterator(const TDevicesList* owner,
                                       const ui64 dev)
                : Owner(owner)
                , Dev(dev)
            {
            }

            inline bool operator!=(const TDeviceListIterator& other) {
                Y_ASSERT(Owner == other.Owner);
                return Dev != other.Dev;
            }

            inline const TDeviceListIterator& operator++() {
                Y_ASSERT(Dev < MaxDeviceCount);
                do {
                    ++Dev;
                } while (!Owner->DevicesSet.test(Dev) && Dev < Owner->End);
                return *this;
            }

            inline const ui64& operator*() const {
                return Dev;
            }
        };

        inline TDeviceListIterator begin() const {
            return {this, Begin};
        }

        inline TDeviceListIterator end() const {
            return {this, End};
        }
    };

    class TDevicesListBuilder {
    public:
        TDevicesListBuilder() {
            Result.Begin = TDevicesList::MaxDeviceCount;
            Result.End = 0;
        }

        TDevicesListBuilder& AddDevice(ui32 device) {
            Result.DevicesSet.set(static_cast<ui64>(device), true);
            Result.Begin = Min<ui64>(Result.Begin, device);
            Result.End = Max<ui64>(Result.End, device + 1);
            return *this;
        }

        static TDevicesList Range(ui32 firstDev, ui32 lastDev) {
            TDevicesList result;
            result.DevicesSet.set();
            result.Begin = firstDev;
            result.End = lastDev;
            return result;
        }

        TDevicesList Build() {
            if (Result.End < Result.Begin) {
                Y_ASSERT(Result.DevicesSet.count() == 0);
                Result.Begin = Result.End = TDevicesList::MaxDeviceCount;
            }
            return Result;
        }

        static TDevicesList SingleDevice(ui64 devId) {
            return TDevicesListBuilder().AddDevice(devId).Build();
        }

    private:
        TDevicesList Result;
    };
}
