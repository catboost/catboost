#pragma once

#include "fwd.h"
#include "cuda_manager.h"
#include "cuda_kernel_buffer.h"
#include "slice.h"

#include <catboost/libs/helpers/exception.h>

#include <util/stream/labeled.h>

namespace NCudaLib {
    template <class TImpl>
    class TMappingBase {
    private:
        ui64 ObjectSize;

    public:
        using TMeta = NKernelHost::TObjectsMeta;

        explicit TMappingBase(ui64 objectSize = 1)
            : ObjectSize(objectSize)
        {
        }

        inline ui64 SingleObjectSize() const {
            return ObjectSize;
        }

        //indexing without GPU-layout
        ui64 MemorySize(const TSlice& slice) const {
            return slice.Size() * ObjectSize;
        }

        ui64 MemorySize() const {
            return MemorySize(static_cast<const TImpl*>(this)->GetObjectsSlice());
        }

        ui64 MemoryOffset(const TSlice& slice) const {
            return slice.Left * SingleObjectSize();
        }

        TMeta At(ui64 dev) const {
            const ui64 devSize = static_cast<const TImpl*>(this)->CountAt(dev);
            return NKernelHost::TObjectsMeta(devSize, ObjectSize);
        }

        //memory offsets and usages for devices
        ui64 MemoryUsageAt(ui64 dev) const {
            return static_cast<const TImpl*>(this)->CountAt(dev) * ObjectSize;
        }

        ui64 DeviceMemoryOffset(ui64 dev, const TSlice& slice) const {
            const TSlice deviceSlice = static_cast<const TImpl*>(this)->DeviceSlice(dev);
            CB_ENSURE(TSlice::Intersection(slice, deviceSlice) == slice);
            const ui64 devSize = slice.Size() != 0u ? (slice.Left - deviceSlice.Left) : 0;
            if (devSize != 0u) {
                CB_ENSURE(slice.Left >= deviceSlice.Left, TStringBuilder() << slice << " " << deviceSlice);
            }
            return devSize * SingleObjectSize();
        }

        TDevicesList NonEmptyDevices() const {
            TDevicesListBuilder builder;
            auto devCount = GetCudaManager().GetDeviceCount();
            for (ui64 dev = 0; dev < devCount; ++dev) {
                if (MemoryUsageAt(dev)) {
                    builder.AddDevice(dev);
                }
            }
            return builder.Build();
        }

        template <class TFunc>
        inline void Apply(TFunc&& trans) const {
            static_cast<const TImpl*>(this)->Transform([&](const TSlice& slice) -> ui64 {
                trans(slice);
                return 0;
            });
        }
    };

    class TSingleMapping: public TMappingBase<TSingleMapping> {
    protected:
        ui64 Count = 0;
        ui32 DeviceId = 0;

    public:
        using TMappingBase::TMeta;

        explicit TSingleMapping(ui32 devId = 0, ui64 count = 0, ui64 size = 1)
            : TMappingBase(size)
            , Count(count)
            , DeviceId(devId)
        {
        }

        explicit TSingleMapping(TVector<TSlice>&& slices, ui64 singleObjectSize = 1)
            : TMappingBase(singleObjectSize)
        {
            CB_ENSURE(slices.size() == NCudaLib::GetCudaManager().GetDeviceCount());

            bool initDone = false;
            for (ui32 i = 0; i < slices.size(); ++i) {
                if (slices[i].Size()) {
                    CB_ENSURE(!initDone);
                    initDone = true;
                    DeviceId = i;
                    Count = slices[i].Size();
                }
            }
        }

        TSlice GetObjectsSlice() const {
            return {0, Count};
        }

        ui64 CountAt(ui64 dev) const {
            if (dev != DeviceId) {
                return 0;
            }
            return Count;
        }

        ui32 GetDeviceId() const {
            return DeviceId;
        }

        TSingleMapping ToLocalSlice(const TSlice& slice) const {
            CB_ENSURE(GetObjectsSlice().Contains(slice));
            return TSingleMapping(DeviceId, slice.Size(), SingleObjectSize());
        }

        TSlice DeviceSlice(ui64 dev) const {
            if (dev != DeviceId) {
                return {0, 0};
            }
            return {0, Count};
        }

        TSingleMapping RepeatOnAllDevices(ui64 objectCount, ui64 objectSize = 1) const {
            return TSingleMapping(DeviceId, objectCount, objectSize);
        }

        template <class TFunc>
        inline TSingleMapping Transform(TFunc&& trans, ui64 objectSize = 1) const {
            ui64 devSize = trans(GetObjectsSlice());
            return TSingleMapping(DeviceId, devSize, objectSize);
        }

        inline TSingleMapping ChangeDevice(ui32 newDeviceId) const {
            return TSingleMapping(newDeviceId, Count, SingleObjectSize());
        }
    };

    class TMirrorMapping: public TMappingBase<TMirrorMapping> {
    protected:
        ui64 Count;

    public:
        using TMappingBase::TMeta;

        explicit TMirrorMapping(ui64 count = 0,
                                ui64 objectSize = 1)
            : TMappingBase(objectSize)
            , Count(count)
        {
        }

        ui64 CountAt(ui32 dev) const {
            Y_UNUSED(dev);
            return Count;
        }

        TSlice DeviceSlice(ui32 dev) const {
            Y_UNUSED(dev);
            return {0, Count};
        }

        TSlice GetObjectsSlice() const {
            return {0, Count};
        }

        TMirrorMapping ToLocalSlice(const TSlice& slice) const {
            Y_ASSERT(GetObjectsSlice().Contains(slice));
            CB_ENSURE(GetObjectsSlice().Contains(slice), TStringBuilder() << "Slice " << slice << " should be subset of " << GetObjectsSlice());

            return TMirrorMapping(slice.Size(), SingleObjectSize());
        }

        TMirrorMapping RepeatOnAllDevices(ui64 objectCount, ui64 objectSize = 1) const {
            return TMirrorMapping(objectCount, objectSize);
        }

        template <class TFunc>
        inline TMirrorMapping Transform(TFunc&& trans, ui64 objectSize = 1) const {
            ui64 devSize = trans(GetObjectsSlice());
            return TMirrorMapping(devSize, objectSize);
        }
    };

    class TStripeMapping: public TMappingBase<TStripeMapping> {
    protected:
        TVector<TSlice> Slices;

    public:
        using TMappingBase::TMeta;

        explicit TStripeMapping(TVector<TSlice>&& slices, ui64 singleObjectSize = 1)
            : TMappingBase(singleObjectSize)
            , Slices(std::move(slices))
        {
            for (ui32 i = 1; i < Slices.size(); ++i) {
                CB_ENSURE(Slices[i].Left == Slices[i - 1].Right, LabeledOutput(i, Slices[i].Left, Slices[i - 1].Right));
            }
        }

        ui64 CountAt(ui32 dev) const {
            return Slices[dev].Size();
        }

        TStripeMapping()
            : TMappingBase(0)
        {
            Slices.resize(GetCudaManager().GetDeviceCount(), TSlice(0, 0));
        }

        TSlice DeviceSlice(ui32 dev) const {
            return dev < Slices.size() ? Slices[dev] : TSlice(0, 0);
        }

        TSlice GetObjectsSlice() const {
            ui64 min = Slices[0].Left;
            ui64 max = Slices[0].Right;
            for (ui64 dev = 0; dev < Slices.size(); ++dev) {
                min = std::min(min, Slices[dev].Left);
                max = std::max(max, Slices[dev].Right);
            }

            return {min, max};
        }

        TStripeMapping ToLocalSlice(const TSlice& slice) const {
            CB_ENSURE(GetObjectsSlice().Contains(slice));
            TVector<TSlice> slices(Slices.begin(), Slices.end());

            for (ui64 i = 0; i < slices.size(); ++i) {
                slices[i] = TSlice::Intersection(slices[i], slice);
                const ui64 left = i > 0 ? slices[i - 1].Right : 0;
                const ui64 right = left + slices[i].Size();
                slices[i] = TSlice(left, right);
            }
            return TStripeMapping(std::move(slices), SingleObjectSize());
        }

        static TStripeMapping SplitBetweenDevices(ui64 objectCount, ui64 objectSize = 1) {
            const ui64 devCount = GetCudaManager().GetDeviceCount();
            TVector<TSlice> slices(devCount);
            const ui64 objectPerDevice = objectCount / devCount;

            ui64 total = 0;

            for (ui32 i = 0; i < devCount; ++i) {
                const ui64 devSize = (i + 1 != devCount ? objectPerDevice : (objectCount - total));
                slices[i] = TSlice(total, total + devSize);
                total += devSize;
            }
            return TStripeMapping(std::move(slices), objectSize);
        }

        static TStripeMapping RepeatOnAllDevices(ui64 objectCount,
                                                 ui64 objectSize = 1) {
            const ui64 devCount = GetCudaManager().GetDeviceCount();
            TVector<TSlice> slices(devCount);
            for (ui64 i = 0; i < slices.size(); ++i) {
                slices[i].Left = i * objectCount;
                slices[i].Right = (i + 1) * objectCount;
            }
            return TStripeMapping(std::move(slices), objectSize);
        }
        template<typename TSizesArray>
        static TStripeMapping CreateFromSizes(const TSizesArray& sizes,
                                              ui64 objectSize = 1) {
            const ui64 devCount = GetCudaManager().GetDeviceCount();
            TVector<TSlice> slices(devCount);
            for (ui64 i = 0; i < slices.size(); ++i) {
                slices[i].Left = i > 0 ? slices[i - 1].Right : 0;
                slices[i].Right = slices[i].Left + sizes[i];
            }
            return TStripeMapping(std::move(slices), objectSize);
        }

        template <class TFunc>
        inline TStripeMapping Transform(TFunc&& trans,
                                        ui64 objectSize = 1) const {
            TVector<TSlice> nextSlices;

            ui64 offset = 0;
            for (ui32 i = 0; i < Slices.size(); ++i) {
                ui64 devSize = trans(Slices[i]);
                nextSlices.push_back(TSlice(offset, offset + devSize));
                offset += devSize;
            }
            return TStripeMapping(std::move(nextSlices), objectSize);
        }
    };

    template <class TMapping>
    class TMappingBuilder;

    template <>
    class TMappingBuilder<TStripeMapping> {
    public:
        TMappingBuilder() {
            DeviceSizes.resize(NCudaLib::GetCudaManager().GetDeviceCount(), 0);
        }

        TMappingBuilder& SetSizeAt(ui32 device, ui64 size) {
            DeviceSizes[device] = size;
            return *this;
        }

        TMappingBuilder& UpdateMaxSizeAt(ui32 device, ui64 size) {
            DeviceSizes[device] = Max<ui64>(size, DeviceSizes[device]);
            return *this;
        }

        TStripeMapping Build(ui64 objectSize = 1) {
            TVector<TSlice> slices;
            ui64 offset = 0;
            for (ui32 dev = 0; dev < DeviceSizes.size(); ++dev) {
                slices.push_back(TSlice(offset, offset + DeviceSizes[dev]));
                offset += DeviceSizes[dev];
            }
            return TStripeMapping(std::move(slices), objectSize);
        }

    private:
        TVector<ui64> DeviceSizes;
    };

    template <>
    class TMappingBuilder<TMirrorMapping> {
    public:
        TMappingBuilder() {
            DeviceSizes.resize(NCudaLib::GetCudaManager().GetDeviceCount(), 0);
        }

        TMappingBuilder& SetSizeAt(ui32 device, ui64 size) {
            DeviceSizes[device] = size;
            return *this;
        }

        TMirrorMapping Build(ui64 objectSize = 1) {
            for (ui32 dev = 0; dev < DeviceSizes.size(); ++dev) {
                CB_ENSURE(DeviceSizes[dev] == DeviceSizes[0]);
            }
            return TMirrorMapping(DeviceSizes[0], objectSize);
        }

    private:
        TVector<ui64> DeviceSizes;
    };

    template <>
    class TMappingBuilder<TSingleMapping> {
    public:
        TMappingBuilder() {
            DeviceSizes.resize(NCudaLib::GetCudaManager().GetDeviceCount(), 0);
        }

        TMappingBuilder& SetSizeAt(ui32 device, ui64 size) {
            DeviceSizes[device] = size;
            return *this;
        }

        TSingleMapping Build() {
            ui32 nonZeroDevices = 0;
            ui32 nonZeroId = 0;
            for (ui32 i = 0; i < DeviceSizes.size(); ++i) {
                if (DeviceSizes[i] != 0) {
                    nonZeroDevices++;
                    nonZeroId = i;
                }
            }
            CB_ENSURE(nonZeroDevices <= 1);
            return TSingleMapping(nonZeroId, DeviceSizes[nonZeroId]);
        }

    private:
        TVector<ui64> DeviceSizes;
    };

}

template <class TMapping, class T>
TMapping CreateMapping(const NCudaLib::TDistributedObject<T>& sizes) {
    NCudaLib::TMappingBuilder<TMapping> builder;
    for (ui32 dev = 0; dev < sizes.DeviceCount(); ++dev) {
        builder.SetSizeAt(dev, sizes.At(dev));
    }
    return builder.Build();
}
