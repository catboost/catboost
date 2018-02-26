#pragma once

#include "cuda_base.h"
#include "device_id.h"
#include "slice.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "kernel.h"
#include <util/ysaveload.h>

namespace NKernelHost {
    class TFixedSizesObjectsMeta {
    private:
        ui64 ObjectCount;
        ui64 SingleObjectSize;

    public:
        TFixedSizesObjectsMeta()
            : ObjectCount(0)
            , SingleObjectSize(0)
        {
        }

        TFixedSizesObjectsMeta(const TFixedSizesObjectsMeta&) = default;

        TFixedSizesObjectsMeta(TFixedSizesObjectsMeta&&) = default;

        TFixedSizesObjectsMeta& operator=(TFixedSizesObjectsMeta&&) = default;
        TFixedSizesObjectsMeta& operator=(const TFixedSizesObjectsMeta&) = default;

        TFixedSizesObjectsMeta(ui64 count, ui64 objectSize)
            : ObjectCount(count)
            , SingleObjectSize(objectSize)
        {
        }

        ui64 GetObjectCount() const {
            return ObjectCount;
        }

        ui64 SliceSize(const TSlice& slice) const {
            return SingleObjectSize * (slice.Right - slice.Left);
        }

        ui64 SliceOffset(const TSlice& slice) const {
            return SingleObjectSize * slice.Left;
        }

        ui64 GetTotalDataSize() const {
            return SliceSize(TSlice(0, ObjectCount));
        }

        Y_SAVELOAD_DEFINE(ObjectCount, SingleObjectSize);
    };

    template <typename T,
              typename TObjectsMeta = TFixedSizesObjectsMeta,
              EPtrType Type = EPtrType::CudaDevice>
    class TDeviceBuffer {
    private:
        NCudaLib::THandleBasedMemoryPointer<T, Type> Data;
        TObjectsMeta Meta;
        ui64 ColumnCount;
        TDeviceId DeviceId;

    public:
        TDeviceBuffer(NCudaLib::THandleBasedMemoryPointer<T, Type> ptr,
                      TObjectsMeta&& meta,
                      ui64 columnCount,
                      TDeviceId deviceId)
            : Data(ptr)
            , Meta(meta)
            , ColumnCount(columnCount)
            , DeviceId(deviceId)
        {
        }

        TDeviceBuffer()
            : Data(NCudaLib::THandleBasedMemoryPointer<T, Type>())
            , Meta(TObjectsMeta())
            , ColumnCount(0)
        {
        }

        TDeviceBuffer& operator=(const TDeviceBuffer& other) = default;
        TDeviceBuffer& operator=(TDeviceBuffer&& other) = default;
        TDeviceBuffer(TDeviceBuffer&& other) = default;
        TDeviceBuffer(const TDeviceBuffer& other) = default;

        T* Get() const {
            return Data.Get();
        }

        T operator[](ui64 idx) const {
            T value;
            TCudaStream& stream = NCudaLib::GetDefaultStream();
            NCudaLib::TMemoryCopier<Type, EPtrType::Host>::CopyMemoryAsync(Get() + idx, &value, 1, stream);
            stream.Synchronize();
            return value;
        }

        ui64 ColumnSize() const {
            return Meta.GetTotalDataSize();
        }

        operator TDeviceBuffer<const T, TObjectsMeta, Type>() {
            TObjectsMeta metaCopy = Meta;
            return TDeviceBuffer<const T, TObjectsMeta, Type>(static_cast<NCudaLib::THandleBasedMemoryPointer<const T, Type>>(Data),
                                                              std::move(metaCopy),
                                                              ColumnCount);
        };

        ui64 Size() const {
            return ColumnSize() * ColumnCount;
        }

        ui64 ObjectCount() const {
            return Meta.GetObjectCount();
        }

        ui64 ObjectSize(ui64 objectId) const {
            return SliceMemorySize(TSlice(objectId, objectId + 1));
        }
        ui64 SliceMemorySize(const TSlice& slice) const {
            return Meta.SliceSize(slice);
        }

        ui64 SliceMemoryOffset(const TSlice& slice) const {
            return Meta.SliceOffset(slice);
        }

        ui64 GetColumnCount() const {
            return ColumnCount;
        }

        void AsyncWrite(const TVector<T>& data,
                        const TCudaStream& stream) const {
            CB_ENSURE(data.size() <= Size());
            NCudaLib::TMemoryCopier<EPtrType::Host, Type>::CopyMemoryAsync(~data, Get(), data.size(), stream);
        }

        void Write(const TVector<T>& data, const TCudaStream& stream) const {
            if (Type == EPtrType::CudaHost) {
                T* Dst = this->Get();
                for (ui32 i = 0; i < data.size(); ++i) {
                    Dst[i] = data[i];
                }
            } else {
                AsyncWrite(data, stream);
                stream.Synchronize();
            }
        }

        TVector<std::remove_const_t<T>> Read(const TCudaStream& stream) const {
            const ui64 size = Size();
            TVector<std::remove_const_t<T>> result;
            result.resize(size);
            NCudaLib::TMemoryCopier<Type, EPtrType::Host>::CopyMemoryAsync(Get(), ~result, size, stream);
            stream.Synchronize();
            return result;
        }

        static TDeviceBuffer Nullptr() {
            return TDeviceBuffer();
        }

        const TDeviceId& GetDeviceId() const {
            return DeviceId;
        }

        Y_SAVELOAD_DEFINE(Data, Meta, ColumnCount, DeviceId);
    };

    template <class T, EPtrType PtrType = EPtrType::CudaDevice>
    using TCudaBufferPtr = TDeviceBuffer<T, TFixedSizesObjectsMeta, PtrType>;

    template <class T, class TObjectsMeta = TFixedSizesObjectsMeta>
    using TCudaHostBufferPtr = TDeviceBuffer<T, TObjectsMeta, EPtrType::CudaHost>;
}
