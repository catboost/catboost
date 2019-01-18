#pragma once

#include "cuda_base.h"
#include "device_id.h"
#include "slice.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "kernel.h"
#include "column_aligment_helper.h"
#include <util/ysaveload.h>

namespace NKernelHost {
    using TDeviceId = NCudaLib::TDeviceId;

    class TObjectsMeta {
    private:
        ui64 ObjectCount;
        ui64 SingleObjectSize;

    public:
        TObjectsMeta()
            : ObjectCount(0)
            , SingleObjectSize(0)
        {
        }

        TObjectsMeta(const TObjectsMeta&) = default;

        TObjectsMeta(TObjectsMeta&&) = default;

        TObjectsMeta& operator=(TObjectsMeta&&) = default;
        TObjectsMeta& operator=(const TObjectsMeta&) = default;

        TObjectsMeta(ui64 count, ui64 objectSize)
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

        ui64 GetSingleObjectSize() const {
            return SingleObjectSize;
        }

        Y_SAVELOAD_DEFINE(ObjectCount, SingleObjectSize);
    };

    template <typename T,
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

        T* GetForObject(ui64 objectId) const {
            return Data.Get() + Meta.GetSingleObjectSize() * objectId;
        }

        T operator[](ui64 idx) const {
            T value;
            TCudaStream& stream = NCudaLib::GetDefaultStream();
            NCudaLib::TMemoryCopier<Type, EPtrType::Host>::CopyMemoryAsync(Get() + idx, &value, 1, stream);
            stream.Synchronize();
            return value;
        }

        T* GetColumn(ui32 columnId) const {
            CB_ENSURE(columnId < ColumnCount, "Column id " << columnId << " should be less than " << ColumnCount);
            return Data.Get() + NCudaLib::NAligment::ColumnShift(Size(), columnId);
        }

        /* this is size of columns with aligments */
        ui64 AlignedColumnSize() const {
            return NCudaLib::NAligment::AlignedColumnSize(Size());
        }

        operator TDeviceBuffer<const T, Type>() {
            TObjectsMeta metaCopy = Meta;
            return TDeviceBuffer<const T, Type>(static_cast<NCudaLib::THandleBasedMemoryPointer<const T, Type>>(Data),
                                                std::move(metaCopy),
                                                ColumnCount,
                                                DeviceId);
        };

        ui64 Size() const {
            return Meta.GetTotalDataSize();
        }

        ui64 ObjectCount() const {
            return Meta.GetObjectCount();
        }

        ui64 ObjectSize() const {
            return Meta.GetSingleObjectSize();
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
            NCudaLib::TMemoryCopier<EPtrType::Host, Type>::CopyMemoryAsync(data.data(), Get(), data.size(), stream);
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
            NCudaLib::TMemoryCopier<Type, EPtrType::Host>::CopyMemoryAsync(Get(), result.data(), size, stream);
            stream.Synchronize();
            return result;
        }

        static TDeviceBuffer Nullptr() {
            return TDeviceBuffer();
        }

        const TDeviceId& GetDeviceId() const {
            return DeviceId;
        }

        NCudaLib::THandleBasedMemoryPointer<T, Type> GetData() const {
            return Data;
        };

        Y_SAVELOAD_DEFINE(Data, Meta, ColumnCount, DeviceId);
    };

    template <class T>
    using TCudaBufferPtr = TDeviceBuffer<T, EPtrType::CudaDevice>;

    template <class T>
    using TCudaHostBufferPtr = TDeviceBuffer<T, EPtrType::CudaHost>;
}
