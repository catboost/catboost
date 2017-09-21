#pragma once

#include "cuda_base.h"
#include "gpu_memory_pool.h"
#include "slice.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "kernel.h"

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

        SAVELOAD(ObjectCount, SingleObjectSize);
    };

    template <typename T,
              typename TObjectsMeta = TFixedSizesObjectsMeta,
              EPtrType Type = EPtrType::CudaDevice>
    class TDeviceBuffer {
    private:
        NCudaLib::THandleBasedMemoryPointer<T, Type> Data;
        TObjectsMeta Meta;
        ui64 ColumnCount;

    public:
        TDeviceBuffer(NCudaLib::THandleBasedMemoryPointer<T, Type> ptr,
                      TObjectsMeta&& meta,
                      ui64 columnCount)
            : Data(ptr)
            , Meta(meta)
            , ColumnCount(columnCount)
        {
        }

        TDeviceBuffer()
            : Data(NCudaLib::THandleBasedMemoryPointer<T, Type>())
            , Meta(TObjectsMeta())
            , ColumnCount(0)
        {
        }

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


        void AsyncWrite(const yvector<T>& data,
                        const TCudaStream& stream) const {
            CB_ENSURE(data.size() <= Size());
            NCudaLib::TMemoryCopier<EPtrType::Host, Type>::CopyMemoryAsync(~data, Get(), data.size(), stream);
        }

        void Write(const yvector<T>& data, const TCudaStream& stream) const {
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

        yvector<typename std::remove_const<T>::type> Read(const TCudaStream& stream) const {
            const ui64 size = Size();
            yvector<typename std::remove_const<T>::type> result;
            result.resize(size);
            NCudaLib::TMemoryCopier<Type, EPtrType::Host>::CopyMemoryAsync(Get(), ~result, size, stream);
            stream.Synchronize();
            return result;
        }

        static TDeviceBuffer Nullptr() {
            return TDeviceBuffer();
        }

        SAVELOAD(Data, Meta, ColumnCount);
    };

    template <class T, class TObjectsMeta = TFixedSizesObjectsMeta>
    using TCudaBufferPtr = TDeviceBuffer<T, TObjectsMeta, EPtrType::CudaDevice>;

    template <class T, class TObjectsMeta = TFixedSizesObjectsMeta>
    using TCudaHostBufferPtr = TDeviceBuffer<T, TObjectsMeta, EPtrType::CudaHost>;
}
