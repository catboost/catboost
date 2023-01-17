#pragma once

#include "fwd.h"
#include "cuda_base.h"
#include "column_aligment_helper.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "single_device.h"
#include "cuda_kernel_buffer.h"
#include "cuda_manager.h"
#include "mapping.h"
#include "cpu_reducers.h"
#include "cache.h"
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_writer.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_reader.h>

#include <util/generic/array_ref.h>
#include <util/generic/vector.h>

namespace NCudaLib {
    template <class T,
              class TMapping,
              EPtrType Type>
    class TCudaBuffer: public TMoveOnly {
    private:
        using TRawPtr = typename TMemoryProviderImplTrait<Type>::TRawFreeMemory;
        using TBuffer = TCudaSingleDevice::TSingleBuffer<T, Type>;
        using TDeviceBuffer = typename NKernelHost::TDeviceBuffer<T, Type>;
        using TConstDeviceBuffer = typename NKernelHost::TDeviceBuffer<const T, Type>;

        TMapping Mapping;
        TVector<TBuffer> Buffers;

        ui64 ColumnCount = 1;
        mutable bool CreatedFromScratchFlag = false;
        mutable bool IsSliceView = false;
        bool ReadOnly = false;

        void EnsureSize(ui32 devId, ui64 size, bool freeMemory) {
            size = NAligment::GetMemorySize(size, ColumnCount);
            if (Buffers.at(devId).NotEmpty() && (Buffers.at(devId).Size() == size || (Buffers.at(devId).Size() > size && !freeMemory))) {
                return;
            }
            Buffers.at(devId) = GetCudaManager().CreateSingleBuffer<T, Type>(devId, size);
        }

        TBuffer& GetBuffer(ui32 devId) const {
            Y_ASSERT(Buffers.at(devId).NotEmpty());
            CB_ENSURE(Buffers.at(devId).NotEmpty(), TStringBuilder() << "Error: no buffer found on device #" << devId);
            //TODO: rid of this
            return const_cast<TBuffer&>(Buffers.at(devId));
        }

        bool CanReset() const {
            if (ReadOnly || IsSliceView) {
                return false;
            }
            if (CreatedFromScratchFlag) {
                return true;
            }
            bool allNullptr = true;
            for (auto& buffer : Buffers) {
                if (buffer.NotEmpty()) {
                    allNullptr = false;
                    break;
                }
            }
            if (allNullptr) {
                CreatedFromScratchFlag = true;
                return true;
            }
            return false;
        }

        template <class TBuffer>
        friend class TCudaBufferReader;

        template <class TBuffer>
        friend class TCudaBufferWriter;

        template <class TSrc, class TDst>
        friend class TCudaBufferResharding;

        template <EPtrType, EPtrType>
        friend class TMemoryCopyPerformance;

        TBuffer& BufferAt(ui32 devId) {
            return Buffers.at(devId);
        }
        const TBuffer& BufferAt(ui32 devId) const {
            return Buffers.at(devId);
        }

        template <class TC,
                  class TFriendMapping, EPtrType>
        friend class TCudaBuffer;

        template <class TC, EPtrType Type2>
        friend TCudaBuffer<const TC, TStripeMapping, Type2> StripeView(const TCudaBuffer<const TC, TMirrorMapping, Type2>& buffer,
                                                                       const TStripeMapping& stripeMapping,
                                                                       ui32 column);

        template <class TC, EPtrType Type2>
        friend TCudaBuffer<TC, TStripeMapping, Type2> ParallelStripeView(const TCudaBuffer<TC, TStripeMapping, Type2>& buffer,
                                                                         const TSlice& slice,
                                                                         ui32 column);

    public:
        using TValueType = T;

        explicit TCudaBuffer(ui64 columnCount = 1)
            : ColumnCount(columnCount)
        {
            Buffers.resize(GetCudaManager().GetDeviceCount());
        }

        template <class U>
        TCudaBuffer<U, TMapping, Type> ReinterpretCast() {
            TCudaBuffer<U, TMapping, Type> buffer;
            buffer.Mapping = Mapping;
            buffer.ColumnCount = ColumnCount;

            for (ui64 i = 0; i < Buffers.size(); ++i) {
                if (Buffers[i].NotEmpty()) {
                    buffer.Buffers[i] = Buffers[i].template ReinterpretCast<U>();
                }
            }

            buffer.IsSliceView = true;
            return buffer;
        }

        TCudaBuffer<const T, TMapping, Type> SliceView(const TSlice& slice,
                                                       ui64 column = 0) const {
            TCudaBuffer<const T, TMapping, Type> buffer;
            buffer.Mapping = Mapping.ToLocalSlice(slice);

            for (ui64 i = 0; i < Buffers.size(); ++i) {
                if (Buffers[i].NotEmpty()) {
                    const ui64 columnsShift = NAligment::ColumnShift(Mapping.MemoryUsageAt(i), column);
                    const auto devSlice = TSlice::Intersection(slice, Mapping.DeviceSlice(i));

                    buffer.Buffers[i] = Buffers[i].ShiftedConstBuffer(
                        columnsShift + Mapping.DeviceMemoryOffset(i, devSlice));
                }
            }
            buffer.IsSliceView = true;
            return buffer;
        }

        TCudaBuffer<T, TMapping, Type> SliceView(const TSlice& slice,
                                                 ui64 column = 0) {
            TCudaBuffer<T, TMapping, Type> buffer;
            buffer.Mapping = Mapping.ToLocalSlice(slice);

            for (ui64 i = 0; i < Buffers.size(); ++i) {
                if (Buffers[i].NotEmpty()) {
                    const ui64 columnsShift = NAligment::ColumnShift(Mapping.MemoryUsageAt(i), column);
                    const auto devSlice = TSlice::Intersection(slice, Mapping.DeviceSlice(i));
                    buffer.Buffers[i] = Buffers[i].ShiftedBuffer(
                        columnsShift + Mapping.DeviceMemoryOffset(i, devSlice));
                }
            }
            buffer.IsSliceView = true;
            return buffer;
        }

        TCudaBuffer<const T, TMapping, Type> ColumnsView(const TSlice& columns) const {
            CB_ENSURE(columns.Right <= ColumnCount);
            TCudaBuffer<const T, TMapping, Type> buffer;
            auto slice = GetObjectsSlice();
            buffer.Mapping = Mapping.ToLocalSlice(slice);
            buffer.ColumnCount = columns.Size();

            for (ui64 i = 0; i < Buffers.size(); ++i) {
                if (Buffers[i].NotEmpty()) {
                    const ui64 columnsShift = NAligment::ColumnShift(Mapping.MemoryUsageAt(i), columns.Left);
                    const auto devSlice = TSlice::Intersection(slice, Mapping.DeviceSlice(i));
                    buffer.Buffers[i] = Buffers[i].ShiftedConstBuffer(
                        columnsShift + Mapping.DeviceMemoryOffset(i, devSlice));
                }
            }
            buffer.IsSliceView = true;
            return buffer;
        }

        TCudaBuffer<T, TMapping, Type> ColumnsView(const TSlice& columns) {
            CB_ENSURE(columns.Right <= ColumnCount);
            TCudaBuffer<T, TMapping, Type> buffer;
            auto slice = GetObjectsSlice();
            buffer.Mapping = Mapping.ToLocalSlice(slice);
            buffer.ColumnCount = columns.Size();

            for (ui64 i = 0; i < Buffers.size(); ++i) {
                if (Buffers[i].NotEmpty()) {
                    const ui64 columnsShift = NAligment::ColumnShift(Mapping.MemoryUsageAt(i), columns.Left);
                    const auto devSlice = TSlice::Intersection(slice, Mapping.DeviceSlice(i));
                    buffer.Buffers[i] = Buffers[i].ShiftedBuffer(
                        columnsShift + Mapping.DeviceMemoryOffset(i, devSlice));
                }
            }
            buffer.IsSliceView = true;
            return buffer;
        }

        TCudaBuffer<const T, TMapping, Type> ColumnView(ui64 column) const {
            return ColumnsView(TSlice(column, column + 1));
        }

        TCudaBuffer<T, TMapping, Type> ColumnView(ui64 column) {
            return ColumnsView(TSlice(column, column + 1));
        }

        TCudaBuffer<T, TMapping, Type> CopyView() {
            return ColumnsView(TSlice(0, ColumnCount));
        }

        TCudaBuffer<const T, TMapping, Type> ConstCopyView() const {
            return ColumnsView(TSlice(0, ColumnCount));
        }

        //TODO: get rid of this. Refactor everything to const in template type instead of holder-objects
        TCudaBuffer<T, TSingleMapping, Type> DeviceView(ui32 devId) const {
            TCudaBuffer<T, TSingleMapping, Type> buffer;
            const auto devSlice = Mapping.DeviceSlice(devId);
            buffer.Mapping = TSingleMapping(devId, devSlice.Size(), Mapping.SingleObjectSize());

            if (Buffers.at(devId).NotEmpty()) {
                buffer.Buffers.at(devId) = Buffers.at(devId).ShiftedBuffer(Mapping.DeviceMemoryOffset(devId, devSlice));
            }
            buffer.IsSliceView = true;
            buffer.ReadOnly = true;
            buffer.ColumnCount = ColumnCount;
            return buffer;
        }

        TCudaBuffer<const T, TSingleMapping, Type> ConstDeviceView(ui32 devId) const {
            TCudaBuffer<const T, TSingleMapping, Type> buffer;
            const auto devSlice = Mapping.DeviceSlice(devId);
            buffer.Mapping = TSingleMapping(devId, devSlice.Size(), Mapping.SingleObjectSize());
            if (Buffers.at(devId).NotEmpty()) {
                buffer.Buffers.at(devId) = Buffers.at(devId).ShiftedConstBuffer(Mapping.DeviceMemoryOffset(devId, devSlice));
            }
            buffer.IsSliceView = true;
            buffer.ColumnCount = ColumnCount;
            return buffer;
        }

        TCudaBuffer<const T, TMapping, Type> AsConstBuf() const {
            TCudaBuffer<const T, TMapping, Type> buffer;

            for (ui64 i = 0; i < Buffers.size(); ++i) {
                if (Buffers[i].NotEmpty()) {
                    buffer.Buffers[i] = Buffers[i].ShiftedConstBuffer(0);
                }
            }
            buffer.Mapping = Mapping;
            buffer.ColumnCount = ColumnCount;
            buffer.IsSliceView = IsSliceView;
            buffer.ReadOnly = ReadOnly;
            buffer.CreatedFromScratchFlag = CreatedFromScratchFlag;

            return buffer;
        };

        TSlice GetObjectsSlice() const {
            return Mapping.GetObjectsSlice();
        }
        const TMapping& GetMapping() const {
            return Mapping;
        }

        TDevicesList NonEmptyDevices() const {
            return Mapping.NonEmptyDevices();
        }

        TDeviceBuffer At(ui64 devId) {
            Y_ASSERT(devId < Buffers.size());

            return TDeviceBuffer(BufferAt(devId).GetPointer(),
                                 Mapping.At(devId),
                                 ColumnCount,
                                 GetCudaManager().GetDeviceId(devId));
        };

        TConstDeviceBuffer At(ui64 devId) const {
            Y_ASSERT(devId < Buffers.size());
            return TConstDeviceBuffer(BufferAt(devId).GetPointer(),
                                      Mapping.At(devId),
                                      ColumnCount,
                                      GetCudaManager().GetDeviceId(devId));
        };

        ui64 GetColumnCount() const {
            return ColumnCount;
        }

        //may reallocate memory
        static void SetMapping(const TMapping& mapping,
                               TCudaBuffer& buffer,
                               bool freeUnusedMemory = false) {
            for (auto dev : mapping.NonEmptyDevices()) {
                buffer.EnsureSize(dev, mapping.At(dev).GetTotalDataSize(), freeUnusedMemory);
            }
            buffer.Mapping = mapping;
        }

        void Reset(const TMapping& mapping) {
            CB_ENSURE(CanReset());
            static_assert(!std::is_const<T>::value, "Can't reset const buffer");
            ColumnCount = 1;
            TCudaBuffer::SetMapping(mapping, *this, false);
        }

        void Reset(const TMapping& mapping,
                   ui32 columnCount) {
            CB_ENSURE(CanReset(), "Error: buffer is view of some other data. can't reset it");
            ColumnCount = columnCount;
            TCudaBuffer::SetMapping(mapping, *this, false);
        }

        static TCudaBuffer Create(const TMapping& mapping,
                                  ui64 columnCount = 1) {
            TCudaBuffer buffer(columnCount);
            SetMapping(mapping, buffer);
            buffer.CreatedFromScratchFlag = true;
            return buffer;
        }

        static void Swap(TCudaBuffer& lhs, TCudaBuffer& rhs) {
            using std::swap;
            swap(lhs.Buffers, rhs.Buffers);
            swap(lhs.ColumnCount, rhs.ColumnCount);
            swap(lhs.Mapping, rhs.Mapping);
            swap(lhs.CreatedFromScratchFlag, rhs.CreatedFromScratchFlag);
            swap(lhs.IsSliceView, rhs.IsSliceView);
        }

        void Clear() {
            TCudaBuffer empty;
            Swap(*this, empty);
        }

        static constexpr EPtrType PtrType() {
            return Type;
        }

        TCudaBufferWriter<TCudaBuffer> CreateWriter(TConstArrayRef<T> src) {
            return TCudaBufferWriter<TCudaBuffer>(src, *this);
        }

        TCudaBufferReader<TCudaBuffer> CreateReader() const {
            return TCudaBufferReader<TCudaBuffer>(*this);
        }

        void Write(TConstArrayRef<T> src, ui32 stream = 0) {
            CreateWriter(src).SetCustomWritingStream(stream).Write();
        }

        void Read(TVector<std::remove_const_t<T>>& dst, ui32 stream = 0) const {
            CreateReader().SetCustomReadingStream(stream).Read(dst);
        }

        template <class TC, EPtrType PtrType>
        static TCudaBuffer CopyMapping(const TCudaBuffer<TC, TMapping, PtrType>& other, ui32 columnCount = 1) {
            return Create(other.Mapping, columnCount);
        }

        template <class TC, EPtrType PtrType>
        static TCudaBuffer CopyMappingAndColumnCount(const TCudaBuffer<TC, TMapping, PtrType>& other) {
            return Create(other.Mapping, other.ColumnCount);
        }

        template <class TC, EPtrType PtrType>
        static TCudaBuffer CopyMapping(const TCudaBuffer<TC, TMapping, PtrType>* other) {
            return CopyMapping(*other);
        }

        template <class TC, EPtrType SrcType>
        inline void Copy(const TCudaBuffer<TC, TMapping, SrcType>& src, ui32 stream = 0) {
            const auto& mapping = GetMapping();
            const TMapping& srcMapping = src.GetMapping();
            CB_ENSURE(src.ColumnCount == ColumnCount);

            TDataCopier copier(stream);

            for (const auto dev : srcMapping.NonEmptyDevices()) {
                ui64 deviceSize = srcMapping.MemorySize(srcMapping.DeviceSlice(dev));
                ui64 thisDevSize = mapping.MemorySize(mapping.DeviceSlice(dev));
                Y_ASSERT(deviceSize);
                Y_ASSERT(deviceSize == thisDevSize);
                for (ui32 column = 0; column < ColumnCount; ++column) {
                    copier.AddAsyncMemoryCopyTask(src.GetBuffer(dev),
                                                  NAligment::ColumnShift(srcMapping.MemoryUsageAt(dev), column),
                                                  GetBuffer(dev),
                                                  NAligment::ColumnShift(mapping.MemoryUsageAt(dev), column),
                                                  deviceSize);
                }
            }

            copier.SubmitCopy();
        }
    };

    template <class T, class TMapping, EPtrType Type>
    class TDeviceObjectExtractor<TCudaBuffer<T, TMapping, Type>> {
    public:
        using TMeta = typename TMapping::TMeta;
        using TRemoteObject = typename NKernelHost::TDeviceBuffer<T, Type>;

        static TRemoteObject At(ui32 devId, TCudaBuffer<T, TMapping, Type>& object) {
            return object.At(devId);
        }
    };

    template <class T, class TMapping, EPtrType Type>
    class TDeviceObjectExtractor<const TCudaBuffer<T, TMapping, Type>> {
    public:
        using TMeta = typename TMapping::TMeta;
        using TRemoteObject = typename NKernelHost::TDeviceBuffer<const T, Type>;

        static TRemoteObject At(ui32 devId, const TCudaBuffer<T, TMapping, Type>& object) {
            return object.At(devId);
        }
    };

    template <class T, class TMapping, EPtrType Type>
    class TDeviceObjectExtractor<TVector<const TCudaBuffer<T, TMapping, Type>*>> {
    public:
        using TMeta = typename TMapping::TMeta;
        using TRemoteObject = TVector<typename NKernelHost::TDeviceBuffer<const T, Type>>;

        static TRemoteObject At(ui32 devId, TVector<const TCudaBuffer<T, TMapping, Type>*> object) {
            using TBuffer = typename NKernelHost::TDeviceBuffer<const T, Type>;
            TRemoteObject deviceVector;
            for (auto ptr : object) {
                CB_ENSURE(ptr != nullptr, "Error: nullptr found");
                TBuffer buffer = ptr->At(devId);
                deviceVector.push_back(std::move(buffer));
            }
            return deviceVector;
        }
    };

    template <class T, class TMapping, EPtrType Type>
    class TDeviceObjectExtractor<const TCudaBuffer<T, TMapping, Type>*> {
    public:
        using TMeta = typename TMapping::TMeta;
        using TRemoteObject = typename NKernelHost::TDeviceBuffer<const T, Type>;

        static TRemoteObject At(ui32 devId, const TCudaBuffer<T, TMapping, Type>* object) {
            using TBuffer = typename NKernelHost::TDeviceBuffer<const T, Type>;
            if (object) {
                return object->At(devId);
            } else {
                return TBuffer();
            }
        }
    };

    template <class T, class TMapping, EPtrType Type>
    class TDeviceObjectExtractor<TCudaBuffer<T, TMapping, Type>*> {
    public:
        using TMeta = typename TMapping::TMeta;
        using TRemoteObject = typename NKernelHost::TDeviceBuffer<T, Type>;

        static TRemoteObject At(ui32 devId, TCudaBuffer<T, TMapping, Type>* object) {
            using TBuffer = typename NKernelHost::TDeviceBuffer<T, Type>;
            if (object) {
                return object->At(devId);
            } else {
                return TBuffer::Nullptr();
            }
        }
    };
}

using NCudaLib::TCudaBuffer;

template <class T>
using TStripeBuffer = NCudaLib::TCudaBuffer<T, NCudaLib::TStripeMapping>;

template <class T>
using TMirrorBuffer = NCudaLib::TCudaBuffer<T, NCudaLib::TMirrorMapping>;

template <class T>
using TSingleBuffer = NCudaLib::TCudaBuffer<T, NCudaLib::TSingleMapping>;

template <class T>
using TStripeHostBuffer = NCudaLib::TCudaBuffer<T, NCudaLib::TStripeMapping, NCudaLib::EPtrType::CudaHost>;

template <class T>
using TMirrorHostBuffer = NCudaLib::TCudaBuffer<T, NCudaLib::TMirrorMapping, NCudaLib::EPtrType::CudaHost>;

template <class T>
using TSingleHostBuffer = NCudaLib::TCudaBuffer<T, NCudaLib::TSingleMapping, NCudaLib::EPtrType::CudaHost>;

namespace NCudaLib {
    template <class T, EPtrType Type>
    inline TCudaBuffer<const T, TStripeMapping, Type> StripeView(const TCudaBuffer<const T, TMirrorMapping, Type>& buffer,
                                                                 const TStripeMapping& stripeMapping,
                                                                 ui32 column = 0) {
        CB_ENSURE(stripeMapping.GetObjectsSlice() == buffer.GetObjectsSlice());
        TCudaBuffer<const T, TStripeMapping, Type> stripeBuffer;
        stripeBuffer.Mapping = stripeMapping;

        auto& mirrorMapping = buffer.GetMapping();

        for (ui32 dev : stripeMapping.NonEmptyDevices()) {
            const ui64 columnsShift = NAligment::ColumnShift(mirrorMapping.MemoryUsageAt(dev), column);
            const auto devSlice = stripeMapping.DeviceSlice(dev);

            if (buffer.Buffers[dev].NotEmpty()) {
                stripeBuffer.Buffers[dev] = buffer.Buffers[dev].ShiftedBuffer(columnsShift + mirrorMapping.DeviceMemoryOffset(dev, devSlice));
            }
        }

        stripeBuffer.IsSliceView = true;
        return stripeBuffer;
    }

    template <class T, EPtrType Type>
    inline TCudaBuffer<const T, TStripeMapping, Type> StripeView(const TCudaBuffer<const T, TMirrorMapping, Type>& buffer) {
        CB_ENSURE(buffer.GetColumnCount() == 0);
        auto stripeMapping = TStripeMapping::SplitBetweenDevices(buffer.GetObjectsSlice().Size(), buffer.GetMapping().SingleObjectSize());
        return StripeView(buffer, stripeMapping, 0);
    }

    template <class T, EPtrType Type>
    inline TCudaBuffer<T, TStripeMapping, Type> ParallelStripeView(const TCudaBuffer<T, TStripeMapping, Type>& buffer,
                                                                   const TSlice& slice,
                                                                   ui32 column = 0) {
        const TStripeMapping& srcMapping = buffer.GetMapping();
        TCudaBuffer<T, TStripeMapping, Type> parallelViewBuffer;
        const ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();

        TVector<TSlice> srcSlices(devCount);
        TVector<TSlice> viewSlices(devCount);

        {
            TSlice firstDevSlice = srcMapping.DeviceSlice(0);
            CB_ENSURE(slice.Size() <= firstDevSlice.Size(), slice << " / " << firstDevSlice);

            ui32 cursor = 0;
            for (ui32 dev = 0; dev < devCount; ++dev) {
                srcSlices[dev] = TSlice(slice.Left + firstDevSlice.Size() * dev, slice.Right + firstDevSlice.Size() * dev);
                viewSlices[dev] = TSlice(cursor, cursor + slice.Size());
                cursor += slice.Size();
                CB_ENSURE(srcMapping.DeviceSlice(dev).Size() == firstDevSlice.Size());
            }

            parallelViewBuffer.Mapping = NCudaLib::TStripeMapping(std::move(viewSlices),
                                                                  srcMapping.SingleObjectSize());
        }

        for (ui32 dev : parallelViewBuffer.Mapping.NonEmptyDevices()) {
            const ui64 columnsShift = NAligment::ColumnShift(srcMapping.MemoryUsageAt(dev), column);

            const auto devSlice = srcSlices[dev];
            if (buffer.Buffers[dev].NotEmpty()) {
                parallelViewBuffer.Buffers[dev] = buffer.Buffers[dev].ShiftedBuffer(columnsShift + srcMapping.DeviceMemoryOffset(dev, devSlice));
            }
        }
        parallelViewBuffer.IsSliceView = true;

        return parallelViewBuffer;
    }

    template <class T>
    class TParallelStripeVectorBuilder {
    public:
        TParallelStripeVectorBuilder() {
            Data.resize(NCudaLib::GetCudaManager().GetDeviceCount());
        }

        void Add(const NCudaLib::TDistributedObject<T>& entry) {
            for (ui32 i = 0; i < Data.size(); ++i) {
                Data[i].push_back(entry.At(i));
            }
        }

        void Add(const TVector<T>& entry) {
            Y_ASSERT(entry.size() == Data.size());
            for (ui32 i = 0; i < Data.size(); ++i) {
                Data[i].push_back(entry[i]);
            }
        }

        void AddAll(const TVector<TVector<T>>& entries) {
            for (const auto& entry : entries) {
                Add(entry);
            }
        }

        template <NCudaLib::EPtrType Type>
        void Build(NCudaLib::TCudaBuffer<T, TStripeMapping, Type>& dst, ui32 stream = 0) {
            TMappingBuilder<NCudaLib::TStripeMapping> builder;
            TVector<T> flatData;
            flatData.reserve(Data.size() * Data[0].size());
            for (ui32 dev = 0; dev < Data.size(); ++dev) {
                builder.SetSizeAt(dev, Data[dev].size());
                for (const auto& entry : Data[dev]) {
                    flatData.push_back(entry);
                }
            }

            dst.Reset(builder.Build());
            dst.Write(flatData, stream);
        }

    private:
        TVector<TVector<T>> Data;
    };

    template <class T>
    class TStripeVectorBuilder {
    public:
        TStripeVectorBuilder() {
            Data.resize(NCudaLib::GetCudaManager().GetDeviceCount());
        }

        TStripeVectorBuilder& Add(ui32 dev, T val) {
            CB_ENSURE(dev < Data.size(), "Error: invalid devices #" << dev);
            Data[dev].push_back(val);
            ++TotalData;
            return *this;
        }

        ui64 GetCurrentSize(ui32 dev) const {
            return Data[dev].size();
        }

        template <NCudaLib::EPtrType Type>
        void Build(NCudaLib::TCudaBuffer<T, TStripeMapping, Type>& dst, ui32 stream = 0) {
            TMappingBuilder<NCudaLib::TStripeMapping> builder;
            TVector<T> flatData;
            flatData.reserve(TotalData);

            for (ui32 dev = 0; dev < Data.size(); ++dev) {
                builder.SetSizeAt(dev, Data[dev].size());
                for (const auto& entry : Data[dev]) {
                    flatData.push_back(entry);
                }
            }

            dst.Reset(builder.Build());
            dst.Write(flatData,
                      stream);
        }

    private:
        TVector<TVector<T>> Data;
        ui64 TotalData = 0;
    };

    template <bool IsConst>
    struct TMaybeConstView;

    template <>
    struct TMaybeConstView<true> {
        template <class T, class TMapping>
        using TBuffer = TCudaBuffer<const T, TMapping>;
    };

    template <>
    struct TMaybeConstView<false> {
        template <class T, class TMapping>
        using TBuffer = TCudaBuffer<T, TMapping>;
    };

    template <class T>
    inline TVector<TDistributedObject<std::remove_const_t<T>>> ReadToDistributedObjectVec(const TStripeBuffer<T>& src) {
        using T_ = std::remove_const_t<T>;
        TVector<T_> tmp;
        src.Read(tmp);
        ui32 devCount = NCudaLib::GetCudaManager().GetDeviceCount();
        TVector<TDistributedObject<T_>> result;

        for (ui32 dev = 0; dev < devCount; ++dev) {
            CB_ENSURE(src.GetMapping().DeviceSlice(dev).Size() == src.GetMapping().DeviceSlice(0).Size());
        }
        ui32 size = static_cast<ui32>(tmp.size() / devCount);

        for (ui32 i = 0; i < size; ++i) {
            TDistributedObject<T_> obj = NCudaLib::GetCudaManager().CreateDistributedObject<T_>();
            for (ui32 dev = 0; dev < devCount; ++dev) {
                obj.Set(dev, tmp[i + dev * size]);
            }
            result.push_back(obj);
        }
        return result;
    }
}
