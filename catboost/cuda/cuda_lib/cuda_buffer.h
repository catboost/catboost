#pragma once

#include "cuda_base.h"
#include "gpu_memory_pool.h"
#include "memory_provider_trait.h"
#include "remote_objects.h"
#include "single_device.h"
#include "cuda_kernel_buffer.h"
#include "cuda_manager.h"
#include "single_host_memory_copy_tasks.h"
#include "mapping.h"
#include "cpu_reducers.h"
#include "cache.h"
#include <util/ysafeptr.h>
#include <util/generic/vector.h>

namespace NCudaLib {
    template <class TCudaBuffer>
    class TCudaBufferReader: public TMoveOnly {
    private:
        const TCudaBuffer* Buffer;
        ui32 Stream = 0;
        TVector<TDeviceEvent> ReadDone;

        TSlice FactorSlice;
        //slice to read: ReadSlice read all slices, that equal mod ReduceSlice to read slice
        TSlice ReadSlice;
        //columns to read
        TSlice ColumnReadSlice;

    protected:
        ui64 ReadMemorySize(const TSlice& slice) const {
            auto& mapping = Buffer->GetMapping();
            return mapping.MemorySize(slice);
        }

        TVector<TSlice> GetReadSlices() {
            Y_ASSERT(ReadSlice.Right <= FactorSlice.Right);
            CB_ENSURE(ReadSlice.Right <= FactorSlice.Right);

            auto& mapping = Buffer->GetMapping();
            TSlice objectsSlice = mapping.GetObjectsSlice();

            TVector<TSlice> readSlices;
            if (ReadSlice.IsEmpty()) {
                return readSlices;
            }

            ui64 step = FactorSlice.Size();
            for (TSlice cursor = ReadSlice; cursor.Right <= objectsSlice.Right; cursor += step) {
                readSlices.push_back(cursor);
            }

            return readSlices;
        }

    public:
        using T = typename TCudaBuffer::TValueType;
        using NonConstT = typename std::remove_const<typename TCudaBuffer::TValueType>::type;

        TCudaBufferReader(const TCudaBuffer& buffer)
            : Buffer(&buffer)
            , FactorSlice(buffer.GetMapping().GetObjectsSlice())
            , ReadSlice(buffer.GetMapping().GetObjectsSlice())
            , ColumnReadSlice(buffer.GetColumnSlice())
        {
        }

        TCudaBufferReader& SetReadSlice(const TSlice& slice) {
            ReadSlice = slice;
            return *this;
        }

        TCudaBufferReader& SetFactorSlice(const TSlice& slice) {
            FactorSlice = slice;
            return *this;
        }

        TCudaBufferReader& SetColumnReadSlice(const TSlice& slice) {
            ColumnReadSlice = slice;
            return *this;
        }

        TCudaBufferReader& SetCustomReadingStream(ui32 stream) {
            Stream = stream;
            return *this;
        }

        void WaitComplete() {
            for (auto& task : ReadDone) {
                task.WaitComplete();
            }
        }

        void ReadAsync(TVector<NonConstT>& dst) {
            auto readSlices = GetReadSlices();
            ui64 singleSliceSize = ReadMemorySize(ReadSlice) * ColumnReadSlice.Size();
            dst.resize(readSlices.size() * singleSliceSize);

            for (ui64 i = 0; i < readSlices.size(); ++i) {
                SubmitReadAsync(~dst + i * singleSliceSize, readSlices[i]);
            }
        }

        void Read(TVector<NonConstT>& dst) {
            ReadAsync(dst);
            WaitComplete();
        }

        template <class TReducer = NReducers::TSumReducer<NonConstT>>
        void ReadReduce(TVector<NonConstT>& dst) {
            Read(dst);
            ui64 reduceStep = FactorSlice.Size();
            auto& mapping = Buffer->GetMapping();
            TSlice objectsSlice = mapping.GetObjectsSlice();

            auto reduceSize = mapping.MemorySize(ReadSlice);

            TSlice reduceSlice = ReadSlice;
            reduceSlice += reduceStep;

            for (; reduceSlice.Right <= objectsSlice.Right; reduceSlice += reduceStep) {
                auto appendOffset = mapping.MemoryOffset(reduceSlice);
                auto appendSize = mapping.MemorySize(reduceSlice);
                CB_ENSURE(appendSize == reduceSize, "Error: reduce size should be equal append size");
                TReducer::Reduce(~dst, ~dst + appendOffset, appendSize);
            }
            dst.resize(reduceSize);
        }

        void SubmitReadAsync(NonConstT* to, const TSlice& readSlice) {
            if (readSlice.Size()) {
                auto& mapping = Buffer->GetMapping();
                const ui64 skipOffset = mapping.MemoryOffset(readSlice);

                for (ui64 column = ColumnReadSlice.Left; column < ColumnReadSlice.Right; ++column) {
                    TVector<TSlice> currentSlices;
                    currentSlices.push_back(readSlice);

                    for (auto dev : Buffer->NonEmptyDevices()) {
                        TVector<TSlice> nextSlices;

                        for (auto& slice : currentSlices) {
                            const TSlice& deviceSlice = mapping.DeviceSlice(dev);
                            auto deviceSlicePart = TSlice::Intersection(slice, deviceSlice);

                            if (!deviceSlicePart.IsEmpty()) {
                                if (slice.Left < deviceSlicePart.Left) {
                                    nextSlices.push_back({slice.Left, deviceSlicePart.Left});
                                }

                                if (slice.Right > deviceSlicePart.Right) {
                                    nextSlices.push_back({deviceSlicePart.Right, slice.Right});
                                }

                                const ui64 localDataOffset = mapping.DeviceMemoryOffset(dev, deviceSlicePart) +
                                                             column * mapping.MemoryUsageAt(dev);
                                const ui64 writeOffset = mapping.MemoryOffset(deviceSlicePart) - skipOffset;

                                ReadDone.push_back(
                                    TDataCopier::AsyncRead(Buffer->GetBuffer(dev), Stream, localDataOffset,
                                                           to + writeOffset,
                                                           mapping.MemorySize(deviceSlicePart)));
                            } else {
                                nextSlices.push_back(slice);
                            }
                        }

                        if (nextSlices.size() == 0) {
                            break;
                        }
                        currentSlices.swap(nextSlices);
                    }

                    to += mapping.MemorySize(readSlice);
                }
            }
        }
    };

    template <class TCudaBuffer>
    class TCudaBufferWriter: public TMoveOnly {
    private:
        using T = typename TCudaBuffer::TValueType;
        const T* Src;
        TCudaBuffer* Dst;
        ui64 SrcMaxSize;
        TSlice WriteSlice;
        ui32 Stream = 0;
        bool Async = false;
        TVector<TDeviceEvent> WriteDone;
        ui64 SrcOffset = 0;
        TSlice ColumnWriteSlice;

    public:
        TCudaBufferWriter(const TVector<T>& src,
                          TCudaBuffer& dst)
            : Src(~src)
            , Dst(&dst)
            , SrcMaxSize(src.size())
            , WriteSlice(dst.GetMapping().GetObjectsSlice())
            , ColumnWriteSlice(dst.GetColumnSlice())
        {
        }

        TCudaBufferWriter(TCudaBufferWriter&& other) = default;

        ~TCudaBufferWriter() throw (yexception) {
            for (auto& event : WriteDone) {
                CB_ENSURE(event.IsComplete());
            }
        }

        TCudaBufferWriter& SetWriteSlice(const TSlice& slice) {
            WriteSlice = slice;
            return *this;
        }

        TCudaBufferWriter& SetCustomWritingStream(ui32 stream) {
            Stream = stream;
            return *this;
        }

        TCudaBufferWriter& SetColumnWriteSlice(const TSlice& slice) {
            ColumnWriteSlice = slice;
            return *this;
        }

        TCudaBufferWriter& SetCustomSrcOffset(ui64 offset) {
            SrcOffset = offset;
            return *this;
        }

        void Write() {
            const auto& mapping = Dst->GetMapping();

            for (auto dev : Dst->NonEmptyDevices()) {
                ui64 columnOffset = 0;

                for (ui64 column = ColumnWriteSlice.Left; column < ColumnWriteSlice.Right; ++column) {
                    auto deviceSlice = mapping.DeviceSlice(dev);
                    auto intersection = TSlice::Intersection(WriteSlice, deviceSlice);
                    const ui64 memoryUsageAtDevice = mapping.MemoryUsageAt(dev);

                    if (!intersection.IsEmpty()) {
                        const auto localWriteOffset = mapping.DeviceMemoryOffset(dev, intersection) + memoryUsageAtDevice * column;
                        const ui64 writeSize = mapping.MemorySize(intersection);
                        ui64 readOffset = mapping.MemoryOffset(intersection);
                        CB_ENSURE(readOffset >= SrcOffset);
                        readOffset -= SrcOffset;
                        CB_ENSURE(writeSize <= SrcMaxSize);

                        WriteDone.push_back(TDataCopier::AsyncWrite(Src + readOffset + columnOffset, Dst->GetBuffer(dev), Stream,
                                                                    localWriteOffset,
                                                                    writeSize));
                    }
                    columnOffset += memoryUsageAtDevice;
                }
            }

            if (!Async) {
                WaitComplete();
            }
        }

        void WaitComplete() {
            for (auto& task : WriteDone) {
                task.WaitComplete();
            }
        }
    };

    template <class T,
              class TMapping,
              EPtrType Type = CudaDevice>
    class TCudaBuffer: public TMoveOnly {
    private:
        using TRawPtr = typename TMemoryProviderImplTrait<Type>::TRawFreeMemory;
        using TBuffer = TCudaSingleDevice::TSingleBuffer<T, Type>;
        using TDeviceMeta = typename TMapping::TMeta;
        using TDeviceBuffer = typename NKernelHost::TDeviceBuffer<T, TDeviceMeta, Type>;
        using TConstDeviceBuffer = typename NKernelHost::TDeviceBuffer<const T, TDeviceMeta, Type>;
        TMapping Mapping;
        TVector<TBuffer> Buffers;
        ui64 ColumnCount = 1;
        mutable bool CreatetedFromScratchFlag = false;
        mutable bool IsSliceView = false;
        bool ReadOnly = false;

        void EnsureSize(ui32 devId, ui64 size, bool freeMemory) {
            size *= ColumnCount;
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
            if (CreatetedFromScratchFlag) {
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
                CreatetedFromScratchFlag = true;
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
        friend class TLatencyAndBandwidthStats;

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
                    const ui64 columnsShift = Mapping.MemoryUsageAt(i) * column;
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
                    const ui64 columnsShift = Mapping.MemoryUsageAt(i) * column;
                    const auto devSlice = TSlice::Intersection(slice, Mapping.DeviceSlice(i));
                    buffer.Buffers[i] = Buffers[i].ShiftedBuffer(
                        columnsShift + Mapping.DeviceMemoryOffset(i, devSlice));
                }
            }
            buffer.IsSliceView = true;
            return buffer;
        }

        TCudaBuffer<const T, TMapping, Type> ColumnView(ui64 column) const {
            TCudaBuffer<const T, TMapping, Type> buffer;
            auto slice = GetObjectsSlice();
            buffer.Mapping = Mapping.ToLocalSlice(slice);

            for (ui64 i = 0; i < Buffers.size(); ++i) {
                if (Buffers[i].NotEmpty()) {
                    const ui64 columnsShift = Mapping.MemoryUsageAt(i) * column;
                    const auto devSlice = TSlice::Intersection(slice, Mapping.DeviceSlice(i));
                    buffer.Buffers[i] = Buffers[i].ShiftedConstBuffer(
                        columnsShift + Mapping.DeviceMemoryOffset(i, devSlice));
                }
            }
            buffer.IsSliceView = true;
            return buffer;
        }

        TCudaBuffer<T, TMapping, Type> ColumnView(ui64 column) {
            TCudaBuffer<T, TMapping, Type> buffer;
            auto slice = GetObjectsSlice();
            buffer.Mapping = Mapping.ToLocalSlice(slice);

            for (ui64 i = 0; i < Buffers.size(); ++i) {
                if (Buffers[i].NotEmpty()) {
                    const ui64 columnsShift = Mapping.MemoryUsageAt(i) * column;
                    const auto devSlice = TSlice::Intersection(slice, Mapping.DeviceSlice(i));
                    buffer.Buffers[i] = Buffers[i].ShiftedBuffer(
                        columnsShift + Mapping.DeviceMemoryOffset(i, devSlice));
                }
            }
            buffer.IsSliceView = true;
            return buffer;
        }

        TCudaBuffer<T, TMapping, Type> CopyView() {
            return SliceView(GetObjectsSlice());
        }

        TCudaBuffer<const T, TMapping, Type> ConstCopyView() const {
            return SliceView(GetObjectsSlice());
        }

        //TODO: get rid of this. Refactor everything to const in template type instead of holder-objects
        TCudaBuffer<T, TSingleMapping, Type> DeviceView(ui32 devId, ui64 column = 0) const {
            TCudaBuffer<T, TSingleMapping, Type> buffer;
            const ui64 columnsShift = Mapping.MemoryUsageAt(devId) * column;
            const auto devSlice = Mapping.DeviceSlice(devId);
            buffer.Mapping = TSingleMapping(devId, devSlice.Size(), Mapping.SingleObjectSize());
            if (Buffers.at(devId).NotEmpty()) {
                buffer.Buffers.at(devId) = Buffers.at(devId).ShiftedBuffer(
                    columnsShift + Mapping.DeviceMemoryOffset(devId, devSlice));
            }
            buffer.IsSliceView = true;
            buffer.ReadOnly = true;
            return buffer;
        }

        TCudaBuffer<const T, TSingleMapping, Type> ConstDeviceView(ui32 devId, ui64 column = 0) const {
            TCudaBuffer<const T, TSingleMapping, Type> buffer;
            const ui64 columnsShift = Mapping.MemoryUsageAt(devId) * column;
            const auto devSlice = Mapping.DeviceSlice(devId);
            buffer.Mapping = TSingleMapping(devId, devSlice.Size(), Mapping.SingleObjectSize());
            if (Buffers.at(devId).NotEmpty()) {
                buffer.Buffers.at(devId) = Buffers.at(devId).ShiftedConstBuffer(
                    columnsShift + Mapping.DeviceMemoryOffset(devId, devSlice));
            }
            buffer.IsSliceView = true;
            return buffer;
        }

        operator TCudaBuffer<const T, TMapping, Type>() const {
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
            buffer.CreatetedFromScratchFlag = CreatetedFromScratchFlag;

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

            return TDeviceBuffer(BufferAt(devId).GetPointer(), Mapping.At(devId), ColumnCount);
        };

        TConstDeviceBuffer At(ui64 devId) const {
            Y_ASSERT(devId < Buffers.size());
            return TConstDeviceBuffer(BufferAt(devId).GetPointer(), Mapping.At(devId), ColumnCount);
        };

        TSlice GetColumnSlice() const {
            return TSlice(0, ColumnCount);
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
            static_assert(std::is_const<T>::value == false, "Can't reset const buffer");
            TCudaBuffer::SetMapping(mapping, *this, false);
        }

        void Reset(const TMapping& mapping,
                   ui32 columnCount) {
            CB_ENSURE(CanReset(), "Error: buffer is view of some other data. can't reset it");
            ColumnCount = columnCount;
            TCudaBuffer::SetMapping(mapping, *this, false);
        }

        static TCudaBuffer Create(const TMapping& mapping, ui64 columnCount = 1) {
            TCudaBuffer buffer(columnCount);
            SetMapping(mapping, buffer);
            buffer.CreatetedFromScratchFlag = true;
            return buffer;
        }

        static void Swap(TCudaBuffer& lhs, TCudaBuffer& rhs) {
            using std::swap;
            swap(lhs.Buffers, rhs.Buffers);
            swap(lhs.ColumnCount, rhs.ColumnCount);
            swap(lhs.Mapping, rhs.Mapping);
            swap(lhs.CreatetedFromScratchFlag, rhs.CreatetedFromScratchFlag);
            swap(lhs.IsSliceView, rhs.IsSliceView);
        }

        void Clear() {
            TCudaBuffer empty;
            Swap(*this, empty);
        }

        static constexpr EPtrType PtrType() {
            return Type;
        }

        TCudaBufferWriter<TCudaBuffer> CreateWriter(const TVector<T>& src) {
            return TCudaBufferWriter<TCudaBuffer>(src, *this);
        }

        TCudaBufferReader<TCudaBuffer> CreateReader() const {
            return TCudaBufferReader<TCudaBuffer>(*this);
        }

        void Write(const TVector<T>& src, ui32 stream = 0) {
            CreateWriter(src).SetCustomWritingStream(stream).Write();
        }

        void Read(TVector<typename std::remove_const<T>::type>& dst, ui32 stream = 0) const {
            CreateReader().SetCustomReadingStream(stream).Read(dst);
        }

        template <class TC, EPtrType PtrType>
        static TCudaBuffer CopyMapping(const TCudaBuffer<TC, TMapping, PtrType>& other) {
            return Create(other.Mapping);
        }

        template <class TC, EPtrType SrcType>
        void Copy(const TCudaBuffer<TC, TMapping, SrcType>& src, ui32 stream = 0) {
            const auto& mapping = GetMapping();
            const TMapping& srcMapping = src.GetMapping();

            TDataCopier copier(stream);

            for (const auto dev : srcMapping.NonEmptyDevices()) {
                ui64 deviceSize = srcMapping.MemorySize(srcMapping.DeviceSlice(dev));
                ui64 thisDevSize = mapping.MemorySize(mapping.DeviceSlice(dev));
                Y_ASSERT(deviceSize);
                Y_ASSERT(deviceSize == thisDevSize);

                copier.AddAsyncMemoryCopyTask(src.GetBuffer(dev), 0,
                                              GetBuffer(dev), 0,
                                              deviceSize);
            }
            copier.SubmitCopy();
        }
    };

    template <class T, class TMapping, EPtrType Type>
    class TDeviceObjectExtractor<TCudaBuffer<T, TMapping, Type>> {
    public:
        using TMeta = typename TMapping::TMeta;
        using TRemoteObject = typename NKernelHost::TDeviceBuffer<T, TMeta, Type>;

        static TRemoteObject At(ui32 devId, TCudaBuffer<T, TMapping, Type>& object) {
            return object.At(devId);
        }
    };

    template <class T, class TMapping, EPtrType Type>
    class TDeviceObjectExtractor<const TCudaBuffer<T, TMapping, Type>> {
    public:
        using TMeta = typename TMapping::TMeta;
        using TRemoteObject = typename NKernelHost::TDeviceBuffer<const T, TMeta, Type>;

        static TRemoteObject At(ui32 devId, const TCudaBuffer<T, TMapping, Type>& object) {
            return object.At(devId);
        }
    };

    template <class T, class TMapping, EPtrType Type>
    class TDeviceObjectExtractor<TVector<const TCudaBuffer<T, TMapping, Type>*>> {
    public:
        using TMeta = typename TMapping::TMeta;
        using TRemoteObject = TVector<typename NKernelHost::TDeviceBuffer<const T, TMeta, Type>>;

        static TRemoteObject At(ui32 devId, TVector<const TCudaBuffer<T, TMapping, Type>*> object) {
            using TBuffer = typename NKernelHost::TDeviceBuffer<const T, TMeta, Type>;
            TRemoteObject deviceVector;
            for (auto ptr : object) {
                Y_ENSURE(ptr != nullptr, "Error: nullptr found");
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
        using TRemoteObject = typename NKernelHost::TDeviceBuffer<const T, TMeta, Type>;

        static TRemoteObject At(ui32 devId, const TCudaBuffer<T, TMapping, Type>* object) {
            using TBuffer = typename NKernelHost::TDeviceBuffer<const T, TMeta, Type>;
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
        using TRemoteObject = typename NKernelHost::TDeviceBuffer<T, TMeta, Type>;

        static TRemoteObject At(ui32 devId, TCudaBuffer<T, TMapping, Type>* object) {
            using TBuffer = typename NKernelHost::TDeviceBuffer<T, TMeta, Type>;
            if (object) {
                return object->At(devId);
            } else {
                return TBuffer();
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
            const ui64 columnsShift = mirrorMapping.MemoryUsageAt(dev) * column;
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
            CB_ENSURE(slice.Size() < firstDevSlice.Size());

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
            const ui64 columnsShift = srcMapping.MemoryUsageAt(dev) * column;

            const auto devSlice = srcSlices[dev];
            if (buffer.Buffers[dev].NotEmpty()) {
                parallelViewBuffer.Buffers[dev] = buffer.Buffers[dev].ShiftedBuffer(columnsShift + srcMapping.DeviceMemoryOffset(dev, devSlice));
            }
        }
        parallelViewBuffer.IsSliceView = true;

        return parallelViewBuffer;
    }
}
