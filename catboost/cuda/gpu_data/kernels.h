#pragma once

#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/cuda/gpu_data/gpu_structures.h>
#include <catboost/cuda/gpu_data/kernel/binarize.cuh>
#include <catboost/cuda/gpu_data/kernel/lazy_write_compresed_index.cuh>
#include <catboost/cuda/gpu_data/kernel/query_helper.cuh>
#include <catboost/cuda/cuda_util/compression_helpers_gpu.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/options/binarization_options.h>
#include <catboost/private/libs/quantized_pool/loader.h>

#include <library/cpp/grid_creator/binarization.h>
#include <util/ysaveload.h>

inline static ui8 ClipWideHistValue(ui16 wideValue, ui16 baseValue) {
    return Min(Max(wideValue - baseValue, 0), 255);
}

namespace NKernelHost {
    class TFindBordersKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Feature;
        EBorderSelectionType BorderType;
        ui32 BorderCount;
        TCudaBufferPtr<float> Dst;

    public:
        TFindBordersKernel() = default;

        TFindBordersKernel(TCudaBufferPtr<const float> feature,
                           EBorderSelectionType borderType,
                           ui32 borderCount,
                           TCudaBufferPtr<float> dst)
            : Feature(feature)
            , BorderType(borderType)
            , BorderCount(borderCount)
            , Dst(dst)
        {
        }

        Y_SAVELOAD_DEFINE(Feature, Dst, BorderCount, BorderType);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Dst.Size() > BorderCount);

            if (BorderType == EBorderSelectionType::Median) {
                NKernel::FastGpuBorders(Feature.Get(), Feature.Size(), Dst.Get(),
                                        BorderCount, stream.GetStream());
            } else if (BorderType == EBorderSelectionType::Uniform) {
                NKernel::ComputeUniformBorders(Feature.Get(), static_cast<ui32>(Feature.Size()),
                                               Dst.Get(), BorderCount,
                                               stream.GetStream());
            } else {
                ythrow TCatBoostException() << "Error: unsupported binarization for combinations ctrs "
                                            << BorderType;
            }
        }
    };

    class TBinarizeFloatFeatureKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> Values;
        TCudaBufferPtr<const float> Borders;
        TCFeature Feature;
        TCudaBufferPtr<ui32> Dst;
        TCudaBufferPtr<const ui32> GatherIndex;
        bool AtomicUpdate;

    public:
        TBinarizeFloatFeatureKernel() = default;

        TBinarizeFloatFeatureKernel(TCudaBufferPtr<const float> values, TCudaBufferPtr<const float> borders, TCFeature feature, TCudaBufferPtr<const ui32> gatherIndex, TCudaBufferPtr<ui32> dst, bool atomicUpdate)
            : Values(values)
            , Borders(borders)
            , Feature(feature)
            , Dst(dst)
            , GatherIndex(gatherIndex)
            , AtomicUpdate(atomicUpdate)
        {
        }

        Y_SAVELOAD_DEFINE(Values, Borders, Feature, Dst, GatherIndex, AtomicUpdate);

        void Run(const TCudaStream& stream) const {
            NKernel::BinarizeFloatFeature(Values.Get(), static_cast<ui32>(Values.Size()),
                                          Borders.Get(), Feature,
                                          Dst.Get(),
                                          GatherIndex.Get(),
                                          AtomicUpdate,
                                          stream.GetStream());
        }
    };

    template <NCudaLib::EPtrType Type>
    class TWriteCompressedIndexKernel: public TStatelessKernel {
    private:
        TDeviceBuffer<const ui8, Type> Bins;
        TCFeature Feature;
        TCudaBufferPtr<ui32> Dst;

    public:
        TWriteCompressedIndexKernel() = default;

        TWriteCompressedIndexKernel(TDeviceBuffer<const ui8, Type> bins,
                                    TCFeature feature,
                                    TCudaBufferPtr<ui32> cindex)
            : Bins(bins)
            , Feature(feature)
            , Dst(cindex)
        {
        }

        Y_SAVELOAD_DEFINE(Bins, Feature, Dst);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Feature.Mask != 0);
            CB_ENSURE(Feature.Offset != (ui64)(-1));
            NKernel::WriteCompressedIndex(Feature,
                                          Bins.Get(),
                                          Bins.Size(),
                                          Dst.Get(),
                                          stream.GetStream());
        }
    };

    class TWriteLazyCompressedIndexKernel: public TKernelBase<NKernel::TLazyWirteCompressedIndexKernelContext, false> {
    private:
        NCB::TPathWithScheme PathWithScheme;
        ui32 DatasetFeatureId;
        ui32 FeatureId;
        TCFeature Feature;
        TCudaBufferPtr<ui32> Dst;
        TSlice DeviceSlice;
        ui64 SingleObjectSize = 1;
        TMaybe<ui16> BaseValue = Nothing();

        NCB::TDatasetSubset GetLoadSubset() const {
            return NCB::TDatasetSubset::MakeRange(DeviceSlice.Left, DeviceSlice.Right);
        }

        TStringBuilder GetDeviceTag() const {
            const auto deviceId = Dst.GetDeviceId();
            return TStringBuilder() << "Device(" << deviceId.DeviceId << ")@Host(" << deviceId.HostId << ")";
        }

    public:
        using TKernelContext = NKernel::TLazyWirteCompressedIndexKernelContext;

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            CATBOOST_DEBUG_LOG << GetDeviceTag() << ": " << __PRETTY_FUNCTION__ << Endl;
            CB_ENSURE_INTERNAL(DeviceSlice.NotEmpty(), GetDeviceTag() << ": slice is empty");
            auto context = MakeHolder<TKernelContext>();
            context->TempStorage = memoryManager.Allocate<ui8>(DeviceSlice.Size() * SingleObjectSize);
            return context;
        }

        TWriteLazyCompressedIndexKernel() = default;

        TWriteLazyCompressedIndexKernel(const NCB::TPathWithScheme& pathWithScheme,
                                    ui32 datasetFeatureId,
                                    const TSlice& deviceSlice,
                                    ui64 singleObjectSize,
                                    ui32 featureId,
                                    TCFeature feature,
                                    TMaybe<ui16> baseValue,
                                    TCudaBufferPtr<ui32> cindex)
            : PathWithScheme(pathWithScheme)
            , DatasetFeatureId(datasetFeatureId)
            , FeatureId(featureId)
            , Feature(feature)
            , Dst(cindex)
            , DeviceSlice(deviceSlice)
            , SingleObjectSize(singleObjectSize)
            , BaseValue(baseValue)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) const {
            CATBOOST_DEBUG_LOG << GetDeviceTag() << ": " << __PRETTY_FUNCTION__ << Endl;
            CB_ENSURE(Feature.Mask != 0);
            CB_ENSURE(Feature.Offset != (ui64)(-1));
            CB_ENSURE_INTERNAL(Dst.Get(), GetDeviceTag() << ": Dst.Get() returns nullptr");
            TDeviceBuffer<ui8, EPtrType::CudaDevice> deviceBins(
                context.TempStorage,
                TObjectsMeta(DeviceSlice.Size(), SingleObjectSize),
                /*columnCount*/1,
                Dst.GetDeviceId());

            auto poolLoader = NCB::TQuantizedPoolLoadersCache::GetLoader(PathWithScheme, GetLoadSubset());
            CATBOOST_DEBUG_LOG << GetDeviceTag() << ": have loader " << Hex((size_t)poolLoader.Get()) << Endl;
            const auto rawBytes = poolLoader->LoadQuantizedColumn(DatasetFeatureId, DeviceSlice.Left, DeviceSlice.Size());
            CB_ENSURE_INTERNAL(rawBytes.size() > 0, GetDeviceTag() << ": LoadQuantizedColumn returns empty vector");
            if (DeviceSlice.Size() == rawBytes.size()) {
                deviceBins.Write(rawBytes, stream);
            } else {
                CB_ENSURE(
                    BaseValue.Defined() && rawBytes.size() == DeviceSlice.Size() * sizeof(ui16),
                    GetDeviceTag() << ": "
                    "wide column size in bytes (" << rawBytes.size() << ") "
                    "mismatches size of device slice (" << DeviceSlice.Size() << ")");
                TVector<ui8> bins;
                bins.yresize(DeviceSlice.Size());
                for (auto i : xrange(DeviceSlice.Size())) {
                    bins[i] = ClipWideHistValue(reinterpret_cast<const ui16*>(rawBytes.data())[i], *BaseValue);
                }
                deviceBins.Write(bins, stream);
            }

            NKernel::WriteCompressedIndex(Feature,
                                          deviceBins.Get(),
                                          deviceBins.Size(),
                                          Dst.Get(),
                                          stream.GetStream());
        }

        inline void Save(IOutputStream* s) const {
            ::SaveMany(s, FeatureId, Feature, Dst, PathWithScheme, DatasetFeatureId, DeviceSlice, SingleObjectSize, BaseValue);
        }

        inline void Load(IInputStream* s) {
            CATBOOST_DEBUG_LOG << GetDeviceTag() << ": " << __PRETTY_FUNCTION__ << Endl;
            ::LoadMany(s, FeatureId, Feature, Dst, PathWithScheme, DatasetFeatureId, DeviceSlice, SingleObjectSize, BaseValue);
            NCB::TQuantizedPoolLoadersCache::GetLoader(PathWithScheme, GetLoadSubset());
        }
    };

    class TDropAllLoaders: public TStatelessKernel {
    public:
        TDropAllLoaders() = default;

        inline void Save(IOutputStream*) const {}

        inline void Load(IInputStream*) {}

        void Run(const TCudaStream&) const {
            CATBOOST_DEBUG_LOG << __PRETTY_FUNCTION__ << Endl;
            NCB::TQuantizedPoolLoadersCache::DropAllLoaders();
        }
    };

    class TComputeQueryIdsKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> QSizes;
        TCudaBufferPtr<const ui32> QOffsets;
        ui32 OffsetsBias;
        TCudaBufferPtr<ui32> Dst;

    public:
        TComputeQueryIdsKernel() = default;

        TComputeQueryIdsKernel(TCudaBufferPtr<const ui32> qSizes, TCudaBufferPtr<const ui32> qOffsets, ui32 offsetsBias, TCudaBufferPtr<ui32> dst)
            : QSizes(qSizes)
            , QOffsets(qOffsets)
            , OffsetsBias(offsetsBias)
            , Dst(dst)
        {
        }

        Y_SAVELOAD_DEFINE(QSizes, QOffsets, OffsetsBias, Dst);

        void Run(const TCudaStream& stream) const {
            NKernel::ComputeGroupIds(QSizes.Get(), QOffsets.Get(), OffsetsBias, QSizes.Size(), Dst.Get(), stream.GetStream());
        }
    };

    class TFillQueryEndMaskKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const ui32> Qids;
        TCudaBufferPtr<const ui32> Docs;
        TCudaBufferPtr<ui32> Masks;

    public:
        TFillQueryEndMaskKernel() = default;

        TFillQueryEndMaskKernel(TCudaBufferPtr<const ui32> qids, TCudaBufferPtr<const ui32> docs, TCudaBufferPtr<ui32> masks)
            : Qids(qids)
            , Docs(docs)
            , Masks(masks)
        {
        }

        Y_SAVELOAD_DEFINE(Qids, Docs, Masks);

        void Run(const TCudaStream& stream) const {
            NKernel::FillQueryEndMask(Qids.Get(), Docs.Get(), Docs.Size(), Masks.Get(), stream.GetStream());
        }
    };

    class TCreateKeysForSegmentedDocsSampleKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<ui64> Seeds;
        TCudaBufferPtr<const ui32> Qids;
        TCudaBufferPtr<ui64> Keys;

    public:
        TCreateKeysForSegmentedDocsSampleKernel() = default;

        TCreateKeysForSegmentedDocsSampleKernel(TCudaBufferPtr<ui64> seeds,
                                                TCudaBufferPtr<const ui32> qids,
                                                TCudaBufferPtr<ui64> keys)
            : Seeds(seeds)
            , Qids(qids)
            , Keys(keys)
        {
        }

        Y_SAVELOAD_DEFINE(Seeds, Qids, Keys);

        void Run(const TCudaStream& stream) const {
            CB_ENSURE(Qids.Size() == Keys.Size(), "Number of keys and query ids should be same");
            NKernel::CreateSortKeys(Seeds.Get(), Seeds.Size(), Qids.Get(), Qids.Size(), Keys.Get(), stream.GetStream());
        }
    };

    class TFillTakenDocsMaskKernel: public TStatelessKernel {
    private:
        TCudaBufferPtr<const float> TakenQueryMasks;
        TCudaBufferPtr<const ui32> Qids;
        TCudaBufferPtr<const ui32> Docs;
        TCudaBufferPtr<const ui32> QueryOffsets;
        ui32 QueryOffsetsBias;
        TCudaBufferPtr<const ui32> QuerySizes;
        float DocwiseSampleRate;
        ui32 MaxQuerySize;
        TCudaBufferPtr<float> TakenMask;

    public:
        TFillTakenDocsMaskKernel() = default;

        TFillTakenDocsMaskKernel(TCudaBufferPtr<const float> takenQueryMasks, TCudaBufferPtr<const ui32> qids, TCudaBufferPtr<const ui32> docs, TCudaBufferPtr<const ui32> queryOffsets, const ui32 queryOffsetsBias, TCudaBufferPtr<const ui32> querySizes, const float docwiseSampleRate, const ui32 maxQuerySize, TCudaBufferPtr<float> takenMask)
            : TakenQueryMasks(takenQueryMasks)
            , Qids(qids)
            , Docs(docs)
            , QueryOffsets(queryOffsets)
            , QueryOffsetsBias(queryOffsetsBias)
            , QuerySizes(querySizes)
            , DocwiseSampleRate(docwiseSampleRate)
            , MaxQuerySize(maxQuerySize)
            , TakenMask(takenMask)
        {
        }

        Y_SAVELOAD_DEFINE(TakenQueryMasks, Qids, Docs, QueryOffsets, QueryOffsetsBias, QuerySizes, DocwiseSampleRate, MaxQuerySize, TakenMask);

        void Run(const TCudaStream& stream) const {
            NKernel::FillTakenDocsMask(TakenQueryMasks.Get(), Qids.Get(), Docs.Get(), Docs.Size(), QueryOffsets.Get(), QueryOffsetsBias, QuerySizes.Get(), DocwiseSampleRate, MaxQuerySize, TakenMask.Get(), stream.GetStream());
        }
    };

    class TRemoveQueryMeans: public TKernelBase<NKernel::TRemoveQueryBiasContext, false> {
    private:
        TCudaBufferPtr<const ui32> Qids;
        TCudaBufferPtr<const ui32> QidsOffsets;
        TCudaBufferPtr<float> Dest;

    public:
        using TKernelContext = NKernel::TRemoveQueryBiasContext;

        Y_SAVELOAD_DEFINE(Qids, QidsOffsets, Dest);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->QueryBias = memoryManager.Allocate<float>(QidsOffsets.Size());
            return context;
        }

        TRemoveQueryMeans() = default;

        TRemoveQueryMeans(const TCudaBufferPtr<const ui32> qids,
                          const TCudaBufferPtr<const ui32> qidOffsets,
                          TCudaBufferPtr<float> dest)
            : Qids(qids)
            , QidsOffsets(qidOffsets)
            , Dest(dest)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            CB_ENSURE(QidsOffsets.Size());
            const ui32 qCount = QidsOffsets.Size() - 1;
            NKernel::ComputeGroupMeans(Dest.Get(), nullptr, QidsOffsets.Get(), qCount, context.QueryBias, stream.GetStream());
            NKernel::RemoveGroupBias(context.QueryBias, Qids.Get(), Dest.Size(), Dest.Get(), stream.GetStream());
        }
    };

    class TRemoveQueryMax: public TKernelBase<NKernel::TRemoveQueryBiasContext, false> {
    private:
        TCudaBufferPtr<const ui32> Qids;
        TCudaBufferPtr<const ui32> QidsOffsets;
        TCudaBufferPtr<float> Dest;

    public:
        using TKernelContext = NKernel::TRemoveQueryBiasContext;

        Y_SAVELOAD_DEFINE(Qids, QidsOffsets, Dest);

        THolder<TKernelContext> PrepareContext(IMemoryManager& memoryManager) const {
            auto context = MakeHolder<TKernelContext>();
            context->QueryBias = memoryManager.Allocate<float>(QidsOffsets.Size());
            return context;
        }

        TRemoveQueryMax() = default;

        TRemoveQueryMax(const TCudaBufferPtr<const ui32> qids,
                        const TCudaBufferPtr<const ui32> qidOffsets,
                        TCudaBufferPtr<float> dest)
            : Qids(qids)
            , QidsOffsets(qidOffsets)
            , Dest(dest)
        {
        }

        void Run(const TCudaStream& stream, TKernelContext& context) {
            CB_ENSURE(QidsOffsets.Size());
            const ui32 qCount = QidsOffsets.Size() - 1;
            NKernel::ComputeGroupMax(Dest.Get(), QidsOffsets.Get(), qCount, context.QueryBias, stream.GetStream());
            NKernel::RemoveGroupBias(context.QueryBias, Qids.Get(), Dest.Size(), Dest.Get(), stream.GetStream());
        }
    };

}

template <class TFloat, class TMapping>
inline void ComputeBordersOnDevice(const TCudaBuffer<TFloat, TMapping>& feature,
                                   const NCatboostOptions::TBinarizationOptions& description,
                                   TCudaBuffer<float, TMapping>& dst,
                                   ui32 stream = 0) {
    LaunchKernels<NKernelHost::TFindBordersKernel>(feature.NonEmptyDevices(), stream, feature, description.BorderSelectionType, description.BorderCount, dst);
}

template <class TValuesFloatType, class TBordersFloatType, class TUi32, class TMapping>
inline void BinarizeOnDevice(const TCudaBuffer<TValuesFloatType, TMapping>& featureValues,
                             const TCudaBuffer<TBordersFloatType, TMapping>& borders,
                             const NCudaLib::TDistributedObject<TCFeature>& feature,
                             TCudaBuffer<ui32, TMapping>& dst,
                             bool atomicUpdate,
                             const TCudaBuffer<TUi32, TMapping>* gatherIndices,
                             ui32 stream = 0) {
    using TKernel = NKernelHost::TBinarizeFloatFeatureKernel;
    LaunchKernels<TKernel>(featureValues.NonEmptyDevices(), stream, featureValues, borders, feature, gatherIndices, dst, atomicUpdate);
};

template <class TUi32,
          class TBinsBuffer>
inline void WriteCompressedFeature(const NCudaLib::TDistributedObject<TCFeature>& feature,
                                   const TBinsBuffer& bins,
                                   TStripeBuffer<TUi32>& cindex,
                                   ui32 stream = 0) {
    using TKernel = NKernelHost::TWriteCompressedIndexKernel<TBinsBuffer::PtrType()>;
    LaunchKernels<TKernel>(bins.NonEmptyDevices(), stream, bins, feature, cindex);
};

inline void WriteLazyCompressedFeature(
    const NCudaLib::TDistributedObject<TCFeature>& feature,
    const NCudaLib::TStripeMapping& docMapping,
    const NCB::TPathWithScheme& pathWithScheme,
    ui32 datasetFeatureId,
    ui32 featureId,
    TMaybe<ui16> baseValue,
    TStripeBuffer<ui32>& cindex,
    ui32 stream = 0
) {
    using TKernel = NKernelHost::TWriteLazyCompressedIndexKernel;

    auto& cudaManager = NCudaLib::GetCudaManager();
    auto deviceSlices = CreateDistributedObject<TSlice>();
    for (auto deviceIdx : xrange(cudaManager.GetDeviceCount())) {
        const auto deviceSlice = docMapping.DeviceSlice(deviceIdx);
        deviceSlices.Set(deviceIdx, deviceSlice);
    }

    LaunchKernels<TKernel>(
        docMapping.NonEmptyDevices(),
        stream,
        pathWithScheme,
        datasetFeatureId,
        deviceSlices,
        docMapping.SingleObjectSize(),
        featureId,
        feature,
        baseValue,
        cindex);
}

inline void DropAllLoaders(const NCudaLib::TDevicesList& deviceList, ui32 stream = 0) {
    using TKernel = NKernelHost::TDropAllLoaders;

    auto deviceListCopy = deviceList;
    LaunchKernels<TKernel>(std::move(deviceListCopy), stream);
}

template <class TUi32, class TMapping, class TQueryOffsetsBias>
inline void ComputeQueryIds(const TCudaBuffer<TUi32, TMapping>& querySizes,
                            const TCudaBuffer<TUi32, TMapping>& queryOffsets,
                            TQueryOffsetsBias bias,
                            TCudaBuffer<ui32, TMapping>* qids,
                            ui32 stream = 0) {
    using TKernel = NKernelHost::TComputeQueryIdsKernel;
    LaunchKernels<TKernel>(querySizes.NonEmptyDevices(), stream, querySizes, queryOffsets, bias, *qids);
};

template <class TUi32, class TMapping>
inline void FillQueryEndMasks(const TCudaBuffer<TUi32, TMapping>& qids,
                              const TCudaBuffer<TUi32, TMapping>& docs,
                              TCudaBuffer<ui32, TMapping>* masks,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TFillQueryEndMaskKernel;
    LaunchKernels<TKernel>(qids.NonEmptyDevices(), stream, qids, docs, *masks);
};

template <class TMapping>
inline void CreateShuffleKeys(TCudaBuffer<ui64, TMapping>& seeds,
                              const TCudaBuffer<ui32, TMapping>& docQids,
                              TCudaBuffer<ui64, TMapping>* keys,
                              ui32 stream = 0) {
    using TKernel = NKernelHost::TCreateKeysForSegmentedDocsSampleKernel;
    LaunchKernels<TKernel>(docQids.NonEmptyDevices(), stream, seeds, docQids, *keys);
};

template <class TMapping, class TQueryOffsetsBias>
inline void CreateTakenDocsMask(const TCudaBuffer<float, TMapping>& takenQueriesMask,
                                const TCudaBuffer<ui32, TMapping>& docQids,
                                const TCudaBuffer<ui32, TMapping>& perQueryShuffledDocs,
                                const TCudaBuffer<const ui32, TMapping>& queryOffsets,
                                TQueryOffsetsBias queryOffsetsBias,
                                const TCudaBuffer<const ui32, TMapping>& querySizes,
                                float docSampleRate,
                                ui32 maxQuerySize,
                                TCudaBuffer<float, TMapping>* sampledWeights,
                                ui32 stream = 0) {
    using TKernel = NKernelHost::TFillTakenDocsMaskKernel;
    LaunchKernels<TKernel>(docQids.NonEmptyDevices(), stream, takenQueriesMask, docQids, perQueryShuffledDocs,
                           queryOffsets, queryOffsetsBias, querySizes,
                           docSampleRate, maxQuerySize, *sampledWeights);
};

template <class TMapping>
inline void RemoveQueryMeans(const TCudaBuffer<ui32, TMapping>& qids,
                             const TCudaBuffer<ui32, TMapping>& qidOffsets,
                             TCudaBuffer<float, TMapping>* vec,
                             ui32 stream = 0) {
    using TKernel = NKernelHost::TRemoveQueryMeans;
    LaunchKernels<TKernel>(vec->NonEmptyDevices(), stream, qids, qidOffsets, *vec);
};

template <class TMapping>
inline void RemoveQueryMax(const TCudaBuffer<ui32, TMapping>& qids,
                           const TCudaBuffer<ui32, TMapping>& qidOffsets,
                           TCudaBuffer<float, TMapping>* vec,
                           ui32 stream = 0) {
    using TKernel = NKernelHost::TRemoveQueryMax;
    LaunchKernels<TKernel>(vec->NonEmptyDevices(), stream, qids, qidOffsets, *vec);
};
