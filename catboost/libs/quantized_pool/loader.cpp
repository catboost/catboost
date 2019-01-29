#include "pool.h"
#include "quantized.h"
#include "serialization.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data_new/loader.h>
#include <catboost/libs/data_new/meta_info.h>
#include <catboost/libs/data_new/unaligned_mem.h>
#include <catboost/libs/data_util/exists_checker.h>
#include <catboost/libs/data_util/path_with_scheme.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/libs/quantization_schema/serialization.h>

#include <util/generic/cast.h>
#include <util/generic/deque.h>
#include <util/generic/mapfindptr.h>
#include <util/generic/scope.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/system/madvise.h>
#include <util/system/types.h>
#include <util/system/unaligned_mem.h>

using NCB::EObjectsOrder;
using NCB::IQuantizedFeaturesDataVisitor;
using NCB::IQuantizedFeaturesDatasetLoader;
using NCB::QuantizationSchemaFromProto;
using NCB::TDataMetaInfo;
using NCB::TDatasetLoaderFactory;
using NCB::TDatasetLoaderPullArgs;
using NCB::TExistsCheckerFactory;
using NCB::TFSExistsChecker;
using NCB::TLoadQuantizedPoolParameters;
using NCB::TMaybeOwningConstArrayHolder;
using NCB::TPathWithScheme;
using NCB::TQuantizedPool;
using NCB::TUnalignedArrayBuf;

namespace {
    class TCBQuantizedDataLoader : public IQuantizedFeaturesDatasetLoader {
    public:
        explicit TCBQuantizedDataLoader(TDatasetLoaderPullArgs&& args);

        void Do(IQuantizedFeaturesDataVisitor* visitor) override;

    private:
        void AddChunk(
            const TQuantizedPool::TChunkDescription& chunk,
            EColumn columnType,
            const size_t* flatFeatureIdx,
            const size_t* baselineIdx,
            IQuantizedFeaturesDataVisitor* visitor) const;

        void AddQuantizedFeatureChunk(
            const TQuantizedPool::TChunkDescription& chunk,
            const size_t flatFeatureIdx,
            IQuantizedFeaturesDataVisitor* visitor) const;

        static TLoadQuantizedPoolParameters GetLoadParameters() {
            return {/*LockMemory*/ false, /*Precharge*/ false};
        }

    private:
        ui32 ObjectCount;
        TVector<bool> IsFeatureIgnored;
        TQuantizedPool QuantizedPool;
        TPathWithScheme PairsPath;
        TPathWithScheme GroupWeightsPath;
        TDataMetaInfo DataMetaInfo;
        EObjectsOrder ObjectsOrder;
    };
}

static TDataMetaInfo MaybeAddArtificialFeatures(
    const TDataMetaInfo& src,
    const NCB::TPoolQuantizationSchema& schema)
{
    TDataMetaInfo metaInfo = src;
    metaInfo.FeaturesLayout = MakeIntrusive<NCB::TFeaturesLayout>(
        src.FeaturesLayout->GetExternalFeatureCount(),
        TVector<ui32>(
            src.FeaturesLayout->GetCatFeatureInternalIdxToExternalIdx().begin(),
            src.FeaturesLayout->GetCatFeatureInternalIdxToExternalIdx().end()),
        src.FeaturesLayout->GetExternalFeatureIds(),
        &schema);

    const auto srcFeaturesMetaInfo = src.FeaturesLayout->GetExternalFeaturesMetaInfo();
    for (size_t i = 0; i < srcFeaturesMetaInfo.size(); ++i) {
        if (srcFeaturesMetaInfo[i].IsIgnored) {
            metaInfo.FeaturesLayout->IgnoreExternalFeature(i);
        }
    }

    // TODO(yazevnul): do we need to modify `res.ColumnsInfo`? It has almoust a year old TODO
    // in it (see r3649186) and technically we are not working with DSV pool we are working
    // with quantized pool.

    return metaInfo;
}

TCBQuantizedDataLoader::TCBQuantizedDataLoader(TDatasetLoaderPullArgs&& args)
    : ObjectCount(0) // inited later
    , QuantizedPool(std::forward<TQuantizedPool>(LoadQuantizedPool(args.PoolPath.Path, GetLoadParameters())))
    , PairsPath(args.CommonArgs.PairsFilePath)
    , GroupWeightsPath(args.CommonArgs.GroupWeightsFilePath)
    , ObjectsOrder(args.CommonArgs.ObjectsOrder)
{
    CB_ENSURE(QuantizedPool.DocumentCount > 0, "Pool is empty");
    CB_ENSURE(
        QuantizedPool.DocumentCount <= (size_t)Max<ui32>(),
        "CatBoost does not support datasets with more than " << Max<ui32>() << " objects"
    );
    // validity of cast checked above
    ObjectCount = (ui32)QuantizedPool.DocumentCount;

    CB_ENSURE(!PairsPath.Inited() || CheckExists(PairsPath),
        "TCBQuantizedDataLoader:PairsFilePath does not exist");
    CB_ENSURE(!GroupWeightsPath.Inited() || CheckExists(GroupWeightsPath),
        "TCBQuantizedDataLoader:GroupWeightsFilePath does not exist");

    DataMetaInfo = GetDataMetaInfo(QuantizedPool, GroupWeightsPath.Inited(), PairsPath.Inited());

    CB_ENSURE(DataMetaInfo.GetFeatureCount() > 0, "Pool should have at least one factor");

    TVector<ui32> allIgnoredFeatures = args.CommonArgs.IgnoredFeatures;
    TVector<ui32> ignoredFeaturesFromPool = GetIgnoredFlatIndices(QuantizedPool);
    allIgnoredFeatures.insert(
        allIgnoredFeatures.end(),
        ignoredFeaturesFromPool.begin(),
        ignoredFeaturesFromPool.end()
    );

    ProcessIgnoredFeaturesList(allIgnoredFeatures, &DataMetaInfo, &IsFeatureIgnored);

    DataMetaInfo = MaybeAddArtificialFeatures(
        DataMetaInfo,
        QuantizationSchemaFromProto(QuantizedPool.QuantizationSchema));
}

namespace {
    struct TChunkRef {
        const TQuantizedPool::TChunkDescription* Description = nullptr;
        ui32 ColumnIndex = 0;
        ui32 LocalIndex = 0;
    };

    class TSequantialChunkEvictor {
    public:
        explicit TSequantialChunkEvictor(ui64 minSizeInBytesToEvict);

        void Push(const TChunkRef& chunk);
        void MaybeEvict(bool force = false) noexcept;

    private:
        size_t MinSizeInBytesToEvict_ = 0;
        bool Evicted_ = false;
        const ui8* Data_ = nullptr;
        size_t Size_ = 0;
    };
}

TSequantialChunkEvictor::TSequantialChunkEvictor(const ui64 minSizeInBytesToEvict)
    : MinSizeInBytesToEvict_(minSizeInBytesToEvict) {
}

void TSequantialChunkEvictor::Push(const TChunkRef& chunk) {
    Y_DEFER { Evicted_ = false; };

    const auto* const data = reinterpret_cast<const ui8*>(chunk.Description->Chunk->Quants()->data());
    const size_t size = chunk.Description->Chunk->Quants()->size();
    CB_ENSURE(
        Data_ + Size_ <= data,
        LabeledOutput(static_cast<const void*>(Data_), Size_, static_cast<const void*>(data), size));

    if (!Data_) {
        Data_ = data;
        Size_ = size;
    } else if (Evicted_) {
        const auto* const nextData = Data_ + Size_;
        Size_ = data - nextData + size;
        Data_ = nextData;
    } else {
        Size_ = data - Data_ + size;
    }
}

void TSequantialChunkEvictor::MaybeEvict(const bool force) noexcept {
    if (Evicted_ || !force && Size_ < MinSizeInBytesToEvict_) {
        return;
    }

    try {
#if !defined(_win_)
        // TODO(akhropov): fix MadviseEvict on Windows: MLTOOLS-2440
        MadviseEvict(Data_, Size_);
#endif
    } catch (const std::exception& e) {
        CATBOOST_DEBUG_LOG
            << "MadviseEvict(Data_, Size_) with "
            << LabeledOutput(static_cast<const void*>(Data_), Size_)
            << "failed with error: " << e.what() << Endl;
    }

    Evicted_ = true;
}

static TDeque<TChunkRef> GatherAndSortChunks(const TQuantizedPool& pool) {
    TDeque<TChunkRef> chunks;
    for (const auto [columnIdx, localIdx] : pool.ColumnIndexToLocalIndex) {
        for (const auto& description : pool.Chunks[localIdx]) {
            chunks.push_back({&description, static_cast<ui32>(columnIdx), static_cast<ui32>(localIdx)});
        }
    }

    const ui32 fakeIndices[] = {
        pool.StringDocIdLocalIndex,
        pool.StringGroupIdLocalIndex,
        pool.StringSubgroupIdLocalIndex};
    for (const auto fakeIdx : fakeIndices) {
        if (fakeIdx == static_cast<ui32>(-1)) {
            continue;
        }

        for (const auto& description : pool.Chunks[fakeIdx]) {
            chunks.push_back({&description, 0, fakeIdx});
        }
    }

    // Sort chunks in ascending order based on the address of the chunks. We'll use it later to
    // process chunks in the same order as if we were reading them from file one by one
    // sequentially.
    Sort(chunks, [](const auto lhs, const auto rhs) {
        return lhs.Description->Chunk->Quants()->data() < rhs.Description->Chunk->Quants()->data();
    });

    return chunks;
}

template <typename T, typename U>
static void AssignUnaligned(const TConstArrayRef<ui8> unaligned, TVector<U>* dst) {
    dst->yresize(unaligned.size() / sizeof(T));
    TUnalignedMemoryIterator<T> it(unaligned.data(), unaligned.size());
    for (auto& v : *dst) {
        v = it.Next();
    }
}

template <typename T>
static void FillArtificialFeatures(
    const TConstArrayRef<ui8> quants,
    const NCB::TQuantizedFeatureIndexRemapper remapper,
    TVector<TVector<ui8>>* const artificialFeatures)
{
    TUnalignedMemoryIterator<T> it(quants.data(), quants.size());
    for (size_t i = 0, iEnd = quants.size() / sizeof(T); i < iEnd; ++i, (void)it.Next()) {
        const auto originalBinIdx = it.Cur();
        const size_t artificialFeatureIdx = originalBinIdx / remapper.GetBinsPerArtificialFeature();
        const auto artificialBinIdx = static_cast<ui8>(originalBinIdx % remapper.GetBinsPerArtificialFeature());
        for (size_t j = 0, jEnd = artificialFeatures->size(); j < jEnd; ++j) {
            (*artificialFeatures)[j][i] = i == artificialFeatureIdx ? artificialBinIdx : 255;
        }
    }
}

void TCBQuantizedDataLoader::AddQuantizedFeatureChunk(
    const TQuantizedPool::TChunkDescription& chunk,
    const size_t flatFeatureIdx,
    IQuantizedFeaturesDataVisitor* const visitor) const
{
    const auto quants = MakeArrayRef(
        reinterpret_cast<const ui8*>(chunk.Chunk->Quants()->data()),
        chunk.Chunk->Quants()->size());

    const auto remapper = DataMetaInfo.FeaturesLayout->GetQuantizedFeatureIndexRemapper(flatFeatureIdx);
    if (!remapper.HasArtificialFeatures()) {
        visitor->AddFloatFeaturePart(
            flatFeatureIdx,
            chunk.DocumentOffset,
            TMaybeOwningConstArrayHolder<ui8>::CreateNonOwning(quants));
        return;
    }

    const auto documentCount = chunk.Chunk->Quants()->size()
        / static_cast<size_t>(chunk.Chunk->BitsPerDocument() / CHAR_BIT) ;
    TVector<TVector<ui8>> artificialFeatures(remapper.GetArtificialFeatureCount() + 1);
    for (auto& artificialFeature : artificialFeatures) {
        artificialFeature.yresize(documentCount);
    }

    switch (const auto bitsPerDocument = chunk.Chunk->BitsPerDocument()) {
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_8:
            ythrow TCatBoostException() << "should not be here (there must be no artificial features)";
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_16:
            FillArtificialFeatures<ui16>(quants, remapper, &artificialFeatures);
            break;
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_32:
            FillArtificialFeatures<ui32>(quants, remapper, &artificialFeatures);
            break;
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_64:
            FillArtificialFeatures<ui64>(quants, remapper, &artificialFeatures);
            break;
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_1:
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_2:
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_4:
            ythrow TCatBoostException() << LabeledOutput((int)bitsPerDocument) << " is not supported";
        case NCB::NIdl::EBitsPerDocumentFeature_BPDF_UKNOWN:
            ythrow TCatBoostException() << "got invalid value for bits perDocument";
    }

    visitor->AddFloatFeaturePart(
        flatFeatureIdx,
        chunk.DocumentOffset,
        TMaybeOwningConstArrayHolder<ui8>::CreateNonOwning(MakeArrayRef(artificialFeatures[0])));

    for (ui32 i = 0, iEnd = remapper.GetArtificialFeatureCount(); i < iEnd; ++i) {
        const auto artificialFeatureIdx = remapper.GetIdxOffset() + i;
        visitor->AddFloatFeaturePart(
            artificialFeatureIdx,
            chunk.DocumentOffset,
            TMaybeOwningConstArrayHolder<ui8>::CreateNonOwning(MakeArrayRef(artificialFeatures[i + 1])));
    }
}

void TCBQuantizedDataLoader::AddChunk(
    const TQuantizedPool::TChunkDescription& chunk,
    const EColumn columnType,
    const size_t* const flatFeatureIdx,
    const size_t* const baselineIdx,
    IQuantizedFeaturesDataVisitor* const visitor) const
{
    const auto quants = MakeArrayRef(
        reinterpret_cast<const ui8*>(chunk.Chunk->Quants()->data()),
        chunk.Chunk->Quants()->size());

    switch (columnType) {
        case EColumn::Num: {
            AddQuantizedFeatureChunk(chunk, *flatFeatureIdx, visitor);
            break;
        } case EColumn::Label: {
            // TODO(akhropov): will be raw strings as was decided for new data formats for MLTOOLS-140.
            visitor->AddTargetPart(chunk.DocumentOffset, TUnalignedArrayBuf<float>(quants));
            break;
        } case EColumn::Baseline: {
            // TODO(akhropov): switch to storing floats - MLTOOLS-2394
            TVector<float> tmp;
            AssignUnaligned<double>(quants, &tmp);
            visitor->AddBaselinePart(chunk.DocumentOffset, *baselineIdx, TUnalignedArrayBuf<float>(tmp.data(), tmp.size() * sizeof(float)));
            break;
        } case EColumn::Weight: {
            visitor->AddWeightPart(chunk.DocumentOffset, TUnalignedArrayBuf<float>(quants));
            break;
        } case EColumn::GroupWeight: {
            visitor->AddGroupWeightPart(chunk.DocumentOffset, TUnalignedArrayBuf<float>(quants));
            break;
        } case EColumn::GroupId: {
            visitor->AddGroupIdPart(chunk.DocumentOffset, TUnalignedArrayBuf<ui64>(quants));
            break;
        } case EColumn::SubgroupId: {
            visitor->AddSubgroupIdPart(chunk.DocumentOffset, TUnalignedArrayBuf<ui32>(quants));
            break;
        } case EColumn::DocId:
            // Are skipped in a caller
        case EColumn::Categ:
            // TODO(yazevnul): categorical feature quantization on YT is still in progress
        case EColumn::Auxiliary:
            // Should not be present in quantized pool
        case EColumn::Timestamp:
            // Not supported by quantized pools right now
        case EColumn::Sparse:
            // Not supported by CatBoost at all
        case EColumn::Prediction: {
            // Can't be present in quantized pool
            ythrow TCatBoostException() << "Unexpected column type " << columnType;
        }
    }
}

void TCBQuantizedDataLoader::Do(IQuantizedFeaturesDataVisitor* visitor) {
    visitor->Start(
        DataMetaInfo,
        ObjectCount,
        ObjectsOrder,
        {},
        QuantizationSchemaFromProto(QuantizedPool.QuantizationSchema));

    const auto columnIdxToFlatIdx = GetColumnIndexToFlatIndexMap(QuantizedPool);
    const auto columnIdxToBaselineIdx = GetColumnIndexToBaselineIndexMap(QuantizedPool);
    const auto chunkRefs = GatherAndSortChunks(QuantizedPool);

    TSequantialChunkEvictor evictor(1ULL << 24);
    for (const auto chunkRef : chunkRefs) {
        evictor.Push(chunkRef);
        Y_DEFER { evictor.MaybeEvict(); };

        const auto columnIdx = chunkRef.ColumnIndex;
        const auto localIdx = chunkRef.LocalIndex;
        const auto isStringColumn = QuantizedPool.HasStringColumns &&
            (localIdx == QuantizedPool.StringDocIdLocalIndex ||
             localIdx == QuantizedPool.StringGroupIdLocalIndex ||
             localIdx == QuantizedPool.StringSubgroupIdLocalIndex);
        if (isStringColumn) {
            // Ignore string columns, they are only needed for fancy output for evaluation.
            continue;
        }

        const auto columnType = QuantizedPool.ColumnTypes[localIdx];
        if (columnType == EColumn::DocId) {
            // Skip DocId columns presented in old pools.
            continue;
        }

        CB_ENSURE(
            columnType == EColumn::Num || columnType == EColumn::Baseline ||
            columnType == EColumn::Label || columnType == EColumn::Categ ||
            columnType == EColumn::Weight || columnType == EColumn::GroupWeight ||
            columnType == EColumn::GroupId || columnType == EColumn::SubgroupId,
            "Expected Num, Baseline, Label, Categ, Weight, GroupWeight, GroupId, or Subgroupid; got "
            LabeledOutput(columnType, columnIdx));

        const auto* const flatFeatureIdx = columnIdxToFlatIdx.FindPtr(columnIdx);
        if (flatFeatureIdx && IsFeatureIgnored[*flatFeatureIdx]) {
            continue;
        }

        const auto* const baselineIdx = columnIdxToBaselineIdx.FindPtr(columnIdx);
        AddChunk(*chunkRef.Description, columnType, flatFeatureIdx, baselineIdx, visitor);
    }

    evictor.MaybeEvict(true);

    QuantizedPool = TQuantizedPool(); // release memory
    SetGroupWeights(GroupWeightsPath, ObjectCount, visitor);
    SetPairs(PairsPath, ObjectCount, visitor);
    visitor->Finish();
}

namespace {
    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSQuantizedExistsCheckerReg("quantized");
    TDatasetLoaderFactory::TRegistrator<TCBQuantizedDataLoader> CBQuantizedDataLoaderReg("quantized");
}

