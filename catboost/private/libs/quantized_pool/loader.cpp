#include "loader.h"
#include "quantized.h"

#include <catboost/idl/pool/flat/quantized_chunk_t.fbs.h>
#include <catboost/libs/column_description/column.h>
#include <catboost/libs/data/baseline.h>
#include <catboost/libs/data/meta_info.h>
#include <catboost/libs/data/unaligned_mem.h>
#include <catboost/private/libs/data_util/exists_checker.h>
#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/labels/helpers.h>
#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/private/libs/quantization_schema/serialization.h>

#include <util/generic/algorithm.h>
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

NCB::TCBQuantizedDataLoader::TCBQuantizedDataLoader(TDatasetLoaderPullArgs&& args)
    : ObjectCount(0) // inited later
    , QuantizedPool(
        std::forward<TQuantizedPool>(
            LoadQuantizedPool(args.PoolPath, GetLoadParameters(args.CommonArgs.DatasetSubset))
        )
      )
    , PairsPath(args.CommonArgs.PairsFilePath)
    , GraphPath(args.CommonArgs.GraphFilePath)
    , GroupWeightsPath(args.CommonArgs.GroupWeightsFilePath)
    , BaselinePath(args.CommonArgs.BaselineFilePath)
    , TimestampsPath(args.CommonArgs.TimestampsFilePath)
    , PoolMetaInfoPath(args.CommonArgs.PoolMetaInfoPath)
    , ObjectsOrder(args.CommonArgs.ObjectsOrder)
    , DatasetSubset(args.CommonArgs.DatasetSubset)
{
    CB_ENSURE(QuantizedPool.DocumentCount > 0, "Pool is empty");
    CB_ENSURE(
        QuantizedPool.DocumentCount <= (size_t)Max<ui32>(),
        "CatBoost does not support datasets with more than " << Max<ui32>() << " objects"
    );
    // validity of cast checked above
    ObjectCount = Min<ui32>(QuantizedPool.DocumentCount, DatasetSubset.Range.End) - DatasetSubset.Range.Begin;

    CB_ENSURE(
        !PairsPath.Inited() || CheckExists(PairsPath),
        "TCBQuantizedDataLoader:PairsFilePath does not exist");
    CB_ENSURE(
        !GraphPath.Inited() || CheckExists(GraphPath),
        "TCBQuantizedDataLoader:GraphFilePath does not supported");
    CB_ENSURE(
        !GroupWeightsPath.Inited() || CheckExists(GroupWeightsPath),
        "TCBQuantizedDataLoader:GroupWeightsFilePath does not exist");
    CB_ENSURE(
        !TimestampsPath.Inited() || CheckExists(TimestampsPath),
        "TCBQuantizedDataLoader:TimestampsPath does not exist");
    CB_ENSURE(
        !FeatureNamesPath.Inited() || CheckExists(FeatureNamesPath),
        "TCBQuantizedDataLoader:FeatureNamesPath does not exist");
    CB_ENSURE(
        !PoolMetaInfoPath.Inited() || CheckExists(PoolMetaInfoPath),
        "TCBQuantizedDataLoader:PoolMetaInfoPath does not exist");

    THolder<NCB::IBaselineReader> baselineReader;
    if (BaselinePath.Inited()) {
        CB_ENSURE(
            CheckExists(BaselinePath),
            "TCBQuantizedDataLoader:BaselineFilePath does not exist");

        baselineReader = GetProcessor<NCB::IBaselineReader, NCB::TBaselineReaderArgs>(
            BaselinePath,
            NCB::TBaselineReaderArgs{
                BaselinePath,
                ClassLabelsToStrings(args.CommonArgs.ClassLabels),
                DatasetSubset.Range
            }
        );
    }

    DataMetaInfo = GetDataMetaInfo(
        QuantizedPool,
        GroupWeightsPath.Inited(),
        TimestampsPath.Inited(),
        PairsPath.Inited(),
        args.CommonArgs.ForceUnitAutoPairWeights,
        baselineReader ? TMaybe<ui32>(baselineReader->GetBaselineCount()) : Nothing(),
        args.CommonArgs.FeatureNamesPath,
        PoolMetaInfoPath);

    CB_ENSURE(DataMetaInfo.GetFeatureCount() > 0, "Pool should have at least one factor");

    TVector<ui32> allIgnoredFeatures = args.CommonArgs.IgnoredFeatures;
    CATBOOST_DEBUG_LOG << "allIgnoredFeatures.size() " << allIgnoredFeatures.size() << Endl;
    TVector<ui32> ignoredFeaturesFromPool = GetIgnoredFlatIndices(QuantizedPool);
    CATBOOST_DEBUG_LOG << "ignoredFeaturesFromPool.size() " << ignoredFeaturesFromPool.size() << Endl;
    allIgnoredFeatures.insert(
        allIgnoredFeatures.end(),
        ignoredFeaturesFromPool.begin(),
        ignoredFeaturesFromPool.end()
    );

    CATBOOST_DEBUG_LOG << "allIgnoredFeatures.size() " << allIgnoredFeatures.size() << Endl;
    ProcessIgnoredFeaturesList(
        allIgnoredFeatures,
        /*allFeaturesIgnoredMessage*/ "All features are either constant or ignored",
        &DataMetaInfo,
        &IsFeatureIgnored);
}

namespace {
    struct TChunkRef {
        const TQuantizedPool::TChunkDescription* Description = nullptr;
        ui32 ColumnIndex = 0;
        ui32 LocalIndex = 0;
    };

    class TSequentialChunkEvictor {
    public:
        explicit TSequentialChunkEvictor(ui64 minSizeInBytesToEvict);

        void Push(const TChunkRef& chunk);
        void MaybeEvict(bool force = false) noexcept;

    private:
        size_t MinSizeInBytesToEvict_ = 0;
        bool Evicted_ = false;
        const ui8* Data_ = nullptr;
        size_t Size_ = 0;
    };
}

TSequentialChunkEvictor::TSequentialChunkEvictor(const ui64 minSizeInBytesToEvict)
    : MinSizeInBytesToEvict_(minSizeInBytesToEvict) {
}

void TSequentialChunkEvictor::Push(const TChunkRef& chunk) {
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

void TSequentialChunkEvictor::MaybeEvict(const bool force) noexcept {
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
        pool.StringSubgroupIdLocalIndex
    };
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
    Sort(
        chunks,
        [](const auto lhs, const auto rhs) {
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

void NCB::TCBQuantizedDataLoader::AddQuantizedFeatureChunk(
    const TQuantizedPool::TChunkDescription& chunk,
    const size_t flatFeatureIdx,
    IQuantizedFeaturesDataVisitor* const visitor) const
{
    const auto quants = ClipByDatasetSubset(chunk);

    if (quants.empty()) {
        return;
    }

    visitor->AddFloatFeaturePart(
        flatFeatureIdx,
        GetDatasetOffset(chunk),
        chunk.Chunk->BitsPerDocument(),
        TMaybeOwningConstArrayHolder<ui8>::CreateNonOwning(quants));
}

 void NCB::TCBQuantizedDataLoader::AddQuantizedCatFeatureChunk(
    const TQuantizedPool::TChunkDescription& chunk,
    const size_t flatFeatureIdx,
    IQuantizedFeaturesDataVisitor* const visitor) const
{
    const auto quants = ClipByDatasetSubset(chunk);

    if (quants.empty()) {
        return;
    }

    visitor->AddCatFeaturePart(
        flatFeatureIdx,
        GetDatasetOffset(chunk),
        chunk.Chunk->BitsPerDocument(),
        TMaybeOwningConstArrayHolder<ui8>::CreateNonOwning(quants));
}

void NCB::TCBQuantizedDataLoader::AddChunk(
    const TQuantizedPool::TChunkDescription& chunk,
    const EColumn columnType,
    const size_t* const targetIdx,
    const size_t* const flatFeatureIdx,
    const size_t* const baselineIdx,
    IQuantizedFeaturesDataVisitor* const visitor) const
{
    const auto quants = ClipByDatasetSubset(chunk);

    if (quants.empty()) {
        return;
    }

    switch (columnType) {
        case EColumn::Num: {
            CB_ENSURE(flatFeatureIdx != nullptr, "Feature not found in index");
            AddQuantizedFeatureChunk(chunk, *flatFeatureIdx, visitor);
            break;
        } case EColumn::Label: {
            // TODO(akhropov): will be raw strings as was decided for new data formats for MLTOOLS-140.
            CB_ENSURE(targetIdx != nullptr, "Target not found in index");
            visitor->AddTargetPart(*targetIdx, GetDatasetOffset(chunk), TUnalignedArrayBuf<float>(quants));
            break;
        } case EColumn::Baseline: {
            // TODO(akhropov): switch to storing floats - MLTOOLS-2394
            CB_ENSURE(baselineIdx != nullptr, "Baseline not found in index");
            TVector<float> tmp;
            AssignUnaligned<double>(quants, &tmp);
            visitor->AddBaselinePart(
                GetDatasetOffset(chunk),
                *baselineIdx,
                TUnalignedArrayBuf<float>(tmp.data(), tmp.size() * sizeof(float)));
            break;
        } case EColumn::Weight: {
            visitor->AddWeightPart(GetDatasetOffset(chunk), TUnalignedArrayBuf<float>(quants));
            break;
        } case EColumn::GroupWeight: {
            visitor->AddGroupWeightPart(GetDatasetOffset(chunk), TUnalignedArrayBuf<float>(quants));
            break;
        } case EColumn::GroupId: {
            visitor->AddGroupIdPart(GetDatasetOffset(chunk), TUnalignedArrayBuf<ui64>(quants));
            break;
        } case EColumn::SubgroupId: {
            visitor->AddSubgroupIdPart(GetDatasetOffset(chunk), TUnalignedArrayBuf<ui32>(quants));
            break;
        } case EColumn::Timestamp: {
            visitor->AddTimestampPart(GetDatasetOffset(chunk), TUnalignedArrayBuf<ui64>(quants));
            break;
        } case EColumn::Categ:
          case EColumn::HashedCateg: {
            CB_ENSURE(flatFeatureIdx != nullptr, "Feature not found in index");
            AddQuantizedCatFeatureChunk(chunk, *flatFeatureIdx, visitor);
            break;
        }
        case EColumn::SampleId:
            // Are skipped in a caller
        case EColumn::Auxiliary:
        case EColumn::Text:
        case EColumn::NumVector:
            // Should not be present in quantized pool
        case EColumn::Sparse:
            // Not supported by CatBoost at all
        case EColumn::Prediction:
        case EColumn::Features: {
            // Can't be present in quantized pool
            ythrow TCatBoostException() << "Unexpected column type " << columnType;
        }
    }
}

TConstArrayRef<ui8> NCB::TCBQuantizedDataLoader::ClipByDatasetSubset(
    const TQuantizedPool::TChunkDescription& chunk) const
{
    const auto valueBytes = static_cast<size_t>(chunk.Chunk->BitsPerDocument() / CHAR_BIT);
    CB_ENSURE(valueBytes > 0, "Cannot read quantized pool with less than " << CHAR_BIT << " bits per value");
    const auto documentCount = chunk.Chunk->Quants()->size() / valueBytes;
    const auto chunkStart = chunk.DocumentOffset;
    const auto chunkEnd = chunkStart + documentCount;
    const auto loadStart = DatasetSubset.Range.Begin;
    const auto loadEnd = DatasetSubset.Range.End;
    if (loadStart <= chunkStart && chunkStart < loadEnd) {
        const auto* clippedStart = reinterpret_cast<const ui8*>(chunk.Chunk->Quants()->data());
        const auto clippedSize = Min<ui64>(chunkEnd - chunkStart, loadEnd - chunkStart) * valueBytes;
        return MakeArrayRef(clippedStart, clippedSize);
    } else if (chunkStart < loadStart && loadStart < chunkEnd) {
        const auto* clippedStart
            = reinterpret_cast<const ui8*>(chunk.Chunk->Quants()->data())
                + (loadStart - chunkStart) * valueBytes;
        const auto clippedSize = Min<ui64>(chunkEnd - loadStart, loadEnd - loadStart) * valueBytes;
        return MakeArrayRef(clippedStart, clippedSize);
    } else {
        CATBOOST_DEBUG_LOG << "All documents in chunk [" << chunkStart << ", " << chunkEnd << ") "
            "are outside load region [" << loadStart << ", " << loadEnd << ")" << Endl;
        return {};
    }
}

ui32 NCB::TCBQuantizedDataLoader::GetDatasetOffset(const TQuantizedPool::TChunkDescription& chunk) const {
    const auto documentCount = chunk.Chunk->Quants()->size()
         / static_cast<size_t>(chunk.Chunk->BitsPerDocument() / CHAR_BIT);
    const auto chunkStart = chunk.DocumentOffset;
    const auto chunkEnd = chunkStart + documentCount;
    const auto loadStart = DatasetSubset.Range.Begin;
    const auto loadEnd = DatasetSubset.Range.End;
    if (loadStart <= chunkStart && chunkStart < loadEnd) {
        return chunkStart - loadStart;
    } else if (chunkStart < loadStart && loadStart < chunkEnd) {
        return 0;
    } else {
        CB_ENSURE(
            false,
            "All documents in chunk [" << chunkStart << ", " << chunkEnd << ") are outside load region ["
            << loadStart << ", " << loadEnd << ")");
    }
}

void NCB::TCBQuantizedDataLoader::Do(IQuantizedFeaturesDataVisitor* visitor) {
    visitor->Start(
        DataMetaInfo,
        ObjectCount,
        ObjectsOrder,
        {},
        QuantizationSchemaFromProto(QuantizedPool.QuantizationSchema),
        /*wholeColumns*/ false);

    const auto columnIdxToTargetIdx = GetColumnIndexToTargetIndexMap(QuantizedPool);
    const auto columnIdxToFlatIdx = GetColumnIndexToFlatIndexMap(QuantizedPool);
    const auto columnIdxToBaselineIdx = GetColumnIndexToBaselineIndexMap(QuantizedPool);
    const auto chunkRefs = GatherAndSortChunks(QuantizedPool);

    TSequentialChunkEvictor evictor(1ULL << 24);
    CATBOOST_DEBUG_LOG << "Number of chunks to process " << chunkRefs.size() << Endl;
    for (const auto chunkRef : chunkRefs) {
        if (QuantizedPool.ChunkStorage.empty()) { // reading from mapped file
            evictor.Push(chunkRef);
        }
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
        if (columnType == EColumn::SampleId) {
            // Skip DocId columns presented in old pools.
            continue;
        }

        CB_ENSURE(
            EqualToOneOf(columnType, EColumn::Num, EColumn::Baseline,
            EColumn::Label, EColumn::Categ, EColumn::Weight,
            EColumn::GroupWeight, EColumn::GroupId, EColumn::SubgroupId,
            EColumn::Timestamp),
            "Expected Num, Baseline, Label, Categ, Weight, GroupWeight, GroupId, Subgroupid, or Timestamp; got "
            LabeledOutput(columnType, columnIdx));
        if (!DatasetSubset.HasFeatures) {
            CB_ENSURE(
                columnType != EColumn::Num && columnType != EColumn::Categ,
                "CollectChunks collected a feature chunk despite HasFeatures = false");
        }

        const auto* const flatFeatureIdx = columnIdxToFlatIdx.FindPtr(columnIdx);
        if (flatFeatureIdx && IsFeatureIgnored[*flatFeatureIdx]) {
            continue;
        }

        const auto* const baselineIdx = columnIdxToBaselineIdx.FindPtr(columnIdx);
        const auto* const targetIdx = columnIdxToTargetIdx.FindPtr(columnIdx);
        AddChunk(*chunkRef.Description, columnType, targetIdx, flatFeatureIdx, baselineIdx, visitor);
    }

    evictor.MaybeEvict(true);

    QuantizedPool = TQuantizedPool(); // release memory
    SetGroupWeights(GroupWeightsPath, ObjectCount, DatasetSubset, visitor);
    SetPairs(PairsPath, DatasetSubset, visitor);
    SetBaseline(
        BaselinePath,
        ObjectCount,
        DatasetSubset,
        NCB::ClassLabelsToStrings(DataMetaInfo.ClassLabels),
        visitor);
    SetTimestamps(TimestampsPath, ObjectCount, DatasetSubset, visitor);
    visitor->Finish();
}

namespace {
    TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSQuantizedExistsCheckerReg("quantized");
    TDatasetLoaderFactory::TRegistrator<NCB::TCBQuantizedDataLoader> CBQuantizedDataLoaderReg("quantized");
}

TAtomicSharedPtr<NCB::IQuantizedPoolLoader> NCB::TQuantizedPoolLoadersCache::GetLoader(
    const TPathWithScheme& pathWithScheme,
    TDatasetSubset loadSubset
) {
    auto& loadersCache = GetRef();
    TAtomicSharedPtr<IQuantizedPoolLoader> loader = nullptr;
    with_lock(loadersCache.Lock) {
        CATBOOST_DEBUG_LOG << __PRETTY_FUNCTION__ << ": loaders cache size " << loadersCache.Cache.size() << Endl;
        const auto loaderKey = std::make_pair(pathWithScheme, loadSubset);
        if (!loadersCache.Cache.contains(loaderKey)) {
            loader = GetProcessor<IQuantizedPoolLoader, const TPathWithScheme&>(
                pathWithScheme,
                pathWithScheme).Release();
            loadersCache.Cache[loaderKey] = loader;
            if (loadSubset.HasFeatures) {
                TLoadQuantizedPoolParameters params{/*LockMemory=*/false, /*Precharge=*/false, loadSubset};
                loader->LoadQuantizedPool(params);
            }
        }
        loader = loadersCache.Cache.at(loaderKey);
    }
    return loader;
}

bool NCB::TQuantizedPoolLoadersCache::HaveLoader(const TPathWithScheme& pathWithScheme, TDatasetSubset loadSubset) {
    auto& loadersCache = GetRef();
    with_lock(loadersCache.Lock) {
        CATBOOST_DEBUG_LOG << __PRETTY_FUNCTION__ << ": loaders cache size " << loadersCache.Cache.size() << Endl;
        return loadersCache.Cache.contains(std::make_pair(pathWithScheme, loadSubset));
    }
}

void NCB::TQuantizedPoolLoadersCache::DropAllLoaders() {
    auto& loadersCache = GetRef();
    with_lock(loadersCache.Lock) {
        CATBOOST_DEBUG_LOG << __PRETTY_FUNCTION__ << ": loaders cache size " << loadersCache.Cache.size() << Endl;
        for (auto& keyValue : loadersCache.Cache) {
            CB_ENSURE(
                keyValue.second.RefCount() <= 1,
                "Loader for " << keyValue.first.first.Scheme << "://" << keyValue.first.first.Path
                << " is still referenced");
        }
        loadersCache.Cache.clear();
    }
}
