
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
#include <catboost/libs/quantization_schema/serialization.h>

#include <util/generic/cast.h>
#include <util/generic/vector.h>
#include <util/generic/ylimits.h>
#include <util/system/madvise.h>
#include <util/system/types.h>


namespace NCB {

    class TCBQuantizedDataLoader final : public IQuantizedFeaturesDatasetLoader {
    public:
        explicit TCBQuantizedDataLoader(TDatasetLoaderPullArgs&& args);

        void Do(IQuantizedFeaturesDataVisitor* visitor) override;

    private:
        void AddColumn(
            const ui32 featureIndex,
            const ui32 baselineIndex,
            const EColumn columnType,
            const ui32 localIndex,
            IQuantizedFeaturesDataVisitor* visitor
        ) const;

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
    }

    void TCBQuantizedDataLoader::AddColumn(
        const ui32 featureIndex,
        const ui32 baselineIndex,
        const EColumn columnType,
        const ui32 localIndex,
        IQuantizedFeaturesDataVisitor* visitor
    ) const {
        auto onColumn = [&](size_t sizeOfElement, auto&& callbackFunction) {
            constexpr size_t MIN_QUANTS_SIZE_TO_FREE_INDIVIDUALLY = 1 << 20;

            const auto& chunks = QuantizedPool.Chunks[localIndex];
            for (const auto& descriptor : chunks) {
                CB_ENSURE(static_cast<size_t>(descriptor.Chunk->BitsPerDocument()) == sizeOfElement * 8);
                // cast is safe, checked at the start
                TConstArrayRef<ui8> quants = *descriptor.Chunk->Quants();
                callbackFunction((ui32)descriptor.DocumentOffset, quants);
#if !defined(_win_)
                // TODO(akhropov): fix MadviseEvict on Windows: MLTOOLS-2440

                // Free no longer needed memory by individual quants if they are big enough
                if (quants.size() > MIN_QUANTS_SIZE_TO_FREE_INDIVIDUALLY) {
                    MadviseEvict(quants.begin(), quants.size());
                }
#endif
            }

#if !defined(_win_)
            // TODO(akhropov): fix MadviseEvict on Windows: MLTOOLS-2440

            // Free no longer needed memory
            if (chunks.size() > 0) {
                auto begin = TConstArrayRef<ui8>(*(chunks.front().Chunk->Quants())).data();
                auto backQuant = TConstArrayRef<ui8>(*(chunks.back().Chunk->Quants()));
                auto end = backQuant.data() + backQuant.size();
                MadviseEvict(begin, end - begin);
            }
#endif
        };

        switch (columnType) {
            case EColumn::Num:
                onColumn(
                    sizeof(ui8),
                    [&] (ui32 objectOffset, TConstArrayRef<ui8> quants) {
                        visitor->AddFloatFeaturePart(
                            featureIndex,
                            objectOffset,
                            TMaybeOwningConstArrayHolder<ui8>::CreateNonOwning(quants)
                        );
                    }
                );
                break;
            case EColumn::Label:
                // TODO(akhropov): will be raw strings as was decided for new data formats for MLTOOLS-140.
                onColumn(
                    sizeof(float),
                    [&] (ui32 objectOffset, TConstArrayRef<ui8> quants) {
                        visitor->AddTargetPart(objectOffset, TUnalignedArrayBuf<float>(quants));
                    }
                );
                break;
            case EColumn::Baseline:
                // TODO(akhropov): switch to storing floats - MLTOOLS-2394
                onColumn(
                    sizeof(double),
                    [&] (ui32 objectOffset, TConstArrayRef<ui8> quants) {
                        TUnalignedArrayBuf<double> doubleData(quants);
                        TVector<float> floatData;
                        floatData.yresize(doubleData.GetSize());

                        TUnalignedMemoryIterator<double> doubleDataSrcIt = doubleData.GetIterator();
                        auto floatDataDstIt = floatData.begin();
                        for ( ; !doubleDataSrcIt.AtEnd(); doubleDataSrcIt.Next(), ++floatDataDstIt) {
                            *floatDataDstIt = (float)doubleDataSrcIt.Cur();
                        }
                        visitor->AddBaselinePart(
                            objectOffset,
                            baselineIndex,
                            TUnalignedArrayBuf<float>(floatData.data(), floatData.size()*sizeof(float))
                        );
                    }
                );
                break;
            case EColumn::Weight:
                onColumn(
                    sizeof(float),
                    [&] (ui32 objectOffset, TConstArrayRef<ui8> quants) {
                        visitor->AddWeightPart(objectOffset, TUnalignedArrayBuf<float>(quants));
                    }
                );
                break;
            case EColumn::GroupWeight: {
                onColumn(
                    sizeof(float),
                    [&] (ui32 objectOffset, TConstArrayRef<ui8> quants) {
                        visitor->AddGroupWeightPart(objectOffset, TUnalignedArrayBuf<float>(quants));
                    }
                );
                break;
            }
            case EColumn::DocId: {
                break;
            }
            case EColumn::GroupId: {
                onColumn(
                    sizeof(ui64),
                    [&] (ui32 objectOffset, TConstArrayRef<ui8> quants) {
                        visitor->AddGroupIdPart(objectOffset, TUnalignedArrayBuf<ui64>(quants));
                    }
                );
                break;
            }
            case EColumn::SubgroupId: {
                onColumn(
                    sizeof(ui32),
                    [&] (ui32 objectOffset, TConstArrayRef<ui8> quants) {
                        visitor->AddSubgroupIdPart(objectOffset, TUnalignedArrayBuf<ui32>(quants));
                    }
                );
                break;
            }
            case EColumn::Categ:
                // TODO(yazevnul): categorical feature quantization on YT is still in progress
            case EColumn::Auxiliary:
                // Should not be present in quantized pool
            case EColumn::Timestamp:
                // not supported by quantized pools right now
            case EColumn::Sparse:
                // not supperted by CatBoost at all
            case EColumn::Prediction: {
                // can't be present in quantized pool
                ythrow TCatboostException() << "Unexpected column type " << columnType;
            }
        }
    }

    void TCBQuantizedDataLoader::Do(IQuantizedFeaturesDataVisitor* visitor) {
        visitor->Start(
            DataMetaInfo,
            ObjectCount,
            ObjectsOrder,
            {},
            QuantizationSchemaFromProto(QuantizedPool.QuantizationSchema)
        );

        ui32 baselineIndex = 0;
        const auto columnIndexToFlatIndex = GetColumnIndexToFlatIndexMap(QuantizedPool);
        const auto columnIndexToNumericFeatureIndex = GetColumnIndexToNumericFeatureIndexMap(QuantizedPool);
        for (const auto [columnIndex, localIndex] : QuantizedPool.ColumnIndexToLocalIndex) {
            const auto columnType = QuantizedPool.ColumnTypes[localIndex];

            if (QuantizedPool.Chunks[localIndex].empty()) {
                continue;
            }
            // skip DocId columns presented in old pools
            if (columnType == EColumn::DocId) {
                continue;
            }
            CB_ENSURE(
                columnType == EColumn::Num || columnType == EColumn::Baseline ||
                columnType == EColumn::Label || columnType == EColumn::Categ ||
                columnType == EColumn::Weight || columnType == EColumn::GroupWeight ||
                columnType == EColumn::GroupId || columnType == EColumn::SubgroupId,
                "Expected Num, Baseline, Label, Categ, Weight, GroupWeight, GroupId, or Subgroupid; got "
                LabeledOutput(columnType, columnIndex));

            const auto it = columnIndexToNumericFeatureIndex.find(columnIndex);
            if (it == columnIndexToNumericFeatureIndex.end()) {
                AddColumn(
                    /*unused featureIndex*/ Max<ui32>(),
                    baselineIndex,
                    columnType,
                    SafeIntegerCast<ui32>(localIndex),
                    visitor
                );
                baselineIndex += (columnType == EColumn::Baseline);
            } else if (!IsFeatureIgnored[columnIndexToFlatIndex.at(columnIndex)]) {
                AddColumn(it->second, baselineIndex, columnType, localIndex, visitor);
            }
        }

        QuantizedPool = TQuantizedPool(); // release memory
        SetGroupWeights(GroupWeightsPath, ObjectCount, visitor);
        SetPairs(PairsPath, ObjectCount, visitor);
        visitor->Finish();
    }

    namespace {
        TExistsCheckerFactory::TRegistrator<TFSExistsChecker> FSQuantizedExistsCheckerReg("quantized");
        TDatasetLoaderFactory::TRegistrator<TCBQuantizedDataLoader> CBQuantizedDataLoaderReg("quantized");
    }
}

