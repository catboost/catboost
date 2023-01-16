#include "data_provider_builders.h"

#include <catboost/libs/cat_feature/cat_feature.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/visitor.h>

#include <catboost/libs/helpers/exception.h>
#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/cast.h>
#include <util/generic/xrange.h>

#include <util/system/env.h>

using namespace NCB;


TRawObjectsDataProviderPtr CreateRawObjectsDataProvider(
    TFeaturesLayoutPtr featuresLayout,
    i64 objectCount,
    TVector<TMaybeOwningConstArrayHolder<float>>* columnwiseFloatFeaturesData,
    TVector<TMaybeOwningConstArrayHolder<i32>>* columnwiseCatFeaturesData,
    i32 maxUniqCatFeatureValues,
    NPar::TLocalExecutor* localExecutor
) {
    auto loaderFunc = [&] (IRawFeaturesOrderDataVisitor* visitor) {
        TDataMetaInfo metaInfo;
        metaInfo.ObjectCount = SafeIntegerCast<ui32>(objectCount);
        metaInfo.FeaturesLayout = featuresLayout;
        visitor->Start(
            metaInfo,
            metaInfo.ObjectCount,
            EObjectsOrder::Undefined,
            {}
        );

        {
            auto srcFloatColumnsIterator = columnwiseFloatFeaturesData->begin();
            featuresLayout->IterateOverAvailableFeatures<EFeatureType::Float>(
                [&] (TFloatFeatureIdx idx) {
                    visitor->AddFloatFeature(
                        featuresLayout->GetFloatFeatureInternalIdxToExternalIdx()[*idx],
                        MakeTypeCastArrayHolder<float, float>(*srcFloatColumnsIterator++)
                    );
                }
            );
        }
        if (!columnwiseCatFeaturesData->empty()) {
            TVector<ui32> integerValueHashes; // hashes for "0", "1" ... etc.
            for (auto i : xrange((ui32)maxUniqCatFeatureValues)) {
                integerValueHashes.push_back(CalcCatFeatureHash(ToString(i)));
            }

            auto srcCatColumnsIterator = columnwiseCatFeaturesData->begin();
            featuresLayout->IterateOverAvailableFeatures<EFeatureType::Categorical>(
                [&] (TCatFeatureIdx idx) {
                    TConstArrayRef<i32> integerCatValues = **(srcCatColumnsIterator++);

                    TVector<ui32> hashedCatFeatureValues;
                    hashedCatFeatureValues.yresize(objectCount);

                    localExecutor->ExecRangeBlockedWithThrow(
                        [&] (int i) {
                            hashedCatFeatureValues[i] = integerValueHashes[integerCatValues[i]];
                        },
                        0,
                        SafeIntegerCast<int>(objectCount),
                        /*batchSizeOrZeroForAutoBatchSize*/ 0,
                        NPar::TLocalExecutor::WAIT_COMPLETE
                    );

                    visitor->AddCatFeature(
                        featuresLayout->GetCatFeatureInternalIdxToExternalIdx()[*idx],
                        TMaybeOwningConstArrayHolder<ui32>::CreateOwning(std::move(hashedCatFeatureValues))
                    );
                }
            );
        }
        visitor->Finish();
    };
    TRawObjectsDataProviderPtr result(
        dynamic_cast<TRawObjectsDataProvider*>(
            CreateDataProvider(std::move(loaderFunc))->ObjectsData.Release()
        )
    );
    {
        TVector<TMaybeOwningConstArrayHolder<float>>().swap(*columnwiseFloatFeaturesData);
        TVector<TMaybeOwningConstArrayHolder<i32>>().swap(*columnwiseCatFeaturesData);
    }
    return result;
}

template <class TDst, class TSrc>
bool TryMakeTransformIterator(
    IDynamicBlockIteratorBasePtr& valuesIterator,
    TVector<IDynamicBlockIteratorPtr<TDst>>* iterators
) {
    if (auto* blockOfTIterator = dynamic_cast<IDynamicBlockIterator<TSrc>*>(valuesIterator.Get())) {
        iterators->push_back(
            MakeBlockTransformerIterator<TDst, TSrc>(
                IDynamicBlockIteratorPtr<TSrc>(blockOfTIterator),
                [] (TConstArrayRef<TSrc> src, TArrayRef<TDst> dst) {
                    Copy(src.begin(), src.end(), dst.begin());
                }
            )
        );
        Y_UNUSED(valuesIterator.Release());
        return true;
    }
    return false;
}

template <class T>
void MakeTransformIterator(
    IDynamicBlockIteratorBasePtr valuesIterator, // moved into
    TVector<IDynamicBlockIteratorPtr<T>>* iterators
) {
    if (TryMakeTransformIterator<T, ui8>(valuesIterator, iterators)) {
        return;
    }
    if (TryMakeTransformIterator<T, ui16>(valuesIterator, iterators)) {
        return;
    }
    if (TryMakeTransformIterator<T, ui32>(valuesIterator, iterators)) {
        return;
    }
    CB_ENSURE_INTERNAL(false, "valuesIterator is of unknown type");
}

template <class T>
void AddColumn(
    IDynamicBlockIteratorBasePtr valuesIterator, // moved into
    TVector<IDynamicBlockIteratorPtr<T>>* iterators,
    TVector<TConstArrayRef<T>>* blocks,
    size_t* lastColumnBlockSize
) {

    if (auto* blockOfTIterator = dynamic_cast<IDynamicBlockIterator<T>*>(valuesIterator.Get())) {
        iterators->push_back(IDynamicBlockIteratorPtr<T>(blockOfTIterator));
        Y_UNUSED(valuesIterator.Release());
    } else {
        MakeTransformIterator(std::move(valuesIterator), iterators);
    }
    blocks->push_back(iterators->back()->Next());
    *lastColumnBlockSize = blocks->back().size();
}


TQuantizedRowAssembler::TQuantizedRowAssembler(
    TQuantizedObjectsDataProviderPtr objectsData
) : ObjectsData(objectsData) {
    const auto& quantizedFeaturesInfo = *(objectsData->GetQuantizedFeaturesInfo());
    const auto& featuresLayout = *(quantizedFeaturesInfo.GetFeaturesLayout());

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx idx) {
            const IQuantizedFloatValuesHolder* valuesHolder = *(objectsData->GetNonPackedFloatFeature(*idx));
            IDynamicBlockIteratorBasePtr valuesIterator = valuesHolder->GetBlockIterator();

            size_t lastColumnBlockSize;
            if (quantizedFeaturesInfo.GetBorders(idx).size() > 255) {
                AddColumn<ui16>(
                    std::move(valuesIterator),
                    &Ui16ColumnIterators,
                    &Ui16ColumnBlocks,
                    &lastColumnBlockSize
                );
            } else {
                AddColumn<ui8>(
                    std::move(valuesIterator),
                    &Ui8ColumnIterators,
                    &Ui8ColumnBlocks,
                    &lastColumnBlockSize
                );
            }
            if (BlocksSize) {
                CB_ENSURE_INTERNAL(lastColumnBlockSize == BlocksSize, "All column block sizes must be equal");
            } else {
                BlocksSize = lastColumnBlockSize;
            }
        }
    );
    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Categorical>(
        [&] (TCatFeatureIdx idx) {
            const IQuantizedCatValuesHolder* valuesHolder = *(objectsData->GetNonPackedCatFeature(*idx));
            IDynamicBlockIteratorBasePtr valuesIterator = valuesHolder->GetBlockIterator();

            ui32 uniqValuesCount = quantizedFeaturesInfo.GetUniqueValuesCounts(idx).OnAll;

            size_t lastColumnBlockSize;
            if (uniqValuesCount > ((ui32)Max<ui16>() + 1)) {
                AddColumn<ui32>(
                    std::move(valuesIterator),
                    &Ui32ColumnIterators,
                    &Ui32ColumnBlocks,
                    &lastColumnBlockSize
                );
            } else if (uniqValuesCount > ((ui32)Max<ui8>() + 1)) {
                AddColumn<ui16>(
                    std::move(valuesIterator),
                    &Ui16ColumnIterators,
                    &Ui16ColumnBlocks,
                    &lastColumnBlockSize
                );
            } else {
                AddColumn<ui8>(
                    std::move(valuesIterator),
                    &Ui8ColumnIterators,
                    &Ui8ColumnBlocks,
                    &lastColumnBlockSize
                );
            }
            if (BlocksSize) {
                CB_ENSURE_INTERNAL(lastColumnBlockSize == BlocksSize, "All column block sizes must be equal");
            } else {
                BlocksSize = lastColumnBlockSize;
            }
        }
    );
}

i32 TQuantizedRowAssembler::GetObjectBlobSize() const {
    return SafeIntegerCast<i32>(
        sizeof(ui8) * Ui8ColumnBlocks.size()
        + sizeof(ui16) * Ui16ColumnBlocks.size()
        + sizeof(ui32) * Ui32ColumnBlocks.size()
    );
}

void TQuantizedRowAssembler::AssembleObjectBlob(i32 objectIdx, TArrayRef<i8> buffer) {
    ui32 unsignedObjectIdx = SafeIntegerCast<ui32>(objectIdx);

    CB_ENSURE(
        unsignedObjectIdx >= BlocksStartOffset,
        "object indices must sequentially increase"
    );

    while (unsignedObjectIdx >= (BlocksStartOffset + BlocksSize)) {
        BlocksStartOffset += BlocksSize;
        BlocksSize = 0;

        auto updateBlocks = [&] (auto& blocks, auto& blockIterators) {
            for (auto i : xrange(blocks.size())) {
                blocks[i] = blockIterators[i]->Next();
                if (BlocksSize) {
                    CB_ENSURE_INTERNAL(
                        blocks[i].size() == BlocksSize,
                        "All column block sizes must be equal"
                    );
                } else {
                    BlocksSize = blocks[i].size();
                }
            }
        };

        updateBlocks(Ui8ColumnBlocks, Ui8ColumnIterators);
        updateBlocks(Ui16ColumnBlocks, Ui16ColumnIterators);
        updateBlocks(Ui32ColumnBlocks, Ui32ColumnIterators);

        CB_ENSURE(BlocksSize != 0, "Dataset does not contain an element with index " << objectIdx);
    }
    ui32 offsetInBlock = unsignedObjectIdx - BlocksStartOffset;

    i8* dstPtr = buffer.data();

    auto writeValues = [&] (auto& blocks, auto dstPtr) {
        for (auto block : blocks) {
            *dstPtr++ = block[offsetInBlock];
        }
    };

    writeValues(Ui8ColumnBlocks, (ui8*)dstPtr);
    dstPtr += sizeof(ui8) * Ui8ColumnBlocks.size();
    writeValues(Ui16ColumnBlocks, (ui16*)dstPtr);
    dstPtr += sizeof(ui16) * Ui16ColumnBlocks.size();
    writeValues(Ui32ColumnBlocks, (ui32*)dstPtr);
}


TDataProviderClosureForJVM::TDataProviderClosureForJVM(
    EDatasetVisitorType visitorType,
    const TDataProviderBuilderOptions& options,
    bool hasFeatures,
    NPar::TLocalExecutor* localExecutor
) {
    DataProviderBuilder = CreateDataProviderBuilder(
        visitorType,
        options,
        TDatasetSubset::MakeColumns(hasFeatures),
        localExecutor
    );
    CB_ENSURE_INTERNAL(
        DataProviderBuilder.Get(),
        "Failed to create data provider builder for visitor of type "
        << visitorType
    );
}

