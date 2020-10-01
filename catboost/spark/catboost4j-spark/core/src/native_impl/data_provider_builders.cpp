#include "data_provider_builders.h"

#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/visitor.h>

#include <catboost/libs/helpers/maybe_owning_array_holder.h>

#include <library/cpp/threading/local_executor/local_executor.h>

#include <util/generic/cast.h>
#include <util/generic/xrange.h>

using namespace NCB;


TRawObjectsDataProviderPtr CreateRawObjectsDataProvider(
    TFeaturesLayoutPtr featuresLayout,
    i64 objectCount,
    TVector<TMaybeOwningConstArrayHolder<float>>* columnwiseFloatFeaturesData
) throw (yexception) {
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

        auto srcColumnsIterator = columnwiseFloatFeaturesData->begin();
        featuresLayout->IterateOverAvailableFeatures<EFeatureType::Float>(
            [&] (TFloatFeatureIdx idx) {
                visitor->AddFloatFeature(
                    *idx,
                    MakeTypeCastArrayHolder<float, float>(*srcColumnsIterator++)
                );
            }
        );
        visitor->Finish();
    };
    TRawObjectsDataProviderPtr result(
        dynamic_cast<TRawObjectsDataProvider*>(
            CreateDataProvider(std::move(loaderFunc))->ObjectsData.Release()
        )
    );
    {
        TVector<TMaybeOwningConstArrayHolder<float>>().swap(*columnwiseFloatFeaturesData);
    }
    return result;
}

TQuantizedRowAssembler::TQuantizedRowAssembler(
    TQuantizedObjectsDataProviderPtr objectsData
) throw (yexception) {
    const auto& quantizedFeaturesInfo = *(objectsData->GetQuantizedFeaturesInfo());
    const auto& featuresLayout = *(quantizedFeaturesInfo.GetFeaturesLayout());

    featuresLayout.IterateOverAvailableFeatures<EFeatureType::Float>(
        [&] (TFloatFeatureIdx idx) {
            const IQuantizedFloatValuesHolder* valuesHolder = *(objectsData->GetNonPackedFloatFeature(*idx));
            IDynamicBlockIteratorBasePtr valuesIterator = valuesHolder->GetBlockIterator();

            size_t lastColumnBlockSize;
            if (quantizedFeaturesInfo.GetBorders(idx).size() > 255) {
                Ui16ColumnIterators.push_back(
                    IDynamicBlockIteratorPtr<ui16>(
                        dynamic_cast<IDynamicBlockIterator<ui16>*>(valuesIterator.Release())
                    )
                );
                Ui16ColumnBlocks.push_back(Ui16ColumnIterators.back()->Next());
                lastColumnBlockSize = Ui16ColumnBlocks.back().size();
            } else {
                Ui8ColumnIterators.push_back(
                    IDynamicBlockIteratorPtr<ui8>(
                        dynamic_cast<IDynamicBlockIterator<ui8>*>(valuesIterator.Release())
                    )
                );
                Ui8ColumnBlocks.push_back(Ui8ColumnIterators.back()->Next());
                lastColumnBlockSize = Ui8ColumnBlocks.back().size();
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
        sizeof(ui8) * Ui8ColumnBlocks.size() + sizeof(ui16) * Ui16ColumnBlocks.size()
    );
}

void TQuantizedRowAssembler::AssembleObjectBlob(i32 objectIdx, TArrayRef<i8> buffer) throw (yexception) {
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
    writeValues(Ui16ColumnBlocks, (ui16*)(dstPtr + Ui8ColumnBlocks.size()));
}


TDataProviderClosureForJVM::TDataProviderClosureForJVM(
    EDatasetVisitorType visitorType,
    const TDataProviderBuilderOptions& options,
    bool hasFeatures,
    i32 threadCount
) throw (yexception) {
    NPar::TLocalExecutor* localExecutor = &NPar::LocalExecutor();
    if ((localExecutor->GetThreadCount() + 1) < threadCount) {
        localExecutor->RunAdditionalThreads(threadCount - 1);
    }
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

