#pragma once

#include "columns.h"

#include <catboost/private/libs/data_util/path_with_scheme.h>
#include <catboost/private/libs/quantized_pool/loader.h>

namespace NCB {

    template <class TBase>
    class TLazyCompressedValuesHolderImpl : public TBase {
    public:
        using TLoadedColumnData = TVector<ui8>; // TODO(kirillovs): support wide historgrams in "lazy" columns

        template<typename T>
        class TLazyCompressedValuesIterator : public IDynamicBlockIterator<T> {
        public:
            TLazyCompressedValuesIterator(
                const TFeaturesArraySubsetIndexing* subsetIndexing,
                std::shared_ptr<TLoadedColumnData>&& columnData,
                size_t offset
            )
                : SubsetIndexing(subsetIndexing)
                , ColumnData(std::move(columnData))
            {
                Iterator = MakeArraySubsetBlockIterator<T>(
                    SubsetIndexing,
                    MakeArrayRef(*ColumnData),
                    offset
                );
            }

            TConstArrayRef<T> Next(size_t blockSize) {
                return Iterator->Next(blockSize);
            }
        private:
            const TFeaturesArraySubsetIndexing* SubsetIndexing;
            std::shared_ptr<TLoadedColumnData> ColumnData;
            IDynamicBlockIteratorPtr<T> Iterator;
        };

        TLazyCompressedValuesHolderImpl(
            ui32 featureId,
            const TFeaturesArraySubsetIndexing* subsetIndexing,
            TAtomicSharedPtr<IQuantizedPoolLoader> poolLoader)
        : TBase(featureId, subsetIndexing->Size())
        , SubsetIndexing(subsetIndexing)
        , PoolLoader(poolLoader)
        {
        }

        bool IsSparse() const override {
            return false;
        }

        ui64 EstimateMemoryForCloning(
            const TCloningParams& cloningParams
        ) const override {
            Y_UNUSED(cloningParams);
            return 0;
        }

        THolder<IFeatureValuesHolder> CloneWithNewSubsetIndexing(
            const TCloningParams& cloningParams,
            NPar::TLocalExecutor* localExecutor
        ) const override {
            Y_UNUSED(localExecutor);
            CB_ENSURE_INTERNAL(!cloningParams.MakeConsecutive, "Making consecutive not supported on Lazy columns for now");
            return MakeHolder<TLazyCompressedValuesHolderImpl>(
                TBase::GetId(),
                cloningParams.SubsetIndexing,
                PoolLoader
            );
        }

        IDynamicBlockIteratorBasePtr GetBlockIterator(ui32 offset) const override {
            return MakeHolder<TLazyCompressedValuesIterator<ui8>>(
                SubsetIndexing,
                GetColumnData(),
                offset
            );
        }

    private:
        std::shared_ptr<TLoadedColumnData> GetColumnData() const {
            with_lock(LoadDataLock) {
                auto cachedResult = LoadedColumnDataWeakPtr.lock();
                if (cachedResult) {
                    return cachedResult;
                }
                std::shared_ptr<TLoadedColumnData> loadedColumn = std::make_shared<TLoadedColumnData>(
                    PoolLoader->LoadQuantizedColumn(TBase::GetId())
                );
                LoadedColumnDataWeakPtr = loadedColumn;
                return loadedColumn;
            }
        }
    private:
        mutable TMutex LoadDataLock;
        mutable std::weak_ptr<TVector<ui8>> LoadedColumnDataWeakPtr;

        const TFeaturesArraySubsetIndexing* SubsetIndexing;
        TAtomicSharedPtr<IQuantizedPoolLoader> PoolLoader;
    };
}
