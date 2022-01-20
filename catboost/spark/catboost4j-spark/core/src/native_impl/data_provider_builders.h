#pragma once

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/data_provider.h>
#include <catboost/libs/data/data_provider_builders.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/data/visitor.h>
#include <catboost/libs/helpers/dynamic_iterator.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/ptr.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>


namespace NPar {
    class TLocalExecutor;
}


NCB::TRawObjectsDataProviderPtr CreateRawObjectsDataProvider(
    NCB::TFeaturesLayoutPtr featuresLayout,
    i64 objectCount,
    TVector<NCB::TMaybeOwningConstArrayHolder<float>>* columnwiseFloatFeaturesData,
    TVector<NCB::TMaybeOwningConstArrayHolder<i32>>* columnwiseCatFeaturesData,
    i32 maxUniqCatFeatureValues,
    NPar::TLocalExecutor* localExecutor
);


class TQuantizedRowAssembler {
public:
    TQuantizedRowAssembler(NCB::TQuantizedObjectsDataProviderPtr objectsData);

    i32 GetObjectBlobSize() const;

    void AssembleObjectBlob(i32 objectIdx, TArrayRef<i8> buffer);

private:
    NCB::TQuantizedObjectsDataProviderPtr ObjectsData;

    size_t BlocksStartOffset = 0;
    size_t BlocksSize = 0;

    TVector<NCB::IDynamicBlockIteratorPtr<ui8>> Ui8ColumnIterators;
    TVector<NCB::IDynamicBlockIteratorPtr<ui16>> Ui16ColumnIterators;
    TVector<NCB::IDynamicBlockIteratorPtr<ui32>> Ui32ColumnIterators;

    TVector<TConstArrayRef<ui8>> Ui8ColumnBlocks;
    TVector<TConstArrayRef<ui16>> Ui16ColumnBlocks;
    TVector<TConstArrayRef<ui32>> Ui32ColumnBlocks;
};


// Difference is that it uses default local executor and threadCount argument instead
class TDataProviderClosureForJVM {
public:
    TDataProviderClosureForJVM(
        NCB::EDatasetVisitorType visitorType,
        const NCB::TDataProviderBuilderOptions& options,
        bool hasFeatures,
        NPar::TLocalExecutor* localExecutor
    );

    template <class IVisitor>
    IVisitor* GetVisitor() {
        return dynamic_cast<IVisitor*>(DataProviderBuilder.Get());
    }

#ifdef SWIG
    %template(GetQuantizedVisitor) GetVisitor<NCB::IQuantizedFeaturesDataVisitor>;
#endif

    NCB::TDataProviderPtr GetResult() {
        return DataProviderBuilder->GetResult();
    }

private:
    THolder<NCB::IDataProviderBuilder> DataProviderBuilder;
};

