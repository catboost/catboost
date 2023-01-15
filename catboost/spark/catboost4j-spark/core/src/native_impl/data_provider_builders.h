#pragma once

#include <catboost/libs/data/features_layout.h>
#include <catboost/libs/data/objects.h>
#include <catboost/libs/helpers/dynamic_iterator.h>

#include <util/generic/fwd.h>
#include <util/generic/array_ref.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/system/types.h>


NCB::TRawObjectsDataProviderPtr CreateRawObjectsDataProvider(
    NCB::TFeaturesLayoutPtr featuresLayout,
    i64 objectCount,
    TVector<NCB::TMaybeOwningConstArrayHolder<float>>* columnwiseFloatFeaturesData
) throw (yexception);


class TQuantizedRowAssembler {
public:
    TQuantizedRowAssembler(NCB::TQuantizedObjectsDataProviderPtr objectsData);

    i32 GetObjectBlobSize() const;

    void AssembleObjectBlob(i32 objectIdx, TArrayRef<i8> buffer);

private:
    size_t BlocksStartOffset = 0;
    size_t BlocksSize = 0;

    TVector<NCB::IDynamicBlockIteratorPtr<ui8>> Ui8ColumnIterators;
    TVector<NCB::IDynamicBlockIteratorPtr<ui16>> Ui16ColumnIterators;

    TVector<TConstArrayRef<ui8>> Ui8ColumnBlocks;
    TVector<TConstArrayRef<ui16>> Ui16ColumnBlocks;
};
