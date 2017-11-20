#include "features.h"

#include "flatbuffers_serializer_helper.h"

flatbuffers::Offset<NCatBoostFbs::TCtrFeature> TCtrFeature::FBSerialize(TModelPartsCachingSerializer& serializer) const {
    return NCatBoostFbs::CreateTCtrFeatureDirect(
        serializer.FlatbufBuilder,
        serializer.GetOffset(Ctr),
        &Borders
    );
}
