// This file was auto-generated. Do not edit!!!
#pragma once

#include <util/generic/serialized_enum.h>
#include <catboost/libs/column_description/column.h>
// I/O for EColumn
const TString& ToString(EColumn);
Y_FORCE_INLINE TStringBuf ToStringBuf(EColumn e) {
    return ::NEnumSerializationRuntime::ToStringBuf<EColumn>(e);
}
bool FromString(const TString& name, EColumn& ret);
bool FromString(const TStringBuf& name, EColumn& ret);
template <>
constexpr size_t GetEnumItemsCount<EColumn>() {
    return 17;
}
