// This file was auto-generated. Do not edit!!!
#include <catboost/libs/column_description/column.h>
#include <tools/enum_parser/enum_serialization_runtime/enum_runtime.h>

#include <tools/enum_parser/enum_parser/stdlib_deps.h>

#include <util/generic/typetraits.h>
#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/map.h>
#include <util/generic/serialized_enum.h>
#include <util/string/cast.h>
#include <util/stream/output.h>

// I/O for EColumn
namespace { namespace NEColumnPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EColumn>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 17> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 17>{{
        TNameBufsBase::EnumStringPair(EColumn::Num, "Num"sv),
        TNameBufsBase::EnumStringPair(EColumn::Categ, "Categ"sv),
        TNameBufsBase::EnumStringPair(EColumn::HashedCateg, "HashedCateg"sv),
        TNameBufsBase::EnumStringPair(EColumn::Label, "Label"sv),
        TNameBufsBase::EnumStringPair(EColumn::Auxiliary, "Auxiliary"sv),
        TNameBufsBase::EnumStringPair(EColumn::Baseline, "Baseline"sv),
        TNameBufsBase::EnumStringPair(EColumn::Weight, "Weight"sv),
        TNameBufsBase::EnumStringPair(EColumn::SampleId, "SampleId"sv),
        TNameBufsBase::EnumStringPair(EColumn::GroupId, "GroupId"sv),
        TNameBufsBase::EnumStringPair(EColumn::GroupWeight, "GroupWeight"sv),
        TNameBufsBase::EnumStringPair(EColumn::SubgroupId, "SubgroupId"sv),
        TNameBufsBase::EnumStringPair(EColumn::Timestamp, "Timestamp"sv),
        TNameBufsBase::EnumStringPair(EColumn::Sparse, "Sparse"sv),
        TNameBufsBase::EnumStringPair(EColumn::Prediction, "Prediction"sv),
        TNameBufsBase::EnumStringPair(EColumn::Text, "Text"sv),
        TNameBufsBase::EnumStringPair(EColumn::NumVector, "NumVector"sv),
        TNameBufsBase::EnumStringPair(EColumn::Features, "Features"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[17]{
        TNameBufsBase::EnumStringPair(EColumn::Auxiliary, "Auxiliary"sv),
        TNameBufsBase::EnumStringPair(EColumn::Baseline, "Baseline"sv),
        TNameBufsBase::EnumStringPair(EColumn::Categ, "Categ"sv),
        TNameBufsBase::EnumStringPair(EColumn::Features, "Features"sv),
        TNameBufsBase::EnumStringPair(EColumn::GroupId, "GroupId"sv),
        TNameBufsBase::EnumStringPair(EColumn::GroupWeight, "GroupWeight"sv),
        TNameBufsBase::EnumStringPair(EColumn::HashedCateg, "HashedCateg"sv),
        TNameBufsBase::EnumStringPair(EColumn::Label, "Label"sv),
        TNameBufsBase::EnumStringPair(EColumn::Num, "Num"sv),
        TNameBufsBase::EnumStringPair(EColumn::NumVector, "NumVector"sv),
        TNameBufsBase::EnumStringPair(EColumn::Prediction, "Prediction"sv),
        TNameBufsBase::EnumStringPair(EColumn::SampleId, "SampleId"sv),
        TNameBufsBase::EnumStringPair(EColumn::Sparse, "Sparse"sv),
        TNameBufsBase::EnumStringPair(EColumn::SubgroupId, "SubgroupId"sv),
        TNameBufsBase::EnumStringPair(EColumn::Text, "Text"sv),
        TNameBufsBase::EnumStringPair(EColumn::Timestamp, "Timestamp"sv),
        TNameBufsBase::EnumStringPair(EColumn::Weight, "Weight"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[17]{
        "Num"sv,
        "Categ"sv,
        "HashedCateg"sv,
        "Label"sv,
        "Auxiliary"sv,
        "Baseline"sv,
        "Weight"sv,
        "SampleId"sv,
        "GroupId"sv,
        "GroupWeight"sv,
        "SubgroupId"sv,
        "Timestamp"sv,
        "Sparse"sv,
        "Prediction"sv,
        "Text"sv,
        "NumVector"sv,
        "Features"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EColumn::"sv,
        "EColumn"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EColumn> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EColumn>;

        inline TNameBufs();

        static constexpr const TNameBufsBase::TInitializationData& EnumInitializationData = ENUM_INITIALIZATION_DATA;
        static constexpr ::NEnumSerializationRuntime::ESortOrder NamesOrder = NAMES_ORDER;
        static constexpr ::NEnumSerializationRuntime::ESortOrder ValuesOrder = VALUES_ORDER;

        static inline const TNameBufs& Instance() {
            return *SingletonWithPriority<TNameBufs, 0>();
        }
    };

    inline TNameBufs::TNameBufs()
        : TBase(TNameBufs::EnumInitializationData)
    {
    }

}}

const TString& ToString(EColumn x) {
    const NEColumnPrivate::TNameBufs& names = NEColumnPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EColumn FromStringImpl<EColumn>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEColumnPrivate::TNameBufs, EColumn>(data, len);
}

template<>
bool TryFromStringImpl<EColumn>(const char* data, size_t len, EColumn& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEColumnPrivate::TNameBufs, EColumn>(data, len, result);
}

bool FromString(const TString& name, EColumn& ret) {
    return ::TryFromStringImpl<EColumn>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EColumn& ret) {
    return ::TryFromStringImpl<EColumn>(name.data(), name.size(), ret);
}

template<>
void Out<EColumn>(IOutputStream& os, TTypeTraits<EColumn>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEColumnPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EColumn>(EColumn e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEColumnPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EColumn> GetEnumAllValuesImpl<EColumn>() {
        const NEColumnPrivate::TNameBufs& names = NEColumnPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EColumn>() {
        const NEColumnPrivate::TNameBufs& names = NEColumnPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EColumn, TString> GetEnumNamesImpl<EColumn>() {
        const NEColumnPrivate::TNameBufs& names = NEColumnPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EColumn>() {
        const NEColumnPrivate::TNameBufs& names = NEColumnPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

