// This file was auto-generated. Do not edit!!!
#include <library/cpp/json/writer/json_value.h>
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

// I/O for NJson::EJsonValueType
namespace { namespace NNJsonEJsonValueTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NJson::EJsonValueType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 9> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 9>{{
        TNameBufsBase::EnumStringPair(NJson::JSON_UNDEFINED, "Undefined"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_NULL, "Null"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_BOOLEAN, "Boolean"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_INTEGER, "Integer"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_DOUBLE, "Double"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_STRING, "String"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_MAP, "Map"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_ARRAY, "Array"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_UINTEGER, "UInteger"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[9]{
        TNameBufsBase::EnumStringPair(NJson::JSON_ARRAY, "Array"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_BOOLEAN, "Boolean"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_DOUBLE, "Double"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_INTEGER, "Integer"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_MAP, "Map"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_NULL, "Null"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_STRING, "String"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_UINTEGER, "UInteger"sv),
        TNameBufsBase::EnumStringPair(NJson::JSON_UNDEFINED, "Undefined"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[9]{
        "JSON_UNDEFINED"sv,
        "JSON_NULL"sv,
        "JSON_BOOLEAN"sv,
        "JSON_INTEGER"sv,
        "JSON_DOUBLE"sv,
        "JSON_STRING"sv,
        "JSON_MAP"sv,
        "JSON_ARRAY"sv,
        "JSON_UINTEGER"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NJson::"sv,
        "NJson::EJsonValueType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NJson::EJsonValueType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NJson::EJsonValueType>;

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

const TString& ToString(NJson::EJsonValueType x) {
    const NNJsonEJsonValueTypePrivate::TNameBufs& names = NNJsonEJsonValueTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NJson::EJsonValueType FromStringImpl<NJson::EJsonValueType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNJsonEJsonValueTypePrivate::TNameBufs, NJson::EJsonValueType>(data, len);
}

template<>
bool TryFromStringImpl<NJson::EJsonValueType>(const char* data, size_t len, NJson::EJsonValueType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNJsonEJsonValueTypePrivate::TNameBufs, NJson::EJsonValueType>(data, len, result);
}

bool FromString(const TString& name, NJson::EJsonValueType& ret) {
    return ::TryFromStringImpl<NJson::EJsonValueType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NJson::EJsonValueType& ret) {
    return ::TryFromStringImpl<NJson::EJsonValueType>(name.data(), name.size(), ret);
}

template<>
void Out<NJson::EJsonValueType>(IOutputStream& os, TTypeTraits<NJson::EJsonValueType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNJsonEJsonValueTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NJson::EJsonValueType>(NJson::EJsonValueType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNJsonEJsonValueTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NJson::EJsonValueType> GetEnumAllValuesImpl<NJson::EJsonValueType>() {
        const NNJsonEJsonValueTypePrivate::TNameBufs& names = NNJsonEJsonValueTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NJson::EJsonValueType>() {
        const NNJsonEJsonValueTypePrivate::TNameBufs& names = NNJsonEJsonValueTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NJson::EJsonValueType, TString> GetEnumNamesImpl<NJson::EJsonValueType>() {
        const NNJsonEJsonValueTypePrivate::TNameBufs& names = NNJsonEJsonValueTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NJson::EJsonValueType>() {
        const NNJsonEJsonValueTypePrivate::TNameBufs& names = NNJsonEJsonValueTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

