// This file was auto-generated. Do not edit!!!
#include <catboost/libs/helpers/sparse_array.h>
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

// I/O for NCB::ESparseArrayIndexingType
namespace { namespace NNCBESparseArrayIndexingTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::ESparseArrayIndexingType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(NCB::ESparseArrayIndexingType::Indices, "Indices"sv),
        TNameBufsBase::EnumStringPair(NCB::ESparseArrayIndexingType::Blocks, "Blocks"sv),
        TNameBufsBase::EnumStringPair(NCB::ESparseArrayIndexingType::HybridIndex, "HybridIndex"sv),
        TNameBufsBase::EnumStringPair(NCB::ESparseArrayIndexingType::Undefined, "Undefined"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(NCB::ESparseArrayIndexingType::Blocks, "Blocks"sv),
        TNameBufsBase::EnumStringPair(NCB::ESparseArrayIndexingType::HybridIndex, "HybridIndex"sv),
        TNameBufsBase::EnumStringPair(NCB::ESparseArrayIndexingType::Indices, "Indices"sv),
        TNameBufsBase::EnumStringPair(NCB::ESparseArrayIndexingType::Undefined, "Undefined"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "Indices"sv,
        "Blocks"sv,
        "HybridIndex"sv,
        "Undefined"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NCB::ESparseArrayIndexingType::"sv,
        "NCB::ESparseArrayIndexingType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NCB::ESparseArrayIndexingType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::ESparseArrayIndexingType>;

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

const TString& ToString(NCB::ESparseArrayIndexingType x) {
    const NNCBESparseArrayIndexingTypePrivate::TNameBufs& names = NNCBESparseArrayIndexingTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NCB::ESparseArrayIndexingType FromStringImpl<NCB::ESparseArrayIndexingType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNCBESparseArrayIndexingTypePrivate::TNameBufs, NCB::ESparseArrayIndexingType>(data, len);
}

template<>
bool TryFromStringImpl<NCB::ESparseArrayIndexingType>(const char* data, size_t len, NCB::ESparseArrayIndexingType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNCBESparseArrayIndexingTypePrivate::TNameBufs, NCB::ESparseArrayIndexingType>(data, len, result);
}

bool FromString(const TString& name, NCB::ESparseArrayIndexingType& ret) {
    return ::TryFromStringImpl<NCB::ESparseArrayIndexingType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NCB::ESparseArrayIndexingType& ret) {
    return ::TryFromStringImpl<NCB::ESparseArrayIndexingType>(name.data(), name.size(), ret);
}

template<>
void Out<NCB::ESparseArrayIndexingType>(IOutputStream& os, TTypeTraits<NCB::ESparseArrayIndexingType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNCBESparseArrayIndexingTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NCB::ESparseArrayIndexingType>(NCB::ESparseArrayIndexingType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNCBESparseArrayIndexingTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NCB::ESparseArrayIndexingType> GetEnumAllValuesImpl<NCB::ESparseArrayIndexingType>() {
        const NNCBESparseArrayIndexingTypePrivate::TNameBufs& names = NNCBESparseArrayIndexingTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NCB::ESparseArrayIndexingType>() {
        const NNCBESparseArrayIndexingTypePrivate::TNameBufs& names = NNCBESparseArrayIndexingTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NCB::ESparseArrayIndexingType, TString> GetEnumNamesImpl<NCB::ESparseArrayIndexingType>() {
        const NNCBESparseArrayIndexingTypePrivate::TNameBufs& names = NNCBESparseArrayIndexingTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NCB::ESparseArrayIndexingType>() {
        const NNCBESparseArrayIndexingTypePrivate::TNameBufs& names = NNCBESparseArrayIndexingTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

