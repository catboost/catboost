// This file was auto-generated. Do not edit!!!
#include <catboost/libs/helpers/distribution_helpers.h>
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

// I/O for NCB::EDistributionType
namespace { namespace NNCBEDistributionTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::EDistributionType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(NCB::EDistributionType::Normal, "Normal"sv),
        TNameBufsBase::EnumStringPair(NCB::EDistributionType::Logistic, "Logistic"sv),
        TNameBufsBase::EnumStringPair(NCB::EDistributionType::Extreme, "Extreme"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(NCB::EDistributionType::Extreme, "Extreme"sv),
        TNameBufsBase::EnumStringPair(NCB::EDistributionType::Logistic, "Logistic"sv),
        TNameBufsBase::EnumStringPair(NCB::EDistributionType::Normal, "Normal"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Normal"sv,
        "Logistic"sv,
        "Extreme"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NCB::EDistributionType::"sv,
        "NCB::EDistributionType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NCB::EDistributionType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::EDistributionType>;

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

const TString& ToString(NCB::EDistributionType x) {
    const NNCBEDistributionTypePrivate::TNameBufs& names = NNCBEDistributionTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NCB::EDistributionType FromStringImpl<NCB::EDistributionType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNCBEDistributionTypePrivate::TNameBufs, NCB::EDistributionType>(data, len);
}

template<>
bool TryFromStringImpl<NCB::EDistributionType>(const char* data, size_t len, NCB::EDistributionType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNCBEDistributionTypePrivate::TNameBufs, NCB::EDistributionType>(data, len, result);
}

bool FromString(const TString& name, NCB::EDistributionType& ret) {
    return ::TryFromStringImpl<NCB::EDistributionType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NCB::EDistributionType& ret) {
    return ::TryFromStringImpl<NCB::EDistributionType>(name.data(), name.size(), ret);
}

template<>
void Out<NCB::EDistributionType>(IOutputStream& os, TTypeTraits<NCB::EDistributionType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNCBEDistributionTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NCB::EDistributionType>(NCB::EDistributionType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNCBEDistributionTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NCB::EDistributionType> GetEnumAllValuesImpl<NCB::EDistributionType>() {
        const NNCBEDistributionTypePrivate::TNameBufs& names = NNCBEDistributionTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NCB::EDistributionType>() {
        const NNCBEDistributionTypePrivate::TNameBufs& names = NNCBEDistributionTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NCB::EDistributionType, TString> GetEnumNamesImpl<NCB::EDistributionType>() {
        const NNCBEDistributionTypePrivate::TNameBufs& names = NNCBEDistributionTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NCB::EDistributionType>() {
        const NNCBEDistributionTypePrivate::TNameBufs& names = NNCBEDistributionTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

