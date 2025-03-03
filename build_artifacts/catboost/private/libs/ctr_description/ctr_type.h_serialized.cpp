// This file was auto-generated. Do not edit!!!
#include <catboost/private/libs/ctr_description/ctr_type.h>
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

// I/O for ECtrType
namespace { namespace NECtrTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ECtrType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 7> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 7>{{
        TNameBufsBase::EnumStringPair(ECtrType::Borders, "Borders"sv),
        TNameBufsBase::EnumStringPair(ECtrType::Buckets, "Buckets"sv),
        TNameBufsBase::EnumStringPair(ECtrType::BinarizedTargetMeanValue, "BinarizedTargetMeanValue"sv),
        TNameBufsBase::EnumStringPair(ECtrType::FloatTargetMeanValue, "FloatTargetMeanValue"sv),
        TNameBufsBase::EnumStringPair(ECtrType::Counter, "Counter"sv),
        TNameBufsBase::EnumStringPair(ECtrType::FeatureFreq, "FeatureFreq"sv),
        TNameBufsBase::EnumStringPair(ECtrType::CtrTypesCount, "CtrTypesCount"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[7]{
        TNameBufsBase::EnumStringPair(ECtrType::BinarizedTargetMeanValue, "BinarizedTargetMeanValue"sv),
        TNameBufsBase::EnumStringPair(ECtrType::Borders, "Borders"sv),
        TNameBufsBase::EnumStringPair(ECtrType::Buckets, "Buckets"sv),
        TNameBufsBase::EnumStringPair(ECtrType::Counter, "Counter"sv),
        TNameBufsBase::EnumStringPair(ECtrType::CtrTypesCount, "CtrTypesCount"sv),
        TNameBufsBase::EnumStringPair(ECtrType::FeatureFreq, "FeatureFreq"sv),
        TNameBufsBase::EnumStringPair(ECtrType::FloatTargetMeanValue, "FloatTargetMeanValue"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[7]{
        "Borders"sv,
        "Buckets"sv,
        "BinarizedTargetMeanValue"sv,
        "FloatTargetMeanValue"sv,
        "Counter"sv,
        "FeatureFreq"sv,
        "CtrTypesCount"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ECtrType::"sv,
        "ECtrType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ECtrType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ECtrType>;

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

const TString& ToString(ECtrType x) {
    const NECtrTypePrivate::TNameBufs& names = NECtrTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ECtrType FromStringImpl<ECtrType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NECtrTypePrivate::TNameBufs, ECtrType>(data, len);
}

template<>
bool TryFromStringImpl<ECtrType>(const char* data, size_t len, ECtrType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NECtrTypePrivate::TNameBufs, ECtrType>(data, len, result);
}

bool FromString(const TString& name, ECtrType& ret) {
    return ::TryFromStringImpl<ECtrType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ECtrType& ret) {
    return ::TryFromStringImpl<ECtrType>(name.data(), name.size(), ret);
}

template<>
void Out<ECtrType>(IOutputStream& os, TTypeTraits<ECtrType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NECtrTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ECtrType>(ECtrType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NECtrTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ECtrType> GetEnumAllValuesImpl<ECtrType>() {
        const NECtrTypePrivate::TNameBufs& names = NECtrTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ECtrType>() {
        const NECtrTypePrivate::TNameBufs& names = NECtrTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ECtrType, TString> GetEnumNamesImpl<ECtrType>() {
        const NECtrTypePrivate::TNameBufs& names = NECtrTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ECtrType>() {
        const NECtrTypePrivate::TNameBufs& names = NECtrTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ECtrHistoryUnit
namespace { namespace NECtrHistoryUnitPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ECtrHistoryUnit>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ECtrHistoryUnit::Group, "Group"sv),
        TNameBufsBase::EnumStringPair(ECtrHistoryUnit::Sample, "Sample"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ECtrHistoryUnit::Group, "Group"sv),
        TNameBufsBase::EnumStringPair(ECtrHistoryUnit::Sample, "Sample"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Group"sv,
        "Sample"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ECtrHistoryUnit::"sv,
        "ECtrHistoryUnit"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ECtrHistoryUnit> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ECtrHistoryUnit>;

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

const TString& ToString(ECtrHistoryUnit x) {
    const NECtrHistoryUnitPrivate::TNameBufs& names = NECtrHistoryUnitPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ECtrHistoryUnit FromStringImpl<ECtrHistoryUnit>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NECtrHistoryUnitPrivate::TNameBufs, ECtrHistoryUnit>(data, len);
}

template<>
bool TryFromStringImpl<ECtrHistoryUnit>(const char* data, size_t len, ECtrHistoryUnit& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NECtrHistoryUnitPrivate::TNameBufs, ECtrHistoryUnit>(data, len, result);
}

bool FromString(const TString& name, ECtrHistoryUnit& ret) {
    return ::TryFromStringImpl<ECtrHistoryUnit>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ECtrHistoryUnit& ret) {
    return ::TryFromStringImpl<ECtrHistoryUnit>(name.data(), name.size(), ret);
}

template<>
void Out<ECtrHistoryUnit>(IOutputStream& os, TTypeTraits<ECtrHistoryUnit>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NECtrHistoryUnitPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ECtrHistoryUnit>(ECtrHistoryUnit e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NECtrHistoryUnitPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ECtrHistoryUnit> GetEnumAllValuesImpl<ECtrHistoryUnit>() {
        const NECtrHistoryUnitPrivate::TNameBufs& names = NECtrHistoryUnitPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ECtrHistoryUnit>() {
        const NECtrHistoryUnitPrivate::TNameBufs& names = NECtrHistoryUnitPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ECtrHistoryUnit, TString> GetEnumNamesImpl<ECtrHistoryUnit>() {
        const NECtrHistoryUnitPrivate::TNameBufs& names = NECtrHistoryUnitPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ECtrHistoryUnit>() {
        const NECtrHistoryUnitPrivate::TNameBufs& names = NECtrHistoryUnitPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

