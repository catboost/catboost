// This file was auto-generated. Do not edit!!!
#include <catboost/private/libs/options/enums.h>
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

// I/O for EOverfittingDetectorType
namespace { namespace NEOverfittingDetectorTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EOverfittingDetectorType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(EOverfittingDetectorType::None, "None"sv),
        TNameBufsBase::EnumStringPair(EOverfittingDetectorType::Wilcoxon, "Wilcoxon"sv),
        TNameBufsBase::EnumStringPair(EOverfittingDetectorType::IncToDec, "IncToDec"sv),
        TNameBufsBase::EnumStringPair(EOverfittingDetectorType::Iter, "Iter"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(EOverfittingDetectorType::IncToDec, "IncToDec"sv),
        TNameBufsBase::EnumStringPair(EOverfittingDetectorType::Iter, "Iter"sv),
        TNameBufsBase::EnumStringPair(EOverfittingDetectorType::None, "None"sv),
        TNameBufsBase::EnumStringPair(EOverfittingDetectorType::Wilcoxon, "Wilcoxon"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "None"sv,
        "Wilcoxon"sv,
        "IncToDec"sv,
        "Iter"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EOverfittingDetectorType::"sv,
        "EOverfittingDetectorType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EOverfittingDetectorType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EOverfittingDetectorType>;

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

const TString& ToString(EOverfittingDetectorType x) {
    const NEOverfittingDetectorTypePrivate::TNameBufs& names = NEOverfittingDetectorTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EOverfittingDetectorType FromStringImpl<EOverfittingDetectorType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEOverfittingDetectorTypePrivate::TNameBufs, EOverfittingDetectorType>(data, len);
}

template<>
bool TryFromStringImpl<EOverfittingDetectorType>(const char* data, size_t len, EOverfittingDetectorType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEOverfittingDetectorTypePrivate::TNameBufs, EOverfittingDetectorType>(data, len, result);
}

bool FromString(const TString& name, EOverfittingDetectorType& ret) {
    return ::TryFromStringImpl<EOverfittingDetectorType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EOverfittingDetectorType& ret) {
    return ::TryFromStringImpl<EOverfittingDetectorType>(name.data(), name.size(), ret);
}

template<>
void Out<EOverfittingDetectorType>(IOutputStream& os, TTypeTraits<EOverfittingDetectorType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEOverfittingDetectorTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EOverfittingDetectorType>(EOverfittingDetectorType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEOverfittingDetectorTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EOverfittingDetectorType> GetEnumAllValuesImpl<EOverfittingDetectorType>() {
        const NEOverfittingDetectorTypePrivate::TNameBufs& names = NEOverfittingDetectorTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EOverfittingDetectorType>() {
        const NEOverfittingDetectorTypePrivate::TNameBufs& names = NEOverfittingDetectorTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EOverfittingDetectorType, TString> GetEnumNamesImpl<EOverfittingDetectorType>() {
        const NEOverfittingDetectorTypePrivate::TNameBufs& names = NEOverfittingDetectorTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EOverfittingDetectorType>() {
        const NEOverfittingDetectorTypePrivate::TNameBufs& names = NEOverfittingDetectorTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ESamplingFrequency
namespace { namespace NESamplingFrequencyPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ESamplingFrequency>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ESamplingFrequency::PerTree, "PerTree"sv),
        TNameBufsBase::EnumStringPair(ESamplingFrequency::PerTreeLevel, "PerTreeLevel"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ESamplingFrequency::PerTree, "PerTree"sv),
        TNameBufsBase::EnumStringPair(ESamplingFrequency::PerTreeLevel, "PerTreeLevel"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "PerTree"sv,
        "PerTreeLevel"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ESamplingFrequency::"sv,
        "ESamplingFrequency"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ESamplingFrequency> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ESamplingFrequency>;

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

const TString& ToString(ESamplingFrequency x) {
    const NESamplingFrequencyPrivate::TNameBufs& names = NESamplingFrequencyPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ESamplingFrequency FromStringImpl<ESamplingFrequency>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NESamplingFrequencyPrivate::TNameBufs, ESamplingFrequency>(data, len);
}

template<>
bool TryFromStringImpl<ESamplingFrequency>(const char* data, size_t len, ESamplingFrequency& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NESamplingFrequencyPrivate::TNameBufs, ESamplingFrequency>(data, len, result);
}

bool FromString(const TString& name, ESamplingFrequency& ret) {
    return ::TryFromStringImpl<ESamplingFrequency>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ESamplingFrequency& ret) {
    return ::TryFromStringImpl<ESamplingFrequency>(name.data(), name.size(), ret);
}

template<>
void Out<ESamplingFrequency>(IOutputStream& os, TTypeTraits<ESamplingFrequency>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NESamplingFrequencyPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ESamplingFrequency>(ESamplingFrequency e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NESamplingFrequencyPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ESamplingFrequency> GetEnumAllValuesImpl<ESamplingFrequency>() {
        const NESamplingFrequencyPrivate::TNameBufs& names = NESamplingFrequencyPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ESamplingFrequency>() {
        const NESamplingFrequencyPrivate::TNameBufs& names = NESamplingFrequencyPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ESamplingFrequency, TString> GetEnumNamesImpl<ESamplingFrequency>() {
        const NESamplingFrequencyPrivate::TNameBufs& names = NESamplingFrequencyPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ESamplingFrequency>() {
        const NESamplingFrequencyPrivate::TNameBufs& names = NESamplingFrequencyPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ESamplingUnit
namespace { namespace NESamplingUnitPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ESamplingUnit>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ESamplingUnit::Object, "Object"sv),
        TNameBufsBase::EnumStringPair(ESamplingUnit::Group, "Group"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ESamplingUnit::Group, "Group"sv),
        TNameBufsBase::EnumStringPair(ESamplingUnit::Object, "Object"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Object"sv,
        "Group"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ESamplingUnit::"sv,
        "ESamplingUnit"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ESamplingUnit> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ESamplingUnit>;

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

const TString& ToString(ESamplingUnit x) {
    const NESamplingUnitPrivate::TNameBufs& names = NESamplingUnitPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ESamplingUnit FromStringImpl<ESamplingUnit>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NESamplingUnitPrivate::TNameBufs, ESamplingUnit>(data, len);
}

template<>
bool TryFromStringImpl<ESamplingUnit>(const char* data, size_t len, ESamplingUnit& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NESamplingUnitPrivate::TNameBufs, ESamplingUnit>(data, len, result);
}

bool FromString(const TString& name, ESamplingUnit& ret) {
    return ::TryFromStringImpl<ESamplingUnit>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ESamplingUnit& ret) {
    return ::TryFromStringImpl<ESamplingUnit>(name.data(), name.size(), ret);
}

template<>
void Out<ESamplingUnit>(IOutputStream& os, TTypeTraits<ESamplingUnit>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NESamplingUnitPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ESamplingUnit>(ESamplingUnit e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NESamplingUnitPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ESamplingUnit> GetEnumAllValuesImpl<ESamplingUnit>() {
        const NESamplingUnitPrivate::TNameBufs& names = NESamplingUnitPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ESamplingUnit>() {
        const NESamplingUnitPrivate::TNameBufs& names = NESamplingUnitPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ESamplingUnit, TString> GetEnumNamesImpl<ESamplingUnit>() {
        const NESamplingUnitPrivate::TNameBufs& names = NESamplingUnitPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ESamplingUnit>() {
        const NESamplingUnitPrivate::TNameBufs& names = NESamplingUnitPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EFeatureType
namespace { namespace NEFeatureTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EFeatureType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(EFeatureType::Float, "Float"sv),
        TNameBufsBase::EnumStringPair(EFeatureType::Categorical, "Categorical"sv),
        TNameBufsBase::EnumStringPair(EFeatureType::Text, "Text"sv),
        TNameBufsBase::EnumStringPair(EFeatureType::Embedding, "Embedding"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(EFeatureType::Categorical, "Categorical"sv),
        TNameBufsBase::EnumStringPair(EFeatureType::Embedding, "Embedding"sv),
        TNameBufsBase::EnumStringPair(EFeatureType::Float, "Float"sv),
        TNameBufsBase::EnumStringPair(EFeatureType::Text, "Text"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "Float"sv,
        "Categorical"sv,
        "Text"sv,
        "Embedding"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EFeatureType::"sv,
        "EFeatureType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EFeatureType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EFeatureType>;

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

const TString& ToString(EFeatureType x) {
    const NEFeatureTypePrivate::TNameBufs& names = NEFeatureTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EFeatureType FromStringImpl<EFeatureType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEFeatureTypePrivate::TNameBufs, EFeatureType>(data, len);
}

template<>
bool TryFromStringImpl<EFeatureType>(const char* data, size_t len, EFeatureType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEFeatureTypePrivate::TNameBufs, EFeatureType>(data, len, result);
}

bool FromString(const TString& name, EFeatureType& ret) {
    return ::TryFromStringImpl<EFeatureType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EFeatureType& ret) {
    return ::TryFromStringImpl<EFeatureType>(name.data(), name.size(), ret);
}

template<>
void Out<EFeatureType>(IOutputStream& os, TTypeTraits<EFeatureType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEFeatureTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EFeatureType>(EFeatureType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEFeatureTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EFeatureType> GetEnumAllValuesImpl<EFeatureType>() {
        const NEFeatureTypePrivate::TNameBufs& names = NEFeatureTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EFeatureType>() {
        const NEFeatureTypePrivate::TNameBufs& names = NEFeatureTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EFeatureType, TString> GetEnumNamesImpl<EFeatureType>() {
        const NEFeatureTypePrivate::TNameBufs& names = NEFeatureTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EFeatureType>() {
        const NEFeatureTypePrivate::TNameBufs& names = NEFeatureTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EEstimatedSourceFeatureType
namespace { namespace NEEstimatedSourceFeatureTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EEstimatedSourceFeatureType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EEstimatedSourceFeatureType::Text, "Text"sv),
        TNameBufsBase::EnumStringPair(EEstimatedSourceFeatureType::Embedding, "Embedding"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EEstimatedSourceFeatureType::Embedding, "Embedding"sv),
        TNameBufsBase::EnumStringPair(EEstimatedSourceFeatureType::Text, "Text"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Text"sv,
        "Embedding"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EEstimatedSourceFeatureType::"sv,
        "EEstimatedSourceFeatureType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EEstimatedSourceFeatureType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EEstimatedSourceFeatureType>;

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

const TString& ToString(EEstimatedSourceFeatureType x) {
    const NEEstimatedSourceFeatureTypePrivate::TNameBufs& names = NEEstimatedSourceFeatureTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EEstimatedSourceFeatureType FromStringImpl<EEstimatedSourceFeatureType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEEstimatedSourceFeatureTypePrivate::TNameBufs, EEstimatedSourceFeatureType>(data, len);
}

template<>
bool TryFromStringImpl<EEstimatedSourceFeatureType>(const char* data, size_t len, EEstimatedSourceFeatureType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEEstimatedSourceFeatureTypePrivate::TNameBufs, EEstimatedSourceFeatureType>(data, len, result);
}

bool FromString(const TString& name, EEstimatedSourceFeatureType& ret) {
    return ::TryFromStringImpl<EEstimatedSourceFeatureType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EEstimatedSourceFeatureType& ret) {
    return ::TryFromStringImpl<EEstimatedSourceFeatureType>(name.data(), name.size(), ret);
}

template<>
void Out<EEstimatedSourceFeatureType>(IOutputStream& os, TTypeTraits<EEstimatedSourceFeatureType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEEstimatedSourceFeatureTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EEstimatedSourceFeatureType>(EEstimatedSourceFeatureType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEEstimatedSourceFeatureTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EEstimatedSourceFeatureType> GetEnumAllValuesImpl<EEstimatedSourceFeatureType>() {
        const NEEstimatedSourceFeatureTypePrivate::TNameBufs& names = NEEstimatedSourceFeatureTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EEstimatedSourceFeatureType>() {
        const NEEstimatedSourceFeatureTypePrivate::TNameBufs& names = NEEstimatedSourceFeatureTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EEstimatedSourceFeatureType, TString> GetEnumNamesImpl<EEstimatedSourceFeatureType>() {
        const NEEstimatedSourceFeatureTypePrivate::TNameBufs& names = NEEstimatedSourceFeatureTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EEstimatedSourceFeatureType>() {
        const NEEstimatedSourceFeatureTypePrivate::TNameBufs& names = NEEstimatedSourceFeatureTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EErrorType
namespace { namespace NEErrorTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EErrorType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(PerObjectError, "PerObjectError"sv),
        TNameBufsBase::EnumStringPair(PairwiseError, "PairwiseError"sv),
        TNameBufsBase::EnumStringPair(QuerywiseError, "QuerywiseError"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(PairwiseError, "PairwiseError"sv),
        TNameBufsBase::EnumStringPair(PerObjectError, "PerObjectError"sv),
        TNameBufsBase::EnumStringPair(QuerywiseError, "QuerywiseError"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "PerObjectError"sv,
        "PairwiseError"sv,
        "QuerywiseError"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        ""sv,
        "EErrorType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EErrorType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EErrorType>;

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

const TString& ToString(EErrorType x) {
    const NEErrorTypePrivate::TNameBufs& names = NEErrorTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EErrorType FromStringImpl<EErrorType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEErrorTypePrivate::TNameBufs, EErrorType>(data, len);
}

template<>
bool TryFromStringImpl<EErrorType>(const char* data, size_t len, EErrorType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEErrorTypePrivate::TNameBufs, EErrorType>(data, len, result);
}

bool FromString(const TString& name, EErrorType& ret) {
    return ::TryFromStringImpl<EErrorType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EErrorType& ret) {
    return ::TryFromStringImpl<EErrorType>(name.data(), name.size(), ret);
}

template<>
void Out<EErrorType>(IOutputStream& os, TTypeTraits<EErrorType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEErrorTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EErrorType>(EErrorType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEErrorTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EErrorType> GetEnumAllValuesImpl<EErrorType>() {
        const NEErrorTypePrivate::TNameBufs& names = NEErrorTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EErrorType>() {
        const NEErrorTypePrivate::TNameBufs& names = NEErrorTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EErrorType, TString> GetEnumNamesImpl<EErrorType>() {
        const NEErrorTypePrivate::TNameBufs& names = NEErrorTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EErrorType>() {
        const NEErrorTypePrivate::TNameBufs& names = NEErrorTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ETaskType
namespace { namespace NETaskTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ETaskType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ETaskType::GPU, "GPU"sv),
        TNameBufsBase::EnumStringPair(ETaskType::CPU, "CPU"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ETaskType::CPU, "CPU"sv),
        TNameBufsBase::EnumStringPair(ETaskType::GPU, "GPU"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "GPU"sv,
        "CPU"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ETaskType::"sv,
        "ETaskType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ETaskType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ETaskType>;

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

const TString& ToString(ETaskType x) {
    const NETaskTypePrivate::TNameBufs& names = NETaskTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ETaskType FromStringImpl<ETaskType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NETaskTypePrivate::TNameBufs, ETaskType>(data, len);
}

template<>
bool TryFromStringImpl<ETaskType>(const char* data, size_t len, ETaskType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NETaskTypePrivate::TNameBufs, ETaskType>(data, len, result);
}

bool FromString(const TString& name, ETaskType& ret) {
    return ::TryFromStringImpl<ETaskType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ETaskType& ret) {
    return ::TryFromStringImpl<ETaskType>(name.data(), name.size(), ret);
}

template<>
void Out<ETaskType>(IOutputStream& os, TTypeTraits<ETaskType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NETaskTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ETaskType>(ETaskType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NETaskTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ETaskType> GetEnumAllValuesImpl<ETaskType>() {
        const NETaskTypePrivate::TNameBufs& names = NETaskTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ETaskType>() {
        const NETaskTypePrivate::TNameBufs& names = NETaskTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ETaskType, TString> GetEnumNamesImpl<ETaskType>() {
        const NETaskTypePrivate::TNameBufs& names = NETaskTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ETaskType>() {
        const NETaskTypePrivate::TNameBufs& names = NETaskTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EBoostingType
namespace { namespace NEBoostingTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EBoostingType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(Ordered, "Ordered"sv),
        TNameBufsBase::EnumStringPair(Plain, "Plain"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(Ordered, "Ordered"sv),
        TNameBufsBase::EnumStringPair(Plain, "Plain"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Ordered"sv,
        "Plain"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        ""sv,
        "EBoostingType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EBoostingType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EBoostingType>;

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

const TString& ToString(EBoostingType x) {
    const NEBoostingTypePrivate::TNameBufs& names = NEBoostingTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EBoostingType FromStringImpl<EBoostingType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEBoostingTypePrivate::TNameBufs, EBoostingType>(data, len);
}

template<>
bool TryFromStringImpl<EBoostingType>(const char* data, size_t len, EBoostingType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEBoostingTypePrivate::TNameBufs, EBoostingType>(data, len, result);
}

bool FromString(const TString& name, EBoostingType& ret) {
    return ::TryFromStringImpl<EBoostingType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EBoostingType& ret) {
    return ::TryFromStringImpl<EBoostingType>(name.data(), name.size(), ret);
}

template<>
void Out<EBoostingType>(IOutputStream& os, TTypeTraits<EBoostingType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEBoostingTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EBoostingType>(EBoostingType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEBoostingTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EBoostingType> GetEnumAllValuesImpl<EBoostingType>() {
        const NEBoostingTypePrivate::TNameBufs& names = NEBoostingTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EBoostingType>() {
        const NEBoostingTypePrivate::TNameBufs& names = NEBoostingTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EBoostingType, TString> GetEnumNamesImpl<EBoostingType>() {
        const NEBoostingTypePrivate::TNameBufs& names = NEBoostingTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EBoostingType>() {
        const NEBoostingTypePrivate::TNameBufs& names = NEBoostingTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EDataPartitionType
namespace { namespace NEDataPartitionTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EDataPartitionType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EDataPartitionType::FeatureParallel, "FeatureParallel"sv),
        TNameBufsBase::EnumStringPair(EDataPartitionType::DocParallel, "DocParallel"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EDataPartitionType::DocParallel, "DocParallel"sv),
        TNameBufsBase::EnumStringPair(EDataPartitionType::FeatureParallel, "FeatureParallel"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "FeatureParallel"sv,
        "DocParallel"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EDataPartitionType::"sv,
        "EDataPartitionType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EDataPartitionType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EDataPartitionType>;

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

const TString& ToString(EDataPartitionType x) {
    const NEDataPartitionTypePrivate::TNameBufs& names = NEDataPartitionTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EDataPartitionType FromStringImpl<EDataPartitionType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEDataPartitionTypePrivate::TNameBufs, EDataPartitionType>(data, len);
}

template<>
bool TryFromStringImpl<EDataPartitionType>(const char* data, size_t len, EDataPartitionType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEDataPartitionTypePrivate::TNameBufs, EDataPartitionType>(data, len, result);
}

bool FromString(const TString& name, EDataPartitionType& ret) {
    return ::TryFromStringImpl<EDataPartitionType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EDataPartitionType& ret) {
    return ::TryFromStringImpl<EDataPartitionType>(name.data(), name.size(), ret);
}

template<>
void Out<EDataPartitionType>(IOutputStream& os, TTypeTraits<EDataPartitionType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEDataPartitionTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EDataPartitionType>(EDataPartitionType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEDataPartitionTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EDataPartitionType> GetEnumAllValuesImpl<EDataPartitionType>() {
        const NEDataPartitionTypePrivate::TNameBufs& names = NEDataPartitionTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EDataPartitionType>() {
        const NEDataPartitionTypePrivate::TNameBufs& names = NEDataPartitionTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EDataPartitionType, TString> GetEnumNamesImpl<EDataPartitionType>() {
        const NEDataPartitionTypePrivate::TNameBufs& names = NEDataPartitionTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EDataPartitionType>() {
        const NEDataPartitionTypePrivate::TNameBufs& names = NEDataPartitionTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ELoadUnimplementedPolicy
namespace { namespace NELoadUnimplementedPolicyPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ELoadUnimplementedPolicy>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(ELoadUnimplementedPolicy::SkipWithWarning, "SkipWithWarning"sv),
        TNameBufsBase::EnumStringPair(ELoadUnimplementedPolicy::Exception, "Exception"sv),
        TNameBufsBase::EnumStringPair(ELoadUnimplementedPolicy::ExceptionOnChange, "ExceptionOnChange"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(ELoadUnimplementedPolicy::Exception, "Exception"sv),
        TNameBufsBase::EnumStringPair(ELoadUnimplementedPolicy::ExceptionOnChange, "ExceptionOnChange"sv),
        TNameBufsBase::EnumStringPair(ELoadUnimplementedPolicy::SkipWithWarning, "SkipWithWarning"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "SkipWithWarning"sv,
        "Exception"sv,
        "ExceptionOnChange"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ELoadUnimplementedPolicy::"sv,
        "ELoadUnimplementedPolicy"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ELoadUnimplementedPolicy> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ELoadUnimplementedPolicy>;

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

const TString& ToString(ELoadUnimplementedPolicy x) {
    const NELoadUnimplementedPolicyPrivate::TNameBufs& names = NELoadUnimplementedPolicyPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ELoadUnimplementedPolicy FromStringImpl<ELoadUnimplementedPolicy>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NELoadUnimplementedPolicyPrivate::TNameBufs, ELoadUnimplementedPolicy>(data, len);
}

template<>
bool TryFromStringImpl<ELoadUnimplementedPolicy>(const char* data, size_t len, ELoadUnimplementedPolicy& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NELoadUnimplementedPolicyPrivate::TNameBufs, ELoadUnimplementedPolicy>(data, len, result);
}

bool FromString(const TString& name, ELoadUnimplementedPolicy& ret) {
    return ::TryFromStringImpl<ELoadUnimplementedPolicy>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ELoadUnimplementedPolicy& ret) {
    return ::TryFromStringImpl<ELoadUnimplementedPolicy>(name.data(), name.size(), ret);
}

template<>
void Out<ELoadUnimplementedPolicy>(IOutputStream& os, TTypeTraits<ELoadUnimplementedPolicy>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NELoadUnimplementedPolicyPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ELoadUnimplementedPolicy>(ELoadUnimplementedPolicy e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NELoadUnimplementedPolicyPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ELoadUnimplementedPolicy> GetEnumAllValuesImpl<ELoadUnimplementedPolicy>() {
        const NELoadUnimplementedPolicyPrivate::TNameBufs& names = NELoadUnimplementedPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ELoadUnimplementedPolicy>() {
        const NELoadUnimplementedPolicyPrivate::TNameBufs& names = NELoadUnimplementedPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ELoadUnimplementedPolicy, TString> GetEnumNamesImpl<ELoadUnimplementedPolicy>() {
        const NELoadUnimplementedPolicyPrivate::TNameBufs& names = NELoadUnimplementedPolicyPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ELoadUnimplementedPolicy>() {
        const NELoadUnimplementedPolicyPrivate::TNameBufs& names = NELoadUnimplementedPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ELeavesEstimation
namespace { namespace NELeavesEstimationPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ELeavesEstimation>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(ELeavesEstimation::Gradient, "Gradient"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimation::Newton, "Newton"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimation::Exact, "Exact"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimation::Simple, "Simple"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(ELeavesEstimation::Exact, "Exact"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimation::Gradient, "Gradient"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimation::Newton, "Newton"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimation::Simple, "Simple"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "Gradient"sv,
        "Newton"sv,
        "Exact"sv,
        "Simple"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ELeavesEstimation::"sv,
        "ELeavesEstimation"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ELeavesEstimation> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ELeavesEstimation>;

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

const TString& ToString(ELeavesEstimation x) {
    const NELeavesEstimationPrivate::TNameBufs& names = NELeavesEstimationPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ELeavesEstimation FromStringImpl<ELeavesEstimation>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NELeavesEstimationPrivate::TNameBufs, ELeavesEstimation>(data, len);
}

template<>
bool TryFromStringImpl<ELeavesEstimation>(const char* data, size_t len, ELeavesEstimation& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NELeavesEstimationPrivate::TNameBufs, ELeavesEstimation>(data, len, result);
}

bool FromString(const TString& name, ELeavesEstimation& ret) {
    return ::TryFromStringImpl<ELeavesEstimation>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ELeavesEstimation& ret) {
    return ::TryFromStringImpl<ELeavesEstimation>(name.data(), name.size(), ret);
}

template<>
void Out<ELeavesEstimation>(IOutputStream& os, TTypeTraits<ELeavesEstimation>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NELeavesEstimationPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ELeavesEstimation>(ELeavesEstimation e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NELeavesEstimationPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ELeavesEstimation> GetEnumAllValuesImpl<ELeavesEstimation>() {
        const NELeavesEstimationPrivate::TNameBufs& names = NELeavesEstimationPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ELeavesEstimation>() {
        const NELeavesEstimationPrivate::TNameBufs& names = NELeavesEstimationPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ELeavesEstimation, TString> GetEnumNamesImpl<ELeavesEstimation>() {
        const NELeavesEstimationPrivate::TNameBufs& names = NELeavesEstimationPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ELeavesEstimation>() {
        const NELeavesEstimationPrivate::TNameBufs& names = NELeavesEstimationPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EScoreFunction
namespace { namespace NEScoreFunctionPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EScoreFunction>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 7> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 7>{{
        TNameBufsBase::EnumStringPair(EScoreFunction::SolarL2, "SolarL2"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::Cosine, "Cosine"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::NewtonL2, "NewtonL2"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::NewtonCosine, "NewtonCosine"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::LOOL2, "LOOL2"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::SatL2, "SatL2"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::L2, "L2"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[7]{
        TNameBufsBase::EnumStringPair(EScoreFunction::Cosine, "Cosine"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::L2, "L2"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::LOOL2, "LOOL2"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::NewtonCosine, "NewtonCosine"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::NewtonL2, "NewtonL2"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::SatL2, "SatL2"sv),
        TNameBufsBase::EnumStringPair(EScoreFunction::SolarL2, "SolarL2"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[7]{
        "SolarL2"sv,
        "Cosine"sv,
        "NewtonL2"sv,
        "NewtonCosine"sv,
        "LOOL2"sv,
        "SatL2"sv,
        "L2"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EScoreFunction::"sv,
        "EScoreFunction"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EScoreFunction> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EScoreFunction>;

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

const TString& ToString(EScoreFunction x) {
    const NEScoreFunctionPrivate::TNameBufs& names = NEScoreFunctionPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EScoreFunction FromStringImpl<EScoreFunction>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEScoreFunctionPrivate::TNameBufs, EScoreFunction>(data, len);
}

template<>
bool TryFromStringImpl<EScoreFunction>(const char* data, size_t len, EScoreFunction& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEScoreFunctionPrivate::TNameBufs, EScoreFunction>(data, len, result);
}

bool FromString(const TString& name, EScoreFunction& ret) {
    return ::TryFromStringImpl<EScoreFunction>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EScoreFunction& ret) {
    return ::TryFromStringImpl<EScoreFunction>(name.data(), name.size(), ret);
}

template<>
void Out<EScoreFunction>(IOutputStream& os, TTypeTraits<EScoreFunction>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEScoreFunctionPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EScoreFunction>(EScoreFunction e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEScoreFunctionPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EScoreFunction> GetEnumAllValuesImpl<EScoreFunction>() {
        const NEScoreFunctionPrivate::TNameBufs& names = NEScoreFunctionPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EScoreFunction>() {
        const NEScoreFunctionPrivate::TNameBufs& names = NEScoreFunctionPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EScoreFunction, TString> GetEnumNamesImpl<EScoreFunction>() {
        const NEScoreFunctionPrivate::TNameBufs& names = NEScoreFunctionPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EScoreFunction>() {
        const NEScoreFunctionPrivate::TNameBufs& names = NEScoreFunctionPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ERandomScoreType
namespace { namespace NERandomScoreTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ERandomScoreType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ERandomScoreType::NormalWithModelSizeDecrease, "NormalWithModelSizeDecrease"sv),
        TNameBufsBase::EnumStringPair(ERandomScoreType::Gumbel, "Gumbel"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ERandomScoreType::Gumbel, "Gumbel"sv),
        TNameBufsBase::EnumStringPair(ERandomScoreType::NormalWithModelSizeDecrease, "NormalWithModelSizeDecrease"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "NormalWithModelSizeDecrease"sv,
        "Gumbel"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ERandomScoreType::"sv,
        "ERandomScoreType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ERandomScoreType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ERandomScoreType>;

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

const TString& ToString(ERandomScoreType x) {
    const NERandomScoreTypePrivate::TNameBufs& names = NERandomScoreTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ERandomScoreType FromStringImpl<ERandomScoreType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NERandomScoreTypePrivate::TNameBufs, ERandomScoreType>(data, len);
}

template<>
bool TryFromStringImpl<ERandomScoreType>(const char* data, size_t len, ERandomScoreType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NERandomScoreTypePrivate::TNameBufs, ERandomScoreType>(data, len, result);
}

bool FromString(const TString& name, ERandomScoreType& ret) {
    return ::TryFromStringImpl<ERandomScoreType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ERandomScoreType& ret) {
    return ::TryFromStringImpl<ERandomScoreType>(name.data(), name.size(), ret);
}

template<>
void Out<ERandomScoreType>(IOutputStream& os, TTypeTraits<ERandomScoreType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NERandomScoreTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ERandomScoreType>(ERandomScoreType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NERandomScoreTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ERandomScoreType> GetEnumAllValuesImpl<ERandomScoreType>() {
        const NERandomScoreTypePrivate::TNameBufs& names = NERandomScoreTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ERandomScoreType>() {
        const NERandomScoreTypePrivate::TNameBufs& names = NERandomScoreTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ERandomScoreType, TString> GetEnumNamesImpl<ERandomScoreType>() {
        const NERandomScoreTypePrivate::TNameBufs& names = NERandomScoreTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ERandomScoreType>() {
        const NERandomScoreTypePrivate::TNameBufs& names = NERandomScoreTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EModelShrinkMode
namespace { namespace NEModelShrinkModePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EModelShrinkMode>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EModelShrinkMode::Constant, "Constant"sv),
        TNameBufsBase::EnumStringPair(EModelShrinkMode::Decreasing, "Decreasing"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EModelShrinkMode::Constant, "Constant"sv),
        TNameBufsBase::EnumStringPair(EModelShrinkMode::Decreasing, "Decreasing"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Constant"sv,
        "Decreasing"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EModelShrinkMode::"sv,
        "EModelShrinkMode"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EModelShrinkMode> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EModelShrinkMode>;

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

const TString& ToString(EModelShrinkMode x) {
    const NEModelShrinkModePrivate::TNameBufs& names = NEModelShrinkModePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EModelShrinkMode FromStringImpl<EModelShrinkMode>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEModelShrinkModePrivate::TNameBufs, EModelShrinkMode>(data, len);
}

template<>
bool TryFromStringImpl<EModelShrinkMode>(const char* data, size_t len, EModelShrinkMode& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEModelShrinkModePrivate::TNameBufs, EModelShrinkMode>(data, len, result);
}

bool FromString(const TString& name, EModelShrinkMode& ret) {
    return ::TryFromStringImpl<EModelShrinkMode>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EModelShrinkMode& ret) {
    return ::TryFromStringImpl<EModelShrinkMode>(name.data(), name.size(), ret);
}

template<>
void Out<EModelShrinkMode>(IOutputStream& os, TTypeTraits<EModelShrinkMode>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEModelShrinkModePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EModelShrinkMode>(EModelShrinkMode e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEModelShrinkModePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EModelShrinkMode> GetEnumAllValuesImpl<EModelShrinkMode>() {
        const NEModelShrinkModePrivate::TNameBufs& names = NEModelShrinkModePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EModelShrinkMode>() {
        const NEModelShrinkModePrivate::TNameBufs& names = NEModelShrinkModePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EModelShrinkMode, TString> GetEnumNamesImpl<EModelShrinkMode>() {
        const NEModelShrinkModePrivate::TNameBufs& names = NEModelShrinkModePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EModelShrinkMode>() {
        const NEModelShrinkModePrivate::TNameBufs& names = NEModelShrinkModePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EBootstrapType
namespace { namespace NEBootstrapTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EBootstrapType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 5> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 5>{{
        TNameBufsBase::EnumStringPair(EBootstrapType::Poisson, "Poisson"sv),
        TNameBufsBase::EnumStringPair(EBootstrapType::Bayesian, "Bayesian"sv),
        TNameBufsBase::EnumStringPair(EBootstrapType::Bernoulli, "Bernoulli"sv),
        TNameBufsBase::EnumStringPair(EBootstrapType::MVS, "MVS"sv),
        TNameBufsBase::EnumStringPair(EBootstrapType::No, "No"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[5]{
        TNameBufsBase::EnumStringPair(EBootstrapType::Bayesian, "Bayesian"sv),
        TNameBufsBase::EnumStringPair(EBootstrapType::Bernoulli, "Bernoulli"sv),
        TNameBufsBase::EnumStringPair(EBootstrapType::MVS, "MVS"sv),
        TNameBufsBase::EnumStringPair(EBootstrapType::No, "No"sv),
        TNameBufsBase::EnumStringPair(EBootstrapType::Poisson, "Poisson"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[5]{
        "Poisson"sv,
        "Bayesian"sv,
        "Bernoulli"sv,
        "MVS"sv,
        "No"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EBootstrapType::"sv,
        "EBootstrapType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EBootstrapType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EBootstrapType>;

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

const TString& ToString(EBootstrapType x) {
    const NEBootstrapTypePrivate::TNameBufs& names = NEBootstrapTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EBootstrapType FromStringImpl<EBootstrapType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEBootstrapTypePrivate::TNameBufs, EBootstrapType>(data, len);
}

template<>
bool TryFromStringImpl<EBootstrapType>(const char* data, size_t len, EBootstrapType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEBootstrapTypePrivate::TNameBufs, EBootstrapType>(data, len, result);
}

bool FromString(const TString& name, EBootstrapType& ret) {
    return ::TryFromStringImpl<EBootstrapType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EBootstrapType& ret) {
    return ::TryFromStringImpl<EBootstrapType>(name.data(), name.size(), ret);
}

template<>
void Out<EBootstrapType>(IOutputStream& os, TTypeTraits<EBootstrapType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEBootstrapTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EBootstrapType>(EBootstrapType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEBootstrapTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EBootstrapType> GetEnumAllValuesImpl<EBootstrapType>() {
        const NEBootstrapTypePrivate::TNameBufs& names = NEBootstrapTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EBootstrapType>() {
        const NEBootstrapTypePrivate::TNameBufs& names = NEBootstrapTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EBootstrapType, TString> GetEnumNamesImpl<EBootstrapType>() {
        const NEBootstrapTypePrivate::TNameBufs& names = NEBootstrapTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EBootstrapType>() {
        const NEBootstrapTypePrivate::TNameBufs& names = NEBootstrapTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EGrowPolicy
namespace { namespace NEGrowPolicyPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EGrowPolicy>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(EGrowPolicy::SymmetricTree, "SymmetricTree"sv),
        TNameBufsBase::EnumStringPair(EGrowPolicy::Lossguide, "Lossguide"sv),
        TNameBufsBase::EnumStringPair(EGrowPolicy::Depthwise, "Depthwise"sv),
        TNameBufsBase::EnumStringPair(EGrowPolicy::Region, "Region"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(EGrowPolicy::Depthwise, "Depthwise"sv),
        TNameBufsBase::EnumStringPair(EGrowPolicy::Lossguide, "Lossguide"sv),
        TNameBufsBase::EnumStringPair(EGrowPolicy::Region, "Region"sv),
        TNameBufsBase::EnumStringPair(EGrowPolicy::SymmetricTree, "SymmetricTree"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "SymmetricTree"sv,
        "Lossguide"sv,
        "Depthwise"sv,
        "Region"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EGrowPolicy::"sv,
        "EGrowPolicy"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EGrowPolicy> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EGrowPolicy>;

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

const TString& ToString(EGrowPolicy x) {
    const NEGrowPolicyPrivate::TNameBufs& names = NEGrowPolicyPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EGrowPolicy FromStringImpl<EGrowPolicy>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEGrowPolicyPrivate::TNameBufs, EGrowPolicy>(data, len);
}

template<>
bool TryFromStringImpl<EGrowPolicy>(const char* data, size_t len, EGrowPolicy& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEGrowPolicyPrivate::TNameBufs, EGrowPolicy>(data, len, result);
}

bool FromString(const TString& name, EGrowPolicy& ret) {
    return ::TryFromStringImpl<EGrowPolicy>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EGrowPolicy& ret) {
    return ::TryFromStringImpl<EGrowPolicy>(name.data(), name.size(), ret);
}

template<>
void Out<EGrowPolicy>(IOutputStream& os, TTypeTraits<EGrowPolicy>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEGrowPolicyPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EGrowPolicy>(EGrowPolicy e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEGrowPolicyPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EGrowPolicy> GetEnumAllValuesImpl<EGrowPolicy>() {
        const NEGrowPolicyPrivate::TNameBufs& names = NEGrowPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EGrowPolicy>() {
        const NEGrowPolicyPrivate::TNameBufs& names = NEGrowPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EGrowPolicy, TString> GetEnumNamesImpl<EGrowPolicy>() {
        const NEGrowPolicyPrivate::TNameBufs& names = NEGrowPolicyPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EGrowPolicy>() {
        const NEGrowPolicyPrivate::TNameBufs& names = NEGrowPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ENanMode
namespace { namespace NENanModePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ENanMode>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(ENanMode::Min, "Min"sv),
        TNameBufsBase::EnumStringPair(ENanMode::Max, "Max"sv),
        TNameBufsBase::EnumStringPair(ENanMode::Forbidden, "Forbidden"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(ENanMode::Forbidden, "Forbidden"sv),
        TNameBufsBase::EnumStringPair(ENanMode::Max, "Max"sv),
        TNameBufsBase::EnumStringPair(ENanMode::Min, "Min"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Min"sv,
        "Max"sv,
        "Forbidden"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ENanMode::"sv,
        "ENanMode"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ENanMode> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ENanMode>;

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

const TString& ToString(ENanMode x) {
    const NENanModePrivate::TNameBufs& names = NENanModePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ENanMode FromStringImpl<ENanMode>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NENanModePrivate::TNameBufs, ENanMode>(data, len);
}

template<>
bool TryFromStringImpl<ENanMode>(const char* data, size_t len, ENanMode& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NENanModePrivate::TNameBufs, ENanMode>(data, len, result);
}

bool FromString(const TString& name, ENanMode& ret) {
    return ::TryFromStringImpl<ENanMode>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ENanMode& ret) {
    return ::TryFromStringImpl<ENanMode>(name.data(), name.size(), ret);
}

template<>
void Out<ENanMode>(IOutputStream& os, TTypeTraits<ENanMode>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NENanModePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ENanMode>(ENanMode e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NENanModePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ENanMode> GetEnumAllValuesImpl<ENanMode>() {
        const NENanModePrivate::TNameBufs& names = NENanModePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ENanMode>() {
        const NENanModePrivate::TNameBufs& names = NENanModePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ENanMode, TString> GetEnumNamesImpl<ENanMode>() {
        const NENanModePrivate::TNameBufs& names = NENanModePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ENanMode>() {
        const NENanModePrivate::TNameBufs& names = NENanModePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ECrossValidation
namespace { namespace NECrossValidationPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ECrossValidation>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(ECrossValidation::Classical, "Classical"sv),
        TNameBufsBase::EnumStringPair(ECrossValidation::Inverted, "Inverted"sv),
        TNameBufsBase::EnumStringPair(ECrossValidation::TimeSeries, "TimeSeries"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(ECrossValidation::Classical, "Classical"sv),
        TNameBufsBase::EnumStringPair(ECrossValidation::Inverted, "Inverted"sv),
        TNameBufsBase::EnumStringPair(ECrossValidation::TimeSeries, "TimeSeries"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Classical"sv,
        "Inverted"sv,
        "TimeSeries"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ECrossValidation::"sv,
        "ECrossValidation"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ECrossValidation> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ECrossValidation>;

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

const TString& ToString(ECrossValidation x) {
    const NECrossValidationPrivate::TNameBufs& names = NECrossValidationPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ECrossValidation FromStringImpl<ECrossValidation>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NECrossValidationPrivate::TNameBufs, ECrossValidation>(data, len);
}

template<>
bool TryFromStringImpl<ECrossValidation>(const char* data, size_t len, ECrossValidation& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NECrossValidationPrivate::TNameBufs, ECrossValidation>(data, len, result);
}

bool FromString(const TString& name, ECrossValidation& ret) {
    return ::TryFromStringImpl<ECrossValidation>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ECrossValidation& ret) {
    return ::TryFromStringImpl<ECrossValidation>(name.data(), name.size(), ret);
}

template<>
void Out<ECrossValidation>(IOutputStream& os, TTypeTraits<ECrossValidation>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NECrossValidationPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ECrossValidation>(ECrossValidation e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NECrossValidationPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ECrossValidation> GetEnumAllValuesImpl<ECrossValidation>() {
        const NECrossValidationPrivate::TNameBufs& names = NECrossValidationPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ECrossValidation>() {
        const NECrossValidationPrivate::TNameBufs& names = NECrossValidationPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ECrossValidation, TString> GetEnumNamesImpl<ECrossValidation>() {
        const NECrossValidationPrivate::TNameBufs& names = NECrossValidationPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ECrossValidation>() {
        const NECrossValidationPrivate::TNameBufs& names = NECrossValidationPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ELossFunction
namespace { namespace NELossFunctionPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ELossFunction>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 79> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 79>{{
        TNameBufsBase::EnumStringPair(ELossFunction::Logloss, "Logloss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::CrossEntropy, "CrossEntropy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::CtrFactor, "CtrFactor"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Focal, "Focal"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::RMSE, "RMSE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::LogCosh, "LogCosh"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Lq, "Lq"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MAE, "MAE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Quantile, "Quantile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiQuantile, "MultiQuantile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Expectile, "Expectile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::LogLinQuantile, "LogLinQuantile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MAPE, "MAPE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Poisson, "Poisson"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MSLE, "MSLE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MedianAbsoluteError, "MedianAbsoluteError"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::SMAPE, "SMAPE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Huber, "Huber"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Tweedie, "Tweedie"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Cox, "Cox"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::RMSEWithUncertainty, "RMSEWithUncertainty"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiClass, "MultiClass"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiClassOneVsAll, "MultiClassOneVsAll"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PairLogit, "PairLogit"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PairLogitPairwise, "PairLogitPairwise"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::YetiRank, "YetiRank"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::YetiRankPairwise, "YetiRankPairwise"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QueryRMSE, "QueryRMSE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::GroupQuantile, "GroupQuantile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QuerySoftMax, "QuerySoftMax"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QueryCrossEntropy, "QueryCrossEntropy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::StochasticFilter, "StochasticFilter"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::LambdaMart, "LambdaMart"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::StochasticRank, "StochasticRank"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PythonUserDefinedPerObject, "PythonUserDefinedPerObject"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PythonUserDefinedMultiTarget, "PythonUserDefinedMultiTarget"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::UserPerObjMetric, "UserPerObjMetric"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::UserQuerywiseMetric, "UserQuerywiseMetric"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::R2, "R2"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::NumErrors, "NumErrors"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::FairLoss, "FairLoss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::AUC, "AUC"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Accuracy, "Accuracy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::BalancedAccuracy, "BalancedAccuracy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::BalancedErrorRate, "BalancedErrorRate"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::BrierScore, "BrierScore"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Precision, "Precision"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Recall, "Recall"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::F1, "F1"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::TotalF1, "TotalF1"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::F, "F"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MCC, "MCC"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::ZeroOneLoss, "ZeroOneLoss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::HammingLoss, "HammingLoss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::HingeLoss, "HingeLoss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Kappa, "Kappa"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::WKappa, "WKappa"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::LogLikelihoodOfPrediction, "LogLikelihoodOfPrediction"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::NormalizedGini, "NormalizedGini"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PRAUC, "PRAUC"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PairAccuracy, "PairAccuracy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::AverageGain, "AverageGain"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QueryAverage, "QueryAverage"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QueryAUC, "QueryAUC"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PFound, "PFound"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PrecisionAt, "PrecisionAt"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::RecallAt, "RecallAt"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MAP, "MAP"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::NDCG, "NDCG"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::DCG, "DCG"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::FilteredDCG, "FilteredDCG"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MRR, "MRR"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::ERR, "ERR"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::SurvivalAft, "SurvivalAft"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiRMSE, "MultiRMSE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiRMSEWithMissingValues, "MultiRMSEWithMissingValues"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiLogloss, "MultiLogloss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiCrossEntropy, "MultiCrossEntropy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Combination, "Combination"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[79]{
        TNameBufsBase::EnumStringPair(ELossFunction::AUC, "AUC"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Accuracy, "Accuracy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::AverageGain, "AverageGain"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::BalancedAccuracy, "BalancedAccuracy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::BalancedErrorRate, "BalancedErrorRate"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::BrierScore, "BrierScore"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Combination, "Combination"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Cox, "Cox"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::CrossEntropy, "CrossEntropy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::CtrFactor, "CtrFactor"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::DCG, "DCG"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::ERR, "ERR"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Expectile, "Expectile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::F, "F"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::F1, "F1"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::FairLoss, "FairLoss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::FilteredDCG, "FilteredDCG"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Focal, "Focal"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::GroupQuantile, "GroupQuantile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::HammingLoss, "HammingLoss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::HingeLoss, "HingeLoss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Huber, "Huber"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Kappa, "Kappa"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::LambdaMart, "LambdaMart"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::LogCosh, "LogCosh"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::LogLikelihoodOfPrediction, "LogLikelihoodOfPrediction"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::LogLinQuantile, "LogLinQuantile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Logloss, "Logloss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Lq, "Lq"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MAE, "MAE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MAP, "MAP"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MAPE, "MAPE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MCC, "MCC"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MRR, "MRR"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MSLE, "MSLE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MedianAbsoluteError, "MedianAbsoluteError"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiClass, "MultiClass"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiClassOneVsAll, "MultiClassOneVsAll"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiCrossEntropy, "MultiCrossEntropy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiLogloss, "MultiLogloss"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiQuantile, "MultiQuantile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiRMSE, "MultiRMSE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::MultiRMSEWithMissingValues, "MultiRMSEWithMissingValues"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::NDCG, "NDCG"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::NormalizedGini, "NormalizedGini"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::NumErrors, "NumErrors"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PFound, "PFound"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PRAUC, "PRAUC"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PairAccuracy, "PairAccuracy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PairLogit, "PairLogit"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PairLogitPairwise, "PairLogitPairwise"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Poisson, "Poisson"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Precision, "Precision"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PrecisionAt, "PrecisionAt"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PythonUserDefinedMultiTarget, "PythonUserDefinedMultiTarget"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::PythonUserDefinedPerObject, "PythonUserDefinedPerObject"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Quantile, "Quantile"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QueryAUC, "QueryAUC"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QueryAverage, "QueryAverage"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QueryCrossEntropy, "QueryCrossEntropy"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QueryRMSE, "QueryRMSE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::QuerySoftMax, "QuerySoftMax"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::R2, "R2"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::RMSE, "RMSE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::RMSEWithUncertainty, "RMSEWithUncertainty"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Recall, "Recall"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::RecallAt, "RecallAt"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::SMAPE, "SMAPE"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::StochasticFilter, "StochasticFilter"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::StochasticRank, "StochasticRank"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::SurvivalAft, "SurvivalAft"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::TotalF1, "TotalF1"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::Tweedie, "Tweedie"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::UserPerObjMetric, "UserPerObjMetric"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::UserQuerywiseMetric, "UserQuerywiseMetric"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::WKappa, "WKappa"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::YetiRank, "YetiRank"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::YetiRankPairwise, "YetiRankPairwise"sv),
        TNameBufsBase::EnumStringPair(ELossFunction::ZeroOneLoss, "ZeroOneLoss"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[79]{
        "Logloss"sv,
        "CrossEntropy"sv,
        "CtrFactor"sv,
        "Focal"sv,
        "RMSE"sv,
        "LogCosh"sv,
        "Lq"sv,
        "MAE"sv,
        "Quantile"sv,
        "MultiQuantile"sv,
        "Expectile"sv,
        "LogLinQuantile"sv,
        "MAPE"sv,
        "Poisson"sv,
        "MSLE"sv,
        "MedianAbsoluteError"sv,
        "SMAPE"sv,
        "Huber"sv,
        "Tweedie"sv,
        "Cox"sv,
        "RMSEWithUncertainty"sv,
        "MultiClass"sv,
        "MultiClassOneVsAll"sv,
        "PairLogit"sv,
        "PairLogitPairwise"sv,
        "YetiRank"sv,
        "YetiRankPairwise"sv,
        "QueryRMSE"sv,
        "GroupQuantile"sv,
        "QuerySoftMax"sv,
        "QueryCrossEntropy"sv,
        "StochasticFilter"sv,
        "LambdaMart"sv,
        "StochasticRank"sv,
        "PythonUserDefinedPerObject"sv,
        "PythonUserDefinedMultiTarget"sv,
        "UserPerObjMetric"sv,
        "UserQuerywiseMetric"sv,
        "R2"sv,
        "NumErrors"sv,
        "FairLoss"sv,
        "AUC"sv,
        "Accuracy"sv,
        "BalancedAccuracy"sv,
        "BalancedErrorRate"sv,
        "BrierScore"sv,
        "Precision"sv,
        "Recall"sv,
        "F1"sv,
        "TotalF1"sv,
        "F"sv,
        "MCC"sv,
        "ZeroOneLoss"sv,
        "HammingLoss"sv,
        "HingeLoss"sv,
        "Kappa"sv,
        "WKappa"sv,
        "LogLikelihoodOfPrediction"sv,
        "NormalizedGini"sv,
        "PRAUC"sv,
        "PairAccuracy"sv,
        "AverageGain"sv,
        "QueryAverage"sv,
        "QueryAUC"sv,
        "PFound"sv,
        "PrecisionAt"sv,
        "RecallAt"sv,
        "MAP"sv,
        "NDCG"sv,
        "DCG"sv,
        "FilteredDCG"sv,
        "MRR"sv,
        "ERR"sv,
        "SurvivalAft"sv,
        "MultiRMSE"sv,
        "MultiRMSEWithMissingValues"sv,
        "MultiLogloss"sv,
        "MultiCrossEntropy"sv,
        "Combination"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ELossFunction::"sv,
        "ELossFunction"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ELossFunction> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ELossFunction>;

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

const TString& ToString(ELossFunction x) {
    const NELossFunctionPrivate::TNameBufs& names = NELossFunctionPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ELossFunction FromStringImpl<ELossFunction>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NELossFunctionPrivate::TNameBufs, ELossFunction>(data, len);
}

template<>
bool TryFromStringImpl<ELossFunction>(const char* data, size_t len, ELossFunction& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NELossFunctionPrivate::TNameBufs, ELossFunction>(data, len, result);
}

bool FromString(const TString& name, ELossFunction& ret) {
    return ::TryFromStringImpl<ELossFunction>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ELossFunction& ret) {
    return ::TryFromStringImpl<ELossFunction>(name.data(), name.size(), ret);
}

template<>
void Out<ELossFunction>(IOutputStream& os, TTypeTraits<ELossFunction>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NELossFunctionPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ELossFunction>(ELossFunction e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NELossFunctionPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ELossFunction> GetEnumAllValuesImpl<ELossFunction>() {
        const NELossFunctionPrivate::TNameBufs& names = NELossFunctionPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ELossFunction>() {
        const NELossFunctionPrivate::TNameBufs& names = NELossFunctionPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ELossFunction, TString> GetEnumNamesImpl<ELossFunction>() {
        const NELossFunctionPrivate::TNameBufs& names = NELossFunctionPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ELossFunction>() {
        const NELossFunctionPrivate::TNameBufs& names = NELossFunctionPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ERankingType
namespace { namespace NERankingTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ERankingType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(ERankingType::CrossEntropy, "CrossEntropy"sv),
        TNameBufsBase::EnumStringPair(ERankingType::AbsoluteValue, "AbsoluteValue"sv),
        TNameBufsBase::EnumStringPair(ERankingType::Order, "Order"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(ERankingType::AbsoluteValue, "AbsoluteValue"sv),
        TNameBufsBase::EnumStringPair(ERankingType::CrossEntropy, "CrossEntropy"sv),
        TNameBufsBase::EnumStringPair(ERankingType::Order, "Order"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "CrossEntropy"sv,
        "AbsoluteValue"sv,
        "Order"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ERankingType::"sv,
        "ERankingType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ERankingType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ERankingType>;

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

const TString& ToString(ERankingType x) {
    const NERankingTypePrivate::TNameBufs& names = NERankingTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ERankingType FromStringImpl<ERankingType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NERankingTypePrivate::TNameBufs, ERankingType>(data, len);
}

template<>
bool TryFromStringImpl<ERankingType>(const char* data, size_t len, ERankingType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NERankingTypePrivate::TNameBufs, ERankingType>(data, len, result);
}

bool FromString(const TString& name, ERankingType& ret) {
    return ::TryFromStringImpl<ERankingType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ERankingType& ret) {
    return ::TryFromStringImpl<ERankingType>(name.data(), name.size(), ret);
}

template<>
void Out<ERankingType>(IOutputStream& os, TTypeTraits<ERankingType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NERankingTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ERankingType>(ERankingType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NERankingTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ERankingType> GetEnumAllValuesImpl<ERankingType>() {
        const NERankingTypePrivate::TNameBufs& names = NERankingTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ERankingType>() {
        const NERankingTypePrivate::TNameBufs& names = NERankingTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ERankingType, TString> GetEnumNamesImpl<ERankingType>() {
        const NERankingTypePrivate::TNameBufs& names = NERankingTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ERankingType>() {
        const NERankingTypePrivate::TNameBufs& names = NERankingTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EHessianType
namespace { namespace NEHessianTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EHessianType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EHessianType::Symmetric, "Symmetric"sv),
        TNameBufsBase::EnumStringPair(EHessianType::Diagonal, "Diagonal"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EHessianType::Diagonal, "Diagonal"sv),
        TNameBufsBase::EnumStringPair(EHessianType::Symmetric, "Symmetric"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Symmetric"sv,
        "Diagonal"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EHessianType::"sv,
        "EHessianType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EHessianType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EHessianType>;

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

const TString& ToString(EHessianType x) {
    const NEHessianTypePrivate::TNameBufs& names = NEHessianTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EHessianType FromStringImpl<EHessianType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEHessianTypePrivate::TNameBufs, EHessianType>(data, len);
}

template<>
bool TryFromStringImpl<EHessianType>(const char* data, size_t len, EHessianType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEHessianTypePrivate::TNameBufs, EHessianType>(data, len, result);
}

bool FromString(const TString& name, EHessianType& ret) {
    return ::TryFromStringImpl<EHessianType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EHessianType& ret) {
    return ::TryFromStringImpl<EHessianType>(name.data(), name.size(), ret);
}

template<>
void Out<EHessianType>(IOutputStream& os, TTypeTraits<EHessianType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEHessianTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EHessianType>(EHessianType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEHessianTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EHessianType> GetEnumAllValuesImpl<EHessianType>() {
        const NEHessianTypePrivate::TNameBufs& names = NEHessianTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EHessianType>() {
        const NEHessianTypePrivate::TNameBufs& names = NEHessianTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EHessianType, TString> GetEnumNamesImpl<EHessianType>() {
        const NEHessianTypePrivate::TNameBufs& names = NEHessianTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EHessianType>() {
        const NEHessianTypePrivate::TNameBufs& names = NEHessianTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ECounterCalc
namespace { namespace NECounterCalcPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ECounterCalc>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ECounterCalc::Full, "Full"sv),
        TNameBufsBase::EnumStringPair(ECounterCalc::SkipTest, "SkipTest"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ECounterCalc::Full, "Full"sv),
        TNameBufsBase::EnumStringPair(ECounterCalc::SkipTest, "SkipTest"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Full"sv,
        "SkipTest"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ECounterCalc::"sv,
        "ECounterCalc"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ECounterCalc> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ECounterCalc>;

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

const TString& ToString(ECounterCalc x) {
    const NECounterCalcPrivate::TNameBufs& names = NECounterCalcPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ECounterCalc FromStringImpl<ECounterCalc>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NECounterCalcPrivate::TNameBufs, ECounterCalc>(data, len);
}

template<>
bool TryFromStringImpl<ECounterCalc>(const char* data, size_t len, ECounterCalc& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NECounterCalcPrivate::TNameBufs, ECounterCalc>(data, len, result);
}

bool FromString(const TString& name, ECounterCalc& ret) {
    return ::TryFromStringImpl<ECounterCalc>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ECounterCalc& ret) {
    return ::TryFromStringImpl<ECounterCalc>(name.data(), name.size(), ret);
}

template<>
void Out<ECounterCalc>(IOutputStream& os, TTypeTraits<ECounterCalc>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NECounterCalcPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ECounterCalc>(ECounterCalc e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NECounterCalcPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ECounterCalc> GetEnumAllValuesImpl<ECounterCalc>() {
        const NECounterCalcPrivate::TNameBufs& names = NECounterCalcPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ECounterCalc>() {
        const NECounterCalcPrivate::TNameBufs& names = NECounterCalcPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ECounterCalc, TString> GetEnumNamesImpl<ECounterCalc>() {
        const NECounterCalcPrivate::TNameBufs& names = NECounterCalcPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ECounterCalc>() {
        const NECounterCalcPrivate::TNameBufs& names = NECounterCalcPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EPredictionType
namespace { namespace NEPredictionTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EPredictionType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 9> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 9>{{
        TNameBufsBase::EnumStringPair(EPredictionType::Probability, "Probability"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::LogProbability, "LogProbability"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::Class, "Class"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::RawFormulaVal, "RawFormulaVal"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::Exponent, "Exponent"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::RMSEWithUncertainty, "RMSEWithUncertainty"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::InternalRawFormulaVal, "InternalRawFormulaVal"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::VirtEnsembles, "VirtEnsembles"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::TotalUncertainty, "TotalUncertainty"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[9]{
        TNameBufsBase::EnumStringPair(EPredictionType::Class, "Class"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::Exponent, "Exponent"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::InternalRawFormulaVal, "InternalRawFormulaVal"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::LogProbability, "LogProbability"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::Probability, "Probability"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::RMSEWithUncertainty, "RMSEWithUncertainty"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::RawFormulaVal, "RawFormulaVal"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::TotalUncertainty, "TotalUncertainty"sv),
        TNameBufsBase::EnumStringPair(EPredictionType::VirtEnsembles, "VirtEnsembles"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[9]{
        "Probability"sv,
        "LogProbability"sv,
        "Class"sv,
        "RawFormulaVal"sv,
        "Exponent"sv,
        "RMSEWithUncertainty"sv,
        "InternalRawFormulaVal"sv,
        "VirtEnsembles"sv,
        "TotalUncertainty"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EPredictionType::"sv,
        "EPredictionType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EPredictionType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EPredictionType>;

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

const TString& ToString(EPredictionType x) {
    const NEPredictionTypePrivate::TNameBufs& names = NEPredictionTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EPredictionType FromStringImpl<EPredictionType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEPredictionTypePrivate::TNameBufs, EPredictionType>(data, len);
}

template<>
bool TryFromStringImpl<EPredictionType>(const char* data, size_t len, EPredictionType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEPredictionTypePrivate::TNameBufs, EPredictionType>(data, len, result);
}

bool FromString(const TString& name, EPredictionType& ret) {
    return ::TryFromStringImpl<EPredictionType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EPredictionType& ret) {
    return ::TryFromStringImpl<EPredictionType>(name.data(), name.size(), ret);
}

template<>
void Out<EPredictionType>(IOutputStream& os, TTypeTraits<EPredictionType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEPredictionTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EPredictionType>(EPredictionType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEPredictionTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EPredictionType> GetEnumAllValuesImpl<EPredictionType>() {
        const NEPredictionTypePrivate::TNameBufs& names = NEPredictionTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EPredictionType>() {
        const NEPredictionTypePrivate::TNameBufs& names = NEPredictionTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EPredictionType, TString> GetEnumNamesImpl<EPredictionType>() {
        const NEPredictionTypePrivate::TNameBufs& names = NEPredictionTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EPredictionType>() {
        const NEPredictionTypePrivate::TNameBufs& names = NEPredictionTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EFstrType
namespace { namespace NEFstrTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EFstrType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 10> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 10>{{
        TNameBufsBase::EnumStringPair(EFstrType::PredictionValuesChange, "PredictionValuesChange"sv),
        TNameBufsBase::EnumStringPair(EFstrType::LossFunctionChange, "LossFunctionChange"sv),
        TNameBufsBase::EnumStringPair(EFstrType::FeatureImportance, "FeatureImportance"sv),
        TNameBufsBase::EnumStringPair(EFstrType::InternalFeatureImportance, "InternalFeatureImportance"sv),
        TNameBufsBase::EnumStringPair(EFstrType::Interaction, "Interaction"sv),
        TNameBufsBase::EnumStringPair(EFstrType::InternalInteraction, "InternalInteraction"sv),
        TNameBufsBase::EnumStringPair(EFstrType::ShapValues, "ShapValues"sv),
        TNameBufsBase::EnumStringPair(EFstrType::PredictionDiff, "PredictionDiff"sv),
        TNameBufsBase::EnumStringPair(EFstrType::ShapInteractionValues, "ShapInteractionValues"sv),
        TNameBufsBase::EnumStringPair(EFstrType::SageValues, "SageValues"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[10]{
        TNameBufsBase::EnumStringPair(EFstrType::FeatureImportance, "FeatureImportance"sv),
        TNameBufsBase::EnumStringPair(EFstrType::Interaction, "Interaction"sv),
        TNameBufsBase::EnumStringPair(EFstrType::InternalFeatureImportance, "InternalFeatureImportance"sv),
        TNameBufsBase::EnumStringPair(EFstrType::InternalInteraction, "InternalInteraction"sv),
        TNameBufsBase::EnumStringPair(EFstrType::LossFunctionChange, "LossFunctionChange"sv),
        TNameBufsBase::EnumStringPair(EFstrType::PredictionDiff, "PredictionDiff"sv),
        TNameBufsBase::EnumStringPair(EFstrType::PredictionValuesChange, "PredictionValuesChange"sv),
        TNameBufsBase::EnumStringPair(EFstrType::SageValues, "SageValues"sv),
        TNameBufsBase::EnumStringPair(EFstrType::ShapInteractionValues, "ShapInteractionValues"sv),
        TNameBufsBase::EnumStringPair(EFstrType::ShapValues, "ShapValues"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[10]{
        "PredictionValuesChange"sv,
        "LossFunctionChange"sv,
        "FeatureImportance"sv,
        "InternalFeatureImportance"sv,
        "Interaction"sv,
        "InternalInteraction"sv,
        "ShapValues"sv,
        "PredictionDiff"sv,
        "ShapInteractionValues"sv,
        "SageValues"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EFstrType::"sv,
        "EFstrType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EFstrType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EFstrType>;

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

const TString& ToString(EFstrType x) {
    const NEFstrTypePrivate::TNameBufs& names = NEFstrTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EFstrType FromStringImpl<EFstrType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEFstrTypePrivate::TNameBufs, EFstrType>(data, len);
}

template<>
bool TryFromStringImpl<EFstrType>(const char* data, size_t len, EFstrType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEFstrTypePrivate::TNameBufs, EFstrType>(data, len, result);
}

bool FromString(const TString& name, EFstrType& ret) {
    return ::TryFromStringImpl<EFstrType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EFstrType& ret) {
    return ::TryFromStringImpl<EFstrType>(name.data(), name.size(), ret);
}

template<>
void Out<EFstrType>(IOutputStream& os, TTypeTraits<EFstrType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEFstrTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EFstrType>(EFstrType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEFstrTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EFstrType> GetEnumAllValuesImpl<EFstrType>() {
        const NEFstrTypePrivate::TNameBufs& names = NEFstrTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EFstrType>() {
        const NEFstrTypePrivate::TNameBufs& names = NEFstrTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EFstrType, TString> GetEnumNamesImpl<EFstrType>() {
        const NEFstrTypePrivate::TNameBufs& names = NEFstrTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EFstrType>() {
        const NEFstrTypePrivate::TNameBufs& names = NEFstrTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EFstrCalculatedInFitType
namespace { namespace NEFstrCalculatedInFitTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EFstrCalculatedInFitType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(EFstrCalculatedInFitType::PredictionValuesChange, "PredictionValuesChange"sv),
        TNameBufsBase::EnumStringPair(EFstrCalculatedInFitType::LossFunctionChange, "LossFunctionChange"sv),
        TNameBufsBase::EnumStringPair(EFstrCalculatedInFitType::FeatureImportance, "FeatureImportance"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(EFstrCalculatedInFitType::FeatureImportance, "FeatureImportance"sv),
        TNameBufsBase::EnumStringPair(EFstrCalculatedInFitType::LossFunctionChange, "LossFunctionChange"sv),
        TNameBufsBase::EnumStringPair(EFstrCalculatedInFitType::PredictionValuesChange, "PredictionValuesChange"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "PredictionValuesChange"sv,
        "LossFunctionChange"sv,
        "FeatureImportance"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EFstrCalculatedInFitType::"sv,
        "EFstrCalculatedInFitType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EFstrCalculatedInFitType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EFstrCalculatedInFitType>;

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

const TString& ToString(EFstrCalculatedInFitType x) {
    const NEFstrCalculatedInFitTypePrivate::TNameBufs& names = NEFstrCalculatedInFitTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EFstrCalculatedInFitType FromStringImpl<EFstrCalculatedInFitType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEFstrCalculatedInFitTypePrivate::TNameBufs, EFstrCalculatedInFitType>(data, len);
}

template<>
bool TryFromStringImpl<EFstrCalculatedInFitType>(const char* data, size_t len, EFstrCalculatedInFitType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEFstrCalculatedInFitTypePrivate::TNameBufs, EFstrCalculatedInFitType>(data, len, result);
}

bool FromString(const TString& name, EFstrCalculatedInFitType& ret) {
    return ::TryFromStringImpl<EFstrCalculatedInFitType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EFstrCalculatedInFitType& ret) {
    return ::TryFromStringImpl<EFstrCalculatedInFitType>(name.data(), name.size(), ret);
}

template<>
void Out<EFstrCalculatedInFitType>(IOutputStream& os, TTypeTraits<EFstrCalculatedInFitType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEFstrCalculatedInFitTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EFstrCalculatedInFitType>(EFstrCalculatedInFitType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEFstrCalculatedInFitTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EFstrCalculatedInFitType> GetEnumAllValuesImpl<EFstrCalculatedInFitType>() {
        const NEFstrCalculatedInFitTypePrivate::TNameBufs& names = NEFstrCalculatedInFitTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EFstrCalculatedInFitType>() {
        const NEFstrCalculatedInFitTypePrivate::TNameBufs& names = NEFstrCalculatedInFitTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EFstrCalculatedInFitType, TString> GetEnumNamesImpl<EFstrCalculatedInFitType>() {
        const NEFstrCalculatedInFitTypePrivate::TNameBufs& names = NEFstrCalculatedInFitTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EFstrCalculatedInFitType>() {
        const NEFstrCalculatedInFitTypePrivate::TNameBufs& names = NEFstrCalculatedInFitTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EPreCalcShapValues
namespace { namespace NEPreCalcShapValuesPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EPreCalcShapValues>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(EPreCalcShapValues::Auto, "Auto"sv),
        TNameBufsBase::EnumStringPair(EPreCalcShapValues::UsePreCalc, "UsePreCalc"sv),
        TNameBufsBase::EnumStringPair(EPreCalcShapValues::NoPreCalc, "NoPreCalc"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(EPreCalcShapValues::Auto, "Auto"sv),
        TNameBufsBase::EnumStringPair(EPreCalcShapValues::NoPreCalc, "NoPreCalc"sv),
        TNameBufsBase::EnumStringPair(EPreCalcShapValues::UsePreCalc, "UsePreCalc"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Auto"sv,
        "UsePreCalc"sv,
        "NoPreCalc"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EPreCalcShapValues::"sv,
        "EPreCalcShapValues"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EPreCalcShapValues> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EPreCalcShapValues>;

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

const TString& ToString(EPreCalcShapValues x) {
    const NEPreCalcShapValuesPrivate::TNameBufs& names = NEPreCalcShapValuesPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EPreCalcShapValues FromStringImpl<EPreCalcShapValues>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEPreCalcShapValuesPrivate::TNameBufs, EPreCalcShapValues>(data, len);
}

template<>
bool TryFromStringImpl<EPreCalcShapValues>(const char* data, size_t len, EPreCalcShapValues& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEPreCalcShapValuesPrivate::TNameBufs, EPreCalcShapValues>(data, len, result);
}

bool FromString(const TString& name, EPreCalcShapValues& ret) {
    return ::TryFromStringImpl<EPreCalcShapValues>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EPreCalcShapValues& ret) {
    return ::TryFromStringImpl<EPreCalcShapValues>(name.data(), name.size(), ret);
}

template<>
void Out<EPreCalcShapValues>(IOutputStream& os, TTypeTraits<EPreCalcShapValues>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEPreCalcShapValuesPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EPreCalcShapValues>(EPreCalcShapValues e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEPreCalcShapValuesPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EPreCalcShapValues> GetEnumAllValuesImpl<EPreCalcShapValues>() {
        const NEPreCalcShapValuesPrivate::TNameBufs& names = NEPreCalcShapValuesPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EPreCalcShapValues>() {
        const NEPreCalcShapValuesPrivate::TNameBufs& names = NEPreCalcShapValuesPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EPreCalcShapValues, TString> GetEnumNamesImpl<EPreCalcShapValues>() {
        const NEPreCalcShapValuesPrivate::TNameBufs& names = NEPreCalcShapValuesPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EPreCalcShapValues>() {
        const NEPreCalcShapValuesPrivate::TNameBufs& names = NEPreCalcShapValuesPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ECalcTypeShapValues
namespace { namespace NECalcTypeShapValuesPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ECalcTypeShapValues>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(ECalcTypeShapValues::Approximate, "Approximate"sv),
        TNameBufsBase::EnumStringPair(ECalcTypeShapValues::Regular, "Regular"sv),
        TNameBufsBase::EnumStringPair(ECalcTypeShapValues::Exact, "Exact"sv),
        TNameBufsBase::EnumStringPair(ECalcTypeShapValues::Independent, "Independent"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(ECalcTypeShapValues::Approximate, "Approximate"sv),
        TNameBufsBase::EnumStringPair(ECalcTypeShapValues::Exact, "Exact"sv),
        TNameBufsBase::EnumStringPair(ECalcTypeShapValues::Independent, "Independent"sv),
        TNameBufsBase::EnumStringPair(ECalcTypeShapValues::Regular, "Regular"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "Approximate"sv,
        "Regular"sv,
        "Exact"sv,
        "Independent"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ECalcTypeShapValues::"sv,
        "ECalcTypeShapValues"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ECalcTypeShapValues> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ECalcTypeShapValues>;

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

const TString& ToString(ECalcTypeShapValues x) {
    const NECalcTypeShapValuesPrivate::TNameBufs& names = NECalcTypeShapValuesPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ECalcTypeShapValues FromStringImpl<ECalcTypeShapValues>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NECalcTypeShapValuesPrivate::TNameBufs, ECalcTypeShapValues>(data, len);
}

template<>
bool TryFromStringImpl<ECalcTypeShapValues>(const char* data, size_t len, ECalcTypeShapValues& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NECalcTypeShapValuesPrivate::TNameBufs, ECalcTypeShapValues>(data, len, result);
}

bool FromString(const TString& name, ECalcTypeShapValues& ret) {
    return ::TryFromStringImpl<ECalcTypeShapValues>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ECalcTypeShapValues& ret) {
    return ::TryFromStringImpl<ECalcTypeShapValues>(name.data(), name.size(), ret);
}

template<>
void Out<ECalcTypeShapValues>(IOutputStream& os, TTypeTraits<ECalcTypeShapValues>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NECalcTypeShapValuesPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ECalcTypeShapValues>(ECalcTypeShapValues e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NECalcTypeShapValuesPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ECalcTypeShapValues> GetEnumAllValuesImpl<ECalcTypeShapValues>() {
        const NECalcTypeShapValuesPrivate::TNameBufs& names = NECalcTypeShapValuesPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ECalcTypeShapValues>() {
        const NECalcTypeShapValuesPrivate::TNameBufs& names = NECalcTypeShapValuesPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ECalcTypeShapValues, TString> GetEnumNamesImpl<ECalcTypeShapValues>() {
        const NECalcTypeShapValuesPrivate::TNameBufs& names = NECalcTypeShapValuesPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ECalcTypeShapValues>() {
        const NECalcTypeShapValuesPrivate::TNameBufs& names = NECalcTypeShapValuesPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EExplainableModelOutput
namespace { namespace NEExplainableModelOutputPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EExplainableModelOutput>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(EExplainableModelOutput::Raw, "Raw"sv),
        TNameBufsBase::EnumStringPair(EExplainableModelOutput::Probability, "Probability"sv),
        TNameBufsBase::EnumStringPair(EExplainableModelOutput::LossFunction, "LossFunction"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(EExplainableModelOutput::LossFunction, "LossFunction"sv),
        TNameBufsBase::EnumStringPair(EExplainableModelOutput::Probability, "Probability"sv),
        TNameBufsBase::EnumStringPair(EExplainableModelOutput::Raw, "Raw"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Raw"sv,
        "Probability"sv,
        "LossFunction"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EExplainableModelOutput::"sv,
        "EExplainableModelOutput"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EExplainableModelOutput> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EExplainableModelOutput>;

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

const TString& ToString(EExplainableModelOutput x) {
    const NEExplainableModelOutputPrivate::TNameBufs& names = NEExplainableModelOutputPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EExplainableModelOutput FromStringImpl<EExplainableModelOutput>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEExplainableModelOutputPrivate::TNameBufs, EExplainableModelOutput>(data, len);
}

template<>
bool TryFromStringImpl<EExplainableModelOutput>(const char* data, size_t len, EExplainableModelOutput& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEExplainableModelOutputPrivate::TNameBufs, EExplainableModelOutput>(data, len, result);
}

bool FromString(const TString& name, EExplainableModelOutput& ret) {
    return ::TryFromStringImpl<EExplainableModelOutput>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EExplainableModelOutput& ret) {
    return ::TryFromStringImpl<EExplainableModelOutput>(name.data(), name.size(), ret);
}

template<>
void Out<EExplainableModelOutput>(IOutputStream& os, TTypeTraits<EExplainableModelOutput>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEExplainableModelOutputPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EExplainableModelOutput>(EExplainableModelOutput e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEExplainableModelOutputPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EExplainableModelOutput> GetEnumAllValuesImpl<EExplainableModelOutput>() {
        const NEExplainableModelOutputPrivate::TNameBufs& names = NEExplainableModelOutputPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EExplainableModelOutput>() {
        const NEExplainableModelOutputPrivate::TNameBufs& names = NEExplainableModelOutputPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EExplainableModelOutput, TString> GetEnumNamesImpl<EExplainableModelOutput>() {
        const NEExplainableModelOutputPrivate::TNameBufs& names = NEExplainableModelOutputPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EExplainableModelOutput>() {
        const NEExplainableModelOutputPrivate::TNameBufs& names = NEExplainableModelOutputPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EObservationsToBootstrap
namespace { namespace NEObservationsToBootstrapPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EObservationsToBootstrap>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EObservationsToBootstrap::LearnAndTest, "LearnAndTest"sv),
        TNameBufsBase::EnumStringPair(EObservationsToBootstrap::TestOnly, "TestOnly"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EObservationsToBootstrap::LearnAndTest, "LearnAndTest"sv),
        TNameBufsBase::EnumStringPair(EObservationsToBootstrap::TestOnly, "TestOnly"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "LearnAndTest"sv,
        "TestOnly"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EObservationsToBootstrap::"sv,
        "EObservationsToBootstrap"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EObservationsToBootstrap> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EObservationsToBootstrap>;

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

const TString& ToString(EObservationsToBootstrap x) {
    const NEObservationsToBootstrapPrivate::TNameBufs& names = NEObservationsToBootstrapPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EObservationsToBootstrap FromStringImpl<EObservationsToBootstrap>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEObservationsToBootstrapPrivate::TNameBufs, EObservationsToBootstrap>(data, len);
}

template<>
bool TryFromStringImpl<EObservationsToBootstrap>(const char* data, size_t len, EObservationsToBootstrap& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEObservationsToBootstrapPrivate::TNameBufs, EObservationsToBootstrap>(data, len, result);
}

bool FromString(const TString& name, EObservationsToBootstrap& ret) {
    return ::TryFromStringImpl<EObservationsToBootstrap>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EObservationsToBootstrap& ret) {
    return ::TryFromStringImpl<EObservationsToBootstrap>(name.data(), name.size(), ret);
}

template<>
void Out<EObservationsToBootstrap>(IOutputStream& os, TTypeTraits<EObservationsToBootstrap>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEObservationsToBootstrapPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EObservationsToBootstrap>(EObservationsToBootstrap e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEObservationsToBootstrapPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EObservationsToBootstrap> GetEnumAllValuesImpl<EObservationsToBootstrap>() {
        const NEObservationsToBootstrapPrivate::TNameBufs& names = NEObservationsToBootstrapPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EObservationsToBootstrap>() {
        const NEObservationsToBootstrapPrivate::TNameBufs& names = NEObservationsToBootstrapPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EObservationsToBootstrap, TString> GetEnumNamesImpl<EObservationsToBootstrap>() {
        const NEObservationsToBootstrapPrivate::TNameBufs& names = NEObservationsToBootstrapPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EObservationsToBootstrap>() {
        const NEObservationsToBootstrapPrivate::TNameBufs& names = NEObservationsToBootstrapPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EGpuCatFeaturesStorage
namespace { namespace NEGpuCatFeaturesStoragePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EGpuCatFeaturesStorage>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EGpuCatFeaturesStorage::CpuPinnedMemory, "CpuPinnedMemory"sv),
        TNameBufsBase::EnumStringPair(EGpuCatFeaturesStorage::GpuRam, "GpuRam"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EGpuCatFeaturesStorage::CpuPinnedMemory, "CpuPinnedMemory"sv),
        TNameBufsBase::EnumStringPair(EGpuCatFeaturesStorage::GpuRam, "GpuRam"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "CpuPinnedMemory"sv,
        "GpuRam"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EGpuCatFeaturesStorage::"sv,
        "EGpuCatFeaturesStorage"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EGpuCatFeaturesStorage> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EGpuCatFeaturesStorage>;

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

const TString& ToString(EGpuCatFeaturesStorage x) {
    const NEGpuCatFeaturesStoragePrivate::TNameBufs& names = NEGpuCatFeaturesStoragePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EGpuCatFeaturesStorage FromStringImpl<EGpuCatFeaturesStorage>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEGpuCatFeaturesStoragePrivate::TNameBufs, EGpuCatFeaturesStorage>(data, len);
}

template<>
bool TryFromStringImpl<EGpuCatFeaturesStorage>(const char* data, size_t len, EGpuCatFeaturesStorage& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEGpuCatFeaturesStoragePrivate::TNameBufs, EGpuCatFeaturesStorage>(data, len, result);
}

bool FromString(const TString& name, EGpuCatFeaturesStorage& ret) {
    return ::TryFromStringImpl<EGpuCatFeaturesStorage>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EGpuCatFeaturesStorage& ret) {
    return ::TryFromStringImpl<EGpuCatFeaturesStorage>(name.data(), name.size(), ret);
}

template<>
void Out<EGpuCatFeaturesStorage>(IOutputStream& os, TTypeTraits<EGpuCatFeaturesStorage>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEGpuCatFeaturesStoragePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EGpuCatFeaturesStorage>(EGpuCatFeaturesStorage e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEGpuCatFeaturesStoragePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EGpuCatFeaturesStorage> GetEnumAllValuesImpl<EGpuCatFeaturesStorage>() {
        const NEGpuCatFeaturesStoragePrivate::TNameBufs& names = NEGpuCatFeaturesStoragePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EGpuCatFeaturesStorage>() {
        const NEGpuCatFeaturesStoragePrivate::TNameBufs& names = NEGpuCatFeaturesStoragePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EGpuCatFeaturesStorage, TString> GetEnumNamesImpl<EGpuCatFeaturesStorage>() {
        const NEGpuCatFeaturesStoragePrivate::TNameBufs& names = NEGpuCatFeaturesStoragePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EGpuCatFeaturesStorage>() {
        const NEGpuCatFeaturesStoragePrivate::TNameBufs& names = NEGpuCatFeaturesStoragePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EProjectionType
namespace { namespace NEProjectionTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EProjectionType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EProjectionType::TreeCtr, "TreeCtr"sv),
        TNameBufsBase::EnumStringPair(EProjectionType::SimpleCtr, "SimpleCtr"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EProjectionType::SimpleCtr, "SimpleCtr"sv),
        TNameBufsBase::EnumStringPair(EProjectionType::TreeCtr, "TreeCtr"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "TreeCtr"sv,
        "SimpleCtr"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EProjectionType::"sv,
        "EProjectionType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EProjectionType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EProjectionType>;

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

const TString& ToString(EProjectionType x) {
    const NEProjectionTypePrivate::TNameBufs& names = NEProjectionTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EProjectionType FromStringImpl<EProjectionType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEProjectionTypePrivate::TNameBufs, EProjectionType>(data, len);
}

template<>
bool TryFromStringImpl<EProjectionType>(const char* data, size_t len, EProjectionType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEProjectionTypePrivate::TNameBufs, EProjectionType>(data, len, result);
}

bool FromString(const TString& name, EProjectionType& ret) {
    return ::TryFromStringImpl<EProjectionType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EProjectionType& ret) {
    return ::TryFromStringImpl<EProjectionType>(name.data(), name.size(), ret);
}

template<>
void Out<EProjectionType>(IOutputStream& os, TTypeTraits<EProjectionType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEProjectionTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EProjectionType>(EProjectionType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEProjectionTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EProjectionType> GetEnumAllValuesImpl<EProjectionType>() {
        const NEProjectionTypePrivate::TNameBufs& names = NEProjectionTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EProjectionType>() {
        const NEProjectionTypePrivate::TNameBufs& names = NEProjectionTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EProjectionType, TString> GetEnumNamesImpl<EProjectionType>() {
        const NEProjectionTypePrivate::TNameBufs& names = NEProjectionTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EProjectionType>() {
        const NEProjectionTypePrivate::TNameBufs& names = NEProjectionTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EPriorEstimation
namespace { namespace NEPriorEstimationPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EPriorEstimation>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EPriorEstimation::No, "No"sv),
        TNameBufsBase::EnumStringPair(EPriorEstimation::BetaPrior, "BetaPrior"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EPriorEstimation::BetaPrior, "BetaPrior"sv),
        TNameBufsBase::EnumStringPair(EPriorEstimation::No, "No"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "No"sv,
        "BetaPrior"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EPriorEstimation::"sv,
        "EPriorEstimation"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EPriorEstimation> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EPriorEstimation>;

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

const TString& ToString(EPriorEstimation x) {
    const NEPriorEstimationPrivate::TNameBufs& names = NEPriorEstimationPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EPriorEstimation FromStringImpl<EPriorEstimation>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEPriorEstimationPrivate::TNameBufs, EPriorEstimation>(data, len);
}

template<>
bool TryFromStringImpl<EPriorEstimation>(const char* data, size_t len, EPriorEstimation& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEPriorEstimationPrivate::TNameBufs, EPriorEstimation>(data, len, result);
}

bool FromString(const TString& name, EPriorEstimation& ret) {
    return ::TryFromStringImpl<EPriorEstimation>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EPriorEstimation& ret) {
    return ::TryFromStringImpl<EPriorEstimation>(name.data(), name.size(), ret);
}

template<>
void Out<EPriorEstimation>(IOutputStream& os, TTypeTraits<EPriorEstimation>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEPriorEstimationPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EPriorEstimation>(EPriorEstimation e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEPriorEstimationPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EPriorEstimation> GetEnumAllValuesImpl<EPriorEstimation>() {
        const NEPriorEstimationPrivate::TNameBufs& names = NEPriorEstimationPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EPriorEstimation>() {
        const NEPriorEstimationPrivate::TNameBufs& names = NEPriorEstimationPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EPriorEstimation, TString> GetEnumNamesImpl<EPriorEstimation>() {
        const NEPriorEstimationPrivate::TNameBufs& names = NEPriorEstimationPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EPriorEstimation>() {
        const NEPriorEstimationPrivate::TNameBufs& names = NEPriorEstimationPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ELaunchMode
namespace { namespace NELaunchModePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ELaunchMode>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(ELaunchMode::Train, "Train"sv),
        TNameBufsBase::EnumStringPair(ELaunchMode::Eval, "Eval"sv),
        TNameBufsBase::EnumStringPair(ELaunchMode::CV, "CV"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(ELaunchMode::CV, "CV"sv),
        TNameBufsBase::EnumStringPair(ELaunchMode::Eval, "Eval"sv),
        TNameBufsBase::EnumStringPair(ELaunchMode::Train, "Train"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Train"sv,
        "Eval"sv,
        "CV"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ELaunchMode::"sv,
        "ELaunchMode"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ELaunchMode> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ELaunchMode>;

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

const TString& ToString(ELaunchMode x) {
    const NELaunchModePrivate::TNameBufs& names = NELaunchModePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ELaunchMode FromStringImpl<ELaunchMode>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NELaunchModePrivate::TNameBufs, ELaunchMode>(data, len);
}

template<>
bool TryFromStringImpl<ELaunchMode>(const char* data, size_t len, ELaunchMode& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NELaunchModePrivate::TNameBufs, ELaunchMode>(data, len, result);
}

bool FromString(const TString& name, ELaunchMode& ret) {
    return ::TryFromStringImpl<ELaunchMode>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ELaunchMode& ret) {
    return ::TryFromStringImpl<ELaunchMode>(name.data(), name.size(), ret);
}

template<>
void Out<ELaunchMode>(IOutputStream& os, TTypeTraits<ELaunchMode>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NELaunchModePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ELaunchMode>(ELaunchMode e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NELaunchModePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ELaunchMode> GetEnumAllValuesImpl<ELaunchMode>() {
        const NELaunchModePrivate::TNameBufs& names = NELaunchModePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ELaunchMode>() {
        const NELaunchModePrivate::TNameBufs& names = NELaunchModePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ELaunchMode, TString> GetEnumNamesImpl<ELaunchMode>() {
        const NELaunchModePrivate::TNameBufs& names = NELaunchModePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ELaunchMode>() {
        const NELaunchModePrivate::TNameBufs& names = NELaunchModePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ENodeType
namespace { namespace NENodeTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ENodeType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ENodeType::Master, "Master"sv),
        TNameBufsBase::EnumStringPair(ENodeType::SingleHost, "SingleHost"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ENodeType::Master, "Master"sv),
        TNameBufsBase::EnumStringPair(ENodeType::SingleHost, "SingleHost"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Master"sv,
        "SingleHost"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ENodeType::"sv,
        "ENodeType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ENodeType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ENodeType>;

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

const TString& ToString(ENodeType x) {
    const NENodeTypePrivate::TNameBufs& names = NENodeTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ENodeType FromStringImpl<ENodeType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NENodeTypePrivate::TNameBufs, ENodeType>(data, len);
}

template<>
bool TryFromStringImpl<ENodeType>(const char* data, size_t len, ENodeType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NENodeTypePrivate::TNameBufs, ENodeType>(data, len, result);
}

bool FromString(const TString& name, ENodeType& ret) {
    return ::TryFromStringImpl<ENodeType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ENodeType& ret) {
    return ::TryFromStringImpl<ENodeType>(name.data(), name.size(), ret);
}

template<>
void Out<ENodeType>(IOutputStream& os, TTypeTraits<ENodeType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NENodeTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ENodeType>(ENodeType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NENodeTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ENodeType> GetEnumAllValuesImpl<ENodeType>() {
        const NENodeTypePrivate::TNameBufs& names = NENodeTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ENodeType>() {
        const NENodeTypePrivate::TNameBufs& names = NENodeTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ENodeType, TString> GetEnumNamesImpl<ENodeType>() {
        const NENodeTypePrivate::TNameBufs& names = NENodeTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ENodeType>() {
        const NENodeTypePrivate::TNameBufs& names = NENodeTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EFinalCtrComputationMode
namespace { namespace NEFinalCtrComputationModePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EFinalCtrComputationMode>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EFinalCtrComputationMode::Skip, "Skip"sv),
        TNameBufsBase::EnumStringPair(EFinalCtrComputationMode::Default, "Default"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EFinalCtrComputationMode::Default, "Default"sv),
        TNameBufsBase::EnumStringPair(EFinalCtrComputationMode::Skip, "Skip"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Skip"sv,
        "Default"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EFinalCtrComputationMode::"sv,
        "EFinalCtrComputationMode"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EFinalCtrComputationMode> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EFinalCtrComputationMode>;

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

const TString& ToString(EFinalCtrComputationMode x) {
    const NEFinalCtrComputationModePrivate::TNameBufs& names = NEFinalCtrComputationModePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EFinalCtrComputationMode FromStringImpl<EFinalCtrComputationMode>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEFinalCtrComputationModePrivate::TNameBufs, EFinalCtrComputationMode>(data, len);
}

template<>
bool TryFromStringImpl<EFinalCtrComputationMode>(const char* data, size_t len, EFinalCtrComputationMode& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEFinalCtrComputationModePrivate::TNameBufs, EFinalCtrComputationMode>(data, len, result);
}

bool FromString(const TString& name, EFinalCtrComputationMode& ret) {
    return ::TryFromStringImpl<EFinalCtrComputationMode>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EFinalCtrComputationMode& ret) {
    return ::TryFromStringImpl<EFinalCtrComputationMode>(name.data(), name.size(), ret);
}

template<>
void Out<EFinalCtrComputationMode>(IOutputStream& os, TTypeTraits<EFinalCtrComputationMode>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEFinalCtrComputationModePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EFinalCtrComputationMode>(EFinalCtrComputationMode e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEFinalCtrComputationModePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EFinalCtrComputationMode> GetEnumAllValuesImpl<EFinalCtrComputationMode>() {
        const NEFinalCtrComputationModePrivate::TNameBufs& names = NEFinalCtrComputationModePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EFinalCtrComputationMode>() {
        const NEFinalCtrComputationModePrivate::TNameBufs& names = NEFinalCtrComputationModePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EFinalCtrComputationMode, TString> GetEnumNamesImpl<EFinalCtrComputationMode>() {
        const NEFinalCtrComputationModePrivate::TNameBufs& names = NEFinalCtrComputationModePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EFinalCtrComputationMode>() {
        const NEFinalCtrComputationModePrivate::TNameBufs& names = NEFinalCtrComputationModePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EFinalFeatureCalcersComputationMode
namespace { namespace NEFinalFeatureCalcersComputationModePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EFinalFeatureCalcersComputationMode>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EFinalFeatureCalcersComputationMode::Skip, "Skip"sv),
        TNameBufsBase::EnumStringPair(EFinalFeatureCalcersComputationMode::Default, "Default"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EFinalFeatureCalcersComputationMode::Default, "Default"sv),
        TNameBufsBase::EnumStringPair(EFinalFeatureCalcersComputationMode::Skip, "Skip"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Skip"sv,
        "Default"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EFinalFeatureCalcersComputationMode::"sv,
        "EFinalFeatureCalcersComputationMode"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EFinalFeatureCalcersComputationMode> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EFinalFeatureCalcersComputationMode>;

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

const TString& ToString(EFinalFeatureCalcersComputationMode x) {
    const NEFinalFeatureCalcersComputationModePrivate::TNameBufs& names = NEFinalFeatureCalcersComputationModePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EFinalFeatureCalcersComputationMode FromStringImpl<EFinalFeatureCalcersComputationMode>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEFinalFeatureCalcersComputationModePrivate::TNameBufs, EFinalFeatureCalcersComputationMode>(data, len);
}

template<>
bool TryFromStringImpl<EFinalFeatureCalcersComputationMode>(const char* data, size_t len, EFinalFeatureCalcersComputationMode& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEFinalFeatureCalcersComputationModePrivate::TNameBufs, EFinalFeatureCalcersComputationMode>(data, len, result);
}

bool FromString(const TString& name, EFinalFeatureCalcersComputationMode& ret) {
    return ::TryFromStringImpl<EFinalFeatureCalcersComputationMode>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EFinalFeatureCalcersComputationMode& ret) {
    return ::TryFromStringImpl<EFinalFeatureCalcersComputationMode>(name.data(), name.size(), ret);
}

template<>
void Out<EFinalFeatureCalcersComputationMode>(IOutputStream& os, TTypeTraits<EFinalFeatureCalcersComputationMode>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEFinalFeatureCalcersComputationModePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EFinalFeatureCalcersComputationMode>(EFinalFeatureCalcersComputationMode e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEFinalFeatureCalcersComputationModePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EFinalFeatureCalcersComputationMode> GetEnumAllValuesImpl<EFinalFeatureCalcersComputationMode>() {
        const NEFinalFeatureCalcersComputationModePrivate::TNameBufs& names = NEFinalFeatureCalcersComputationModePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EFinalFeatureCalcersComputationMode>() {
        const NEFinalFeatureCalcersComputationModePrivate::TNameBufs& names = NEFinalFeatureCalcersComputationModePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EFinalFeatureCalcersComputationMode, TString> GetEnumNamesImpl<EFinalFeatureCalcersComputationMode>() {
        const NEFinalFeatureCalcersComputationModePrivate::TNameBufs& names = NEFinalFeatureCalcersComputationModePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EFinalFeatureCalcersComputationMode>() {
        const NEFinalFeatureCalcersComputationModePrivate::TNameBufs& names = NEFinalFeatureCalcersComputationModePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ELeavesEstimationStepBacktracking
namespace { namespace NELeavesEstimationStepBacktrackingPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ELeavesEstimationStepBacktracking>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(ELeavesEstimationStepBacktracking::No, "No"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimationStepBacktracking::AnyImprovement, "AnyImprovement"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimationStepBacktracking::Armijo, "Armijo"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(ELeavesEstimationStepBacktracking::AnyImprovement, "AnyImprovement"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimationStepBacktracking::Armijo, "Armijo"sv),
        TNameBufsBase::EnumStringPair(ELeavesEstimationStepBacktracking::No, "No"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "No"sv,
        "AnyImprovement"sv,
        "Armijo"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ELeavesEstimationStepBacktracking::"sv,
        "ELeavesEstimationStepBacktracking"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ELeavesEstimationStepBacktracking> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ELeavesEstimationStepBacktracking>;

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

const TString& ToString(ELeavesEstimationStepBacktracking x) {
    const NELeavesEstimationStepBacktrackingPrivate::TNameBufs& names = NELeavesEstimationStepBacktrackingPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ELeavesEstimationStepBacktracking FromStringImpl<ELeavesEstimationStepBacktracking>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NELeavesEstimationStepBacktrackingPrivate::TNameBufs, ELeavesEstimationStepBacktracking>(data, len);
}

template<>
bool TryFromStringImpl<ELeavesEstimationStepBacktracking>(const char* data, size_t len, ELeavesEstimationStepBacktracking& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NELeavesEstimationStepBacktrackingPrivate::TNameBufs, ELeavesEstimationStepBacktracking>(data, len, result);
}

bool FromString(const TString& name, ELeavesEstimationStepBacktracking& ret) {
    return ::TryFromStringImpl<ELeavesEstimationStepBacktracking>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ELeavesEstimationStepBacktracking& ret) {
    return ::TryFromStringImpl<ELeavesEstimationStepBacktracking>(name.data(), name.size(), ret);
}

template<>
void Out<ELeavesEstimationStepBacktracking>(IOutputStream& os, TTypeTraits<ELeavesEstimationStepBacktracking>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NELeavesEstimationStepBacktrackingPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ELeavesEstimationStepBacktracking>(ELeavesEstimationStepBacktracking e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NELeavesEstimationStepBacktrackingPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ELeavesEstimationStepBacktracking> GetEnumAllValuesImpl<ELeavesEstimationStepBacktracking>() {
        const NELeavesEstimationStepBacktrackingPrivate::TNameBufs& names = NELeavesEstimationStepBacktrackingPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ELeavesEstimationStepBacktracking>() {
        const NELeavesEstimationStepBacktrackingPrivate::TNameBufs& names = NELeavesEstimationStepBacktrackingPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ELeavesEstimationStepBacktracking, TString> GetEnumNamesImpl<ELeavesEstimationStepBacktracking>() {
        const NELeavesEstimationStepBacktrackingPrivate::TNameBufs& names = NELeavesEstimationStepBacktrackingPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ELeavesEstimationStepBacktracking>() {
        const NELeavesEstimationStepBacktrackingPrivate::TNameBufs& names = NELeavesEstimationStepBacktrackingPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EKappaMetricType
namespace { namespace NEKappaMetricTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EKappaMetricType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EKappaMetricType::Cohen, "Cohen"sv),
        TNameBufsBase::EnumStringPair(EKappaMetricType::Weighted, "Weighted"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EKappaMetricType::Cohen, "Cohen"sv),
        TNameBufsBase::EnumStringPair(EKappaMetricType::Weighted, "Weighted"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Cohen"sv,
        "Weighted"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EKappaMetricType::"sv,
        "EKappaMetricType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EKappaMetricType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EKappaMetricType>;

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

const TString& ToString(EKappaMetricType x) {
    const NEKappaMetricTypePrivate::TNameBufs& names = NEKappaMetricTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EKappaMetricType FromStringImpl<EKappaMetricType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEKappaMetricTypePrivate::TNameBufs, EKappaMetricType>(data, len);
}

template<>
bool TryFromStringImpl<EKappaMetricType>(const char* data, size_t len, EKappaMetricType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEKappaMetricTypePrivate::TNameBufs, EKappaMetricType>(data, len, result);
}

bool FromString(const TString& name, EKappaMetricType& ret) {
    return ::TryFromStringImpl<EKappaMetricType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EKappaMetricType& ret) {
    return ::TryFromStringImpl<EKappaMetricType>(name.data(), name.size(), ret);
}

template<>
void Out<EKappaMetricType>(IOutputStream& os, TTypeTraits<EKappaMetricType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEKappaMetricTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EKappaMetricType>(EKappaMetricType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEKappaMetricTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EKappaMetricType> GetEnumAllValuesImpl<EKappaMetricType>() {
        const NEKappaMetricTypePrivate::TNameBufs& names = NEKappaMetricTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EKappaMetricType>() {
        const NEKappaMetricTypePrivate::TNameBufs& names = NEKappaMetricTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EKappaMetricType, TString> GetEnumNamesImpl<EKappaMetricType>() {
        const NEKappaMetricTypePrivate::TNameBufs& names = NEKappaMetricTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EKappaMetricType>() {
        const NEKappaMetricTypePrivate::TNameBufs& names = NEKappaMetricTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ENdcgMetricType
namespace { namespace NENdcgMetricTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ENdcgMetricType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ENdcgMetricType::Base, "Base"sv),
        TNameBufsBase::EnumStringPair(ENdcgMetricType::Exp, "Exp"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ENdcgMetricType::Base, "Base"sv),
        TNameBufsBase::EnumStringPair(ENdcgMetricType::Exp, "Exp"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Base"sv,
        "Exp"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ENdcgMetricType::"sv,
        "ENdcgMetricType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ENdcgMetricType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ENdcgMetricType>;

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

const TString& ToString(ENdcgMetricType x) {
    const NENdcgMetricTypePrivate::TNameBufs& names = NENdcgMetricTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ENdcgMetricType FromStringImpl<ENdcgMetricType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NENdcgMetricTypePrivate::TNameBufs, ENdcgMetricType>(data, len);
}

template<>
bool TryFromStringImpl<ENdcgMetricType>(const char* data, size_t len, ENdcgMetricType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NENdcgMetricTypePrivate::TNameBufs, ENdcgMetricType>(data, len, result);
}

bool FromString(const TString& name, ENdcgMetricType& ret) {
    return ::TryFromStringImpl<ENdcgMetricType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ENdcgMetricType& ret) {
    return ::TryFromStringImpl<ENdcgMetricType>(name.data(), name.size(), ret);
}

template<>
void Out<ENdcgMetricType>(IOutputStream& os, TTypeTraits<ENdcgMetricType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NENdcgMetricTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ENdcgMetricType>(ENdcgMetricType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NENdcgMetricTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ENdcgMetricType> GetEnumAllValuesImpl<ENdcgMetricType>() {
        const NENdcgMetricTypePrivate::TNameBufs& names = NENdcgMetricTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ENdcgMetricType>() {
        const NENdcgMetricTypePrivate::TNameBufs& names = NENdcgMetricTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ENdcgMetricType, TString> GetEnumNamesImpl<ENdcgMetricType>() {
        const NENdcgMetricTypePrivate::TNameBufs& names = NENdcgMetricTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ENdcgMetricType>() {
        const NENdcgMetricTypePrivate::TNameBufs& names = NENdcgMetricTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ENdcgDenominatorType
namespace { namespace NENdcgDenominatorTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ENdcgDenominatorType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(ENdcgDenominatorType::LogPosition, "LogPosition"sv),
        TNameBufsBase::EnumStringPair(ENdcgDenominatorType::Position, "Position"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(ENdcgDenominatorType::LogPosition, "LogPosition"sv),
        TNameBufsBase::EnumStringPair(ENdcgDenominatorType::Position, "Position"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "LogPosition"sv,
        "Position"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ENdcgDenominatorType::"sv,
        "ENdcgDenominatorType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ENdcgDenominatorType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ENdcgDenominatorType>;

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

const TString& ToString(ENdcgDenominatorType x) {
    const NENdcgDenominatorTypePrivate::TNameBufs& names = NENdcgDenominatorTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ENdcgDenominatorType FromStringImpl<ENdcgDenominatorType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NENdcgDenominatorTypePrivate::TNameBufs, ENdcgDenominatorType>(data, len);
}

template<>
bool TryFromStringImpl<ENdcgDenominatorType>(const char* data, size_t len, ENdcgDenominatorType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NENdcgDenominatorTypePrivate::TNameBufs, ENdcgDenominatorType>(data, len, result);
}

bool FromString(const TString& name, ENdcgDenominatorType& ret) {
    return ::TryFromStringImpl<ENdcgDenominatorType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ENdcgDenominatorType& ret) {
    return ::TryFromStringImpl<ENdcgDenominatorType>(name.data(), name.size(), ret);
}

template<>
void Out<ENdcgDenominatorType>(IOutputStream& os, TTypeTraits<ENdcgDenominatorType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NENdcgDenominatorTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ENdcgDenominatorType>(ENdcgDenominatorType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NENdcgDenominatorTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ENdcgDenominatorType> GetEnumAllValuesImpl<ENdcgDenominatorType>() {
        const NENdcgDenominatorTypePrivate::TNameBufs& names = NENdcgDenominatorTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ENdcgDenominatorType>() {
        const NENdcgDenominatorTypePrivate::TNameBufs& names = NENdcgDenominatorTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ENdcgDenominatorType, TString> GetEnumNamesImpl<ENdcgDenominatorType>() {
        const NENdcgDenominatorTypePrivate::TNameBufs& names = NENdcgDenominatorTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ENdcgDenominatorType>() {
        const NENdcgDenominatorTypePrivate::TNameBufs& names = NENdcgDenominatorTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ENdcgSortType
namespace { namespace NENdcgSortTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ENdcgSortType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(ENdcgSortType::None, "None"sv),
        TNameBufsBase::EnumStringPair(ENdcgSortType::ByPrediction, "ByPrediction"sv),
        TNameBufsBase::EnumStringPair(ENdcgSortType::ByTarget, "ByTarget"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(ENdcgSortType::ByPrediction, "ByPrediction"sv),
        TNameBufsBase::EnumStringPair(ENdcgSortType::ByTarget, "ByTarget"sv),
        TNameBufsBase::EnumStringPair(ENdcgSortType::None, "None"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "None"sv,
        "ByPrediction"sv,
        "ByTarget"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "ENdcgSortType::"sv,
        "ENdcgSortType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ENdcgSortType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ENdcgSortType>;

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

const TString& ToString(ENdcgSortType x) {
    const NENdcgSortTypePrivate::TNameBufs& names = NENdcgSortTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ENdcgSortType FromStringImpl<ENdcgSortType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NENdcgSortTypePrivate::TNameBufs, ENdcgSortType>(data, len);
}

template<>
bool TryFromStringImpl<ENdcgSortType>(const char* data, size_t len, ENdcgSortType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NENdcgSortTypePrivate::TNameBufs, ENdcgSortType>(data, len, result);
}

bool FromString(const TString& name, ENdcgSortType& ret) {
    return ::TryFromStringImpl<ENdcgSortType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ENdcgSortType& ret) {
    return ::TryFromStringImpl<ENdcgSortType>(name.data(), name.size(), ret);
}

template<>
void Out<ENdcgSortType>(IOutputStream& os, TTypeTraits<ENdcgSortType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NENdcgSortTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ENdcgSortType>(ENdcgSortType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NENdcgSortTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ENdcgSortType> GetEnumAllValuesImpl<ENdcgSortType>() {
        const NENdcgSortTypePrivate::TNameBufs& names = NENdcgSortTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ENdcgSortType>() {
        const NENdcgSortTypePrivate::TNameBufs& names = NENdcgSortTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ENdcgSortType, TString> GetEnumNamesImpl<ENdcgSortType>() {
        const NENdcgSortTypePrivate::TNameBufs& names = NENdcgSortTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ENdcgSortType>() {
        const NENdcgSortTypePrivate::TNameBufs& names = NENdcgSortTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EMetricBestValue
namespace { namespace NEMetricBestValuePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EMetricBestValue>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(EMetricBestValue::Max, "Max"sv),
        TNameBufsBase::EnumStringPair(EMetricBestValue::Min, "Min"sv),
        TNameBufsBase::EnumStringPair(EMetricBestValue::FixedValue, "FixedValue"sv),
        TNameBufsBase::EnumStringPair(EMetricBestValue::Undefined, "Undefined"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(EMetricBestValue::FixedValue, "FixedValue"sv),
        TNameBufsBase::EnumStringPair(EMetricBestValue::Max, "Max"sv),
        TNameBufsBase::EnumStringPair(EMetricBestValue::Min, "Min"sv),
        TNameBufsBase::EnumStringPair(EMetricBestValue::Undefined, "Undefined"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "Max"sv,
        "Min"sv,
        "FixedValue"sv,
        "Undefined"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EMetricBestValue::"sv,
        "EMetricBestValue"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EMetricBestValue> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EMetricBestValue>;

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

const TString& ToString(EMetricBestValue x) {
    const NEMetricBestValuePrivate::TNameBufs& names = NEMetricBestValuePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EMetricBestValue FromStringImpl<EMetricBestValue>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEMetricBestValuePrivate::TNameBufs, EMetricBestValue>(data, len);
}

template<>
bool TryFromStringImpl<EMetricBestValue>(const char* data, size_t len, EMetricBestValue& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEMetricBestValuePrivate::TNameBufs, EMetricBestValue>(data, len, result);
}

bool FromString(const TString& name, EMetricBestValue& ret) {
    return ::TryFromStringImpl<EMetricBestValue>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EMetricBestValue& ret) {
    return ::TryFromStringImpl<EMetricBestValue>(name.data(), name.size(), ret);
}

template<>
void Out<EMetricBestValue>(IOutputStream& os, TTypeTraits<EMetricBestValue>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEMetricBestValuePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EMetricBestValue>(EMetricBestValue e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEMetricBestValuePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EMetricBestValue> GetEnumAllValuesImpl<EMetricBestValue>() {
        const NEMetricBestValuePrivate::TNameBufs& names = NEMetricBestValuePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EMetricBestValue>() {
        const NEMetricBestValuePrivate::TNameBufs& names = NEMetricBestValuePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EMetricBestValue, TString> GetEnumNamesImpl<EMetricBestValue>() {
        const NEMetricBestValuePrivate::TNameBufs& names = NEMetricBestValuePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EMetricBestValue>() {
        const NEMetricBestValuePrivate::TNameBufs& names = NEMetricBestValuePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EFeatureCalcerType
namespace { namespace NEFeatureCalcerTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EFeatureCalcerType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 5> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 5>{{
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::BoW, "BoW"sv),
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::NaiveBayes, "NaiveBayes"sv),
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::BM25, "BM25"sv),
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::LDA, "LDA"sv),
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::KNN, "KNN"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[5]{
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::BM25, "BM25"sv),
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::BoW, "BoW"sv),
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::KNN, "KNN"sv),
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::LDA, "LDA"sv),
        TNameBufsBase::EnumStringPair(EFeatureCalcerType::NaiveBayes, "NaiveBayes"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[5]{
        "BoW"sv,
        "NaiveBayes"sv,
        "BM25"sv,
        "LDA"sv,
        "KNN"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EFeatureCalcerType::"sv,
        "EFeatureCalcerType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EFeatureCalcerType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EFeatureCalcerType>;

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

const TString& ToString(EFeatureCalcerType x) {
    const NEFeatureCalcerTypePrivate::TNameBufs& names = NEFeatureCalcerTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EFeatureCalcerType FromStringImpl<EFeatureCalcerType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEFeatureCalcerTypePrivate::TNameBufs, EFeatureCalcerType>(data, len);
}

template<>
bool TryFromStringImpl<EFeatureCalcerType>(const char* data, size_t len, EFeatureCalcerType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEFeatureCalcerTypePrivate::TNameBufs, EFeatureCalcerType>(data, len, result);
}

bool FromString(const TString& name, EFeatureCalcerType& ret) {
    return ::TryFromStringImpl<EFeatureCalcerType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EFeatureCalcerType& ret) {
    return ::TryFromStringImpl<EFeatureCalcerType>(name.data(), name.size(), ret);
}

template<>
void Out<EFeatureCalcerType>(IOutputStream& os, TTypeTraits<EFeatureCalcerType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEFeatureCalcerTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EFeatureCalcerType>(EFeatureCalcerType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEFeatureCalcerTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EFeatureCalcerType> GetEnumAllValuesImpl<EFeatureCalcerType>() {
        const NEFeatureCalcerTypePrivate::TNameBufs& names = NEFeatureCalcerTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EFeatureCalcerType>() {
        const NEFeatureCalcerTypePrivate::TNameBufs& names = NEFeatureCalcerTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EFeatureCalcerType, TString> GetEnumNamesImpl<EFeatureCalcerType>() {
        const NEFeatureCalcerTypePrivate::TNameBufs& names = NEFeatureCalcerTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EFeatureCalcerType>() {
        const NEFeatureCalcerTypePrivate::TNameBufs& names = NEFeatureCalcerTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EAutoClassWeightsType
namespace { namespace NEAutoClassWeightsTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EAutoClassWeightsType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(EAutoClassWeightsType::Balanced, "Balanced"sv),
        TNameBufsBase::EnumStringPair(EAutoClassWeightsType::SqrtBalanced, "SqrtBalanced"sv),
        TNameBufsBase::EnumStringPair(EAutoClassWeightsType::None, "None"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(EAutoClassWeightsType::Balanced, "Balanced"sv),
        TNameBufsBase::EnumStringPair(EAutoClassWeightsType::None, "None"sv),
        TNameBufsBase::EnumStringPair(EAutoClassWeightsType::SqrtBalanced, "SqrtBalanced"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Balanced"sv,
        "SqrtBalanced"sv,
        "None"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EAutoClassWeightsType::"sv,
        "EAutoClassWeightsType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EAutoClassWeightsType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EAutoClassWeightsType>;

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

const TString& ToString(EAutoClassWeightsType x) {
    const NEAutoClassWeightsTypePrivate::TNameBufs& names = NEAutoClassWeightsTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EAutoClassWeightsType FromStringImpl<EAutoClassWeightsType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEAutoClassWeightsTypePrivate::TNameBufs, EAutoClassWeightsType>(data, len);
}

template<>
bool TryFromStringImpl<EAutoClassWeightsType>(const char* data, size_t len, EAutoClassWeightsType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEAutoClassWeightsTypePrivate::TNameBufs, EAutoClassWeightsType>(data, len, result);
}

bool FromString(const TString& name, EAutoClassWeightsType& ret) {
    return ::TryFromStringImpl<EAutoClassWeightsType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EAutoClassWeightsType& ret) {
    return ::TryFromStringImpl<EAutoClassWeightsType>(name.data(), name.size(), ret);
}

template<>
void Out<EAutoClassWeightsType>(IOutputStream& os, TTypeTraits<EAutoClassWeightsType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEAutoClassWeightsTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EAutoClassWeightsType>(EAutoClassWeightsType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEAutoClassWeightsTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EAutoClassWeightsType> GetEnumAllValuesImpl<EAutoClassWeightsType>() {
        const NEAutoClassWeightsTypePrivate::TNameBufs& names = NEAutoClassWeightsTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EAutoClassWeightsType>() {
        const NEAutoClassWeightsTypePrivate::TNameBufs& names = NEAutoClassWeightsTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EAutoClassWeightsType, TString> GetEnumNamesImpl<EAutoClassWeightsType>() {
        const NEAutoClassWeightsTypePrivate::TNameBufs& names = NEAutoClassWeightsTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EAutoClassWeightsType>() {
        const NEAutoClassWeightsTypePrivate::TNameBufs& names = NEAutoClassWeightsTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EAucType
namespace { namespace NEAucTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EAucType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(EAucType::Classic, "Classic"sv),
        TNameBufsBase::EnumStringPair(EAucType::Ranking, "Ranking"sv),
        TNameBufsBase::EnumStringPair(EAucType::Mu, "Mu"sv),
        TNameBufsBase::EnumStringPair(EAucType::OneVsAll, "OneVsAll"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(EAucType::Classic, "Classic"sv),
        TNameBufsBase::EnumStringPair(EAucType::Mu, "Mu"sv),
        TNameBufsBase::EnumStringPair(EAucType::OneVsAll, "OneVsAll"sv),
        TNameBufsBase::EnumStringPair(EAucType::Ranking, "Ranking"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "Classic"sv,
        "Ranking"sv,
        "Mu"sv,
        "OneVsAll"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EAucType::"sv,
        "EAucType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EAucType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EAucType>;

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

const TString& ToString(EAucType x) {
    const NEAucTypePrivate::TNameBufs& names = NEAucTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EAucType FromStringImpl<EAucType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEAucTypePrivate::TNameBufs, EAucType>(data, len);
}

template<>
bool TryFromStringImpl<EAucType>(const char* data, size_t len, EAucType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEAucTypePrivate::TNameBufs, EAucType>(data, len, result);
}

bool FromString(const TString& name, EAucType& ret) {
    return ::TryFromStringImpl<EAucType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EAucType& ret) {
    return ::TryFromStringImpl<EAucType>(name.data(), name.size(), ret);
}

template<>
void Out<EAucType>(IOutputStream& os, TTypeTraits<EAucType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEAucTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EAucType>(EAucType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEAucTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EAucType> GetEnumAllValuesImpl<EAucType>() {
        const NEAucTypePrivate::TNameBufs& names = NEAucTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EAucType>() {
        const NEAucTypePrivate::TNameBufs& names = NEAucTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EAucType, TString> GetEnumNamesImpl<EAucType>() {
        const NEAucTypePrivate::TNameBufs& names = NEAucTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EAucType>() {
        const NEAucTypePrivate::TNameBufs& names = NEAucTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EF1AverageType
namespace { namespace NEF1AverageTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EF1AverageType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(EF1AverageType::Micro, "Micro"sv),
        TNameBufsBase::EnumStringPair(EF1AverageType::Macro, "Macro"sv),
        TNameBufsBase::EnumStringPair(EF1AverageType::Weighted, "Weighted"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(EF1AverageType::Macro, "Macro"sv),
        TNameBufsBase::EnumStringPair(EF1AverageType::Micro, "Micro"sv),
        TNameBufsBase::EnumStringPair(EF1AverageType::Weighted, "Weighted"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Micro"sv,
        "Macro"sv,
        "Weighted"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EF1AverageType::"sv,
        "EF1AverageType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EF1AverageType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EF1AverageType>;

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

const TString& ToString(EF1AverageType x) {
    const NEF1AverageTypePrivate::TNameBufs& names = NEF1AverageTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EF1AverageType FromStringImpl<EF1AverageType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEF1AverageTypePrivate::TNameBufs, EF1AverageType>(data, len);
}

template<>
bool TryFromStringImpl<EF1AverageType>(const char* data, size_t len, EF1AverageType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEF1AverageTypePrivate::TNameBufs, EF1AverageType>(data, len, result);
}

bool FromString(const TString& name, EF1AverageType& ret) {
    return ::TryFromStringImpl<EF1AverageType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EF1AverageType& ret) {
    return ::TryFromStringImpl<EF1AverageType>(name.data(), name.size(), ret);
}

template<>
void Out<EF1AverageType>(IOutputStream& os, TTypeTraits<EF1AverageType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEF1AverageTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EF1AverageType>(EF1AverageType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEF1AverageTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EF1AverageType> GetEnumAllValuesImpl<EF1AverageType>() {
        const NEF1AverageTypePrivate::TNameBufs& names = NEF1AverageTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EF1AverageType>() {
        const NEF1AverageTypePrivate::TNameBufs& names = NEF1AverageTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EF1AverageType, TString> GetEnumNamesImpl<EF1AverageType>() {
        const NEF1AverageTypePrivate::TNameBufs& names = NEF1AverageTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EF1AverageType>() {
        const NEF1AverageTypePrivate::TNameBufs& names = NEF1AverageTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for EAccuracyType
namespace { namespace NEAccuracyTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EAccuracyType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(EAccuracyType::Classic, "Classic"sv),
        TNameBufsBase::EnumStringPair(EAccuracyType::PerClass, "PerClass"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(EAccuracyType::Classic, "Classic"sv),
        TNameBufsBase::EnumStringPair(EAccuracyType::PerClass, "PerClass"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Classic"sv,
        "PerClass"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EAccuracyType::"sv,
        "EAccuracyType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EAccuracyType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EAccuracyType>;

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

const TString& ToString(EAccuracyType x) {
    const NEAccuracyTypePrivate::TNameBufs& names = NEAccuracyTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EAccuracyType FromStringImpl<EAccuracyType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEAccuracyTypePrivate::TNameBufs, EAccuracyType>(data, len);
}

template<>
bool TryFromStringImpl<EAccuracyType>(const char* data, size_t len, EAccuracyType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEAccuracyTypePrivate::TNameBufs, EAccuracyType>(data, len, result);
}

bool FromString(const TString& name, EAccuracyType& ret) {
    return ::TryFromStringImpl<EAccuracyType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EAccuracyType& ret) {
    return ::TryFromStringImpl<EAccuracyType>(name.data(), name.size(), ret);
}

template<>
void Out<EAccuracyType>(IOutputStream& os, TTypeTraits<EAccuracyType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEAccuracyTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EAccuracyType>(EAccuracyType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEAccuracyTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EAccuracyType> GetEnumAllValuesImpl<EAccuracyType>() {
        const NEAccuracyTypePrivate::TNameBufs& names = NEAccuracyTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EAccuracyType>() {
        const NEAccuracyTypePrivate::TNameBufs& names = NEAccuracyTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EAccuracyType, TString> GetEnumNamesImpl<EAccuracyType>() {
        const NEAccuracyTypePrivate::TNameBufs& names = NEAccuracyTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EAccuracyType>() {
        const NEAccuracyTypePrivate::TNameBufs& names = NEAccuracyTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NCB::EFeatureEvalMode
namespace { namespace NNCBEFeatureEvalModePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeatureEvalMode>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(NCB::EFeatureEvalMode::OneVsNone, "OneVsNone"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeatureEvalMode::OneVsOthers, "OneVsOthers"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeatureEvalMode::OneVsAll, "OneVsAll"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeatureEvalMode::OthersVsAll, "OthersVsAll"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(NCB::EFeatureEvalMode::OneVsAll, "OneVsAll"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeatureEvalMode::OneVsNone, "OneVsNone"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeatureEvalMode::OneVsOthers, "OneVsOthers"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeatureEvalMode::OthersVsAll, "OthersVsAll"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "OneVsNone"sv,
        "OneVsOthers"sv,
        "OneVsAll"sv,
        "OthersVsAll"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NCB::EFeatureEvalMode::"sv,
        "NCB::EFeatureEvalMode"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeatureEvalMode> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeatureEvalMode>;

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

const TString& ToString(NCB::EFeatureEvalMode x) {
    const NNCBEFeatureEvalModePrivate::TNameBufs& names = NNCBEFeatureEvalModePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NCB::EFeatureEvalMode FromStringImpl<NCB::EFeatureEvalMode>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNCBEFeatureEvalModePrivate::TNameBufs, NCB::EFeatureEvalMode>(data, len);
}

template<>
bool TryFromStringImpl<NCB::EFeatureEvalMode>(const char* data, size_t len, NCB::EFeatureEvalMode& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNCBEFeatureEvalModePrivate::TNameBufs, NCB::EFeatureEvalMode>(data, len, result);
}

bool FromString(const TString& name, NCB::EFeatureEvalMode& ret) {
    return ::TryFromStringImpl<NCB::EFeatureEvalMode>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NCB::EFeatureEvalMode& ret) {
    return ::TryFromStringImpl<NCB::EFeatureEvalMode>(name.data(), name.size(), ret);
}

template<>
void Out<NCB::EFeatureEvalMode>(IOutputStream& os, TTypeTraits<NCB::EFeatureEvalMode>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNCBEFeatureEvalModePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NCB::EFeatureEvalMode>(NCB::EFeatureEvalMode e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNCBEFeatureEvalModePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NCB::EFeatureEvalMode> GetEnumAllValuesImpl<NCB::EFeatureEvalMode>() {
        const NNCBEFeatureEvalModePrivate::TNameBufs& names = NNCBEFeatureEvalModePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NCB::EFeatureEvalMode>() {
        const NNCBEFeatureEvalModePrivate::TNameBufs& names = NNCBEFeatureEvalModePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NCB::EFeatureEvalMode, TString> GetEnumNamesImpl<NCB::EFeatureEvalMode>() {
        const NNCBEFeatureEvalModePrivate::TNameBufs& names = NNCBEFeatureEvalModePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NCB::EFeatureEvalMode>() {
        const NNCBEFeatureEvalModePrivate::TNameBufs& names = NNCBEFeatureEvalModePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NCB::ERawTargetType
namespace { namespace NNCBERawTargetTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::ERawTargetType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 5> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 5>{{
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::Boolean, "Boolean"sv),
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::Integer, "Integer"sv),
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::Float, "Float"sv),
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::String, "String"sv),
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::None, "None"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[5]{
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::Boolean, "Boolean"sv),
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::Float, "Float"sv),
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::Integer, "Integer"sv),
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::None, "None"sv),
        TNameBufsBase::EnumStringPair(NCB::ERawTargetType::String, "String"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[5]{
        "Boolean"sv,
        "Integer"sv,
        "Float"sv,
        "String"sv,
        "None"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NCB::ERawTargetType::"sv,
        "NCB::ERawTargetType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NCB::ERawTargetType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::ERawTargetType>;

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

const TString& ToString(NCB::ERawTargetType x) {
    const NNCBERawTargetTypePrivate::TNameBufs& names = NNCBERawTargetTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NCB::ERawTargetType FromStringImpl<NCB::ERawTargetType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNCBERawTargetTypePrivate::TNameBufs, NCB::ERawTargetType>(data, len);
}

template<>
bool TryFromStringImpl<NCB::ERawTargetType>(const char* data, size_t len, NCB::ERawTargetType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNCBERawTargetTypePrivate::TNameBufs, NCB::ERawTargetType>(data, len, result);
}

bool FromString(const TString& name, NCB::ERawTargetType& ret) {
    return ::TryFromStringImpl<NCB::ERawTargetType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NCB::ERawTargetType& ret) {
    return ::TryFromStringImpl<NCB::ERawTargetType>(name.data(), name.size(), ret);
}

template<>
void Out<NCB::ERawTargetType>(IOutputStream& os, TTypeTraits<NCB::ERawTargetType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNCBERawTargetTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NCB::ERawTargetType>(NCB::ERawTargetType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNCBERawTargetTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NCB::ERawTargetType> GetEnumAllValuesImpl<NCB::ERawTargetType>() {
        const NNCBERawTargetTypePrivate::TNameBufs& names = NNCBERawTargetTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NCB::ERawTargetType>() {
        const NNCBERawTargetTypePrivate::TNameBufs& names = NNCBERawTargetTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NCB::ERawTargetType, TString> GetEnumNamesImpl<NCB::ERawTargetType>() {
        const NNCBERawTargetTypePrivate::TNameBufs& names = NNCBERawTargetTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NCB::ERawTargetType>() {
        const NNCBERawTargetTypePrivate::TNameBufs& names = NNCBERawTargetTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NCB::EFeaturesSelectionAlgorithm
namespace { namespace NNCBEFeaturesSelectionAlgorithmPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeaturesSelectionAlgorithm>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange, "RecursiveByPredictionValuesChange"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionAlgorithm::RecursiveByLossFunctionChange, "RecursiveByLossFunctionChange"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionAlgorithm::RecursiveByShapValues, "RecursiveByShapValues"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionAlgorithm::RecursiveByLossFunctionChange, "RecursiveByLossFunctionChange"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionAlgorithm::RecursiveByPredictionValuesChange, "RecursiveByPredictionValuesChange"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionAlgorithm::RecursiveByShapValues, "RecursiveByShapValues"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "RecursiveByPredictionValuesChange"sv,
        "RecursiveByLossFunctionChange"sv,
        "RecursiveByShapValues"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NCB::EFeaturesSelectionAlgorithm::"sv,
        "NCB::EFeaturesSelectionAlgorithm"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeaturesSelectionAlgorithm> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeaturesSelectionAlgorithm>;

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

const TString& ToString(NCB::EFeaturesSelectionAlgorithm x) {
    const NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs& names = NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NCB::EFeaturesSelectionAlgorithm FromStringImpl<NCB::EFeaturesSelectionAlgorithm>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs, NCB::EFeaturesSelectionAlgorithm>(data, len);
}

template<>
bool TryFromStringImpl<NCB::EFeaturesSelectionAlgorithm>(const char* data, size_t len, NCB::EFeaturesSelectionAlgorithm& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs, NCB::EFeaturesSelectionAlgorithm>(data, len, result);
}

bool FromString(const TString& name, NCB::EFeaturesSelectionAlgorithm& ret) {
    return ::TryFromStringImpl<NCB::EFeaturesSelectionAlgorithm>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NCB::EFeaturesSelectionAlgorithm& ret) {
    return ::TryFromStringImpl<NCB::EFeaturesSelectionAlgorithm>(name.data(), name.size(), ret);
}

template<>
void Out<NCB::EFeaturesSelectionAlgorithm>(IOutputStream& os, TTypeTraits<NCB::EFeaturesSelectionAlgorithm>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NCB::EFeaturesSelectionAlgorithm>(NCB::EFeaturesSelectionAlgorithm e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NCB::EFeaturesSelectionAlgorithm> GetEnumAllValuesImpl<NCB::EFeaturesSelectionAlgorithm>() {
        const NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs& names = NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NCB::EFeaturesSelectionAlgorithm>() {
        const NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs& names = NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NCB::EFeaturesSelectionAlgorithm, TString> GetEnumNamesImpl<NCB::EFeaturesSelectionAlgorithm>() {
        const NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs& names = NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NCB::EFeaturesSelectionAlgorithm>() {
        const NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs& names = NNCBEFeaturesSelectionAlgorithmPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NCB::EFeaturesSelectionGrouping
namespace { namespace NNCBEFeaturesSelectionGroupingPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeaturesSelectionGrouping>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionGrouping::Individual, "Individual"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionGrouping::ByTags, "ByTags"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionGrouping::ByTags, "ByTags"sv),
        TNameBufsBase::EnumStringPair(NCB::EFeaturesSelectionGrouping::Individual, "Individual"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Individual"sv,
        "ByTags"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NCB::EFeaturesSelectionGrouping::"sv,
        "NCB::EFeaturesSelectionGrouping"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeaturesSelectionGrouping> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NCB::EFeaturesSelectionGrouping>;

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

const TString& ToString(NCB::EFeaturesSelectionGrouping x) {
    const NNCBEFeaturesSelectionGroupingPrivate::TNameBufs& names = NNCBEFeaturesSelectionGroupingPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NCB::EFeaturesSelectionGrouping FromStringImpl<NCB::EFeaturesSelectionGrouping>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNCBEFeaturesSelectionGroupingPrivate::TNameBufs, NCB::EFeaturesSelectionGrouping>(data, len);
}

template<>
bool TryFromStringImpl<NCB::EFeaturesSelectionGrouping>(const char* data, size_t len, NCB::EFeaturesSelectionGrouping& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNCBEFeaturesSelectionGroupingPrivate::TNameBufs, NCB::EFeaturesSelectionGrouping>(data, len, result);
}

bool FromString(const TString& name, NCB::EFeaturesSelectionGrouping& ret) {
    return ::TryFromStringImpl<NCB::EFeaturesSelectionGrouping>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NCB::EFeaturesSelectionGrouping& ret) {
    return ::TryFromStringImpl<NCB::EFeaturesSelectionGrouping>(name.data(), name.size(), ret);
}

template<>
void Out<NCB::EFeaturesSelectionGrouping>(IOutputStream& os, TTypeTraits<NCB::EFeaturesSelectionGrouping>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNCBEFeaturesSelectionGroupingPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NCB::EFeaturesSelectionGrouping>(NCB::EFeaturesSelectionGrouping e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNCBEFeaturesSelectionGroupingPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NCB::EFeaturesSelectionGrouping> GetEnumAllValuesImpl<NCB::EFeaturesSelectionGrouping>() {
        const NNCBEFeaturesSelectionGroupingPrivate::TNameBufs& names = NNCBEFeaturesSelectionGroupingPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NCB::EFeaturesSelectionGrouping>() {
        const NNCBEFeaturesSelectionGroupingPrivate::TNameBufs& names = NNCBEFeaturesSelectionGroupingPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NCB::EFeaturesSelectionGrouping, TString> GetEnumNamesImpl<NCB::EFeaturesSelectionGrouping>() {
        const NNCBEFeaturesSelectionGroupingPrivate::TNameBufs& names = NNCBEFeaturesSelectionGroupingPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NCB::EFeaturesSelectionGrouping>() {
        const NNCBEFeaturesSelectionGroupingPrivate::TNameBufs& names = NNCBEFeaturesSelectionGroupingPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

