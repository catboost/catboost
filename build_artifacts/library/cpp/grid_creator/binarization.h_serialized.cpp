// This file was auto-generated. Do not edit!!!
#include <library/cpp/grid_creator/binarization.h>
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

// I/O for EBorderSelectionType
namespace { namespace NEBorderSelectionTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<EBorderSelectionType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 7> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 7>{{
        TNameBufsBase::EnumStringPair(EBorderSelectionType::Median, "Median"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::GreedyLogSum, "GreedyLogSum"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::UniformAndQuantiles, "UniformAndQuantiles"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::MinEntropy, "MinEntropy"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::MaxLogSum, "MaxLogSum"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::Uniform, "Uniform"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::GreedyMinEntropy, "GreedyMinEntropy"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[7]{
        TNameBufsBase::EnumStringPair(EBorderSelectionType::GreedyLogSum, "GreedyLogSum"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::GreedyMinEntropy, "GreedyMinEntropy"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::MaxLogSum, "MaxLogSum"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::Median, "Median"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::MinEntropy, "MinEntropy"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::Uniform, "Uniform"sv),
        TNameBufsBase::EnumStringPair(EBorderSelectionType::UniformAndQuantiles, "UniformAndQuantiles"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[7]{
        "Median"sv,
        "GreedyLogSum"sv,
        "UniformAndQuantiles"sv,
        "MinEntropy"sv,
        "MaxLogSum"sv,
        "Uniform"sv,
        "GreedyMinEntropy"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "EBorderSelectionType::"sv,
        "EBorderSelectionType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<EBorderSelectionType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<EBorderSelectionType>;

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

const TString& ToString(EBorderSelectionType x) {
    const NEBorderSelectionTypePrivate::TNameBufs& names = NEBorderSelectionTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
EBorderSelectionType FromStringImpl<EBorderSelectionType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NEBorderSelectionTypePrivate::TNameBufs, EBorderSelectionType>(data, len);
}

template<>
bool TryFromStringImpl<EBorderSelectionType>(const char* data, size_t len, EBorderSelectionType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NEBorderSelectionTypePrivate::TNameBufs, EBorderSelectionType>(data, len, result);
}

bool FromString(const TString& name, EBorderSelectionType& ret) {
    return ::TryFromStringImpl<EBorderSelectionType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, EBorderSelectionType& ret) {
    return ::TryFromStringImpl<EBorderSelectionType>(name.data(), name.size(), ret);
}

template<>
void Out<EBorderSelectionType>(IOutputStream& os, TTypeTraits<EBorderSelectionType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NEBorderSelectionTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<EBorderSelectionType>(EBorderSelectionType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NEBorderSelectionTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<EBorderSelectionType> GetEnumAllValuesImpl<EBorderSelectionType>() {
        const NEBorderSelectionTypePrivate::TNameBufs& names = NEBorderSelectionTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<EBorderSelectionType>() {
        const NEBorderSelectionTypePrivate::TNameBufs& names = NEBorderSelectionTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<EBorderSelectionType, TString> GetEnumNamesImpl<EBorderSelectionType>() {
        const NEBorderSelectionTypePrivate::TNameBufs& names = NEBorderSelectionTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<EBorderSelectionType>() {
        const NEBorderSelectionTypePrivate::TNameBufs& names = NEBorderSelectionTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NSplitSelection::NImpl::EPenaltyType
namespace { namespace NNSplitSelectionNImplEPenaltyTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NSplitSelection::NImpl::EPenaltyType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EPenaltyType::MinEntropy, "MinEntropy"sv),
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EPenaltyType::MaxSumLog, "MaxSumLog"sv),
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EPenaltyType::W2, "W2"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EPenaltyType::MaxSumLog, "MaxSumLog"sv),
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EPenaltyType::MinEntropy, "MinEntropy"sv),
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EPenaltyType::W2, "W2"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "MinEntropy"sv,
        "MaxSumLog"sv,
        "W2"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NSplitSelection::NImpl::EPenaltyType::"sv,
        "NSplitSelection::NImpl::EPenaltyType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NSplitSelection::NImpl::EPenaltyType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NSplitSelection::NImpl::EPenaltyType>;

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

const TString& ToString(NSplitSelection::NImpl::EPenaltyType x) {
    const NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs& names = NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NSplitSelection::NImpl::EPenaltyType FromStringImpl<NSplitSelection::NImpl::EPenaltyType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs, NSplitSelection::NImpl::EPenaltyType>(data, len);
}

template<>
bool TryFromStringImpl<NSplitSelection::NImpl::EPenaltyType>(const char* data, size_t len, NSplitSelection::NImpl::EPenaltyType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs, NSplitSelection::NImpl::EPenaltyType>(data, len, result);
}

bool FromString(const TString& name, NSplitSelection::NImpl::EPenaltyType& ret) {
    return ::TryFromStringImpl<NSplitSelection::NImpl::EPenaltyType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NSplitSelection::NImpl::EPenaltyType& ret) {
    return ::TryFromStringImpl<NSplitSelection::NImpl::EPenaltyType>(name.data(), name.size(), ret);
}

template<>
void Out<NSplitSelection::NImpl::EPenaltyType>(IOutputStream& os, TTypeTraits<NSplitSelection::NImpl::EPenaltyType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NSplitSelection::NImpl::EPenaltyType>(NSplitSelection::NImpl::EPenaltyType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NSplitSelection::NImpl::EPenaltyType> GetEnumAllValuesImpl<NSplitSelection::NImpl::EPenaltyType>() {
        const NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs& names = NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NSplitSelection::NImpl::EPenaltyType>() {
        const NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs& names = NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NSplitSelection::NImpl::EPenaltyType, TString> GetEnumNamesImpl<NSplitSelection::NImpl::EPenaltyType>() {
        const NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs& names = NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NSplitSelection::NImpl::EPenaltyType>() {
        const NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs& names = NNSplitSelectionNImplEPenaltyTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NSplitSelection::NImpl::EOptimizationType
namespace { namespace NNSplitSelectionNImplEOptimizationTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NSplitSelection::NImpl::EOptimizationType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EOptimizationType::Exact, "Exact"sv),
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EOptimizationType::Greedy, "Greedy"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EOptimizationType::Exact, "Exact"sv),
        TNameBufsBase::EnumStringPair(NSplitSelection::NImpl::EOptimizationType::Greedy, "Greedy"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Exact"sv,
        "Greedy"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NSplitSelection::NImpl::EOptimizationType::"sv,
        "NSplitSelection::NImpl::EOptimizationType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NSplitSelection::NImpl::EOptimizationType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NSplitSelection::NImpl::EOptimizationType>;

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

const TString& ToString(NSplitSelection::NImpl::EOptimizationType x) {
    const NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs& names = NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NSplitSelection::NImpl::EOptimizationType FromStringImpl<NSplitSelection::NImpl::EOptimizationType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs, NSplitSelection::NImpl::EOptimizationType>(data, len);
}

template<>
bool TryFromStringImpl<NSplitSelection::NImpl::EOptimizationType>(const char* data, size_t len, NSplitSelection::NImpl::EOptimizationType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs, NSplitSelection::NImpl::EOptimizationType>(data, len, result);
}

bool FromString(const TString& name, NSplitSelection::NImpl::EOptimizationType& ret) {
    return ::TryFromStringImpl<NSplitSelection::NImpl::EOptimizationType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NSplitSelection::NImpl::EOptimizationType& ret) {
    return ::TryFromStringImpl<NSplitSelection::NImpl::EOptimizationType>(name.data(), name.size(), ret);
}

template<>
void Out<NSplitSelection::NImpl::EOptimizationType>(IOutputStream& os, TTypeTraits<NSplitSelection::NImpl::EOptimizationType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NSplitSelection::NImpl::EOptimizationType>(NSplitSelection::NImpl::EOptimizationType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NSplitSelection::NImpl::EOptimizationType> GetEnumAllValuesImpl<NSplitSelection::NImpl::EOptimizationType>() {
        const NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs& names = NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NSplitSelection::NImpl::EOptimizationType>() {
        const NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs& names = NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NSplitSelection::NImpl::EOptimizationType, TString> GetEnumNamesImpl<NSplitSelection::NImpl::EOptimizationType>() {
        const NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs& names = NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NSplitSelection::NImpl::EOptimizationType>() {
        const NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs& names = NNSplitSelectionNImplEOptimizationTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

