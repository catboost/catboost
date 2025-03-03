// This file was auto-generated. Do not edit!!!
#include <catboost/libs/logging/logging_level.h>
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

// I/O for ELoggingLevel
namespace { namespace NELoggingLevelPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ELoggingLevel>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(Silent, "Silent"sv),
        TNameBufsBase::EnumStringPair(Verbose, "Verbose"sv),
        TNameBufsBase::EnumStringPair(Info, "Info"sv),
        TNameBufsBase::EnumStringPair(Debug, "Debug"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(Debug, "Debug"sv),
        TNameBufsBase::EnumStringPair(Info, "Info"sv),
        TNameBufsBase::EnumStringPair(Silent, "Silent"sv),
        TNameBufsBase::EnumStringPair(Verbose, "Verbose"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "Silent"sv,
        "Verbose"sv,
        "Info"sv,
        "Debug"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        ""sv,
        "ELoggingLevel"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ELoggingLevel> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ELoggingLevel>;

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

const TString& ToString(ELoggingLevel x) {
    const NELoggingLevelPrivate::TNameBufs& names = NELoggingLevelPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ELoggingLevel FromStringImpl<ELoggingLevel>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NELoggingLevelPrivate::TNameBufs, ELoggingLevel>(data, len);
}

template<>
bool TryFromStringImpl<ELoggingLevel>(const char* data, size_t len, ELoggingLevel& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NELoggingLevelPrivate::TNameBufs, ELoggingLevel>(data, len, result);
}

bool FromString(const TString& name, ELoggingLevel& ret) {
    return ::TryFromStringImpl<ELoggingLevel>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ELoggingLevel& ret) {
    return ::TryFromStringImpl<ELoggingLevel>(name.data(), name.size(), ret);
}

template<>
void Out<ELoggingLevel>(IOutputStream& os, TTypeTraits<ELoggingLevel>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NELoggingLevelPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ELoggingLevel>(ELoggingLevel e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NELoggingLevelPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ELoggingLevel> GetEnumAllValuesImpl<ELoggingLevel>() {
        const NELoggingLevelPrivate::TNameBufs& names = NELoggingLevelPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ELoggingLevel>() {
        const NELoggingLevelPrivate::TNameBufs& names = NELoggingLevelPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ELoggingLevel, TString> GetEnumNamesImpl<ELoggingLevel>() {
        const NELoggingLevelPrivate::TNameBufs& names = NELoggingLevelPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ELoggingLevel>() {
        const NELoggingLevelPrivate::TNameBufs& names = NELoggingLevelPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

