// This file was auto-generated. Do not edit!!!
#include <library/cpp/logger/priority.h>
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

// I/O for ELogPriority
namespace { namespace NELogPriorityPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ELogPriority>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 9> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 9>{{
        TNameBufsBase::EnumStringPair(TLOG_EMERG, "EMERG"sv),
        TNameBufsBase::EnumStringPair(TLOG_ALERT, "ALERT"sv),
        TNameBufsBase::EnumStringPair(TLOG_CRIT, "CRITICAL_INFO"sv),
        TNameBufsBase::EnumStringPair(TLOG_ERR, "ERROR"sv),
        TNameBufsBase::EnumStringPair(TLOG_WARNING, "WARNING"sv),
        TNameBufsBase::EnumStringPair(TLOG_NOTICE, "NOTICE"sv),
        TNameBufsBase::EnumStringPair(TLOG_INFO, "INFO"sv),
        TNameBufsBase::EnumStringPair(TLOG_DEBUG, "DEBUG"sv),
        TNameBufsBase::EnumStringPair(TLOG_RESOURCES, "RESOURCES"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[9]{
        TNameBufsBase::EnumStringPair(TLOG_ALERT, "ALERT"sv),
        TNameBufsBase::EnumStringPair(TLOG_CRIT, "CRITICAL_INFO"sv),
        TNameBufsBase::EnumStringPair(TLOG_DEBUG, "DEBUG"sv),
        TNameBufsBase::EnumStringPair(TLOG_EMERG, "EMERG"sv),
        TNameBufsBase::EnumStringPair(TLOG_ERR, "ERROR"sv),
        TNameBufsBase::EnumStringPair(TLOG_INFO, "INFO"sv),
        TNameBufsBase::EnumStringPair(TLOG_NOTICE, "NOTICE"sv),
        TNameBufsBase::EnumStringPair(TLOG_RESOURCES, "RESOURCES"sv),
        TNameBufsBase::EnumStringPair(TLOG_WARNING, "WARNING"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[9]{
        "TLOG_EMERG"sv,
        "TLOG_ALERT"sv,
        "TLOG_CRIT"sv,
        "TLOG_ERR"sv,
        "TLOG_WARNING"sv,
        "TLOG_NOTICE"sv,
        "TLOG_INFO"sv,
        "TLOG_DEBUG"sv,
        "TLOG_RESOURCES"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        ""sv,
        "ELogPriority"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ELogPriority> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ELogPriority>;

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

const TString& ToString(ELogPriority x) {
    const NELogPriorityPrivate::TNameBufs& names = NELogPriorityPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ELogPriority FromStringImpl<ELogPriority>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NELogPriorityPrivate::TNameBufs, ELogPriority>(data, len);
}

template<>
bool TryFromStringImpl<ELogPriority>(const char* data, size_t len, ELogPriority& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NELogPriorityPrivate::TNameBufs, ELogPriority>(data, len, result);
}

bool FromString(const TString& name, ELogPriority& ret) {
    return ::TryFromStringImpl<ELogPriority>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ELogPriority& ret) {
    return ::TryFromStringImpl<ELogPriority>(name.data(), name.size(), ret);
}

template<>
void Out<ELogPriority>(IOutputStream& os, TTypeTraits<ELogPriority>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NELogPriorityPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ELogPriority>(ELogPriority e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NELogPriorityPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ELogPriority> GetEnumAllValuesImpl<ELogPriority>() {
        const NELogPriorityPrivate::TNameBufs& names = NELogPriorityPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ELogPriority>() {
        const NELogPriorityPrivate::TNameBufs& names = NELogPriorityPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ELogPriority, TString> GetEnumNamesImpl<ELogPriority>() {
        const NELogPriorityPrivate::TNameBufs& names = NELogPriorityPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ELogPriority>() {
        const NELogPriorityPrivate::TNameBufs& names = NELogPriorityPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

