// This file was auto-generated. Do not edit!!!
#include <library/cpp/token/formtype.h>
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

// I/O for TFormType
namespace { namespace NTFormTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<TFormType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 4> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 4>{{
        TNameBufsBase::EnumStringPair(fGeneral, "fGeneral"sv),
        TNameBufsBase::EnumStringPair(fExactWord, "fExactWord"sv),
        TNameBufsBase::EnumStringPair(fExactLemma, "fExactLemma"sv),
        TNameBufsBase::EnumStringPair(fWeirdExactWord, "fWeirdExactWord"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[4]{
        TNameBufsBase::EnumStringPair(fExactLemma, "fExactLemma"sv),
        TNameBufsBase::EnumStringPair(fExactWord, "fExactWord"sv),
        TNameBufsBase::EnumStringPair(fGeneral, "fGeneral"sv),
        TNameBufsBase::EnumStringPair(fWeirdExactWord, "fWeirdExactWord"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[4]{
        "fGeneral"sv,
        "fExactWord"sv,
        "fExactLemma"sv,
        "fWeirdExactWord"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        ""sv,
        "TFormType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<TFormType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<TFormType>;

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

const TString& ToString(TFormType x) {
    const NTFormTypePrivate::TNameBufs& names = NTFormTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
TFormType FromStringImpl<TFormType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NTFormTypePrivate::TNameBufs, TFormType>(data, len);
}

template<>
bool TryFromStringImpl<TFormType>(const char* data, size_t len, TFormType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NTFormTypePrivate::TNameBufs, TFormType>(data, len, result);
}

bool FromString(const TString& name, TFormType& ret) {
    return ::TryFromStringImpl<TFormType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, TFormType& ret) {
    return ::TryFromStringImpl<TFormType>(name.data(), name.size(), ret);
}

template<>
void Out<TFormType>(IOutputStream& os, TTypeTraits<TFormType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NTFormTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<TFormType>(TFormType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NTFormTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<TFormType> GetEnumAllValuesImpl<TFormType>() {
        const NTFormTypePrivate::TNameBufs& names = NTFormTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<TFormType>() {
        const NTFormTypePrivate::TNameBufs& names = NTFormTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<TFormType, TString> GetEnumNamesImpl<TFormType>() {
        const NTFormTypePrivate::TNameBufs& names = NTFormTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<TFormType>() {
        const NTFormTypePrivate::TNameBufs& names = NTFormTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

