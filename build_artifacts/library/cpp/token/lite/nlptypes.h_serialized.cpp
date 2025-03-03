// This file was auto-generated. Do not edit!!!
#include <library/cpp/token/nlptypes.h>
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

// I/O for NLP_TYPE
namespace { namespace NNLP_TYPEPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NLP_TYPE>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 8> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 8>{{
        TNameBufsBase::EnumStringPair(NLP_END, "NLP_END"sv),
        TNameBufsBase::EnumStringPair(NLP_WORD, "NLP_WORD"sv),
        TNameBufsBase::EnumStringPair(NLP_INTEGER, "NLP_INTEGER"sv),
        TNameBufsBase::EnumStringPair(NLP_FLOAT, "NLP_FLOAT"sv),
        TNameBufsBase::EnumStringPair(NLP_MARK, "NLP_MARK"sv),
        TNameBufsBase::EnumStringPair(NLP_SENTBREAK, "NLP_SENTBREAK"sv),
        TNameBufsBase::EnumStringPair(NLP_PARABREAK, "NLP_PARABREAK"sv),
        TNameBufsBase::EnumStringPair(NLP_MISCTEXT, "NLP_MISCTEXT"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[8]{
        TNameBufsBase::EnumStringPair(NLP_END, "NLP_END"sv),
        TNameBufsBase::EnumStringPair(NLP_FLOAT, "NLP_FLOAT"sv),
        TNameBufsBase::EnumStringPair(NLP_INTEGER, "NLP_INTEGER"sv),
        TNameBufsBase::EnumStringPair(NLP_MARK, "NLP_MARK"sv),
        TNameBufsBase::EnumStringPair(NLP_MISCTEXT, "NLP_MISCTEXT"sv),
        TNameBufsBase::EnumStringPair(NLP_PARABREAK, "NLP_PARABREAK"sv),
        TNameBufsBase::EnumStringPair(NLP_SENTBREAK, "NLP_SENTBREAK"sv),
        TNameBufsBase::EnumStringPair(NLP_WORD, "NLP_WORD"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[8]{
        "NLP_END"sv,
        "NLP_WORD"sv,
        "NLP_INTEGER"sv,
        "NLP_FLOAT"sv,
        "NLP_MARK"sv,
        "NLP_SENTBREAK"sv,
        "NLP_PARABREAK"sv,
        "NLP_MISCTEXT"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        ""sv,
        "NLP_TYPE"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NLP_TYPE> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NLP_TYPE>;

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

const TString& ToString(NLP_TYPE x) {
    const NNLP_TYPEPrivate::TNameBufs& names = NNLP_TYPEPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NLP_TYPE FromStringImpl<NLP_TYPE>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNLP_TYPEPrivate::TNameBufs, NLP_TYPE>(data, len);
}

template<>
bool TryFromStringImpl<NLP_TYPE>(const char* data, size_t len, NLP_TYPE& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNLP_TYPEPrivate::TNameBufs, NLP_TYPE>(data, len, result);
}

bool FromString(const TString& name, NLP_TYPE& ret) {
    return ::TryFromStringImpl<NLP_TYPE>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NLP_TYPE& ret) {
    return ::TryFromStringImpl<NLP_TYPE>(name.data(), name.size(), ret);
}

template<>
void Out<NLP_TYPE>(IOutputStream& os, TTypeTraits<NLP_TYPE>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNLP_TYPEPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NLP_TYPE>(NLP_TYPE e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNLP_TYPEPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NLP_TYPE> GetEnumAllValuesImpl<NLP_TYPE>() {
        const NNLP_TYPEPrivate::TNameBufs& names = NNLP_TYPEPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NLP_TYPE>() {
        const NNLP_TYPEPrivate::TNameBufs& names = NNLP_TYPEPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NLP_TYPE, TString> GetEnumNamesImpl<NLP_TYPE>() {
        const NNLP_TYPEPrivate::TNameBufs& names = NNLP_TYPEPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NLP_TYPE>() {
        const NNLP_TYPEPrivate::TNameBufs& names = NNLP_TYPEPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for ESpaceType
namespace { namespace NESpaceTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<ESpaceType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 5> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 5>{{
        TNameBufsBase::EnumStringPair(ST_NOBRK, "ST_NOBRK"sv),
        TNameBufsBase::EnumStringPair(ST_SENTBRK, "ST_SENTBRK"sv),
        TNameBufsBase::EnumStringPair(ST_PARABRK, "ST_PARABRK"sv),
        TNameBufsBase::EnumStringPair(ST_ZONEOPN, "ST_ZONEOPN"sv),
        TNameBufsBase::EnumStringPair(ST_ZONECLS, "ST_ZONECLS"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[5]{
        TNameBufsBase::EnumStringPair(ST_NOBRK, "ST_NOBRK"sv),
        TNameBufsBase::EnumStringPair(ST_PARABRK, "ST_PARABRK"sv),
        TNameBufsBase::EnumStringPair(ST_SENTBRK, "ST_SENTBRK"sv),
        TNameBufsBase::EnumStringPair(ST_ZONECLS, "ST_ZONECLS"sv),
        TNameBufsBase::EnumStringPair(ST_ZONEOPN, "ST_ZONEOPN"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[5]{
        "ST_NOBRK"sv,
        "ST_SENTBRK"sv,
        "ST_PARABRK"sv,
        "ST_ZONEOPN"sv,
        "ST_ZONECLS"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        ""sv,
        "ESpaceType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<ESpaceType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<ESpaceType>;

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

const TString& ToString(ESpaceType x) {
    const NESpaceTypePrivate::TNameBufs& names = NESpaceTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
ESpaceType FromStringImpl<ESpaceType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NESpaceTypePrivate::TNameBufs, ESpaceType>(data, len);
}

template<>
bool TryFromStringImpl<ESpaceType>(const char* data, size_t len, ESpaceType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NESpaceTypePrivate::TNameBufs, ESpaceType>(data, len, result);
}

bool FromString(const TString& name, ESpaceType& ret) {
    return ::TryFromStringImpl<ESpaceType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, ESpaceType& ret) {
    return ::TryFromStringImpl<ESpaceType>(name.data(), name.size(), ret);
}

template<>
void Out<ESpaceType>(IOutputStream& os, TTypeTraits<ESpaceType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NESpaceTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<ESpaceType>(ESpaceType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NESpaceTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<ESpaceType> GetEnumAllValuesImpl<ESpaceType>() {
        const NESpaceTypePrivate::TNameBufs& names = NESpaceTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<ESpaceType>() {
        const NESpaceTypePrivate::TNameBufs& names = NESpaceTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<ESpaceType, TString> GetEnumNamesImpl<ESpaceType>() {
        const NESpaceTypePrivate::TNameBufs& names = NESpaceTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<ESpaceType>() {
        const NESpaceTypePrivate::TNameBufs& names = NESpaceTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

