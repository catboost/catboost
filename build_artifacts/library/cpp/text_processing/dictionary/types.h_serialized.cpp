// This file was auto-generated. Do not edit!!!
#include <library/cpp/text_processing/dictionary/types.h>
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

// I/O for NTextProcessing::NDictionary::ETokenLevelType
namespace { namespace NNTextProcessingNDictionaryETokenLevelTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::ETokenLevelType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::ETokenLevelType::Word, "Word"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::ETokenLevelType::Letter, "Letter"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::ETokenLevelType::Letter, "Letter"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::ETokenLevelType::Word, "Word"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Word"sv,
        "Letter"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NDictionary::ETokenLevelType::"sv,
        "NTextProcessing::NDictionary::ETokenLevelType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::ETokenLevelType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::ETokenLevelType>;

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

const TString& ToString(NTextProcessing::NDictionary::ETokenLevelType x) {
    const NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NDictionary::ETokenLevelType FromStringImpl<NTextProcessing::NDictionary::ETokenLevelType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs, NTextProcessing::NDictionary::ETokenLevelType>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NDictionary::ETokenLevelType>(const char* data, size_t len, NTextProcessing::NDictionary::ETokenLevelType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs, NTextProcessing::NDictionary::ETokenLevelType>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NDictionary::ETokenLevelType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::ETokenLevelType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NDictionary::ETokenLevelType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::ETokenLevelType>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NDictionary::ETokenLevelType>(IOutputStream& os, TTypeTraits<NTextProcessing::NDictionary::ETokenLevelType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NDictionary::ETokenLevelType>(NTextProcessing::NDictionary::ETokenLevelType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NDictionary::ETokenLevelType> GetEnumAllValuesImpl<NTextProcessing::NDictionary::ETokenLevelType>() {
        const NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NDictionary::ETokenLevelType>() {
        const NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NDictionary::ETokenLevelType, TString> GetEnumNamesImpl<NTextProcessing::NDictionary::ETokenLevelType>() {
        const NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NDictionary::ETokenLevelType>() {
        const NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryETokenLevelTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NTextProcessing::NDictionary::EUnknownTokenPolicy
namespace { namespace NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EUnknownTokenPolicy>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EUnknownTokenPolicy::Skip, "Skip"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EUnknownTokenPolicy::Insert, "Insert"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EUnknownTokenPolicy::Insert, "Insert"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EUnknownTokenPolicy::Skip, "Skip"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Skip"sv,
        "Insert"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NDictionary::EUnknownTokenPolicy::"sv,
        "NTextProcessing::NDictionary::EUnknownTokenPolicy"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EUnknownTokenPolicy> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EUnknownTokenPolicy>;

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

const TString& ToString(NTextProcessing::NDictionary::EUnknownTokenPolicy x) {
    const NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NDictionary::EUnknownTokenPolicy FromStringImpl<NTextProcessing::NDictionary::EUnknownTokenPolicy>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs, NTextProcessing::NDictionary::EUnknownTokenPolicy>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NDictionary::EUnknownTokenPolicy>(const char* data, size_t len, NTextProcessing::NDictionary::EUnknownTokenPolicy& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs, NTextProcessing::NDictionary::EUnknownTokenPolicy>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NDictionary::EUnknownTokenPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EUnknownTokenPolicy>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NDictionary::EUnknownTokenPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EUnknownTokenPolicy>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NDictionary::EUnknownTokenPolicy>(IOutputStream& os, TTypeTraits<NTextProcessing::NDictionary::EUnknownTokenPolicy>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NDictionary::EUnknownTokenPolicy>(NTextProcessing::NDictionary::EUnknownTokenPolicy e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NDictionary::EUnknownTokenPolicy> GetEnumAllValuesImpl<NTextProcessing::NDictionary::EUnknownTokenPolicy>() {
        const NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NDictionary::EUnknownTokenPolicy>() {
        const NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NDictionary::EUnknownTokenPolicy, TString> GetEnumNamesImpl<NTextProcessing::NDictionary::EUnknownTokenPolicy>() {
        const NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NDictionary::EUnknownTokenPolicy>() {
        const NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEUnknownTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NTextProcessing::NDictionary::EEndOfWordTokenPolicy
namespace { namespace NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EEndOfWordTokenPolicy::Skip, "Skip"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EEndOfWordTokenPolicy::Insert, "Insert"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EEndOfWordTokenPolicy::Insert, "Insert"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EEndOfWordTokenPolicy::Skip, "Skip"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Skip"sv,
        "Insert"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NDictionary::EEndOfWordTokenPolicy::"sv,
        "NTextProcessing::NDictionary::EEndOfWordTokenPolicy"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EEndOfWordTokenPolicy> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>;

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

const TString& ToString(NTextProcessing::NDictionary::EEndOfWordTokenPolicy x) {
    const NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NDictionary::EEndOfWordTokenPolicy FromStringImpl<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs, NTextProcessing::NDictionary::EEndOfWordTokenPolicy>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>(const char* data, size_t len, NTextProcessing::NDictionary::EEndOfWordTokenPolicy& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs, NTextProcessing::NDictionary::EEndOfWordTokenPolicy>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NDictionary::EEndOfWordTokenPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NDictionary::EEndOfWordTokenPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>(IOutputStream& os, TTypeTraits<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>(NTextProcessing::NDictionary::EEndOfWordTokenPolicy e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NDictionary::EEndOfWordTokenPolicy> GetEnumAllValuesImpl<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>() {
        const NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>() {
        const NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NDictionary::EEndOfWordTokenPolicy, TString> GetEnumNamesImpl<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>() {
        const NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NDictionary::EEndOfWordTokenPolicy>() {
        const NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfWordTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy
namespace { namespace NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy::Skip, "Skip"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy::Insert, "Insert"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy::Insert, "Insert"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy::Skip, "Skip"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Skip"sv,
        "Insert"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy::"sv,
        "NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>;

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

const TString& ToString(NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy x) {
    const NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy FromStringImpl<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs, NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>(const char* data, size_t len, NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs, NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>(IOutputStream& os, TTypeTraits<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>(NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy> GetEnumAllValuesImpl<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>() {
        const NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>() {
        const NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy, TString> GetEnumNamesImpl<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>() {
        const NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NDictionary::EEndOfSentenceTokenPolicy>() {
        const NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs& names = NNTextProcessingNDictionaryEEndOfSentenceTokenPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NTextProcessing::NDictionary::EContextLevel
namespace { namespace NNTextProcessingNDictionaryEContextLevelPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EContextLevel>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EContextLevel::Word, "Word"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EContextLevel::Sentence, "Sentence"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EContextLevel::Sentence, "Sentence"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EContextLevel::Word, "Word"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Word"sv,
        "Sentence"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NDictionary::EContextLevel::"sv,
        "NTextProcessing::NDictionary::EContextLevel"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EContextLevel> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EContextLevel>;

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

const TString& ToString(NTextProcessing::NDictionary::EContextLevel x) {
    const NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs& names = NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NDictionary::EContextLevel FromStringImpl<NTextProcessing::NDictionary::EContextLevel>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs, NTextProcessing::NDictionary::EContextLevel>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NDictionary::EContextLevel>(const char* data, size_t len, NTextProcessing::NDictionary::EContextLevel& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs, NTextProcessing::NDictionary::EContextLevel>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NDictionary::EContextLevel& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EContextLevel>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NDictionary::EContextLevel& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EContextLevel>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NDictionary::EContextLevel>(IOutputStream& os, TTypeTraits<NTextProcessing::NDictionary::EContextLevel>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NDictionary::EContextLevel>(NTextProcessing::NDictionary::EContextLevel e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NDictionary::EContextLevel> GetEnumAllValuesImpl<NTextProcessing::NDictionary::EContextLevel>() {
        const NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs& names = NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NDictionary::EContextLevel>() {
        const NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs& names = NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NDictionary::EContextLevel, TString> GetEnumNamesImpl<NTextProcessing::NDictionary::EContextLevel>() {
        const NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs& names = NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NDictionary::EContextLevel>() {
        const NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs& names = NNTextProcessingNDictionaryEContextLevelPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NTextProcessing::NDictionary::EDictionaryType
namespace { namespace NNTextProcessingNDictionaryEDictionaryTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EDictionaryType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EDictionaryType::FrequencyBased, "FrequencyBased"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EDictionaryType::Bpe, "Bpe"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EDictionaryType::Bpe, "Bpe"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NDictionary::EDictionaryType::FrequencyBased, "FrequencyBased"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "FrequencyBased"sv,
        "Bpe"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NDictionary::EDictionaryType::"sv,
        "NTextProcessing::NDictionary::EDictionaryType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EDictionaryType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NDictionary::EDictionaryType>;

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

const TString& ToString(NTextProcessing::NDictionary::EDictionaryType x) {
    const NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NDictionary::EDictionaryType FromStringImpl<NTextProcessing::NDictionary::EDictionaryType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs, NTextProcessing::NDictionary::EDictionaryType>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NDictionary::EDictionaryType>(const char* data, size_t len, NTextProcessing::NDictionary::EDictionaryType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs, NTextProcessing::NDictionary::EDictionaryType>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NDictionary::EDictionaryType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EDictionaryType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NDictionary::EDictionaryType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NDictionary::EDictionaryType>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NDictionary::EDictionaryType>(IOutputStream& os, TTypeTraits<NTextProcessing::NDictionary::EDictionaryType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NDictionary::EDictionaryType>(NTextProcessing::NDictionary::EDictionaryType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NDictionary::EDictionaryType> GetEnumAllValuesImpl<NTextProcessing::NDictionary::EDictionaryType>() {
        const NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NDictionary::EDictionaryType>() {
        const NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NDictionary::EDictionaryType, TString> GetEnumNamesImpl<NTextProcessing::NDictionary::EDictionaryType>() {
        const NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NDictionary::EDictionaryType>() {
        const NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs& names = NNTextProcessingNDictionaryEDictionaryTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

