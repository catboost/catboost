// This file was auto-generated. Do not edit!!!
#include <library/cpp/text_processing/tokenizer/options.h>
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

// I/O for NTextProcessing::NTokenizer::ESeparatorType
namespace { namespace NNTextProcessingNTokenizerESeparatorTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ESeparatorType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ESeparatorType::ByDelimiter, "ByDelimiter"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ESeparatorType::BySense, "BySense"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ESeparatorType::ByDelimiter, "ByDelimiter"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ESeparatorType::BySense, "BySense"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "ByDelimiter"sv,
        "BySense"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NTokenizer::ESeparatorType::"sv,
        "NTextProcessing::NTokenizer::ESeparatorType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ESeparatorType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ESeparatorType>;

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

const TString& ToString(NTextProcessing::NTokenizer::ESeparatorType x) {
    const NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NTokenizer::ESeparatorType FromStringImpl<NTextProcessing::NTokenizer::ESeparatorType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs, NTextProcessing::NTokenizer::ESeparatorType>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NTokenizer::ESeparatorType>(const char* data, size_t len, NTextProcessing::NTokenizer::ESeparatorType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs, NTextProcessing::NTokenizer::ESeparatorType>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NTokenizer::ESeparatorType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::ESeparatorType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NTokenizer::ESeparatorType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::ESeparatorType>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NTokenizer::ESeparatorType>(IOutputStream& os, TTypeTraits<NTextProcessing::NTokenizer::ESeparatorType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NTokenizer::ESeparatorType>(NTextProcessing::NTokenizer::ESeparatorType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NTokenizer::ESeparatorType> GetEnumAllValuesImpl<NTextProcessing::NTokenizer::ESeparatorType>() {
        const NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NTokenizer::ESeparatorType>() {
        const NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NTokenizer::ESeparatorType, TString> GetEnumNamesImpl<NTextProcessing::NTokenizer::ESeparatorType>() {
        const NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NTokenizer::ESeparatorType>() {
        const NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerESeparatorTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NTextProcessing::NTokenizer::ETokenType
namespace { namespace NNTextProcessingNTokenizerETokenTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ETokenType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 6> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 6>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::Word, "Word"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::Number, "Number"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::Punctuation, "Punctuation"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::SentenceBreak, "SentenceBreak"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::ParagraphBreak, "ParagraphBreak"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::Unknown, "Unknown"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[6]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::Number, "Number"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::ParagraphBreak, "ParagraphBreak"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::Punctuation, "Punctuation"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::SentenceBreak, "SentenceBreak"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::Unknown, "Unknown"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenType::Word, "Word"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[6]{
        "Word"sv,
        "Number"sv,
        "Punctuation"sv,
        "SentenceBreak"sv,
        "ParagraphBreak"sv,
        "Unknown"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NTokenizer::ETokenType::"sv,
        "NTextProcessing::NTokenizer::ETokenType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ETokenType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ETokenType>;

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

const TString& ToString(NTextProcessing::NTokenizer::ETokenType x) {
    const NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NTokenizer::ETokenType FromStringImpl<NTextProcessing::NTokenizer::ETokenType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs, NTextProcessing::NTokenizer::ETokenType>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NTokenizer::ETokenType>(const char* data, size_t len, NTextProcessing::NTokenizer::ETokenType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs, NTextProcessing::NTokenizer::ETokenType>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NTokenizer::ETokenType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::ETokenType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NTokenizer::ETokenType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::ETokenType>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NTokenizer::ETokenType>(IOutputStream& os, TTypeTraits<NTextProcessing::NTokenizer::ETokenType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NTokenizer::ETokenType>(NTextProcessing::NTokenizer::ETokenType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NTokenizer::ETokenType> GetEnumAllValuesImpl<NTextProcessing::NTokenizer::ETokenType>() {
        const NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NTokenizer::ETokenType>() {
        const NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NTokenizer::ETokenType, TString> GetEnumNamesImpl<NTextProcessing::NTokenizer::ETokenType>() {
        const NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NTokenizer::ETokenType>() {
        const NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NTextProcessing::NTokenizer::ESubTokensPolicy
namespace { namespace NNTextProcessingNTokenizerESubTokensPolicyPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ESubTokensPolicy>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ESubTokensPolicy::SingleToken, "SingleToken"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ESubTokensPolicy::SeveralTokens, "SeveralTokens"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ESubTokensPolicy::SeveralTokens, "SeveralTokens"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ESubTokensPolicy::SingleToken, "SingleToken"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "SingleToken"sv,
        "SeveralTokens"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NTokenizer::ESubTokensPolicy::"sv,
        "NTextProcessing::NTokenizer::ESubTokensPolicy"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ESubTokensPolicy> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ESubTokensPolicy>;

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

const TString& ToString(NTextProcessing::NTokenizer::ESubTokensPolicy x) {
    const NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NTokenizer::ESubTokensPolicy FromStringImpl<NTextProcessing::NTokenizer::ESubTokensPolicy>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs, NTextProcessing::NTokenizer::ESubTokensPolicy>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NTokenizer::ESubTokensPolicy>(const char* data, size_t len, NTextProcessing::NTokenizer::ESubTokensPolicy& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs, NTextProcessing::NTokenizer::ESubTokensPolicy>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NTokenizer::ESubTokensPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::ESubTokensPolicy>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NTokenizer::ESubTokensPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::ESubTokensPolicy>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NTokenizer::ESubTokensPolicy>(IOutputStream& os, TTypeTraits<NTextProcessing::NTokenizer::ESubTokensPolicy>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NTokenizer::ESubTokensPolicy>(NTextProcessing::NTokenizer::ESubTokensPolicy e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NTokenizer::ESubTokensPolicy> GetEnumAllValuesImpl<NTextProcessing::NTokenizer::ESubTokensPolicy>() {
        const NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NTokenizer::ESubTokensPolicy>() {
        const NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NTokenizer::ESubTokensPolicy, TString> GetEnumNamesImpl<NTextProcessing::NTokenizer::ESubTokensPolicy>() {
        const NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NTokenizer::ESubTokensPolicy>() {
        const NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerESubTokensPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

// I/O for NTextProcessing::NTokenizer::ETokenProcessPolicy
namespace { namespace NNTextProcessingNTokenizerETokenProcessPolicyPrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ETokenProcessPolicy>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 3> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 3>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenProcessPolicy::Skip, "Skip"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenProcessPolicy::LeaveAsIs, "LeaveAsIs"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenProcessPolicy::Replace, "Replace"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[3]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenProcessPolicy::LeaveAsIs, "LeaveAsIs"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenProcessPolicy::Replace, "Replace"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::ETokenProcessPolicy::Skip, "Skip"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[3]{
        "Skip"sv,
        "LeaveAsIs"sv,
        "Replace"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NTokenizer::ETokenProcessPolicy::"sv,
        "NTextProcessing::NTokenizer::ETokenProcessPolicy"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ETokenProcessPolicy> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::ETokenProcessPolicy>;

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

const TString& ToString(NTextProcessing::NTokenizer::ETokenProcessPolicy x) {
    const NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NTokenizer::ETokenProcessPolicy FromStringImpl<NTextProcessing::NTokenizer::ETokenProcessPolicy>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs, NTextProcessing::NTokenizer::ETokenProcessPolicy>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NTokenizer::ETokenProcessPolicy>(const char* data, size_t len, NTextProcessing::NTokenizer::ETokenProcessPolicy& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs, NTextProcessing::NTokenizer::ETokenProcessPolicy>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NTokenizer::ETokenProcessPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::ETokenProcessPolicy>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NTokenizer::ETokenProcessPolicy& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::ETokenProcessPolicy>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NTokenizer::ETokenProcessPolicy>(IOutputStream& os, TTypeTraits<NTextProcessing::NTokenizer::ETokenProcessPolicy>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NTokenizer::ETokenProcessPolicy>(NTextProcessing::NTokenizer::ETokenProcessPolicy e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NTokenizer::ETokenProcessPolicy> GetEnumAllValuesImpl<NTextProcessing::NTokenizer::ETokenProcessPolicy>() {
        const NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NTokenizer::ETokenProcessPolicy>() {
        const NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NTokenizer::ETokenProcessPolicy, TString> GetEnumNamesImpl<NTextProcessing::NTokenizer::ETokenProcessPolicy>() {
        const NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NTokenizer::ETokenProcessPolicy>() {
        const NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs& names = NNTextProcessingNTokenizerETokenProcessPolicyPrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

