// This file was auto-generated. Do not edit!!!
#include <library/cpp/text_processing/tokenizer/lemmer_impl.h>
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

// I/O for NTextProcessing::NTokenizer::EImplementationType
namespace { namespace NNTextProcessingNTokenizerEImplementationTypePrivate {
    using TNameBufsBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::EImplementationType>;

    static constexpr const std::array<TNameBufsBase::TEnumStringPair, 2> NAMES_INITIALIZATION_PAIRS_PAYLOAD = ::NEnumSerializationRuntime::TryStableSortKeys(std::array<TNameBufsBase::TEnumStringPair, 2>{{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::EImplementationType::Trivial, "Trivial"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::EImplementationType::YandexSpecific, "YandexSpecific"sv),
    }});
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> NAMES_INITIALIZATION_PAIRS{NAMES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TNameBufsBase::TEnumStringPair VALUES_INITIALIZATION_PAIRS_PAYLOAD[2]{
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::EImplementationType::Trivial, "Trivial"sv),
        TNameBufsBase::EnumStringPair(NTextProcessing::NTokenizer::EImplementationType::YandexSpecific, "YandexSpecific"sv),
    };
    static constexpr const TArrayRef<const TNameBufsBase::TEnumStringPair> VALUES_INITIALIZATION_PAIRS{VALUES_INITIALIZATION_PAIRS_PAYLOAD};

    static constexpr const TStringBuf CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD[2]{
        "Trivial"sv,
        "YandexSpecific"sv,
    };
    static constexpr const TArrayRef<const TStringBuf> CPP_NAMES_INITIALIZATION_ARRAY{CPP_NAMES_INITIALIZATION_ARRAY_PAYLOAD};

    static constexpr const TNameBufsBase::TInitializationData ENUM_INITIALIZATION_DATA{
        NAMES_INITIALIZATION_PAIRS,
        VALUES_INITIALIZATION_PAIRS,
        CPP_NAMES_INITIALIZATION_ARRAY,
        "NTextProcessing::NTokenizer::EImplementationType::"sv,
        "NTextProcessing::NTokenizer::EImplementationType"sv
    };

    static constexpr ::NEnumSerializationRuntime::ESortOrder NAMES_ORDER = ::NEnumSerializationRuntime::GetKeyFieldSortOrder(NAMES_INITIALIZATION_PAIRS);
    static constexpr ::NEnumSerializationRuntime::ESortOrder VALUES_ORDER = ::NEnumSerializationRuntime::GetNameFieldSortOrder(VALUES_INITIALIZATION_PAIRS);

    class TNameBufs : public ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::EImplementationType> {
    public:
        using TBase = ::NEnumSerializationRuntime::TEnumDescription<NTextProcessing::NTokenizer::EImplementationType>;

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

const TString& ToString(NTextProcessing::NTokenizer::EImplementationType x) {
    const NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs::Instance();
    return names.ToString(x);
}

template<>
NTextProcessing::NTokenizer::EImplementationType FromStringImpl<NTextProcessing::NTokenizer::EImplementationType>(const char* data, size_t len) {
    return ::NEnumSerializationRuntime::DispatchFromStringImplFn<NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs, NTextProcessing::NTokenizer::EImplementationType>(data, len);
}

template<>
bool TryFromStringImpl<NTextProcessing::NTokenizer::EImplementationType>(const char* data, size_t len, NTextProcessing::NTokenizer::EImplementationType& result) {
    return ::NEnumSerializationRuntime::DispatchTryFromStringImplFn<NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs, NTextProcessing::NTokenizer::EImplementationType>(data, len, result);
}

bool FromString(const TString& name, NTextProcessing::NTokenizer::EImplementationType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::EImplementationType>(name.data(), name.size(), ret);
}

bool FromString(const TStringBuf& name, NTextProcessing::NTokenizer::EImplementationType& ret) {
    return ::TryFromStringImpl<NTextProcessing::NTokenizer::EImplementationType>(name.data(), name.size(), ret);
}

template<>
void Out<NTextProcessing::NTokenizer::EImplementationType>(IOutputStream& os, TTypeTraits<NTextProcessing::NTokenizer::EImplementationType>::TFuncParam n) {
    return ::NEnumSerializationRuntime::DispatchOutFn<NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs>(os, n);
}

namespace NEnumSerializationRuntime {
    template<>
    TStringBuf ToStringBuf<NTextProcessing::NTokenizer::EImplementationType>(NTextProcessing::NTokenizer::EImplementationType e) {
        return ::NEnumSerializationRuntime::DispatchToStringBufFn<NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs>(e);
    }

    template<>
    TMappedArrayView<NTextProcessing::NTokenizer::EImplementationType> GetEnumAllValuesImpl<NTextProcessing::NTokenizer::EImplementationType>() {
        const NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs::Instance();
        return names.AllEnumValues();
    }

    template<>
    const TString& GetEnumAllNamesImpl<NTextProcessing::NTokenizer::EImplementationType>() {
        const NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs::Instance();
        return names.AllEnumNames();
    }

    template<>
    TMappedDictView<NTextProcessing::NTokenizer::EImplementationType, TString> GetEnumNamesImpl<NTextProcessing::NTokenizer::EImplementationType>() {
        const NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs::Instance();
        return names.EnumNames();
    }

    template<>
    const TVector<TString>& GetEnumAllCppNamesImpl<NTextProcessing::NTokenizer::EImplementationType>() {
        const NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs& names = NNTextProcessingNTokenizerEImplementationTypePrivate::TNameBufs::Instance();
        return names.AllEnumCppNames();
    }
}

