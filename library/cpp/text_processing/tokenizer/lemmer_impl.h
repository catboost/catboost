#pragma once

#include <library/cpp/langs/langs.h>
#include <library/cpp/object_factory/object_factory.h>

namespace NTextProcessing::NTokenizer {
    enum class EImplementationType {
        Trivial,
        YandexSpecific
    };

    class ILemmerImplementation {
    public:
        virtual void Lemmatize(TUtf16String* token) const = 0;
        virtual ~ILemmerImplementation() = default;
    };

     using TLemmerImplementationFactory = NObjectFactory::TParametrizedObjectFactory<ILemmerImplementation, EImplementationType, const TVector<ELanguage>&>;
}

