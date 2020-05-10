#include "lemmer_impl.h"

namespace {
    class TTrivialLemmerImplementation : public NTextProcessing::NTokenizer::ILemmerImplementation {
    public:
        TTrivialLemmerImplementation(const TVector<ELanguage>&) {}
        void Lemmatize(TUtf16String*) const override {}
    };
}

NTextProcessing::NTokenizer::TLemmerImplementationFactory::TRegistrator<TTrivialLemmerImplementation> TrivialLemmerImplementationRegistrator(NTextProcessing::NTokenizer::EImplementationType::Trivial);
