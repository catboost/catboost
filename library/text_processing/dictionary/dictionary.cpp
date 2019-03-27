#include "dictionary.h"

#include "frequency_based_dictionary.h"

using NTextProcessing::NDictionary::IDictionary;
using NTextProcessing::NDictionary::TDictionary;

THolder<IDictionary> IDictionary::Load(IInputStream *stream) {
    auto dictionary = MakeHolder<TDictionary>();
    dictionary->Load(stream);
    return dictionary;
}
