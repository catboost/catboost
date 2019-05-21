#pragma once
#include <library/text_processing/dictionary/dictionary.h>

namespace NCB {

    using IDictionary = NTextProcessing::NDictionary::IDictionary;
    using TDictionaryPtr = TIntrusivePtr<IDictionary>;
}
