#pragma once

#include "embedding.h"
#include <util/generic/fwd.h>

namespace NCB {

    TEmbeddingPtr LoadEmbedding(const TString& path, const TDictionaryProxy& dictionary);

}
