#pragma once

#include "embedding.h"

namespace NCB {

    TEmbeddingPtr LoadEmbedding(const TString& path, const IDictionary& dictionary);

}
