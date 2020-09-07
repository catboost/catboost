#include <util/generic/vector.h>

namespace NHnsw {
    TVector<size_t> GetLevelSizes(size_t numVectors, size_t levelSizeDecay) {
        TVector<size_t> levelSizes;
        if (numVectors == 1) {
            levelSizes.push_back(numVectors);
        } else {
            for (; numVectors > 1; numVectors /= levelSizeDecay) {
                levelSizes.push_back(numVectors);
            }
        }
        return levelSizes;
    }
}
