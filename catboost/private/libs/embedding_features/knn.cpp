#include "knn.h"

namespace NCB {
    void TKNNCalcer::Compute(const TEmbeddingsArray& embed,
                             TOutputFloatIterator iterator) const {
        TVector<float> result(NumClasses, 0);
        if (Cloud.GetNumItems() >= CloseNum) {
            auto neighbors = Cloud.GetNearestNeighbors(embed.data(), CloseNum);
            for (size_t pos = 1; pos < neighbors.size(); ++pos) {
                ++result[Targets.at(neighbors[pos].Id)];
            }
        }
        ForEachActiveFeature(
            [&result, &iterator](ui32 featureId){
                *iterator = result[featureId];
                ++iterator;
            }
        );
    }

    void TKNNCalcerVisitor::Update(ui32 classId,
                const TEmbeddingsArray& embed,
                TEmbeddingFeatureCalcer* featureCalcer) {
        auto knn = dynamic_cast<TKNNCalcer*>(featureCalcer);
        Y_ASSERT(knn);
        knn->Cloud.GetNearestNeighborsAndAddItem(embed.data());
        knn->Targets.push_back(classId);
    }
};
