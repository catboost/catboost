#include "knn.h"

#include <catboost/private/libs/embedding_features/flatbuffers/embedding_feature_calcers.fbs.h>

#include <util/stream/length.h>

namespace NCB {

    TVector<ui32> TKNNUpdatableCloud::GetNearestNeighbors(const float* embed, ui32 knum) const  {
        TVector<ui32> result;
        auto neighbors = Cloud.GetNearestNeighbors(embed, knum);
        for (size_t pos = 0; pos < neighbors.size(); ++pos) {
            result.push_back(neighbors[pos].Id);
        }
        return result;
    }

    TVector<ui32> TKNNCloud::GetNearestNeighbors(const float* embed, ui32 knum) const  {
        TVector<ui32> result;
        auto neighbors = Cloud.GetNearestNeighbors<NOnlineHnsw::TDenseVectorExtendableItemStorage<float>,
                                                   TL2Distance<NHnsw::TL2SqrDistance<float>>>(embed,
                                                   knum, 2 * knum, Points, Dist);
        for (size_t pos = 0; pos < neighbors.size(); ++pos) {
            result.push_back(neighbors[pos].Id);
        }
        return result;
    }

    void TKNNCalcer::Compute(const TEmbeddingsArray& embed,
                             TOutputFloatIterator iterator) const {
        TVector<float> result(NumClasses, 0);
        auto neighbors = Cloud->GetNearestNeighbors(embed.data(), CloseNum);
        for (size_t pos = 0; pos < neighbors.size(); ++pos) {
            ++result[Targets.at(neighbors[pos])];
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
        auto cloudPtr = dynamic_cast<TKNNUpdatableCloud*>(knn->Cloud.Get());
        Y_ASSERT(cloudPtr);
        cloudPtr->AddItem(embed.data());
        knn->Targets.push_back(classId);
        ++knn->Size;
    }

    TEmbeddingFeatureCalcer::TEmbeddingCalcerFbs TKNNCalcer::SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const {
        using namespace NCatBoostFbs;

        const auto& fbLDA = CreateTKNN(
            builder,
            TotalDimension,
            NumClasses,
            CloseNum,
            Size
        );
        return TEmbeddingCalcerFbs(TAnyEmbeddingCalcer_TKNN, fbLDA.Union());
    }

    void TKNNCalcer::LoadParametersFromFB(const NCatBoostFbs::TEmbeddingCalcer* calcer) {
        auto fbKNN = calcer->FeatureCalcerImpl_as_TKNN();
        TotalDimension = fbKNN->TotalDimension();
        NumClasses = fbKNN->NumClasses();
        CloseNum = fbKNN->KNum();
        Size = fbKNN->Size();
    }

    void TKNNCalcer::SaveLargeParameters(IOutputStream* stream) const {
        ::Save(stream, Targets);
        auto cl = dynamic_cast<TKNNUpdatableCloud*>(Cloud.Get());
        NOnlineHnsw::TOnlineHnswIndexData indexData = cl->GetCloud().ConstructIndexData();
        ::SaveSize(stream, NOnlineHnsw::ExpectedSize(indexData));
        TCountingOutput ss(stream);
        NOnlineHnsw::WriteIndex(indexData, ss);
        ::Save(stream, cl->GetVector());
    }

    void TKNNCalcer::LoadLargeParameters(IInputStream* stream) {
        ::Load(stream, Targets);
        size_t indexSize = ::LoadSize(stream);
        TArrayHolder<ui8> indexArray = TArrayHolder<ui8>(new ui8[indexSize]);
        stream->Load(indexArray.Get(), indexSize);
        TVector<float> points(TotalDimension * Size);
        ::Load(stream, points);
        auto cloudPtr = MakeHolder<TKNNCloud>(std::move(indexArray), indexSize,
                                              std::move(points), TotalDimension, Size);
    }

    TEmbeddingFeatureCalcerFactory::TRegistrator<TKNNCalcer> KNNRegistrator(EFeatureCalcerType::KNN);

};
