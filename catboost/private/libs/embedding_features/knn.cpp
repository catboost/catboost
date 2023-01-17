#include "knn.h"

#include <catboost/private/libs/embedding_features/flatbuffers/embedding_feature_calcers.fbs.h>

#include <util/stream/length.h>
#include <util/system/yassert.h>

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
                                                   TL2Distance>(embed,
                                                   knum, 300, Points, Dist);
        for (size_t pos = 0; pos < neighbors.size(); ++pos) {
            result.push_back(neighbors[pos].Id);
        }
        return result;
    }

    void TKNNCalcer::Compute(const TEmbeddingsArray& embed,
                             TOutputFloatIterator iterator) const {
        TVector<float> result(FeatureCount_, 0);
        auto neighbors = Cloud->GetNearestNeighbors(embed.data(), CloseNum);
        if (IsClassification) {
           for (size_t pos = 0; pos < neighbors.size(); ++pos) {
                ++result[TargetClasses.at(neighbors[pos])];
            }
        } else {
            if (neighbors.size()) {
                for (size_t pos = 0; pos < neighbors.size(); ++pos) {
                    result[0] += Targets.at(neighbors[pos]);
                }
                result[0] /= (float)neighbors.size();
            }
        }
        ForEachActiveFeature(
            [&result, &iterator](ui32 featureId){
                *iterator = result[featureId];
                ++iterator;
            }
        );
    }

    void TKNNCalcerVisitor::Update(float target,
                const TEmbeddingsArray& embed,
                TEmbeddingFeatureCalcer* featureCalcer) {
        auto knn = dynamic_cast<TKNNCalcer*>(featureCalcer);
        Y_ASSERT(knn);
        auto cloudPtr = dynamic_cast<TKNNUpdatableCloud*>(knn->Cloud.Get());
        Y_ASSERT(cloudPtr);
        cloudPtr->AddItem(embed.data());
        if (knn->IsClassification) {
            knn->TargetClasses.push_back((ui32)target);
        } else {
            knn->Targets.push_back(target);
        }
        ++knn->Size;
    }

    TEmbeddingFeatureCalcer::TEmbeddingCalcerFbs TKNNCalcer::SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const {
        using namespace NCatBoostFbs::NEmbeddings;

        const auto& fbLDA = CreateTKNN(
            builder,
            TotalDimension,
            FeatureCount_,
            CloseNum,
            Size,
            IsClassification
        );
        return TEmbeddingCalcerFbs(TAnyEmbeddingCalcer_TKNN, fbLDA.Union());
    }

    void TKNNCalcer::LoadParametersFromFB(const NCatBoostFbs::NEmbeddings::TEmbeddingCalcer* calcer) {
        auto fbKNN = calcer->FeatureCalcerImpl_as_TKNN();
        TotalDimension = fbKNN->TotalDimension();
        IsClassification = fbKNN->IsClassification();
        FeatureCount_ = IsClassification ? fbKNN->NumClasses() : 1; // will always be 1 for regression
        CloseNum = fbKNN->KNum();
        Size = fbKNN->Size();
        
    }

    void TKNNCalcer::SaveLargeParameters(IOutputStream* stream) const {
        if (IsClassification) {
            ::Save(stream, TargetClasses);
        } else {
            ::Save(stream, Targets);
        }
        if (auto updatableCloud = dynamic_cast<TKNNUpdatableCloud*>(Cloud.Get())) {
            NOnlineHnsw::TOnlineHnswIndexData indexData = updatableCloud->GetCloud().ConstructIndexData();
            auto expectedIndexSize = NOnlineHnsw::ExpectedSize(indexData);
            ::SaveSize(stream, expectedIndexSize);
            TCountingOutput countingOutput(stream);
            NOnlineHnsw::WriteIndex(indexData, countingOutput);
            CB_ENSURE(
                countingOutput.Counter() == expectedIndexSize,
                LabeledOutput(countingOutput.Counter(), expectedIndexSize) << " should be equal."
            );
            ::Save(stream, updatableCloud->GetVector());
        } else {
            const auto staticCloud = dynamic_cast<const TKNNCloud*>(Cloud.Get());
            CB_ENSURE(staticCloud, "Expected NCB::TKNNCloud pointer");
            const auto& indexDataBlob = staticCloud->GetIndexDataBlob();
            ::SaveSize(stream, indexDataBlob.Size());
            stream->Write(indexDataBlob.Data(), indexDataBlob.Size());
            ::Save(stream, staticCloud->GetPointsVector());
        }
    }

    void TKNNCalcer::LoadLargeParameters(IInputStream* stream) {
        if (IsClassification) {
            ::Load(stream, TargetClasses);
        } else {
            ::Load(stream, Targets);
        }
        size_t indexSize = ::LoadSize(stream);
        TLengthLimitedInput indexArrayStream(stream, indexSize);
        auto indexArray = TBlob::FromStream(indexArrayStream);
        CB_ENSURE(indexArray.Size() == indexSize);
        TVector<float> points(TotalDimension * Size);
        ::Load(stream, points);
        Cloud = MakeHolder<TKNNCloud>(
            std::move(indexArray),
            std::move(points),
            Size,
            TotalDimension
        );
    }

    TEmbeddingFeatureCalcerFactory::TRegistrator<TKNNCalcer> KNNRegistrator(EFeatureCalcerType::KNN);

};
