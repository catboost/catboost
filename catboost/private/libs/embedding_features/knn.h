#pragma once

#include "embedding_feature_calcer.h"

#include <library/cpp/hnsw/index/index_base.h>
#include <library/cpp/hnsw/index/index_item_storage_base.h>
#include <library/cpp/hnsw/index/dense_vector_distance.h>
#include <library/cpp/hnsw/index/dense_vector_index.h>
#include <library/cpp/online_hnsw/base/item_storage_index.h>
#include <library/cpp/online_hnsw/base/index_reader.h>
#include <library/cpp/online_hnsw/base/index_writer.h>
#include <library/cpp/online_hnsw/dense_vectors/index.h>

#include <util/stream/buffer.h>
#include <util/memory/blob.h>
#include <util/random/fast.h>

using TOnlineHnswCloud = NOnlineHnsw::TOnlineHnswDenseVectorIndex<float, NHnsw::TL2SqrDistance<float>>;
using THnswCloud = NHnsw::THnswDenseVectorIndex<float>;

namespace NCB {

    class IKNNCloud : public TThrRefBase {
    public:
        virtual TVector<ui32> GetNearestNeighbors(const float*, ui32) const = 0;
    };

    using TKNNCloudPtr = THolder<IKNNCloud>;

    class TKNNUpdatableCloud : public IKNNCloud {
    public:
        explicit TKNNUpdatableCloud(NOnlineHnsw::TOnlineHnswBuildOptions Opts, int Dimension)
            : Cloud(Opts, Dimension) {
        }
        TVector<ui32> GetNearestNeighbors(const float* embed, ui32 knum) const override;
        void AddItem(const float* embed) {
            Cloud.GetNearestNeighborsAndAddItem(embed);
        }
        const TVector<float>& GetVector() const {
            return Cloud.GetVector();
        }
        const TOnlineHnswCloud& GetCloud() const {
            return Cloud;
        }
    private:
        TOnlineHnswCloud Cloud;
    };

    struct TL2Distance {
    public:
        TL2Distance(size_t dim)
            : Dim(dim)
        {}
        using TResult = float;
        using TLess = ::TLess<TResult>;
        TResult operator()(const float* a, const float* b) const {
            return Dist(a, b, Dim);
        }
    private:
        NHnsw::TL2SqrDistance<float> Dist;
        size_t Dim;
    };

    class TKNNCloud : public IKNNCloud {
    public:
        TKNNCloud(
            TBlob&& indexData,
            TVector<float>&& vectorData,
            size_t size,
            size_t dim
        )
            : IndexData(std::move(indexData))
            , Dist(dim)
            , Cloud(IndexData, NOnlineHnsw::TOnlineHnswIndexReader())
            , Points(std::move(vectorData), dim, size)
        {
            CB_ENSURE(vectorData.size() == dim * size);
        }
        TVector<ui32> GetNearestNeighbors(const float* embed, ui32 knum) const override;

        const TBlob& GetIndexDataBlob() const {
            return IndexData;
        }

        const TVector<float>& GetPointsVector() const {
            return Points.GetVector();
        }
    private:
        TBlob IndexData;
        TL2Distance Dist;
        NHnsw::THnswIndexBase Cloud;
        NOnlineHnsw::TDenseVectorExtendableItemStorage<float> Points;
    };

    class TKNNCalcer final : public TEmbeddingFeatureCalcer {
    public:
        explicit TKNNCalcer(
            int totalDimension = 2,
            int numClasses = 2,
            ui32 closeNum = 5,
            float samplingProbability = 1.0,
            ui64 randSeed = 0,
            const TGuid& calcerId = CreateGuid()
        )
            : TEmbeddingFeatureCalcer(numClasses, calcerId)
            , TotalDimension(totalDimension)
            , NumClasses(numClasses)
            , CloseNum(closeNum)
            , SamplingProbability(samplingProbability)
            , Size(0)
            , Cloud(MakeHolder<TKNNUpdatableCloud>(NOnlineHnsw::TOnlineHnswBuildOptions({CloseNum, 300}),
                                                   totalDimension))
            , Rand(randSeed)
        {}

        void Compute(const TEmbeddingsArray& embed, TOutputFloatIterator outputFeaturesIterator) const override;

        ui32 FeatureCount() const override {
            return NumClasses;
        }

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::KNN;
        }

    protected:
        TEmbeddingFeatureCalcer::TEmbeddingCalcerFbs SaveParametersToFB(flatbuffers::FlatBufferBuilder& builder) const override;
        void LoadParametersFromFB(const NCatBoostFbs::NEmbeddings::TEmbeddingCalcer* calcerFbs) override;

        void SaveLargeParameters(IOutputStream*) const override;
        void LoadLargeParameters(IInputStream*) override;

    private:
        int TotalDimension;
        int NumClasses;
        ui32 CloseNum;
        float SamplingProbability;
        ui32 Size;
        TKNNCloudPtr Cloud;
        TVector<ui32> Targets;
        TFastRng64 Rand;

    protected:
        friend class TKNNCalcerVisitor;
    };

    class TKNNCalcerVisitor final : public IEmbeddingCalcerVisitor{
    public:
        void Update(ui32 classId, const TEmbeddingsArray& embed, TEmbeddingFeatureCalcer* featureCalcer) override;
    };

};
