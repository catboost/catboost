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
            bool isClassification = true,
            int featureCount = 2,
            ui32 closeNum = 5,
            const TGuid& calcerId = CreateGuid()
        )
            : TEmbeddingFeatureCalcer(featureCount, calcerId)
            , TotalDimension(totalDimension)
            , IsClassification(isClassification)
            , FeatureCount_(featureCount)
            , CloseNum(closeNum)
            , Size(0)
            , Cloud(MakeHolder<TKNNUpdatableCloud>(NOnlineHnsw::TOnlineHnswBuildOptions({CloseNum, 300}),
                                                   totalDimension))
        {}

        void Compute(const TEmbeddingsArray& embed, TOutputFloatIterator outputFeaturesIterator) const override;

        ui32 FeatureCount() const override {
            return FeatureCount_;
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
        bool IsClassification;
        int FeatureCount_;
        ui32 CloseNum;
        ui32 Size;
        TKNNCloudPtr Cloud;
        TVector<ui32> TargetClasses; // used for classification
        TVector<float> Targets;       // used for regression

    protected:
        friend class TKNNCalcerVisitor;
    };

    class TKNNCalcerVisitor final : public IEmbeddingCalcerVisitor{
    public:
        void Update(float target, const TEmbeddingsArray& embed, TEmbeddingFeatureCalcer* featureCalcer) override;
    };

};
