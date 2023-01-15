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
    private:
        TOnlineHnswCloud Cloud;
    };

    class TKNNCloud : public IKNNCloud {
    public:
        TKNNCloud(TArrayHolder<ui8>&& indexData, TArrayHolder<ui8>&& vectorData,
                  size_t idxSize, size_t stSize, size_t dim)
            : IndexData(std::move(indexData))
            , VectorData(std::move(vectorData))
            , Cloud(TBlob::NoCopy(IndexData.Get(), idxSize),
                    TBlob::NoCopy(VectorData.Get(), stSize), dim)
        { }
        TVector<ui32> GetNearestNeighbors(const float* embed, ui32 knum) const override;
    private:
        TArrayHolder<ui8> IndexData;
        TArrayHolder<ui8> VectorData;
        THnswCloud Cloud;
    };

    class TKNNCalcer final : public TEmbeddingFeatureCalcer {
    public:
        explicit TKNNCalcer(
            int totalDimension = 2,
            int numClasses = 2,
            ui32 closeNum = 5,
            const TGuid& calcerId = CreateGuid()
        )
            : TEmbeddingFeatureCalcer(numClasses, calcerId)
            , TotalDimension(totalDimension)
            , NumClasses(numClasses)
            , CloseNum(closeNum)
            , Size(0)
            , Cloud(MakeHolder<TKNNUpdatableCloud>(NOnlineHnsw::TOnlineHnswBuildOptions({CloseNum, 300}),
                                                   totalDimension))
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
        void LoadParametersFromFB(const NCatBoostFbs::TEmbeddingCalcer* calcerFbs) override;

        void SaveLargeParameters(IOutputStream*) const override;
        void LoadLargeParameters(IInputStream*) override;

    private:
        int TotalDimension;
        int NumClasses;
        ui32 CloseNum;
        ui32 Size;
        TKNNCloudPtr Cloud;
        TVector<ui32> Targets;

    protected:
        friend class TKNNCalcerVisitor;
    };

    class TKNNCalcerVisitor final : public IEmbeddingCalcerVisitor{
    public:
        void Update(ui32 classId, const TEmbeddingsArray& embed, TEmbeddingFeatureCalcer* featureCalcer) override;
    };

};
