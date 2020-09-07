#pragma once

#include <library/cpp/hnsw/index/index_base.h>
#include <library/cpp/hnsw/index/index_item_storage_base.h>
#include <library/cpp/hnsw/index/dense_vector_distance.h>
#include <library/cpp/hnsw/index/dense_vector_index.h>
#include <library/cpp/online_hnsw/base/item_storage_index.h>
#include <library/cpp/online_hnsw/base/index_reader.h>
#include <library/cpp/online_hnsw/base/index_writer.h>
#include <library/cpp/online_hnsw/dense_vectors/index.h>

#include "embedding_feature_calcer.h"

using TOnlineHnswCloud = NOnlineHnsw::TOnlineHnswDenseVectorIndex<float, NHnsw::TL2SqrDistance<float>>;

namespace NCB {

    class TKNNCalcer final : public TEmbeddingFeatureCalcer {
    public:
        explicit TKNNCalcer(
            int totalDimension,
            int numClasses,
            ui32 closeNum,
            const TGuid& calcerId = CreateGuid()
        )
            : TEmbeddingFeatureCalcer(numClasses, calcerId)
            , TotalDimension(totalDimension)
            , NumClasses(numClasses)
            , CloseNum(closeNum)
            , Opts({CloseNum, 300})
            , Cloud(Opts, TotalDimension)
        {}

        void Compute(const TEmbeddingsArray& embed, TOutputFloatIterator outputFeaturesIterator) const override;

        ui32 FeatureCount() const override {
            return NumClasses;
        }

        EFeatureCalcerType Type() const override {
            return EFeatureCalcerType::KNN;
        }

    private:
        int TotalDimension;
        int NumClasses;
        ui32 CloseNum;
        NOnlineHnsw::TOnlineHnswBuildOptions Opts;
        TOnlineHnswCloud Cloud;
        TVector<ui32> Targets;

    protected:
        friend class TKNNCalcerVisitor;
    };

    class TKNNCalcerVisitor final : public IEmbeddingCalcerVisitor{
    public:
        void Update(ui32 classId, const TEmbeddingsArray& embed, TEmbeddingFeatureCalcer* featureCalcer) override;
    };

};
