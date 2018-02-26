#pragma once

#include "data_provider.h"
#include "data_utils.h"
#include "grid_creator.h"
#include "proto_helpers.h"
#include "binarizations_manager.h"
#include <catboost/cuda/data/columns.h>
#include <catboost/cuda/data/pool_proto/pool.pb.h>
#include <library/protobuf/protofile/protofile.h>
#include <library/grid_creator/binarization.h>
#include <util/stream/file.h>
#include <util/generic/vector.h>
#include <util/generic/map.h>

namespace NCatboostCuda {
    class TCatBoostProtoPoolReader {
    public:
        template <class TContainer>
        TCatBoostProtoPoolReader& AddIgnoredFeatures(const TContainer& container) {
            for (const auto& f : container) {
                IgnoreFeatures.insert(f);
            }
            return *this;
        }

        TCatBoostProtoPoolReader& SetClassesWeights(const TVector<float>& classesWeights) {
            ClassesWeights = classesWeights;
            return *this;
        }

        TCatBoostProtoPoolReader& SetShuffleSeed(ui64 seed = 0) {
            Seed = seed;
            return *this;
        }

        void SetPairs(const TVector<TPair>& pairs) {
            Pairs = pairs;
        }

        TDataProvider Read(TIFStream& input);

        inline TDataProvider Read(const TString& filename) {
            TIFStream input(filename);
            return Read(input);
        }

        explicit TCatBoostProtoPoolReader(TBinarizedFeaturesManager& featuresManager,
                                          bool hasTime)
            : FeaturesManager(featuresManager)
            , HasTime(hasTime)
        {
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;
        TSet<ui32> IgnoreFeatures;
        TVector<float> ClassesWeights;
        TVector<TPair> Pairs;

        bool HasTime = false;
        ui64 Seed = 0;

        void AddFeatureColumn(TIFStream& input,
                              ui32 docCount,
                              const TVector<ui64>* order,
                              TVector<TFeatureColumnPtr>* nzColumns,
                              TVector<TString>* featureNames,
                              TSet<int>* catFeatureIds);

        template <class T>
        static inline TVector<T> FromProtoToVector(const ::google::protobuf::RepeatedField<T>& data) {
            return TVector<T>(data.begin(), data.end());
        }

        NCompressedPool::TFeatureColumn FeatureColumn;
    };

}
