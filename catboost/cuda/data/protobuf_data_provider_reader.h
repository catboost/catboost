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

namespace NCatboostCuda
{
    class TCatBoostProtoPoolReader
    {
    public:
        TDataProvider Read(TIFStream& input)
        {
            TDataProvider dataProvider;
            NCompressedPool::TPoolStructure poolStructure;
            ReadMessage(input, poolStructure);

            ReadFloatColumn(input, dataProvider.Targets);

            if (poolStructure.GetDocIdColumn())
            {
                ReadUnsignedIntColumn(input, dataProvider.DocIds);
            }

            if (poolStructure.GetQueryIdColumn())
            {
                ReadIntColumn(input, dataProvider.QueryIds);
                GroupQueries(dataProvider.QueryIds, &dataProvider.Queries);
            }

            if (poolStructure.GetWeightColumn())
            {
                ReadFloatColumn(input, dataProvider.Weights);
            } else
            {
                dataProvider.Weights.resize(dataProvider.Targets.size());
                std::fill(dataProvider.Weights.begin(), dataProvider.Weights.begin(), 1.0f);
            }

            dataProvider.Baseline.resize(poolStructure.GetBaselineColumn());
            for (ui32 i = 0; i < poolStructure.GetBaselineColumn(); ++i)
            {
                ReadFloatColumn(input, dataProvider.Baseline[i]);
            }

            for (ui32 feature = 0; feature < poolStructure.GetFeatureCount(); ++feature)
            {
                AddFeatureColumn(input, dataProvider.Features, poolStructure.GetDocCount());
            }

            dataProvider.BuildIndicesRemap();
            if (FeaturesManager.GetTargetBorders().size() == 0)
            {
                FeaturesManager.SetTargetBorders(TBordersBuilder(*GridBuilderFactory,
                                                                 dataProvider.GetTargets())(
                        FeaturesManager.GetTargetBinarizationDescription()));
            }

            return dataProvider;
        }

        inline TDataProvider Read(const TString& filename)
        {
            TIFStream input(filename);
            return Read(input);
        }

        TCatBoostProtoPoolReader& SetBinarizer(THolder<IFactory < IGridBuilder>>

        && gridBuilderFactory) {
            GridBuilderFactory = std::move(gridBuilderFactory);
            return *this;
        }

        explicit TCatBoostProtoPoolReader(TBinarizedFeaturesManager& featuresManager)
                : FeaturesManager(featuresManager)
        {
        }

    private:
        TBinarizedFeaturesManager& FeaturesManager;

        void AddFeatureColumn(TIFStream& input, yvector<TFeatureColumnPtr>& features, ui32 docCount);

        template<class T>
        static inline yvector<T> FromProtoToVector(const ::google::protobuf::RepeatedField<T>& data)
        {
            return yvector<T>(data.begin(), data.end());
        }

        THolder<IFactory < IGridBuilder>> GridBuilderFactory;
        NCompressedPool::TFeatureColumn FeatureColumn;
    };
}
