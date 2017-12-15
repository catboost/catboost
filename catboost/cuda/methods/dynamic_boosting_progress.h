#pragma once

#include "serialization_helper.h"
#include <util/ysaveload.h>
#include <util/folder/path.h>
#include <catboost/libs/logging/logging.h>
#include <catboost/cuda/cuda_lib/read_and_write_helpers.h>

namespace NCatboostCuda {
    template <class TModel>
    struct TDynamicBoostingProgress {
        TVector<TVector<TVector<float>>> PermutationCursor;
        TVector<float> EstimationCursor;
        TVector<float> TestCursor;
        TModel Model;
        TModelFeaturesMap ModelFeaturesMap;

        Y_SAVELOAD_DEFINE(PermutationCursor, EstimationCursor, TestCursor, Model, ModelFeaturesMap);
    };

    template <class TModel,
              class TCursor,
              class TVec>
    TDynamicBoostingProgress<TModel> MakeProgress(const TBinarizedFeaturesManager& featuresManager,
                                                  const TModel& currentModel,
                                                  const TCursor& cursor, //with folds and other
                                                  const TVec* testCursor //test is one perm
    ) {
        TDynamicBoostingProgress<TModel> progress;
        progress.Model = currentModel;
        cursor.Estimation.Read(progress.EstimationCursor);
        Read(cursor.FoldData, progress.PermutationCursor);
        if (testCursor) {
            testCursor->Read(progress.TestCursor);
        }
        TModelFeaturesBuilder<TModel>::Write(featuresManager, currentModel, progress.ModelFeaturesMap);
        return progress;
    };

    template <class TModel,
              class TCursor,
              class TVec>
    void WriteProgressToGpu(const TDynamicBoostingProgress<TModel>& progress,
                            TBinarizedFeaturesManager& featuresManager,
                            TModel& writeModel,
                            TCursor& writeCursor,
                            TVec* testCursor) {
        writeModel = TFeatureIdsRemaper<TModel>::Remap(featuresManager, progress.ModelFeaturesMap, progress.Model);
        writeCursor.Estimation.Write(progress.EstimationCursor);
        Write(progress.PermutationCursor, writeCursor.FoldData);

        if (testCursor) {
            CB_ENSURE(progress.TestCursor.size() == testCursor->GetObjectsSlice().Size(), "Error: expect equal size of test set. Got " << progress.TestCursor.size() << " / " << testCursor->GetObjectsSlice().Size());
            testCursor->Write(progress.TestCursor);
        }
    };
}
