#pragma once

#include <library/cpp/json/json_reader.h>
#include <util/generic/fwd.h>
#include <util/generic/strbuf.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>
#include <util/memory/blob.h>
#include <util/ysaveload.h>

#include <stddef.h>


namespace NHnsw {
    struct THnswBuildOptions {
        size_t MaxNeighbors = 32;
        size_t BatchSize = 1000;
        size_t UpperLevelBatchSize = 40000;
        size_t SearchNeighborhoodSize = 300;
        size_t NumExactCandidates = 100;
        size_t LevelSizeDecay = AutoSelect;
        size_t NumThreads = AutoSelect;
        bool Verbose = false;
        bool ReportProgress = true;
        TString SnapshotFile = "";
        TBlob* SnapshotBlobPtr = nullptr;
        double SnapshotInterval = 600; // seconds

        static constexpr size_t AutoSelect = 0;

        static THnswBuildOptions FromJsonString(const TString& jsonString) {
            NJson::TJsonValue json;
            Y_ENSURE(ReadJsonTree(TStringBuf(jsonString), &json));

            THnswBuildOptions options;
            options.MaxNeighbors = json["max_neighbors"].GetUIntegerSafe(options.MaxNeighbors);
            options.BatchSize = json["batch_size"].GetUIntegerSafe(options.BatchSize);
            options.UpperLevelBatchSize = json["upper_level_batch_size"].GetUIntegerSafe(options.UpperLevelBatchSize);
            options.SearchNeighborhoodSize = json["search_neighborhood_size"].GetUIntegerSafe(options.SearchNeighborhoodSize);
            options.NumExactCandidates = json["num_exact_candidates"].GetUIntegerSafe(options.NumExactCandidates);
            options.LevelSizeDecay = json["level_size_decay"].GetUIntegerSafe(options.LevelSizeDecay);
            options.NumThreads = json["num_threads"].GetUIntegerSafe(options.NumThreads);
            options.Verbose = json["verbose"].GetBooleanSafe(options.Verbose);
            options.ReportProgress = json["report_progress"].GetBooleanSafe(options.ReportProgress);
            options.SnapshotFile = json["snapshot_file"].GetStringSafe(options.SnapshotFile);
            options.SnapshotInterval = json["snapshot_interval"].GetDoubleSafe(options.SnapshotInterval);

            return options;
        }

        void CheckOptions() const {
            Y_ENSURE(1 <= MaxNeighbors && MaxNeighbors <= SearchNeighborhoodSize);
            Y_ENSURE(1 <= MaxNeighbors && MaxNeighbors <= NumExactCandidates);
            Y_ENSURE(BatchSize > MaxNeighbors);
            Y_ENSURE(LevelSizeDecay == THnswBuildOptions::AutoSelect || LevelSizeDecay > 1);
        }

        Y_SAVELOAD_DEFINE(
            MaxNeighbors,
            BatchSize,
            UpperLevelBatchSize,
            SearchNeighborhoodSize,
            NumExactCandidates,
            LevelSizeDecay,
            NumThreads,
            Verbose,
            ReportProgress);
    };

}
