#pragma once

#include "build_options.h"

#include <util/generic/string.h>
#include <util/generic/utility.h>
#include <util/system/info.h>

#include <stddef.h>


namespace NHnsw {
    struct THnswInternalBuildOptions {
        size_t MaxNeighbors;
        size_t BatchSize;
        size_t UpperLevelBatchSize;
        size_t SearchNeighborhoodSize;
        size_t NumExactCandidates;
        size_t LevelSizeDecay;
        size_t NumThreads;
        bool Verbose = false;
        bool ReportProgress = true;
        TString SnapshotFile;
        TBlob* SnapshotBlobPtr;
        double SnapshotInterval;

        THnswInternalBuildOptions() = default;

        explicit THnswInternalBuildOptions(const THnswBuildOptions& opts) {
            opts.CheckOptions();

            MaxNeighbors = opts.MaxNeighbors;
            BatchSize = opts.BatchSize;
            UpperLevelBatchSize = Max(opts.UpperLevelBatchSize, opts.BatchSize);
            SearchNeighborhoodSize = opts.SearchNeighborhoodSize;
            NumExactCandidates = opts.NumExactCandidates;
            LevelSizeDecay = opts.LevelSizeDecay;
            if (opts.LevelSizeDecay == THnswBuildOptions::AutoSelect) {
                LevelSizeDecay = Max<size_t>(2, opts.MaxNeighbors / 2);
            }
            NumThreads = opts.NumThreads;
            if (opts.NumThreads == THnswBuildOptions::AutoSelect) {
                NumThreads = NSystemInfo::CachedNumberOfCpus();
            }
            Verbose = opts.Verbose;
            ReportProgress = opts.ReportProgress;
            SnapshotFile = opts.SnapshotFile;
            SnapshotInterval = opts.SnapshotInterval;
            SnapshotBlobPtr = opts.SnapshotBlobPtr;
        }
    };
}
