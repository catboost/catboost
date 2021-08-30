#include "distance.h"
#include "vector_component_type.h"

#include <library/cpp/hnsw/index/dense_vector_distance.h>
#include <library/cpp/hnsw/index_builder/dense_vector_index_builder.h>
#include <library/cpp/hnsw/index_builder/dense_vector_storage.h>
#include <library/cpp/hnsw/index_builder/mobius_transform.h>
#include <library/cpp/hnsw/index_builder/index_writer.h>

#include <library/cpp/getopt/last_getopt.h>

struct TOptions {
    NHnsw::THnswBuildOptions BuildOpts;
    TString VectorFilename;
    size_t Dimension;
    EVectorComponentType VectorComponentType = EVectorComponentType::Unknown;
    EDistance Distance = EDistance::Unknown;
    TString OutputFilename;
    bool MobiusTransform = false;

    TOptions(int argc, char** argv) {
        NLastGetopt::TOpts opts = NLastGetopt::TOpts::Default();
        opts
            .AddHelpOption();
        opts
            .AddLongOption('v', "vectors")
            .RequiredArgument("FILE")
            .StoreResult(&VectorFilename)
            .Required()
            .Help("Binary file containing matrix num_vectors x dim in a row-major order.");
        opts
            .AddLongOption('t', "type")
            .RequiredArgument("STRING")
            .StoreResult(&VectorComponentType)
            .Required()
            .Help("One of { i8, i32, float }. Type of vectors' components.");
        opts
            .AddLongOption('d', "dim")
            .RequiredArgument("INT")
            .StoreResult(&Dimension)
            .Required()
            .Help("Dimension of vectors.");
        opts
            .AddLongOption('D', "distance")
            .RequiredArgument("STRING")
            .StoreResult(&Distance)
            .Required()
            .Help("One of { l1_distance, l2_sqr_distance, dot_product }.");
        opts.AddLongOption('o', "output")
            .RequiredArgument("FILE")
            .StoreResult(&OutputFilename)
            .Required();
        opts
            .AddLongOption('T', "num-threads")
            .RequiredArgument("INT")
            .DefaultValue("Number of CPUs");
        opts
            .AddLongOption('m', "max-neighbors")
            .RequiredArgument("INT")
            .StoreResult(&BuildOpts.MaxNeighbors)
            .DefaultValue(BuildOpts.MaxNeighbors)
            .Help("Number of neighbors in KNN-graph. Higher values generally result in more accurate search in expense of increased search time and memory consumption.");
        opts
            .AddLongOption('b', "batch-size")
            .RequiredArgument("INT")
            .StoreResult(&BuildOpts.BatchSize)
            .DefaultValue(BuildOpts.BatchSize)
            .Help("Batch size. Affects performance.");
        opts
            .AddLongOption('u', "upper-level-batch-size")
            .RequiredArgument("INT")
            .StoreResult(&BuildOpts.UpperLevelBatchSize)
            .DefaultValue(BuildOpts.UpperLevelBatchSize)
            .Help("Batch size for building upper levels. Affects accuracy.");
        opts
            .AddLongOption('s', "search-neighborhood-size")
            .RequiredArgument("INT")
            .StoreResult(&BuildOpts.SearchNeighborhoodSize)
            .DefaultValue(BuildOpts.SearchNeighborhoodSize)
            .Help("Search neighborhood size for ANN-search. Higher values improve search quality in expense of building time.");
        opts
            .AddLongOption('e', "num-exact-candidates")
            .RequiredArgument("INT")
            .StoreResult(&BuildOpts.NumExactCandidates)
            .DefaultValue(BuildOpts.NumExactCandidates)
            .Help("Number of nearest vectors in batch.");
        opts
            .AddLongOption('l', "level-size-decay")
            .RequiredArgument("INT")
            .DefaultValue("max-neighbors / 2")
            .Help("Base of exponent for decaying level sizes.");
        opts
            .AddLongOption("verbose")
            .NoArgument()
            .SetFlag(&BuildOpts.Verbose);
        opts
            .AddLongOption("no-progress", "Do not write progress.")
            .NoArgument()
            .StoreResult(&BuildOpts.ReportProgress, false);
        opts
            .AddLongOption("snapshot-file", "Snapshot file.")
            .RequiredArgument("PATH")
            .StoreResult(&BuildOpts.SnapshotFile);
        opts
            .AddLongOption("snapshot-interval", "Interval between saving snapshots (seconds).")
            .RequiredArgument("INT")
            .DefaultValue(BuildOpts.SnapshotInterval)
            .StoreResult(&BuildOpts.SnapshotInterval);
        opts
            .AddLongOption("mobius-transform")
            .NoArgument()
            .StoreValue(&MobiusTransform, true)
            .Help("Apply Mobius transform (may be useful for unnormalized data with dot product similarity, requires L2Sqr distance when building)");
        opts.SetFreeArgsNum(0);
        opts.AddHelpOption('h');

        NLastGetopt::TOptsParseResult parsedOpts(&opts, argc, argv);

        if (!TryFromString<size_t>(parsedOpts.Get("level-size-decay"), BuildOpts.LevelSizeDecay)) {
            BuildOpts.LevelSizeDecay = NHnsw::THnswBuildOptions::AutoSelect;
        }

        if (!TryFromString<size_t>(parsedOpts.Get("num-threads"), BuildOpts.NumThreads)) {
            BuildOpts.NumThreads = NHnsw::THnswBuildOptions::AutoSelect;
        }

        Y_VERIFY(!MobiusTransform || Distance == EDistance::L2SqrDistance, "Mobius Transformation requires L2 distance");
    }
};

template <class T, template <typename> class TDistance>
void BuildIndex(const TOptions& opts, const NHnsw::TDenseVectorStorage<T>& itemStorage) {
    auto index = NHnsw::BuildDenseVectorIndex<T, TDistance<T>>(opts.BuildOpts, itemStorage, opts.Dimension);
    NHnsw::WriteIndex(index, opts.OutputFilename);
}

template <class T>
void DispatchDistance(const TOptions& opts, const NHnsw::TDenseVectorStorage<T>& itemStorage) {
    switch (opts.Distance) {
        case EDistance::L1Distance: {
            BuildIndex<T, NHnsw::TL1Distance>(opts, itemStorage);
            break;
        }
        case EDistance::L2SqrDistance: {
            BuildIndex<T, NHnsw::TL2SqrDistance>(opts, itemStorage);
            break;
        }
        case EDistance::DotProduct: {
            BuildIndex<T, NHnsw::TDotProduct>(opts, itemStorage);
            break;
        }
        default: {
            Y_VERIFY(false, "Unknown distance!");
        }
    }
}

template <class T>
void LoadVectors(const TOptions& opts) {
    NHnsw::TDenseVectorStorage<T> itemStorage(opts.VectorFilename, opts.Dimension);
    if (opts.MobiusTransform) {
        DispatchDistance(opts, NHnsw::TransformMobius(itemStorage));
    } else {
        DispatchDistance(opts, itemStorage);
    }
}

void DispatchVectorComponentType(const TOptions& opts) {
    switch (opts.VectorComponentType) {
        case EVectorComponentType::I8: {
            LoadVectors<i8>(opts);
            break;
        }
        case EVectorComponentType::I32: {
            LoadVectors<i32>(opts);
            break;
        }
        case EVectorComponentType::Float: {
            LoadVectors<float>(opts);
            break;
        }
        default: {
            Y_VERIFY(false, "Unknown vector component type!");
        }
    }
}

void DistpatchAndBuildIndex(const TOptions& opts) {
    DispatchVectorComponentType(opts);
}

int main(int argc, char** argv) {
    TOptions opts(argc, argv);

    DistpatchAndBuildIndex(opts);

    return 0;
}
