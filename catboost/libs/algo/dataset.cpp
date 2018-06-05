#include "dataset.h"

TDataset BuildDataset(const TPool& pool) {
    TDataset data;
    data.Target = pool.Docs.Target;
    data.Weights = pool.Docs.Weight;
    data.QueryId = pool.Docs.QueryId;
    data.SubgroupId = pool.Docs.SubgroupId;
    data.Baseline = pool.Docs.Baseline;
    data.Pairs = pool.Pairs;
    data.HasGroupWeight = pool.MetaInfo.HasGroupWeight;
    return data;
}
