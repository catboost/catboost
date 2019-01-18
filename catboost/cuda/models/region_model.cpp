#include "region_model.h"
#include "add_region_doc_parallel.h"

namespace NCatboostCuda {
    void TRegionModel::ComputeBins(const TDocParallelDataSet& dataSet,
                                   TStripeBuffer<ui32>* dst) const {
        ComputeBinsForModel(ModelStructure, dataSet, dst);
    }
}
