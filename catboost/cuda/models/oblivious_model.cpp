#include "oblivious_model.h"
#include "add_oblivious_tree_model_doc_parallel.h"

namespace NCatboostCuda {
    void TObliviousTreeModel::ComputeBins(const TDocParallelDataSet& dataSet,
                                          TStripeBuffer<ui32>* dst) const {
        ComputeBinsForModel(ModelStructure, dataSet, dst);
    }

}
