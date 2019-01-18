#include "non_symmetric_tree.h"
#include "add_non_symmetric_tree_doc_parallel.h"

namespace NCatboostCuda {
    void TNonSymmetricTree::ComputeBins(const TDocParallelDataSet& dataSet,
                                        TStripeBuffer<ui32>* dst) const {
        ComputeBinsForModel(ModelStructure, dataSet, dst);
    }
}
