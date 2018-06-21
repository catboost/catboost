#include "feature_parallel_dataset_builder.h"

namespace NCatboostCuda {
    template
    class TFeatureParallelDataSetHoldersBuilder<NCudaLib::EPtrType::CudaDevice>;

    template
    class TFeatureParallelDataSetHoldersBuilder<NCudaLib::EPtrType::CudaHost>;
}
