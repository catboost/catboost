#include "cat_features_dataset.h"

namespace NCatboostCuda {
     template class TCompressedCatFeatureDataSet<NCudaLib::EPtrType::CudaDevice>;
     template class TCompressedCatFeatureDataSetBuilder<NCudaLib::EPtrType::CudaDevice>;
     template class TMirrorCatFeatureProvider<NCudaLib::EPtrType::CudaDevice>;

     template class TCompressedCatFeatureDataSet<NCudaLib::EPtrType::CudaHost>;
     template class TCompressedCatFeatureDataSetBuilder<NCudaLib::EPtrType::CudaHost>;
     template class TMirrorCatFeatureProvider<NCudaLib::EPtrType::CudaHost>;
}
