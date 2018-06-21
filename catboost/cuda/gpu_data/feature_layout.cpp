#include "feature_layout_doc_parallel.h"
#include "feature_layout_feature_parallel.h"
#include "feature_layout_single.h"

namespace NCatboostCuda {
    template struct TGpuFeaturesBlockDescription<NCudaLib::TSingleMapping, NCudaLib::TSingleMapping>;
    template struct TGpuFeaturesBlockDescription<NCudaLib::TStripeMapping, NCudaLib::TStripeMapping>;
    template struct TGpuFeaturesBlockDescription<NCudaLib::TStripeMapping, NCudaLib::TMirrorMapping>;

    template struct TCudaFeaturesLayoutHelper<TSingleDevLayout>;
    template struct TCudaFeaturesLayoutHelper<TFeatureParallelLayout>;
    template struct TCudaFeaturesLayoutHelper<TDocParallelLayout>;



}
