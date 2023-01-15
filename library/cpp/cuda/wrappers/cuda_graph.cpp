#include "cuda_graph.h"

TCudaGraph::TInner::~TInner() {
    CUDA_SAFE_CALL_FOR_DESTRUCTOR(cudaGraphDestroy(Graph_));
}

TCudaGraphInstance::TInner::~TInner() {
    CUDA_SAFE_CALL_FOR_DESTRUCTOR(cudaGraphExecDestroy(Instance_));
}

TCudaGraphInstance::TCudaGraphInstance(TCudaGraph graph)
    : Graph_(graph)
{
    cudaGraphExec_t instance;
    CUDA_SAFE_CALL(cudaGraphInstantiate(&instance, Graph_, NULL, NULL, 0));
    Impl_ = new TInner(instance);
}
void TCudaGraphInstance::Launch(TCudaStream stream) const {
    CUDA_SAFE_CALL(cudaGraphLaunch(Impl_->Instance_, stream));
}
