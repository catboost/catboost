#include "stream_capture.h"

#include <library/cpp/cuda/exception/exception.h>

#include <cuda_runtime.h>


namespace NCuda {
    TStreamCapture::~TStreamCapture() {
        cudaGraph_t graph;
        CUDA_SAFE_CALL_FOR_DESTRUCTOR(cudaStreamEndCapture(Stream_, &graph));
        (*CapturedGraph_) = TCudaGraph(graph);
    }

    TStreamCapture TStreamCapture::Capture(TCudaStream stream,
                                        TCudaGraph* graph) {
        CUDA_SAFE_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        return TStreamCapture(graph, stream);
    }
}
