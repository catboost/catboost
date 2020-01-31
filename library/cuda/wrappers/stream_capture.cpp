#include "stream_capture.h"

#include "cuda_graph.h"

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
