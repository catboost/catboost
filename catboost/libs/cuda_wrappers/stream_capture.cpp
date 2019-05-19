#include "stream_capture.h"

#include "cuda_graph.h"

TStreamCapture::~TStreamCapture() {
    cudaGraph_t graph;
    CUDA_SAFE_CALL(cudaStreamEndCapture(Stream_, &graph));
    (*CapturedGraph_) = TCudaGraph(graph);
}

TStreamCapture TStreamCapture::Capture(TCudaStream stream,
                                       TCudaGraph* graph) {
    CUDA_SAFE_CALL(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    return TStreamCapture(graph, stream);
}
