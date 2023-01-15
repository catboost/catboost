#pragma once

#include "base.h"
#include "cuda_graph.h"

class TStreamCapture {
public:
    ~TStreamCapture();

    static TStreamCapture Capture(TCudaStream stream, TCudaGraph* graph);

    explicit inline operator bool() const noexcept {
        return true;
    }

private:
    TStreamCapture(TCudaGraph* graph, TCudaStream stream)
        : CapturedGraph_(graph)
        , Stream_(stream)
    {
    }

private:
    TCudaGraph* CapturedGraph_;
    TCudaStream Stream_;
};

static inline TStreamCapture Capture(TCudaGraph& graph, TCudaStream stream) {
    return TStreamCapture::Capture(stream, &graph);
}

#define stream_capture(graph, stream)                                    \
    if (auto Y_GENERATE_UNIQUE_ID(__guard) = ::Capture(graph, stream)) { \
        goto Y_CAT(CAPTURE_LABEL, __LINE__);                             \
    } else                                                               \
        Y_CAT(CAPTURE_LABEL, __LINE__)                                   \
            :

