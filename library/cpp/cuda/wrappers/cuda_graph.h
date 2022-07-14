#pragma once

#include "base.h"

class TStreamCapture;

class TCudaGraph {
private:
    struct TInner: public TThrRefBase {
        TInner(cudaGraph_t graph)
            : Graph_(graph)
        {
        }

        ~TInner();

        cudaGraph_t Graph_;
    };

private:
    TIntrusivePtr<TInner> Impl_;

    friend class TStreamCapture;

    explicit TCudaGraph(cudaGraph_t graph)
        : Impl_(new TInner(graph))
    {
    }

public:
    TCudaGraph() {
    }

    operator bool() const {
        return Impl_ != nullptr;
    }

    operator cudaGraph_t() const {
        return Impl_->Graph_;
    }
};

class TCudaGraphInstance {
private:
    struct TInner: public TThrRefBase {
        TInner(cudaGraphExec_t exec)
            : Instance_(exec)
        {
        }

        ~TInner();

        cudaGraphExec_t Instance_;
    };

private:
    TCudaGraph Graph_;
    TIntrusivePtr<TInner> Impl_;

public:
    explicit TCudaGraphInstance(TCudaGraph graph);

    TCudaGraphInstance() = default;

    TCudaGraphInstance(const TCudaGraphInstance& other) = default;
    TCudaGraphInstance& operator=(const TCudaGraphInstance& other) = default;

    operator bool() const {
        return Graph_;
    }

    void Launch(TCudaStream stream) const;
};
