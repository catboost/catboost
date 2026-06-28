#include "conn.h"

#include <util/network/socket.h>

namespace {
    class TBlockingSocketStreams: public THttpServerConn::ISocketStreams {
    public:
        explicit TBlockingSocketStreams(const TSocket& s)
            : Socket_(s)
            , Input_(Socket_)
            , Output_(Socket_)
        {
        }

        IInputStream* Input() override {
            return &Input_;
        }

        IOutputStream* Output() override {
            return &Output_;
        }

        void Reset() override {
            if (Socket_ != INVALID_SOCKET) {
                // send RST packet to client
                Socket_.SetLinger(true, 0);
                Socket_.Close();
           }
        }
    private:
        TSocket Socket_;
        TSocketInput Input_;
        TSocketOutput Output_;
    };
}

THttpServerConn::THttpServerConn(const TSocket& s)
    : THttpServerConn(s, s.MaximumTransferUnit())
{
}

THttpServerConn::THttpServerConn(const TSocket& s, size_t outputBufferSize)
    : THttpServerConn(MakeHolder<TBlockingSocketStreams>(s), outputBufferSize)
{
}

THttpServerConn::THttpServerConn(THolder<ISocketStreams> socketStreams, size_t outputBufferSize)
    : SocketStreams_(std::move(socketStreams))
    , BufferedOutput_(SocketStreams_->Output(), outputBufferSize)
    , HttpInput_(SocketStreams_->Input())
    , HttpOutput_(&BufferedOutput_, &HttpInput_)
{
}

THttpServerConn::~THttpServerConn() {
}

void THttpServerConn::Reset() {
    return SocketStreams_->Reset();
}
