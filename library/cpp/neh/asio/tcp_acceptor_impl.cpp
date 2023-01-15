#include "tcp_acceptor_impl.h"

using namespace NAsio;

bool TOperationAccept::Execute(int errorCode) {
    if (errorCode) {
        H_(errorCode, *this);

        return true;
    }

    struct sockaddr_storage addr;
    socklen_t sz = sizeof(addr);

    SOCKET res = ::accept(Fd(), (sockaddr*)&addr, &sz);

    if (res == INVALID_SOCKET) {
        H_(LastSystemError(), *this);
    } else {
        NS_.Assign(res, TEndpoint(new NAddr::TOpaqueAddr((sockaddr*)&addr)));
        H_(0, *this);
    }

    return true;
}
