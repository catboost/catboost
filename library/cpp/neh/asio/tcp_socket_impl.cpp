#include "tcp_socket_impl.h"

using namespace NAsio;

TSocketOperation::TSocketOperation(TTcpSocket::TImpl& s, TPollType pt, TInstant deadline)
    : TFdOperation(s.Fd(), pt, deadline)
    , S_(s)
{
}

bool TOperationWrite::Execute(int errorCode) {
    if (errorCode) {
        H_(errorCode, Written_, *this);

        return true; //op. completed
    }

    TErrorCode ec;
    TContIOVector& iov = *Buffs_->GetIOvec();

    size_t n = S_.WriteSome(iov, ec);

    if (ec && ec.Value() != EAGAIN && ec.Value() != EWOULDBLOCK) {
        H_(ec, Written_ + n, *this);

        return true;
    }

    if (n) {
        Written_ += n;
        iov.Proceed(n);
        if (!iov.Bytes()) {
            H_(ec, Written_, *this);

            return true; //op. completed
        }
    }

    return false; //operation not compleled
}

bool TOperationWriteVector::Execute(int errorCode) {
    if (errorCode) {
        H_(errorCode, Written_, *this);

        return true; //op. completed
    }

    TErrorCode ec;

    size_t n = S_.WriteSome(V_, ec);

    if (ec && ec.Value() != EAGAIN && ec.Value() != EWOULDBLOCK) {
        H_(ec, Written_ + n, *this);

        return true;
    }

    if (n) {
        Written_ += n;
        V_.Proceed(n);
        if (!V_.Bytes()) {
            H_(ec, Written_, *this);

            return true; //op. completed
        }
    }

    return false; //operation not compleled
}

bool TOperationReadSome::Execute(int errorCode) {
    if (errorCode) {
        H_(errorCode, 0, *this);

        return true; //op. completed
    }

    TErrorCode ec;

    H_(ec, S_.ReadSome(Buff_, Size_, ec), *this);

    return true;
}

bool TOperationRead::Execute(int errorCode) {
    if (errorCode) {
        H_(errorCode, Read_, *this);

        return true; //op. completed
    }

    TErrorCode ec;
    size_t n = S_.ReadSome(Buff_, Size_, ec);
    Read_ += n;

    if (ec && ec.Value() != EAGAIN && ec.Value() != EWOULDBLOCK) {
        H_(ec, Read_, *this);

        return true; //op. completed
    }

    if (n) {
        Size_ -= n;
        if (!Size_) {
            H_(ec, Read_, *this);

            return true;
        }
        Buff_ += n;
    } else if (!ec) {  // EOF while read not all
        H_(ec, Read_, *this);
        return true;
    }

    return false;
}
