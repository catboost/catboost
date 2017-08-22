#include "null.h"
#include "multi.h"

TMultiInput::TMultiInput(IInputStream* f, IInputStream* s) noexcept
    : C_(f)
    , N_(s)
{
}

TMultiInput::~TMultiInput() = default;

size_t TMultiInput::DoRead(void* buf, size_t len) {
    const size_t ret = C_->Read(buf, len);

    if (ret) {
        return ret;
    }

    C_ = N_;
    N_ = &Cnull;

    return C_->Read(buf, len);
}

size_t TMultiInput::DoReadTo(TString& st, char ch) {
    size_t ret = C_->ReadTo(st, ch);
    if (ret == st.size() + 1) { // found a symbol, not eof
        return ret;
    }

    C_ = N_;
    N_ = &Cnull;

    if (ret == 0) {
        ret += C_->ReadTo(st, ch);
    } else {
        TString tmp;
        ret += C_->ReadTo(tmp, ch);
        st += tmp;
    }

    return ret;
}

size_t TMultiInput::DoSkip(size_t len) {
    const size_t ret = C_->Skip(len);

    if (ret) {
        return ret;
    }

    C_ = N_;
    N_ = &Cnull;

    return C_->Skip(len);
}
