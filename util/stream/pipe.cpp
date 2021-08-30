#include "pipe.h"

#include <util/generic/yexception.h>

#include <cstdio>
#include <cerrno>

class TPipeBase::TImpl {
public:
    inline TImpl(const TString& command, const char* mode)
        : Pipe_(nullptr)
    {
#ifndef _freebsd_
        if (strcmp(mode, "r+") == 0) {
            ythrow TSystemError(EINVAL) << "pipe \"r+\" mode is implemented only on FreeBSD";
        }
#endif
        Pipe_ = ::popen(command.data(), mode);
        if (Pipe_ == nullptr) {
            ythrow TSystemError() << "failed to open pipe: " << command.Quote();
        }
    }

    inline ~TImpl() {
        if (Pipe_ != nullptr) {
            ::pclose(Pipe_);
        }
    }

public:
    FILE* Pipe_;
};

TPipeBase::TPipeBase(const TString& command, const char* mode)
    : Impl_(new TImpl(command, mode))
{
}

TPipeBase::~TPipeBase() = default;

TPipeInput::TPipeInput(const TString& command)
    : TPipeBase(command, "r")
{
}

size_t TPipeInput::DoRead(void* buf, size_t len) {
    if (Impl_->Pipe_ == nullptr) {
        return 0;
    }

    size_t bytesRead = ::fread(buf, 1, len, Impl_->Pipe_);
    if (bytesRead == 0) {
        int exitStatus = ::pclose(Impl_->Pipe_);
        Impl_->Pipe_ = nullptr;
        if (exitStatus == -1) {
            ythrow TSystemError() << "pclose() failed";
        } else if (exitStatus != 0) {
            ythrow yexception() << "subprocess exited with non-zero status(" << exitStatus << ")";
        }
    }
    return bytesRead;
}

TPipeOutput::TPipeOutput(const TString& command)
    : TPipeBase(command, "w")
{
}

void TPipeOutput::DoWrite(const void* buf, size_t len) {
    if (Impl_->Pipe_ == nullptr || len != ::fwrite(buf, 1, len, Impl_->Pipe_)) {
        ythrow TSystemError() << "fwrite failed";
    }
}

void TPipeOutput::Close() {
    int exitStatus = ::pclose(Impl_->Pipe_);
    Impl_->Pipe_ = nullptr;
    if (exitStatus == -1) {
        ythrow TSystemError() << "pclose() failed";
    } else if (exitStatus != 0) {
        ythrow yexception() << "subprocess exited with non-zero status(" << exitStatus << ")";
    }
}

TPipedBase::TPipedBase(PIPEHANDLE fd)
    : Handle_(fd)
{
}

TPipedBase::~TPipedBase() {
    if (Handle_.IsOpen()) {
        Handle_.Close();
    }
}

TPipedInput::TPipedInput(PIPEHANDLE fd)
    : TPipedBase(fd)
{
}

TPipedInput::~TPipedInput() = default;

size_t TPipedInput::DoRead(void* buf, size_t len) {
    if (!Handle_.IsOpen()) {
        return 0;
    }
    return Handle_.Read(buf, len);
}

TPipedOutput::TPipedOutput(PIPEHANDLE fd)
    : TPipedBase(fd)
{
}

TPipedOutput::~TPipedOutput() = default;

void TPipedOutput::DoWrite(const void* buf, size_t len) {
    if (!Handle_.IsOpen() || static_cast<ssize_t>(len) != Handle_.Write(buf, len)) {
        ythrow TSystemError() << "pipe writing failed";
    }
}
