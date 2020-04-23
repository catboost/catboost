#pragma once

#include <util/generic/yexception.h>

class TIOStatus {
public:
    TIOStatus(int status) noexcept
        : Status_(status)
    {
    }

    static TIOStatus Error(int status) noexcept {
        return TIOStatus(status);
    }

    static TIOStatus Error() noexcept {
        return TIOStatus(LastSystemError());
    }

    static TIOStatus Success() noexcept {
        return TIOStatus(0);
    }

    void Check() const {
        if (Status_) {
            ythrow TSystemError(Status_) << "io error";
        }
    }

    bool Failed() const noexcept {
        return (bool)Status_;
    }

    bool Succeed() const noexcept {
        return !Failed();
    }

    int Status() const noexcept {
        return Status_;
    }

private:
    int Status_;
};


class TContIOStatus {
public:
    TContIOStatus(size_t processed, TIOStatus status) noexcept
        : Processed_(processed)
        , Status_(status)
    {
    }

    static TContIOStatus Error(TIOStatus status) noexcept {
        return TContIOStatus(0, status);
    }

    static TContIOStatus Error() noexcept {
        return TContIOStatus(0, TIOStatus::Error());
    }

    static TContIOStatus Success(size_t processed) noexcept {
        return TContIOStatus(processed, TIOStatus::Success());
    }

    static TContIOStatus Eof() noexcept {
        return Success(0);
    }

    ~TContIOStatus() {
    }

    size_t Processed() const noexcept {
        return Processed_;
    }

    int Status() const noexcept {
        return Status_.Status();
    }

    size_t Checked() const {
        Status_.Check();

        return Processed_;
    }

private:
    size_t Processed_;
    TIOStatus Status_;
};
