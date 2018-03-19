#pragma once

#include <util/generic/yexception.h>

class TIOStatus {
public:
    inline TIOStatus(int status) noexcept
        : Status_(status)
    {
    }

    static inline TIOStatus Error(int status) noexcept {
        return TIOStatus(status);
    }

    static inline TIOStatus Error() noexcept {
        return TIOStatus(LastSystemError());
    }

    static inline TIOStatus Success() noexcept {
        return TIOStatus(0);
    }

    inline void Check() const {
        if (Status_) {
            ythrow TSystemError(Status_) << "io error";
        }
    }

    inline bool Failed() const noexcept {
        return (bool)Status_;
    }

    inline bool Succeed() const noexcept {
        return !Failed();
    }

    inline int Status() const noexcept {
        return Status_;
    }

private:
    int Status_;
};

class TContIOStatus {
public:
    inline TContIOStatus(size_t processed, TIOStatus status) noexcept
        : Processed_(processed)
        , Status_(status)
    {
    }

    static inline TContIOStatus Error(TIOStatus status) noexcept {
        return TContIOStatus(0, status);
    }

    static inline TContIOStatus Error() noexcept {
        return TContIOStatus(0, TIOStatus::Error());
    }

    static inline TContIOStatus Success(size_t processed) noexcept {
        return TContIOStatus(processed, TIOStatus::Success());
    }

    static inline TContIOStatus Eof() noexcept {
        return Success(0);
    }

    inline ~TContIOStatus() {
    }

    inline size_t Processed() const noexcept {
        return Processed_;
    }

    inline int Status() const noexcept {
        return Status_.Status();
    }

    inline size_t Checked() const {
        Status_.Check();

        return Processed_;
    }

private:
    size_t Processed_;
    TIOStatus Status_;
};
