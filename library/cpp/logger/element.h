#pragma once

#include "priority.h"
#include "record.h"


#include <util/string/cast.h>
#include <util/stream/tempbuf.h>


class TLog;

/// @warning Better don't use directly.
class TLogElement: public TTempBufOutput {
public:
    explicit TLogElement(const TLog* parent);
    explicit TLogElement(const TLog* parent, ELogPriority priority);

    TLogElement(TLogElement&&) noexcept = default;
    TLogElement& operator=(TLogElement&&) noexcept = default;

    ~TLogElement() override;

    template <class T>
    inline TLogElement& operator<<(const T& t) {
        static_cast<IOutputStream&>(*this) << t;

        return *this;
    }

    /// @note For pretty usage: logger << TLOG_ERROR << "Error description";
    inline TLogElement& operator<<(ELogPriority priority) {
        Flush();
        Priority_ = priority;
        return *this;
    }

    template<typename T>
    TLogElement& With(const TStringBuf key, const T value) {
        Context_.emplace_back(key, ToString(value));

        return *this;
    }

    ELogPriority Priority() const noexcept {
        return Priority_;
    }

protected:
    void DoFlush() override;

protected:
    const TLog* Parent_ = nullptr;
    ELogPriority Priority_;
    TLogRecord::TMetaFlags Context_;
};
