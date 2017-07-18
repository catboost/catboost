#pragma once

#include "priority.h"

#include <util/stream/tempbuf.h>

class TLog;

/*
 * better do not use directly
 */
class TLogElement: public TGrowingTempBufOutput {
public:
    TLogElement(const TLog* parent);
    TLogElement(const TLog* parent, TLogPriority priority);

    TLogElement(TLogElement&&) noexcept = default;
    TLogElement& operator=(TLogElement&&) noexcept = default;

    ~TLogElement() override;

    template <class T>
    inline TLogElement& operator<<(const T& t) {
        static_cast<TOutputStream&>(*this) << t;

        return *this;
    }

    /*
         * for pretty usage: logger << TLOG_ERROR << "Error description";
         */
    inline TLogElement& operator<<(TLogPriority priority) {
        Flush();
        Priority_ = priority;
        return *this;
    }

    TLogPriority Priority() const noexcept {
        return Priority_;
    }

protected:
    void DoFlush() override;

protected:
    const TLog* Parent_;
    TLogPriority Priority_;
};
