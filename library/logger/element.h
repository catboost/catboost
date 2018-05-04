#pragma once

#include "priority.h"

#include <util/stream/tempbuf.h>

class TLog;

/*
 * better do not use directly
 */
class TLogElement: public TTempBufOutput {
public:
    TLogElement(const TLog* parent);
    TLogElement(const TLog* parent, ELogPriority priority);

    TLogElement(TLogElement&&) noexcept = default;
    TLogElement& operator=(TLogElement&&) noexcept = default;

    ~TLogElement() override;

    template <class T>
    inline TLogElement& operator<<(const T& t) {
        static_cast<IOutputStream&>(*this) << t;

        return *this;
    }

    /*
         * for pretty usage: logger << TLOG_ERROR << "Error description";
         */
    inline TLogElement& operator<<(ELogPriority priority) {
        Flush();
        Priority_ = priority;
        return *this;
    }

    ELogPriority Priority() const noexcept {
        return Priority_;
    }

protected:
    void DoFlush() override;

protected:
    const TLog* Parent_;
    ELogPriority Priority_;
};
