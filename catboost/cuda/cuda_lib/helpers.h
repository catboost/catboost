#pragma once

#include <future>
#include <util/generic/string.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>

namespace NHelpers {
    template <class T>
    inline bool IsFutureReady(std::future<T>& future) {
        auto stopStatus = future.wait_for(std::chrono::nanoseconds(0));
        if (stopStatus == std::future_status::ready) {
            return true;
        }
        return false;
    }

    static inline TSet<ui32> ParseRangeString(const TString& str, ui32 devCountLimit) {
        TSet<ui32> data;
        size_t cur = 0, prev = 0;
        while (cur != TString::npos && data.size() < devCountLimit) {
            cur = str.find(':', prev);
            TString range = str.substr(prev, cur - prev);
            size_t dash = range.find('-');
            if (range.length()) {
                if (dash == TString::npos)
                    data.insert(atoi(range.c_str()));
                else if (dash > 0 && dash < range.length() - 1) {
                    int first = atoi(range.substr(0, dash).c_str());
                    int last = atoi(range.substr(dash + 1).c_str());
                    while (first <= last && data.size() < devCountLimit)
                        data.insert(first++);
                } else
                    ythrow yexception() << "Invalid range: " << range;
            }
            prev = cur + 1;
        }
        return data;
    }

    template <class T, class U>
    constexpr inline T CeilDivide(T x, U y) {
        assert(y > 0);
        return (x + y - 1) / y;
    }
}
