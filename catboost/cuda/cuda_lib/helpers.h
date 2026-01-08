#pragma once

#include <future>
#include <util/generic/string.h>
#include <util/generic/set.h>
#include <util/generic/vector.h>
#include <util/generic/yexception.h>
#include <util/string/cast.h>

namespace NHelpers {
    template <class T>
    inline bool IsFutureReady(std::future<T>& future) {
        auto stopStatus = future.wait_for(std::chrono::nanoseconds(0));
        if (stopStatus == std::future_status::ready) {
            return true;
        }
        return false;
    }

    static inline TSet<ui32> ParseRangeString(const TString& str, ui32 devIdLimit) {
        TSet<ui32> data;
        size_t cur = 0, prev = 0;
        while (cur != TString::npos) {
            cur = str.find(':', prev);
            TString range = str.substr(prev, cur - prev);
            try {
                auto addToData = [&] (ui32 devId) {
                    Y_ENSURE_EX(
                        devId < devIdLimit,
                        TBadArgumentException() << "id " << devId << " greater than limit " << devIdLimit
                    );
                    data.insert(devId);
                };

                Y_ENSURE_EX(!range.empty(), TBadArgumentException() << "empty");
                size_t dash = range.find('-');
                if (dash == TString::npos)
                    addToData(FromString<ui32>(range));
                else if (dash > 0 && dash < range.length() - 1) {
                    auto first = FromString<ui32>(range.substr(0, dash));
                    auto last = FromString<ui32>(range.substr(dash + 1));
                    Y_ENSURE_EX(last >= first, TBadArgumentException() << "The start of the range is greater than the end");
                    while (first <= last)
                        addToData(first++);
                } else
                    throw TBadArgumentException() << "Should be an single numeric id or '<start_id>-<end_id>'";
            } catch (TBadArgumentException& e) {
                ythrow TBadArgumentException() << "Range specification string \"" << range << "\" is invalid: " << e.what();
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
