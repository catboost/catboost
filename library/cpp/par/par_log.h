#pragma once

#include <library/cpp/logger/global/global.h>
#include <library/cpp/containers/ring_buffer/ring_buffer.h>
#include <util/string/builder.h>
#include <util/system/mutex.h>
#include <util/generic/singleton.h>

namespace NPar {
    class TParLogger {
        static constexpr int MaxHistory = 512;
        TStaticRingBuffer<TString, MaxHistory> LoggedStringsBuffer;
        bool OutputToConsole = false;
        TMutex LogMutex;

    public:
        void SetOutputToConsole(bool value) {
            OutputToConsole = value;
        }

        void GetLogHistory(TVector<TString>* res) {
            res->resize(0);
            TGuard<TMutex> lock(LogMutex);
            const auto begin = LoggedStringsBuffer.FirstIndex();
            const auto end = LoggedStringsBuffer.TotalSize();
            for (size_t i = begin; i < end; ++i) {
                res->push_back(LoggedStringsBuffer[i]);
            }
        }

        void OutputLogTailToCout() {
            TGuard<TMutex> lock(LogMutex);
            const auto begin = LoggedStringsBuffer.FirstIndex();
            const auto end = LoggedStringsBuffer.TotalSize();
            for (size_t i = begin; i < end; ++i) {
                Cout << LoggedStringsBuffer[i] << Endl;
            }
        }

        void LogString(const TString& str) {
            TGuard<TMutex> lock(LogMutex);
            LoggedStringsBuffer.PushBack(str);
            if (OutputToConsole) {
                Cout << str;
            }
        }
    };

    inline TStringBuf StripFileName(TStringBuf string) {
        return string.RNextTok(LOCSLASH_C);
    }

    struct TParLoggingHelper: public TStringOutput {
        TString Buf;
        explicit TParLoggingHelper(const TSourceLocation& location)
            : TStringOutput(Buf)
        {
            (*this) << "PAR_LOG: " << NLoggingImpl::GetLocalTimeS() << " " << StripFileName(location.File) << ":" << location.Line << " ";
        }

        template <class T>
        inline TParLoggingHelper& operator<<(const T& t) {
            static_cast<IOutputStream&>(*this) << t;

            return *this;
        }

        struct THelperEater {
            Y_FORCE_INLINE bool operator|(const TParLoggingHelper& helper) const {
                Singleton<TParLogger>()->LogString(helper.Buf);
                return true;
            }
        };
    };
}

#define PAR_DEBUG_LOG NPar::TParLoggingHelper::THelperEater() | NPar::TParLoggingHelper(__LOCATION__)
