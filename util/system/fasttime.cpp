#include "dynlib.h"
#include "fasttime.h"

#include <util/generic/singleton.h>
#include <util/generic/yexception.h>
#include <utility>

#include <util/thread/singleton.h>

#if defined(_win_) || defined(_arm32_) || defined(_cygwin_)
ui64 InterpolatedMicroSeconds() {
    return MicroSeconds();
}
#else

    #include <dlfcn.h>
    #include <sys/time.h>

    #if defined(_musl_)
        #include <util/generic/hash.h>
        #include <util/generic/vector.h>
        #include <util/generic/string.h>

        #include <contrib/libs/linuxvdso/interface.h>
    #endif

namespace {
    using TTime = ui64;

    struct TSymbols {
        using TFunc = int (*)(struct timeval*, struct timezone*);

        inline TSymbols()
            : Func(nullptr)
        {
            // not DEFAULT, cause library/cpp/gettimeofday
            Func = reinterpret_cast<TFunc>(dlsym(RTLD_NEXT, "gettimeofday"));

    #if defined(_musl_)
            if (!Func) {
                Func = reinterpret_cast<TFunc>(NVdso::Function("__vdso_gettimeofday", "LINUX_2.6"));
            }
    #endif

            if (!Func) {
                Func = reinterpret_cast<TFunc>(Libc()->Sym("gettimeofday"));
            }
        }

        inline TTime SystemTime() {
            timeval tv;

            Zero(tv);

            Func(&tv, nullptr);

            return (((TTime)1000000) * (TTime)tv.tv_sec) + (TTime)tv.tv_usec;
        }

        static inline THolder<TDynamicLibrary> OpenLibc() {
            const char* libs[] = {
                "/lib/libc.so.8",
                "/lib/libc.so.7",
                "/lib/libc.so.6",
            };

            for (auto& lib : libs) {
                try {
                    return MakeHolder<TDynamicLibrary>(lib);
                } catch (...) {
                    // ¯\_(ツ)_/¯
                }
            }

            ythrow yexception() << "can not load libc";
        }

        inline TDynamicLibrary* Libc() {
            if (!Lib) {
                Lib = OpenLibc();
            }

            return Lib.Get();
        }

        THolder<TDynamicLibrary> Lib;
        TFunc Func;
    };

    static inline TTime SystemTime() {
        return Singleton<TSymbols>()->SystemTime();
    }

    struct TInitialTimes {
        inline TInitialTimes()
            : ITime(TimeBase())
            , IProc(RdtscBase())
        {
        }

        static TTime RdtscBase() {
            return GetCycleCount() / (TTime)1000;
        }

        static TTime TimeBase() {
            return SystemTime();
        }

        inline TTime Rdtsc() {
            return RdtscBase() - IProc;
        }

        inline TTime Time() {
            return TimeBase() - ITime;
        }

        const TTime ITime;
        const TTime IProc;
    };

    template <size_t N, class A, class B>
    class TLinePredictor {
    public:
        using TSample = std::pair<A, B>;

        inline TLinePredictor()
            : C_(0)
            , A_(0)
            , B_(0)
        {
        }

        inline void Add(const A& a, const B& b) noexcept {
            Add(TSample(a, b));
        }

        inline void Add(const TSample& s) noexcept {
            S_[(C_++) % N] = s;
            if (C_ > 1) {
                ReCalc();
            }
        }

        inline B Predict(A a) const noexcept {
            return A_ + a * B_;
        }

        inline size_t Size() const noexcept {
            return C_;
        }

        inline bool Enough() const noexcept {
            return Size() >= N;
        }

        inline A LastX() const noexcept {
            return S_[(C_ - 1) % N].first;
        }

    private:
        inline void ReCalc() noexcept {
            const size_t n = Min(N, C_);

            double sx = 0;
            double sy = 0;
            double sxx = 0;
            double sxy = 0;

            for (size_t i = 0; i < n; ++i) {
                const double x = S_[i].first;
                const double y = S_[i].second;

                sx += x;
                sy += y;
                sxx += x * x;
                sxy += x * y;
            }

            B_ = (n * sxy - sx * sy) / (n * sxx - sx * sx);
            A_ = (sy - B_ * sx) / n;
        }

    private:
        size_t C_;
        TSample S_[N];
        double A_;
        double B_;
    };

    class TTimePredictor: public TInitialTimes {
    public:
        inline TTimePredictor()
            : Next_(1)
        {
        }

        inline TTime Get() {
            return GetBase() + ITime;
        }

    private:
        inline TTime GetBase() {
            const TTime x = Rdtsc();

            if (TimeToSync(x)) {
                const TTime y = Time();

                P_.Add(x, y);

                return y;
            }

            if (P_.Enough()) {
                return P_.Predict(x);
            }

            return Time();
        }

        inline bool TimeToSync(TTime x) {
            if (x > Next_) {
                Next_ = Min(x + x / 10, x + 1000000);

                return true;
            }

            return false;
        }

    private:
        TLinePredictor<16, TTime, TTime> P_;
        TTime Next_;
    };
} // namespace

ui64 InterpolatedMicroSeconds() {
    return FastTlsSingleton<TTimePredictor>()->Get();
}

#endif
