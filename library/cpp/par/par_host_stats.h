#pragma once

#include <library/cpp/binsaver/bin_saver.h>

#include <util/generic/singleton.h>
#include <util/system/types.h>
#include <util/system/yassert.h>

#include <atomic>

namespace NPar {
    class TTiming {
        std::atomic<double> Timing;

    public:
        TTiming()
            : Timing{0} {
        }

        TTiming(double val) {
            *this = val;
        }

        operator double() const {
            return Timing.load();
        }

        int operator&(IBinSaver& f) {
            if (f.IsReading()) {
                double timing;
                f.Add(2, &timing);
                Timing.store(timing);
            } else {
                double timing = Timing.load();
                f.Add(2, &timing);
            }
            return 0;
        }

        TTiming& operator=(double val) {
            Timing.store(val);
            return *this;
        }

        TTiming& operator+=(double val) {
            double current = Timing.load();
            // using weak compare&exchange, because it may be faster
            while (!Timing.compare_exchange_weak(current, current + val)) {
            }
            return *this;
        }

        TTiming& operator-=(double val) {
            double current = Timing.load();
            while (!Timing.compare_exchange_weak(current, current - val)) {
            }
            return *this;
        }
    };

    enum class ETimingTag {
        QueryFullTime,
        QueryExecutionTime,
        TimingsCount
    };

    struct TParTimings {
        TParTimings(const TParTimings&) {
            Y_ABORT("you should not be there");
        }
        TParTimings(TParTimings&&) {
            Y_ABORT("you should not be there");
        }
        TParTimings& operator=(const TParTimings&) {
            Y_ABORT("you should not be there");
        }
        TVector<THolder<TTiming>> Timings;

        TParTimings() {
            for (size_t i = 0; i < static_cast<size_t>(ETimingTag::TimingsCount); ++i) {
                Timings.emplace_back(new TTiming);
            }
        }

        int operator&(IBinSaver& f) {
            if (f.IsReading()) {
                TVector<double> timings;
                f.Add(2, &timings);
                Y_ASSERT(timings.size() == Timings.size());
                for (size_t i = 0; i < static_cast<size_t>(ETimingTag::TimingsCount); ++i) {
                    *Timings[i] = timings[i];
                }
            } else {
                TVector<double> timings(Timings.size());
                for (size_t i = 0; i < static_cast<size_t>(ETimingTag::TimingsCount); ++i) {
                    timings[i] = *Timings[i];
                }
                f.Add(2, &timings);
            }
            return 0;
        }
    };

    struct TParHostStats {
        TParTimings ParTimings;
        SAVELOAD(ParTimings);
        static TTiming& GetTiming(ETimingTag timingTag) {
            return *Singleton<TParHostStats>()->ParTimings.Timings[static_cast<size_t>(timingTag)];
        }
    };
}
