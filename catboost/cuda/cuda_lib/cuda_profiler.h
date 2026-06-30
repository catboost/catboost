#pragma once

#include "cuda_manager.h"
#include <cmath>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

namespace NCudaLib {
    enum class EProfileMode {
        //profile host + GPU operation time
        ImplicitLabelSync,
        //profile host send and recieve result time
        LabelAsync,
        NoProfile
    };

    class TLabeledInterval {
        TString Label;
        std::chrono::high_resolution_clock::time_point Time;
        ui64 Count;
        double Max;
        double Sum;
        double Sum2;
        bool Active;
        EProfileMode ProfileMode;
        ui32* Nestedness;
        ui32 TabSize = 0;
        TMaybe<std::chrono::high_resolution_clock::time_point> Timestamp;

        void UpdateTabSize(ui32 tabSize) {
            if (TabSize != tabSize) {
                CATBOOST_WARNING_LOG
                    << "Warning: found " << Label << " at different level in call stack"
                    << " -- will show this label at highest level" << Endl;
                TabSize = Min(TabSize, tabSize);
            }
        }

    public:
        TLabeledInterval(const TString& label, ui32* nestedness,
                         EProfileMode profileMode = EProfileMode::LabelAsync)
            : Label(label)
            , Count(0)
            , Max(0.0)
            , Sum(0.0)
            , Sum2(0.0)
            , Active(false)
            , ProfileMode(profileMode)
            , Nestedness(nestedness)
        {
            CB_ENSURE(nestedness, "Need nestedness counter");
            TabSize = *nestedness;
        }

        ~TLabeledInterval() noexcept(false) {
            if (Active) {
                CATBOOST_WARNING_LOG << "Profiled code terminates before profiling has been finished. "
                    << "Profile data may be inconsistent" << Endl;
                if (!std::uncaught_exceptions()) {
                    CB_ENSURE(!Active);
                }
            }
        }

        void Add(const TLabeledInterval& other) {
            CB_ENSURE(other.Label == Label);
            CB_ENSURE(!other.Active, "Can't add running label interval. Inconsistent cuda-manager's state");
            Max = std::max(Max, other.Max);
            Sum += other.Sum;
            Sum2 += other.Sum2;
            Count += other.Count;
            UpdateTabSize(other.TabSize);
        }

        void PrintInfo() const {
            if (Count == 0) {
                return;
            }

            double mean = Sum / Count;
            CATBOOST_INFO_LOG << TString(TabSize * 2, ' ');
            CATBOOST_INFO_LOG << Label
                              << " count " << Count
                              << " mean: " << mean
                              << " max: " << Max
                              << " rmse: " << sqrt((Sum2 - Sum * mean) / Count)
                              << Endl;
        }

        void Acquire() {
            CB_ENSURE(!Active, "Error: label is already aquired " + Label);
            Active = true;
            if (ProfileMode == EProfileMode::NoProfile) {
                return;
            }
            if (ProfileMode == EProfileMode::ImplicitLabelSync) {
                GetCudaManager().WaitComplete();
            }
            Time = std::chrono::high_resolution_clock::now();
            if (!Timestamp) {
                Timestamp = Time;
            }
            ++*Nestedness;
        }

        void Release() {
            CB_ENSURE(Active, "Can't release non-active label " + Label);
            Active = false;
            if (ProfileMode == EProfileMode::NoProfile) {
                return;
            }

            if (ProfileMode == EProfileMode::ImplicitLabelSync) {
                GetCudaManager().WaitComplete();
            }

            auto elapsed = std::chrono::high_resolution_clock::now() - Time;
            double val = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() * 1.0 / 1000 / 1000;

            Max = std::max(Max, val);
            Sum += val;
            Sum2 += val * val;
            Count++;
            --*Nestedness;
        }

        inline bool operator<(const TLabeledInterval& right) const {
            return Timestamp < right.Timestamp;
        }
    };

    class TCudaProfiler {
    private:
        TMap<TString, THolder<TLabeledInterval>> Labels;
        EProfileMode DefaultProfileMode;
        ui64 MinProfileLevel;
        TLabeledInterval EmptyLabel;
        bool PrintOnDelete = true;
        ui32 Nestedness = 0;

    public:
        TCudaProfiler(EProfileMode profileMode = EProfileMode::LabelAsync,
                      ui64 level = 0,
                      bool printOnDelete = true)
            : DefaultProfileMode(profileMode)
            , MinProfileLevel(level)
            , EmptyLabel("fake", &Nestedness, EProfileMode::NoProfile)
            , PrintOnDelete(printOnDelete)
        {
        }

        ~TCudaProfiler() {
            if (PrintOnDelete) {
                PrintInfo();
            }
        }

        void PrintInfo() {
            TVector<TLabeledInterval> intervals;
            intervals.reserve(Labels.size());
            for (const auto& [label, interval] : Labels) {
                intervals.push_back(*interval);
            }
            Sort(intervals);
            for (const auto& interval : intervals) {
                interval.PrintInfo();
            }
        }

        inline void SetProfileLevel(ui64 level) {
            MinProfileLevel = level;
        }

        inline void Add(const TCudaProfiler& other) {
            for (const auto& entry : other.Labels) {
                const auto& label = entry.first;
                if (Labels.count(label) == 0) {
                    Labels[entry.first] = MakeHolder<TLabeledInterval>(label,
                                                                       &Nestedness,
                                                                       DefaultProfileMode);
                }
                Labels[entry.first]->Add(*entry.second);
            }
        }

        inline void SetDefaultProfileMode(EProfileMode mode) {
            DefaultProfileMode = mode;
        }

        inline EProfileMode GetDefaultProfileMode() {
            return DefaultProfileMode;
        }

        inline TGuard<TLabeledInterval> Profile(const TString& label,
                                                ui64 profileLevel = 0) {
            if (profileLevel < MinProfileLevel) {
                return Guard(EmptyLabel);
            }

            if (!Labels.count(label)) {
                Labels[label] = MakeHolder<TLabeledInterval>(label,
                                                             &Nestedness,
                                                             DefaultProfileMode);
            }
            return Guard(*Labels[label]);
        }
    };

    inline TCudaProfiler& GetProfiler() {
        auto& manager = GetCudaManager();
        return manager.GetProfiler();
    }

    inline void SetDefaultProfileMode(EProfileMode mode) {
        auto& manager = GetCudaManager();
        manager
            .GetProfiler()
            .SetDefaultProfileMode(mode);
    }

    inline void SetProfileLevel(ui64 level) {
        auto& manager = GetCudaManager();
        manager
            .GetProfiler()
            .SetProfileLevel(level);
    }
}
