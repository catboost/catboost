#pragma once

#include "cuda_manager.h"
#include <cmath>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

namespace NCudaLib {
    enum class EProfileMode {
        //profile host + gpu operation time
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

    public:
        TLabeledInterval(const TString& label,
                         EProfileMode profileMode = EProfileMode::LabelAsync)
            : Label(label)
            , Count(0)
            , Max(0.0)
            , Sum(0.0)
            , Sum2(0.0)
            , Active(false)
            , ProfileMode(profileMode)
        {
        }

        ~TLabeledInterval() {
            Y_VERIFY(!Active, "Exit application before stopping LabelInterval");
        }

        void Add(const TLabeledInterval& other) {
            CB_ENSURE(other.Label == Label);
            CB_ENSURE(!other.Active, "Can't add running label interval. Inconsistent cuda-mangers state");
            Max = std::max(Max, other.Max);
            Sum += other.Sum;
            Sum2 += other.Sum2;
            Count += other.Count;
        }

        void PrintInfo() {
            if (Count == 0) {
                return;
            }

            double mean = Sum / Count;
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
        }
    };

    class TCudaProfiler {
    private:
        TMap<TString, THolder<TLabeledInterval>> Labels;
        EProfileMode DefaultProfileMode;
        ui64 MinProfileLevel;
        TLabeledInterval EmptyLabel;
        bool PrintOnDelete = true;

    public:
        TCudaProfiler(EProfileMode profileMode = EProfileMode::LabelAsync,
                      ui64 level = 0,
                      bool printOnDelete = true)
            : DefaultProfileMode(profileMode)
            , MinProfileLevel(level)
            , EmptyLabel("fake", EProfileMode::NoProfile)
            , PrintOnDelete(printOnDelete)
        {
        }

        ~TCudaProfiler() {
            if (PrintOnDelete) {
                PrintInfo();
            }
        }

        void PrintInfo() {
            for (auto& label : Labels) {
                label.second->PrintInfo();
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
