#include <library/cpp/colorizer/colors.h>
#include <library/statistics/statistics.h>
#include <library/testing/benchmark/bench.h>
#include <library/cpp/threading/thread_local/thread_local.h>

#include <util/datetime/base.h>
#include <util/generic/xrange.h>
#include <util/stream/format.h>
#include <util/thread/pool.h>

#include <array>

namespace {
    struct SmallStruct {
        std::array<ui8, 64> Padding;
    };

    template <NThreading::EThreadLocalImpl Impl>
    class TWrapper {
    public:
        SmallStruct& Get() {
            return *ThreadLocal_.Get();
        }

    private:
        NThreading::TThreadLocalValue<SmallStruct, Impl> ThreadLocal_{};
    };

    template <NThreading::EThreadLocalImpl Impl>
    class TWorker final: public IObjectInQueue {
    public:
        TWorker(ui32 iters, TWrapper<Impl>& wrapper)
            : Iters_{iters}
            , Wrapper_{wrapper} {
        }

        void Process(void*) override {
            for (ui32 i : xrange(Iters_)) {
                Y_UNUSED(i);
                Y_DO_NOT_OPTIMIZE_AWAY(Wrapper_.Get());
                if ((i + 1) % 100 == 0) {
                    std::this_thread::yield();
                }
            }
        }

    private:
        ui32 Iters_ = 0;
        TWrapper<Impl>& Wrapper_;
    };

    struct TBenchBase {
        virtual ~TBenchBase() {
        }

        virtual TDuration RunIter() const = 0;
        virtual TStringBuf Name() const = 0;
        virtual size_t Threads() const = 0;
        virtual size_t Reps() const = 0;
    };

    template <NThreading::EThreadLocalImpl ImplType>
    class TBench: public TBenchBase {
    public:
        TBench(ui32 threads, ui32 reps)
            : NumThreads_{threads}
            , NumReps_{reps} {
        }

        TDuration RunIter() const override {
            THolder<IThreadPool> pool = CreateThreadPool(NumThreads_);
            TVector<TWorker<ImplType>> workers(Reserve(NumThreads_));

            TWrapper<ImplType> wrapper;
            for (ui32 thread : xrange(NumThreads_)) {
                Y_UNUSED(thread);
                workers.emplace_back(NumReps_, wrapper);
                pool->SafeAdd(&workers.back());
            }

            auto start = Now();
            pool->Stop();
            return Now() - start;
        }

        TStringBuf Name() const override {
            return ToString(ImplType);
        }

        size_t Threads() const override {
            return NumThreads_;
        }

        size_t Reps() const override {
            return NumReps_;
        }

    private:
        ui32 NumThreads_;
        ui32 NumReps_;
    };
}

int main() {
    TVector<THolder<TBenchBase>> benchmarks;
    for (ui32 threads : {1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100}) {
        for (ui32 reps : {1000, 100000}) {
            benchmarks.emplace_back(new TBench<NThreading::EThreadLocalImpl::HotSwap>(threads, reps));
            benchmarks.emplace_back(new TBench<NThreading::EThreadLocalImpl::SkipList>(threads, reps));
            benchmarks.emplace_back(new TBench<NThreading::EThreadLocalImpl::ForwardList>(threads, reps));
        }
    }

    SortBy(benchmarks, [](const THolder<TBenchBase>& bench) {
        return std::make_tuple(bench->Name(), bench->Threads(), bench->Reps());
    });

    for (THolder<TBenchBase>& bench : benchmarks) {
        static constexpr TDuration timeLimit{TDuration::Seconds(3)};

        TInstant start = Now();
        NStatistics::TStatisticsCalculator<double, double> stats;
        while (Now() < start + timeLimit) {
            TDuration delta = bench->RunIter();
            stats.Push(delta.SecondsFloat());
        }

        auto& fmt = NColorizer::StdErr();
        TDuration mean = TDuration::Seconds(stats.Mean());
        TDuration stddev = TDuration::Seconds(stats.StandardDeviation());
        Cerr << fmt.LightRedColor() << bench->Name() << " [threads = " << bench->Threads() << ", reps = " << bench->Reps() << "]" << fmt.OldColor() << ": ";
        Cerr << fmt.LightCyanColor() << HumanReadable(mean) << u8" ± " << HumanReadable(stddev) << fmt.OldColor() << Endl;
    }
}
