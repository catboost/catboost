#include "bench.h"

#include <contrib/libs/re2/re2/re2.h>

#include <library/cpp/colorizer/output.h>
#include <library/cpp/getopt/small/last_getopt.h>
#include <library/cpp/json/json_value.h>
#include <library/cpp/linear_regression/linear_regression.h>
#include <library/cpp/threading/poor_man_openmp/thread_helper.h>

#include <util/generic/ptr.h>
#include <util/system/hp_timer.h>
#include <util/system/info.h>
#include <util/stream/output.h>
#include <util/datetime/base.h>
#include <util/random/random.h>
#include <util/string/cast.h>
#include <util/generic/xrange.h>
#include <util/generic/algorithm.h>
#include <util/generic/singleton.h>
#include <util/system/spinlock.h>
#include <util/generic/function.h>
#include <util/generic/maybe.h>
#include <util/generic/strbuf.h>
#include <util/generic/intrlist.h>
#include <util/stream/format.h>
#include <util/stream/file.h>
#include <util/system/yield.h>

using re2::RE2;

using namespace NBench;
using namespace NColorizer;
using namespace NLastGetopt;

namespace {
    struct TOptions {
        double TimeBudget;
    };

    struct TResult {
        TStringBuf TestName;
        ui64 Samples;
        ui64 Iterations;
        TMaybe<double> CyclesPerIteration;
        TMaybe<double> SecondsPerIteration;
        double RunTime;
        size_t TestId;  //  Sequential test id (zero-based)
    };

    struct ITestRunner: public TIntrusiveListItem<ITestRunner> {
        virtual ~ITestRunner() = default;
        void Register();

        virtual TStringBuf Name() const noexcept = 0;
        virtual TResult Run(const TOptions& opts) = 0;
        size_t SequentialId = 0;
    };

    struct TCpuBenchmark: public ITestRunner {
        inline TCpuBenchmark(const char* name, NCpu::TUserFunc func)
            : F(func)
            , N(name)
        {
            Register();
        }

        TResult Run(const TOptions& opts) override;

        TStringBuf Name() const noexcept override {
            return N;
        }

        std::function<NCpu::TUserFunc> F;
        const TStringBuf N;
    };

    inline TString DoFmtTime(double t) {
        if (t > 0.1) {
            return ToString(t) + " seconds";
        }

        t *= 1000.0;

        if (t > 0.1) {
            return ToString(t) + " milliseconds";
        }

        t *= 1000.0;

        if (t > 0.1) {
            return ToString(t) + " microseconds";
        }

        t *= 1000.0;

        if (t < 0.05) {
            t = 0.0;
        }

        return ToString(t) + " nanoseconds";
    }

    struct THiPerfTimer: public THPTimer {
        static inline TString FmtTime(double t) {
            return DoFmtTime(t);
        }
    };

    struct TSimpleTimer {
        inline double Passed() const noexcept {
            return (TInstant::Now() - N).MicroSeconds() / 1000000.0;
        }

        static inline TString FmtTime(double t) {
            return DoFmtTime(t);
        }

        const TInstant N = TInstant::Now();
    };

    struct TCycleTimer {
        inline ui64 Passed() const noexcept {
            return GetCycleCount() - N;
        }

        static inline TString FmtTime(double t) {
            if (t < 0.5) {
                t = 0.0;
            }

            TString hr;
            if (t > 10 * 1000) {
                hr = " (" + ToString(HumanReadableSize(t, ESizeFormat::SF_QUANTITY)) + ")";
            }

            return ToString(t) + hr + " cycles";
        }

        const ui64 N = GetCycleCount();
    };

    template <class TMyTimer, class T>
    inline double Measure(T&& t, size_t n) {
        TMyTimer timer;

        t(n);

        return timer.Passed();
    }

    struct TSampleIterator {
        inline size_t Next() noexcept {
            return M++;

            N *= 1.02;
            M += 1;

            return Max<double>(N, M);
        }

        double N = 1.0;
        size_t M = 1;
    };

    using TSample = std::pair<size_t, double>;
    using TSamples = TVector<TSample>;

    struct TLinFunc {
        double A;
        double B;

        inline double operator()(double x) const noexcept {
            return A * x + B;
        }
    };

    TLinFunc CalcModel(const TSamples& s) {
        TKahanSLRSolver solver;

        for (const auto& p : s) {
            solver.Add(p.first, p.second);
        }

        double c = 0;
        double i = 0;

        solver.Solve(c, i);

        return TLinFunc{c, i};
    }

    inline TSamples RemoveOutliers(const TSamples& s, double fraction) {
        if (s.size() < 20) {
            return s;
        }

        const auto predictor = CalcModel(s);

        const auto errfunc = [&predictor](const TSample& p) -> double {
            //return (1.0 + fabs(predictor(p.first) - p.second)) / (1.0 + fabs(p.second));
            //return fabs((predictor(p.first) - p.second)) / (1.0 + fabs(p.second));
            //return fabs((predictor(p.first) - p.second)) / (1.0 + p.first);
            return fabs((predictor(p.first) - p.second));
        };

        using TSampleWithError = std::pair<const TSample*, double>;
        TVector<TSampleWithError> v;

        v.reserve(s.size());

        for (const auto& p : s) {
            v.emplace_back(&p, errfunc(p));
        }

        Sort(v.begin(), v.end(), [](const TSampleWithError& l, const TSampleWithError& r) -> bool {
            return (l.second < r.second) || ((l.second == r.second) && (l.first < r.first));
        });

        if (0) {
            for (const auto& x : v) {
                Cout << x.first->first << ", " << x.first->second << " -> " << x.second << Endl;
            }
        }

        TSamples ret;

        ret.reserve(v.size());

        for (const auto i : xrange<size_t>(0, fraction * v.size())) {
            ret.push_back(*v[i].first);
        }

        return ret;
    }

    template <class TMyTimer, class T>
    static inline TResult RunTest(T&& func, double budget, ITestRunner& test) {
        THPTimer start;

        start.Passed();

        TSampleIterator sample;
        TSamples samples;
        ui64 iters = 0;

        //warm up
        func(1);

        while (start.Passed() < budget) {
            if (start.Passed() < ((budget * samples.size()) / 2000000.0)) {
                ThreadYield();
            } else {
                const size_t n = sample.Next();

                iters += (ui64)n;
                samples.emplace_back(n, Measure<TMyTimer>(func, n));
            }
        }

        auto filtered = RemoveOutliers(samples, 0.9);

        return {test.Name(), filtered.size(), iters, CalcModel(filtered).A, Nothing(), start.Passed(), test.SequentialId};
    }

    using TTests = TIntrusiveListWithAutoDelete<ITestRunner, TDestructor>;

    inline TTests& Tests() {
        return *Singleton<TTests>();
    }

    void ITestRunner::Register() {
        Tests().PushBack(this);
    }

    TResult TCpuBenchmark::Run(const TOptions& opts) {
        return RunTest<TCycleTimer>([this](size_t n) {
            NCpu::TParams params{n};

            F(params);
        }, opts.TimeBudget, *this);
    }

    enum EOutFormat {
        F_CONSOLE = 0 /* "console" */,
        F_CSV /* "csv" */,
        F_JSON /* "json" */
    };

    TAdaptiveLock STDOUT_LOCK;

    struct IReporter {
        virtual void Report(TResult&& result) = 0;

        virtual void Finish() {
        }

        virtual ~IReporter() {
        }
    };

    class TConsoleReporter: public IReporter {
    public:
        TConsoleReporter(IOutputStream& outputStream) : OutputStream(outputStream) {
        }

        ~TConsoleReporter() override {
        }

        void Report(TResult&& r) override {
            with_lock (STDOUT_LOCK) {
                OutputStream << r;
            }
        }
    private:
        IOutputStream& OutputStream;
    };

    class TCSVReporter: public IReporter {
    public:
        TCSVReporter(IOutputStream& outputStream) : OutputStream(outputStream) {
            outputStream << "Name\tSamples\tIterations\tRun_time\tPer_iteration_sec\tPer_iteration_cycles" << Endl;
        }

        ~TCSVReporter() override {
        }

        void Report(TResult&& r) override {
            with_lock (STDOUT_LOCK) {
                OutputStream << r.TestName
                     << '\t' << r.Samples
                     << '\t' << r.Iterations
                     << '\t' << r.RunTime;

                OutputStream << '\t';
                if (r.CyclesPerIteration) {
                    OutputStream << TCycleTimer::FmtTime(*r.CyclesPerIteration);
                } else {
                    OutputStream << '-';
                }

                OutputStream << '\t';
                if (r.SecondsPerIteration) {
                    OutputStream << DoFmtTime(*r.SecondsPerIteration);
                } else {
                    OutputStream << '-';
                }

                OutputStream << Endl;
            }
        }
    private:
        IOutputStream& OutputStream;
    };

    class TJSONReporter: public IReporter {
    public:
        TJSONReporter(IOutputStream& outputStream) : OutputStream(outputStream) {
        }

        ~TJSONReporter() override {
        }

        void Report(TResult&& r) override {
            with_lock (ResultsLock_) {
                Results_.emplace_back(std::move(r));
            }
        }

        void Finish() override {
            NJson::TJsonValue report;
            auto& bench = report["benchmark"];
            bench.SetType(NJson::JSON_ARRAY);

            NJson::TJsonValue benchReport;

            for (const auto& result : Results_) {
                NJson::TJsonValue{}.Swap(benchReport);
                benchReport["name"] = result.TestName;
                benchReport["samples"] = result.Samples;
                benchReport["run_time"] = result.RunTime;
                benchReport["iterations"] = result.Iterations;

                if (result.CyclesPerIteration) {
                    benchReport["per_iteration_cycles"] = *result.CyclesPerIteration;
                }

                if (result.SecondsPerIteration) {
                    benchReport["per_iteration_secons"] = *result.SecondsPerIteration;
                }

                bench.AppendValue(benchReport);
            }

            OutputStream << report << Endl;
        }

    private:
        IOutputStream& OutputStream;
        TAdaptiveLock ResultsLock_;
        TVector<TResult> Results_;
    };

    class TOrderedReporter: public IReporter {
    public:
        TOrderedReporter(THolder<IReporter> slave)
            : Slave_(std::move(slave))
        {
        }

        void Report(TResult&& result) override {
            with_lock (ResultsLock_) {
                OrderedResultQueue_.emplace(result.TestId, std::move(result));
                while (!OrderedResultQueue_.empty() && OrderedResultQueue_.begin()->first <= ExpectedTestId_) {
                    Slave_->Report(std::move(OrderedResultQueue_.begin()->second));
                    OrderedResultQueue_.erase(OrderedResultQueue_.begin());
                    ++ExpectedTestId_;
                }
            }
        }

        void Finish() override {
            for (auto& it : OrderedResultQueue_) {
                Slave_->Report(std::move(it.second));
            }
            OrderedResultQueue_.clear();
            Slave_->Finish();
        }

    private:
        THolder<IReporter> Slave_;
        size_t ExpectedTestId_ = 0;
        TMap<size_t, TResult> OrderedResultQueue_;
        TAdaptiveLock ResultsLock_;
    };

    THolder<IReporter> MakeReporter(const EOutFormat type, IOutputStream& outputStream) {
        switch (type) {
            case F_CONSOLE:
                return MakeHolder<TConsoleReporter>(outputStream);

            case F_CSV:
                return MakeHolder<TCSVReporter>(outputStream);

            case F_JSON:
                return MakeHolder<TJSONReporter>(outputStream);

            default:
                break;
        }

        return MakeHolder<TConsoleReporter>(outputStream); // make compiler happy
    }

    THolder<IReporter> MakeOrderedReporter(const EOutFormat type, IOutputStream& outputStream) {
        return MakeHolder<TOrderedReporter>(MakeReporter(type, outputStream));
    }

    void EnumerateTests(TVector<ITestRunner*>& tests) {
        for (size_t id : xrange(tests.size())) {
            tests[id]->SequentialId = id;
        }
    }
}

template <>
EOutFormat FromStringImpl<EOutFormat>(const char* data, size_t len) {
    const auto s = TStringBuf{data, len};

    if (TStringBuf("console") == s) {
        return F_CONSOLE;
    } else if (TStringBuf("csv") == s) {
        return F_CSV;
    } else if (TStringBuf("json") == s) {
        return F_JSON;
    }

    ythrow TFromStringException{} << "failed to convert '" << s << '\'';
}

template <>
void Out<TResult>(IOutputStream& out, const TResult& r) {
    out << "----------- " << LightRed() << r.TestName << Old() << " ---------------" << Endl
        << " samples:       " << White() << r.Samples << Old() << Endl
        << " iterations:    " << White() << r.Iterations << Old() << Endl
        << " iterations hr:    " << White() << HumanReadableSize(r.Iterations, SF_QUANTITY) << Old() << Endl
        << " run time:      " << White() << r.RunTime << Old() << Endl;

    if (r.CyclesPerIteration) {
        out << " per iteration: " << White() << TCycleTimer::FmtTime(*r.CyclesPerIteration) << Old() << Endl;
    }

    if (r.SecondsPerIteration) {
        out << " per iteration: " << White() << DoFmtTime(*r.SecondsPerIteration) << Old() << Endl;
    }
}

NCpu::TRegistar::TRegistar(const char* name, TUserFunc func) {
    static_assert(sizeof(TCpuBenchmark) + alignof(TCpuBenchmark) < sizeof(Buf), "fix Buf size");

    new (AlignUp(Buf, alignof(TCpuBenchmark))) TCpuBenchmark(name, func);
}

namespace {
    struct TProgOpts {
        TProgOpts(int argc, char** argv) {
            TOpts opts = TOpts::Default();

            opts.AddHelpOption();

            opts.AddLongOption('b', "budget")
                .StoreResult(&TimeBudget)
                .RequiredArgument("SEC")
                .Optional()
                .Help("overall time budget");

            opts.AddLongOption('l', "list")
                .NoArgument()
                .StoreValue(&ListTests, true)
                .Help("list all tests");

            opts.AddLongOption('t', "threads")
                .StoreResult(&Threads)
                .OptionalValue(ToString((NSystemInfo::CachedNumberOfCpus() + 1) / 2), "JOBS")
                .DefaultValue("1")
                .Help("run benchmarks in parallel");

            opts.AddLongOption('f', "format")
                .AddLongName("benchmark_format")
                .StoreResult(&OutFormat)
                .RequiredArgument("FORMAT")
                .DefaultValue("console")
                .Help("output format (console|csv|json)");

            opts.AddLongOption('r', "report_path")
                .StoreResult(&ReportPath)
                .Optional()
                .Help("path to save report");

            opts.SetFreeArgDefaultTitle("REGEXP", "RE2 regular expression to filter tests");

            const TOptsParseResult parseResult{&opts, argc, argv};

            for (const auto& regexp : parseResult.GetFreeArgs()) {
                Filters.push_back(MakeHolder<RE2>(regexp.data(), RE2::Quiet));
                Y_ENSURE(Filters.back()->ok(), "incorrect RE2 expression '" << regexp << "'");
            }
        }

        bool MatchFilters(const TStringBuf& name) const {
            if (!Filters) {
                return true;
            }

            for (auto&& re : Filters) {
                if (RE2::FullMatchN({name.data(), name.size()}, *re, nullptr, 0)) {
                    return true;
                }
            }

            return false;
        }

        bool ListTests = false;
        double TimeBudget = -1.0;
        TVector<THolder<RE2>> Filters;
        size_t Threads = 0;
        EOutFormat OutFormat;
        std::string ReportPath;
    };
}

int NBench::Main(int argc, char** argv) {
    const TProgOpts opts(argc, argv);

    TVector<ITestRunner*> tests;

    for (auto&& it : Tests()) {
        if (opts.MatchFilters(it.Name())) {
            tests.push_back(&it);
        }
    }
    EnumerateTests(tests);

    if (opts.ListTests) {
        for (const auto* const it : tests) {
            Cout << it->Name() << Endl;
        }

        return 0;
    }

    if (!tests) {
        return 0;
    }

    double timeBudget = opts.TimeBudget;

    if (timeBudget < 0) {
        timeBudget = 5.0 * tests.size();
    }


    THolder<IOutputStream> outputHolder;
    IOutputStream* outputStream = &Cout;

    if (opts.ReportPath != "") {
        TString filePath(opts.ReportPath);
        outputHolder.Reset(outputStream = new TFileOutput(filePath));
    }

    const TOptions testOpts = {timeBudget / tests.size()};
    const auto reporter = MakeOrderedReporter(opts.OutFormat, *outputStream);

    std::function<void(ITestRunner**)> func = [&](ITestRunner** it) {
        auto&& res = (*it)->Run(testOpts);

        reporter->Report(std::move(res));
    };

    if (opts.Threads > 1) {
        NYmp::SetThreadCount(opts.Threads);
        NYmp::ParallelForStaticChunk(tests.data(), tests.data() + tests.size(), 1, func);
    } else {
        for (auto it : tests) {
            func(&it);
        }
    }

    reporter->Finish();

    return 0;
}
