#include <library/cpp/threading/skip_list/skiplist.h>

#include <library/cpp/getopt/small/last_getopt.h>

#include <library/cpp/charset/ci_string.h>
#include <util/datetime/base.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <functional>
#include <util/memory/pool.h>
#include <util/random/random.h>
#include <util/string/join.h>
#include <util/system/mutex.h>
#include <util/system/thread.h>

namespace {
    using namespace NThreading;

    ////////////////////////////////////////////////////////////////////////////////

    IOutputStream& LogInfo() {
        return Cerr << TInstant::Now() << " INFO: ";
    }

    IOutputStream& LogError() {
        return Cerr << TInstant::Now() << " ERROR: ";
    }

    ////////////////////////////////////////////////////////////////////////////////

    struct TListItem {
        TStringBuf Key;
        TStringBuf Value;

        TListItem(const TStringBuf& key, const TStringBuf& value)
            : Key(key)
            , Value(value)
        {
        }

        int Compare(const TListItem& other) const {
            return Key.compare(other.Key);
        }
    };

    using TListType = TSkipList<TListItem>;

    ////////////////////////////////////////////////////////////////////////////////

    class TRandomData {
    private:
        TVector<char> Buffer;

    public:
        TRandomData()
            : Buffer(1024 * 1024)
        {
            for (size_t i = 0; i < Buffer.size(); ++i) {
                Buffer[i] = RandomNumber<char>();
            }
        }

        TStringBuf GetString(size_t len) const {
            size_t start = RandomNumber(Buffer.size() - len);
            return TStringBuf(&Buffer[start], len);
        }

        TStringBuf GetString(size_t min, size_t max) const {
            return GetString(min + RandomNumber(max - min));
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    class TWorkerThread: public ISimpleThread {
    private:
        std::function<void()> Func;
        TDuration Time;

    public:
        TWorkerThread(std::function<void()> func)
            : Func(func)
        {
        }

        TDuration GetTime() const {
            return Time;
        }

    private:
        void* ThreadProc() noexcept override {
            TInstant started = TInstant::Now();
            Func();
            Time = TInstant::Now() - started;
            return nullptr;
        }
    };

    inline TAutoPtr<TWorkerThread> StartThread(std::function<void()> func) {
        TAutoPtr<TWorkerThread> thread = new TWorkerThread(func);
        thread->Start();
        return thread;
    }

    ////////////////////////////////////////////////////////////////////////////////

    typedef std::function<void()> TTestFunc;

    struct TTest {
        TString Name;
        TTestFunc Func;

        TTest() {
        }

        TTest(const TString& name, const TTestFunc& func)
            : Name(name)
            , Func(func)
        {
        }
    };

    ////////////////////////////////////////////////////////////////////////////////

    class TTestSuite {
    private:
        size_t Iterations = 1000000;
        size_t KeyLen = 10;
        size_t ValueLen = 100;
        size_t NumReaders = 4;
        size_t NumWriters = 1;
        size_t BatchSize = 20;

        TMemoryPool MemoryPool;
        TListType List;
        TMutex Mutex;
        TRandomData Random;

        TMap<TCiString, TTest> AllTests;
        TVector<TTest> Tests;

    public:
        TTestSuite()
            : MemoryPool(64 * 1024)
            , List(MemoryPool)
        {
        }

        bool Init(int argc, const char* argv[]) {
            TVector<TString> tests;
            try {
                NLastGetopt::TOpts opts;
                opts.AddHelpOption();

#define OPTION(opt, x)             \
    opts.AddLongOption(opt, #x)    \
        .Optional()                \
        .DefaultValue(ToString(x)) \
        .StoreResult(&x) // end of OPTION

                OPTION('i', Iterations);
                OPTION('k', KeyLen);
                OPTION('v', ValueLen);
                OPTION('r', NumReaders);
                OPTION('w', NumWriters);
                OPTION('b', BatchSize);

#undef OPTION

                NLastGetopt::TOptsParseResultException optsRes(&opts, argc, argv);
                for (const auto& opt : opts.Opts_) {
                    const NLastGetopt::TOptParseResult* r = optsRes.FindOptParseResult(opt.Get(), true);
                    if (r) {
                        LogInfo() << "[-" << opt->GetChar() << "] " << opt->GetName() << ": " << r->Back() << Endl;
                    }
                }
                tests = optsRes.GetFreeArgs();
            } catch (...) {
                LogError() << CurrentExceptionMessage() << Endl;
                return false;
            }

#define TEST(type) \
    AddTest(#type, std::bind(&TTestSuite::Y_CAT(TEST_, type), this)) // end of TEST

            TEST(Clear);
            TEST(InsertRandom);
            TEST(InsertSequential);
            TEST(InsertSequentialSimple);
            TEST(LookupRandom);
            TEST(Concurrent);

#undef TEST

            if (tests.empty()) {
                LogError() << "no tests specified, choose from: " << PrintTests() << Endl;
                return false;
            }

            for (size_t i = 0; i < tests.size(); ++i) {
                if (!AllTests.contains(tests[i])) {
                    LogError() << "unknown test name: " << tests[i] << Endl;
                    return false;
                }
                Tests.push_back(AllTests[tests[i]]);
            }

            return true;
        }

        void Run() {
#if !defined(NDEBUG)
            LogInfo() << "*** DEBUG build! ***" << Endl;
#endif

            for (const TTest& test : Tests) {
                LogInfo() << "Starting test " << test.Name << Endl;

                TInstant started = TInstant::Now();
                try {
                    test.Func();
                } catch (...) {
                    LogError() << "test " << test.Name
                               << " failed: " << CurrentExceptionMessage()
                               << Endl;
                }

                LogInfo() << "List size = " << List.GetSize() << Endl;

                TDuration duration = TInstant::Now() - started;
                LogInfo() << "test " << test.Name
                          << " duration: " << duration
                          << " (" << (double)duration.MicroSeconds() / (Iterations * NumWriters) << "us per iteration)"
                          << Endl;
                LogInfo() << "Finished test " << test.Name << Endl;
            }
        }

    private:
        void AddTest(const char* name, TTestFunc func) {
            AllTests[name] = TTest(name, func);
        }

        TString PrintTests() const {
            TVector<TString> names;
            for (const auto& it : AllTests) {
                names.push_back(it.first);
            }
            return JoinSeq(", ", names);
        }

        void TEST_Clear() {
            List.Clear();
        }

        void TEST_InsertRandom() {
            for (size_t i = 0; i < Iterations; ++i) {
                List.Insert(TListItem(Random.GetString(KeyLen), Random.GetString(ValueLen)));
            }
        }

        void TEST_InsertSequential() {
            TString key;
            for (size_t i = 0; i < Iterations;) {
                key.assign(Random.GetString(KeyLen));
                size_t batch = BatchSize / 2 + RandomNumber(BatchSize);
                for (size_t j = 0; j < batch; ++j, ++i) {
                    key.resize(KeyLen - 1);
                    key.append((char)j);
                    List.Insert(TListItem(key, Random.GetString(ValueLen)));
                }
            }
        }

        void TEST_InsertSequentialSimple() {
            for (size_t i = 0; i < Iterations; ++i) {
                List.Insert(TListItem(Random.GetString(KeyLen), Random.GetString(ValueLen)));
            }
        }

        void TEST_LookupRandom() {
            for (size_t i = 0; i < Iterations; ++i) {
                List.SeekTo(TListItem(Random.GetString(KeyLen), TStringBuf()));
            }
        }

        void TEST_Concurrent() {
            LogInfo() << "starting producers..." << Endl;

            TVector<TAutoPtr<TWorkerThread>> producers(NumWriters);
            for (size_t i1 = 0; i1 < producers.size(); ++i1) {
                producers[i1] = StartThread([&] {
                    TInstant started = TInstant::Now();
                    for (size_t i2 = 0; i2 < Iterations; ++i2) {
                        {
                            TGuard<TMutex> guard(Mutex);
                            List.Insert(TListItem(Random.GetString(KeyLen), Random.GetString(ValueLen)));
                        }
                    }
                    TDuration duration = TInstant::Now() - started;
                    LogInfo()
                        << "Average time for producer = "
                        << (double)duration.MicroSeconds() / Iterations << "us per iteration"
                        << Endl;
                });
            }

            LogInfo() << "starting consumers..." << Endl;

            TVector<TAutoPtr<TWorkerThread>> consumers(NumReaders);
            for (size_t i1 = 0; i1 < consumers.size(); ++i1) {
                consumers[i1] = StartThread([&] {
                    TInstant started = TInstant::Now();
                    for (size_t i2 = 0; i2 < Iterations; ++i2) {
                        List.SeekTo(TListItem(Random.GetString(KeyLen), TStringBuf()));
                    }
                    TDuration duration = TInstant::Now() - started;
                    LogInfo()
                        << "Average time for consumer = "
                        << (double)duration.MicroSeconds() / Iterations << "us per iteration"
                        << Endl;
                });
            }

            LogInfo() << "wait for producers..." << Endl;

            TDuration producerTime;
            for (size_t i = 0; i < producers.size(); ++i) {
                producers[i]->Join();
                producerTime += producers[i]->GetTime();
            }

            LogInfo() << "wait for consumers..." << Endl;

            TDuration consumerTime;
            for (size_t i = 0; i < consumers.size(); ++i) {
                consumers[i]->Join();
                consumerTime += consumers[i]->GetTime();
            }

            LogInfo() << "average producer time: "
                      << producerTime.SecondsFloat() / producers.size() << " seconds"
                      << Endl;

            LogInfo() << "average consumer time: "
                      << consumerTime.SecondsFloat() / consumers.size() << " seconds"
                      << Endl;
        }
    };

}

////////////////////////////////////////////////////////////////////////////////

int main(int argc, const char* argv[]) {
    TTestSuite suite;
    if (!suite.Init(argc, argv)) {
        return -1;
    }
    suite.Run();
    return 0;
}
