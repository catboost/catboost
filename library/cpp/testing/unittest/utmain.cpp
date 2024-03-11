#include "junit.h"
#include "plugin.h"
#include "registar.h"
#include "utmain.h"

#include <library/cpp/colorizer/colors.h>

#include <library/cpp/json/writer/json.h>
#include <library/cpp/json/writer/json_value.h>
#include <library/cpp/testing/common/env.h>
#include <library/cpp/testing/hook/hook.h>

#include <util/datetime/base.h>

#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/scope.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

#include <util/network/init.h>

#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/string/join.h>
#include <util/string/util.h>

#include <util/system/defaults.h>
#include <util/system/env.h>
#include <util/system/execpath.h>
#include <util/system/valgrind.h>
#include <util/system/shellcommand.h>

#include <filesystem>

#if defined(_win_)
    #include <fcntl.h>
    #include <io.h>
    #include <windows.h>
    #include <crtdbg.h>
#endif

#if defined(_unix_)
    #include <unistd.h>
#endif

#ifdef WITH_VALGRIND
    #define NOTE_IN_VALGRIND(test) VALGRIND_PRINTF("%s::%s", test->unit->name.data(), test->name)
#else
    #define NOTE_IN_VALGRIND(test)
#endif

const size_t MAX_COMMENT_MESSAGE_LENGTH = 1024 * 1024; // 1 MB

using namespace NUnitTest;

class TNullTraceWriterProcessor: public ITestSuiteProcessor {
};

class TMultiTraceProcessor: public ITestSuiteProcessor {
public:
    TMultiTraceProcessor(std::vector<std::shared_ptr<ITestSuiteProcessor>>&& processors)
        : Processors(std::move(processors))
    {
    }

    void SetForkTestsParams(bool forkTests, bool isForked) override {
        ITestSuiteProcessor::SetForkTestsParams(forkTests, isForked);
        for (const auto& proc : Processors) {
            proc->SetForkTestsParams(forkTests, isForked);
        }
    }

private:
    void OnStart() override {
        for (const auto& proc : Processors) {
            proc->Start();
        }
    }

    void OnEnd() override {
        for (const auto& proc : Processors) {
            proc->End();
        }
    }

    void OnUnitStart(const TUnit* unit) override {
        for (const auto& proc : Processors) {
            proc->UnitStart(*unit);
        }
    }

    void OnUnitStop(const TUnit* unit) override {
        for (const auto& proc : Processors) {
            proc->UnitStop(*unit);
        }
    }

    void OnError(const TError* error) override {
        for (const auto& proc : Processors) {
            proc->Error(*error);
        }
    }

    void OnFinish(const TFinish* finish) override {
        for (const auto& proc : Processors) {
            proc->Finish(*finish);
        }
    }

    void OnBeforeTest(const TTest* test) override {
        for (const auto& proc : Processors) {
            proc->BeforeTest(*test);
        }
    }

private:
    std::vector<std::shared_ptr<ITestSuiteProcessor>> Processors;
};

class TTraceWriterProcessor: public ITestSuiteProcessor {
public:
    inline TTraceWriterProcessor(const char* traceFilePath, EOpenMode mode)
        : PrevTime(TInstant::Now())
    {
        TraceFile = new TUnbufferedFileOutput(TFile(traceFilePath, mode | WrOnly | Seq));
    }

private:
    TAutoPtr<TUnbufferedFileOutput> TraceFile;
    TString TraceFilePath;
    TInstant PrevTime;
    TVector<TString> ErrorMessages;

    inline void Trace(const TString eventName, const NJson::TJsonValue eventValue) {
        NJsonWriter::TBuf json(NJsonWriter::HEM_UNSAFE);
        json.BeginObject();

        json.WriteKey("name").WriteString(eventName);
        json.WriteKey("value").WriteJsonValue(&eventValue);
        json.WriteKey("timestamp").WriteDouble(TInstant::Now().SecondsFloat(), PREC_NDIGITS, 14);

        json.EndObject();

        json.FlushTo(TraceFile.Get());
        *TraceFile << "\n";
    }

    inline void TraceSubtestFinished(const char* className, const char* subtestName, const char* status, const TString comment, const TTestContext* context) {
        const TInstant now = TInstant::Now();
        NJson::TJsonValue event;
        event.InsertValue("class", className);
        event.InsertValue("subtest", subtestName);
        event.InsertValue("status", status);
        event.InsertValue("comment", comment.data());
        event.InsertValue("time", (now - PrevTime).SecondsFloat());
        if (context) {
            for (const auto& metric : context->Metrics) {
                event["metrics"].InsertValue(metric.first, metric.second);
            }
        }
        Trace("subtest-finished", event);

        PrevTime = now;
        TString marker = Join("", "\n###subtest-finished:", className, "::", subtestName, "\n");
        Cout << marker;
        Cout.Flush();
        Cerr << comment;
        Cerr << marker;
        Cerr.Flush();
    }

    virtual TString BuildComment(const char* message, const char* backTrace) {
        return NUnitTest::GetFormatTag("bad") +
               TString(message).substr(0, MAX_COMMENT_MESSAGE_LENGTH) +
               NUnitTest::GetResetTag() +
               TString("\n") +
               NUnitTest::GetFormatTag("alt1") +
               TString(backTrace).substr(0, MAX_COMMENT_MESSAGE_LENGTH) +
               NUnitTest::GetResetTag();
    }

    void OnBeforeTest(const TTest* test) override {
        NJson::TJsonValue event;
        event.InsertValue("class", test->unit->name);
        event.InsertValue("subtest", test->name);
        Trace("subtest-started", event);
        TString marker = Join("", "\n###subtest-started:", test->unit->name, "::", test->name, "\n");
        Cout << marker;
        Cout.Flush();
        Cerr << marker;
        Cerr.Flush();
    }

    void OnUnitStart(const TUnit* unit) override {
        NJson::TJsonValue event;
        event.InsertValue("class", unit->name);
    }

    void OnUnitStop(const TUnit* unit) override {
        NJson::TJsonValue event;
        event.InsertValue("class", unit->name);
    }

    void OnError(const TError* descr) override {
        const TString comment = BuildComment(descr->msg, descr->BackTrace.data());
        ErrorMessages.push_back(comment);
    }

    void OnFinish(const TFinish* descr) override {
        if (descr->Success) {
            TraceSubtestFinished(descr->test->unit->name.data(), descr->test->name, "good", "", descr->Context);
        } else {
            TStringBuilder msgs;
            for (const TString& m : ErrorMessages) {
                if (msgs) {
                    msgs << TStringBuf("\n");
                }
                msgs << m;
            }
            if (msgs) {
                msgs << TStringBuf("\n");
            }
            TraceSubtestFinished(descr->test->unit->name.data(), descr->test->name, "fail", msgs, descr->Context);
            ErrorMessages.clear();
        }
    }
};

class TColoredProcessor: public ITestSuiteProcessor, public NColorizer::TColors {
public:
    inline TColoredProcessor(const TString& appName)
        : PrintBeforeSuite_(true)
        , PrintBeforeTest_(true)
        , PrintAfterTest_(true)
        , PrintAfterSuite_(true)
        , PrintTimes_(false)
        , PrintSummary_(true)
        , PrevTime_(TInstant::Now())
        , ShowFails(true)
        , Start(0)
        , End(Max<size_t>())
        , AppName(appName)
        , Loop(false)
        , ForkExitedCorrectly(false)
        , TraceProcessor(new TNullTraceWriterProcessor())
    {
    }

    ~TColoredProcessor() override {
    }

    inline void Disable(const char* name) {
        size_t colon = TString(name).find("::");
        if (colon == TString::npos) {
            DisabledSuites_.insert(name);
        } else {
            TString suite = TString(name).substr(0, colon);
            DisabledTests_.insert(name);
        }
    }

    inline void Enable(const char* name) {
        size_t colon = TString(name).rfind("::");
        if (colon == TString::npos) {
            EnabledSuites_.insert(name);
            EnabledTests_.insert(TString() + name + "::*");
        } else {
            TString suite = TString(name).substr(0, colon);
            EnabledSuites_.insert(suite);
            EnabledSuites_.insert(name);
            EnabledTests_.insert(name);
            EnabledTests_.insert(TString() + name + "::*");
        }
    }

    inline void FilterFromFile(TString filename) {
        TString filterLine;

        TFileInput filtersStream(filename);

        while (filtersStream.ReadLine(filterLine)) {
            if (filterLine.StartsWith("-")) {
                Disable(filterLine.c_str() + 1);
            } else if(filterLine.StartsWith("+")) {
                Enable(filterLine.c_str() + 1);
            }
        }
    }

    inline void SetPrintBeforeSuite(bool print) {
        PrintBeforeSuite_ = print;
    }

    inline void SetPrintAfterSuite(bool print) {
        PrintAfterSuite_ = print;
    }

    inline void SetPrintBeforeTest(bool print) {
        PrintBeforeTest_ = print;
    }

    inline void SetPrintAfterTest(bool print) {
        PrintAfterTest_ = print;
    }

    inline void SetPrintTimes(bool print) {
        PrintTimes_ = print;
    }

    inline void SetPrintSummary(bool print) {
        PrintSummary_ = print;
    }

    inline bool GetPrintSummary() {
        return PrintSummary_;
    }

    inline void SetShowFails(bool show) {
        ShowFails = show;
    }

    inline void BeQuiet() {
        SetPrintTimes(false);
        SetPrintBeforeSuite(false);
        SetPrintAfterSuite(false);
        SetPrintBeforeTest(false);
        SetPrintAfterTest(false);
        SetPrintSummary(false);
    }

    inline void SetStart(size_t val) {
        Start = val;
    }

    inline void SetEnd(size_t val) {
        End = val;
    }

    inline void SetForkTestsParams(bool forkTests, bool isForked) override {
        ITestSuiteProcessor::SetForkTestsParams(forkTests, isForked);
        TraceProcessor->SetForkTestsParams(forkTests, isForked);

        SetIsTTY(GetIsForked() || CalcIsTTY(stderr));
    }

    inline void SetLoop(bool loop) {
        Loop = loop;
    }

    inline bool IsLoop() const {
        return Loop;
    }

    inline void SetTraceProcessor(std::shared_ptr<ITestSuiteProcessor> traceProcessor) {
        TraceProcessor = std::move(traceProcessor);
    }

private:
    void OnUnitStart(const TUnit* unit) override {
        TraceProcessor->UnitStart(*unit);
        if (GetIsForked()) {
            return;
        }
        if (PrintBeforeSuite_ || PrintBeforeTest_) {
            fprintf(stderr, "%s<-----%s %s\n", LightBlueColor().data(), OldColor().data(), unit->name.data());
        }
    }

    void OnUnitStop(const TUnit* unit) override {
        TraceProcessor->UnitStop(*unit);
        if (GetIsForked()) {
            return;
        }
        if (!PrintAfterSuite_) {
            return;
        }

        fprintf(stderr, "%s----->%s %s -> ok: %s%u%s",
                LightBlueColor().data(), OldColor().data(), unit->name.data(),
                LightGreenColor().data(), GoodTestsInCurrentUnit(), OldColor().data());
        if (FailTestsInCurrentUnit()) {
            fprintf(stderr, ", err: %s%u%s",
                    LightRedColor().data(), FailTestsInCurrentUnit(), OldColor().data());
        }
        fprintf(stderr, "\n");
    }

    void OnBeforeTest(const TTest* test) override {
        if (!GetIsForked() && PrintBeforeTest_) {
            fprintf(stderr, "[%sexec%s] %s::%s...\n", LightBlueColor().data(), OldColor().data(), test->unit->name.data(), test->name);
        }
        TraceProcessor->BeforeTest(*test);
    }

    void OnError(const TError* descr) override {
        TraceProcessor->Error(*descr);
        if (!GetIsForked() && ForkExitedCorrectly) {
            return;
        }
        if (!PrintAfterTest_) {
            return;
        }

        const TString err = Sprintf("[%sFAIL%s] %s::%s -> %s%s%s\n%s%s%s", LightRedColor().data(), OldColor().data(),
                                    descr->test->unit->name.data(),
                                    descr->test->name,
                                    LightRedColor().data(), descr->msg, OldColor().data(), LightCyanColor().data(), descr->BackTrace.data(), OldColor().data());
        const TDuration test_duration = SaveTestDuration();
        if (ShowFails) {
            if (PrintTimes_) {
                Fails.push_back(Sprintf("%s %s", test_duration.ToString().data(), err.data()));
            } else {
                Fails.push_back(err);
            }
        }
        fprintf(stderr, "%s", err.data());
        NOTE_IN_VALGRIND(descr->test);
        PrintTimes(test_duration);
        if (GetIsForked()) {
            fprintf(stderr, "%s", ForkCorrectExitMsg);
        }
    }

    void OnFinish(const TFinish* descr) override {
        TraceProcessor->Finish(*descr);
        if (!GetIsForked() && ForkExitedCorrectly) {
            return;
        }
        if (!PrintAfterTest_) {
            return;
        }

        if (descr->Success) {
            fprintf(stderr, "[%sgood%s] %s::%s\n", LightGreenColor().data(), OldColor().data(),
                    descr->test->unit->name.data(),
                    descr->test->name);
            NOTE_IN_VALGRIND(descr->test);
            PrintTimes(SaveTestDuration());
            if (GetIsForked()) {
                fprintf(stderr, "%s", ForkCorrectExitMsg);
            }
        }
    }

    inline TDuration SaveTestDuration() {
        const TInstant now = TInstant::Now();
        TDuration d = now - PrevTime_;
        PrevTime_ = now;
        return d;
    }

    inline void PrintTimes(TDuration d) {
        if (!PrintTimes_) {
            return;
        }

        Cerr << d << "\n";
    }

    void OnEnd() override {
        TraceProcessor->End();
        if (GetIsForked()) {
            return;
        }

        if (!PrintSummary_) {
            return;
        }

        fprintf(stderr, "[%sDONE%s] ok: %s%u%s",
                YellowColor().data(), OldColor().data(),
                LightGreenColor().data(), GoodTests(), OldColor().data());
        if (FailTests())
            fprintf(stderr, ", err: %s%u%s",
                    LightRedColor().data(), FailTests(), OldColor().data());
        fprintf(stderr, "\n");

        if (ShowFails) {
            for (size_t i = 0; i < Fails.size(); ++i) {
                printf("%s", Fails[i].data());
            }
        }
    }

    bool CheckAccess(TString name, size_t num) override {
        if (num < Start) {
            return false;
        }

        if (num >= End) {
            return false;
        }

        if (DisabledSuites_.find(name.data()) != DisabledSuites_.end()) {
            return false;
        }

        if (EnabledSuites_.empty()) {
            return true;
        }

        return EnabledSuites_.find(name.data()) != EnabledSuites_.end();
    }

    bool CheckAccessTest(TString suite, const char* test) override {
        TString name = suite + "::" + test;
        if (DisabledTests_.find(name) != DisabledTests_.end()) {
            return false;
        }

        if (EnabledTests_.empty()) {
            return true;
        }

        if (EnabledTests_.find(TString() + suite + "::*") != EnabledTests_.end()) {
            return true;
        }

        return EnabledTests_.find(name) != EnabledTests_.end();
    }

    void Run(std::function<void()> f, const TString& suite, const char* name, const bool forceFork) override {
        if (!(GetForkTests() || forceFork) || GetIsForked()) {
            return f();
        }

        TList<TString> args(1, "--is-forked-internal");
        args.push_back(Sprintf("+%s::%s", suite.data(), name));

        // stdin is ignored - unittest should not need them...
        TShellCommandOptions options;
        options
            .SetUseShell(false)
            .SetCloseAllFdsOnExec(true)
            .SetAsync(false)
            .SetLatency(1);

        TShellCommand cmd(AppName, args, options);
        cmd.Run();

        const TString& err = cmd.GetError();
        const size_t msgIndex = err.find(ForkCorrectExitMsg);

        // everything is printed by parent process except test's result output ("good" or "fail")
        // which is printed by child. If there was no output - parent process prints default message.
        ForkExitedCorrectly = msgIndex != TString::npos;

        // TODO: stderr output is always printed after stdout
        Cout.Write(cmd.GetOutput());
        Cerr.Write(err.c_str(), Min(msgIndex, err.size()));

        // do not use default case, so gcc will warn if new element in enum will be added
        switch (cmd.GetStatus()) {
            case TShellCommand::SHELL_FINISHED: {
                // test could fail with zero status if it calls exit(0) in the middle.
                if (ForkExitedCorrectly)
                    break;
                [[fallthrough]];
            }
            case TShellCommand::SHELL_ERROR: {
                ythrow yexception() << "Forked test failed";
            }

            case TShellCommand::SHELL_NONE: {
                ythrow yexception() << "Forked test finished with unknown status";
            }
            case TShellCommand::SHELL_RUNNING: {
                Y_ABORT_UNLESS(false, "This can't happen, we used sync mode, it's a bug!");
            }
            case TShellCommand::SHELL_INTERNAL_ERROR: {
                ythrow yexception() << "Forked test failed with internal error: " << cmd.GetInternalError();
            }
        }
    }

private:
    bool PrintBeforeSuite_;
    bool PrintBeforeTest_;
    bool PrintAfterTest_;
    bool PrintAfterSuite_;
    bool PrintTimes_;
    bool PrintSummary_;
    THashSet<TString> DisabledSuites_;
    THashSet<TString> EnabledSuites_;
    THashSet<TString> DisabledTests_;
    THashSet<TString> EnabledTests_;
    TInstant PrevTime_;
    bool ShowFails;
    TVector<TString> Fails;
    size_t Start;
    size_t End;
    TString AppName;
    bool Loop;
    static const char* const ForkCorrectExitMsg;
    bool ForkExitedCorrectly;
    std::shared_ptr<ITestSuiteProcessor> TraceProcessor;
};

const char* const TColoredProcessor::ForkCorrectExitMsg = "--END--";

class TEnumeratingProcessor: public ITestSuiteProcessor {
public:
    TEnumeratingProcessor(bool verbose, IOutputStream& stream) noexcept
        : Verbose_(verbose)
        , Stream_(stream)
    {
    }

    ~TEnumeratingProcessor() override {
    }

    bool CheckAccess(TString name, size_t /*num*/) override {
        if (Verbose_) {
            return true;
        } else {
            Stream_ << name << "\n";
            return false;
        }
    }

    bool CheckAccessTest(TString suite, const char* name) override {
        Stream_ << suite << "::" << name << "\n";
        return false;
    }

private:
    bool Verbose_;
    IOutputStream& Stream_;
};

#ifdef _win_
class TWinEnvironment {
public:
    TWinEnvironment()
        : OutputCP(GetConsoleOutputCP())
    {
        setmode(fileno(stdout), _O_BINARY);
        SetConsoleOutputCP(CP_UTF8);

        _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);

        if (!IsDebuggerPresent()) {
            _CrtSetReportMode(_CRT_ASSERT, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
            _CrtSetReportFile(_CRT_ASSERT, _CRTDBG_FILE_STDERR);
            _CrtSetReportMode(_CRT_ERROR, _CRTDBG_MODE_FILE | _CRTDBG_MODE_DEBUG);
            _CrtSetReportFile(_CRT_ERROR, _CRTDBG_FILE_STDERR);
        }
    }
    ~TWinEnvironment() {
        if (!IsDebuggerPresent()) {
            _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
            _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
        }

        SetConsoleOutputCP(OutputCP); // restore original output CP at program exit
    }

private:
    UINT OutputCP; // original codepage
};
static const TWinEnvironment Instance;
#endif // _win_

static int DoList(bool verbose, IOutputStream& stream) {
    TEnumeratingProcessor eproc(verbose, stream);
    TTestFactory::Instance().SetProcessor(&eproc);
    TTestFactory::Instance().Execute();
    return 0;
}

static int DoUsage(const char* progname) {
    Cout << "Usage: " << progname << " [options] [[+|-]test]...\n\n"
         << "Options:\n"
         << "  -h, --help            print this help message\n"
         << "  -l, --list            print a list of available tests\n"
         << "  -A --list-verbose        print a list of available subtests\n"
         << "  --print-before-test   print each test name before running it\n"
         << "  --print-before-suite  print each test suite name before running it\n"
         << "  --show-fails          print a list of all failed tests at the end\n"
         << "  --dont-show-fails     do not print a list of all failed tests at the end\n"
         << "  --print-times         print wall clock duration of each test\n"
         << "  --fork-tests          run each test in a separate process\n"
         << "  --trace-path          path to the trace file to be generated\n"
         << "  --trace-path-append   path to the trace file to be appended\n"
         << "  --filter-file         path to the test filters ([+|-]test) file (" << Y_UNITTEST_TEST_FILTER_FILE_OPTION << ")\n";
    return 0;
}

#if defined(_linux_) && defined(CLANG_COVERAGE)
extern "C" int __llvm_profile_write_file(void);

static void GracefulShutdownHandler(int) {
    try {
        __llvm_profile_write_file();
    } catch (...) {
    }
    abort();
}
#endif

int NUnitTest::RunMain(int argc, char** argv) {
#if defined(_linux_) && defined(CLANG_COVERAGE)
    {
        struct sigaction sa;
        memset(&sa, 0, sizeof(sa));
        sa.sa_handler = GracefulShutdownHandler;
        sa.sa_flags = SA_SIGINFO | SA_RESTART;
        Y_ABORT_UNLESS(!sigaction(SIGUSR2, &sa, nullptr));
    }
#endif
    NTesting::THook::CallBeforeInit();
    InitNetworkSubSystem();
    Singleton<::NPrivate::TTestEnv>();

    try {
        GetExecPath();
    } catch (...) {
    }

#ifndef UT_SKIP_EXCEPTIONS
    try {
#endif
        NTesting::THook::CallBeforeRun();
        Y_DEFER {
            NTesting::THook::CallAfterRun();
        };

        NPlugin::OnStartMain(argc, argv);
        Y_DEFER {
            NPlugin::OnStopMain(argc, argv);
        };

        TColoredProcessor processor(GetExecPath());
        IOutputStream* listStream = &Cout;
        THolder<IOutputStream> listFile;

        enum EListType {
            DONT_LIST,
            LIST,
            LIST_VERBOSE
        };
        EListType listTests = DONT_LIST;

        bool hasJUnitProcessor = false;
        bool forkTests = false;
        bool isForked = false;
        std::vector<std::shared_ptr<ITestSuiteProcessor>> traceProcessors;


        // load filters from environment variable
        TString filterFn = GetEnv(Y_UNITTEST_TEST_FILTER_FILE_OPTION);
        if (!filterFn.empty()) {
            processor.FilterFromFile(filterFn);
        }

        auto processJunitOption = [&](const TStringBuf& v) {
            if (!hasJUnitProcessor) {
                hasJUnitProcessor = true;
                bool xmlFormat = false;
                constexpr TStringBuf xmlPrefix = "xml:";
                constexpr TStringBuf jsonPrefix = "json:";
                if ((xmlFormat = v.StartsWith(xmlPrefix)) || v.StartsWith(jsonPrefix)) {
                    TStringBuf fileName = v;
                    const TStringBuf prefix = xmlFormat ? xmlPrefix : jsonPrefix;
                    fileName = fileName.SubString(prefix.size(), TStringBuf::npos);
                    const TJUnitProcessor::EOutputFormat format = xmlFormat ? TJUnitProcessor::EOutputFormat::Xml : TJUnitProcessor::EOutputFormat::Json;
                    NUnitTest::ShouldColorizeDiff = false;
                    traceProcessors.push_back(std::make_shared<TJUnitProcessor>(TString(fileName),
                                                                                std::filesystem::path(argv[0]).stem().string(),
                                                                                format));
                }
            }
        };

        for (size_t i = 1; i < (size_t)argc; ++i) {
            const char* name = argv[i];

            if (name && *name) {
                if (strcmp(name, "--help") == 0 || strcmp(name, "-h") == 0) {
                    return DoUsage(argv[0]);
                } else if (strcmp(name, "--list") == 0 || strcmp(name, "-l") == 0) {
                    listTests = LIST;
                } else if (strcmp(name, "--list-verbose") == 0 || strcmp(name, "-A") == 0) {
                    listTests = LIST_VERBOSE;
                } else if (strcmp(name, "--print-before-suite=false") == 0) {
                    processor.SetPrintBeforeSuite(false);
                } else if (strcmp(name, "--print-before-test=false") == 0) {
                    processor.SetPrintBeforeTest(false);
                } else if (strcmp(name, "--print-before-suite") == 0) {
                    processor.SetPrintBeforeSuite(true);
                } else if (strcmp(name, "--print-before-test") == 0) {
                    processor.SetPrintBeforeTest(true);
                } else if (strcmp(name, "--show-fails") == 0) {
                    processor.SetShowFails(true);
                } else if (strcmp(name, "--dont-show-fails") == 0) {
                    processor.SetShowFails(false);
                } else if (strcmp(name, "--print-times") == 0) {
                    processor.SetPrintTimes(true);
                } else if (strcmp(name, "--from") == 0) {
                    ++i;
                    processor.SetStart(FromString<size_t>(argv[i]));
                } else if (strcmp(name, "--to") == 0) {
                    ++i;
                    processor.SetEnd(FromString<size_t>(argv[i]));
                } else if (strcmp(name, "--fork-tests") == 0) {
                    forkTests = true;
                } else if (strcmp(name, "--is-forked-internal") == 0) {
                    isForked = true;
                } else if (strcmp(name, "--loop") == 0) {
                    processor.SetLoop(true);
                } else if (strcmp(name, "--trace-path") == 0) {
                    ++i;
                    processor.BeQuiet();
                    NUnitTest::ShouldColorizeDiff = false;
                    traceProcessors.push_back(std::make_shared<TTraceWriterProcessor>(argv[i], CreateAlways));
                } else if (strcmp(name, "--trace-path-append") == 0) {
                    ++i;
                    processor.BeQuiet();
                    NUnitTest::ShouldColorizeDiff = false;
                    traceProcessors.push_back(std::make_shared<TTraceWriterProcessor>(argv[i], OpenAlways | ForAppend));
                } else if (strcmp(name, "--list-path") == 0) {
                    ++i;
                    listFile = MakeHolder<TFixedBufferFileOutput>(argv[i]);
                    listStream = listFile.Get();
                } else if (strcmp(name, "--test-param") == 0) {
                    ++i;
                    TString param(argv[i]);
                    size_t assign = param.find('=');
                    Singleton<::NPrivate::TTestEnv>()->AddTestParam(param.substr(0, assign), param.substr(assign + 1));
                } else if (strcmp(name, "--output") == 0) {
                    ++i;
                    Y_ENSURE((int)i < argc);
                    processJunitOption(argv[i]);
                } else if (strcmp(name, "--filter-file") == 0) {
                    ++i;
                    TString filename(argv[i]);
                    processor.FilterFromFile(filename);
                } else if (TString(name).StartsWith("--")) {
                    return DoUsage(argv[0]), 1;
                } else if (*name == '-') {
                    processor.Disable(name + 1);
                } else if (*name == '+') {
                    processor.Enable(name + 1);
                } else {
                    processor.Enable(name);
                }
            }
        }
        if (listTests != DONT_LIST) {
            return DoList(listTests == LIST_VERBOSE, *listStream);
        }

        if (!hasJUnitProcessor) {
            if (TString oo = GetEnv(Y_UNITTEST_OUTPUT_CMDLINE_OPTION)) {
                processJunitOption(oo);
            }
        }

        if (traceProcessors.size() > 1) {
            processor.SetTraceProcessor(std::make_shared<TMultiTraceProcessor>(std::move(traceProcessors)));
        } else if (traceProcessors.size() == 1) {
            processor.SetTraceProcessor(std::move(traceProcessors[0]));
        }

        processor.SetForkTestsParams(forkTests, isForked);

        TTestFactory::Instance().SetProcessor(&processor);

        unsigned ret;
        for (;;) {
            ret = TTestFactory::Instance().Execute();
            if (!processor.GetIsForked() && ret && processor.GetPrintSummary()) {
                Cerr << "SOME TESTS FAILED!!!!" << Endl;
            }

            if (0 != ret || !processor.IsLoop()) {
                break;
            }
        }
        return ret;
#ifndef UT_SKIP_EXCEPTIONS
    } catch (...) {
        Cerr << "caught exception in test suite(" << CurrentExceptionMessage() << ")" << Endl;
    }
#endif

    return 1;
}
