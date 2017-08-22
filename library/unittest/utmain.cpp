#include "plugin.h"
#include "registar.h"
#include "utmain.h"

#include <library/colorizer/colors.h>

#include <library/json/writer/json.h>
#include <library/json/writer/json_value.h>

#include <util/datetime/base.h>

#include <util/generic/hash.h>
#include <util/generic/hash_set.h>
#include <util/generic/string.h>
#include <util/generic/yexception.h>

#include <util/network/init.h>

#include <util/stream/file.h>
#include <util/stream/output.h>
#include <util/string/join.h>
#include <util/string/util.h>

#include <util/system/defaults.h>
#include <util/system/execpath.h>
#include <util/system/valgrind.h>
#include <util/system/shellcommand.h>

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
#define NOTE_IN_VALGRIND(test) VALGRIND_PRINTF("%s::%s", ~test->unit->name, test->name)
#else
#define NOTE_IN_VALGRIND(test)
#endif

const size_t MAX_COMMENT_MESSAGE_LENGTH = 1024 * 1024; // 1 MB

using namespace NUnitTest;

class TNullTraceWriterProcessor: public ITestSuiteProcessor {
};

class TTraceWriterProcessor: public ITestSuiteProcessor {
public:
    inline TTraceWriterProcessor(const char* traceFilePath, EOpenMode mode)
        : PrevTime(TInstant::Now())
    {
        TraceFile = new TFileOutput(TFile(traceFilePath, mode | WrOnly | Seq));
    }

private:
    TAutoPtr<TFileOutput> TraceFile;
    TString TraceFilePath;
    TInstant PrevTime;
    yvector<TString> ErrorMessages;

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
        event.InsertValue("comment", ~comment);
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
        Trace("test-started", event);
    }

    void OnUnitStop(const TUnit* unit) override {
        NJson::TJsonValue event;
        event.InsertValue("class", unit->name);
        Trace("test-finished", event);
    }

    void OnError(const TError* descr) override {
        const TString comment = BuildComment(descr->msg, ~descr->BackTrace);
        ErrorMessages.push_back(comment);
    }

    void OnFinish(const TFinish* descr) override {
        if (descr->Success) {
            TraceSubtestFinished(~descr->test->unit->name, descr->test->name, "good", "", descr->Context);
        } else {
            TStringBuilder msgs;
            for (const TString& m : ErrorMessages) {
                if (msgs) {
                    msgs << STRINGBUF("\n");
                }
                msgs << m;
            }
            if (msgs) {
                msgs << STRINGBUF("\n");
            }
            TraceSubtestFinished(~descr->test->unit->name, descr->test->name, "fail", msgs, descr->Context);
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
        , ShowFails(false)
        , Start(0)
        , End(Max<size_t>())
        , AppName(appName)
        , ForkTests(false)
        , IsForked(false)
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

    void SetContinueOnFail(bool val) {
        NUnitTest::ContinueOnFail = val;
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

    inline void SetForkTests(bool val) {
        ForkTests = val;
    }

    inline bool GetForkTests() const override {
        return ForkTests;
    }

    inline void SetIsForked(bool val) {
        IsForked = val;
        SetIsTTY(IsForked || CalcIsTTY(stderr));
    }

    inline bool GetIsForked() const override {
        return IsForked;
    }

    inline void SetLoop(bool loop) {
        Loop = loop;
    }

    inline bool IsLoop() const {
        return Loop;
    }

    inline void SetTraceProcessor(TAutoPtr<ITestSuiteProcessor> traceProcessor) {
        TraceProcessor = traceProcessor;
    }

private:
    void OnUnitStart(const TUnit* unit) override {
        TraceProcessor->UnitStart(*unit);
        if (IsForked) {
            return;
        }
        if (PrintBeforeSuite_ || PrintBeforeTest_) {
            fprintf(stderr, "%s<-----%s %s\n", ~LightBlueColor(), ~OldColor(), ~unit->name);
        }
    }

    void OnUnitStop(const TUnit* unit) override {
        TraceProcessor->UnitStop(*unit);
        if (IsForked) {
            return;
        }
        if (!PrintAfterSuite_) {
            return;
        }

        fprintf(stderr, "%s----->%s %s -> ok: %s%u%s",
                ~LightBlueColor(), ~OldColor(), ~unit->name,
                ~LightGreenColor(), GoodTestsInCurrentUnit(), ~OldColor());
        if (FailTestsInCurrentUnit()) {
            fprintf(stderr, ", err: %s%u%s",
                    ~LightRedColor(), FailTestsInCurrentUnit(), ~OldColor());
        }
        fprintf(stderr, "\n");
    }

    void OnBeforeTest(const TTest* test) override {
        TraceProcessor->BeforeTest(*test);
        if (IsForked) {
            return;
        }
        if (PrintBeforeTest_) {
            fprintf(stderr, "[%sexec%s] %s::%s...\n", ~LightBlueColor(), ~OldColor(), ~test->unit->name, test->name);
        }
    }

    void OnError(const TError* descr) override {
        TraceProcessor->Error(*descr);
        if (!IsForked && ForkExitedCorrectly) {
            return;
        }
        if (!PrintAfterTest_) {
            return;
        }

        const TString err = Sprintf("[%sFAIL%s] %s::%s -> %s%s%s\n%s%s%s", ~LightRedColor(), ~OldColor(),
                                    ~descr->test->unit->name,
                                    descr->test->name,
                                    ~LightRedColor(), descr->msg, ~OldColor(), ~LightCyanColor(), ~descr->BackTrace, ~OldColor());
        if (ShowFails) {
            Fails.push_back(err);
        }
        fprintf(stderr, "%s", ~err);
        NOTE_IN_VALGRIND(descr->test);
        PrintTimes();
        if (IsForked) {
            fprintf(stderr, "%s", ForkCorrectExitMsg);
        }
    }

    void OnFinish(const TFinish* descr) override {
        TraceProcessor->Finish(*descr);
        if (!IsForked && ForkExitedCorrectly) {
            return;
        }
        if (!PrintAfterTest_) {
            return;
        }

        if (descr->Success) {
            fprintf(stderr, "[%sgood%s] %s::%s\n", ~LightGreenColor(), ~OldColor(),
                    ~descr->test->unit->name,
                    descr->test->name);
            NOTE_IN_VALGRIND(descr->test);
            PrintTimes();
            if (IsForked) {
                fprintf(stderr, "%s", ForkCorrectExitMsg);
            }
        }
    }

    inline void PrintTimes() {
        if (!PrintTimes_) {
            return;
        }

        const TInstant now = TInstant::Now();
        Cerr << now - PrevTime_ << "\n";
        PrevTime_ = now;
    }

    void OnEnd() override {
        TraceProcessor->End();
        if (IsForked) {
            return;
        }

        if (!PrintSummary_) {
            return;
        }

        fprintf(stderr, "[%sDONE%s] ok: %s%u%s",
                ~YellowColor(), ~OldColor(),
                ~LightGreenColor(), GoodTests(), ~OldColor());
        if (FailTests())
            fprintf(stderr, ", err: %s%u%s",
                    ~LightRedColor(), FailTests(), ~OldColor());
        fprintf(stderr, "\n");

        if (ShowFails) {
            for (size_t i = 0; i < Fails.size(); ++i) {
                printf("%s", ~Fails[i]);
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

        if (DisabledSuites_.find(~name) != DisabledSuites_.end()) {
            return false;
        }

        if (EnabledSuites_.empty()) {
            return true;
        }

        return EnabledSuites_.find(~name) != EnabledSuites_.end();
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

    void Run(std::function<void()> f, TString suite, const char* name, const bool forceFork) override {
        if (!(ForkTests || forceFork) || GetIsForked()) {
            return f();
        }

        ylist<TString> args(1, "--is-forked-internal");
        args.push_back(Sprintf("+%s::%s", ~suite, name));

        // stdin is ignored - unittest should not need them...
        TShellCommand cmd(AppName, args,
                          TShellCommandOptions().SetUseShell(false).SetCloseAllFdsOnExec(true).SetAsync(false).SetLatency(1));
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
                // fallthrough
            }
            case TShellCommand::SHELL_ERROR: {
                ythrow yexception() << "Forked test failed";
            }

            case TShellCommand::SHELL_NONE: {
                ythrow yexception() << "Forked test finished with unknown status";
            }
            case TShellCommand::SHELL_RUNNING: {
                Y_VERIFY(false, "This can't happen, we used sync mode, it's a bug!");
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
    yhash_set<TString> DisabledSuites_;
    yhash_set<TString> EnabledSuites_;
    yhash_set<TString> DisabledTests_;
    yhash_set<TString> EnabledTests_;
    TInstant PrevTime_;
    bool ShowFails;
    yvector<TString> Fails;
    size_t Start;
    size_t End;
    TString AppName;
    bool ForkTests;
    bool IsForked;
    bool Loop;
    static const char* const ForkCorrectExitMsg;
    bool ForkExitedCorrectly;
    TAutoPtr<ITestSuiteProcessor> TraceProcessor;
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
            _CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
            _CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
        }
    }
    ~TWinEnvironment() {
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
         << "  --list-verbose        print a list of available subtests\n"
         << "  --print-before-test   print each test name before running it\n"
         << "  --print-before-suite  print each test suite name before running it\n"
         << "  --show-fails          print a list of all failed tests at the end\n"
         << "  --continue-on-fail    print a message and continue running test suite instead of break\n"
         << "  --print-times         print wall clock duration of each test\n"
         << "  --fork-tests          run each test in a separate process\n"
         << "  --trace-path          path to the trace file to be generated\n"
         << "  --trace-path-append   path to the trace file to be appended\n";
    return 0;
}

#if !defined(UTMAIN)
#define UTMAIN NUnitTest::RunMain
#endif

int UTMAIN(int argc, char** argv) {
    InitNetworkSubSystem();

    try {
        GetExecPath();
    } catch (...) {
    }

#ifndef UT_SKIP_EXCEPTIONS
    try {
#endif
        NPlugin::OnStartMain(argc, argv);

        TColoredProcessor processor(GetExecPath());
        IOutputStream* listStream = &Cout;
        THolder<IOutputStream> listFile;

        enum EListType {
            DONT_LIST,
            LIST,
            LIST_VERBOSE
        };
        EListType listTests = DONT_LIST;

        for (size_t i = 1; i < (size_t)argc; ++i) {
            const char* name = argv[i];

            if (name && *name) {
                if (strcmp(name, "--help") == 0 || strcmp(name, "-h") == 0) {
                    return DoUsage(argv[0]);
                } else if (strcmp(name, "--list") == 0 || strcmp(name, "-l") == 0) {
                    listTests = LIST;
                } else if (strcmp(name, "--list-verbose") == 0) {
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
                } else if (strcmp(name, "--continue-on-fail") == 0) {
                    processor.SetContinueOnFail(true);
                } else if (strcmp(name, "--print-times") == 0) {
                    processor.SetPrintTimes(true);
                } else if (strcmp(name, "--from") == 0) {
                    ++i;
                    processor.SetStart(FromString<size_t>(argv[i]));
                } else if (strcmp(name, "--to") == 0) {
                    ++i;
                    processor.SetEnd(FromString<size_t>(argv[i]));
                } else if (strcmp(name, "--fork-tests") == 0) {
                    processor.SetForkTests(true);
                } else if (strcmp(name, "--is-forked-internal") == 0) {
                    processor.SetIsForked(true);
                } else if (strcmp(name, "--loop") == 0) {
                    processor.SetLoop(true);
                } else if (strcmp(name, "--trace-path") == 0) {
                    ++i;
                    processor.BeQuiet();
                    NUnitTest::ShouldColorizeDiff = false;
                    processor.SetTraceProcessor(new TTraceWriterProcessor(argv[i], CreateAlways));
                } else if (strcmp(name, "--trace-path-append") == 0) {
                    ++i;
                    processor.BeQuiet();
                    NUnitTest::ShouldColorizeDiff = false;
                    processor.SetTraceProcessor(new TTraceWriterProcessor(argv[i], OpenAlways | ForAppend));
                } else if (strcmp(name, "--list-path") == 0) {
                    ++i;
                    listFile = new TBufferedFileOutput(argv[i]);
                    listStream = listFile.Get();
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

        TTestFactory::Instance().SetProcessor(&processor);

        for (;;) {
            const unsigned ret = TTestFactory::Instance().Execute();

            if (!processor.GetIsForked() && ret && processor.GetPrintSummary()) {
                Cerr << "SOME TESTS FAILED!!!!" << Endl;
            }

            if (ret || !processor.IsLoop()) {
                return ret;
            }
        }

        NPlugin::OnStopMain(argc, argv);
#ifndef UT_SKIP_EXCEPTIONS
    } catch (...) {
        Cerr << "caught exception in test suite(" << CurrentExceptionMessage() << ")" << Endl;
    }
#endif

    return 1;
}
