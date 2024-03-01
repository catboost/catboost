#include "junit.h"

#include <library/cpp/json/json_reader.h>
#include <library/cpp/json/writer/json.h>
#include <library/cpp/json/writer/json_value.h>

#include <util/charset/utf8.h>
#include <util/generic/scope.h>
#include <util/generic/size_literals.h>
#include <util/stream/file.h>
#include <util/stream/input.h>
#include <util/system/backtrace.h>
#include <util/system/env.h>
#include <util/system/file.h>
#include <util/system/fs.h>
#include <util/system/file.h>
#include <util/system/fstat.h>
#include <util/system/tempfile.h>

#include <stdio.h>
#include <signal.h>

#if defined(_win_)
#include <io.h>
#endif

namespace NUnitTest {

extern const TString Y_UNITTEST_OUTPUT_CMDLINE_OPTION = "Y_UNITTEST_OUTPUT";
extern const TString Y_UNITTEST_TEST_FILTER_FILE_OPTION = "Y_UNITTEST_FILTER_FILE";

static bool IsAllowed(wchar32 c) {
    // https://en.wikipedia.org/wiki/Valid_characters_in_XML
    return c == 0x9
        || c == 0xA
        || c == 0xD
        || c >= 0x20 && c <= 0xD7FF
        || c >= 0xE000 && c <= 0xFFFD
        || c >= 0x10000 && c <= 0x10FFFF;
}

static TString SanitizeString(TString s) {
    TString escaped;
    bool fixedSomeChars = false;
    const unsigned char* i = reinterpret_cast<const unsigned char*>(s.data());
    const unsigned char* end = i + s.size();
    auto replaceChar = [&]() {
        if (!fixedSomeChars) {
            fixedSomeChars = true;
            escaped.reserve(s.size());
            escaped.insert(escaped.end(), s.data(), reinterpret_cast<const char*>(i));
        }
        escaped.push_back('?');
    };
    while (i < end) {
        wchar32 rune;
        size_t runeLen;
        const RECODE_RESULT result = SafeReadUTF8Char(rune, runeLen, i, end);
        if (result == RECODE_OK) {
            if (IsAllowed(rune)) {
                if (fixedSomeChars) {
                    escaped.insert(escaped.end(), reinterpret_cast<const char*>(i), reinterpret_cast<const char*>(i + runeLen));
                }
            } else {
                replaceChar();
            }
            i += runeLen;
        } else {
            replaceChar();
            ++i;
        }
    }
    if (fixedSomeChars) {
        return escaped;
    } else {
        return s;
    }
}

struct TJUnitProcessor::TOutputCapturer {
    static constexpr int STDOUT_FD = 1;
    static constexpr int STDERR_FD = 2;

    TOutputCapturer(int fd)
        : FdToCapture(fd)
        , TmpFile(MakeTempName())
    {
        {
#if defined(_win_)
            TFileHandle f((FHANDLE)_get_osfhandle(FdToCapture));
#else
            TFileHandle f(FdToCapture);
#endif
            TFileHandle other(f.Duplicate());
            Original.Swap(other);
            f.Release();
        }

        TFileHandle captured(TmpFile.Name(), EOpenModeFlag::OpenAlways | EOpenModeFlag::RdWr);

        fflush(nullptr);
        captured.Duplicate2Posix(FdToCapture);
    }

    ~TOutputCapturer() {
        Uncapture();
    }

    void Uncapture() {
        if (Original.IsOpen()) {
            fflush(nullptr);
            Original.Duplicate2Posix(FdToCapture);
            Original.Close();
        }
    }

    TString GetTmpFileName() {
        Uncapture();
        return TmpFile.Name();
    }

    TString GetCapturedString() {
        Uncapture();

        TFile captured(TmpFile.Name(), EOpenModeFlag::RdOnly);
        i64 len = captured.GetLength();
        if (len > 0) {
            try {
                constexpr size_t LIMIT = 10_KB;
                constexpr size_t PART_LIMIT = 5_KB;
                TStringBuilder out;
                if (static_cast<size_t>(len) <= LIMIT) {
                    out.resize(len);
                    captured.Read((void*)out.data(), len);
                } else {
                    // Read first 5_KB
                    {
                        TString first;
                        first.resize(PART_LIMIT);
                        captured.Read((void*)first.data(), PART_LIMIT);
                        size_t lastNewLine = first.find_last_of('\n');
                        if (lastNewLine == TString::npos) {
                            out << first << Endl;
                        } else {
                            out << TStringBuf(first.c_str(), lastNewLine);
                        }
                    }

                    out << Endl << Endl << "...SKIPPED..." << Endl << Endl;

                    // Read last 5_KB
                    {
                        TString last;
                        last.resize(PART_LIMIT);
                        captured.Seek(-PART_LIMIT, sEnd);
                        captured.Read((void*)last.data(), PART_LIMIT);
                        size_t newLine = last.find_first_of('\n');
                        if (newLine == TString::npos) {
                            out << last << Endl;
                        } else {
                            out << TStringBuf(last.c_str() + newLine + 1);
                        }
                    }
                }
                if (out.back() != '\n') {
                    out << Endl;
                }
                return std::move(out);
            } catch (const std::exception& ex) {
                Cerr << "Failed to read from captured output: " << ex.what() << Endl;
            }
        }
        return {};
    }

    const int FdToCapture;
    TFileHandle Original;
    TTempFile TmpFile;
};

TJUnitProcessor::TJUnitProcessor(TString file, TString exec, EOutputFormat outputFormat)
    : FileName(file)
    , ExecName(exec)
    , OutputFormat(outputFormat)
{
}

TJUnitProcessor::~TJUnitProcessor() {
    Save();
}

void TJUnitProcessor::OnBeforeTest(const TTest* test) {
    CurrentTest.emplace(test);
    CaptureSignal(this);
    if (!GetForkTests() || GetIsForked()) {
        StdErrCapturer = MakeHolder<TOutputCapturer>(TOutputCapturer::STDERR_FD);
        StdOutCapturer = MakeHolder<TOutputCapturer>(TOutputCapturer::STDOUT_FD);
        StartCurrentTestTime = TInstant::Now();
    }
}

void TJUnitProcessor::OnError(const TError* descr) {
    if (!GetForkTests() || GetIsForked()) {
        auto* testCase = GetTestCase(descr->test);
        TFailure& failure = testCase->Failures.emplace_back();
        failure.Message = SanitizeString(descr->msg);
        failure.BackTrace = SanitizeString(descr->BackTrace);
    }
}

void TJUnitProcessor::TransferFromCapturer(THolder<TJUnitProcessor::TOutputCapturer>& capturer, TString& out, IOutputStream& outStream) {
    if (capturer) {
        capturer->Uncapture();
        {
            TFileInput fileStream(capturer->GetTmpFileName());
            TransferData(&fileStream, &outStream);
            out = SanitizeString(capturer->GetCapturedString());
        }
        capturer = nullptr;
    }
}

void TJUnitProcessor::OnFinish(const TFinish* descr) {
    if (!GetForkTests() || GetIsForked()) {
        auto* testCase = GetTestCase(descr->test);
        testCase->Success = descr->Success;
        if (StartCurrentTestTime != TInstant::Zero()) {
            testCase->DurationSecods = (TInstant::Now() - StartCurrentTestTime).SecondsFloat();
        }
        StartCurrentTestTime = TInstant::Zero();
        TransferFromCapturer(StdOutCapturer, testCase->StdOut, Cout);
        TransferFromCapturer(StdErrCapturer, testCase->StdErr, Cerr);
    } else {
        MergeSubprocessReport();
    }
    UncaptureSignal();
}

TString TJUnitProcessor::BuildFileName(size_t index, const TStringBuf extension) const {
    TStringBuilder result;
    result << FileName << ExecName;
    if (index > 0) {
        result << "-"sv << index;
    }
    result << extension;
    return std::move(result);
}

TStringBuf TJUnitProcessor::GetFileExtension() const {
    switch (OutputFormat) {
    case EOutputFormat::Xml:
        return ".xml"sv;
    case EOutputFormat::Json:
        return ".json"sv;
    }
    return TStringBuf();
}

void TJUnitProcessor::MakeReportFileName() {
    constexpr size_t MaxReps = 200;

#if defined(_win_)
    constexpr char DirSeparator = '\\';
#else
    constexpr char DirSeparator = '/';
#endif

    if (!ResultReportFileName.empty()) {
        return;
    }

    if (GetIsForked() || !FileName.empty() && FileName.back() != DirSeparator) {
        ResultReportFileName = FileName;
    } else { // Directory is specified, => make unique report name
        if (!FileName.empty()) {
            NFs::MakeDirectoryRecursive(FileName);
        }
        for (size_t i = 0; i < MaxReps; ++i) {
            TString uniqReportFileName = BuildFileName(i, GetFileExtension());
            try {
                TFile newUniqReportFile(uniqReportFileName, EOpenModeFlag::CreateNew);
                newUniqReportFile.Close();
                ResultReportFileName = std::move(uniqReportFileName);
                break;
            } catch (const TFileError&) {
                // File already exists => try next name
            }
        }
    }

    if (ResultReportFileName.empty()) {
        Cerr << "Could not find a vacant file name to write report for path " << FileName << ", maximum number of reports: " << MaxReps << Endl;
        Y_ABORT("Cannot write report");
    }
}

void TJUnitProcessor::Save() {
    MakeReportFileName();
    SerializeToFile();
}

void TJUnitProcessor::SetForkTestsParams(bool forkTests, bool isForked) {
    ITestSuiteProcessor::SetForkTestsParams(forkTests, isForked);
    MakeTmpFileNameForForkedTests();
}

void TJUnitProcessor::MakeTmpFileNameForForkedTests() {
    if (GetForkTests() && !GetIsForked()) {
        TmpReportFile.ConstructInPlace(MakeTempName());
        // Replace option for child processes
        SetEnv(Y_UNITTEST_OUTPUT_CMDLINE_OPTION, TStringBuilder() << "json:" << TmpReportFile->Name());
    }
}

static TJUnitProcessor* CurrentJUnitProcessor = nullptr;

void TJUnitProcessor::CaptureSignal(TJUnitProcessor* processor) {
    CurrentJUnitProcessor = processor;
    processor->PrevAbortHandler = signal(SIGABRT, &TJUnitProcessor::SignalHandler);
    if (processor->PrevAbortHandler == SIG_ERR) {
        processor->PrevAbortHandler = nullptr;
    }
    processor->PrevSegvHandler = signal(SIGSEGV, &TJUnitProcessor::SignalHandler);
    if (processor->PrevSegvHandler == SIG_ERR) {
        processor->PrevSegvHandler = nullptr;
    }
}

void TJUnitProcessor::UncaptureSignal() {
    if (CurrentJUnitProcessor) {
        if (CurrentJUnitProcessor->PrevAbortHandler != nullptr) {
            signal(SIGABRT, CurrentJUnitProcessor->PrevAbortHandler);
        } else {
            signal(SIGABRT, SIG_DFL);
        }

        if (CurrentJUnitProcessor->PrevSegvHandler != nullptr) {
            signal(SIGSEGV, CurrentJUnitProcessor->PrevSegvHandler);
        } else {
            signal(SIGSEGV, SIG_DFL);
        }
    }
}

void TJUnitProcessor::SignalHandler(int signal) {
    if (CurrentJUnitProcessor) {
        if (CurrentJUnitProcessor->CurrentTest) {
            TError errDesc;
            errDesc.test = *CurrentJUnitProcessor->CurrentTest;
            if (signal == SIGABRT) {
                errDesc.msg = "Test aborted";
            } else {
                errDesc.msg = "Segmentation fault";
                PrintBackTrace();
            }
            CurrentJUnitProcessor->OnError(&errDesc);

            TFinish finishDesc;
            finishDesc.Success = false;
            finishDesc.test = *CurrentJUnitProcessor->CurrentTest;
            CurrentJUnitProcessor->OnFinish(&finishDesc);
        }

        CurrentJUnitProcessor->Save();

        if (signal == SIGABRT) {
            if (CurrentJUnitProcessor->PrevAbortHandler) {
                CurrentJUnitProcessor->PrevAbortHandler(signal);
            }
        } else {
            if (CurrentJUnitProcessor->PrevSegvHandler) {
                CurrentJUnitProcessor->PrevSegvHandler(signal);
            }
        }
    }
}

void TJUnitProcessor::SerializeToFile() {
    switch (OutputFormat) {
    case EOutputFormat::Json:
        SerializeToJson();
        break;
    case EOutputFormat::Xml:
        [[fallthrough]];
    default:
        SerializeToXml();
        break;
    }
}

void TJUnitProcessor::SerializeToJson() {
    TFileOutput out(ResultReportFileName);
    NJsonWriter::TBuf json(NJsonWriter::HEM_UNSAFE, &out);
    json.SetIndentSpaces(1);
    json.BeginObject();
    {
        json.WriteKey("tests"sv).WriteInt(GetTestsCount());
        json.WriteKey("failures"sv).WriteInt(GetFailuresCount());
        json.WriteKey("testsuites"sv).BeginList();
        for (const auto& [suiteName, suite] : Suites) {
            json.BeginObject();
            json.WriteKey("name"sv).WriteString(suiteName);
            json.WriteKey("id"sv).WriteString(suiteName);
            json.WriteKey("tests"sv).WriteInt(suite.GetTestsCount());
            json.WriteKey("failures"sv).WriteInt(suite.GetFailuresCount());
            json.WriteKey("time"sv).WriteDouble(suite.GetDurationSeconds());
            json.WriteKey("testcases"sv).BeginList();
            for (const auto& [testName, test] : suite.Cases) {
                json.BeginObject();
                json.WriteKey("classname"sv).WriteString(suiteName);
                json.WriteKey("name"sv).WriteString(testName);
                json.WriteKey("id"sv).WriteString(testName);
                json.WriteKey("time"sv).WriteDouble(test.DurationSecods);
                json.WriteKey("failures"sv).BeginList();
                for (const auto& failure : test.Failures) {
                    json.BeginObject();
                    json.WriteKey("message"sv).WriteString(failure.Message);
                    json.WriteKey("type"sv).WriteString("ERROR"sv);
                    if (failure.BackTrace) {
                        json.WriteKey("backtrace"sv).WriteString(failure.BackTrace);
                    }
                    json.EndObject();
                }
                json.EndList();

                if (!test.StdOut.empty()) {
                    json.WriteKey("system-out"sv).WriteString(test.StdOut);
                }
                if (!test.StdErr.empty()) {
                    json.WriteKey("system-err"sv).WriteString(test.StdErr);
                }
                json.EndObject();
            }
            json.EndList();
            json.EndObject();
        }
        json.EndList();
    }
    json.EndObject();
}

class TXmlWriter {
public:
    class TTag {
        friend class TXmlWriter;

        explicit TTag(TXmlWriter* parent, TStringBuf name, size_t indent)
            : Parent(parent)
            , Name(name)
            , Indent(indent)
        {
            Start();
        }

    public:
        TTag(TTag&& tag)
            : Parent(tag.Parent)
            , Name(tag.Name)
        {
            tag.Parent = nullptr;
        }

        ~TTag() {
            if (Parent) {
                End();
            }
        }

        template <class T>
        TTag& Attribute(TStringBuf name, const T& value) {
            return Attribute(name, TStringBuf(ToString(value)));
        }

        TTag& Attribute(TStringBuf name, const TStringBuf& value) {
            Y_ABORT_UNLESS(!HasChildren);
            Parent->Out << ' ';
            Parent->Escape(name);
            Parent->Out << "=\"";
            Parent->Escape(value);
            Parent->Out << '\"';
            return *this;
        }

        TTag Tag(TStringBuf name) {
            if (!HasChildren) {
                HasChildren = true;
                Close();
            }
            return TTag(Parent, name, Indent + 1);
        }

        TTag& Text(TStringBuf text) {
            if (!HasChildren) {
                HasChildren = true;
                Close();
            }
            Parent->Escape(text);
            if (!text.empty() && text.back() == '\n') {
                NewLineBeforeIndent = false;
            }
            return *this;
        }

    private:
        void Start() {
            Parent->Indent(Indent);
            Parent->Out << '<';
            Parent->Escape(Name);
        }

        void Close() {
            Parent->Out << '>';
        }

        void End() {
            if (HasChildren) {
                Parent->Indent(Indent, NewLineBeforeIndent);
                Parent->Out << "</";
                Parent->Escape(Name);
                Parent->Out << ">";
            } else {
                Parent->Out << "/>";
            }
        }

    private:
        TXmlWriter* Parent = nullptr;
        TStringBuf Name;
        size_t Indent = 0;
        bool HasChildren = false;
        bool NewLineBeforeIndent = true;
    };

public:
    explicit TXmlWriter(const TString& fileName)
        : Out(fileName)
    {
        StartFile();
    }

    ~TXmlWriter() {
        Out << '\n';
    }

    TTag Tag(TStringBuf name) {
        return TTag(this, name, 0);
    }

private:
    void StartFile() {
        Out << R"(<?xml version="1.0" encoding="UTF-8"?>)"sv;
    }

    void Indent(size_t count, bool insertNewLine = true) {
        if (insertNewLine) {
            Out << '\n';
        }

        while (count--) {
            Out << ' ';
        }
    }

    void Escape(const TStringBuf str) {
        const unsigned char* i = reinterpret_cast<const unsigned char*>(str.data());
        const unsigned char* end = i + str.size();
        while (i < end) {
            wchar32 rune;
            size_t runeLen;
            const RECODE_RESULT result = SafeReadUTF8Char(rune, runeLen, i, end);
            if (result == RECODE_OK) { // string is expected not to have unallowed characters now
                switch (rune) {
                    case '\'':
                        Out.Write("&apos;");
                        break;
                    case '\"':
                        Out.Write("&quot;");
                        break;
                    case '<':
                        Out.Write("&lt;");
                        break;
                    case '>':
                        Out.Write("&gt;");
                        break;
                    case '&':
                        Out.Write("&amp;");
                        break;
                    default:
                        Out.Write(i, runeLen);
                        break;
                }
                i += runeLen;
            }
        }
    }

private:
    TFileOutput Out;
};

void TJUnitProcessor::SerializeToXml() {
    TXmlWriter report(ResultReportFileName);
    TXmlWriter::TTag testSuites = report.Tag("testsuites"sv);
    testSuites
        .Attribute("tests"sv, GetTestsCount())
        .Attribute("failures"sv, GetFailuresCount());

    for (const auto& [suiteName, suite] : Suites) {
        auto testSuite = testSuites.Tag("testsuite"sv);
        testSuite
            .Attribute("name"sv, suiteName)
            .Attribute("id"sv, suiteName)
            .Attribute("tests"sv, suite.GetTestsCount())
            .Attribute("failures"sv, suite.GetFailuresCount())
            .Attribute("time"sv, suite.GetDurationSeconds());

        for (const auto& [testName, test] : suite.Cases) {
            auto testCase = testSuite.Tag("testcase"sv);
            testCase
                .Attribute("classname"sv, suiteName)
                .Attribute("name"sv, testName)
                .Attribute("id"sv, testName)
                .Attribute("time"sv, test.DurationSecods);

            for (const auto& failure : test.Failures) {
                auto testFailure = testCase.Tag("failure"sv);
                testFailure
                    .Attribute("message"sv, failure.Message)
                    .Attribute("type"sv, "ERROR"sv);
                if (!failure.BackTrace.empty()) {
                    testFailure.Text(failure.BackTrace);
                }
            }

            if (!test.StdOut.empty()) {
                testCase.Tag("system-out"sv).Text(test.StdOut);
            }
            if (!test.StdErr.empty()) {
                testCase.Tag("system-err"sv).Text(test.StdErr);
            }
        }
    }
}

void TJUnitProcessor::MergeSubprocessReport() {
    {
        const i64 len = GetFileLength(TmpReportFile->Name());
        if (len < 0) {
            Cerr << "Failed to get length of the output file for subprocess" << Endl;
            return;
        }
        if (len == 0) {
            return; // Empty file
        }
    }

    Y_DEFER {
        TFile file(TmpReportFile->Name(), EOpenModeFlag::TruncExisting);
        file.Close();
    };

    NJson::TJsonValue testsReportJson;
    {
        TFileInput in(TmpReportFile->Name());
        if (!NJson::ReadJsonTree(&in, &testsReportJson)) {
            Cerr << "Failed to read json report for subprocess" << Endl;
            return;
        }
    }

    if (!testsReportJson.IsMap()) {
        Cerr << "Invalid subprocess report format: report is not a map" << Endl;
        return;
    }

    const NJson::TJsonValue* testSuitesJson = nullptr;
    if (!testsReportJson.GetValuePointer("testsuites"sv, &testSuitesJson)) {
        // no tests for some reason
        Cerr << "No tests found in subprocess report" << Endl;
        return;
    }

    if (!testSuitesJson->IsArray()) {
        Cerr << "Invalid subprocess report format: testsuites is not an array" << Endl;
        return;
    }

    for (const NJson::TJsonValue& suiteJson : testSuitesJson->GetArray()) {
        if (!suiteJson.IsMap()) {
            Cerr << "Invalid subprocess report format: suite is not a map" << Endl;
            continue;
        }
        const NJson::TJsonValue* suiteIdJson = nullptr;
        if (!suiteJson.GetValuePointer("id"sv, &suiteIdJson)) {
            Cerr << "Invalid subprocess report format: suite does not have id" << Endl;
            continue;
        }

        const TString& suiteId = suiteIdJson->GetString();
        if (suiteId.empty()) {
            Cerr << "Invalid subprocess report format: suite has empty id" << Endl;
            continue;
        }

        TTestSuite& suiteInfo = Suites[suiteId];
        const NJson::TJsonValue* testCasesJson = nullptr;
        if (!suiteJson.GetValuePointer("testcases"sv, &testCasesJson)) {
            Cerr << "No test cases found in suite \"" << suiteId << "\"" << Endl;
            continue;
        }
        if (!testCasesJson->IsArray()) {
            Cerr << "Invalid subprocess report format: testcases value is not an array" << Endl;
            continue;
        }

        for (const NJson::TJsonValue& testCaseJson : testCasesJson->GetArray()) {
            const NJson::TJsonValue* testCaseIdJson = nullptr;
            if (!testCaseJson.GetValuePointer("id"sv, &testCaseIdJson)) {
                Cerr << "Invalid subprocess report format: test case does not have id" << Endl;
                continue;
            }

            const TString& testCaseId = testCaseIdJson->GetString();
            if (testCaseId.empty()) {
                Cerr << "Invalid subprocess report format: test case has empty id" << Endl;
                continue;
            }

            TTestCase& testCaseInfo = suiteInfo.Cases[testCaseId];

            const NJson::TJsonValue* testCaseDurationJson = nullptr;
            if (testCaseJson.GetValuePointer("time"sv, &testCaseDurationJson)) {
                testCaseInfo.DurationSecods = testCaseDurationJson->GetDouble(); // Will handle also integers as double
            }

            const NJson::TJsonValue* stdOutJson = nullptr;
            if (testCaseJson.GetValuePointer("system-out"sv, &stdOutJson)) {
                testCaseInfo.StdOut = stdOutJson->GetString();
            }

            const NJson::TJsonValue* stdErrJson = nullptr;
            if (testCaseJson.GetValuePointer("system-err"sv, &stdErrJson)) {
                testCaseInfo.StdErr = stdErrJson->GetString();
            }

            const NJson::TJsonValue* failuresJson = nullptr;
            if (!testCaseJson.GetValuePointer("failures"sv, &failuresJson)) {
                continue;
            }

            if (!failuresJson->IsArray()) {
                Cerr << "Invalid subprocess report format: failures is not an array" << Endl;
                continue;
            }

            for (const NJson::TJsonValue& failureJson : failuresJson->GetArray()) {
                TFailure& failureInfo = testCaseInfo.Failures.emplace_back();

                const NJson::TJsonValue* messageJson = nullptr;
                if (failureJson.GetValuePointer("message"sv, &messageJson)) {
                    failureInfo.Message = messageJson->GetString();
                }

                const NJson::TJsonValue* backtraceJson = nullptr;
                if (failureJson.GetValuePointer("backtrace"sv, &backtraceJson)) {
                    failureInfo.BackTrace = backtraceJson->GetString();
                }
            }
        }
    }
}

} // namespace NUnitTest
