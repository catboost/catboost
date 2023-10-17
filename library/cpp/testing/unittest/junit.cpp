#include "junit.h"

#include <libxml/parser.h>
#include <libxml/xmlwriter.h>

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

static bool IsAllowedInXml(wchar32 c) {
    // https://en.wikipedia.org/wiki/Valid_characters_in_XML
    return c == 0x9
        || c == 0xA
        || c == 0xD
        || c >= 0x20 && c <= 0xD7FF
        || c >= 0xE000 && c <= 0xFFFD
        || c >= 0x10000 && c <= 0x10FFFF;
}

static TString SanitizeXmlString(TString s) {
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
            if (IsAllowedInXml(rune)) {
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

TJUnitProcessor::TJUnitProcessor(TString file, TString exec)
    : FileName(file)
    , ExecName(exec)
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
        failure.Message = SanitizeXmlString(descr->msg);
        failure.BackTrace = SanitizeXmlString(descr->BackTrace);
    }
}

void TJUnitProcessor::TransferFromCapturer(THolder<TJUnitProcessor::TOutputCapturer>& capturer, TString& out, IOutputStream& outStream) {
    if (capturer) {
        capturer->Uncapture();
        {
            TFileInput fileStream(capturer->GetTmpFileName());
            TransferData(&fileStream, &outStream);
            out = SanitizeXmlString(capturer->GetCapturedString());
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
            TString uniqReportFileName = BuildFileName(i, ".xml"sv);
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
        SetEnv(Y_UNITTEST_OUTPUT_CMDLINE_OPTION, TStringBuilder() << "xml:" << TmpReportFile->Name());
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

#define CHECK_CALL(expr) if (int resultCode = (expr); resultCode < 0) {    \
    Cerr << "Faield to write to xml. Result code: " << resultCode << Endl; \
    return;                                                                \
}

#define XML_STR(s) ((const xmlChar*)(s))

void TJUnitProcessor::SerializeToFile() {
    auto file = xmlNewTextWriterFilename(ResultReportFileName.c_str(), 0);
    if (!file) {
        Cerr << "Failed to open xml file for writing: " << ResultReportFileName << Endl;
        return;
    }

    Y_DEFER {
        xmlFreeTextWriter(file);
    };

    CHECK_CALL(xmlTextWriterSetIndent(file, 1));

    CHECK_CALL(xmlTextWriterStartDocument(file, nullptr, "UTF-8", nullptr));
    CHECK_CALL(xmlTextWriterStartElement(file, XML_STR("testsuites")));
    CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("tests"), XML_STR(ToString(GetTestsCount()).c_str())));
    CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("failures"), XML_STR(ToString(GetFailuresCount()).c_str())));

    for (const auto& [suiteName, suite] : Suites) {
        CHECK_CALL(xmlTextWriterStartElement(file, XML_STR("testsuite")));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("name"), XML_STR(suiteName.c_str())));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("id"), XML_STR(suiteName.c_str())));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("tests"), XML_STR(ToString(suite.GetTestsCount()).c_str())));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("failures"), XML_STR(ToString(suite.GetFailuresCount()).c_str())));
        CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("time"), XML_STR(ToString(suite.GetDurationSeconds()).c_str())));

        for (const auto& [testName, test] : suite.Cases) {
            CHECK_CALL(xmlTextWriterStartElement(file, XML_STR("testcase")));
            CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("classname"), XML_STR(suiteName.c_str())));
            CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("name"), XML_STR(testName.c_str())));
            CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("id"), XML_STR(testName.c_str())));
            CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("time"), XML_STR(ToString(test.DurationSecods).c_str())));

            for (const auto& failure : test.Failures) {
                CHECK_CALL(xmlTextWriterStartElement(file, XML_STR("failure")));
                CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("message"), XML_STR(failure.Message.c_str())));
                CHECK_CALL(xmlTextWriterWriteAttribute(file, XML_STR("type"), XML_STR("ERROR")));
                if (failure.BackTrace) {
                    CHECK_CALL(xmlTextWriterWriteString(file, XML_STR(failure.BackTrace.c_str())));
                }
                CHECK_CALL(xmlTextWriterEndElement(file));
            }

            if (!test.StdOut.empty()) {
                CHECK_CALL(xmlTextWriterWriteElement(file, XML_STR("system-out"), XML_STR(test.StdOut.c_str())));
            }
            if (!test.StdErr.empty()) {
                CHECK_CALL(xmlTextWriterWriteElement(file, XML_STR("system-err"), XML_STR(test.StdErr.c_str())));
            }

            CHECK_CALL(xmlTextWriterEndElement(file));
        }

        CHECK_CALL(xmlTextWriterEndElement(file));
    }

    CHECK_CALL(xmlTextWriterEndElement(file));
    CHECK_CALL(xmlTextWriterEndDocument(file));
}

#define C_STR(s) ((const char*)(s))
#define STRBUF(s) TStringBuf(C_STR(s))

#define NODE_NAME(node) STRBUF((node)->name)
#define SAFE_CONTENT(node) (node && node->children ? STRBUF(node->children->content) : TStringBuf())

#define CHECK_NODE_NAME(node, expectedName) if (NODE_NAME(node) != (expectedName)) { \
    ythrow yexception() << "Expected node name: \"" << (expectedName)                \
        << "\", but got \"" << TStringBuf(C_STR((node)->name)) << "\"";              \
}

static TString GetAttrValue(xmlNodePtr node, TStringBuf name, bool required = true) {
    for (xmlAttrPtr attr = node->properties; attr != nullptr; attr = attr->next) {
        if (NODE_NAME(attr) == name) {
            return TString(SAFE_CONTENT(attr));
        }
    }
    if (required) {
        ythrow yexception() << "Attribute \"" << name << "\" was not found";
    }
    return {};
}

static xmlNodePtr NextElement(xmlNodePtr node) {
    if (!node) {
        return nullptr;
    }

    do {
        node = node->next;
    } while (node && node->type != XML_ELEMENT_NODE);

    return node;
}

static xmlNodePtr ChildElement(xmlNodePtr node) {
    xmlNodePtr child = node->children;
    if (child && child->type != XML_ELEMENT_NODE) {
        return NextElement(child);
    }
    return child;
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

    xmlDocPtr doc = xmlParseFile(TmpReportFile->Name().c_str());
    if (!doc) {
        Cerr << "Failed to parse xml output for subprocess" << Endl;
        return;
    }

    Y_DEFER {
        xmlFreeDoc(doc);
    };

    xmlNodePtr root = xmlDocGetRootElement(doc);
    if (!root) {
        Cerr << "Failed to parse xml output for subprocess: empty document" << Endl;
        return;
    }

    CHECK_NODE_NAME(root, "testsuites");
    for (xmlNodePtr suite = ChildElement(root); suite != nullptr; suite = NextElement(suite)) {
        try {
            CHECK_NODE_NAME(suite, "testsuite");
            TString suiteName = GetAttrValue(suite, "id");
            TTestSuite& suiteInfo = Suites[suiteName];

            // Test cases
            for (xmlNodePtr testCase = ChildElement(suite); testCase != nullptr; testCase = NextElement(testCase)) {
                try {
                    CHECK_NODE_NAME(testCase, "testcase");
                    TString caseName = GetAttrValue(testCase, "id");
                    TTestCase& testCaseInfo = suiteInfo.Cases[caseName];

                    if (TString duration = GetAttrValue(testCase, "time")) {
                        TryFromString(duration, testCaseInfo.DurationSecods);
                    }

                    // Failures/stderr/stdout
                    for (xmlNodePtr testProp = ChildElement(testCase); testProp != nullptr; testProp = NextElement(testProp)) {
                        try {
                            if (NODE_NAME(testProp) == "failure") {
                                TString message = GetAttrValue(testProp, "message");
                                auto& failure = testCaseInfo.Failures.emplace_back();
                                failure.Message = message;
                                failure.BackTrace = TString(SAFE_CONTENT(testProp));
                            } else if (NODE_NAME(testProp) == "system-out") {
                                testCaseInfo.StdOut = TString(SAFE_CONTENT(testProp));
                            } else if (NODE_NAME(testProp) == "system-err") {
                                testCaseInfo.StdErr = TString(SAFE_CONTENT(testProp));
                            } else {
                                ythrow yexception() << "Unknown test case subprop: \"" << NODE_NAME(testProp) << "\"";
                            }
                        } catch (const std::exception& ex) {
                            auto& failure = testCaseInfo.Failures.emplace_back();
                            failure.Message = TStringBuilder() << "Failed to read part of test case info from unit test tool: " << ex.what();
                            Cerr << "Failed to load test case " << caseName << " failure in suite " << suiteName << ": " << ex.what() << Endl;
                            continue;
                        }
                    }
                } catch (const std::exception& ex) {
                    Cerr << "Failed to load test case info in suite " << suiteName << ": " << ex.what() << Endl;
                    continue;
                }
            }
        } catch (const std::exception& ex) {
            Cerr << "Failed to load test suite info from xml: " << ex.what() << Endl;
            continue;
        }
    }
}

} // namespace NUnitTest
