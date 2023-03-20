#include "junit.h"

#include <libxml/parser.h>
#include <libxml/xmlwriter.h>

#include <util/generic/scope.h>
#include <util/generic/size_literals.h>
#include <util/system/env.h>
#include <util/system/file.h>
#include <util/system/fs.h>
#include <util/system/file.h>
#include <util/system/fstat.h>
#include <util/system/tempfile.h>

#include <stdio.h>

#if defined(_win_)
#include <io.h>
#endif

namespace NUnitTest {

extern const TString Y_UNITTEST_OUTPUT_CMDLINE_OPTION = "Y_UNITTEST_OUTPUT";

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

    TString GetCapturedString() {
        Uncapture();

        TFile captured(TmpFile.Name(), EOpenModeFlag::RdOnly);
        i64 len = captured.GetLength();
        if (len > 0) {
            TString out;
            if (static_cast<size_t>(len) > 10_KB) {
                len = static_cast<i64>(10_KB);
            }
            out.resize(len);
            try {
                captured.Read((void*)out.data(), len);
                return out;
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
    Y_UNUSED(test);
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
        failure.Message = descr->msg;
        failure.BackTrace = descr->BackTrace;
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
        if (StdOutCapturer) {
            testCase->StdOut = StdOutCapturer->GetCapturedString();
            StdOutCapturer = nullptr;
            Cout.Write(testCase->StdOut);
        }
        if (StdErrCapturer) {
            testCase->StdErr = StdErrCapturer->GetCapturedString();
            StdErrCapturer = nullptr;
            Cerr.Write(testCase->StdErr);
        }
    } else {
        MergeSubprocessReport();
    }
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
        Y_FAIL("Cannot write report");
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
    for (xmlNodePtr suite = root->children; suite != nullptr; suite = suite->next) {
        try {
            CHECK_NODE_NAME(suite, "testsuite");
            TString suiteName = GetAttrValue(suite, "id");
            TTestSuite& suiteInfo = Suites[suiteName];

            // Test cases
            for (xmlNodePtr testCase = suite->children; testCase != nullptr; testCase = testCase->next) {
                try {
                    CHECK_NODE_NAME(testCase, "testcase");
                    TString caseName = GetAttrValue(testCase, "id");
                    TTestCase& testCaseInfo = suiteInfo.Cases[caseName];

                    if (TString duration = GetAttrValue(testCase, "time")) {
                        TryFromString(duration, testCaseInfo.DurationSecods);
                    }

                    // Failures/stderr/stdout
                    for (xmlNodePtr testProp = testCase->children; testProp != nullptr; testProp = testProp->next) {
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
