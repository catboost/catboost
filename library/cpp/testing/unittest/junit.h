#include "registar.h"

#include <util/datetime/base.h>
#include <util/generic/maybe.h>
#include <util/system/tempfile.h>

namespace NUnitTest {

extern const TString Y_UNITTEST_OUTPUT_CMDLINE_OPTION;

class TJUnitProcessor : public ITestSuiteProcessor {
    struct TFailure {
        TString Message;
        TString BackTrace;
    };

    struct TTestCase {
        TString Name;
        bool Success;
        TVector<TFailure> Failures;
        TString StdOut;
        TString StdErr;
        double DurationSecods = 0.0;

        size_t GetFailuresCount() const {
            return Failures.size();
        }
    };

    struct TTestSuite {
        TMap<TString, TTestCase> Cases;

        size_t GetTestsCount() const {
            return Cases.size();
        }

        size_t GetFailuresCount() const {
            size_t sum = 0;
            for (const auto& [name, testCase] : Cases) {
                sum += testCase.GetFailuresCount();
            }
            return sum;
        }

        double GetDurationSeconds() const {
            double sum = 0.0;
            for (const auto& [name, testCase] : Cases) {
                sum += testCase.DurationSecods;
            }
            return sum;
        }
    };

    struct TOutputCapturer;

public:
    TJUnitProcessor(TString file, TString exec);
    ~TJUnitProcessor();

    void SetForkTestsParams(bool forkTests, bool isForked) override;

    void OnBeforeTest(const TTest* test) override;
    void OnError(const TError* descr) override;
    void OnFinish(const TFinish* descr) override;

private:
    TTestCase* GetTestCase(const TTest* test) {
        auto& suite = Suites[test->unit->name];
        return &suite.Cases[test->name];
    }

    void Save();

    size_t GetTestsCount() const {
        size_t sum = 0;
        for (const auto& [name, suite] : Suites) {
            sum += suite.GetTestsCount();
        }
        return sum;
    }

    size_t GetFailuresCount() const {
        size_t sum = 0;
        for (const auto& [name, suite] : Suites) {
            sum += suite.GetFailuresCount();
        }
        return sum;
    }

    void SerializeToFile();
    void MergeSubprocessReport();

    TString BuildFileName(size_t index, const TStringBuf extension) const;
    void MakeReportFileName();
    void MakeTmpFileNameForForkedTests();
    static void TransferFromCapturer(THolder<TJUnitProcessor::TOutputCapturer>& capturer, TString& out, IOutputStream& outStream);

private:
    const TString FileName; // cmd line param
    const TString ExecName; // cmd line param
    TString ResultReportFileName;
    TMaybe<TTempFile> TmpReportFile;
    TMap<TString, TTestSuite> Suites;
    THolder<TOutputCapturer> StdErrCapturer;
    THolder<TOutputCapturer> StdOutCapturer;
    TInstant StartCurrentTestTime;
};

} // namespace NUnitTest
