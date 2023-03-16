#include "registar.h"

namespace NUnitTest {

class TJUnitProcessor : public ITestSuiteProcessor {
    struct TTestCase {
        TString Name;
        bool Success;
        TVector<TString> Failures;

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
    };

public:
    TJUnitProcessor(TString file, TString exec)
        : FileName(file)
        , ExecName(exec)
    {
    }

    ~TJUnitProcessor() {
        Save();
    }

    void OnError(const TError* descr) override {
        auto* testCase = GetTestCase(descr->test);
        testCase->Failures.emplace_back(descr->msg);
    }

    void OnFinish(const TFinish* descr) override {
        GetTestCase(descr->test)->Success = descr->Success;
    }

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

private:
    TString FileName;
    TString ExecName;
    TMap<TString, TTestSuite> Suites;
};

} // namespace NUnitTest
