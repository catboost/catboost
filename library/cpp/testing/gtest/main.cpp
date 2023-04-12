#include "main.h"
#include "gtest.h"

#include <library/cpp/string_utils/relaxed_escaper/relaxed_escaper.h>
#include <library/cpp/testing/common/env.h>
#include <library/cpp/testing/hook/hook.h>
#include <util/generic/scope.h>
#include <util/string/join.h>
#include <util/system/src_root.h>

#include <fstream>

namespace {
    bool StartsWith(const char* str, const char* pre) {
        return strncmp(pre, str, strlen(pre)) == 0;
    }

    void Unsupported(const char* flag) {
        std::cerr << "This GTest wrapper does not support flag " << flag << std::endl;
        exit(2);
    }

    void Unknown(const char* flag) {
        std::cerr << "Unknown support flag " << flag << std::endl;
        exit(2);
    }

    std::pair<std::string_view, std::string_view> ParseName(std::string_view name) {
        auto pos = name.find("::");
        if (pos == std::string_view::npos) {
            return {name, "*"};
        } else {
            return {name.substr(0, pos), name.substr(pos + 2, name.size())};
        }
    }

    std::pair<std::string_view, std::string_view> ParseParam(std::string_view param) {
        auto pos = param.find("=");
        if (pos == std::string_view::npos) {
            return {param, ""};
        } else {
            return {param.substr(0, pos), param.substr(pos + 1, param.size())};
        }
    }

    constexpr std::string_view StripRoot(std::string_view f) noexcept {
        return ::NPrivate::StripRoot(::NPrivate::TStaticBuf(f.data(), f.size())).As<std::string_view>();
    }

    std::string EscapeJson(std::string_view str) {
        TString result;
        NEscJ::EscapeJ<true, true>(str, result);
        return result;
    }

    class TTraceWriter: public ::testing::EmptyTestEventListener {
    public:
        explicit TTraceWriter(std::ostream* trace)
            : Trace_(trace)
        {
        }

    private:
        void OnTestProgramStart(const testing::UnitTest& test) override {
            auto ts = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::system_clock::now().time_since_epoch());

            for (int i = 0; i < test.total_test_suite_count(); ++i) {
                auto suite = test.GetTestSuite(i);
                for (int j = 0; j < suite->total_test_count(); ++j) {
                    auto testInfo = suite->GetTestInfo(j);
                    if (testInfo->is_reportable() && !testInfo->should_run()) {
                        PrintTestStatus(*testInfo, "skipped", "test is disabled", {}, ts);
                    }
                }
            }
        }

        void OnTestStart(const ::testing::TestInfo& testInfo) override {
            // We fully format this marker before printing it to stderr/stdout because we want to print it atomically.
            // If we were to write `std::cout << "\n###subtest-finished:" << name`, there would be a chance that
            // someone else could sneak in and print something between `"\n###subtest-finished"` and `name`
            // (this happens when test binary uses both `Cout` and `std::cout`).
            auto marker = Join("", "\n###subtest-started:", testInfo.test_suite_name(), "::", testInfo.name(), "\n");

            // Theoretically, we don't need to flush both `Cerr` and `std::cerr` here because both ultimately
            // result in calling `fflush(stderr)`. However, there may be additional buffering logic
            // going on (custom `std::cerr.tie()`, for example), so just to be sure, we flush both of them.
            std::cout << std::flush;
            Cout << marker << Flush;

            std::cerr << std::flush;
            Cerr << marker << Flush;

            auto ts = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::system_clock::now().time_since_epoch());

            (*Trace_)
                << "{"
                <<   "\"name\": \"subtest-started\", "
                <<   "\"timestamp\": " << std::setprecision(14) << ts.count() << ", "
                <<   "\"value\": {"
                <<     "\"class\": " << EscapeJson(testInfo.test_suite_name()) << ", "
                <<     "\"subtest\": " << EscapeJson(testInfo.name())
                <<   "}"
                << "}"
                << "\n"
                << std::flush;
        }

        void OnTestPartResult(const testing::TestPartResult& result) override {
            if (!result.passed()) {
                if (result.file_name()) {
                    std::cerr << StripRoot(result.file_name()) << ":" << result.line_number() << ":" << "\n";
                }
                std::cerr << result.message() << "\n";
                std::cerr << std::flush;
            }
        }

        void OnTestEnd(const ::testing::TestInfo& testInfo) override {
            auto ts = std::chrono::duration_cast<std::chrono::duration<double>>(
                std::chrono::system_clock::now().time_since_epoch());

            std::string_view status = "good";
            if (testInfo.result()->Failed()) {
                status = "fail";
            } else if (testInfo.result()->Skipped()) {
                status = "skipped";
            }

            std::ostringstream messages;
            std::unordered_map<std::string, double> properties;

            {
                if (testInfo.value_param()) {
                    messages << "Value param:\n  " << testInfo.value_param() << "\n";
                }

                if (testInfo.type_param()) {
                    messages << "Type param:\n  " << testInfo.type_param() << "\n";
                }

                std::string_view sep;
                for (int i = 0; i < testInfo.result()->total_part_count(); ++i) {
                    auto part = testInfo.result()->GetTestPartResult(i);
                    if (part.failed()) {
                        messages << sep;
                        if (part.file_name()) {
                            messages << StripRoot(part.file_name()) << ":" << part.line_number() << ":\n";
                        }
                        messages << part.message();
                        messages << "\n";
                        sep = "\n";
                    }
                }

                for (int i = 0; i < testInfo.result()->test_property_count(); ++i) {
                    auto& property = testInfo.result()->GetTestProperty(i);

                    double value;

                    try {
                        value = std::stod(property.value());
                    } catch (std::invalid_argument&) {
                        messages
                            << sep
                            << "Arcadia CI only supports numeric properties, property "
                            << property.key() << "=" << EscapeJson(property.value()) << " is not a number\n";
                        std::cerr
                            << "Arcadia CI only supports numeric properties, property "
                            << property.key() << "=" << EscapeJson(property.value()) << " is not a number\n"
                            << std::flush;
                        status = "fail";
                        sep = "\n";
                        continue;
                    } catch (std::out_of_range&) {
                        messages
                            << sep
                            << "Property " << property.key() << "=" << EscapeJson(property.value())
                            << " is too big for a double precision value\n";
                        std::cerr
                            << "Property " << property.key() << "=" << EscapeJson(property.value())
                            << " is too big for a double precision value\n"
                            << std::flush;
                        status = "fail";
                        sep = "\n";
                        continue;
                    }

                    properties[property.key()] = value;
                }
            }

            auto marker = Join("", "\n###subtest-finished:", testInfo.test_suite_name(), "::", testInfo.name(), "\n");

            std::cout << std::flush;
            Cout << marker << Flush;

            std::cerr << std::flush;
            Cerr << marker << Flush;

            PrintTestStatus(testInfo, status, messages.str(), properties, ts);
        }

        void PrintTestStatus(
                const ::testing::TestInfo& testInfo,
                std::string_view status,
                std::string_view messages,
                const std::unordered_map<std::string, double>& properties,
                std::chrono::duration<double> ts)
        {
            (*Trace_)
                << "{"
                <<   "\"name\": \"subtest-finished\", "
                <<   "\"timestamp\": " << std::setprecision(14) << ts.count() << ", "
                <<   "\"value\": {"
                <<     "\"class\": " << EscapeJson(testInfo.test_suite_name()) << ", "
                <<     "\"subtest\": " << EscapeJson(testInfo.name()) << ", "
                <<     "\"comment\": " << EscapeJson(messages) << ", "
                <<     "\"status\": " << EscapeJson(status) << ", "
                <<     "\"time\": " << (testInfo.result()->elapsed_time() * (1 / 1000.0)) << ", "
                <<     "\"metrics\": {";
            {
                std::string_view sep = "";
                for (auto& [key, value]: properties) {
                    (*Trace_) << sep << EscapeJson(key) << ": " << value;
                    sep = ", ";
                }
            }
            (*Trace_)
                <<     "}"
                <<   "}"
                << "}"
                << "\n"
                << std::flush;
        }

        std::ostream* Trace_;
    };
}

int NGTest::Main(int argc, char** argv) {
    auto flags = ParseFlags(argc, argv);

    ::testing::GTEST_FLAG(filter) = flags.Filter;

    std::ofstream trace;
    if (!flags.TracePath.empty()) {
        trace.open(flags.TracePath, (flags.AppendTrace ? std::ios::app : std::ios::out) | std::ios::binary);

        if (!trace.is_open()) {
            std::cerr << "Failed to open file " << flags.TracePath << " for write" << std::endl;
            exit(2);
        }

        UnsetDefaultReporter();
        SetTraceReporter(&trace);
    }

    NTesting::THook::CallBeforeInit();

    ::testing::InitGoogleMock(&flags.GtestArgc, flags.GtestArgv.data());

    ListTests(flags.ListLevel, flags.ListPath);

    NTesting::THook::CallBeforeRun();

    Y_DEFER { NTesting::THook::CallAfterRun(); };

    return RUN_ALL_TESTS();
}

NGTest::TFlags NGTest::ParseFlags(int argc, char** argv) {
    TFlags result;

    std::ostringstream filtersPos;
    std::string_view filterPosSep = "";
    std::ostringstream filtersNeg;
    std::string_view filterNegSep = "";

    if (argc > 0) {
        result.GtestArgv.push_back(argv[0]);
    }

    for (int i = 1; i < argc; ++i) {
        auto name = argv[i];

        if (strcmp(name, "--help") == 0) {
            result.GtestArgv.push_back(name);
            break;
        } else if (StartsWith(name, "--gtest_") || StartsWith(name, "--gmock_")) {
            result.GtestArgv.push_back(name);
        } else if (strcmp(name, "--list") == 0 || strcmp(name, "-l") == 0) {
            result.ListLevel = std::max(result.ListLevel, 1);
        } else if (strcmp(name, "--list-verbose") == 0) {
            result.ListLevel = std::max(result.ListLevel, 2);
        } else if (strcmp(name, "--print-before-suite") == 0) {
            Unsupported("--print-before-suite");
        } else if (strcmp(name, "--print-before-test") == 0) {
            Unsupported("--print-before-test");
        } else if (strcmp(name, "--show-fails") == 0) {
            Unsupported("--show-fails");
        } else if (strcmp(name, "--dont-show-fails") == 0) {
            Unsupported("--dont-show-fails");
        } else if (strcmp(name, "--print-times") == 0) {
            Unsupported("--print-times");
        } else if (strcmp(name, "--from") == 0) {
            Unsupported("--from");
        } else if (strcmp(name, "--to") == 0) {
            Unsupported("--to");
        } else if (strcmp(name, "--fork-tests") == 0) {
            Unsupported("--fork-tests");
        } else if (strcmp(name, "--is-forked-internal") == 0) {
            Unsupported("--is-forked-internal");
        } else if (strcmp(name, "--loop") == 0) {
            Unsupported("--loop");
        } else if (strcmp(name, "--trace-path") == 0 || strcmp(name, "--trace-path-append") == 0) {
            ++i;

            if (i >= argc) {
                std::cerr << "Missing value for argument --trace-path" << std::endl;
                exit(2);
            } else if (!result.TracePath.empty()) {
                std::cerr << "Multiple --trace-path or --trace-path-append given" << std::endl;
                exit(2);
            }

            result.TracePath = argv[i];
            result.AppendTrace = strcmp(name, "--trace-path-append") == 0;
        } else if (strcmp(name, "--list-path") == 0) {
            ++i;

            if (i >= argc) {
                std::cerr << "Missing value for argument --list-path" << std::endl;
                exit(2);
            }

            result.ListPath = argv[i];
        } else if (strcmp(name, "--test-param") == 0) {
            ++i;

            if (i >= argc) {
                std::cerr << "Missing value for argument --test-param" << std::endl;
                exit(2);
            }

            auto [key, value] = ParseParam(argv[i]);

        Singleton<NPrivate::TTestEnv>()->AddTestParam(key, value);
        } else if (StartsWith(name, "--")) {
            Unknown(name);
        } else if (*name == '-') {
            auto [suite, test] = ParseName(name + 1);
            filtersNeg << filterNegSep << suite << "." << test;
            filterNegSep = ":";
        } else if (*name == '+') {
            auto [suite, test] = ParseName(name + 1);
            filtersPos << filterPosSep << suite << "." << test;
            filterPosSep = ":";
        } else {
            auto [suite, test] = ParseName(name);
            filtersPos << filterPosSep << suite << "." << test;
            filterPosSep = ":";
        }
    }

    if (!filtersPos.str().empty() || !filtersNeg.str().empty()) {
        result.Filter = filtersPos.str();
        if (!filtersNeg.str().empty()) {
            result.Filter += "-";
            result.Filter += filtersNeg.str();
        }
    }

    // Main-like functions need a null sentinel at the end of `argv' argument.
    // This sentinel is not counted in `argc' argument.
    result.GtestArgv.push_back(nullptr);
    result.GtestArgc = static_cast<int>(result.GtestArgv.size()) - 1;

    return result;
}

void NGTest::ListTests(int listLevel, const std::string& listPath) {
    // NOTE: do not use `std::endl`, use `\n`; `std::endl` produces `\r\n`s on windows,
    // and ya make does not handle them well.

    if (listLevel > 0) {
        std::ostream* listOut = &std::cout;
        std::ofstream listFile;

        if (!listPath.empty()) {
            listFile.open(listPath, std::ios::out | std::ios::binary);
            if (!listFile.is_open()) {
                std::cerr << "Failed to open file " << listPath << " for write" << std::endl;
                exit(2);
            }
            listOut = &listFile;
        }

        for (int i = 0; i < testing::UnitTest::GetInstance()->total_test_suite_count(); ++i) {
            auto suite = testing::UnitTest::GetInstance()->GetTestSuite(i);
            if (listLevel > 1) {
                for (int j = 0; j < suite->total_test_count(); ++j) {
                    auto test = suite->GetTestInfo(j);
                    (*listOut) << suite->name() << "::" << test->name() << "\n";
                }
            } else {
                (*listOut) << suite->name() << "\n";
            }
        }

        (*listOut) << std::flush;

        exit(0);
    }
}

void NGTest::UnsetDefaultReporter() {
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
}

void NGTest::SetTraceReporter(std::ostream* traceFile) {
    ::testing::TestEventListeners& listeners = ::testing::UnitTest::GetInstance()->listeners();
    listeners.Append(new TTraceWriter{traceFile});
}
