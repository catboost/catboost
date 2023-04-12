#pragma once

#include <iosfwd>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>


/**
 * You need to use these functions if you're customizing tests initialization
 * or writing a custom `main`.
 */

namespace NGTest {
    /**
     * Default `main` implementation.
     */
    int Main(int argc, char** argv);

    /**
     * CLI parsing result.
     */
    struct TFlags {
        /**
         * Argument for `ListTests` function.
         */
        int ListLevel = 0;

        /**
         * Where to print listed tests. Empty string means print to `stdout`.
         */
        std::string ListPath = "";

        /**
         * Path to trace file. If empty, tracing is not enabled.
         */
        std::string TracePath = "";

        /**
         * Should trace file be opened for append rather than just write.
         */
        bool AppendTrace = false;

        /**
         * Test filters.
         */
        std::string Filter = "*";

        /**
         * Number of CLI arguments for GTest init function (not counting the last null one).
         */
        int GtestArgc = 0;

        /**
         * CLI arguments for GTest init function.
         * The last one is nullptr.
         */
        std::vector<char*> GtestArgv = {};
    };

    /**
     * Parse unittest-related flags. Test binaries support flags from `library/cpp/testing/unittest` and flags from gtest.
     * This means that there are usually two parsing passes. The first one parses arguments as recognized
     * by the `library/cpp/testing/unittest`, so things like `--trace-path` and filters. The second one parses flags
     * as recognized by gtest.
     */
    TFlags ParseFlags(int argc, char** argv);

    /**
     * List tests using the unittest style and exit.
     *
     * This function should be called after initializing google tests because test parameters are instantiated
     * during initialization.
     *
     * @param listLevel     verbosity of test list. `0` means don't print anything and don't exit, `1` means print
     *                      test suites, `2` means print individual tests.
     */
    void ListTests(int listLevel, const std::string& listPath);

    /**
     * Remove default result reporter, the one that prints to stdout.
     */
    void UnsetDefaultReporter();

    /**
     * Set trace reporter.
     *
     * Trace files are used by arcadia CI to interact with test runner. They consist of JSON objects, one per line.
     * Each object represents an event, such as 'test started' or 'test finished'.
     *
     * @param traceFile     where to write trace file. This stream should exist for the entire duration of test run.
     */
    void SetTraceReporter(std::ostream* traceFile);
}
