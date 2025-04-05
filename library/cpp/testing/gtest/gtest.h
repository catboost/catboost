#pragma once

// IWYU pragma: begin_exports
#include <library/cpp/testing/gtest/matchers.h>

#include <library/cpp/testing/gtest_extensions/gtest_extensions.h>

#include <gtest/gtest.h>
#include <gmock/gmock.h>
// IWYU pragma: end_exports

#include <optional>
#include <string_view>

/**
 * Bridge between GTest framework and Arcadia CI.
 */
namespace NGTest {
    /**
     * Get custom test parameter.
     *
     * You can pass custom parameters to your test using the `--test-param` flag:
     *
     * ```
     * $ ya make -t --test-param NAME=VALUE
     * ```
     *
     * You can later access these parameters from tests using this function:
     *
     * ```
     * TEST(Suite, Name) {
     *     EXPECT_EQ(GetTestParam("NAME").value_or("NOT_SET"), "VALUE");
     * }
     * ```
     *
     * @param name          name of the parameter.
     * @return              value of the parameter, as passed to the program arguments,
     *                      or nothing, if parameter not found.
     */
    std::optional<std::string_view> GetTestParam(std::string_view name);
}
