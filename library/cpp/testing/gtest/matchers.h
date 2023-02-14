#pragma once

#include <util/generic/string.h>

#include <gmock/gmock.h>

namespace NGTest {
    namespace NDetail {
        [[nodiscard]] bool MatchOrUpdateGolden(std::string_view actualContent, const TString& goldenFilename);
    }

    /**
     * Matches a string or std::vector<char> equal to the specified file content.
     * The file must be brought to the test using macro DATA in ya.make.
     *
     * The matcher is able to update the file by the actual content during
     * the special test run with the argument '--test-param GTEST_UPDATE_GOLDEN=1'.
     * Any change in such files should be added to VCS manually.
     *
     * The matcher should not be used for a binary data. Use it for a content whose
     * diff will be visual during a code review: text, config, image.
     *
     * Example:
     * ```
     * TEST(Suite, Name) {
     *     std::string data = RenderSomeTextData();
     *     EXPECT_THAT(data, NGTest::GoldenFileEq(SRC_("golden/data.txt")));
     * }
     * ```
     */
    MATCHER_P(GoldenFileEq, filename, "")
    {
        if (!NDetail::MatchOrUpdateGolden(std::string_view(arg.data(), arg.size()), TString(filename))) {
            *result_listener
                << "\nCall `ya m -rA --test-param GTEST_UPDATE_GOLDEN=1` to update the golden file";
            return false;
        }
        return true;
    }
}
