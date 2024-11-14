#pragma once

#include <contrib/libs/clang16/tools/extra/clang-tidy/ClangTidyCheck.h>

namespace clang::tidy::arcadia {
    /// Finds uses of AsciiCompareIgnoreCase that can be replaced with AsciiEqualsIgnoreCase.
    class AsciiCompareIgnoreCaseCheck : public ClangTidyCheck {
    public:
        AsciiCompareIgnoreCaseCheck(StringRef name, ClangTidyContext* context)
            : ClangTidyCheck(name, context)
        {}

        bool isLanguageVersionSupported(const LangOptions& langOpts) const override {
            return langOpts.CPlusPlus;
        }

        void registerMatchers(ast_matchers::MatchFinder* finder) override;
        void check(const ast_matchers::MatchFinder::MatchResult& result) override;
    };

}  // namespace clang::tidy::arcadia
