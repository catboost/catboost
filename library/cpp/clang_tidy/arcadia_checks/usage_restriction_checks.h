#pragma once

#include <contrib/libs/clang16/tools/extra/clang-tidy/ClangTidyCheck.h>

namespace clang::tidy::arcadia {
    /// Finds usage of `typeid(smth).name`
    /// For more info see https://st.yandex-team.ru/IGNIETFERRO-1522
    class TypeidNameRestrictionCheck: public ClangTidyCheck {
    public:
        TypeidNameRestrictionCheck(StringRef Name, ClangTidyContext* Context)
            : ClangTidyCheck(Name, Context)
        {
        }
        void registerMatchers(ast_matchers::MatchFinder* Finder) override;
        void check(const ast_matchers::MatchFinder::MatchResult& Result) override;
    };

}
