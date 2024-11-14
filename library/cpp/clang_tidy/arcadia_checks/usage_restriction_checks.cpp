#include "usage_restriction_checks.h"
#include <contrib/libs/clang16/include/clang/AST/ASTContext.h>
#include <contrib/libs/clang16/include/clang/ASTMatchers/ASTMatchFinder.h>

using namespace clang::ast_matchers;

namespace clang::tidy::arcadia {
    void TypeidNameRestrictionCheck::registerMatchers(MatchFinder* Finder) {
        Finder->addMatcher(cxxMemberCallExpr(on(expr(hasType(namedDecl(hasName("::std::type_info")))).bind("expr")),
                                             callee(cxxMethodDecl(hasName("name"), parameterCountIs(0)))),
                           this);
        Finder->addMatcher(cxxMemberCallExpr(on(expr(hasType(namedDecl(hasName("::std::type_index")))).bind("expr")),
                                             callee(cxxMethodDecl(hasName("name"), parameterCountIs(0)))),
                           this);
    }

    void TypeidNameRestrictionCheck::check(const MatchFinder::MatchResult& Result) {
        const auto node = Result.Nodes.getNodeAs<Expr>("expr");

        diag(node->getBeginLoc(), "Both std::type_info::name() and std::type_index::name() return mangled typename. "
                                  "Consider using TypeName() functions from <util/system/type_name.h> instead");
    }

}
