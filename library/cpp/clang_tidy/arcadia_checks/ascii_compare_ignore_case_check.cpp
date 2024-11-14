#include "ascii_compare_ignore_case_check.h"

#include <contrib/libs/clang16/include/clang/AST/ASTContext.h>
#include <contrib/libs/clang16/include/clang/ASTMatchers/ASTMatchFinder.h>
#include <contrib/libs/clang16/include/clang/Basic/Diagnostic.h>


using namespace clang::ast_matchers;

namespace clang::tidy::arcadia {

    // This check is based on readability-string-compare.

    static const StringRef DiagMessage =
        "do not use 'AsciiCompareIgnoreCase' to test case-insensitive equality "
        "of strings; use 'AsciiEqualsIgnoreCase' instead";

    void AsciiCompareIgnoreCaseCheck::registerMatchers(MatchFinder* finder) {
        const auto compareMatcher = callExpr(
            callee(functionDecl(
                hasName("AsciiCompareIgnoreCase"),
                parameterCountIs(2)
            ))
        );

        // Case 1: AsciiCompareIgnoreCase(...) is casted (maybe implicitly) to boolean.
        finder->addMatcher(
            traverse(
                TK_AsIs,
                // Explicit casts also contain an implicit cast inside
                implicitCastExpr(hasImplicitDestinationType(booleanType()), has(compareMatcher))
                    .bind("match1")),
            this);

        // Case 2: AsciiCompareIgnoreCase == 0 (!= 0)
        finder->addMatcher(
            binaryOperator(
                hasAnyOperatorName("==", "!="),
                hasOperands(
                    compareMatcher.bind("compare"),
                    integerLiteral(equals(0)).bind("zero")))
                .bind("match2"),
            this);
    }

    void AsciiCompareIgnoreCaseCheck::check(const MatchFinder::MatchResult& result) {
        if (const auto* matched = result.Nodes.getNodeAs<Stmt>("match1")) {
            diag(matched->getBeginLoc(), DiagMessage);
        } else if (const auto* matched = result.Nodes.getNodeAs<Stmt>("match2")) {
            const auto* op = cast<BinaryOperator>(matched);
            const auto* compareCall = result.Nodes.getNodeAs<CallExpr>("compare");
            const auto* zero = result.Nodes.getNodeAs<Stmt>("zero");

            const ASTContext &ctx = *result.Context;
            auto diagBuilder = diag(matched->getBeginLoc(), DiagMessage);
            if (op->getOpcode() == BinaryOperatorKind::BO_NE) {
                diagBuilder << FixItHint::CreateInsertion(compareCall->getCallee()->getBeginLoc(), "!");
            }
            diagBuilder << FixItHint::CreateReplacement(compareCall->getCallee()->getSourceRange(), "AsciiEqualsIgnoreCase");
            diagBuilder << FixItHint::CreateRemoval(op->getOperatorLoc());
            diagBuilder << FixItHint::CreateRemoval(zero->getSourceRange());
        }
    }

}  // namespace clang::tidy::arcadia
