//===--- taxi_dangling_config_ref_check.cpp - clang-tidy ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "taxi_dangling_config_ref_check.h"
#include <contrib/libs/clang16/include/clang/AST/ASTContext.h>
#include <contrib/libs/clang16/include/clang/ASTMatchers/ASTMatchFinder.h>

using namespace clang::ast_matchers;

namespace clang {
namespace tidy {
namespace misc {

namespace {
const auto kConfig = "config";
}

void TaxiDanglingConfigRefCheck::registerMatchers(MatchFinder *Finder) {
  Finder->addMatcher(
      varDecl(
          hasType(references(cxxRecordDecl())),
          has(exprWithCleanups(has(cxxMemberCallExpr(has(memberExpr(
              member(hasName("Get")),
              has(cxxOperatorCallExpr(
                  has(implicitCastExpr(has(materializeTemporaryExpr(
                      has(cxxBindTemporaryExpr(has(cxxMemberCallExpr(has(
                          memberExpr(member(hasName("Get")))))))))))))))))))))
          .bind("config"),
      this);
}

void TaxiDanglingConfigRefCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *MatchedDecl = Result.Nodes.getNodeAs<VarDecl>("config");
  if (MatchedDecl) {
    diag(MatchedDecl->getBeginLoc(),
         "don't init reference with member of temporary")
        << SourceRange(MatchedDecl->getBeginLoc(), MatchedDecl->getEndLoc());
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
