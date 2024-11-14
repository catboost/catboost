//===--- taxi_coroutine_unsafe_check.cpp - clang-tidy ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "taxi_coroutine_unsafe_check.h"
#include <contrib/libs/clang16/include/clang/AST/ASTContext.h>
#include <contrib/libs/clang16/include/clang/ASTMatchers/ASTMatchFinder.h>

using namespace clang::ast_matchers;
using namespace clang::ast_matchers::internal;

namespace clang {
namespace tidy {
namespace misc {

namespace {
const auto kFuncall = "funcall";
const auto kCtr = "ctr";
const auto kFieldDecl = "field_decl";
} // namespace

TaxiCoroutineUnsafeCheck::TaxiCoroutineUnsafeCheck(StringRef Name,
                                                   ClangTidyContext *Context)
    : ClangTidyCheck(Name, Context) {}

void TaxiCoroutineUnsafeCheck::registerMatchers(MatchFinder *Finder) {
  const auto UnsafeTypes = anyOf(                      //
      matchesName("^::std::thread"),                   //
      matchesName("^::std::future"),                   //
      matchesName("^::std::condition_variable"),       //
      matchesName("^::std::condition_variable_any"),   //
      matchesName("^::std::shared_mutex"),             //
      matchesName("^::std::mutex"),                    //
      matchesName("^::boost::thread"),                 //
      matchesName("^::boost::future"),                 //
      matchesName("^::boost::condition_variable"),     //
      matchesName("^::boost::condition_variable_any"), //
      matchesName("^::boost::shared_mutex"),           //
      matchesName("^::boost::mutex")                   //
  );
  Finder->addMatcher(
      cxxConstructExpr(hasDeclaration(namedDecl(UnsafeTypes))).bind(kCtr),
      this);

  const auto UnsafeFuncs = anyOf(     //
      matchesName("^::thrd_create$"), //
      matchesName("^::mtx_init$"),    //
      matchesName("^::cnd_init$"),    //
      matchesName("^::pthread_"),     //
      matchesName("^::system$"),      //
      matchesName("^::fork$"),        //
      matchesName("^::sleep$"),       //
      matchesName("^::usleep$")       //
  );

  Finder->addMatcher(callExpr(callee(functionDecl(UnsafeFuncs))).bind(kFuncall),
                     this);

  Finder->addMatcher(
      fieldDecl(hasType(namedDecl(UnsafeTypes))).bind(kFieldDecl), this);
}

void TaxiCoroutineUnsafeCheck::check(const MatchFinder::MatchResult &Result) {
  const auto *Call = Result.Nodes.getNodeAs<CallExpr>(kFuncall);
  if (Call) {
    diag(Call->getBeginLoc(), "function is not coroutine safe")
        << SourceRange(Call->getBeginLoc(), Call->getEndLoc());
  }

  const auto *Decl = Result.Nodes.getNodeAs<CXXConstructExpr>(kCtr);
  if (Decl) {
    diag(Decl->getBeginLoc(), "type is not coroutine safe")
        << SourceRange(Decl->getBeginLoc(), Decl->getEndLoc());
  }

  const auto *FDecl = Result.Nodes.getNodeAs<FieldDecl>(kFieldDecl);
  if (FDecl) {
    diag(FDecl->getBeginLoc(), "type is not coroutine safe")
        << SourceRange(FDecl->getBeginLoc(), FDecl->getEndLoc());
  }
}

} // namespace misc
} // namespace tidy
} // namespace clang
