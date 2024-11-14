//===--- taxi_dangling_config_ref_check.h - clang-tidy ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_TAXIDANGLINGCONFIGREFCHECK_H
#define LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_TAXIDANGLINGCONFIGREFCHECK_H

#include <contrib/libs/clang16/tools/extra/clang-tidy/ClangTidyCheck.h>

namespace clang {
namespace tidy {
namespace misc {

class TaxiDanglingConfigRefCheck : public ClangTidyCheck {
public:
  TaxiDanglingConfigRefCheck(StringRef Name, ClangTidyContext *Context)
      : ClangTidyCheck(Name, Context) {}

  void registerMatchers(ast_matchers::MatchFinder *Finder) override;

  void check(const ast_matchers::MatchFinder::MatchResult &Result) override;
};

} // namespace misc
} // namespace tidy
} // namespace clang

#endif // LLVM_CLANG_TOOLS_EXTRA_CLANG_TIDY_MISC_TAXIDANGLINGCONFIGREFCHECK_H
