#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

//===- InstructionNamer.h - Give anonymous instructions names -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UTILS_INSTRUCTIONNAMER_H
#define LLVM_TRANSFORMS_UTILS_INSTRUCTIONNAMER_H

#include "llvm/IR/PassManager.h"

namespace llvm {
struct InstructionNamerPass : PassInfoMixin<InstructionNamerPass> {
  PreservedAnalyses run(Function &, FunctionAnalysisManager &);
};
} // namespace llvm

#endif // LLVM_TRANSFORMS_UTILS_INSTRUCTIONNAMER_H

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
