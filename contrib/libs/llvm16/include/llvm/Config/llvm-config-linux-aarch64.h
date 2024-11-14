#pragma once

#include "llvm-config-linux.h"

/* Host triple LLVM will be executed on */
#undef LLVM_HOST_TRIPLE
#define LLVM_HOST_TRIPLE "aarch64-unknown-linux-gnu"

/* LLVM architecture name for the native architecture, if available */
#undef LLVM_NATIVE_ARCH
#define LLVM_NATIVE_ARCH AArch64

/* LLVM name for the native AsmParser init function, if available */
#undef LLVM_NATIVE_ASMPARSER
#define LLVM_NATIVE_ASMPARSER LLVMInitializeAArch64AsmParser

/* LLVM name for the native AsmPrinter init function, if available */
#undef LLVM_NATIVE_ASMPRINTER
#define LLVM_NATIVE_ASMPRINTER LLVMInitializeAArch64AsmPrinter

/* LLVM name for the native Disassembler init function, if available */
#undef LLVM_NATIVE_DISASSEMBLER
#define LLVM_NATIVE_DISASSEMBLER LLVMInitializeAArch64Disassembler

/* LLVM name for the native Target init function, if available */
#undef LLVM_NATIVE_TARGET
#define LLVM_NATIVE_TARGET LLVMInitializeAArch64Target

/* LLVM name for the native TargetInfo init function, if available */
#undef LLVM_NATIVE_TARGETINFO
#define LLVM_NATIVE_TARGETINFO LLVMInitializeAArch64TargetInfo

/* LLVM name for the native target MC init function, if available */
#undef LLVM_NATIVE_TARGETMC
#define LLVM_NATIVE_TARGETMC LLVMInitializeAArch64TargetMC
