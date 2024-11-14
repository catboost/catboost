#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

//===--- TextAPIWriter.h - Text API Writer ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TEXTAPI_TEXTAPIWRITER_H
#define LLVM_TEXTAPI_TEXTAPIWRITER_H

namespace llvm {

class Error;
class raw_ostream;

namespace MachO {

class InterfaceFile;

class TextAPIWriter {
public:
  TextAPIWriter() = delete;

  static Error writeToStream(raw_ostream &os, const InterfaceFile &);
};

} // end namespace MachO.
} // end namespace llvm.

#endif // LLVM_TEXTAPI_TEXTAPIWRITER_H

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
