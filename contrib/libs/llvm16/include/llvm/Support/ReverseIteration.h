#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#ifndef LLVM_SUPPORT_REVERSEITERATION_H
#define LLVM_SUPPORT_REVERSEITERATION_H

#include "llvm/Config/abi-breaking.h"
#include "llvm/Support/PointerLikeTypeTraits.h"

namespace llvm {

template<class T = void *>
bool shouldReverseIterate() {
#if LLVM_ENABLE_REVERSE_ITERATION
  return detail::IsPointerLike<T>::value;
#else
  return false;
#endif
}

}
#endif

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
