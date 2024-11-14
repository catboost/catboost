#pragma once

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

#ifndef LLVM_SUPPORT_LOCALE_H
#define LLVM_SUPPORT_LOCALE_H

namespace llvm {
class StringRef;

namespace sys {
namespace locale {

int columnWidth(StringRef s);
bool isPrint(int c);

}
}
}

#endif // LLVM_SUPPORT_LOCALE_H

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
