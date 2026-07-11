#ifndef RSEQ_INL_H_
#error "Direct inclusion of this file is not allowed, include rseq.h"
// For the sake of sane code completion.
#include "rseq.h"
#endif

#include <util/system/compiler.h>

namespace NYT::NRseq {

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE T ReadField(std::ptrdiff_t fieldOffset)
{
    auto* threadPointer = static_cast<char*>(__builtin_thread_pointer());
    // volatile: the kernel updates the area asynchronously, so the read must not be
    // hoisted or cached across loop iterations.
    return *reinterpret_cast<volatile const T*>(threadPointer + fieldOffset);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq
