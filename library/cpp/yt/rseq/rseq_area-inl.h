#ifndef RSEQ_AREA_INL_H_
#error "Direct inclusion of this file is not allowed, include rseq_area.h"
// For the sake of sane code completion.
#include "rseq_area.h"
#endif

#include <util/system/compiler.h>

namespace NYT::NRseq::NDetail {

////////////////////////////////////////////////////////////////////////////////

template <class T>
Y_FORCE_INLINE T ReadRseqAreaField(std::ptrdiff_t fieldOffset)
{
    auto* threadPointer = static_cast<char*>(__builtin_thread_pointer());
    // volatile: the kernel updates the area asynchronously, so the read must not be
    // hoisted or cached across loop iterations.
    return *reinterpret_cast<volatile const T*>(threadPointer + RseqAreaOffset + fieldOffset);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT::NRseq::NDetail
