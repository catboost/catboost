#ifndef TLS_INL_H_
#error "Direct inclusion of this file is not allowed, include tls.h"
// For the sake of sane code completion.
#include "tls.h"
#endif

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

template <class T>
T& GetTlsRef(volatile T& arg)
{
    return const_cast<T&>(arg);
}
////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
