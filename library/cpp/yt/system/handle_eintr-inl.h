#ifndef HANDLE_EINTR_INL_H_
#error "Direct inclusion of this file is not allowed, include handle_eintr.h"
// For the sake of sane code completion.
#include "handle_eintr.h"
#endif

#include <errno.h>

namespace NYT {

////////////////////////////////////////////////////////////////////////////////

#ifndef _win_
template <class F,  class... Args>
auto HandleEintr(F f, Args&&... args) -> decltype(f(args...))
{
    while (true) {
        auto x = f(args...);
        if (x != -1 || errno != EINTR) {
            return x;
        }
    }
}
#else
template <class F,  class... Args>
auto HandleEintr(F f, Args&&... args) -> decltype(f(args...))
{
     return f(args...);
}
#endif

////////////////////////////////////////////////////////////////////////////////

} // namespace NYT
