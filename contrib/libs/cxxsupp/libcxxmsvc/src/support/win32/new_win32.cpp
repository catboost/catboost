#include <atomic>
#include <new>

namespace std {

void
__throw_bad_alloc()
{
#ifndef _LIBCPP_NO_EXCEPTIONS
    throw bad_alloc();
#endif
}

static std::atomic<std::new_handler> __new_handler;

new_handler
set_new_handler(new_handler handler) _NOEXCEPT
{
    return __new_handler.exchange(handler);
}

new_handler
get_new_handler() _NOEXCEPT
{
    return __new_handler.load();
}

}