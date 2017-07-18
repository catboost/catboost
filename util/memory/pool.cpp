
#include "pool.h"

TMemoryPool::IGrowPolicy* TMemoryPool::TLinearGrow::Instance() noexcept {
    return SingletonWithPriority<TLinearGrow, 0>();
}

TMemoryPool::IGrowPolicy* TMemoryPool::TExpGrow::Instance() noexcept {
    return SingletonWithPriority<TExpGrow, 0>();
}
