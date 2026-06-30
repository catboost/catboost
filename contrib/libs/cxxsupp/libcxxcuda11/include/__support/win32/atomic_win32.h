#ifndef _LIBCPP_ATOMIC_WIN32
#define _LIBCPP_ATOMIC_WIN32

#include <__config>
#include <type_traits>
#include <intrin.h>

_LIBCPP_BEGIN_NAMESPACE_STD

#define _Atomic(x) x

#ifdef _M_ARM
#define _MemoryBarrier __dmb(_ARM_BARRIER_ISH)
#endif

#ifdef _M_ARM64
#define _MemoryBarrier __dmb(_ARM64_BARRIER_ISH)
#endif

#ifdef _M_X64
#define _MemoryBarrier __faststorefence()
#endif

#ifdef _M_IX86
#define _MemoryBarrier _mm_mfence()
#endif

namespace __atomic {
template <typename _Tp> _Tp __create();

template <typename _Tp, typename _Td>
enable_if_t<sizeof(__create<_Tp>() = __create<_Td>()), char>
    __test_atomic_assignable(int);
template <typename _Tp, typename _Up>
__two __test_atomic_assignable(...);

template <typename _Tp, typename _Td>
struct __can_assign {
  static const bool value =
      sizeof(__test_atomic_assignable<_Tp, _Td>(1)) == sizeof(char);
};
}  // namespace __atomic

template <typename _Tp>
static inline enable_if_t<is_assignable<volatile _Tp, _Tp>::value>
__c11_atomic_init(volatile _Atomic(_Tp)* __a,  _Tp __val) {
    *__a = __val;
}

template <typename _Tp>
static inline enable_if_t<!is_assignable<volatile _Tp, _Tp>::value>
__c11_atomic_init(volatile _Atomic(_Tp)* __a,  _Tp __val) {
  volatile char* to = reinterpret_cast<volatile char*>(__a);
  volatile char* end = to + sizeof(_Tp);
  char* from = reinterpret_cast<char*>(&__val);
  while (to != end) {
    *to++ = *from++;
  }
}

template <typename _Tp>
static inline void __c11_atomic_init(_Atomic(_Tp)* __a,  _Tp __val) {
    *__a = __val;
}

static inline void __c11_atomic_thread_fence(int __order) {
    if (__order != static_cast<int>(memory_order_relaxed)) {
#if defined(_M_IX86) || defined(_M_X64)
        if (__order == static_cast<int>(memory_order_seq_cst)) {
            _MemoryBarrier;
        } else {
            _ReadWriteBarrier();
        }
#else // ARM
        _MemoryBarrier;
#endif
    }
}

static inline void __c11_atomic_signal_fence(int /*__order*/) {
    _ReadWriteBarrier();
}

void __msvc_lock(void* p);
void __msvc_unlock(void* p);

template<class _Out, class _Tp>
static inline _Out __msvc_cast(_Tp __val) {
    _Out __result;
    volatile char* to = reinterpret_cast<volatile char*>(&__result);
    volatile char* end = to + sizeof(_Tp);
    char* from = reinterpret_cast<char*>(&__val);
    while (to != end) {
      *to++ = *from++;
    }
    return __result;
}



static inline void __msvc_atomic_store8(volatile char* __a, char __val,
                                        memory_order __order) {
    if (__order == memory_order_relaxed) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __iso_volatile_store8(__a, __val);
#else
        *__a = __val;
#endif
    } else if (__order == memory_order_release) {
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __iso_volatile_store8(__a, __val);
#else
        _ReadWriteBarrier();
        *__a = __val;
#endif
    } else { // __order == memory_order_seq_cst)
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __iso_volatile_store8(__a, __val);
        _MemoryBarrier;
#else
        _InterlockedExchange8(__a, __val);
#endif
    }
}

static inline void __msvc_atomic_store16(volatile short* __a, short __val,
                                         memory_order __order) {
    if (__order == memory_order_relaxed) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __iso_volatile_store16(__a, __val);
#else
        *__a = __val;
#endif
    } else if (__order == memory_order_release) {
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __iso_volatile_store16(__a, __val);
#else
        _ReadWriteBarrier();
        *__a = __val;
#endif
    } else { // __order == memory_order_seq_cst)
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __iso_volatile_store16(__a, __val);
        _MemoryBarrier;
#else
        _InterlockedExchange16(__a, __val);
#endif
    }
}

static inline void __msvc_atomic_store32(volatile long* __a, long __val,
                                         memory_order __order) {
    if (__order == memory_order_relaxed) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __iso_volatile_store32(__a, __val);
#else
        *__a = __val;
#endif
    } else if (__order == memory_order_release) {
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __iso_volatile_store32(__a, __val);
#else
        _ReadWriteBarrier();
        *__a = __val;
#endif
    } else { // __order == memory_order_seq_cst)
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __iso_volatile_store32(__a, __val);
        _MemoryBarrier;
#else
        _InterlockedExchange(__a, __val);
#endif
    }
}

static inline void __msvc_atomic_store64(volatile __int64* __a, __int64 __val,
                                         memory_order __order) {
#if defined(_M_IX86)
    __int64 __tmp;
    do {
        __tmp = *__a;
    } while (__tmp != _InterlockedCompareExchange64(__a, __val, __tmp));
#else
    if (__order == memory_order_relaxed) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __iso_volatile_store64(__a, __val);
#else
        *__a = __val;
#endif
    } else if (__order == memory_order_release) {
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __iso_volatile_store64(__a, __val);
#else
        _ReadWriteBarrier();
        *__a = __val;
#endif
    } else { // __order == memory_order_seq_cst)
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __iso_volatile_store64(__a, __val);
        _MemoryBarrier;
#else
        _InterlockedExchange64(__a, __val);
#endif
    }
#endif
}

template <typename _Tp>
static inline void __c11_atomic_store(volatile _Atomic(_Tp)* __a,  _Tp __val,
                                      int __order) {
    if (sizeof(_Tp) == 1) {
        __msvc_atomic_store8((volatile char*)__a, __msvc_cast<char>(__val), (memory_order)__order);
    } else if (sizeof(_Tp) == 2 && alignof(_Tp) % 2 == 0) {
        __msvc_atomic_store16((volatile short*)__a, __msvc_cast<short>(__val), (memory_order)__order);
    } else if (sizeof(_Tp) == 4 && alignof(_Tp) % 4 == 0) {
        __msvc_atomic_store32((volatile long*)__a, __msvc_cast<long>(__val), (memory_order)__order);
    } else if (sizeof(_Tp) == 8 && alignof(_Tp) % 8 == 0) {
        __msvc_atomic_store64((volatile __int64*)__a, __msvc_cast<__int64>(__val), (memory_order)__order);
    } else {
        __msvc_lock((void*)__a);
        *(_Atomic(_Tp)*)__a = __val;
        __msvc_unlock((void*)__a);
    }
}

template<typename _Tp>
static inline void __c11_atomic_store(_Atomic(_Tp)* __a, _Tp __val, int __order) {
    __c11_atomic_store((volatile _Atomic(_Tp)*)__a, __val, __order);
}

static inline char __msvc_atomic_load8(volatile char* __a, memory_order __order) {
    char __result;
    if (__order == memory_order_relaxed) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __result = __iso_volatile_load8(__a);
#else
        __result = *__a;
#endif
    } else if (__order == memory_order_acquire ||
               __order == memory_order_consume) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __result = __iso_volatile_load8(__a);
        _MemoryBarrier;
#else
        __result = *__a;
        _ReadWriteBarrier();
#endif
    } else { // __order == memory_order_seq_cst
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __result = __iso_volatile_load8(__a);
        _MemoryBarrier;
#else
        _ReadWriteBarrier();
        __result = *__a;
        _ReadWriteBarrier();
#endif
    }
    return __result;
}

static inline short __msvc_atomic_load16(volatile short* __a, memory_order __order) {
    short __result;
    if (__order == memory_order_relaxed) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __result = __iso_volatile_load16(__a);
#else
        __result = *__a;
#endif
    } else if (__order == memory_order_acquire ||
               __order == memory_order_consume) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __result = __iso_volatile_load16(__a);
        _MemoryBarrier;
#else
        __result = *__a;
        _ReadWriteBarrier();
#endif
    } else { // __order == memory_order_seq_cst
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __result = __iso_volatile_load16(__a);
        _MemoryBarrier;
#else
        _ReadWriteBarrier();
        __result = *__a;
        _ReadWriteBarrier();
#endif
    }
    return __result;
}

static inline long __msvc_atomic_load32(volatile long* __a, memory_order __order) {
    long __result;
    if (__order == memory_order_relaxed) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __result = __iso_volatile_load32(__a);
#else
        __result = *__a;
#endif
    } else if (__order == memory_order_acquire ||
               __order == memory_order_consume) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __result = __iso_volatile_load32(__a);
        _MemoryBarrier;
#else
        __result = *__a;
        _ReadWriteBarrier();
#endif
    } else { // __order == memory_order_seq_cst
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __result = __iso_volatile_load32(__a);
        _MemoryBarrier;
#else
        _ReadWriteBarrier();
        __result = *__a;
        _ReadWriteBarrier();
#endif
    }
    return __result;
}

static inline __int64 __msvc_atomic_load64(volatile __int64* __a, memory_order __order) {
    __int64 __result;
#if defined(_M_X86)
    do {
        __result = *__a;
    } while (__result != _InterlockedCompareExchange64(__a, __result, __result));
#else
    if (__order == memory_order_relaxed) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __result = __iso_volatile_load64(__a);
#else
        __result = *__a;
#endif
    } else if (__order == memory_order_acquire ||
               __order == memory_order_consume) {
#if defined(_M_ARM) || defined(_M_ARM64)
        __result = __iso_volatile_load64(__a);
        _MemoryBarrier;
#else
        __result = *__a;
        _ReadWriteBarrier();
#endif
    } else { // __order == memory_order_seq_cst
#if defined(_M_ARM) || defined(_M_ARM64)
        _MemoryBarrier;
        __result = __iso_volatile_load64(__a);
        _MemoryBarrier;
#else
        _ReadWriteBarrier();
        __result = *__a;
        _ReadWriteBarrier();
#endif
    }
#endif
    return __result;
}

template<typename _Tp>
static inline _Tp __c11_atomic_load(volatile _Atomic(_Tp)* __a, int __order) {
    _Tp __result;
    if (sizeof(_Tp) == 1) {
        __result = __msvc_cast<_Tp>(__msvc_atomic_load8((volatile char*)__a, (memory_order)__order));
    } else if (sizeof(_Tp) == 2 && alignof(_Tp) % 2 == 0) {
        __result = __msvc_cast<_Tp>(__msvc_atomic_load16((volatile short*)__a, (memory_order)__order));
    } else if (sizeof(_Tp) == 4 && alignof(_Tp) % 4 == 0) {
        __result = __msvc_cast<_Tp>(__msvc_atomic_load32((volatile long*)__a, (memory_order)__order));
    } else if (sizeof(_Tp) == 8 && alignof(_Tp) % 8 == 0) {
        __result = __msvc_cast<_Tp>(__msvc_atomic_load64((volatile __int64*)__a, (memory_order)__order));
    } else {
        __msvc_lock((void*)__a);
        __result = *(_Atomic(_Tp)*)__a;
        __msvc_unlock((void*)__a);
    }
    return __result;
}

template<typename _Tp>
static inline _Tp __c11_atomic_load(_Atomic(_Tp)* __a, int __order) {
    return __c11_atomic_load((volatile _Atomic(_Tp)*)__a, __order);
}

static inline char __msvc_atomic_exchange8(volatile char* __a, char __val,
                                           memory_order __order) {
    char __result;
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        __result = _InterlockedExchange8_nf(__a, __val);
    } else if (__order == memory_order_acquire ||
               __order == memory_order_consume) {
        __result = _InterlockedExchange8_acq(__a, __val);
    } else if (__order == memory_order_release) {
        __result = _InterlockedExchange8_rel(__a, __val);
    } else {
        __result = _InterlockedExchange8(__a, __val);
    }
#else
  (void)__order;
  __result = _InterlockedExchange8(__a, __val);
#endif
  return __result;
}

static inline short __msvc_atomic_exchange16(volatile short* __a, short __val,
                                             memory_order __order) {
    short __result;
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        __result = _InterlockedExchange16_nf(__a, __val);
    } else if (__order == memory_order_acquire ||
               __order == memory_order_consume) {
        __result = _InterlockedExchange16_acq(__a, __val);
    } else if (__order == memory_order_release) {
        __result = _InterlockedExchange16_rel(__a, __val);
    } else {
        __result = _InterlockedExchange16(__a, __val);
    }
#else
  (void)__order;
  __result = _InterlockedExchange16(__a, __val);
#endif
  return __result;
}

static inline long __msvc_atomic_exchange32(volatile long* __a, long __val,
                                            memory_order __order) {
    long __result;
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        __result = _InterlockedExchange_nf(__a, __val);
    } else if (__order == memory_order_acquire ||
               __order == memory_order_consume) {
        __result = _InterlockedExchange_acq(__a, __val);
    } else if (__order == memory_order_release) {
        __result = _InterlockedExchange_rel(__a, __val);
    } else {
        __result = _InterlockedExchange(__a, __val);
    }
#else
  (void)__order;
  __result = _InterlockedExchange(__a, __val);
#endif
  return __result;
}

static inline __int64 __msvc_atomic_exchange64(volatile __int64* __a, __int64 __val,
                                            memory_order __order) {
    __int64 __result;
#if defined(_M_IX86)
    do {
        __result = *__a;
    } while (__result != _InterlockedCompareExchange64(__a, __val, __result));
#elif defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        __result = _InterlockedExchange64_nf(__a, __val);
    } else if (__order == memory_order_acquire ||
               __order == memory_order_consume) {
        __result = _InterlockedExchange64_acq(__a, __val);
    } else if (__order == memory_order_release) {
        __result = _InterlockedExchange64_rel(__a, __val);
    } else {
        __result = _InterlockedExchange64(__a, __val);
    }
#else
  (void)__order;
  __result = _InterlockedExchange64(__a, __val);
#endif
  return __result;
}

template<typename _Tp>
static inline _Tp __c11_atomic_exchange(volatile _Atomic(_Tp)* __a, _Tp __val,
                                        int __order) {
    _Tp __result;
    if (sizeof(_Tp) == 1) {
        __result = __msvc_cast<_Tp>(
            __msvc_atomic_exchange8((volatile char*)__a, __msvc_cast<char>(__val), (memory_order)__order));
    } else if (sizeof(_Tp) == 2 && alignof(_Tp) % 2 == 0) {
        __result = __msvc_cast<_Tp>(
            __msvc_atomic_exchange16((volatile short*)__a, __msvc_cast<short>(__val), (memory_order)__order));
    } else if (sizeof(_Tp) == 4 && alignof(_Tp) % 4 == 0) {
        __result = __msvc_cast<_Tp>(
            __msvc_atomic_exchange32((volatile long*)__a, __msvc_cast<long>(__val), (memory_order)__order));
    } else if (sizeof(_Tp) == 8 && alignof(_Tp) % 8 == 0) {
        __result = __msvc_cast<_Tp>(
            __msvc_atomic_exchange64((volatile __int64*)__a, __msvc_cast<__int64>(__val), (memory_order)__order));
    } else {
        __msvc_lock((void*)__a);
        __result = *(_Atomic(_Tp)*)__a;
        *(_Atomic(_Tp)*)__a = __val;
        __msvc_unlock((void*)__a);
    }
    return __result;
}

template<typename _Tp>
static inline _Tp __c11_atomic_exchange(_Atomic(_Tp)* __a, _Tp __val,
                                        int __order) {
    return __c11_atomic_exchange((volatile _Atomic(_Tp)*)__a, __val, __order);
}

static inline bool __msvc_atomic_compare_exchange8(volatile char* __a, char __value, char* __expected,
                                                   memory_order __order) {
    char __compare = *__expected;
    char __before;
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        __before = _InterlockedCompareExchange8_nf(__a, __value, __compare);
    } else if (__order == memory_order_acquire) {
        __before = _InterlockedCompareExchange8_acq(__a, __value, __compare);
    } else if (__order == memory_order_release) {
        __before = _InterlockedCompareExchange8_rel(__a, __value, __compare);
    } else {
        __before = _InterlockedCompareExchange8(__a, __value, __compare);
    }
#else
    (void)__order;
    __before = _InterlockedCompareExchange8(__a, __value, __compare);
#endif
    if (__before == __compare) {
        return true;
    }
    *__expected = __before;
    return false;
}

static inline bool __msvc_atomic_compare_exchange16(volatile short* __a, short __value, short* __expected,
                                                    memory_order __order) {
    short __compare = *__expected;
    short __before;
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        __before = _InterlockedCompareExchange16_nf(__a, __value, __compare);
    } else if (__order == memory_order_acquire) {
        __before = _InterlockedCompareExchange16_acq(__a, __value, __compare);
    } else if (__order == memory_order_release) {
        __before = _InterlockedCompareExchange16_rel(__a, __value, __compare);
    } else {
        __before = _InterlockedCompareExchange16(__a, __value, __compare);
    }
#else
    (void)__order;
    __before = _InterlockedCompareExchange16(__a, __value, __compare);
#endif
    if (__before == __compare) {
        return true;
    }
    *__expected = __before;
    return false;
}

static inline bool __msvc_atomic_compare_exchange32(volatile long* __a, long __value, long* __expected,
                                                    memory_order __order) {
    long __compare = *__expected;
    long __before;
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        __before = _InterlockedCompareExchange_nf(__a, __value, __compare);
    } else if (__order == memory_order_acquire) {
        __before = _InterlockedCompareExchange_acq(__a, __value, __compare);
    } else if (__order == memory_order_release) {
        __before = _InterlockedCompareExchange_rel(__a, __value, __compare);
    } else {
        __before = _InterlockedCompareExchange(__a, __value, __compare);
    }
#else
    (void)__order;
    __before = _InterlockedCompareExchange(__a, __value, __compare);
#endif
    if (__before == __compare) {
        return true;
    }
    *__expected = __before;
    return false;
}

static inline bool __msvc_atomic_compare_exchange64(volatile __int64* __a, __int64 __value, __int64* __expected,
                                                    memory_order __order) {
    __int64 __compare = *__expected;
    __int64 __before;
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        __before = _InterlockedCompareExchange64_nf(__a, __value, __compare);
    } else if (__order == memory_order_acquire) {
        __before = _InterlockedCompareExchange64_acq(__a, __value, __compare);
    } else if (__order == memory_order_release) {
        __before = _InterlockedCompareExchange64_rel(__a, __value, __compare);
    } else {
        __before = _InterlockedCompareExchange64(__a, __value, __compare);
    }
#else
    (void)__order;
    __before = _InterlockedCompareExchange64(__a, __value, __compare);
#endif
    if (__before == __compare) {
        return true;
    }
    *__expected = __before;
    return false;
}


static inline memory_order constexpr __msvc_top_memory_order(memory_order __order1, memory_order __order2) {
    return
        (__order1 == memory_order_relaxed && __order2 == memory_order_relaxed) ? memory_order_relaxed :
        ((__order1 == memory_order_relaxed || __order1 == memory_order_acquire || __order1 == memory_order_consume) &&
         (__order2 == memory_order_relaxed || __order2 == memory_order_relaxed || __order2 == memory_order_consume)) ? memory_order_acquire :
        ((__order1 == memory_order_relaxed || __order1 == memory_order_release) &&
         (__order2 == memory_order_relaxed || __order2 == memory_order_release)) ? memory_order_release :
        (__order1 != memory_order_seq_cst && __order2 != memory_order_seq_cst) ? memory_order_acq_rel :
        memory_order_seq_cst;
}

template<typename _Tp>
static inline bool __c11_atomic_compare_exchange_strong(
    volatile _Atomic(_Tp)* __a, _Tp* __expected, _Tp __value,
    int __order_success, int __order_failure) {
    memory_order __order = __msvc_top_memory_order((memory_order)__order_success, (memory_order)__order_failure);
    if (sizeof(_Tp) == 1) {
        return __msvc_atomic_compare_exchange8((volatile char*)__a, __msvc_cast<char>(__value), (char*)__expected, __order);
    } else if (sizeof(_Tp) == 2 && alignof(_Tp) % 2 == 0) {
        return __msvc_atomic_compare_exchange16((volatile short*)__a, __msvc_cast<short>(__value), (short*)__expected, __order);
    } else if (sizeof(_Tp) == 4 && alignof(_Tp) % 4 == 0) {
        return __msvc_atomic_compare_exchange32((volatile long*)__a, __msvc_cast<long>(__value), (long*)__expected, __order);
    } else if (sizeof(_Tp) == 8 && alignof(_Tp) % 8 == 0) {
        return __msvc_atomic_compare_exchange64((volatile __int64*)__a, __msvc_cast<__int64>(__value), (__int64*)__expected, __order);
    } else {
        bool __result;
        __msvc_lock((void*)__a);
        volatile char* __p_a = reinterpret_cast<volatile char*>(__a);
        volatile char* __p_a_end = __p_a + sizeof(_Atomic(_Tp));
        volatile char* __p_expected = reinterpret_cast<volatile char*>(__expected);
        bool __equal = true;
        while (__p_a != __p_a_end) {
            if (*__p_a++ != *__p_expected++) {
                __equal = false;
                break;
            }
        }
        if (__equal) {
            *(_Atomic(_Tp)*)__a = __value;
            __result = true;
        } else {
            *__expected = *(_Atomic(_Tp)*)__a;
            __result = false;
        }
        __msvc_unlock((void*)__a);
        return __result;
    }
}

template<typename _Tp>
static inline bool __c11_atomic_compare_exchange_strong(
    _Atomic(_Tp)* __a, _Tp* __expected, _Tp __value,
    int __order_success, int __order_failure) {
    return __c11_atomic_compare_exchange_strong(
        (volatile _Atomic(_Tp)*)__a, __expected, __value, __order_success, __order_failure);
}

template<typename _Tp>
static inline bool __c11_atomic_compare_exchange_weak(
    volatile _Atomic(_Tp)* __a, _Tp* __expected, _Tp __value,
    int __order_success, int __order_failure) {
    return __c11_atomic_compare_exchange_strong(__a, __expected, __value, __order_success, __order_failure);
}

template<typename _Tp>
static inline bool __c11_atomic_compare_exchange_weak(
    _Atomic(_Tp)* __a, _Tp* __expected, _Tp __value,
    int __order_success, int __order_failure) {
    return __c11_atomic_compare_exchange_strong(__a, __expected, __value, __order_success, __order_failure);
}

template <typename _Tp>
struct __msvc_skip { enum {value = 1}; };

template <typename _Tp>
struct __msvc_skip<_Tp*> { enum {value = sizeof(_Tp)}; };

// FIXME: Haven't figured out what the spec says about using arrays with
// atomic_fetch_add. Force a failure rather than creating bad behavior.
template <typename _Tp>
struct __msvc_skip<_Tp[]> { };
template <typename _Tp, int n>
struct __msvc_skip<_Tp[n]> { };

static inline char __msvc_atomic_fetch_add8(volatile char* __a, char __delta,
                                            memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedExchangeAdd8_nf(__a, __delta);
    } else if (__order == memory_order_acquire) {
        return _InterlockedExchangeAdd8_acq(__a, __delta);
    } else if (__order == memory_order_release) {
        return _InterlockedExchangeAdd8_rel(__a, __delta);
    } else {
        return _InterlockedExchangeAdd8(__a, __delta);
    }
#else
    (void)__order;
    return _InterlockedExchangeAdd8(__a, __delta);
#endif
}

static inline short __msvc_atomic_fetch_add16(volatile short* __a, short __delta,
                                              memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedExchangeAdd16_nf(__a, __delta);
    } else if (__order == memory_order_acquire) {
        return _InterlockedExchangeAdd16_acq(__a, __delta);
    } else if (__order == memory_order_release) {
        return _InterlockedExchangeAdd16_rel(__a, __delta);
    } else {
        return _InterlockedExchangeAdd16(__a, __delta);
    }
#else
    (void)__order;
    return _InterlockedExchangeAdd16(__a, __delta);
#endif
}

static inline long __msvc_atomic_fetch_add32(volatile long* __a, long __delta,
                                             memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedExchangeAdd_nf(__a, __delta);
    } else if (__order == memory_order_acquire) {
        return _InterlockedExchangeAdd_acq(__a, __delta);
    } else if (__order == memory_order_release) {
        return _InterlockedExchangeAdd_rel(__a, __delta);
    } else {
        return _InterlockedExchangeAdd(__a, __delta);
    }
#else
    (void)__order;
    return _InterlockedExchangeAdd(__a, __delta);
#endif
}

static inline __int64 __msvc_atomic_fetch_add64(volatile __int64* __a, __int64 __delta,
                                                memory_order __order) {
#if defined(_M_IX86)
    __int64 __tmp;
    do {
        __tmp = *__a;
    } while (__tmp != _InterlockedCompareExchange64(__a, __tmp + __delta, __tmp));
    return __tmp;
#elif defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedExchangeAdd64_nf(__a, __delta);
    } else if (__order == memory_order_acquire) {
        return _InterlockedExchangeAdd64_acq(__a, __delta);
    } else if (__order == memory_order_release) {
        return _InterlockedExchangeAdd64_rel(__a, __delta);
    } else {
        return _InterlockedExchangeAdd64(__a, __delta);
    }
#else
    (void)__order;
    return _InterlockedExchangeAdd64(__a, __delta);
#endif
}

template <typename _Tp, typename _Td>
static inline _Tp __c11_atomic_fetch_add(volatile _Atomic(_Tp)* __a,
                                         _Td __delta, int __order) {
    _Td __real_delta = __delta * __msvc_skip<_Tp>::value;
    if (sizeof(_Tp) == 1 && std::is_integral<_Tp>::value) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_add8((volatile char*)__a, (char)__real_delta, (memory_order)__order));
    } else if (sizeof(_Tp) == 2 && std::is_integral<_Tp>::value) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_add16((volatile short*)__a, (short)__real_delta, (memory_order)__order));
    } else if (sizeof(_Tp) == 4 && (std::is_integral<_Tp>::value || std::is_pointer<_Tp>::value)) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_add32((volatile long*)__a, (long)__real_delta, (memory_order)__order));
    } else if (sizeof(_Tp) == 8 && (std::is_integral<_Tp>::value || std::is_pointer<_Tp>::value)) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_add64((volatile __int64*)__a, (__int64)__real_delta, (memory_order)__order));
    } else {
        __msvc_lock((void*)__a);
        _Tp __result = *(_Atomic(_Tp)*)__a;
        *(_Atomic(_Tp)*)__a += __delta;
        __msvc_unlock((void*)__a);
        return __result;
    }
}

template <typename _Tp, typename _Td>
static inline _Tp __c11_atomic_fetch_add(_Atomic(_Tp)* __a,
                                         _Td __delta, int __order) {
    return __c11_atomic_fetch_add((volatile _Atomic(_Tp)*) __a, __delta, __order);
}

template <typename _Tp, typename _Td>
static inline _Tp __c11_atomic_fetch_sub(volatile _Atomic(_Tp)* __a,
                                         _Td __delta, int __order) {
    _Td __real_delta = __delta * __msvc_skip<_Tp>::value;
    // Cast __real_delta to unsigned types to avoid integer overflow on negation.
    if (sizeof(_Tp) == 1 && std::is_integral<_Tp>::value) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_add8((volatile char*)__a, -(unsigned char)__real_delta, (memory_order)__order));
    } else if (sizeof(_Tp) == 2 && std::is_integral<_Tp>::value) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_add16((volatile short*)__a, -(unsigned short)__real_delta, (memory_order)__order));
    } else if (sizeof(_Tp) == 4 && (std::is_integral<_Tp>::value || std::is_pointer<_Tp>::value)) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_add32((volatile long*)__a, -(unsigned long)__real_delta, (memory_order)__order));
    } else if (sizeof(_Tp) == 8 && (std::is_integral<_Tp>::value || std::is_pointer<_Tp>::value)) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_add64((volatile __int64*)__a, -(unsigned __int64)__real_delta, (memory_order)__order));
    } else {
        __msvc_lock((void*)__a);
        _Tp __result = *(_Atomic(_Tp)*)__a;
        *(_Atomic(_Tp)*)__a -= __delta;
        __msvc_unlock((void*)__a);
        return __result;
    }
}

template <typename _Tp, typename _Td>
static inline _Tp __c11_atomic_fetch_sub(_Atomic(_Tp)* __a,
                                         _Td __delta, int __order) {
    return __c11_atomic_fetch_sub((volatile _Atomic(_Tp)*) __a, __delta, __order);
}

static inline char __msvc_atomic_fetch_and8(volatile char* __a, char __value,
                                            memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedAnd8_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedAnd8_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedAnd8_rel(__a, __value);
    } else {
        return _InterlockedAnd8(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedAnd8(__a, __value);
#endif
}

static inline short __msvc_atomic_fetch_and16(volatile short* __a, short __value,
                                              memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedAnd16_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedAnd16_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedAnd16_rel(__a, __value);
    } else {
        return _InterlockedAnd16(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedAnd16(__a, __value);
#endif
}

static inline long __msvc_atomic_fetch_and32(volatile long* __a, long __value,
                                             memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedAnd_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedAnd_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedAnd_rel(__a, __value);
    } else {
        return _InterlockedAnd(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedAnd(__a, __value);
#endif
}

static inline __int64 __msvc_atomic_fetch_and64(volatile __int64* __a, __int64 __value,
                                                memory_order __order) {
#if defined(_M_IX86)
    __int64 __tmp;
    do {
        __tmp = *__a;
    } while (__tmp != _InterlockedCompareExchange64(__a, __tmp & __value, __tmp));
    return __tmp;
#elif defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedAnd64_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedAnd64_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedAnd64_rel(__a, __value);
    } else {
        return _InterlockedAnd64(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedAnd64(__a, __value);
#endif
}

template <typename _Tp>
static inline _Tp __c11_atomic_fetch_and(volatile _Atomic(_Tp)* __a,
                                         _Tp __value, int __order) {
    if (sizeof(_Tp) == 1 && (std::is_integral<_Tp>::value || std::is_same<std::remove_cv_t<_Tp>, bool>::value)) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_and8((volatile char*)__a, __msvc_cast<char>(__value), (memory_order)__order));
    } else if (sizeof(_Tp) == 2 && std::is_integral<_Tp>::value) {
        return (_Tp)__msvc_atomic_fetch_and16((volatile short*)__a, (short)__value, (memory_order)__order);
    } else if (sizeof(_Tp) == 4 && std::is_integral<_Tp>::value) {
        return (_Tp) __msvc_atomic_fetch_and32((volatile long*)__a, (long)__value, (memory_order)__order);
    } else if (sizeof(_Tp) == 8 && std::is_integral<_Tp>::value) {
        return (_Tp) __msvc_atomic_fetch_and64((volatile __int64*)__a, (__int64)__value, (memory_order)__order);
    } else {
        __msvc_lock((void*)__a);
        _Tp __result = *(_Atomic(_Tp)*)__a;
        *(_Atomic(_Tp)*)__a &= __value;
        __msvc_unlock((void*)__a);
        return __result;
    }
}

template <typename _Tp>
static inline _Tp __c11_atomic_fetch_and(_Atomic(_Tp)* __a,
                                         _Tp __value, int __order) {
    return __c11_atomic_fetch_and((volatile _Atomic(_Tp)*)__a, __value, __order);
}

static inline char __msvc_atomic_fetch_or8(volatile char* __a, char __value,
                                           memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedOr8_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedOr8_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedOr8_rel(__a, __value);
    } else {
        return _InterlockedOr8(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedOr8(__a, __value);
#endif
}

static inline short __msvc_atomic_fetch_or16(volatile short* __a, short __value,
                                             memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedOr16_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedOr16_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedOr16_rel(__a, __value);
    } else {
        return _InterlockedOr16(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedOr16(__a, __value);
#endif
}

static inline long __msvc_atomic_fetch_or32(volatile long* __a, long __value,
                                            memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedOr_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedOr_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedOr_rel(__a, __value);
    } else {
        return _InterlockedOr(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedOr(__a, __value);
#endif
}

static inline __int64 __msvc_atomic_fetch_or64(volatile __int64* __a, __int64 __value,
                                               memory_order __order) {
#if defined(_M_IX86)
    __int64 __tmp;
    do {
        __tmp = *__a;
    } while (__tmp != _InterlockedCompareExchange64(__a, __tmp | __value, __tmp));
    return __tmp;
#elif defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedOr64_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedOr64_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedOr64_rel(__a, __value);
    } else {
        return _InterlockedOr64(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedOr64(__a, __value);
#endif
}

template <typename _Tp>
static inline _Tp __c11_atomic_fetch_or(volatile _Atomic(_Tp)* __a,
                                        _Tp __value, int __order) {
    if (sizeof(_Tp) == 1 && (std::is_integral<_Tp>::value || std::is_same<std::remove_cv_t<_Tp>, bool>::value)) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_or8((volatile char*)__a, __msvc_cast<char>(__value), (memory_order)__order));
    } else if (sizeof(_Tp) == 2 && std::is_integral<_Tp>::value) {
        return (_Tp)__msvc_atomic_fetch_or16((volatile short*)__a, (short)__value, (memory_order)__order);
    } else if (sizeof(_Tp) == 4 && std::is_integral<_Tp>::value) {
        return (_Tp) __msvc_atomic_fetch_or32((volatile long*)__a, (long)__value, (memory_order)__order);
    } else if (sizeof(_Tp) == 8 && std::is_integral<_Tp>::value) {
        return (_Tp) __msvc_atomic_fetch_or64((volatile __int64*)__a, (__int64)__value, (memory_order)__order);
    } else {
        __msvc_lock((void*)__a);
        _Tp __result = *(_Atomic(_Tp)*)__a;
        *(_Atomic(_Tp)*)__a |= __value;
        __msvc_unlock((void*)__a);
        return __result;
    }
}

template <typename _Tp>
static inline _Tp __c11_atomic_fetch_or(_Atomic(_Tp)* __a,
                                        _Tp __value, int __order) {
    return __c11_atomic_fetch_or((volatile _Atomic(_Tp)*)__a, __value, __order);
}

static inline char __msvc_atomic_fetch_xor8(volatile char* __a, char __value,
                                            memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedXor8_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedXor8_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedXor8_rel(__a, __value);
    } else {
        return _InterlockedXor8(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedXor8(__a, __value);
#endif
}

static inline short __msvc_atomic_fetch_xor16(volatile short* __a, short __value,
                                              memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedXor16_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedXor16_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedXor16_rel(__a, __value);
    } else {
        return _InterlockedXor16(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedXor16(__a, __value);
#endif
}

static inline long __msvc_atomic_fetch_xor32(volatile long* __a, long __value,
                                             memory_order __order) {
#if defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedXor_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedXor_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedXor_rel(__a, __value);
    } else {
        return _InterlockedXor(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedXor(__a, __value);
#endif
}

static inline __int64 __msvc_atomic_fetch_xor64(volatile __int64* __a, __int64 __value,
                                                memory_order __order) {
#if defined(_M_IX86)
    __int64 __tmp;
    do {
        __tmp = *__a;
    } while (__tmp != _InterlockedCompareExchange64(__a, __tmp ^ __value, __tmp));
    return __tmp;
#elif defined(_M_ARM) || defined(_M_ARM64)
    if (__order == memory_order_relaxed) {
        return _InterlockedXor64_nf(__a, __value);
    } else if (__order == memory_order_acquire) {
        return _InterlockedXor64_acq(__a, __value);
    } else if (__order == memory_order_release) {
        return _InterlockedXor64_rel(__a, __value);
    } else {
        return _InterlockedXor64(__a, __value);
    }
#else
    (void)__order;
    return _InterlockedXor64(__a, __value);
#endif
}

template <typename _Tp>
static inline _Tp __c11_atomic_fetch_xor(volatile _Atomic(_Tp)* __a,
                                         _Tp __value, int __order) {
    if (sizeof(_Tp) == 1 && (std::is_integral<_Tp>::value || std::is_same<std::remove_cv_t<_Tp>, bool>::value)) {
        return __msvc_cast<_Tp>(__msvc_atomic_fetch_xor8((volatile char*)__a, __msvc_cast<char>(__value), (memory_order)__order));
    } else if (sizeof(_Tp) == 2 && std::is_integral<_Tp>::value) {
        return (_Tp)__msvc_atomic_fetch_xor16((volatile short*)__a, (short)__value, (memory_order)__order);
    } else if (sizeof(_Tp) == 4 && std::is_integral<_Tp>::value) {
        return (_Tp) __msvc_atomic_fetch_xor32((volatile long*)__a, (long)__value, (memory_order)__order);
    } else if (sizeof(_Tp) == 8 && std::is_integral<_Tp>::value) {
        return (_Tp) __msvc_atomic_fetch_xor64((volatile __int64*)__a, (__int64)__value, (memory_order)__order);
    } else {
        __msvc_lock((void*)__a);
        _Tp __result = *(_Atomic(_Tp)*)__a;
        *(_Atomic(_Tp)*)__a ^= __value;
        __msvc_unlock((void*)__a);
        return __result;
    }
}

template <typename _Tp>
static inline _Tp __c11_atomic_fetch_xor(_Atomic(_Tp)* __a,
                                         _Tp __value, int __order) {
    return __c11_atomic_fetch_xor((volatile _Atomic(_Tp)*)__a, __value, __order);
}

static constexpr bool __atomic_is_lock_free(size_t __size, void*) {
    return __size <= 8;
}

static constexpr bool __atomic_always_lock_free(size_t __size, void*) {
    return __size <= 8;
}

#define __CLANG_ATOMIC_BOOL_LOCK_FREE 2
#define __CLANG_ATOMIC_CHAR_LOCK_FREE 2
#define __CLANG_ATOMIC_CHAR8_T_LOCK_FREE 2
#define __CLANG_ATOMIC_CHAR16_T_LOCK_FREE 2
#define __CLANG_ATOMIC_CHAR32_T_LOCK_FREE 2
#define __CLANG_ATOMIC_WCHAR_T_LOCK_FREE 2
#define __CLANG_ATOMIC_SHORT_LOCK_FREE 2
#define __CLANG_ATOMIC_INT_LOCK_FREE 2
#define __CLANG_ATOMIC_LONG_LOCK_FREE 2
#define __CLANG_ATOMIC_LLONG_LOCK_FREE 2
#define __CLANG_ATOMIC_POINTER_LOCK_FREE 2

_LIBCPP_END_NAMESPACE_STD

#endif
