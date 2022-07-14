#pragma once

#include <util/system/compiler.h>
#include <util/system/types.h>

#include <utility>

namespace NBench {
    namespace NCpu {
        struct TParams {
            inline size_t Iterations() const noexcept {
                return Iterations_;
            }

            const size_t Iterations_;
        };

        using TUserFunc = void(TParams&);

        struct TRegistar {
            TRegistar(const char* name, TUserFunc func);

            char Buf[128];
        };
    }

    /**
     * Functions that states "I can read and write everywhere in memory".
     *
     * Use it to prevent optimizer from reordering or discarding memory writes prior to it's call,
     * and force memory reads after it's call.
     */
    void Clobber();

    /**
     * Forces whatever `p` points to be in memory and not in register.
     *
     * @param       Pointer to data.
     */
    template <typename T>
    void Escape(T* p);

#if defined(__GNUC__)
    Y_FORCE_INLINE void Clobber() {
        asm volatile(""
                     :
                     :
                     : "memory");
    }
#elif defined(_MSC_VER)
    Y_FORCE_INLINE void Clobber() {
        _ReadWriteBarrier();
    }

#else
    Y_FORCE_INLINE void Clobber() {
    }
#endif

#if defined(__GNUC__)
    template <typename T>
    Y_FORCE_INLINE void Escape(T* p) {
        asm volatile(""
                     :
                     : "g"(p)
                     : "memory");
    }
#else
    template <typename T>
    Y_FORCE_INLINE void Escape(T*) {
    }
#endif

    /**
     * Use this function to prevent unused variables elimination.
     *
     * @param       Unused variable (e.g. return value of benchmarked function).
     */
    template <typename T>
    Y_FORCE_INLINE void DoNotOptimize(T&& datum) {
        ::DoNotOptimizeAway(std::forward<T>(datum));
    }

    int Main(int argc, char** argv);
}

#define Y_CPU_BENCHMARK(name, cnt)                              \
    namespace N_bench_##name {                                  \
        static void Run(::NBench::NCpu::TParams&);              \
        const ::NBench::NCpu::TRegistar benchmark(#name, &Run); \
    }                                                           \
    static void N_bench_##name::Run(::NBench::NCpu::TParams& cnt)
