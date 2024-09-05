#include "random.h"
#include "entropy.h"
#include "mersenne.h"

#include <util/system/getpid.h>
#include <util/thread/singleton.h>
#include <util/stream/multi.h>
#include <util/stream/mem.h>
#include <util/digest/numeric.h>

namespace {
    struct TProcStream {
        ui32 Extra;
        TMemoryInput MI;
        TMultiInput TI;

        static inline ui32 ExtraData() noexcept {
            ui32 data;

            EntropyPool().LoadOrFail(&data, sizeof(data));

            return IntHash(data ^ GetPID());
        }

        inline TProcStream() noexcept
            : Extra(ExtraData())
            , MI(&Extra, sizeof(Extra))
            , TI(&MI, &EntropyPool())
        {
        }

        inline IInputStream& S() noexcept {
            return TI;
        }
    };

    template <class T>
    struct TRndGen: public TMersenne<T> {
        inline TRndGen()
            : TMersenne<T>(TProcStream().S())
        {
        }

        inline TRndGen(T seed)
            : TMersenne<T>(seed)
        {
        }
    };

    template <class T>
    static inline TRndGen<T>* GetRndGen() {
        return FastTlsSingletonWithPriority<TRndGen<T>, 2>();
    }

    template <unsigned N>
    struct TToRealTypeBySize {
        using TResult = ui32;
    };

    template <>
    struct TToRealTypeBySize<8> {
        using TResult = ui64;
    };

    template <class T>
    struct TToRealType {
        using TResult = typename TToRealTypeBySize<sizeof(T)>::TResult;
    };
} // namespace

#define DEF_RND(TY)                                               \
    template <>                                                   \
    TY RandomNumber<TY>() {                                       \
        return GetRndGen<TToRealType<TY>::TResult>()->GenRand();  \
    }                                                             \
                                                                  \
    template <>                                                   \
    TY RandomNumber<TY>(TY n) {                                   \
        return GetRndGen<TToRealType<TY>::TResult>()->Uniform(n); \
    }

DEF_RND(char)
DEF_RND(unsigned char)
DEF_RND(unsigned int)
DEF_RND(unsigned long)
DEF_RND(unsigned short)
DEF_RND(unsigned long long)

#undef DEF_RND

template <>
bool RandomNumber<bool>() {
    return RandomNumber<ui8>() % 2 == 0;
}

template <>
float RandomNumber<float>() {
    float ret;

    do {
        ret = (float)GetRndGen<ui64>()->GenRandReal2();
    } while (ret >= 1);

    return ret;
}

template <>
double RandomNumber<double>() {
    return GetRndGen<ui64>()->GenRandReal4();
}

template <>
long double RandomNumber<long double>() {
    return RandomNumber<double>();
}

void ResetRandomState() {
    *GetRndGen<ui32>() = TRndGen<ui32>();
    *GetRndGen<ui64>() = TRndGen<ui64>();
}

void SetRandomSeed(int seed) {
    *GetRndGen<ui32>() = TRndGen<ui32>(seed);
    *GetRndGen<ui64>() = TRndGen<ui64>(seed);
}
