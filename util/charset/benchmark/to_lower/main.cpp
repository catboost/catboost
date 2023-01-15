#include <library/testing/benchmark/bench.h>

#include <util/charset/wide.h>
#include <util/generic/singleton.h>
#include <util/generic/vector.h>
#include <util/generic/string.h>

static const auto ShortAscii = UTF8ToWide("hELlo");
static const auto LongAscii = UTF8ToWide(
    "The first plane, plane 0, the Basic Multilingual Plane (BMP) contains characters for almost "
    "all modern languages, and a large number of symbols. A primary objective for the BMP is to "
    "support the unification of prior character sets as well as characters for writing. Most of "
    "the assigned code points in the BMP are used to encode Chinese, Japanese, and Korean (CJK) "
    "characters.");

static const auto ShortRussian = UTF8ToWide("пРИвет");
static const auto LongRussian = UTF8ToWide(
    "Плоскость 0 (Основная многоязычная плоскость, англ. Basic Multilingual Plane, BMP) отведена "
    "для символов практически всех современных письменностей и большого числа специальных символов. "
    "Большая часть таблицы занята китайско-японскими иероглифами и своеобразными корейскими"
    "буквами. В Юникоде 10.0 в этой плоскости представлены следующие блоки");

#define DEFINE_INPLACE_BENCH(s)                                        \
    Y_CPU_BENCHMARK(s##CopyDetach, iface) {                            \
        for (size_t i = 0, iEnd = iface.Iterations(); i < iEnd; ++i) { \
            NBench::Clobber();                                         \
            auto copy = s;                                             \
            NBench::Escape(copy.Detach());                             \
            NBench::Clobber();                                         \
        }                                                              \
    }                                                                  \
                                                                       \
    Y_CPU_BENCHMARK(s##Inplace, iface) {                               \
        for (size_t i = 0, iEnd = iface.Iterations(); i < iEnd; ++i) { \
            NBench::Clobber();                                         \
            auto copy = s;                                             \
            ToLower(copy);                                             \
            NBench::Escape(copy.data());                               \
            NBench::Clobber();                                         \
        }                                                              \
    }

#define DEFINE_RET_BENCH(s)                                            \
    Y_CPU_BENCHMARK(s##Ret, iface) {                                   \
        for (size_t i = 0, iEnd = iface.Iterations(); i < iEnd; ++i) { \
            NBench::Clobber();                                         \
            const auto res = ToLowerRet(TWtringBuf{s});                \
            NBench::Escape(res.data());                                \
            NBench::Clobber();                                         \
        }                                                              \
    }

DEFINE_INPLACE_BENCH(ShortAscii)
DEFINE_INPLACE_BENCH(LongAscii)
DEFINE_INPLACE_BENCH(ShortRussian)
DEFINE_INPLACE_BENCH(LongRussian)

DEFINE_RET_BENCH(ShortAscii)
DEFINE_RET_BENCH(LongAscii)
DEFINE_RET_BENCH(ShortRussian)
DEFINE_RET_BENCH(LongRussian)
