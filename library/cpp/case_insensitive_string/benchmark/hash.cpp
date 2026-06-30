#include <library/cpp/case_insensitive_string/case_insensitive_string.h>
#include <library/cpp/case_insensitive_string/ut_gtest/util/locale_guard.h>

#include <benchmark/benchmark.h>
#include <library/cpp/digest/murmur/murmur.h>

#include <util/generic/hash_table.h>
#include <util/generic/vector.h>
#include <util/string/ascii.h>
#include <util/random/random.h>

namespace {
    // THash<TCaseInsensitiveStringBuf>::operator() is not inlined
    Y_NO_INLINE size_t NaiveHash(TCaseInsensitiveStringBuf str) noexcept {
        TMurmurHash2A<size_t> hash;
        for (size_t i = 0; i < str.size(); ++i) {
            char lower = std::tolower(str[i]);
            hash.Update(&lower, 1);
        }
        return hash.Value();
    }

    [[maybe_unused]] Y_NO_INLINE size_t OptimizedHashV1(TCaseInsensitiveStringBuf str) noexcept {
        TMurmurHash2A<size_t> hash;
        std::array<char, sizeof(size_t)> buf;
        size_t headSize = str.size() - str.size() % buf.size();
        for (size_t i = 0; i < headSize; i += buf.size()) {
            for (size_t j = 0; j < buf.size(); ++j) {
                buf[j] = std::tolower(str[i + j]);
            }
            hash.Update(buf.data(), buf.size());
        }
        for (size_t i = headSize; i < str.size(); ++i) {
            char lower = std::tolower(str[i]);
            hash.Update(&lower, 1);
        }
        return hash.Value();
    }

    [[maybe_unused]] Y_NO_INLINE size_t OptimizedHashDuplicateTailLoop(TCaseInsensitiveStringBuf str) noexcept {
        TMurmurHash2A<size_t> hash;

        if (str.size() < sizeof(size_t)) {
            for (size_t i = 0; i < str.size(); ++i) {
                char lower = std::tolower(str[i]);
                hash.Update(&lower, 1);
            }
            return hash.Value();
        }
        std::array<char, sizeof(size_t)> buf;
        size_t headSize = str.size() - str.size() % buf.size();
        for (size_t i = 0; i < headSize; i += buf.size()) {
            for (size_t j = 0; j < buf.size(); ++j) {
                buf[j] = std::tolower(str[i + j]);
            }
            hash.Update(buf.data(), buf.size());
        }
        for (size_t i = headSize; i < str.size(); ++i) {
            char lower = std::tolower(str[i]);
            hash.Update(&lower, 1);
        }
        return hash.Value();
    }

    size_t HashTail(TMurmurHash2A<size_t>& hash, const char* data, size_t size) {
        for (size_t i = 0; i < size; ++i) {
            char lower = std::tolower(data[i]);
            hash.Update(&lower, 1);
        }
        return hash.Value();
    }

    [[maybe_unused]] Y_NO_INLINE size_t OptimizedHashDuplicateTailLoopInFunc(TCaseInsensitiveStringBuf str) noexcept {
        TMurmurHash2A<size_t> hash;
        if (str.size() < sizeof(size_t)) {
            return HashTail(hash, str.data(), str.size());
        }
        std::array<char, sizeof(size_t)> buf;
        size_t headSize = str.size() - str.size() % buf.size();
        for (size_t i = 0; i < headSize; i += buf.size()) {
            for (size_t j = 0; j < buf.size(); ++j) {
                buf[j] = std::tolower(str[i + j]);
            }
            hash.Update(buf.data(), buf.size());
        }
        return HashTail(hash, str.data() + headSize, str.size() - headSize);
    }

    // Currently shows the best performance, comparable with NaiveHash for short strings, only slower for empty strings.
    // Should be equivalent to OptimizedHashV1 after inlining HashTail, but for some reason it's a bit larger but at the same time faster than OptimizedHashV1.
    [[maybe_unused]] Y_NO_INLINE size_t OptimizedHashTailLoopInFunc(TCaseInsensitiveStringBuf str) noexcept {
        TMurmurHash2A<size_t> hash;
        std::array<char, sizeof(size_t)> buf;
        size_t headSize = str.size() - str.size() % buf.size();
        for (size_t i = 0; i < headSize; i += buf.size()) {
            for (size_t j = 0; j < buf.size(); ++j) {
                buf[j] = std::tolower(str[i + j]);
            }
            hash.Update(buf.data(), buf.size());
        }
        return HashTail(hash, str.data() + headSize, str.size() - headSize);
    }

    Y_FORCE_INLINE size_t DefaultHash(TCaseInsensitiveStringBuf str) {
        return ComputeHash(str);
    }

    Y_FORCE_INLINE size_t DefaultHashAscii(TCaseInsensitiveAsciiStringBuf str) {
        return ComputeHash(str);
    }
}

template <auto Impl, typename TTraits = TCaseInsensitiveCharTraits>
void CaseInsensitiveHash(benchmark::State& state) {
    TLocaleGuard loc("C");
    Y_ABORT_IF(loc.Error());
    SetRandomSeed(123 + state.range());
    TBasicString<char, TTraits> str;
    for (int i = 0; i < state.range(); ++i) {
        str.push_back(RandomNumber<unsigned char>());
    }
    Y_ENSURE(Impl(str) == NaiveHash(str), "Hashes differ: got " << Impl(str) << ", expected " <<  NaiveHash(str));
    for (auto _ : state) {
        size_t hash = Impl(str);
        benchmark::DoNotOptimize(hash);
    }
}

template <auto Impl, typename TTraits = TCaseInsensitiveCharTraits>
void CaseInsensitiveHashRandomSizes(benchmark::State& state) {
    TLocaleGuard loc("C");
    Y_ABORT_IF(loc.Error());
    SetRandomSeed(123);
    size_t minStrLen = static_cast<size_t>(state.range(0));
    size_t maxStrLen = static_cast<size_t>(state.range(1));
    static constexpr size_t nStrings = 64;
    TVector<TString> stringStorage(Reserve(nStrings));
    std::array<TBasicStringBuf<char, TTraits>, nStrings> strings;
    for (size_t i = 0; i < nStrings; ++i) {
        auto& str = stringStorage.emplace_back();
        size_t strLen = minStrLen + RandomNumber(maxStrLen - minStrLen + 1);
        for (size_t i = 0; i < strLen; ++i) {
            str.push_back(RandomNumber<unsigned char>());
        }
        strings[i] = str;
    }
    for (auto _ : state) {
        for (auto str : strings) {
            size_t hash = Impl(str);
            benchmark::DoNotOptimize(hash);
        }
    }
}

#define BENCH_ARGS ArgName("strlen")->DenseRange(0, sizeof(size_t) + 1)->RangeMultiplier(2)->Range(sizeof(size_t) * 2, 64)

BENCHMARK(CaseInsensitiveHash<NaiveHash>)->BENCH_ARGS;
BENCHMARK(CaseInsensitiveHash<DefaultHash>)->BENCH_ARGS;
#ifdef BENCHMARK_ALL_IMPLS
BENCHMARK(CaseInsensitiveHash<OptimizedHashV1>)->BENCH_ARGS;
BENCHMARK(CaseInsensitiveHash<OptimizedHashDuplicateTailLoop>)->BENCH_ARGS;
BENCHMARK(CaseInsensitiveHash<OptimizedHashDuplicateTailLoopInFunc>)->BENCH_ARGS;
BENCHMARK(CaseInsensitiveHash<OptimizedHashTailLoopInFunc>)->BENCH_ARGS;
#endif

BENCHMARK(CaseInsensitiveHash<DefaultHashAscii, TCaseInsensitiveAsciiCharTraits>)->BENCH_ARGS;

#undef BENCH_ARGS

#define BENCH_ARGS \
    ArgNames({"minStrLen", "maxStrLen"}) \
    ->ArgPair(4, 16) /*Approximate http header lengths (real distribution may differ)*/ \
    ->ArgPair(4, 11) /*1/2 short, 1/2 long to check possible branch mispredictions*/ \
    ->ArgPair(1, 8) /*Mostly short strings to check performance regression*/

BENCHMARK(CaseInsensitiveHashRandomSizes<NaiveHash>)->BENCH_ARGS;
BENCHMARK(CaseInsensitiveHashRandomSizes<DefaultHash>)->BENCH_ARGS;

BENCHMARK(CaseInsensitiveHashRandomSizes<DefaultHashAscii, TCaseInsensitiveAsciiCharTraits>)->BENCH_ARGS;

#undef BENCH_ARGS
