#include <library/cpp/string_utils/base64/base64.h>

#include <library/cpp/testing/benchmark/bench.h>

#include <util/generic/buffer.h>
#include <util/generic/singleton.h>
#include <util/generic/string.h>
#include <util/generic/vector.h>
#include <util/generic/xrange.h>
#include <util/generic/yexception.h>
#include <util/random/random.h>

#include <array>

static TString GenerateRandomData(const size_t minSize, const size_t maxSize) {
    Y_ENSURE(minSize <= maxSize, "wow");
    TString r;
    for (size_t i = 0; i < minSize; ++i) {
        r.push_back(RandomNumber<char>());
    }

    if (minSize == maxSize) {
        return r;
    }

    const size_t size = RandomNumber<size_t>() % (maxSize - minSize + 1);
    for (size_t i = 0; i < size; ++i) {
        r.push_back(RandomNumber<char>());
    }

    return r;
}

template <size_t N>
static std::array<TString, N> GenerateRandomDataVector(const size_t minSize, const size_t maxSize) {
    std::array<TString, N> r;
    for (size_t i = 0; i < N; ++i) {
        r[i] = GenerateRandomData(minSize, maxSize);
    }

    return r;
}

template <size_t N>
static std::array<TString, N> Encode(const std::array<TString, N>& d) {
    std::array<TString, N> r;
    for (size_t i = 0, iEnd = d.size(); i < iEnd; ++i) {
        r[i] = Base64Encode(d[i]);
    }

    return r;
}

namespace {
    template <size_t N, size_t MinSize, size_t MaxSize>
    struct TRandomDataHolder {
        TRandomDataHolder()
            : Data(GenerateRandomDataVector<N>(MinSize, MaxSize))
            , DataEncoded(Encode<N>(Data))
        {
            for (size_t i = 0; i < N; ++i) {
                const size_t size = Data[i].size();
                const size_t sizeEnc = DataEncoded[i].size();
                PlaceToEncode[i].Resize(Base64EncodeBufSize(size));
                PlaceToDecode[i].Resize(Base64DecodeBufSize(sizeEnc));
            }
        }

        static constexpr size_t Size = N;
        const std::array<TString, N> Data;
        const std::array<TString, N> DataEncoded;
        std::array<TBuffer, N> PlaceToEncode;
        std::array<TBuffer, N> PlaceToDecode;
    };

    template <size_t N, size_t Size>
    using TFixedSizeRandomDataHolder = TRandomDataHolder<N, Size, Size>;

    using FSRDH_1 = TFixedSizeRandomDataHolder<10, 1>;
    using FSRDH_2 = TFixedSizeRandomDataHolder<10, 2>;
    using FSRDH_4 = TFixedSizeRandomDataHolder<10, 4>;
    using FSRDH_8 = TFixedSizeRandomDataHolder<10, 8>;
    using FSRDH_16 = TFixedSizeRandomDataHolder<10, 16>;
    using FSRDH_32 = TFixedSizeRandomDataHolder<10, 32>;
    using FSRDH_64 = TFixedSizeRandomDataHolder<10, 64>;
    using FSRDH_128 = TFixedSizeRandomDataHolder<10, 128>;
    using FSRDH_1024 = TFixedSizeRandomDataHolder<10, 1024>;
    using FSRDH_10240 = TFixedSizeRandomDataHolder<10, 10240>;
    using FSRDH_102400 = TFixedSizeRandomDataHolder<10, 102400>;
    using FSRDH_1048576 = TFixedSizeRandomDataHolder<10, 1048576>;
    using FSRDH_10485760 = TFixedSizeRandomDataHolder<10, 10485760>;
}

template <typename T>
static inline void BenchEncode(T& d, const NBench::NCpu::TParams& iface) {
    for (const auto it : xrange(iface.Iterations())) {
        Y_UNUSED(it);
        for (size_t i = 0; i < d.Size; ++i) {
            NBench::Escape(d.PlaceToEncode[i].data());
            Y_DO_NOT_OPTIMIZE_AWAY(
                Base64Encode(d.PlaceToEncode[i].data(), (const unsigned char*)d.Data[i].data(), d.Data[i].size()));
            NBench::Clobber();
        }
    }
}

template <typename T>
static inline void BenchEncodeUrl(T& d, const NBench::NCpu::TParams& iface) {
    for (const auto it : xrange(iface.Iterations())) {
        Y_UNUSED(it);
        for (size_t i = 0; i < d.Size; ++i) {
            NBench::Escape(d.PlaceToEncode[i].data());
            Y_DO_NOT_OPTIMIZE_AWAY(
                Base64EncodeUrl(d.PlaceToEncode[i].data(), (const unsigned char*)d.Data[i].data(), d.Data[i].size()));
            NBench::Clobber();
        }
    }
}

template <typename T>
static inline void BenchDecode(T& d, const NBench::NCpu::TParams& iface) {
    for (const auto it : xrange(iface.Iterations())) {
        Y_UNUSED(it);
        for (size_t i = 0; i < d.Size; ++i) {
            NBench::Escape(d.PlaceToDecode[i].data());
            Y_DO_NOT_OPTIMIZE_AWAY(
                Base64Decode(d.PlaceToDecode[i].data(), (const char*)d.DataEncoded[i].data(), (const char*)(d.DataEncoded[i].data() + d.DataEncoded[i].size())));
            NBench::Clobber();
        }
    }
}

Y_CPU_BENCHMARK(EncodeF1, iface) {
    auto& d = *Singleton<FSRDH_1>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF1, iface) {
    auto& d = *Singleton<FSRDH_1>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF2, iface) {
    auto& d = *Singleton<FSRDH_2>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF2, iface) {
    auto& d = *Singleton<FSRDH_2>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF4, iface) {
    auto& d = *Singleton<FSRDH_4>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF4, iface) {
    auto& d = *Singleton<FSRDH_4>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF8, iface) {
    auto& d = *Singleton<FSRDH_8>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF8, iface) {
    auto& d = *Singleton<FSRDH_8>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF16, iface) {
    auto& d = *Singleton<FSRDH_16>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF16, iface) {
    auto& d = *Singleton<FSRDH_16>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF32, iface) {
    auto& d = *Singleton<FSRDH_32>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF32, iface) {
    auto& d = *Singleton<FSRDH_32>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF64, iface) {
    auto& d = *Singleton<FSRDH_64>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF64, iface) {
    auto& d = *Singleton<FSRDH_64>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF128, iface) {
    auto& d = *Singleton<FSRDH_128>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF128, iface) {
    auto& d = *Singleton<FSRDH_128>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF1024, iface) {
    auto& d = *Singleton<FSRDH_1024>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF1024, iface) {
    auto& d = *Singleton<FSRDH_1024>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF10240, iface) {
    auto& d = *Singleton<FSRDH_10240>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF10240, iface) {
    auto& d = *Singleton<FSRDH_10240>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF102400, iface) {
    auto& d = *Singleton<FSRDH_102400>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF102400, iface) {
    auto& d = *Singleton<FSRDH_102400>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF1048576, iface) {
    auto& d = *Singleton<FSRDH_1048576>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF1048576, iface) {
    auto& d = *Singleton<FSRDH_1048576>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeF10485760, iface) {
    auto& d = *Singleton<FSRDH_10485760>();
    BenchEncode(d, iface);
}

Y_CPU_BENCHMARK(DecodeF10485760, iface) {
    auto& d = *Singleton<FSRDH_10485760>();
    BenchDecode(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF1, iface) {
    auto& d = *Singleton<FSRDH_1>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF2, iface) {
    auto& d = *Singleton<FSRDH_2>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF4, iface) {
    auto& d = *Singleton<FSRDH_4>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF8, iface) {
    auto& d = *Singleton<FSRDH_8>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF16, iface) {
    auto& d = *Singleton<FSRDH_16>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF32, iface) {
    auto& d = *Singleton<FSRDH_32>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF64, iface) {
    auto& d = *Singleton<FSRDH_64>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF128, iface) {
    auto& d = *Singleton<FSRDH_128>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF1024, iface) {
    auto& d = *Singleton<FSRDH_1024>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF10240, iface) {
    auto& d = *Singleton<FSRDH_10240>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF102400, iface) {
    auto& d = *Singleton<FSRDH_102400>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF1048576, iface) {
    auto& d = *Singleton<FSRDH_1048576>();
    BenchEncodeUrl(d, iface);
}

Y_CPU_BENCHMARK(EncodeUrlF10485760, iface) {
    auto& d = *Singleton<FSRDH_10485760>();
    BenchEncodeUrl(d, iface);
}
