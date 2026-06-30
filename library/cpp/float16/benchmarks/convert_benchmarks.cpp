
#include <benchmark/benchmark.h>
#include <library/cpp/float16/float16.h>


static void ConvertFromZero(benchmark::State& state) {
  // Perform setup here
  for (auto _ : state) {
    // This code gets timed
    Y_DO_NOT_OPTIMIZE_AWAY(TFloat16(0));
  }
}

constexpr size_t VectorSize = 8096;

static void FillSimple(benchmark::State& state) {
  // Perform setup here
  std::vector<TFloat16> data(VectorSize);
  for (auto _ : state) {
    // This code gets timed
    std::fill(data.data(), data.data() + data.size(), 0);
    Y_DO_NOT_OPTIMIZE_AWAY(data);
  }
}

static void FillConverted(benchmark::State& state) {
  // Perform setup here
  std::vector<TFloat16> data(VectorSize);
  for (auto _ : state) {
    // This code gets timed
    std::fill(data.data(), data.data() + data.size(), TFloat16(0));
    Y_DO_NOT_OPTIMIZE_AWAY(data);
  }
}

static void FillMemset(benchmark::State& state) {
  // Perform setup here
  std::vector<TFloat16> data(VectorSize);
  for (auto _ : state) {
    // This code gets timed
    memset(data.data(), 0, data.size() * sizeof(TFloat16));
    Y_DO_NOT_OPTIMIZE_AWAY(data);
  }
}

BENCHMARK(ConvertFromZero);
BENCHMARK(FillSimple);
BENCHMARK(FillConverted);
BENCHMARK(FillMemset);
