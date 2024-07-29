#include <library/cpp/threading/future/future.h>
#include <library/cpp/threading/future/core/coroutine_traits.h>

#include <benchmark/benchmark.h>

class TContext {
public:
    TContext()
        : NextInputPromise_(NThreading::NewPromise<bool>())
    {}
    ~TContext() {
        UpdateNextInput(false);
    }

    NThreading::TFuture<bool> NextInput() {
        return NextInputPromise_.GetFuture();
    }

    void UpdateNextInput(bool hasInput = true) {
        auto prevNextInputPromise = NextInputPromise_;
        NextInputPromise_ = NThreading::NewPromise<bool>();
        prevNextInputPromise.SetValue(hasInput);
    }

private:
    NThreading::TPromise<bool> NextInputPromise_;
};

static void TestPureFutureChainSubscribe(benchmark::State& state) {
    TContext context;
    size_t cnt = 0;
    std::function<void(const NThreading::TFuture<bool>&)> processInput = [&context, &cnt, &processInput](const NThreading::TFuture<bool>& hasInput) {
        if (hasInput.GetValue()) {
            benchmark::DoNotOptimize(++cnt);
            context.NextInput().Subscribe(processInput);
        }
    };

    processInput(NThreading::MakeFuture<bool>(true));
    for (auto _ : state) {
        context.UpdateNextInput();
    }
    context.UpdateNextInput(false);
}

static void TestPureFutureChainApply(benchmark::State& state) {
    TContext context;
    size_t cnt = 0;
    std::function<void(const NThreading::TFuture<bool>&)> processInput = [&context, &cnt, &processInput](const NThreading::TFuture<bool>& hasInput) {
        if (hasInput.GetValue()) {
            benchmark::DoNotOptimize(++cnt);
            context.NextInput().Apply(processInput);
        }
    };

    processInput(NThreading::MakeFuture<bool>(true));
    for (auto _ : state) {
        context.UpdateNextInput();
    }
    context.UpdateNextInput(false);
}

static void TestCoroFutureChain(benchmark::State& state) {
    TContext context;
    size_t cnt = 0;
    auto coroutine = [&context, &cnt]() -> NThreading::TFuture<void> {
        while (co_await context.NextInput()) {
            benchmark::DoNotOptimize(++cnt);
        }
    };

    auto coroutineFuture = coroutine();
    for (auto _ : state) {
        context.UpdateNextInput();
    }
    context.UpdateNextInput(false);
    coroutineFuture.GetValueSync();
}

BENCHMARK(TestPureFutureChainSubscribe);
BENCHMARK(TestPureFutureChainApply);
BENCHMARK(TestCoroFutureChain);
