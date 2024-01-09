#include <library/cpp/http/io/headers.h>

#include <benchmark/benchmark.h>

#include <util/stream/str.h>

void FindHeaderFirstMatch(benchmark::State& state) {
    THttpHeaders headers;
    headers.AddHeader("Host", "example.com");
    Y_ENSURE(headers.FindHeader("Host"));
    for (auto _ : state) {
        auto header = headers.FindHeader("Host");
        benchmark::DoNotOptimize(header);
    }
}

void FindHeaderNoMatchSameSize(benchmark::State& state) {
    THttpHeaders headers;
    for (char c = 'a'; c <= 'z'; ++c) {
        headers.AddHeader(TString::Join(c, "aaa"), "some value");
    }
    Y_ENSURE(!headers.FindHeader("Host"));
    for (auto _ : state) {
        auto header = headers.FindHeader("Host");
        benchmark::DoNotOptimize(header);
    }
}

void FindHeaderNoMatchDifferentSizesNoCommonPrefix(benchmark::State& state) {
    THttpHeaders headers;
    for (char c = 'a'; c <= 'z'; ++c) {  // same number of headers as above
        headers.AddHeader("aaaaa", "some value");
    }
    Y_ENSURE(!headers.FindHeader("Host"));
    for (auto _ : state) {
        auto header = headers.FindHeader("Host");
        benchmark::DoNotOptimize(header);
    }
}

void FindHeaderNoMatchDifferentSizesCommonPrefix(benchmark::State& state) {
    THttpHeaders headers;
    for (char c = 'a'; c <= 'z'; ++c) {
        headers.AddHeader("Host2", "some value");
    }
    Y_ENSURE(!headers.FindHeader("Host"));
    for (auto _ : state) {
        auto header = headers.FindHeader("Host");
        benchmark::DoNotOptimize(header);
    }
}

void FindHeaderMoreRealisticUseCase(benchmark::State& state) {
    TString requestHeaders(R"(Host: yandex.ru
User-Agent: Mozilla/5.0 ...
Accept: */*
Accept-Language: en-US,en;q=0.5
Accept-Encoding: gzip, deflate, br
Content-Type: text/plain;charset=UTF-8
Content-Length: 1234
Origin: https://a.yandex-team.ru
Connection: keep-alive
Referer: https://a.yandex-team.ru/
Sec-Fetch-Dest: empty
Sec-Fetch-Mode: no-cors
Sec-Fetch-Site: cross-site
TE: trailers)");
    TStringInput stream(requestHeaders);
    THttpHeaders headers(&stream);
    Y_ENSURE(headers.FindHeader("Content-Type"));
    for (auto _ : state) {
        auto header = headers.FindHeader("Content-Type");
        benchmark::DoNotOptimize(header);
    }
}

BENCHMARK(FindHeaderFirstMatch);
BENCHMARK(FindHeaderNoMatchSameSize);
BENCHMARK(FindHeaderNoMatchDifferentSizesNoCommonPrefix);
BENCHMARK(FindHeaderNoMatchDifferentSizesCommonPrefix);
BENCHMARK(FindHeaderMoreRealisticUseCase);
