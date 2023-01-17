#include <benchmark/benchmark.h>

#include <library/cpp/yson/node/node_builder.h>

using namespace NYT;

static void BM_Map(benchmark::State& state, const std::tuple<int, bool>& input) {
    for (auto _ : state) {
        TNode node;
        TNodeBuilder builder(&node);
        if (std::get<1>(input)) {
            builder.OnBeginMap(std::get<0>(input));
        } else {
            builder.OnBeginMap();
        }
        for (auto i = 0; i < std::get<0>(input); ++i) {
            builder.OnKeyedItem(ToString(i));
            builder.OnUint64Scalar(i);
        }
        builder.OnEndMap();
    }
}

static void BM_List(benchmark::State& state, const std::tuple<int, bool>& input) {
    for (auto _ : state) {
        TNode node;
        TNodeBuilder builder(&node);
        if (std::get<1>(input)) {
            builder.OnBeginList(std::get<0>(input));
        } else {
            builder.OnBeginList();
        }
        for (auto i = 0; i < std::get<0>(input); ++i) {
            builder.OnListItem();
            builder.OnUint64Scalar(i);
        }
        builder.OnEndList();
    }
}

BENCHMARK_CAPTURE(BM_Map, map_10, std::make_tuple(10u, false));
BENCHMARK_CAPTURE(BM_Map, map_100, std::make_tuple(100u, false));
BENCHMARK_CAPTURE(BM_Map, map_300, std::make_tuple(300u, false));
BENCHMARK_CAPTURE(BM_Map, map_500, std::make_tuple(500u, false));
BENCHMARK_CAPTURE(BM_Map, map_1000, std::make_tuple(1000u, false));
BENCHMARK_CAPTURE(BM_Map, map_reserve_10, std::make_tuple(10u, true));
BENCHMARK_CAPTURE(BM_Map, map_reserve_100, std::make_tuple(100u, true));
BENCHMARK_CAPTURE(BM_Map, map_reserve_300, std::make_tuple(300u, true));
BENCHMARK_CAPTURE(BM_Map, map_reserve_500, std::make_tuple(500u, true));
BENCHMARK_CAPTURE(BM_Map, map_reserve_1000, std::make_tuple(1000u, true));
BENCHMARK_CAPTURE(BM_List, list_10, std::make_tuple(10u, false));
BENCHMARK_CAPTURE(BM_List, list_100, std::make_tuple(100u, false));
BENCHMARK_CAPTURE(BM_List, list_300, std::make_tuple(300u, false));
BENCHMARK_CAPTURE(BM_List, list_500, std::make_tuple(500u, false));
BENCHMARK_CAPTURE(BM_List, list_1000, std::make_tuple(1000u, false));
BENCHMARK_CAPTURE(BM_List, list_reserve_10, std::make_tuple(10u, true));
BENCHMARK_CAPTURE(BM_List, list_reserve_100, std::make_tuple(100u, true));
BENCHMARK_CAPTURE(BM_List, list_reserve_300, std::make_tuple(300u, true));
BENCHMARK_CAPTURE(BM_List, list_reserve_500, std::make_tuple(500u, true));
BENCHMARK_CAPTURE(BM_List, list_reserve_1000, std::make_tuple(1000u, true));
