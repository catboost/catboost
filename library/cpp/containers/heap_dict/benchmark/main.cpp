#include "old_heap_dict.h"

#include <library/cpp/containers/heap_dict/heap_dict.h>

#include <library/testing/benchmark/bench.h>

#include <util/random/fast.h>
#include <util/generic/set.h>

static const size_t topSize = 1000;
static const size_t datasetSize = 10 * 1000 * 1000;
static const size_t numKeys = 30 * 1000;
static const i64 maxAbsCount = 500 * 1000;

template <class T>
static void UpdateHash(ui64& hash, T value) {
    static const ui64 base = 0x4906ba494954cb65ull;
    hash = hash * base + static_cast<ui64>(value);
}

Y_CPU_BENCHMARK(HashAndSortBenchmark, iface) {
    for (size_t iteration = 0; iteration < iface.Iterations(); ++iteration) {
        TFastRng<ui64> rng(iteration);

        THashMap<ui64, i64> dict;
        ui64 erasions = 0;
        for (size_t i = 0; i < datasetSize; ++i) {
            ui64 key = rng.GenRand() % numKeys;
            i64 count = static_cast<i64>(rng.GenRand() % (2 * maxAbsCount)) - maxAbsCount;
            auto& value = dict[key];
            value += count;
            if (value <= 0) {
                ++erasions;
                dict.erase(key);
            }
        }
        //Cout << '!' << iteration << ' ' << dict.size() << ' ' << erasions << Endl;
        TVector<std::pair<ui64, i64>> keysAndValues(dict.begin(), dict.end());
        std::sort(keysAndValues.begin(), keysAndValues.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.second > rhs.second;
        });
        TVector<ui64> topKeys;
        ui64 hash = 0;
        for (size_t i = 0; i < Min(topSize, keysAndValues.size()); ++i) {
            topKeys.push_back(keysAndValues[i].first);
            UpdateHash(hash, keysAndValues[i].first);
            UpdateHash(hash, keysAndValues[i].second);
        }
        //Cout << iteration << ' ' << hash << Endl;
    }
}

Y_CPU_BENCHMARK(SetBenchmark, iface) {
    for (size_t iteration = 0; iteration < iface.Iterations(); ++iteration) {
        TFastRng<ui64> rng(iteration);

        TSet<std::pair<i64, ui64>> set;
        THashMap<ui64, i64> dict;
        ui64 erasions = 0;
        for (size_t i = 0; i < datasetSize; ++i) {
            ui64 key = rng.GenRand() % numKeys;
            i64 count = static_cast<i64>(rng.GenRand() % (2 * maxAbsCount)) - maxAbsCount;
            auto& value = dict[key];
            set.erase(std::make_pair(value, key));
            value += count;
            if (value <= 0) {
                ++erasions;
                dict.erase(key);
            } else {
                set.insert(std::make_pair(value, key));
            }
        }
        //Cout << ',' << iteration << ' ' << set.size() << ' ' << erasions << Endl;
        TVector<ui64> topKeys;
        ui64 hash = 0;
        for (; topKeys.size() < topSize && !set.empty();) {
            auto top = *set.rbegin();
            topKeys.push_back(top.second);
            UpdateHash(hash, top.second);
            UpdateHash(hash, top.first);
            set.erase(top);
        }
        //Cout << iteration << ' ' << hash << Endl;
    }
}

Y_CPU_BENCHMARK(HeapDictBenchmark, iface) {
    for (size_t iteration = 0; iteration < iface.Iterations(); ++iteration) {
        TFastRng<ui64> rng(iteration);

        THeapDict<ui64, i64> heapDict;
        ui64 erasions = 0;
        for (size_t i = 0; i < datasetSize; ++i) {
            ui64 key = rng.GenRand() % numKeys;
            i64 count = static_cast<i64>(rng.GenRand() % (2 * maxAbsCount)) - maxAbsCount;
            auto& value = heapDict[key];
            value += count;
            if (value <= 0) {
                ++erasions;
                heapDict.erase(key);
            }
        }
        //Cout << '?' << iteration << ' ' << heapDict.size() << ' ' << erasions << Endl;
        TVector<ui64> topKeys;
        ui64 hash = 0;
        for (; topKeys.size() < topSize && !heapDict.empty(); heapDict.pop()) {
            topKeys.push_back(heapDict.top().first);
            UpdateHash(hash, heapDict.top().first);
            UpdateHash(hash, heapDict.top().second);
        }
        //Cout << iteration << ' ' << hash << Endl;
    }
}

Y_CPU_BENCHMARK(OldHeapDictBenchmark, iface) {
    for (size_t iteration = 0; iteration < iface.Iterations(); ++iteration) {
        TFastRng<ui64> rng(iteration);

        NDeprecated::THeapDict<ui64, bool, i64> heapDict;
        ui64 erasions = 0;
        for (size_t i = 0; i < datasetSize; ++i) {
            ui64 key = rng.GenRand() % numKeys;
            i64 count = static_cast<i64>(rng.GenRand() % (2 * maxAbsCount)) - maxAbsCount;
            auto value = heapDict[key];
            i64 newCount = value.Priority + count;
            heapDict.SetPriority(value, newCount);
            if (newCount <= 0) {
                ++erasions;
                heapDict.Pop(key);
            }
        }
        //Cout << '.' << iteration << ' ' << heapDict.GetSize() << ' ' << erasions << Endl;
        TVector<ui64> topKeys;
        ui64 hash = 0;
        for (; topKeys.size() < topSize && !heapDict.IsEmpty(); heapDict.Pop()) {
            topKeys.push_back(heapDict.GetTop().Key);
            UpdateHash(hash, heapDict.GetTop().Key);
            UpdateHash(hash, heapDict.GetTop().Priority);
        }
        //Cout << iteration << ' ' << hash << Endl;
    }
}
