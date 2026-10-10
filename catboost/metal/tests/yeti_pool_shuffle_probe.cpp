// Golden grouped Pool preprocessing order for test_native_yeti_ctr_p4.py.
// Mirrors objects_grouping.cpp::Shuffle, including within-query shuffles.
// No model fitting. From root: clang++ -std=c++20 -DNDEBUG -nostdinc++
// -I. -Icontrib/libs/cxxsupp/libcxx/include this_file.cpp -o probe
#include <util/random/fast.h>
#include <util/random/shuffle.h>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

// Keep this bounded reference executable independent of CatBoost's logger.
namespace NPrivate {
    void Panic(const TStaticBuf&, int, const char*, const char*, const char*, ...) noexcept {
        std::abort();
    }
}

int main() {
    TFastRng64 random(735);
    std::vector<unsigned> sizes, offsets{0}, groups(24);
    for (unsigned repeat = 0; repeat < 3; ++repeat)
        for (unsigned n : {3, 5, 8, 7, 12, 4, 9, 16}) {
            sizes.push_back(n); offsets.push_back(offsets.back() + n);
        }
    std::iota(groups.begin(), groups.end(), 0);
    Shuffle(groups.begin(), groups.end(), random);
    std::printf("{\"order\":["); bool first = true;
    for (unsigned group : groups) {
        std::vector<unsigned> rows(sizes[group]);
        std::iota(rows.begin(), rows.end(), offsets[group]);
        Shuffle(rows.begin(), rows.end(), random);
        for (unsigned row : rows) { std::printf("%s%u", first ? "" : ",", row); first = false; }
    }
    std::printf("],\"group_sizes\":["); first = true;
    for (unsigned group : groups) { std::printf("%s%u", first ? "" : ",", sizes[group]); first = false; }
    std::printf("]}\n");
}
