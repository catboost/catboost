// Independent source Pool row order for the model-based-eval prefix oracle.
// Header-only host program; no trainer, Metal runtime or model reader is used.
#include <util/random/fast.h>
#include <util/random/shuffle.h>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace NPrivate {
    void Panic(const TStaticBuf&, int, const char*, const char*, const char*, ...) noexcept {
        std::abort();
    }
}

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    const unsigned long long seed = std::strtoull(argv[1], nullptr, 10);
    const unsigned rows = std::strtoul(argv[2], nullptr, 10);
    if (!rows || rows > 1000000) return 2;
    TFastRng64 random(seed);
    std::vector<unsigned> order(rows);
    std::iota(order.begin(), order.end(), 0);
    Shuffle(order.begin(), order.end(), random);
    for (unsigned row : order) std::printf("%u\n", row);
}
