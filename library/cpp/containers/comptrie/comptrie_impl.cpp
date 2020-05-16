#include "comptrie_impl.h"

#include <util/system/rusage.h>
#include <util/stream/output.h>

// Unpack the leaf value. The algorithm can store up to 8 full bytes in leafs.

namespace NCompactTrie {
    size_t MeasureOffset(size_t offset) {
        int n = 0;

        while (offset) {
            offset >>= 8;
            ++n;
        }

        return n;
    }

    size_t PackOffset(char* buffer, size_t offset) {
        size_t len = MeasureOffset(offset);
        size_t i = len;

        while (i--) {
            buffer[i] = (char)(offset & 0xFF);
            offset >>= 8;
        }

        return len;
    }

    void ShowProgress(size_t n) {
        if (n % 1000000 == 0)
            Cerr << n << ", RSS=" << (TRusage::Get().MaxRss >> 20) << "mb" << Endl;
        else if (n % 20000 == 0)
            Cerr << ".";
    }

}
