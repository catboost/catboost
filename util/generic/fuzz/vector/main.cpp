#include <util/generic/vector.h>
#include <util/stream/mem.h>

template <class T>
static inline T Read(IInputStream& in) {
    T t;

    in.LoadOrFail(&t, sizeof(t));

    return t;
}

extern "C" int LLVMFuzzerTestOneInput(const ui8* data, size_t size) {
    TMemoryInput mi(data, size);

    try {
        yvector<ui16> v;

        while (mi.Avail()) {
            char cmd = Read<char>(mi);

            switch (cmd % 2) {
                case 0: {
                    const size_t cnt = 1 + Read<ui8>(mi) % 16;

                    for (size_t i = 0; i < cnt; ++i) {
                        v.push_back(i);
                    }

                    break;
                }

                case 1: {
                    if (v) {
                        v.pop_back();
                    }

                    break;
                }
            }
        }
    } catch (...) {
    }

    return 0; // Non-zero return values are reserved for future use.
}
