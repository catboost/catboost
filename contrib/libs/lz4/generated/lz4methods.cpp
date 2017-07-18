#include "iface.h"

namespace lz4_10 { extern struct TLZ4Methods ytbl; }
namespace lz4_11 { extern struct TLZ4Methods ytbl; }
namespace lz4_12 { extern struct TLZ4Methods ytbl; }
namespace lz4_13 { extern struct TLZ4Methods ytbl; }
namespace lz4_14 { extern struct TLZ4Methods ytbl; }
namespace lz4_15 { extern struct TLZ4Methods ytbl; }
namespace lz4_16 { extern struct TLZ4Methods ytbl; }
namespace lz4_17 { extern struct TLZ4Methods ytbl; }
namespace lz4_18 { extern struct TLZ4Methods ytbl; }
namespace lz4_19 { extern struct TLZ4Methods ytbl; }
namespace lz4_20 { extern struct TLZ4Methods ytbl; }

extern "C" {

struct TLZ4Methods* LZ4Methods(int memory) {
    switch (memory) {
        case 10: return &lz4_10::ytbl;
        case 11: return &lz4_11::ytbl;
        case 12: return &lz4_12::ytbl;
        case 13: return &lz4_13::ytbl;
        case 14: return &lz4_14::ytbl;
        case 15: return &lz4_15::ytbl;
        case 16: return &lz4_16::ytbl;
        case 17: return &lz4_17::ytbl;
        case 18: return &lz4_18::ytbl;
        case 19: return &lz4_19::ytbl;
        case 20: return &lz4_20::ytbl;
    }

    return 0;
}

}
