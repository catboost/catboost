#include "normalization.h"

static const wchar32 S_BASE = 0xAC00;
static const wchar32 L_BASE = 0x1100;
static const wchar32 V_BASE = 0x1161;
static const wchar32 T_BASE = 0x11A7;
static const int L_COUNT = 19;
static const int V_COUNT = 21;
static const int T_COUNT = 28;
static const int N_COUNT = V_COUNT * T_COUNT; // 588
static const int S_COUNT = L_COUNT * N_COUNT; // 11172

static inline wchar32 ComposeHangul(wchar32 lead, wchar32 tail) {
    // 1. check to see if two current characters are L and V
    int lIndex = lead - L_BASE;
    if (0 <= lIndex && lIndex < L_COUNT) {
        int vIndex = tail - V_BASE;
        if (0 <= vIndex && vIndex < V_COUNT) {
            // make syllable of form LV
            lead = (wchar32)(S_BASE + (lIndex * V_COUNT + vIndex) * T_COUNT);
            return lead;
        }
    }

    // 2. check to see if two current characters are LV and T
    int sIndex = lead - S_BASE;
    if (0 <= sIndex && sIndex < S_COUNT && (sIndex % T_COUNT) == 0) {
        int TIndex = tail - T_BASE;
        if (0 < TIndex && TIndex < T_COUNT) {
            // make syllable of form LVT
            lead += TIndex;
            return lead;
        }
    }

    return 0;
}

NUnicode::NPrivate::TComposition::TComposition() {
    for (size_t i = 0; i != RawDataSize; ++i) {
        const TRawData& data = RawData[i];

        if (DecompositionCombining(data.Lead) != 0)
            continue;

        Data[TKey(data.Lead, data.Tail)] = data.Comp;
    }

    for (wchar32 s = 0xAC00; s != 0xD7A4; ++s) {
        const wchar32* decompBegin = NUnicode::Decomposition<true>(s);

        if (decompBegin == nullptr)
            continue;

        wchar32 lead = *(decompBegin++);
        while (*decompBegin) {
            wchar32 tail = *(decompBegin++);
            wchar32 comp = ComposeHangul(lead, tail);
            Y_ASSERT(comp != 0);

            Data[TKey(lead, tail)] = comp;

            lead = comp;
        }
    }
}
