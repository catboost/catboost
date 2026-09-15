#pragma once
#include "query.h"
#include <util/string/cast.h>
#include <util/string/split.h>
#include <cmath>

namespace NCB {
    // Match GetApproxScaleQueryCrossEntropy selection without allocating a
    // square table indexed by the largest user-supplied size. GPU queries are
    // limited to 256 rows, so entries beyond this range need only validation.
    inline TVector<float> SelectMetalQueryCrossEntropyScales(
        const NCatboostOptions::TLossDescription& loss,
        TConstArrayRef<float> targets,TConstArrayRef<ui32> offsets)
    {
        CB_ENSURE(offsets.size() >= 2 && offsets.front() == 0 && offsets.back() == targets.size(),
            "QueryCrossEntropy scale selection needs complete query offsets");
        const auto& params = loss.GetLossParamsMap();
        const auto found = params.find("raw_values_scale");
        const TString raw = found == params.end() ? TString() : found->second;
        CB_ENSURE(raw.size() <= (1u << 20), "QueryCrossEntropy scale description exceeds 1 MiB");
        TVector<float> table(257 * 257, 1.f);
        TVector<bool> assigned(257 * 257, false);
        float defaultScale = 1.f;
        bool hasDefault = false;
        if (!raw.empty()) {
            const TVector<TStringBuf> tokens = StringSplitter(raw).Split(' ');
            for (const auto& token : tokens) {
                const TVector<TString> item = StringSplitter(token).Split(':').Limit(2);
                CB_ENSURE(item.size() == 2, "raw_values_scale requires group_size,true_count:scale entries");
                const TVector<TString> index = StringSplitter(item[0]).Split(',').Limit(2);
                ui32 size = 0, count = 0;float scale = 0;
                CB_ENSURE(index.size() == 2 && TryFromString<ui32>(index[0],size) && TryFromString<ui32>(index[1],count)
                    && count <= size && TryFromString<float>(item[1],scale) && std::isfinite(scale),
                    "raw_values_scale requires uint32 sizes, true_count <= size, and finite float32 scales");
                if (!size && !hasDefault) { defaultScale = scale;hasDefault = true; }
                if (size <= 256) { table[size * 257 + count] = scale;assigned[size * 257 + count] = true; }
            }
        }
        TVector<float> result;result.reserve(offsets.size()-1);
        for (size_t group=0;group+1<offsets.size();++group) {
            const ui32 begin=offsets[group],end=offsets[group+1];
            CB_ENSURE(begin<end && end<=targets.size() && end-begin<=256,
                "Metal QueryCrossEntropy supports queries of 1..256 rows like CUDA");
            ui32 count=0;
            for (ui32 row=begin;row<end;++row) {
                CB_ENSURE(std::isfinite(targets[row]) && targets[row]>=0 && targets[row]<=1,
                    "Metal QueryCrossEntropy targets must be in [0,1]");
                count += targets[row] > .5f;
            }
            const ui32 index=(end-begin)*257+count;
            result.push_back(assigned[index] ? table[index] : defaultScale);
        }
        return result;
    }
}
