#pragma once

#include <util/generic/utility.h>
#include <util/generic/hash.h>
#include "net_queue_stat.h"

#include <atomic>

namespace NNetliba_v12 {
    struct TRequesterPendingDataStats: public IPeerQueueStats {
        int GetPacketCount() override {
            return InpCount + OutCount;
        }

        int InpCount, OutCount; // transfers
        ui64 InpDataSize, OutDataSize;
        TRequesterPendingDataStats() {
            // Can't call Zero(*this) here, we are derived class!
            InpCount = OutCount = InpDataSize = OutDataSize = 0;
        }
    };
    inline bool operator==(const TRequesterPendingDataStats& a, const TRequesterPendingDataStats& b) {
        return a.InpCount == b.InpCount && a.OutCount == b.OutCount &&
               a.InpDataSize == b.InpDataSize && a.OutDataSize == b.OutDataSize;
    }

    constexpr int DEFAULT_NETLIBA_COLOR = 0;

    class TColoredRequesterPendingDataStats {
        THashMap<ui8, TIntrusivePtr<TRequesterPendingDataStats>> ColoredStats;

    public:
        using const_iterator = THashMap<ui8, TIntrusivePtr<TRequesterPendingDataStats>>::const_iterator;

        TColoredRequesterPendingDataStats DeepCopy() {
            TColoredRequesterPendingDataStats tmp;
            for (const auto& i : ColoredStats) {
                tmp.ColoredStats[i.first] = new TRequesterPendingDataStats(*i.second);
            }
            return tmp;
        }
        const_iterator Begin() const {
            return ColoredStats.begin();
        }
        const_iterator End() const {
            return ColoredStats.end();
        }
        TIntrusivePtr<TRequesterPendingDataStats>& operator[](const ui8 color) {
            if (ColoredStats[color].Get() == nullptr) {
                ColoredStats[color] = new TRequesterPendingDataStats;
            }
            return ColoredStats[color];
        }
        void Swap(TColoredRequesterPendingDataStats& stat) {
            ColoredStats.swap(stat.ColoredStats);
        }
    };
    typedef std::function<void(const TRequesterPendingDataStats&, const TColoredRequesterPendingDataStats&)>
        TRequesterPendingDataAllStatsCb;
    typedef std::function<void(const TString& debugStr)>
        TDebugStringCb;

    struct TRequesterQueueStats {
        int ReqCount, RespCount;
        ui64 ReqQueueSize, RespQueueSize;
        TRequesterQueueStats() {
            Zero(*this);
        }
    };

    class TStatAggregator {
        float Swx;
        float Sw;
        std::atomic<float> Result;

    public:
        TStatAggregator()
            : Swx(0.0)
            , Sw(0.0001) //no zero, but small positive value;
            , Result(0.0)
        {
        }
        void AddPoint(float value) {
            //Weighted arithmetic mean calculation, where weight is value^3
            // Example: assume "worst" value is 0.5 "good" value is 0.01 and we want
            // to get weighted arithmetic mean in range [(0.5 - eps), 0.5) so:
            // ( N * 0.01 * 0.01^p + 0.5 * 0.5^p ) / ( N * 0.01^p + 0.5^p ) >= 0.5-eps
            // where N = 10000 (expected active connection num)
            // eps = 0.1 (accuracy)
            // walid if p >= 3
            const float weight = value * value * value;
            Sw += weight;
            Swx += weight * value;
        }
        void Update() {
            Result.store(Swx / Sw, std::memory_order_release);
            Swx = 0.0f;
            Sw = 0.0001f;
        }
        float GetResult() const {
            return Result.load(std::memory_order_acquire);
        }
    };

}
