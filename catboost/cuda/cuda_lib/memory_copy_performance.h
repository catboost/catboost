#pragma once

#include "cuda_buffer.h"
#include <util/system/hp_timer.h>
#include <catboost/libs/helpers/exception.h>

//WARNING: Not thread-safe
namespace NCudaLib {
    inline double BandwidthMbPerSec(double bandwidth) {
        return (1000 * 1000 / bandwidth) / 1024 / 1024;
    }

    class TLazyPointToPointTable {
    public:
        template <class TCalcer>
        void SetIfNotExist(TDeviceId fromDevice, TDeviceId toDevice,
                           TCalcer&& calcer) {
            if (fromDevice > toDevice) {
                using std::swap;
                swap(fromDevice, toDevice);
            }
            const bool notCompute = !(Table.contains(fromDevice) && Table[fromDevice].contains(toDevice));

            if (Y_UNLIKELY(notCompute)) {
                Table[fromDevice][toDevice] = calcer();
            }
        }

        double Get(TDeviceId fromDevice, TDeviceId toDevice) const {
            if (fromDevice > toDevice) {
                using std::swap;
                swap(fromDevice, toDevice);
            }
            return Table.at(fromDevice).at(toDevice);
        }

    private:
        mutable TMap<TDeviceId, TMap<TDeviceId, double>> Table;
    };

    struct TLatencyAndBandwidthStatsHelper {
        static constexpr ui32 BandwidthIterations = 5;
        static constexpr ui32 LatencyIterations = 200;
        static constexpr ui32 BandwidthDataSize = 16 * 1024 * 1024;

        template <class TSampler>
        static inline double ComputeMean(TSampler&& sampler, const ui32 iters) {
            //init

            double total = 0;
            for (ui32 iter = 0; iter < iters; ++iter) {
                total += sampler();
            }
            total /= iters;
            return total;
        }
    };

    template <EPtrType From,
              EPtrType To>
    class TMemoryCopyPerformance {
    public:
        static double ComputeTime(ui32 sourceDevice,
                                  ui32 destDevice,
                                  const ui32 messageSize) {
            auto from = TCudaBuffer<char, TSingleMapping, From>::Create(TSingleMapping(sourceDevice, messageSize));
            auto to = TCudaBuffer<char, TSingleMapping, To>::Create(TSingleMapping(destDevice, messageSize));
            auto& manager = GetCudaManager();
            TDataCopier copier;
            copier.AddAsyncMemoryCopyTask(from.Buffers[sourceDevice], 0,
                                          to.Buffers[destDevice], 0,
                                          messageSize);
            TDevicesListBuilder builder;
            builder.AddDevice(sourceDevice);
            builder.AddDevice(destDevice);

            manager.WaitComplete();
            auto start = std::chrono::high_resolution_clock::now();
            copier.SubmitCopy();
            manager.WaitComplete(builder.Build());
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() * 1.0 / 1000;
        }

        //        microseconds
        double Latency(ui32 fromDevice, ui32 toDevice) const {
            auto& manager = GetCudaManager();
            TDeviceId leftDevice = manager.GetDeviceId(fromDevice);
            TDeviceId rightDevice = manager.GetDeviceId(toDevice);
            auto oneSampleLatency = [&]() -> double {
                return ComputeTime(fromDevice, toDevice, 1);
            };

            LatencyTable.SetIfNotExist(leftDevice, rightDevice, [&]() -> double {
                return TLatencyAndBandwidthStatsHelper::ComputeMean(oneSampleLatency,
                                                                    TLatencyAndBandwidthStatsHelper::LatencyIterations);
            });
            return LatencyTable.Get(leftDevice, rightDevice);
        }

        //bytes per microsecond
        double Bandwidth(ui32 fromDevice, ui32 toDevice) const {
            auto& manager = GetCudaManager();
            TDeviceId leftDevice = manager.GetDeviceId(fromDevice);
            TDeviceId rightDevice = manager.GetDeviceId(toDevice);
            auto oneSampleBandwidth = [&]() -> double {
                const ui32 messageSize = TLatencyAndBandwidthStatsHelper::BandwidthDataSize;
                return ComputeTime(fromDevice, toDevice, messageSize) / messageSize;
            };
            BandwidthTable.SetIfNotExist(leftDevice, rightDevice, [&]() -> double {
                return TLatencyAndBandwidthStatsHelper::ComputeMean(oneSampleBandwidth,
                                                                    TLatencyAndBandwidthStatsHelper::BandwidthIterations);
            });
            return BandwidthTable.Get(leftDevice, rightDevice);
        }

    private:
        mutable TLazyPointToPointTable LatencyTable;
        mutable TLazyPointToPointTable BandwidthTable;
    };

    template <>
    class TMemoryCopyPerformance<EPtrType::Host, EPtrType::Host>; //useless

    template <class TImpl>
    class TMasterToDeviceStats {
    public:
        //bytes pert microsecond
        double Latency(ui32 device) const {
            auto& manager = GetCudaManager();
            TDeviceId deviceId = manager.GetDeviceId(device);

            if (Y_UNLIKELY(!LatencyTable.contains(deviceId))) {
                auto oneSampleLatency = [&]() -> double {
                    const ui32 messageSize = 1;
                    return TImpl::ComputeTime(device, messageSize) / messageSize;
                };
                LatencyTable[deviceId] = TLatencyAndBandwidthStatsHelper::ComputeMean(oneSampleLatency,
                                                                                      TLatencyAndBandwidthStatsHelper::LatencyIterations);
            }
            return LatencyTable[deviceId];
        }

        //bytes per microsecond
        double Bandwidth(ui32 device) const {
            auto& manager = GetCudaManager();
            TDeviceId deviceId = manager.GetDeviceId(device);

            if (Y_UNLIKELY(!BandwidthTable.contains(deviceId))) {
                auto oneSampleLatency = [&]() -> double {
                    const ui32 messageSize = TLatencyAndBandwidthStatsHelper::BandwidthDataSize;
                    return TImpl::ComputeTime(device, messageSize) / messageSize;
                };
                BandwidthTable[deviceId] = TLatencyAndBandwidthStatsHelper::ComputeMean(oneSampleLatency,
                                                                                        TLatencyAndBandwidthStatsHelper::BandwidthIterations);
            }
            return BandwidthTable[deviceId];
        }

    private:
        mutable TMap<TDeviceId, double> LatencyTable;
        mutable TMap<TDeviceId, double> BandwidthTable;
    };

    template <EPtrType To>
    class TMemoryCopyPerformance<EPtrType::Host, To>: public TMasterToDeviceStats<TMemoryCopyPerformance<EPtrType::Host, To>> {
    public:
        static double ComputeTime(ui32 destDevice,
                                  ui32 messageSize) {
            auto& manager = GetCudaManager();
            manager.WaitComplete();

            TVector<char> source(messageSize);
            auto start = std::chrono::high_resolution_clock::now();
            auto to = TCudaBuffer<char, TSingleMapping, To>::Create(TSingleMapping(destDevice, messageSize));
            to.Write(source);
            TDevicesListBuilder builder;
            builder.AddDevice(destDevice);
            manager.WaitComplete(builder.Build());
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() * 1.0 / 1000;
        }
    };

    template <EPtrType From>
    class TMemoryCopyPerformance<From, EPtrType::Host>: public TMasterToDeviceStats<TMemoryCopyPerformance<From, EPtrType::Host>> {
    public:
        static double ComputeTime(ui32 destDevice,
                                  ui32 messageSize) {
            auto& manager = GetCudaManager();
            manager.WaitComplete();
            TVector<char> source(messageSize);
            auto start = std::chrono::high_resolution_clock::now();
            auto from = TCudaBuffer<char, TSingleMapping, From>::Create(TSingleMapping(destDevice, messageSize));
            from.Read(source);
            TDevicesListBuilder builder;
            builder.AddDevice(destDevice);
            manager.WaitComplete(builder.Build());
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            return std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() * 1.0 / 1000;
        }
    };

    template <EPtrType From,
              EPtrType To>
    inline TMemoryCopyPerformance<From, To>& GetMemoryCopyPerformance() {
        using TTableType = TMemoryCopyPerformance<From, To>;
        return *Singleton<TTableType>();
    }
}
