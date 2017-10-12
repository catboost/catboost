#pragma once

#include "single_host_memory_copy_tasks.h"
#include "cuda_buffer.h"

#include <catboost/libs/helpers/exception.h>

namespace NCudaLib {
    template <EPtrType From = EPtrType::CudaDevice, EPtrType To = EPtrType::CudaDevice>
    class TLatencyAndBandwidthStats {
    private:
        static constexpr ui64 BandwithIterations = 5;
        static constexpr ui64 LatencyIterations = 200;
        static constexpr ui64 DataSize = 16 * 1024 * 1024;

        TArray2D<double> LatencyTable;
        TArray2D<double> BandwidthTable;

        void BuildTlsLatencyTable() {
            auto& manager = GetCudaManager();
            CB_ENSURE(manager.HasDevices());

            auto deviceCount = manager.GetDeviceCount();
            LatencyTable.SetSizes(deviceCount, deviceCount);
            LatencyTable.FillZero();

            auto from = TCudaBuffer<char, TMirrorMapping, From>::Create(TMirrorMapping(1));
            auto to = TCudaBuffer<char, TMirrorMapping, To>::Create(TMirrorMapping(1));

            for (ui32 dev = 0; dev < deviceCount; ++dev) {
                for (ui32 secondDev = 0; secondDev <= dev; ++secondDev) {
                    for (ui32 iter = 0; iter < LatencyIterations; ++iter) {
                        manager.WaitComplete();
                        auto start = std::chrono::high_resolution_clock::now();
                        TDataCopier copier;
                        copier.AddAsyncMemoryCopyTask(from.GetBuffer(dev), 0,
                                                      to.GetBuffer(secondDev), 0, 1);
                        copier.SubmitCopy();
                        manager.WaitComplete(TDevicesList((1ULL << dev) | (1ULL << secondDev)));
                        auto elapsed = std::chrono::high_resolution_clock::now() - start;

                        LatencyTable[dev][secondDev] += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
                    }
                }
            }

            for (ui32 dev = 0; dev < deviceCount; ++dev) {
                for (ui64 secondDev = 0; secondDev <= dev; ++secondDev) {
                    LatencyTable[dev][secondDev] /= 1000;              //microseconds
                    LatencyTable[dev][secondDev] /= LatencyIterations; //mean
                    LatencyTable[secondDev][dev] = LatencyTable[dev][secondDev];
                }
            }
        }

        void BuildBandwidthTable() {
            auto& manager = GetCudaManager();

            auto deviceCount = manager.GetDeviceCount();
            BandwidthTable.SetSizes(deviceCount, deviceCount);
            BandwidthTable.FillZero();

            auto from = TCudaBuffer<char, TMirrorMapping, From>::Create(TMirrorMapping(DataSize));
            auto to = TCudaBuffer<char, TMirrorMapping, To>::Create(TMirrorMapping(DataSize));

            for (ui32 iter = 0; iter < BandwithIterations; ++iter) {
                for (ui32 dev = 0; dev < deviceCount; ++dev) {
                    for (ui32 secondDev = 0; secondDev <= dev; ++secondDev) {
                        manager.WaitComplete();
                        auto start = std::chrono::high_resolution_clock::now();
                        TDataCopier copier;
                        copier.AddAsyncMemoryCopyTask(from.GetBuffer(dev), 0,
                                                      to.GetBuffer(secondDev), 0, DataSize);
                        copier.SubmitCopy();
                        manager.WaitComplete();
                        auto elapsed = std::chrono::high_resolution_clock::now() - start;

                        BandwidthTable[dev][secondDev] += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count();
                    }
                }
            }

            for (ui32 dev = 0; dev < deviceCount; ++dev) {
                for (ui64 secondDev = 0; secondDev <= dev; ++secondDev) {
                    BandwidthTable[dev][secondDev] /= 1000;               //microseconds
                    BandwidthTable[dev][secondDev] /= BandwithIterations; //mean
                    BandwidthTable[dev][secondDev] /= DataSize;           //for one byte
                    BandwidthTable[secondDev][dev] = BandwidthTable[dev][secondDev];
                }
            }
        }

    public:
        TLatencyAndBandwidthStats() {
            BuildBandwidthTable();
            BuildTlsLatencyTable();
        }

        //microseconds
        double Latency(ui64 fromDevice, ui64 toDevice) const {
            return LatencyTable[fromDevice][toDevice];
        }

        //bytes per microsecond
        double Bandwidth(ui64 fromDevice, ui64 toDevice) const {
            return BandwidthTable[fromDevice][toDevice];
        }

        double BandwidthMbPerSec(ui64 fromDevice, ui64 toDevice) const {
            return (1000 * 1000 / Bandwidth(fromDevice, toDevice)) / 1024 / 1024;
        }
    };

    template <EPtrType From, EPtrType To>
    inline TLatencyAndBandwidthStats<From, To>& GetLatencyAndBandwidthStats() {
        using TStatsType = TLatencyAndBandwidthStats<From, To>;
        return *FastTlsSingleton<TStatsType>();
    }
}
