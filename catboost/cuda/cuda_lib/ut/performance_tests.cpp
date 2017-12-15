#include <library/unittest/registar.h>
#include <iostream>
#include <thread>
#include <catboost/cuda/utils/cpu_random.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/bandwidth_latency_calcer.h>
#include <catboost/cuda/cuda_lib/buffer_resharding.h>

using namespace NCudaLib;

SIMPLE_UNIT_TEST_SUITE(TPerformanceTests) {
    SIMPLE_UNIT_TEST(TestRunKernelPerformance) {
        StartCudaManager();
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        {
            TRandom rand(0);
            const ui64 size = 2048;
            const ui64 kernelRuns = 10000;
            const ui64 tries = 100;

            auto& profiler = GetProfiler();
            for (ui32 i = 0; i < tries; ++i) {
                TVector<TStripeBuffer<float>> buffers;
                TStripeMapping stripeMapping = TStripeMapping::SplitBetweenDevices(size);
                auto guard = profiler.Profile("runKernelBatch");
                for (ui32 k = 0; k < kernelRuns; ++k) {
                    buffers.push_back(TCudaBuffer<float, TStripeMapping>::Create(stripeMapping));
                    FillBuffer(buffers.back(), 1.0f);
                }
            }
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(TestRunKernelAndReadResultPerformance) {
        StartCudaManager();
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        {
            TRandom rand(0);
            const ui64 size = 2048;
            const ui64 kernelRuns = 1000;
            const ui64 tries = 100;

            //            EnableProfiler();
            //            auto& profiler = CudaProfiler();
            auto& profiler = GetProfiler();
            for (ui32 i = 0; i < tries; ++i) {
                TVector<TStripeBuffer<float>> buffers;
                TStripeMapping stripeMapping = TStripeMapping::SplitBetweenDevices(size);

                auto guard = profiler.Profile("runKernelBatch");
                for (ui32 k = 0; k < kernelRuns; ++k) {
                    buffers.push_back(TCudaBuffer<float, TStripeMapping>::Create(stripeMapping));
                    FillBuffer(buffers.back(), 1.0f);
                    TVector<float> tmp;
                    buffers.back().Read(tmp);
                }
            }
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(BandwidthAndLatencyDeviceDevice) {
        StartCudaManager();

        {
            auto& latencyAndBandwidth = GetLatencyAndBandwidthStats<EPtrType::CudaDevice, EPtrType::CudaDevice>();

            ui64 devCount = GetCudaManager().GetDeviceCount();
            MATRIXNET_INFO_LOG << "Bandwitdh MB/s" << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.BandwidthMbPerSec(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }

            MATRIXNET_INFO_LOG << "Bandwitdh " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.Bandwidth(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }
            MATRIXNET_INFO_LOG << "Latency " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.Latency(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(PureCudaLatencyTest) {
        SetDevice(0);
        auto src = TCudaMemoryAllocation<CudaDevice>::Allocate<float>((ui64)2);
        SetDevice(1);
        auto dst = TCudaMemoryAllocation<CudaDevice>::Allocate<float>((ui64)2);

        TCudaStream stream;
        auto event = CreateCudaEvent();
        double val = 0;
        for (ui64 iter = 0; iter < 10000; ++iter) {
            stream.Synchronize();
            auto start = std::chrono::high_resolution_clock::now();
            TMemoryCopier<CudaDevice, CudaDevice>::CopyMemoryAsync(src, dst, (ui64)2, stream);
            event->Record(stream);
            event->WaitComplete();
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            val += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() * 1.0 / 1000;
        }
        val /= 10000;
        MATRIXNET_INFO_LOG << "Latency 0-1 " << val << Endl;
    }

    SIMPLE_UNIT_TEST(BandwidthAndLatencyDeviceHost) {
        StartCudaManager();
        {
            auto& latencyAndBandwidth = GetLatencyAndBandwidthStats<EPtrType::CudaDevice, EPtrType::CudaHost>();
            ui64 devCount = GetCudaManager().GetDeviceCount();
            MATRIXNET_INFO_LOG << "Bandwitdh MB/s" << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.BandwidthMbPerSec(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }

            MATRIXNET_INFO_LOG << "Bandwitdh " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.Bandwidth(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }
            MATRIXNET_INFO_LOG << "Latency " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.Latency(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(BandwidthAndLatencyHostHost) {
        StartCudaManager();
        {
            auto& latencyAndBandwidth = GetLatencyAndBandwidthStats<EPtrType::CudaHost, EPtrType::CudaHost>();
            ui64 devCount = GetCudaManager().GetDeviceCount();
            MATRIXNET_INFO_LOG << "Bandwitdh MB/s" << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.BandwidthMbPerSec(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }

            MATRIXNET_INFO_LOG << "Bandwitdh " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.Bandwidth(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }
            MATRIXNET_INFO_LOG << "Latency " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.Latency(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(BandwidthAndLatencyHostDevice) {
        StartCudaManager();
        {
            auto& latencyAndBandwidth = GetLatencyAndBandwidthStats<EPtrType::CudaHost, EPtrType::CudaDevice>();
            ui64 devCount = GetCudaManager().GetDeviceCount();
            MATRIXNET_INFO_LOG << "Bandwitdh MB/s" << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.BandwidthMbPerSec(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }

            MATRIXNET_INFO_LOG << "Bandwitdh " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.Bandwidth(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }
            MATRIXNET_INFO_LOG << "Latency " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    MATRIXNET_INFO_LOG << latencyAndBandwidth.Latency(dev, secondDev) << "\t";
                }
                MATRIXNET_INFO_LOG << Endl;
            }
        }
        StopCudaManager();
    }

    SIMPLE_UNIT_TEST(LatencyProfile) {
        auto& manager = NCudaLib::GetCudaManager();
        manager.Start();

        if (manager.GetDeviceCount() > 1) {
            auto leftMapping = TSingleMapping(0, 1);
            auto rightMapping = TSingleMapping(1, 1);
            auto bufferLeft = TSingleBuffer<float>::Create(leftMapping);
            auto bufferRight = TSingleBuffer<float>::Create(rightMapping);

            double val = 0;
            for (ui64 i = 0; i < 100000; ++i) {
                auto start = std::chrono::high_resolution_clock::now();
                Reshard(bufferLeft, bufferRight);
                manager.WaitComplete();
                auto elapsed = std::chrono::high_resolution_clock::now() - start;
                val += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() * 1.0 / 1000;
            }
            MATRIXNET_INFO_LOG << "Latency 0-1 " << val / 100000 << Endl;
        }
        manager.Stop();
    }

    SIMPLE_UNIT_TEST(BroadcastTest) {
        auto& manager = NCudaLib::GetCudaManager();
        manager.Start();

        if (manager.GetDeviceCount() > 1) {
            bool initialized = false;

            for (ui64 i = 1; i < 28; ++i) {
                TCudaProfiler profiler(EProfileMode::ImplicitLabelSync);

                const ui64 tries = 10;
                const ui64 innerTries = 10;
                //
                auto singleMapping = TSingleMapping(1, 1 << i);
                auto mirrorMapping = TMirrorMapping(1 << i);

                auto bufferSingle = TSingleBuffer<float>::Create(singleMapping);
                //                auto bufferSingleCpu = TCudaBuffer<float, TStripeMapping, EPtrType::CudaHost>::Create(TStripeMapping::SplitBetweenDevices(1 << i));
                //                auto bufferSingle = TCudaBuffer<float, TSingleMapping, EPtrType::CudaHost>::Create(singleMapping);
                auto bufferMirror = TMirrorBuffer<float>::Create(mirrorMapping);
                //                auto bufferMirror = TSingleBuffer<float>::Create(TSingleMapping(0, 1<< i));
                if (!initialized) {
                    Reshard(bufferSingle, bufferMirror);
                    initialized = true;
                }

                for (ui32 iter = 0; iter < tries; ++iter) {
                    auto guard = profiler.Profile("Broadcast " + ToString(1.0 * sizeof(float) * (1 << i) / 1024 / 1024) + "MBx" + ToString(innerTries));
                    for (ui32 innerIter = 0; innerIter < innerTries; ++innerIter) {
                        //                        Reshard(bufferSingleCpu, bufferSingle);
                        Reshard(bufferSingle, bufferMirror);
                    }
                }
            }

            {
                TCudaProfiler profiler(EProfileMode::ImplicitLabelSync);

                const ui64 tries = 10;
                //
                auto singleMapping = TSingleMapping(1, (ui64)(5500 * 1024 * 1024ULL));
                auto mirrorMapping = TMirrorMapping((ui64)(5500 * 1024 * 1024ULL));

                auto bufferSingle = TSingleBuffer<char>::Create(singleMapping);
                auto bufferMirror = TMirrorBuffer<char>::Create(mirrorMapping);

                for (ui32 iter = 0; iter < tries; ++iter) {
                    auto guard = profiler.Profile("Broadcast  5500MB");
                    Reshard(bufferSingle, bufferMirror);
                }
            }
        }
        manager.Stop();
    }

    SIMPLE_UNIT_TEST(StripeToSingleBroadcastTest) {
        auto& manager = NCudaLib::GetCudaManager();
        manager.Start();

        if (manager.GetDeviceCount() > 1) {
            bool initialized = false;

            for (ui64 i = 4; i < 28; ++i) {
                TCudaProfiler profiler(EProfileMode::ImplicitLabelSync);

                const ui64 tries = 10;
                const ui64 innerTries = 10;
                //
                auto singleMapping = TSingleMapping(1, 1 << i);
                auto stripeMapping = TStripeMapping::SplitBetweenDevices(1 << i);

                auto bufferSingle = TSingleBuffer<float>::Create(singleMapping);
                //                auto bufferSingleCpu = TCudaBuffer<float, TStripeMapping, EPtrType::CudaHost>::Create(TStripeMapping::SplitBetweenDevices(1 << i));
                //                auto bufferSingle = TCudaBuffer<float, TSingleMapping, EPtrType::CudaHost>::Create(singleMapping);
                //                auto bufferStripe = TCudaBuffer<float, TStripeMapping, EPtrType::CudaHost>::Create(stripeMapping);
                auto bufferStripe = TStripeBuffer<float>::Create(stripeMapping);
                //                auto bufferMirror = TSingleBuffer<float>::Create(TSingleMapping(0, 1<< i));
                if (!initialized) {
                    Reshard(bufferStripe, bufferSingle);
                    initialized = true;
                }

                for (ui32 iter = 0; iter < tries; ++iter) {
                    auto guard = profiler.Profile("Broadcast " + ToString(1.0 * sizeof(float) * (1 << i) / 1024 / 1024) + "MBx" + ToString(innerTries));
                    for (ui32 innerIter = 0; innerIter < innerTries; ++innerIter) {
                        Reshard(bufferStripe, bufferSingle);
                    }
                }
            }
        }
        manager.Stop();
    }
    SIMPLE_UNIT_TEST(StripeToMirrorBroadcastTest) {
        auto& manager = NCudaLib::GetCudaManager();
        manager.Start();

        if (manager.GetDeviceCount() > 1) {
            bool initialized = false;

            for (ui64 i = 4; i < 28; ++i) {
                TCudaProfiler profiler(EProfileMode::ImplicitLabelSync);

                const ui64 tries = 10;
                const ui64 innerTries = 10;
                //
                auto mirrorMapping = TMirrorMapping(1 << i);
                auto stripeMapping = TStripeMapping::SplitBetweenDevices(1 << i);

                auto bufferMirror = TMirrorBuffer<float>::Create(mirrorMapping);
                //                auto bufferSingleCpu = TCudaBuffer<float, TStripeMapping, EPtrType::CudaHost>::Create(TStripeMapping::SplitBetweenDevices(1 << i));
                //                auto bufferSingle = TCudaBuffer<float, TSingleMapping, EPtrType::CudaHost>::Create(singleMapping);
                //                auto bufferStripe = TCudaBuffer<float, TStripeMapping, EPtrType::CudaHost>::Create(stripeMapping);
                auto bufferStripe = TStripeBuffer<float>::Create(stripeMapping);
                //                auto bufferMirror = TSingleBuffer<float>::Create(TSingleMapping(0, 1<< i));
                if (!initialized) {
                    Reshard(bufferStripe, bufferMirror);
                    initialized = true;
                }

                for (ui32 iter = 0; iter < tries; ++iter) {
                    auto guard = profiler.Profile("Broadcast " + ToString(1.0 * sizeof(float) * (1 << i) / 1024 / 1024) + "MBx" + ToString(innerTries));
                    for (ui32 innerIter = 0; innerIter < innerTries; ++innerIter) {
                        Reshard(bufferStripe, bufferMirror);
                    }
                }
            }
        }
        manager.Stop();
    }

    SIMPLE_UNIT_TEST(TestReadAndWrite) {
        auto& manager = NCudaLib::GetCudaManager();
        StartCudaManager();
        {
            TCudaProfiler& profiler = manager.GetProfiler();
            profiler.SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
            auto profileGuard = profiler.Profile(TStringBuilder() << "Total time");

            ui64 tries = 30;

            const ui32 device = 0;
            for (ui32 size = 10; size < 10000000; size *= 10) {
                for (ui32 k = 0; k < tries; ++k) {
                    TVector<float> data(size);
                    TVector<float> tmp;
                    TSingleMapping mapping = TSingleMapping(device, size);

                    auto cudaVec = TCudaBuffer<float, TSingleMapping>::Create(mapping);
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Write #" << size << " floats");
                        cudaVec.Write(data);
                    }
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "Read #" << size << " floats");
                        cudaVec.Read(tmp);
                    }
                }
            }
        }
        StopCudaManager();
    }
}
