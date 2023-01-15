#include <library/cpp/testing/unittest/registar.h>
#include <iostream>
#include <thread>
#include <catboost/libs/helpers/cpu_random.h>
#include <catboost/cuda/cuda_util/fill.h>
#include <catboost/cuda/cuda_lib/cuda_profiler.h>
#include <catboost/cuda/cuda_lib/memory_copy_performance.h>
#include <catboost/cuda/cuda_lib/cuda_buffer_helpers/buffer_resharding.h>
#include <catboost/cuda/cuda_util/bootstrap.h>

using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TPerformanceTests) {
    Y_UNIT_TEST(TestRunKernelPerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        {
            TRandom rand(0);
            const ui64 size = 4;
            const ui64 maxKernelRuns = 100001;
            const ui64 tries = 100;

            auto& profiler = GetProfiler();
            for (ui32 i = 0; i < tries; ++i) {
                for (ui64 kernelRuns = 1; kernelRuns < maxKernelRuns; kernelRuns *= 10) {
                    TVector<TStripeBuffer<float>> buffers;
                    TStripeMapping stripeMapping = TStripeMapping::RepeatOnAllDevices(size);
                    {
                        auto guard = profiler.Profile(TStringBuilder() << "runKernelBatch_" << kernelRuns);
                        for (ui32 k = 0; k < kernelRuns; ++k) {
                            buffers.push_back(TCudaBuffer<float, TStripeMapping>::Create(stripeMapping));
                            FillBuffer(buffers.back(), 1.0f);
                        }
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestRunKernelInStreamsPerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        {
            TRandom rand(0);
            const ui64 size = 300000;
            const ui64 maxKernelRuns = 100001;
            const ui64 tries = 100;

            auto& profiler = GetProfiler();
            TVector<TComputationStream> streams;
            for (int i = 0; i < 32; ++i) {
                streams.push_back(GetCudaManager().RequestStream());
            }

            for (ui32 i = 0; i < tries; ++i) {
                for (ui64 kernelRuns = 1; kernelRuns < maxKernelRuns; kernelRuns *= 10) {
                    TStripeMapping stripeMapping = TStripeMapping::RepeatOnAllDevices(size);
                    TStripeBuffer<float> buffer = TCudaBuffer<float, TStripeMapping>::Create(stripeMapping);

                    auto guard = profiler.Profile(TStringBuilder() << "runKernelBatch_" << kernelRuns);
                    for (ui32 k = 0; k < kernelRuns; ++k) {
                        FillBuffer(buffer, 1.0f, streams[k % streams.size()].GetId());
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestRunOnlyKernelPerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        {
            TRandom rand(0);
            const ui64 size = 4;
            const ui64 maxKernelRuns = 100001;
            const ui64 tries = 100;

            auto& profiler = GetProfiler();
            for (ui32 i = 0; i < tries; ++i) {
                TStripeMapping stripeMapping = TStripeMapping::RepeatOnAllDevices(size);
                TStripeBuffer<float> buffer = TCudaBuffer<float, TStripeMapping>::Create(stripeMapping);
                for (ui64 kernelRuns = 1; kernelRuns < maxKernelRuns; kernelRuns *= 10) {
                    auto guard = profiler.Profile(TStringBuilder() << "runKernelBatch_" << kernelRuns);
                    for (ui32 k = 0; k < kernelRuns; ++k) {
                        FillBuffer(buffer, 1.0f);
                    }
                }
            }
        }
    }

    Y_UNIT_TEST(TestMemcpyPerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
        SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);
        {
            TRandom rand(0);
            const ui64 tries = 100;

            auto& profiler = GetProfiler();
            for (int size = 10; size < 10000001; size *= 10) {
                for (ui32 i = 0; i < tries; ++i) {
                    TStripeMapping stripeMapping = TStripeMapping::RepeatOnAllDevices(size);
                    TStripeBuffer<float> from = TCudaBuffer<float, TStripeMapping>::Create(stripeMapping);
                    TStripeBuffer<float> to = TCudaBuffer<float, TStripeMapping>::Create(stripeMapping);
                    auto guard = profiler.Profile(TStringBuilder() << "memcpy_" << size);
                    from.Copy(to);
                }
            }
        }
    }

    Y_UNIT_TEST(TestRunKernelAndReadResultPerformance) {
        auto stopCudaManagerGuard = StartCudaManager();
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
    }

    Y_UNIT_TEST(BandwidthAndLatencyDeviceDevice) {
        auto stopCudaManagerGuard = StartCudaManager();

        {
            auto& stats = GetMemoryCopyPerformance<EPtrType::CudaDevice, EPtrType::CudaDevice>();

            ui64 devCount = GetCudaManager().GetDeviceCount();
            CATBOOST_INFO_LOG << "Bandwitdh MB/s" << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << BandwidthMbPerSec(stats.Bandwidth(dev, secondDev)) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }

            CATBOOST_INFO_LOG << "Bandwitdh " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << stats.Bandwidth(dev, secondDev) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }
            CATBOOST_INFO_LOG << "Latency " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                CATBOOST_INFO_LOG << "Dev #" << dev << "\t";
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << stats.Latency(dev, secondDev) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }
        }
    }

    Y_UNIT_TEST(PureCudaLatencyTest) {
        SetDevice(0);
        auto src = TCudaMemoryAllocation<EPtrType::CudaDevice>::Allocate<float>((ui64)2);
        SetDevice(1);
        auto dst = TCudaMemoryAllocation<EPtrType::CudaDevice>::Allocate<float>((ui64)2);

        TCudaStream stream = GetStreamsProvider().RequestStream();
        auto event = CreateCudaEvent();
        double val = 0;
        for (ui64 iter = 0; iter < 10000; ++iter) {
            stream.Synchronize();
            auto start = std::chrono::high_resolution_clock::now();
            TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaDevice>::CopyMemoryAsync(src, dst, (ui64)2, stream);
            event->Record(stream);
            event->WaitComplete();
            auto elapsed = std::chrono::high_resolution_clock::now() - start;
            val += std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed).count() * 1.0 / 1000;
        }
        val /= 10000;
        CATBOOST_INFO_LOG << "Latency 0-1 " << val << Endl;
    }

    Y_UNIT_TEST(BandwidthAndLatencyDeviceHost) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            auto& latencyAndBandwidth = GetMemoryCopyPerformance<EPtrType::CudaDevice, EPtrType::CudaHost>();
            ui64 devCount = GetCudaManager().GetDeviceCount();
            CATBOOST_INFO_LOG << "Bandwitdh MB/s" << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << BandwidthMbPerSec(latencyAndBandwidth.Bandwidth(dev, secondDev)) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }

            CATBOOST_INFO_LOG << "Bandwitdh " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << latencyAndBandwidth.Bandwidth(dev, secondDev) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }
            CATBOOST_INFO_LOG << "Latency " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << latencyAndBandwidth.Latency(dev, secondDev) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }
        }
    }

    Y_UNIT_TEST(BandwidthAndLatencyHostHost) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            auto& latencyAndBandwidth = GetMemoryCopyPerformance<EPtrType::CudaHost, EPtrType::CudaHost>();
            ui64 devCount = GetCudaManager().GetDeviceCount();
            CATBOOST_INFO_LOG << "Bandwitdh MB/s" << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << BandwidthMbPerSec(latencyAndBandwidth.Bandwidth(dev, secondDev)) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }

            CATBOOST_INFO_LOG << "Bandwitdh " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << latencyAndBandwidth.Bandwidth(dev, secondDev) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }
            CATBOOST_INFO_LOG << "Latency " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << latencyAndBandwidth.Latency(dev, secondDev) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }
        }
    }

    Y_UNIT_TEST(BandwidthAndLatencyHostDevice) {
        auto stopCudaManagerGuard = StartCudaManager();
        {
            auto& latencyAndBandwidth = GetMemoryCopyPerformance<EPtrType::CudaHost, EPtrType::CudaDevice>();
            ui64 devCount = GetCudaManager().GetDeviceCount();
            CATBOOST_INFO_LOG << "Bandwitdh MB/s" << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << BandwidthMbPerSec(latencyAndBandwidth.Bandwidth(dev, secondDev)) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }

            CATBOOST_INFO_LOG << "Bandwitdh " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << latencyAndBandwidth.Bandwidth(dev, secondDev) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }
            CATBOOST_INFO_LOG << "Latency " << Endl;
            for (ui64 dev = 0; dev < devCount; ++dev) {
                for (ui64 secondDev = 0; secondDev < devCount; ++secondDev) {
                    CATBOOST_INFO_LOG << latencyAndBandwidth.Latency(dev, secondDev) << "\t";
                }
                CATBOOST_INFO_LOG << Endl;
            }
        }
    }

    Y_UNIT_TEST(LatencyProfile) {
        auto& manager = NCudaLib::GetCudaManager();
        auto stopCudaManagerGuard = StartCudaManager();

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
            CATBOOST_INFO_LOG << "Latency 0-1 " << val / 100000 << Endl;
        }
    }

    Y_UNIT_TEST(BroadcastTest) {
        auto& manager = NCudaLib::GetCudaManager();
        auto stopCudaManagerGuard = StartCudaManager();

        if (manager.GetDeviceCount() > 1) {
            bool initialized = false;

            auto seeds = TSingleBuffer<ui64>::Create(TSingleMapping(1, 16384));
            MakeSequence(seeds);

            for (ui64 i = 1; i < 28; ++i) {
                TCudaProfiler profiler(EProfileMode::ImplicitLabelSync);
                profiler.SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

                const ui64 tries = 10;
                const ui64 innerTries = 10;
                //
                auto singleMapping = TSingleMapping(1, 1 << i);
                auto mirrorMapping = TMirrorMapping(1 << i);

                auto bufferSingle = TSingleBuffer<float>::Create(singleMapping);
                BayesianBootstrap(seeds, bufferSingle, 1.0f);

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

#if defined(USE_MPI)
                for (ui32 iter = 0; iter < tries; ++iter) {
                    auto guard = profiler.Profile("Broadcast compressed " + ToString(1.0 * sizeof(float) * (1 << i) / 1024 / 1024) + "MBx" + ToString(innerTries));
                    for (ui32 innerIter = 0; innerIter < innerTries; ++innerIter) {
                        //                        Reshard(bufferSingleCpu, bufferSingle);
                        Reshard(bufferSingle, bufferMirror, 0u, true);
                    }
                }
#endif
            }
        }
    }

    Y_UNIT_TEST(StripeToSingleBroadcastTest) {
        auto& manager = NCudaLib::GetCudaManager();
        auto stopCudaManagerGuard = StartCudaManager();

        if (manager.GetDeviceCount() > 1) {
            bool initialized = false;

            for (ui64 i = 4; i < 28; ++i) {
                TCudaProfiler profiler(EProfileMode::ImplicitLabelSync);
                profiler.SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

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
    }
    Y_UNIT_TEST(StripeToMirrorBroadcastTest) {
        auto& manager = NCudaLib::GetCudaManager();
        auto stopCudaManagerGuard = StartCudaManager();

        if (manager.GetDeviceCount() > 1) {
            bool initialized = false;

            for (ui64 i = 4; i < 28; ++i) {
                TCudaProfiler profiler(EProfileMode::ImplicitLabelSync);
                profiler.SetDefaultProfileMode(EProfileMode::ImplicitLabelSync);

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
    }

    Y_UNIT_TEST(TestReadAndWrite) {
        auto& manager = NCudaLib::GetCudaManager();
        auto stopCudaManagerGuard = StartCudaManager();
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
    }
}
