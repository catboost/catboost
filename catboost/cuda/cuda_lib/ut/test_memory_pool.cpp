#include <catboost/cuda/cuda_lib/memory_pool/stack_like_memory_pool.h>
#include <library/cpp/testing/unittest/registar.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>
#include <catboost/libs/helpers/cpu_random.h>

using namespace NCudaLib;

Y_UNIT_TEST_SUITE(TMemoryPoolTest) {
    Y_UNIT_TEST(TestMalloc) {
        TCudaMemoryAllocation<EPtrType::CudaDevice>::FreeMemory(TCudaMemoryAllocation<EPtrType::CudaDevice>::template Allocate<char>(19));
    }

    Y_UNIT_TEST(TestStackAllocations) {
        {
            TStackLikeMemoryPool<EPtrType::Host> pool(1280 + 8192);
            using TPtr = THolder<std::remove_pointer<decltype(pool.Create(103))>::type>;

            {
                TPtr block1(pool.Create(100));
                TPtr block2(pool.Create(101));
                TPtr block3(pool.Create(102));
                TPtr block4(pool.Create(103));
                UNIT_ASSERT(block1->Get() < block2->Get());
                UNIT_ASSERT(block2->Get() < block3->Get());
                UNIT_ASSERT(block3->Get() < block4->Get());
            }
        }
    }

    Y_UNIT_TEST(TestSimpleDefragment) {
        TSetLoggingVerbose inThisScope;
        const ui64 MB = 1024 * 1024;

        TCudaStream defaultStream = GetStreamsProvider().RequestStream();
        SetDefaultStream(defaultStream);

        TStackLikeMemoryPool<EPtrType::CudaDevice> pool(512 * MB);

        using TPtr = THolder<std::remove_pointer<decltype(pool.Create(103))>::type>;
        TVector<char> tmp(255 * MB);
        TVector<char> tmp2(255 * MB);
        for (ui32 i = 0; i < tmp.size(); ++i) {
            tmp[i] = (char)i;
            tmp2[i] = 2;
        }

        {
            TVector<char> tmp0(511 * MB, 2);
            TPtr block0(pool.Create(511 * MB));
            TMemoryCopier<EPtrType::CudaHost, EPtrType::CudaDevice>::CopyMemorySync<char>(tmp0.data(), block0->Get(), 511 * MB);
        }

        TPtr block2;
        {
            TPtr block1(pool.Create(255 * MB));
            block2 = TPtr(pool.Create(255 * MB));
            TMemoryCopier<EPtrType::Host, EPtrType::CudaDevice>::CopyMemorySync<char>(tmp.data(), block2->Get(), 255 * MB);
        }

        TPtr block3(pool.Create(255 * MB));
        TMemoryCopier<EPtrType::CudaDevice, EPtrType::Host>::CopyMemorySync<char>(block2->Get(), tmp2.data(), 255 * MB);
        for (ui32 i = 0; i < tmp2.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp2[i]);
        }
    }

    Y_UNIT_TEST(TestSimpleDefragment2) {
        TSetLoggingVerbose inThisScope;
        const ui64 MB = 1024 * 1024;

        TCudaStream defaultStream = GetStreamsProvider().RequestStream();
        SetDefaultStream(defaultStream);

        TStackLikeMemoryPool<EPtrType::CudaDevice> pool(512 * MB);

        using TPtr = THolder<std::remove_pointer<decltype(pool.Create(0))>::type>;
        TVector<char> tmp1(128 * MB, 1);
        TVector<char> tmp2(120 * MB, 2);
        TVector<char> tmp3(120 * MB, 3);
        TVector<char> tmp4(120 * MB, 4);
        TVector<char> tmp5(140 * MB, 5);

        TPtr block1(pool.Create(tmp1.size()));
        TPtr block2(pool.Create(tmp2.size()));
        TPtr block3(pool.Create(tmp3.size()));
        TPtr block4(pool.Create(tmp4.size()));

        TMemoryCopier<EPtrType::CudaHost, EPtrType::CudaDevice>::CopyMemorySync<char>(tmp1.data(), block1->Get(), tmp1.size());
        TMemoryCopier<EPtrType::CudaHost, EPtrType::CudaDevice>::CopyMemorySync<char>(tmp2.data(), block2->Get(), tmp2.size());
        TMemoryCopier<EPtrType::CudaHost, EPtrType::CudaDevice>::CopyMemorySync<char>(tmp3.data(), block3->Get(), tmp3.size());
        TMemoryCopier<EPtrType::CudaHost, EPtrType::CudaDevice>::CopyMemorySync<char>(tmp4.data(), block4->Get(), tmp4.size());

        block2.Reset(nullptr);

        TPtr block5(pool.Create(tmp5.size()));

        TMemoryCopier<EPtrType::CudaHost, EPtrType::CudaDevice>::CopyMemorySync<char>(tmp5.data(), block5->Get(), tmp5.size());

        {
            TVector<char> tmp(tmp1.size());
            TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaHost>::CopyMemorySync<char>(block1->Get(), tmp.data(), tmp1.size());
            for (ui32 i = 0; i < tmp1.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp1[i]);
            }
        }

        {
            TVector<char> tmp(tmp3.size());
            TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaHost>::CopyMemorySync<char>(block3->Get(), tmp.data(), tmp3.size());
            for (ui32 i = 0; i < tmp3.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp3[i]);
            }
        }

        {
            TVector<char> tmp(tmp4.size());
            TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaHost>::CopyMemorySync<char>(block4->Get(), tmp.data(), tmp4.size());
            for (ui32 i = 0; i < tmp4.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp4[i]);
            }
        }

        {
            TVector<char> tmp(tmp5.size());
            TMemoryCopier<EPtrType::CudaDevice, EPtrType::CudaHost>::CopyMemorySync<char>(block5->Get(), tmp.data(), tmp5.size());
            for (ui32 i = 0; i < tmp5.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp5[i]);
            }
        }
    }
    //
    //
    //
    //    Y_UNIT_TEST(MemoryDefragmentStressTest) {
    //        TDeviceRequestConfig requestConfig = GetDefaultDeviceRequestConfig();
    //        requestConfig.DeviceConfig = "0";
    //        auto stopCudaManagerGuard = StartCudaManager(requestConfig, ELoggingLevel::Debug);
    //        auto& manager = NCudaLib::GetCudaManager();
    //
    //        TVector<TSingleBuffer<ui32>> data;
    //        TRandom random(0);
    //
    //        double memoryRequestSize =
    //
    //
    //    }
}
