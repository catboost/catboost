#include <catboost/cuda/cuda_lib/gpu_memory_pool.h>
#include <library/unittest/registar.h>
#include <catboost/cuda/cuda_lib/cuda_manager.h>
#include <catboost/cuda/cuda_lib/cuda_buffer.h>

using namespace NCudaLib;

SIMPLE_UNIT_TEST_SUITE(TMemoryPoolTest) {
    SIMPLE_UNIT_TEST(TestMalloc) {
        TCudaMemoryAllocation<CudaDevice>::FreeMemory(TCudaMemoryAllocation<CudaDevice>::template Allocate<char>(19));
    }

    SIMPLE_UNIT_TEST(TestStackAllocations) {
        {
            TStackLikeMemoryPool<Host> pool(1280);
            using TPtr = THolder<std::remove_pointer<decltype(pool.Create(103))>::type>;

            {
                TPtr block1 = pool.Create(100);
                TPtr block2 = pool.Create(101);
                TPtr block3 = pool.Create(102);
                TPtr block4 = pool.Create(103);
                UNIT_ASSERT(block1->Get() < block2->Get());
                UNIT_ASSERT(block2->Get() < block3->Get());
                UNIT_ASSERT(block3->Get() < block4->Get());
            }
        }
    }

    SIMPLE_UNIT_TEST(TestSimpleDefragment) {
        SetVerboseLogingMode();
        const ui64 MB = 1024 * 1024;

        TCudaStream defaultStream;
        SetDefaultStream(defaultStream);

        TStackLikeMemoryPool<CudaDevice> pool(512 * MB);

        using TPtr = THolder<std::remove_pointer<decltype(pool.Create(103))>::type>;
        yvector<char> tmp(255 * MB);
        yvector<char> tmp2(255 * MB);
        for (ui32 i = 0; i < tmp.size(); ++i) {
            tmp[i] = (char)i;
            tmp2[i] = 2;
        }

        {
            yvector<char> tmp0(511 * MB, 2);
            TPtr block0 = pool.Create(511 * MB);
            TMemoryCopier<CudaHost, CudaDevice>::CopyMemorySync<char>(~tmp0, block0->Get(), 511 * MB);
        }

        TPtr block2;
        {
            TPtr block1 = pool.Create(255 * MB);
            block2 = pool.Create(255 * MB);
            TMemoryCopier<Host, CudaDevice>::CopyMemorySync<char>(~tmp, block2->Get(), 255 * MB);
        }

        TPtr block3 = pool.Create(255 * MB);
        TMemoryCopier<CudaDevice, Host>::CopyMemorySync<char>(block2->Get(), ~tmp2, 255 * MB);
        for (ui32 i = 0; i < tmp2.size(); ++i) {
            UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp2[i]);
        }
    }

    SIMPLE_UNIT_TEST(TestSimpleDefragment2) {
        SetVerboseLogingMode();
        const ui64 MB = 1024 * 1024;

        TCudaStream defaultStream;
        SetDefaultStream(defaultStream);

        TStackLikeMemoryPool<CudaDevice> pool(512 * MB);

        using TPtr = THolder<std::remove_pointer<decltype(pool.Create(0))>::type>;
        yvector<char> tmp1(128 * MB, 1);
        yvector<char> tmp2(120 * MB, 2);
        yvector<char> tmp3(120 * MB, 3);
        yvector<char> tmp4(120 * MB, 4);
        yvector<char> tmp5(140 * MB, 5);

        TPtr block1 = pool.Create(tmp1.size());
        TPtr block2 = pool.Create(tmp2.size());
        TPtr block3 = pool.Create(tmp3.size());
        TPtr block4 = pool.Create(tmp4.size());

        TMemoryCopier<CudaHost, CudaDevice>::CopyMemorySync<char>(~tmp1, block1->Get(), tmp1.size());
        TMemoryCopier<CudaHost, CudaDevice>::CopyMemorySync<char>(~tmp2, block2->Get(), tmp2.size());
        TMemoryCopier<CudaHost, CudaDevice>::CopyMemorySync<char>(~tmp3, block3->Get(), tmp3.size());
        TMemoryCopier<CudaHost, CudaDevice>::CopyMemorySync<char>(~tmp4, block4->Get(), tmp4.size());

        block2.Reset(nullptr);

        TPtr block5 = pool.Create(tmp5.size());

        TMemoryCopier<CudaHost, CudaDevice>::CopyMemorySync<char>(~tmp5, block5->Get(), tmp5.size());

        {
            yvector<char> tmp(tmp1.size());
            TMemoryCopier<CudaDevice, CudaHost>::CopyMemorySync<char>(block1->Get(), ~tmp, tmp1.size());
            for (ui32 i = 0; i < tmp1.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp1[i]);
            }
        }

        {
            yvector<char> tmp(tmp3.size());
            TMemoryCopier<CudaDevice, CudaHost>::CopyMemorySync<char>(block3->Get(), ~tmp, tmp3.size());
            for (ui32 i = 0; i < tmp3.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp3[i]);
            }
        }

        {
            yvector<char> tmp(tmp4.size());
            TMemoryCopier<CudaDevice, CudaHost>::CopyMemorySync<char>(block4->Get(), ~tmp, tmp4.size());
            for (ui32 i = 0; i < tmp4.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp4[i]);
            }
        }

        {
            yvector<char> tmp(tmp5.size());
            TMemoryCopier<CudaDevice, CudaHost>::CopyMemorySync<char>(block5->Get(), ~tmp, tmp5.size());
            for (ui32 i = 0; i < tmp5.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(tmp[i], tmp5[i]);
            }
        }
    }
}
