#include "pool.h"

#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>

class TCheckedAllocator: public TDefaultAllocator {
public:
    inline TCheckedAllocator()
        : Alloced_(0)
        , Released_(0)
        , Allocs_(0)
        , Frees_(0)
    {
    }

    TBlock Allocate(size_t len) override {
        Check();

        Alloced_ += len;
        ++Allocs_;

        return TDefaultAllocator::Allocate(len);
    }

    void Release(const TBlock& block) override {
        Released_ += block.Len;
        ++Frees_;

        Check();

        TDefaultAllocator::Release(block);
    }

    inline void CheckAtEnd() {
        UNIT_ASSERT_EQUAL(Alloced_, Released_);
        UNIT_ASSERT_EQUAL(Allocs_, Frees_);
    }

private:
    inline void Check() {
        UNIT_ASSERT(Alloced_ >= Released_);
        UNIT_ASSERT(Allocs_ >= Frees_);
    }

private:
    size_t Alloced_;
    size_t Released_;
    size_t Allocs_;
    size_t Frees_;
};

class TErrorOnCopy {
public:
    TErrorOnCopy() = default;
    TErrorOnCopy(TErrorOnCopy&&) = default;

    TErrorOnCopy(const TErrorOnCopy&) {
        UNIT_ASSERT(false);
    }
};

class TNoCopy {
public:
    TNoCopy() = default;
    TNoCopy(TNoCopy&&) = default;

    TNoCopy(const TNoCopy&) = delete;
};

class TNoMove {
public:
    TNoMove() = default;
    TNoMove(const TNoMove&) = default;

    TNoMove(TNoMove&&) = delete;
};

class TMemPoolTest: public TTestBase {
    UNIT_TEST_SUITE(TMemPoolTest);
    UNIT_TEST(TestMemPool)
    UNIT_TEST(TestAlign)
    UNIT_TEST(TestZeroArray)
    UNIT_TEST(TestLargeStartingAlign)
    UNIT_TEST(TestMoveAlloc)
    UNIT_TEST(TestRoundUpToNextPowerOfTwoOption)
    UNIT_TEST(TestMemoryPoolBookmark)
    UNIT_TEST_SUITE_END();

private:
    inline void TestMemPool() {
        TCheckedAllocator alloc;

        {
            TMemoryPool pool(123, TMemoryPool::TExpGrow::Instance(), &alloc);

            for (size_t i = 0; i < 1000; ++i) {
                UNIT_ASSERT(pool.Allocate(i));
            }
        }

        alloc.CheckAtEnd();

        {
            TMemoryPool pool(150, TMemoryPool::TExpGrow::Instance(), &alloc);

            pool.Allocate(8);

            size_t memavail = pool.Available();
            size_t memwaste = pool.MemoryWaste();
            size_t memalloc = pool.MemoryAllocated();

            for (size_t i = 0; i < 1000; ++i) {
                void* m = pool.Allocate(i);
                UNIT_ASSERT(m);
                memset(m, 0, i);
            }

            UNIT_ASSERT_VALUES_EQUAL(pool.ClearReturnUsedChunkCount(true), 11);

            UNIT_ASSERT_VALUES_EQUAL(memalloc - 8, pool.MemoryAllocated());
            UNIT_ASSERT_VALUES_EQUAL(memwaste + 8, pool.MemoryWaste());
            UNIT_ASSERT_VALUES_EQUAL(memavail + 8, pool.Available());

            for (size_t i = 0; i < 1000; ++i) {
                void* m = pool.Allocate(i);
                UNIT_ASSERT(m);
                memset(m, 0, i);
            }

            UNIT_ASSERT_VALUES_EQUAL(pool.ClearReturnUsedChunkCount(false), 12);

            UNIT_ASSERT_VALUES_EQUAL(0, pool.MemoryAllocated());
            UNIT_ASSERT_VALUES_EQUAL(0, pool.MemoryWaste());
            UNIT_ASSERT_VALUES_EQUAL(0, pool.Available());
        }

        alloc.CheckAtEnd();

        struct TConstructorTest {
            int ConstructorType;
            TConstructorTest()
                : ConstructorType(1)
            {
            }
            TConstructorTest(int)
                : ConstructorType(2)
            {
            }
            TConstructorTest(const TString&, const TString&)
                : ConstructorType(3)
            {
            }
            TConstructorTest(TString&&, TString&&)
                : ConstructorType(4)
            {
            }
        };

        {
            TMemoryPool pool(123, TMemoryPool::TExpGrow::Instance(), &alloc);
            THolder<TConstructorTest, TDestructor> data1{pool.New<TConstructorTest>()};
            THolder<TConstructorTest, TDestructor> data2{pool.New<TConstructorTest>(42)};
            THolder<TConstructorTest, TDestructor> data3{pool.New<TConstructorTest>("hello", "world")};
            UNIT_ASSERT_VALUES_EQUAL(data1->ConstructorType, 1);
            UNIT_ASSERT_VALUES_EQUAL(data2->ConstructorType, 2);
            UNIT_ASSERT_VALUES_EQUAL(data3->ConstructorType, 4);
        }

        alloc.CheckAtEnd();
    }

    inline void TestAlign() {
        TMemoryPool pool(1);

        void* aligned16 = pool.Allocate(3, 16);
        void* aligned2 = pool.Allocate(3, 2);
        void* aligned128 = pool.Allocate(3, 128);
        void* aligned4 = pool.Allocate(3, 4);
        void* aligned256 = pool.Allocate(3, 256);
        void* aligned8 = pool.Allocate(3, 8);
        void* aligned1024 = pool.Allocate(3, 1024);

        UNIT_ASSERT_VALUES_UNEQUAL(aligned16, nullptr);
        UNIT_ASSERT_VALUES_UNEQUAL(aligned2, nullptr);
        UNIT_ASSERT_VALUES_UNEQUAL(aligned128, nullptr);
        UNIT_ASSERT_VALUES_UNEQUAL(aligned4, nullptr);
        UNIT_ASSERT_VALUES_UNEQUAL(aligned256, nullptr);
        UNIT_ASSERT_VALUES_UNEQUAL(aligned8, nullptr);
        UNIT_ASSERT_VALUES_UNEQUAL(aligned1024, nullptr);

        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned2) & 1, 0);
        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned4) & 3, 0);
        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned8) & 7, 0);
        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned16) & 15, 0);
        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned128) & 127, 0);
        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned256) & 255, 0);
        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned1024) & 1023, 0);
    }

    void TestZeroArray() {
        TMemoryPool pool(1);
        size_t size = 10;
        i32* intArray = pool.AllocateZeroArray<i32>(size);
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT(intArray[i] == 0);
        }

        size_t align = 256;
        ui8* byteArray = pool.AllocateZeroArray<ui8>(size, align);
        UNIT_ASSERT(size_t(byteArray) % align == 0);
        for (size_t i = 0; i < size; ++i) {
            UNIT_ASSERT(byteArray[i] == 0);
        }
    }

    void TestLargeStartingAlign() {
        TMemoryPool pool(1);

        void* aligned4k1 = pool.Allocate(1, 4096);
        void* aligned4k2 = pool.Allocate(1, 4096);
        UNIT_ASSERT_VALUES_UNEQUAL(aligned4k1, nullptr);
        UNIT_ASSERT_VALUES_UNEQUAL(aligned4k2, nullptr);
        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned4k1) & 4095, 0);
        UNIT_ASSERT_VALUES_EQUAL(reinterpret_cast<uintptr_t>(aligned4k2) & 4095, 0);
    }

    template <typename T>
    void CheckMoveAlloc() {
        TMemoryPool pool(10 * sizeof(T));

        TVector<T, TPoolAllocator> elems(&pool);
        elems.reserve(1);
        elems.emplace_back();
        elems.resize(100);
    }

    void TestMoveAlloc() {
        CheckMoveAlloc<TNoMove>();
        CheckMoveAlloc<TNoCopy>();
        CheckMoveAlloc<TErrorOnCopy>();
    }

    void TestRoundUpToNextPowerOfTwoOption() {
        const size_t MEMORY_POOL_BLOCK_SIZE = (1024 - 16) * 4096 - 16 - 16 - 32;

        class TFixedBlockSizeMemoryPoolPolicy final: public TMemoryPool::IGrowPolicy {
        public:
            size_t Next(size_t /*prev*/) const noexcept override {
                return MEMORY_POOL_BLOCK_SIZE;
            }
        };
        TFixedBlockSizeMemoryPoolPolicy allocationPolicy;

        class TTestAllocator final: public TDefaultAllocator {
        public:
            TBlock Allocate(size_t len) override {
                Size_ += len;
                return TDefaultAllocator::Allocate(len);
            }

            size_t GetSize() const {
                return Size_;
            }

        private:
            size_t Size_ = 0;
        };

        TTestAllocator allocator;

        TMemoryPool::TOptions options;
        options.RoundUpToNextPowerOfTwo = false;

        constexpr size_t EXPECTED_ALLOCATION_SIZE = MEMORY_POOL_BLOCK_SIZE + 32;
        TMemoryPool pool(MEMORY_POOL_BLOCK_SIZE, &allocationPolicy, &allocator, options);

        pool.Allocate(MEMORY_POOL_BLOCK_SIZE);
        UNIT_ASSERT_VALUES_EQUAL(EXPECTED_ALLOCATION_SIZE, allocator.GetSize());

        pool.Allocate(1);
        UNIT_ASSERT_VALUES_EQUAL(2 * EXPECTED_ALLOCATION_SIZE, allocator.GetSize());
    }

    void TestMemoryPoolBookmark() {
        TCheckedAllocator alloc;

        {
            TMemoryPool pool(200U, TMemoryPool::TExpGrow::Instance(), &alloc);
            ui64* someData = pool.Allocate<ui64>();
            static const ui64 TESTING{0x123456789ABCDEF};
            *someData = TESTING;

            const auto ma = pool.MemoryAllocated();
            const auto chunkOverhead = ma - sizeof(ui64);
            const auto firstChunkTotal = pool.MemoryWaste() + ma;

            // Allocate some memory in pool but not enough to need new chunks:
            {
                TMemoryPool::TBookmark bookmarkA(pool);
                for (size_t i = 0U; i != 10; ++i) {
                    UNIT_ASSERT(pool.Allocate<ui64>());
                }
            }
            UNIT_ASSERT_VALUES_EQUAL(pool.MemoryAllocated(), ma);
            UNIT_ASSERT_VALUES_EQUAL(*someData, TESTING);

            // Allocate some memory in pool enough to need a new single chunk:
            {
                TMemoryPool::TBookmark bookmarkB(pool);
                for (size_t i = 0U; i != 50; ++i) {
                    UNIT_ASSERT(pool.Allocate<ui64>());
                }
            }
            UNIT_ASSERT_VALUES_EQUAL(pool.MemoryAllocated(), firstChunkTotal + chunkOverhead); // The last (second) chunk is completely free
            UNIT_ASSERT_VALUES_EQUAL(*someData, TESTING);
        }

        alloc.CheckAtEnd();
    }
};

UNIT_TEST_SUITE_REGISTRATION(TMemPoolTest);
