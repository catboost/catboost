#include "vector.h"

#include <library/cpp/testing/unittest/registar.h>

#include <utility>
#include "yexception.h"

#include <stdexcept>

class TYVectorTest: public TTestBase {
    UNIT_TEST_SUITE(TYVectorTest);
    UNIT_TEST(TestConstructorsAndAssignments)
    UNIT_TEST(TestTildeEmptyToNull)
    UNIT_TEST(TestTilde)
    UNIT_TEST(Test1)
    UNIT_TEST(Test2)
    UNIT_TEST(Test3)
    UNIT_TEST(Test4)
    UNIT_TEST(Test5)
    UNIT_TEST(Test6)
    UNIT_TEST(Test7)
    UNIT_TEST(TestCapacity)
    UNIT_TEST(TestAt)
    UNIT_TEST(TestPointer)
    UNIT_TEST(TestAutoRef)
    UNIT_TEST(TestIterators)
    UNIT_TEST(TestShrink)
    // UNIT_TEST(TestEbo)
    UNIT_TEST(TestFillInConstructor)
    UNIT_TEST(TestYResize)
    UNIT_TEST(TestCrop)
    UNIT_TEST(TestInitializeList)
    UNIT_TEST_SUITE_END();

private:
    void TestConstructorsAndAssignments() {
        using container = TVector<int>;

        container c1;
        c1.push_back(100);
        c1.push_back(200);

        container c2(c1);

        UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
        UNIT_ASSERT_VALUES_EQUAL(2, c2.size());
        UNIT_ASSERT_VALUES_EQUAL(100, c1.at(0));
        UNIT_ASSERT_VALUES_EQUAL(200, c2.at(1));

        container c3(std::move(c1));

        UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
        UNIT_ASSERT_VALUES_EQUAL(2, c3.size());
        UNIT_ASSERT_VALUES_EQUAL(100, c3.at(0));

        c2.push_back(300);
        c3 = c2;

        UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
        UNIT_ASSERT_VALUES_EQUAL(3, c3.size());
        UNIT_ASSERT_VALUES_EQUAL(300, c3.at(2));

        c2.push_back(400);
        c3 = std::move(c2);

        UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
        UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
        UNIT_ASSERT_VALUES_EQUAL(400, c3.at(3));
    }

    inline void TestTildeEmptyToNull() {
        TVector<int> v;
        UNIT_ASSERT_EQUAL(nullptr, v.data());
    }

    inline void TestTilde() {
        TVector<int> v;
        v.push_back(10);
        v.push_back(20);

        UNIT_ASSERT_EQUAL(10, (v.data())[0]);
        UNIT_ASSERT_EQUAL(20, (v.data())[1]);

        for (int i = 0; i < 10000; ++i) {
            v.push_back(99);
        }

        UNIT_ASSERT_EQUAL(10, (v.data())[0]);
        UNIT_ASSERT_EQUAL(20, (v.data())[1]);
        UNIT_ASSERT_EQUAL(99, (v.data())[3]);
        UNIT_ASSERT_EQUAL(99, (v.data())[4]);
    }

    // Copy-paste of STLPort tests

    void Test1() {
        TVector<int> v1; // Empty vector of integers.

        UNIT_ASSERT(v1.empty() == true);
        UNIT_ASSERT(v1.size() == 0);
        UNIT_ASSERT(!v1);

        // UNIT_ASSERT(v1.max_size() == INT_MAX / sizeof(int));
        // cout << "max_size = " << v1.max_size() << endl;
        v1.push_back(42); // Add an integer to the vector.

        UNIT_ASSERT(v1.size() == 1);
        UNIT_ASSERT(v1);

        UNIT_ASSERT(v1[0] == 42);

        {
            TVector<TVector<int>> vect(10);
            TVector<TVector<int>>::iterator it(vect.begin()), end(vect.end());
            for (; it != end; ++it) {
                UNIT_ASSERT((*it).empty());
                UNIT_ASSERT((*it).size() == 0);
                UNIT_ASSERT((*it).capacity() == 0);
                UNIT_ASSERT((*it).begin() == (*it).end());
            }
        }
    }

    void Test2() {
        TVector<double> v1; // Empty vector of doubles.
        v1.push_back(32.1);
        v1.push_back(40.5);
        TVector<double> v2; // Another empty vector of doubles.
        v2.push_back(3.56);

        UNIT_ASSERT(v1.size() == 2);
        UNIT_ASSERT(v1[0] == 32.1);
        UNIT_ASSERT(v1[1] == 40.5);

        UNIT_ASSERT(v2.size() == 1);
        UNIT_ASSERT(v2[0] == 3.56);
        v1.swap(v2); // Swap the vector's contents.

        UNIT_ASSERT(v1.size() == 1);
        UNIT_ASSERT(v1[0] == 3.56);

        UNIT_ASSERT(v2.size() == 2);
        UNIT_ASSERT(v2[0] == 32.1);
        UNIT_ASSERT(v2[1] == 40.5);

        v2 = v1; // Assign one vector to another.

        UNIT_ASSERT(v2.size() == 1);
        UNIT_ASSERT(v2[0] == 3.56);
    }

    void Test3() {
        using vec_type = TVector<char>;

        vec_type v1; // Empty vector of characters.
        v1.push_back('h');
        v1.push_back('i');

        UNIT_ASSERT(v1.size() == 2);
        UNIT_ASSERT(v1[0] == 'h');
        UNIT_ASSERT(v1[1] == 'i');

        vec_type v2(v1.begin(), v1.end());
        v2[1] = 'o'; // Replace second character.

        UNIT_ASSERT(v2.size() == 2);
        UNIT_ASSERT(v2[0] == 'h');
        UNIT_ASSERT(v2[1] == 'o');

        UNIT_ASSERT((v1 == v2) == false);

        UNIT_ASSERT((v1 < v2) == true);
    }

    void Test4() {
        TVector<int> v(4);

        v[0] = 1;
        v[1] = 4;
        v[2] = 9;
        v[3] = 16;

        UNIT_ASSERT(v.front() == 1);
        UNIT_ASSERT(v.back() == 16);

        v.push_back(25);

        UNIT_ASSERT(v.back() == 25);
        UNIT_ASSERT(v.size() == 5);

        v.pop_back();

        UNIT_ASSERT(v.back() == 16);
        UNIT_ASSERT(v.size() == 4);
    }

    void Test5() {
        int array[] = {1, 4, 9, 16};

        TVector<int> v(array, array + 4);

        UNIT_ASSERT(v.size() == 4);

        UNIT_ASSERT(v[0] == 1);
        UNIT_ASSERT(v[1] == 4);
        UNIT_ASSERT(v[2] == 9);
        UNIT_ASSERT(v[3] == 16);
    }

    void Test6() {
        int array[] = {1, 4, 9, 16, 25, 36};

        TVector<int> v(array, array + 6);
        TVector<int>::iterator vit;

        UNIT_ASSERT(v.size() == 6);
        UNIT_ASSERT(v[0] == 1);
        UNIT_ASSERT(v[1] == 4);
        UNIT_ASSERT(v[2] == 9);
        UNIT_ASSERT(v[3] == 16);
        UNIT_ASSERT(v[4] == 25);
        UNIT_ASSERT(v[5] == 36);

        vit = v.erase(v.begin()); // Erase first element.
        UNIT_ASSERT(*vit == 4);

        UNIT_ASSERT(v.size() == 5);
        UNIT_ASSERT(v[0] == 4);
        UNIT_ASSERT(v[1] == 9);
        UNIT_ASSERT(v[2] == 16);
        UNIT_ASSERT(v[3] == 25);
        UNIT_ASSERT(v[4] == 36);

        vit = v.erase(v.end() - 1); // Erase last element.
        UNIT_ASSERT(vit == v.end());

        UNIT_ASSERT(v.size() == 4);
        UNIT_ASSERT(v[0] == 4);
        UNIT_ASSERT(v[1] == 9);
        UNIT_ASSERT(v[2] == 16);
        UNIT_ASSERT(v[3] == 25);

        v.erase(v.begin() + 1, v.end() - 1); // Erase all but first and last.

        UNIT_ASSERT(v.size() == 2);
        UNIT_ASSERT(v[0] == 4);
        UNIT_ASSERT(v[1] == 25);
    }

    void Test7() {
        int array1[] = {1, 4, 25};
        int array2[] = {9, 16};

        TVector<int> v(array1, array1 + 3);
        TVector<int>::iterator vit;
        vit = v.insert(v.begin(), 0); // Insert before first element.
        UNIT_ASSERT(*vit == 0);

        vit = v.insert(v.end(), 36); // Insert after last element.
        UNIT_ASSERT(*vit == 36);

        UNIT_ASSERT(v.size() == 5);
        UNIT_ASSERT(v[0] == 0);
        UNIT_ASSERT(v[1] == 1);
        UNIT_ASSERT(v[2] == 4);
        UNIT_ASSERT(v[3] == 25);
        UNIT_ASSERT(v[4] == 36);

        // Insert contents of array2 before fourth element.
        v.insert(v.begin() + 3, array2, array2 + 2);

        UNIT_ASSERT(v.size() == 7);

        UNIT_ASSERT(v[0] == 0);
        UNIT_ASSERT(v[1] == 1);
        UNIT_ASSERT(v[2] == 4);
        UNIT_ASSERT(v[3] == 9);
        UNIT_ASSERT(v[4] == 16);
        UNIT_ASSERT(v[5] == 25);
        UNIT_ASSERT(v[6] == 36);

        size_t curCapacity = v.capacity();
        v.clear();
        UNIT_ASSERT(v.empty());

        // check that clear save reserved data
        UNIT_ASSERT_EQUAL(curCapacity, v.capacity());

        v.insert(v.begin(), 5, 10);
        UNIT_ASSERT(v.size() == 5);
        UNIT_ASSERT(v[0] == 10);
        UNIT_ASSERT(v[1] == 10);
        UNIT_ASSERT(v[2] == 10);
        UNIT_ASSERT(v[3] == 10);
        UNIT_ASSERT(v[4] == 10);
    }

    struct TestStruct {
        unsigned int a[3];
    };

    void TestCapacity() {
        {
            TVector<int> v;

            UNIT_ASSERT(v.capacity() == 0);
            v.push_back(42);
            UNIT_ASSERT(v.capacity() >= 1);
            v.reserve(5000);
            UNIT_ASSERT(v.capacity() >= 5000);
        }

        {
            TVector<int> v(Reserve(100));

            UNIT_ASSERT(v.capacity() >= 100);
            UNIT_ASSERT(v.size() == 0);
        }

        {
            // Test that used to generate an assertion when using __debug_alloc.
            TVector<TestStruct> va;
            va.reserve(1);
            va.reserve(2);
        }
    }

    void TestAt() {
        TVector<int> v;
        TVector<int> const& cv = v;

        v.push_back(10);
        UNIT_ASSERT(v.at(0) == 10);
        v.at(0) = 20;
        UNIT_ASSERT(cv.at(0) == 20);

        UNIT_ASSERT_EXCEPTION(v.at(1) = 20, std::out_of_range);
    }

    void TestPointer() {
        TVector<int*> v1;
        TVector<int*> v2 = v1;
        TVector<int*> v3;

        v3.insert(v3.end(), v1.begin(), v1.end());
    }

    void TestAutoRef() {
        TVector<int> ref;
        for (int i = 0; i < 5; ++i) {
            ref.push_back(i);
        }

        TVector<TVector<int>> v_v_int(1, ref);
        v_v_int.push_back(v_v_int[0]);
        v_v_int.push_back(ref);
        v_v_int.push_back(v_v_int[0]);
        v_v_int.push_back(v_v_int[0]);
        v_v_int.push_back(ref);

        TVector<TVector<int>>::iterator vvit(v_v_int.begin()), vvitEnd(v_v_int.end());
        for (; vvit != vvitEnd; ++vvit) {
            UNIT_ASSERT(*vvit == ref);
        }
    }

    struct Point {
        int x, y;
    };

    struct PointEx: public Point {
        PointEx()
            : builtFromBase(false)
        {
        }
        PointEx(const Point&)
            : builtFromBase(true)
        {
        }

        bool builtFromBase;
    };

    void TestIterators() {
        TVector<int> vint(10, 0);
        TVector<int> const& crvint = vint;

        UNIT_ASSERT(vint.begin() == vint.begin());
        UNIT_ASSERT(crvint.begin() == vint.begin());
        UNIT_ASSERT(vint.begin() == crvint.begin());
        UNIT_ASSERT(crvint.begin() == crvint.begin());

        UNIT_ASSERT(vint.begin() != vint.end());
        UNIT_ASSERT(crvint.begin() != vint.end());
        UNIT_ASSERT(vint.begin() != crvint.end());
        UNIT_ASSERT(crvint.begin() != crvint.end());

        UNIT_ASSERT(vint.rbegin() == vint.rbegin());
        // Not Standard:
        // UNIT_ASSERT(vint.rbegin() == crvint.rbegin());
        // UNIT_ASSERT(crvint.rbegin() == vint.rbegin());
        UNIT_ASSERT(crvint.rbegin() == crvint.rbegin());

        UNIT_ASSERT(vint.rbegin() != vint.rend());
        // Not Standard:
        // UNIT_ASSERT(vint.rbegin() != crvint.rend());
        // UNIT_ASSERT(crvint.rbegin() != vint.rend());
        UNIT_ASSERT(crvint.rbegin() != crvint.rend());
    }

    void TestShrink() {
        TVector<int> v;
        v.resize(1000);
        v.resize(10);
        v.shrink_to_fit();
        UNIT_ASSERT_EQUAL(v.capacity(), 10);
        v.push_back(0);
        v.shrink_to_fit();
        UNIT_ASSERT_EQUAL(v.capacity(), 11);
    }

    /* This test check a potential issue with empty base class
     * optimization. Some compilers (VC6) do not implement it
     * correctly resulting ina wrong behavior. */
    void TestEbo() {
        // We use heap memory as test failure can corrupt vector internal
        // representation making executable crash on vector destructor invocation.
        // We prefer a simple memory leak, internal corruption should be reveal
        // by size or capacity checks.
        using V = TVector<int>;
        V* pv1 = new V(1, 1);
        V* pv2 = new V(10, 2);

        size_t v1Capacity = pv1->capacity();
        size_t v2Capacity = pv2->capacity();

        pv1->swap(*pv2);

        UNIT_ASSERT(pv1->size() == 10);
        UNIT_ASSERT(pv1->capacity() == v2Capacity);
        UNIT_ASSERT((*pv1)[5] == 2);

        UNIT_ASSERT(pv2->size() == 1);
        UNIT_ASSERT(pv2->capacity() == v1Capacity);
        UNIT_ASSERT((*pv2)[0] == 1);

        delete pv2;
        delete pv1;
    }

    void TestFillInConstructor() {
        for (int k = 0; k < 3; ++k) {
            TVector<int> v(100);
            UNIT_ASSERT_VALUES_EQUAL(100u, v.size());
            for (size_t i = 0; i < v.size(); ++i) {
                UNIT_ASSERT_VALUES_EQUAL(0, v[i]);
            }
            // fill with garbage for the next iteration
            for (size_t i = 0; i < v.size(); ++i) {
                v[i] = 10;
            }
        }
    }

    struct TPod {
        int x;

        operator int() {
            return x;
        }
    };

    struct TNonPod {
        int x;
        TNonPod() {
            x = 0;
        }

        operator int() {
            return x;
        }
    };

    template <typename T>
    class TDebugAlloc: public std::allocator<T> {
    public:
        using TBase = std::allocator<T>;

        T* allocate(typename TBase::size_type n) {
            auto p = TBase::allocate(n);
            for (size_t i = 0; i < n; ++i) {
                memset(p + i, 0xab, sizeof(T));
            }
            return p;
        }

        template <class TOther>
        struct rebind {
            using other = TDebugAlloc<TOther>;
        };
    };

    template <typename T>
    void TestYResize() {
#ifdef _YNDX_LIBCXX_ENABLE_VECTOR_POD_RESIZE_UNINITIALIZED
        constexpr bool ALLOW_UNINITIALIZED = std::is_pod_v<T>;
#else
        constexpr bool ALLOW_UNINITIALIZED = false;
#endif

        TVector<T, TDebugAlloc<T>> v;

        v.reserve(5);
        auto firstBegin = v.begin();

        v.yresize(5); // No realloc, no initialization if allowed
        UNIT_ASSERT(firstBegin == v.begin());
        for (int i = 0; i < 5; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(bool(v[i]), ALLOW_UNINITIALIZED);
        }

        v.yresize(20); // Realloc, still no initialization
        UNIT_ASSERT(firstBegin != v.begin());
        for (int i = 0; i < 20; ++i) {
            UNIT_ASSERT_VALUES_EQUAL(bool(v[i]), ALLOW_UNINITIALIZED);
        }
    }

    struct TNoDefaultConstructor {
        TNoDefaultConstructor() = delete;
        explicit TNoDefaultConstructor(int val)
            : Val(val)
        {
        }

        int Val;
    };

    void TestCrop() {
        TVector<TNoDefaultConstructor> vec;
        vec.emplace_back(42);
        vec.emplace_back(1337);
        vec.emplace_back(8888);
        vec.crop(1); // Should not require default constructor
        UNIT_ASSERT(vec.size() == 1);
        UNIT_ASSERT(vec[0].Val == 42);
        vec.crop(50); // Does nothing if new size is greater than the current size()
        UNIT_ASSERT(vec.size() == 1);
        UNIT_ASSERT(vec[0].Val == 42);
    }

    void TestYResize() {
        TestYResize<int>();
        TestYResize<TPod>();
        TestYResize<TNonPod>();
    }

    void CheckInitializeList(const TVector<int>& v) {
        for (size_t i = 0; i < v.size(); ++i) {
            UNIT_ASSERT_EQUAL(v[i], static_cast<int>(i));
        }
    }

    void TestInitializeList() {
        {
            TVector<int> v;
            v.assign({0, 1, 2});
            CheckInitializeList(v);
        }
        {
            TVector<int> v = {0, 1, 2};
            CheckInitializeList(v);
        }
        {
            TVector<int> v;
            v = {0, 1, 2};
            CheckInitializeList(v);
        }
        {
            TVector<int> v = {0, 3};
            v.insert(v.begin() + 1, {1, 2});
            CheckInitializeList(v);
        }
    }
};

UNIT_TEST_SUITE_REGISTRATION(TYVectorTest);
