#include <library/cpp/containers/paged_vector/paged_vector.h>
#include <library/cpp/testing/unittest/registar.h>

#include <stdexcept>

class TPagedVectorTest: public TTestBase {
    UNIT_TEST_SUITE(TPagedVectorTest);
    UNIT_TEST(Test0)
    UNIT_TEST(Test1)
    UNIT_TEST(Test2)
    UNIT_TEST(Test3)
    UNIT_TEST(Test4)
    UNIT_TEST(Test5)
    UNIT_TEST(Test6)
    UNIT_TEST(Test7)
    UNIT_TEST(TestAt)
    UNIT_TEST(TestAutoRef)
    UNIT_TEST(TestIterators)
    UNIT_TEST(TestEmplaceBack1)
    UNIT_TEST(TestEmplaceBack2)
    //UNIT_TEST(TestEbo)
    UNIT_TEST_SUITE_END();

private:
    // Copy-paste of STLPort tests
    void Test0() {
        using NPagedVector::TPagedVector;
        TPagedVector<int, 16> v1; // Empty vector of integers.

        UNIT_ASSERT(v1.empty() == true);
        UNIT_ASSERT(v1.size() == 0);

        for (size_t i = 0; i < 256; ++i) {
            v1.resize(i + 1);
            UNIT_ASSERT_VALUES_EQUAL(v1.size(), i + 1);
        }

        for (size_t i = 256; i-- > 0;) {
            v1.resize(i);
            UNIT_ASSERT_VALUES_EQUAL(v1.size(), i);
        }
    }

    void Test1() {
        using NPagedVector::TPagedVector;
        TPagedVector<int, 3> v1; // Empty vector of integers.

        UNIT_ASSERT(v1.empty() == true);
        UNIT_ASSERT(v1.size() == 0);

        // UNIT_ASSERT(v1.max_size() == INT_MAX / sizeof(int));
        // cout << "max_size = " << v1.max_size() << endl;
        v1.push_back(42); // Add an integer to the vector.

        UNIT_ASSERT(v1.size() == 1);

        UNIT_ASSERT(v1[0] == 42);

        {
            TPagedVector<TPagedVector<int, 3>, 3> vect;
            vect.resize(10);
            UNIT_ASSERT(vect.size() == 10);
            TPagedVector<TPagedVector<int, 3>, 3>::iterator it(vect.begin()), end(vect.end());
            for (; it != end; ++it) {
                UNIT_ASSERT((*it).empty());
                UNIT_ASSERT((*it).size() == 0);
                UNIT_ASSERT((*it).begin() == (*it).end());
            }
        }
    }

    void Test2() {
        using NPagedVector::TPagedVector;
        TPagedVector<double, 3> v1; // Empty vector of doubles.
        v1.push_back(32.1);
        v1.push_back(40.5);
        v1.push_back(45.5);
        v1.push_back(33.4);
        TPagedVector<double, 3> v2; // Another empty vector of doubles.
        v2.push_back(3.56);

        UNIT_ASSERT(v1.size() == 4);
        UNIT_ASSERT(v1[0] == 32.1);
        UNIT_ASSERT(v1[1] == 40.5);
        UNIT_ASSERT(v1[2] == 45.5);
        UNIT_ASSERT(v1[3] == 33.4);

        UNIT_ASSERT(v2.size() == 1);
        UNIT_ASSERT(v2[0] == 3.56);
        v1.swap(v2); // Swap the vector's contents.

        UNIT_ASSERT(v1.size() == 1);
        UNIT_ASSERT(v1[0] == 3.56);

        UNIT_ASSERT(v2.size() == 4);
        UNIT_ASSERT(v2[0] == 32.1);
        UNIT_ASSERT(v2[1] == 40.5);
        UNIT_ASSERT(v2[2] == 45.5);
        UNIT_ASSERT(v2[3] == 33.4);

        v2 = v1; // Assign one vector to another.

        UNIT_ASSERT(v2.size() == 1);
        UNIT_ASSERT(v2[0] == 3.56);

        v2.pop_back();
        UNIT_ASSERT(v2.size() == 0);
        UNIT_ASSERT(v2.empty());
    }

    void Test3() {
        using NPagedVector::TPagedVector;
        TPagedVector<char, 1> v1;

        v1.push_back('h');
        v1.push_back('i');

        UNIT_ASSERT(v1.size() == 2);
        UNIT_ASSERT(v1[0] == 'h');
        UNIT_ASSERT(v1[1] == 'i');

        TPagedVector<char, 1> v2;
        v2.resize(v1.size());

        for (size_t i = 0; i < v1.size(); ++i)
            v2[i] = v1[i];

        v2[1] = 'o'; // Replace second character.

        UNIT_ASSERT(v2.size() == 2);
        UNIT_ASSERT(v2[0] == 'h');
        UNIT_ASSERT(v2[1] == 'o');

        UNIT_ASSERT((v1 == v2) == false);

        UNIT_ASSERT((v1 < v2) == true);
    }

    void Test4() {
        using NPagedVector::TPagedVector;
        TPagedVector<int, 3> v;
        v.resize(4);

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

        typedef NPagedVector::TPagedVector<int, 3> TVectorType;
        TVectorType v(array, array + 4);

        UNIT_ASSERT(v.size() == 4);

        UNIT_ASSERT(v[0] == 1);
        UNIT_ASSERT(v[1] == 4);
        UNIT_ASSERT(v[2] == 9);
        UNIT_ASSERT(v[3] == 16);
    }

    void Test6() {
        int array[] = {1, 4, 9, 16, 25, 36};

        typedef NPagedVector::TPagedVector<int, 3> TVectorType;
        TVectorType v(array, array + 6);
        TVectorType::iterator vit;

        UNIT_ASSERT_VALUES_EQUAL(v.size(), 6u);
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

        typedef NPagedVector::TPagedVector<int, 3> TVectorType;

        TVectorType v(array1, array1 + 3);
        TVectorType::iterator vit;
        vit = v.insert(v.begin(), 0); // Insert before first element.
        UNIT_ASSERT_VALUES_EQUAL(*vit, 0);

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

        v.clear();
        UNIT_ASSERT(v.empty());
    }

    void TestAt() {
        using NPagedVector::TPagedVector;
        TPagedVector<int, 3> v;
        TPagedVector<int, 3> const& cv = v;

        v.push_back(10);
        UNIT_ASSERT(v.at(0) == 10);
        v.at(0) = 20;
        UNIT_ASSERT(cv.at(0) == 20);

        for (;;) {
            try {
                v.at(1) = 20;
                UNIT_ASSERT(false);
            } catch (std::out_of_range const&) {
                return;
            } catch (...) {
                UNIT_ASSERT(false);
            }
        }
    }

    void TestAutoRef() {
        using NPagedVector::TPagedVector;
        typedef TPagedVector<int, 3> TVec;
        TVec ref;
        for (int i = 0; i < 5; ++i) {
            ref.push_back(i);
        }

        TPagedVector<TVec, 3> v_v_int;
        v_v_int.push_back(ref);
        v_v_int.push_back(v_v_int[0]);
        v_v_int.push_back(ref);
        v_v_int.push_back(v_v_int[0]);
        v_v_int.push_back(v_v_int[0]);
        v_v_int.push_back(ref);

        TPagedVector<TVec, 3>::iterator vvit(v_v_int.begin()), vvitEnd(v_v_int.end());
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
        using NPagedVector::TPagedVector;
        TPagedVector<int, 3> vint;
        vint.resize(10);
        TPagedVector<int, 3> const& crvint = vint;

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
        //UNIT_ASSERT(vint.rbegin() == crvint.rbegin());
        //UNIT_ASSERT(crvint.rbegin() == vint.rbegin());
        UNIT_ASSERT(crvint.rbegin() == crvint.rbegin());

        UNIT_ASSERT(vint.rbegin() != vint.rend());
        // Not Standard:
        //UNIT_ASSERT(vint.rbegin() != crvint.rend());
        //UNIT_ASSERT(crvint.rbegin() != vint.rend());
        UNIT_ASSERT(crvint.rbegin() != crvint.rend());
    }

    void TestEmplaceBack1() {
        NPagedVector::TPagedVector<int, 3> vint;

        for (int i = 0; i < 55; ++i) {
            UNIT_ASSERT_EQUAL(vint.emplace_back(i), i);
        }

        UNIT_ASSERT_EQUAL(vint.size(), 55);

        for (int i = 0; i < 55; ++i) {
            UNIT_ASSERT_EQUAL(vint[i], i);
        }
    }

    void TestEmplaceBack2() {
        using TPair = std::pair<int, TString>;
        NPagedVector::TPagedVector<TPair, 5> arr;

        for (int i = 0; i < 55; ++i) {
            auto s = ToString(i);
            auto& element = arr.emplace_back(i, s);
            UNIT_ASSERT_EQUAL(element, std::make_pair(i, s));
            UNIT_ASSERT_UNEQUAL(element, std::make_pair(i + 1, s));
        }

        UNIT_ASSERT_EQUAL(arr.size(), 55);

        for (int i = 0; i < 55; ++i) {
            UNIT_ASSERT_EQUAL(arr[i].first, i);
            UNIT_ASSERT_EQUAL(arr[i].second, ToString(i));
        }
    }

    /* This test check a potential issue with empty base class
         * optimization. Some compilers (VC6) do not implement it
         * correctly resulting ina wrong behavior. */
    void TestEbo() {
        using NPagedVector::TPagedVector;
        // We use heap memory as test failure can corrupt vector internal
        // representation making executable crash on vector destructor invocation.
        // We prefer a simple memory leak, internal corruption should be reveal
        // by size or capacity checks.
        typedef TPagedVector<int, 3> V;
        V* pv1 = new V;

        pv1->resize(1);
        pv1->at(0) = 1;

        V* pv2 = new V;

        pv2->resize(10);
        for (int i = 0; i < 10; ++i)
            pv2->at(i) = 2;

        pv1->swap(*pv2);

        UNIT_ASSERT(pv1->size() == 10);
        UNIT_ASSERT((*pv1)[5] == 2);

        UNIT_ASSERT(pv2->size() == 1);
        UNIT_ASSERT((*pv2)[0] == 1);

        delete pv2;
        delete pv1;
    }
};

UNIT_TEST_SUITE_REGISTRATION(TPagedVectorTest);
