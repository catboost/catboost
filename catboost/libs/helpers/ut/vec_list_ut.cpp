#include <catboost/libs/helpers/vec_list.h>

#include <util/generic/array_ref.h>

#include <algorithm>

#include <library/unittest/registar.h>


Y_UNIT_TEST_SUITE(TVecList) {
    Y_UNIT_TEST(Simple) {
        TVecList<ui32> vl;

        UNIT_ASSERT_VALUES_EQUAL(vl.size(), 0);

        vl.push_back(0);

        UNIT_ASSERT_VALUES_EQUAL(vl.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(*vl.begin(), 0);

        vl.erase(vl.begin());

        UNIT_ASSERT_VALUES_EQUAL(vl.size(), 0);
    }

    Y_UNIT_TEST(Simple2) {
        TVecList<ui32> vl;

        TVector<ui32> v = {1, 12, 33, 4, 3};

        for (auto e : v) {
            vl.push_back(e);
        }

        UNIT_ASSERT_VALUES_EQUAL(vl.size(), v.size());
        UNIT_ASSERT(std::equal(vl.begin(), vl.end(), v.begin(), v.end()));

        {
            auto it = vl.begin();
            vl.erase(it);
            UNIT_ASSERT(std::equal(vl.begin(), vl.end(), v.begin() + 1, v.end()));
        }
        {
            auto it = vl.begin();
            ++it;
            vl.erase(it);

            TVector<ui32> v2 = {12, 4, 3};

            UNIT_ASSERT(std::equal(vl.begin(), vl.end(), v2.begin(), v2.end()));
        }

        {
            auto it = vl.begin();
            std::advance(it, 2);
            vl.erase(it);

            TVector<ui32> v2 = {12, 4};

            UNIT_ASSERT(std::equal(vl.begin(), vl.end(), v2.begin(), v2.end()));
        }

        {
            auto it = vl.begin();
            ++it;
            vl.erase(it);

            TVector<ui32> v2 = {12};

            UNIT_ASSERT(std::equal(vl.begin(), vl.end(), v2.begin(), v2.end()));
        }

        {
            vl.push_back(22);
            vl.push_back(9);
            vl.push_back(11);

            TVector<ui32> v2 = {12, 22, 9, 11};
            UNIT_ASSERT(std::equal(vl.begin(), vl.end(), v2.begin(), v2.end()));
        }
    }

    Y_UNIT_TEST(AssignAndMove) {
        TVector<ui32> v = {1, 12, 33, 4, 3};
        TVector<ui32> vWithErased = {12, 4};

        auto getVLSimple = [&] () -> TVecList<ui32> {
            TVecList<ui32> vl;
            for (auto e : v) {
                vl.push_back(e);
            }
            return vl;
        };

        auto getVLWithErased = [&] () -> TVecList<ui32> {
            TVecList<ui32> vl;
            for (auto e : v) {
                vl.push_back(e);
            }
            auto it = vl.erase(vl.begin()); // 1
            ++it;
            it = vl.erase(it); // 33
            ++it;
            vl.erase(it); // 3

            return vl;
        };

        auto checkEqual = [&] (TConstArrayRef<ui32> canonData, const TVecList<ui32>& testData) {
            UNIT_ASSERT_VALUES_EQUAL(canonData.size(), testData.size());
            UNIT_ASSERT(std::equal(canonData.begin(), canonData.end(), testData.begin(), testData.end()));
        };

        // copy constructor
        {
            TVecList<ui32> vl = getVLSimple();
            TVecList<ui32> vl2(vl);
            checkEqual(v, vl2);
        }
        {
            TVecList<ui32> vl = getVLWithErased();
            TVecList<ui32> vl2(vl);
            checkEqual(vWithErased, vl2);
        }

        // move constructor
        {
            TVecList<ui32> vl = getVLSimple();
            TVecList<ui32> vl2(std::move(vl));
            checkEqual(v, vl2);
        }
        {
            TVecList<ui32> vl = getVLWithErased();
            TVecList<ui32> vl2(std::move(vl));
            checkEqual(vWithErased, vl2);
        }

        // assignment
        {
            TVecList<ui32> vl = getVLSimple();
            TVecList<ui32> vl2;
            vl2 = vl;
            checkEqual(v, vl2);
        }
        {
            TVecList<ui32> vl = getVLWithErased();
            TVecList<ui32> vl2;
            vl2 = vl;
            checkEqual(vWithErased, vl2);
        }

        // move assignment
        {
            TVecList<ui32> vl = getVLSimple();
            TVecList<ui32> vl2;
            vl2 = std::move(vl);
            checkEqual(v, vl2);
        }
        {
            TVecList<ui32> vl = getVLWithErased();
            TVecList<ui32> vl2;
            vl2 = std::move(vl);
            checkEqual(vWithErased, vl2);
        }
    }

    Y_UNIT_TEST(EraseAllAndAdd) {
        TVector<ui32> v = {1, 12, 9, 7, 3, 2};

        TVecList<ui32> vl;
        for (auto e : v) {
            vl.push_back(e);
        }

        auto it = vl.begin();
        while (!vl.empty()) {
            it = vl.erase(it);
        }

        TVector<ui32> v2 = {4, 5, 11};
        for (auto e : v2) {
            vl.push_back(e);
        }

        UNIT_ASSERT(std::equal(vl.begin(), vl.end(), v2.begin(), v2.end()));
    }
}
