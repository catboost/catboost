//===- llvm/unittest/ADT/SmallSetTest.cpp ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// CompactSet unit tests.
//
//===----------------------------------------------------------------------===//

#include <library/cpp/yt/small_containers/compact_set.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <string>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

TEST(TCompactSetTest, Insert)
{
    TCompactSet<int, 4> s1;

    for (int i = 0; i < 4; i++)
        s1.insert(i);

    for (int i = 0; i < 4; i++)
        s1.insert(i);

    EXPECT_EQ(4u, s1.size());

    for (int i = 0; i < 4; i++)
        EXPECT_EQ(1u, s1.count(i));

    EXPECT_EQ(0u, s1.count(4));
}

TEST(TCompactSetTest, Grow)
{
    TCompactSet<int, 4> s1;

    for (int i = 0; i < 8; i++)
        s1.insert(i);

    EXPECT_EQ(8u, s1.size());

    for (int i = 0; i < 8; i++)
        EXPECT_EQ(1u, s1.count(i));

    EXPECT_EQ(0u, s1.count(8));
}

TEST(TCompactSetTest, Erase)
{
    TCompactSet<int, 4> s1;

    for (int i = 0; i < 8; i++)
        s1.insert(i);

    EXPECT_EQ(8u, s1.size());

    // Remove elements one by one and check if all other elements are still there.
    for (int i = 0; i < 8; i++) {
        EXPECT_EQ(1u, s1.count(i));
        EXPECT_TRUE(s1.erase(i));
        EXPECT_EQ(0u, s1.count(i));
        EXPECT_EQ(8u - i - 1, s1.size());
        for (int j = i + 1; j < 8; j++)
            EXPECT_EQ(1u, s1.count(j));
    }

    EXPECT_EQ(0u, s1.count(8));
}

TEST(TCompactSetTest, IteratorInt)
{
    TCompactSet<int, 4> s1;

    // Test the 'small' case.
    for (int i = 0; i < 3; i++)
        s1.insert(i);

    std::vector<int> V(s1.begin(), s1.end());
    // Make sure the elements are in the expected order.
    std::sort(V.begin(), V.end());
    for (int i = 0; i < 3; i++)
        EXPECT_EQ(i, V[i]);

    // Test the 'big' case by adding a few more elements to switch to std::set
    // internally.
    for (int i = 3; i < 6; i++)
        s1.insert(i);

    V.assign(s1.begin(), s1.end());
    // Make sure the elements are in the expected order.
    std::sort(V.begin(), V.end());
    for (int i = 0; i < 6; i++)
        EXPECT_EQ(i, V[i]);
}

TEST(TCompactSetTest, IteratorString)
{
    // Test CompactSetIterator for TCompactSet with a type with non-trivial
    // ctors/dtors.
    TCompactSet<std::string, 2> s1;

    s1.insert("str 1");
    s1.insert("str 2");
    s1.insert("str 1");

    std::vector<std::string> V(s1.begin(), s1.end());
    std::sort(V.begin(), V.end());
    EXPECT_EQ(2u, s1.size());
    EXPECT_EQ("str 1", V[0]);
    EXPECT_EQ("str 2", V[1]);

    s1.insert("str 4");
    s1.insert("str 0");
    s1.insert("str 4");

    V.assign(s1.begin(), s1.end());
    // Make sure the elements are in the expected order.
    std::sort(V.begin(), V.end());
    EXPECT_EQ(4u, s1.size());
    EXPECT_EQ("str 0", V[0]);
    EXPECT_EQ("str 1", V[1]);
    EXPECT_EQ("str 2", V[2]);
    EXPECT_EQ("str 4", V[3]);
}

TEST(TCompactSetTest, IteratorIncMoveCopy)
{
    // Test CompactSetIterator for TCompactSet with a type with non-trivial
    // ctors/dtors.
    TCompactSet<std::string, 2> s1;

    s1.insert("str 1");
    s1.insert("str 2");

    auto Iter = s1.begin();
    EXPECT_EQ("str 1", *Iter);
    ++Iter;
    EXPECT_EQ("str 2", *Iter);

    s1.insert("str 4");
    s1.insert("str 0");
    auto Iter2 = s1.begin();
    Iter = std::move(Iter2);
    EXPECT_EQ("str 0", *Iter);
}

// These test weren't taken from llvm.

TEST(TCompactSetTest, Empty)
{
    TCompactSet<int, 10> v;
    EXPECT_TRUE(v.empty());

    auto data = {1, 2, 3, 4, 5};

    v.insert(data.begin(), data.end()); // not crossing size threshold
    v.erase(4);
    v.erase(2);
    v.erase(3);
    v.erase(5);
    v.erase(1);
    EXPECT_TRUE(v.empty());

    auto data2 = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    v.insert(data2.begin(), data2.end()); // crossing size threshold
    v.erase(7);
    v.erase(3);
    v.erase(1);
    v.erase(10);
    v.erase(9);
    v.erase(0);
    v.erase(2);
    v.erase(6);
    v.erase(4);
    v.erase(5);
    v.erase(8);
    EXPECT_TRUE(v.empty());
}

TEST(TCompactSetTest, ForEach)
{
    TCompactSet<int, 10> v;

    auto data = {10, 9, 3, 4, 1, 5, 6, 8};

    v.insert(data.begin(), data.end());

    std::vector<int> buf(v.begin(), v.end());
    std::vector<int> expected{1, 3, 4, 5, 6, 8, 9, 10};
    EXPECT_EQ(expected, buf);

    auto data2 = {7, 1, 2, 0};
    v.insert(data2.begin(), data2.end());
    buf.assign(v.begin(), v.end());
    expected = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    EXPECT_EQ(expected, buf);
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
