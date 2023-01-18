#include <library/cpp/yt/small_containers/compact_flat_map.h>

#include <library/cpp/testing/gtest/gtest.h>

#include <string>
#include <vector>

namespace NYT {
namespace {

////////////////////////////////////////////////////////////////////////////////

using TMap = TCompactFlatMap<std::string, std::string, 2>;

TMap CreateMap()
{
    std::vector<std::pair<std::string, std::string>> data = {{"I", "met"}, {"a", "traveller"}, {"from", "an"}, {"antique", "land"}};
    return {data.begin(), data.end()};
}

TEST(TCompactFlatMapTest, DefaultEmpty)
{
    TMap m;
    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.begin(), m.end());
}

TEST(TCompactFlatMapTest, Reserve)
{
    // No real way to test reserve - just use it and wiggle about.
    auto m1 = CreateMap();
    TMap m2;
    m2.reserve(m1.size());
    m2.insert(m1.begin(), m1.end());
    EXPECT_EQ(m1.size(), m2.size());
}

TEST(TCompactFlatMapTest, Size)
{
    auto m = CreateMap();

    EXPECT_EQ(m.size(), 4u);
    EXPECT_EQ(m.ssize(), 4);

    m.insert({"Who", "said"});

    EXPECT_EQ(m.size(), 5u);
    EXPECT_EQ(m.ssize(), 5);

    m.erase("antique");

    EXPECT_EQ(m.size(), 4u);
    EXPECT_EQ(m.ssize(), 4);
}

TEST(TCompactFlatMapTest, ClearAndEmpty)
{
    auto m = CreateMap();

    EXPECT_FALSE(m.empty());
    EXPECT_NE(m.begin(), m.end());

    m.clear();

    EXPECT_TRUE(m.empty());
    EXPECT_EQ(m.begin(), m.end());

    m.insert({"Who", "said"});

    EXPECT_FALSE(m.empty());
    EXPECT_NE(m.begin(), m.end());
}

TEST(TCompactFlatMapTest, FindMutable)
{
    auto m = CreateMap();
    {
        auto it = m.find("from");
        EXPECT_NE(it, m.end());
        EXPECT_EQ(it->second, "an");
        it->second = "the";
    }
    {
        auto it = m.find("from");
        EXPECT_NE(it, m.end());
        EXPECT_EQ(it->second, "the");
    }
    {
        auto it = m.find("Who");
        EXPECT_EQ(it, m.end());
    }
}

TEST(TCompactFlatMapTest, FindConst)
{
    const auto& m = CreateMap();
    {
        auto it = m.find("from");
        EXPECT_NE(it, m.end());
        EXPECT_EQ(it->second, "an");
    }
    {
        auto it = m.find("Who");
        EXPECT_EQ(it, m.end());
    }
}

TEST(TCompactFlatMapTest, Insert)
{
    auto m = CreateMap();

    auto [it, inserted] = m.insert({"Who", "said"});
    EXPECT_TRUE(inserted);
    EXPECT_EQ(m.ssize(), 5);
    EXPECT_NE(it, m.end());
    EXPECT_EQ(it, m.find("Who"));
    EXPECT_EQ(it->second, "said");

    auto [it2, inserted2] = m.insert({"Who", "told"});
    EXPECT_FALSE(inserted2);
    EXPECT_EQ(m.ssize(), 5);
    EXPECT_EQ(it2, it);
    EXPECT_EQ(it->second, "said");

    std::vector<std::pair<std::string, std::string>> data = {{"Two", "vast"}, {"and", "trunkless"}, {"legs", "of"}, {"stone", "Stand"}, {"in", "the"}, {"desert", "..."}};
    m.insert(data.begin(), data.end());
    EXPECT_EQ(m.ssize(), 11);
    EXPECT_NE(m.find("and"), m.end());
    EXPECT_EQ(m.find("and")->second, "trunkless");
}

TEST(TCompactFlatMapTest, Emplace)
{
    auto m = CreateMap();

    auto [it, inserted] = m.emplace("Far", "place");
    EXPECT_TRUE(inserted);
    EXPECT_EQ(m.ssize(), 5);
    EXPECT_NE(it, m.end());
    EXPECT_EQ(it, m.find("Far"));
    EXPECT_EQ(it->second, "place");

    auto [it2, inserted2] = m.emplace("Far", "behind");
    EXPECT_FALSE(inserted2);
    EXPECT_EQ(m.ssize(), 5);
    EXPECT_EQ(it2, it);
    EXPECT_EQ(it->second, "place");
}

TEST(TCompactFlatMapTest, Subscript)
{
    auto m = CreateMap();

    EXPECT_EQ(m["antique"], "land");
    EXPECT_EQ(m.ssize(), 4);

    EXPECT_EQ(m["Who"], "");
    EXPECT_EQ(m.ssize(), 5);
}

TEST(TCompactFlatMapTest, Erase)
{
    auto m = CreateMap();

    m.erase("antique");
    EXPECT_EQ(m.ssize(), 3);

    m.erase("Who");
    EXPECT_EQ(m.ssize(), 3);

    m.erase(m.begin(), m.end());
    EXPECT_TRUE(m.empty());
}

TEST(TCompactFlatMapTest, GrowShrink)
{
    TMap m;
    m.insert({"Two", "vast"});
    m.insert({"and", "trunkless"});
    m.insert({"legs", "of"});
    m.insert({"stone", "Stand"});
    m.insert({"in", "the"});
    m.insert({"desert", "..."});

    m.erase("legs");
    m.erase("stone");
    m.erase("in");
    m.erase("desert");

    EXPECT_EQ(m.ssize(), 2);

    // Must not crash or trigger asan.
}

TEST(TCompactFlatMapTest, GrowShrinkGrow)
{
    TMap m;
    m.insert({"Two", "vast"});
    m.insert({"and", "trunkless"});
    m.insert({"legs", "of"});
    m.insert({"stone", "Stand"});
    m.insert({"in", "the"});
    m.insert({"desert", "..."});

    m.erase("legs");
    m.erase("stone");
    m.erase("in");
    m.erase("desert");

    EXPECT_EQ(m.ssize(), 2);

    m.insert({"I", "met"});
    m.insert({"a", "traveller"});
    m.insert({"from", "an"});
    m.insert({"antique", "land"});

    EXPECT_EQ(m.ssize(), 6);

    // Must not crash or trigger asan.
}

////////////////////////////////////////////////////////////////////////////////

} // namespace
} // namespace NYT
