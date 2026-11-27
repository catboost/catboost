#include "hash.h"
#include "hash_multi_map.h"
#include "vector.h"
#include "hash_set.h"

#include <library/cpp/case_insensitive_string/case_insensitive_string.h>
#include <library/cpp/testing/common/probe.h>
#include <library/cpp/testing/unittest/registar.h>

#include <utility>
#include <util/str_stl.h>
#include <util/digest/multi.h>
#include <util/string/join.h>

static const char star = 42;

class THashTest: public TTestBase {
    UNIT_TEST_SUITE(THashTest);
    UNIT_TEST(TestHMapConstructorsAndAssignments);
    UNIT_TEST(TestHMap1);
    UNIT_TEST(TestHMapEqualityOperator);
    UNIT_TEST(TestHMapEqualityOperatorWithCustomPred);
    UNIT_TEST(TestHMMapEqualityOperator);
    UNIT_TEST(TestHMMapEqualityOperatorWithCustomPred);
    UNIT_TEST(TestHMMapConstructorsAndAssignments);
    UNIT_TEST(TestHMMap1);
    UNIT_TEST(TestHMMapHas);
    UNIT_TEST(TestHSetConstructorsAndAssignments);
    UNIT_TEST(TestHSetSize);
    UNIT_TEST(TestHSet2);
    UNIT_TEST(TestHSetEqualityOperator);
    UNIT_TEST(TestHSetEqualityOperatorWithCustomPred);
    UNIT_TEST(TestHMSetConstructorsAndAssignments);
    UNIT_TEST(TestHMSetSize);
    UNIT_TEST(TestHMSet1);
    UNIT_TEST(TestHMSetEqualityOperator);
    UNIT_TEST(TestHMSetEqualityOperatorWithCustomPred);
    UNIT_TEST(TestHMSetEmplace);
    UNIT_TEST(TestInsertErase);
    UNIT_TEST(TestResizeOnInsertSmartPtrBug)
    UNIT_TEST(TestEmpty);
    UNIT_TEST(TestDefaultConstructor);
    UNIT_TEST(TestSizeOf);
    UNIT_TEST(TestInvariants);
    UNIT_TEST(TestAllocation);
    UNIT_TEST(TestInsertCopy);
    UNIT_TEST(TestEmplace);
    UNIT_TEST(TestEmplaceNoresize);
    UNIT_TEST(TestEmplaceDirect);
    UNIT_TEST(TestTryEmplace);
    UNIT_TEST(TestTryEmplaceCopyKey);
    UNIT_TEST(TestInsertOrAssign);
    UNIT_TEST(TestHMMapEmplace);
    UNIT_TEST(TestHMMapEmplaceNoresize);
    UNIT_TEST(TestHMMapEmplaceDirect);
    UNIT_TEST(TestHSetEmplace);
    UNIT_TEST(TestHSetEmplaceNoresize);
    UNIT_TEST(TestHSetEmplaceDirect);
    UNIT_TEST(TestNonCopyable);
    UNIT_TEST(TestValueInitialization);
    UNIT_TEST(TestAssignmentClear);
    UNIT_TEST(TestReleaseNodes);
    UNIT_TEST(TestAt);
    UNIT_TEST(TestHMapInitializerList);
    UNIT_TEST(TestHMMapInitializerList);
    UNIT_TEST(TestHMapStringKeyTransparency);
    UNIT_TEST(TestTryEmplaceIsCopyFree);
    UNIT_TEST(TestOperatorSquareIsCopyFree);
    UNIT_TEST(TestMapInsertRvalueIsCopyFree);
    UNIT_TEST(TestSetInsertRvalueIsCopyFree);
    UNIT_TEST(TestMultiMapInsertRvalueIsCopyFree);
    UNIT_TEST(TestMultiSetInsertRvalueIsCopyFree);
    UNIT_TEST(TestHeterogeneousLookup);
    UNIT_TEST(TestBitFieldKey);
    UNIT_TEST(TestConvertibleKeyType);
    UNIT_TEST(TestRangeInsert);
    UNIT_TEST(TestHSetInitializerList);
    UNIT_TEST(TestHMSetInitializerList);
    UNIT_TEST(TestHSetInsertInitializerList);
    UNIT_TEST(TestTupleHash);
    UNIT_TEST(TestStringHash);
    UNIT_TEST(TestFloatingPointHash);
    UNIT_TEST_SUITE_END();

    using hmset = THashMultiSet<char, hash<char>, TEqualTo<char>>;

protected:
    void TestHMapConstructorsAndAssignments();
    void TestHMap1();
    void TestHMapEqualityOperator();
    void TestHMapEqualityOperatorWithCustomPred();
    void TestHMMapEqualityOperator();
    void TestHMMapEqualityOperatorWithCustomPred();
    void TestHMMapConstructorsAndAssignments();
    void TestHMMap1();
    void TestHMMapHas();
    void TestHSetConstructorsAndAssignments();
    void TestHSetSize();
    void TestHSet2();
    void TestHSetEqualityOperator();
    void TestHSetEqualityOperatorWithCustomPred();
    void TestHMSetConstructorsAndAssignments();
    void TestHMSetSize();
    void TestHMSet1();
    void TestHMSetEqualityOperator();
    void TestHMSetEqualityOperatorWithCustomPred();
    void TestHMSetEmplace();
    void TestInsertErase();
    void TestResizeOnInsertSmartPtrBug();
    void TestEmpty();
    void TestDefaultConstructor();
    void TestSizeOf();
    void TestInvariants();
    void TestAllocation();
    void TestInsertCopy();
    void TestEmplace();
    void TestEmplaceNoresize();
    void TestEmplaceDirect();
    void TestTryEmplace();
    void TestTryEmplaceCopyKey();
    void TestInsertOrAssign();
    void TestHSetEmplace();
    void TestHSetEmplaceNoresize();
    void TestHSetEmplaceDirect();
    void TestHMMapEmplace();
    void TestHMMapEmplaceNoresize();
    void TestHMMapEmplaceDirect();
    void TestNonCopyable();
    void TestValueInitialization();
    void TestAssignmentClear();
    void TestReleaseNodes();
    void TestAt();
    void TestHMapInitializerList();
    void TestHMMapInitializerList();
    void TestHMapStringKeyTransparency();
    void TestTryEmplaceIsCopyFree();
    void TestOperatorSquareIsCopyFree();
    void TestMapInsertRvalueIsCopyFree();
    void TestSetInsertRvalueIsCopyFree();
    void TestMultiMapInsertRvalueIsCopyFree();
    void TestMultiSetInsertRvalueIsCopyFree();
    void TestHeterogeneousLookup();
    void TestBitFieldKey();
    void TestConvertibleKeyType();
    void TestRangeInsert();
    void TestHSetInitializerList();
    void TestHMSetInitializerList();
    void TestHSetInsertInitializerList();
    void TestTupleHash();
    void TestStringHash();
    void TestFloatingPointHash();
};

UNIT_TEST_SUITE_REGISTRATION(THashTest);

template <class TContainer, class T, class... TLHSParams, class... TRHSParams>
static void TestStringEquality(
    const TStringBuf testSubCaseName,
    const bool expected,
    const std::initializer_list<T>& lhsList,
    const std::initializer_list<T>& rhsList,
    const std::tuple<TLHSParams...>& lhsParams = std::tuple<>{},
    const std::tuple<TRHSParams...>& rhsParams = std::tuple<>{})
{
    auto make = [](const std::initializer_list<T>& list, const auto& params) {
        TContainer r = std::make_from_tuple<TContainer>(params);
        for (const auto& v : list) {
            r.emplace(v);
        }
        return r;
    };
    auto makeRepr = [](const TContainer& c) -> TString {
        TStringBuilder sb;
        for (const auto& v : c) {
            if (sb) {
                sb << ", "sv;
            }
            if constexpr (requires {v.first; v.second; }) {
                sb << "{" << v.first.Quote() << "," << v.second << "}"sv;
            } else {
                sb << v.Quote();
            }
        }
        return '{' + std::move(sb) + '}';
    };

    const TContainer lhs = make(lhsList, lhsParams);
    const TContainer rhs = make(rhsList, rhsParams);

    const bool eq = (lhs == rhs);
    const TStringBuf expectedStr = expected ? "=="sv : "!="sv;

    UNIT_ASSERT_C(eq == expected, (TStringBuilder() << "case '" << testSubCaseName << "' failed: "
                                                    << makeRepr(lhs) << expectedStr << makeRepr(rhs)));
}

void THashTest::TestHMapConstructorsAndAssignments() {
    using container = THashMap<TString, int>;

    container c1;
    c1["one"] = 1;
    c1["two"] = 2;

    container c2(c1);

    UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
    UNIT_ASSERT_VALUES_EQUAL(2, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(1, c1.at("one")); /* Note: fails under MSVC since it does not support implicit generation of move constructors. */
    UNIT_ASSERT_VALUES_EQUAL(2, c2.at("two"));

    container c3(std::move(c1));

    UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
    UNIT_ASSERT_VALUES_EQUAL(2, c3.size());
    UNIT_ASSERT_VALUES_EQUAL(1, c3.at("one"));

    c2["three"] = 3;
    c3 = c2;

    UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(3, c3.size());
    UNIT_ASSERT_VALUES_EQUAL(3, c3.at("three"));

    c2["four"] = 4;
    c3 = std::move(c2);

    UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
    UNIT_ASSERT_VALUES_EQUAL(4, c3.at("four"));

    const container c4{
        {"one", 1},
        {"two", 2},
        {"three", 3},
        {"four", 4},
    };

    UNIT_ASSERT_VALUES_EQUAL(4, c4.size());
    UNIT_ASSERT_VALUES_EQUAL(1, c4.at("one"));
    UNIT_ASSERT_VALUES_EQUAL(2, c4.at("two"));
    UNIT_ASSERT_VALUES_EQUAL(3, c4.at("three"));
    UNIT_ASSERT_VALUES_EQUAL(4, c4.at("four"));

    // non-existent values must be zero-initialized
    UNIT_ASSERT_VALUES_EQUAL(c1["nonexistent"], 0);
}

void THashTest::TestHMap1() {
    using maptype = THashMap<char, TString, THash<char>, TEqualTo<char>>;
    maptype m;
    // Store mappings between roman numerals and decimals.
    m['l'] = "50";
    m['x'] = "20"; // Deliberate mistake.
    m['v'] = "5";
    m['i'] = "1";
    UNIT_ASSERT(!strcmp(m['x'].c_str(), "20"));
    m['x'] = "10"; // Correct mistake.
    UNIT_ASSERT(!strcmp(m['x'].c_str(), "10"));

    UNIT_ASSERT(!m.contains('z'));
    UNIT_ASSERT(!strcmp(m['z'].c_str(), ""));
    UNIT_ASSERT(m.contains('z'));

    UNIT_ASSERT(m.count('z') == 1);
    auto p = m.insert(std::pair<const char, TString>('c', TString("100")));

    UNIT_ASSERT(p.second);

    p = m.insert(std::pair<const char, TString>('c', TString("100")));
    UNIT_ASSERT(!p.second);

    // Some iterators compare check, really compile time checks
    maptype::iterator ite(m.begin());
    maptype::const_iterator cite(m.begin());
    cite = m.begin();
    maptype const& cm = m;
    cite = cm.begin();

    UNIT_ASSERT((maptype::const_iterator)ite == cite);
    UNIT_ASSERT(!((maptype::const_iterator)ite != cite));
    UNIT_ASSERT(cite == (maptype::const_iterator)ite);
    UNIT_ASSERT(!(cite != (maptype::const_iterator)ite));
}

void THashTest::TestHMapEqualityOperator() {
    using container = THashMap<TString, int>;

    container base;
    base["one"] = 1;
    base["two"] = 2;

    container c1(base);
    UNIT_ASSERT(c1 == base);

    container c2;
    c2["two"] = 2;
    c2["one"] = 1;
    UNIT_ASSERT(c2 == base);

    c2["three"] = 3;
    UNIT_ASSERT(c2 != base);

    container c3(base);
    c3["one"] = 0;
    UNIT_ASSERT(c3 != base);
}

void THashTest::TestHMapEqualityOperatorWithCustomPred() {
    using container = THashMap<TString, int, THash<TCaseInsensitiveStringBuf>, TEqualTo<TCaseInsensitiveStringBuf>>;
    using containerCI = THashMap<TCaseInsensitiveString, int, THash<TCaseInsensitiveStringBuf>, TEqualTo<TCaseInsensitiveStringBuf>>;

    TestStringEquality<container, std::pair<const TStringBuf, int>>("idenitity_1", true, {{"one", 1}, {"two", 2}}, {{"one", 1}, {"two", 2}});
    TestStringEquality<containerCI, std::pair<const TStringBuf, int>>("idenitity_2", true, {{"one", 1}, {"two", 2}}, {{"ONE", 1}, {"TWO", 2}});
    TestStringEquality<container, std::pair<const TStringBuf, int>>("idenitity_1", false, {{"one", 1}, {"two", 2}}, {{"ONE", 1}, {"two", 2}});
    TestStringEquality<container, std::pair<const TStringBuf, int>>("idenitity_1", false, {{"one", 1}, {"two", 2}}, {{"onw", 0}, {"two", 2}});
}

void THashTest::TestHMMapEqualityOperator() {
    using container = THashMultiMap<TString, int>;
    using value = container::value_type;

    container base;
    base.insert(value("one", 1));
    base.insert(value("one", -1));
    base.insert(value("two", 2));

    container c1(base);
    UNIT_ASSERT(c1 == base);

    container c2;
    c2.insert(value("two", 2));
    c2.insert(value("one", -1));
    c2.insert(value("one", 1));
    UNIT_ASSERT(c2 == base);

    c2.insert(value("three", 3));
    UNIT_ASSERT(c2 != base);

    container c3;
    c3.insert(value("one", 0));
    c3.insert(value("one", -1));
    c3.insert(value("two", 2));
    UNIT_ASSERT(c3 != base);

    container c4;
    c4.insert(value("one", 1));
    c4.insert(value("one", -1));
    c4.insert(value("one", 0));
    c4.insert(value("two", 2));
    UNIT_ASSERT(c3 != base);
}

void THashTest::TestHMMapEqualityOperatorWithCustomPred() {
    using container = THashMultiMap<TString, int, THash<TCaseInsensitiveStringBuf>, TEqualTo<TCaseInsensitiveStringBuf>>;

    TestStringEquality<container, std::pair<const TStringBuf, int>>("idenitity", true, {{"string1", 1}, {"string2", -2}, {"string2", 2}}, {{"string1", 1}, {"string2", -2}, {"string2", 2}});
    TestStringEquality<container, std::pair<const TStringBuf, int>>("case2", false, {{"string1", 1}, {"string2", -2}, {"string2", 2}}, {{"string1", 1}, {"string1", -1}, {"string2", 2}});
    TestStringEquality<container, std::pair<const TStringBuf, int>>("case3", false, {{"string1", 1}, {"string2", -2}, {"string2", 2}}, {{"string1", 1}, {"STRING2", -2}, {"STRING2", 2}});
    TestStringEquality<container, std::pair<const TStringBuf, int>>("case4", false, {{"string1", 1}, {"string2", -2}, {"string2", 2}}, {{"string1", 1}, {"STRING2", -2}, {"string2", 2}});
    TestStringEquality<container, std::pair<const TStringBuf, int>>("case5", false, {{"string1", 1}, {"string2", -2}, {"string2", 2}}, {{"string1", 1}, {"string2", 2}, {"STRING2", -2}});
    TestStringEquality<container, std::pair<const TStringBuf, int>>("case6", true, {{"string1", 1}, {"STRING2", -2}, {"string2", 2}}, {{"string1", 1}, {"string2", 2}, {"STRING2", -2}});
}

void THashTest::TestHMMapConstructorsAndAssignments() {
    using container = THashMultiMap<TString, int>;

    container c1;
    c1.insert(container::value_type("one", 1));
    c1.insert(container::value_type("two", 2));

    container c2(c1);

    UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
    UNIT_ASSERT_VALUES_EQUAL(2, c2.size());

    container c3(std::move(c1));

    UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
    UNIT_ASSERT_VALUES_EQUAL(2, c3.size());

    c2.insert(container::value_type("three", 3));
    c3 = c2;

    UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(3, c3.size());

    c2.insert(container::value_type("four", 4));
    c3 = std::move(c2);

    UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
}

void THashTest::TestHMMap1() {
    using mmap = THashMultiMap<char, int, THash<char>, TEqualTo<char>>;
    mmap m;

    UNIT_ASSERT(m.count('X') == 0);
    m.insert(std::pair<const char, int>('X', 10)); // Standard way.
    UNIT_ASSERT(m.count('X') == 1);

    m.insert(std::pair<const char, int>('X', 20)); // jbuck: standard way
    UNIT_ASSERT(m.count('X') == 2);

    m.insert(std::pair<const char, int>('Y', 32)); // jbuck: standard way
    mmap::iterator i = m.find('X');                // Find first match.

    UNIT_ASSERT((*i).first == 'X');
    UNIT_ASSERT((*i).second == 10);
    ++i;
    UNIT_ASSERT((*i).first == 'X');
    UNIT_ASSERT((*i).second == 20);

    i = m.find('Y');
    UNIT_ASSERT((*i).first == 'Y');
    UNIT_ASSERT((*i).second == 32);

    i = m.find('Z');
    UNIT_ASSERT(i == m.end());

    size_t count = m.erase('X');
    UNIT_ASSERT(count == 2);

    // Some iterators compare check, really compile time checks
    mmap::iterator ite(m.begin());
    mmap::const_iterator cite(m.begin());

    UNIT_ASSERT((mmap::const_iterator)ite == cite);
    UNIT_ASSERT(!((mmap::const_iterator)ite != cite));
    UNIT_ASSERT(cite == (mmap::const_iterator)ite);
    UNIT_ASSERT(!(cite != (mmap::const_iterator)ite));

    using HMapType = THashMultiMap<size_t, size_t>;
    HMapType hmap;

    // We fill the map to implicitely start a rehash.
    for (size_t counter = 0; counter < 3077; ++counter) {
        hmap.insert(HMapType::value_type(1, counter));
    }

    hmap.insert(HMapType::value_type(12325, 1));
    hmap.insert(HMapType::value_type(12325, 2));

    UNIT_ASSERT(hmap.count(12325) == 2);

    // At this point 23 goes to the same bucket as 12325, it used to reveal a bug.
    hmap.insert(HMapType::value_type(23, 0));

    UNIT_ASSERT(hmap.count(12325) == 2);

    UNIT_ASSERT(hmap.bucket_count() > 3000);
    for (size_t n = 0; n < 10; n++) {
        hmap.clear();
        hmap.insert(HMapType::value_type(1, 2));
    }
    UNIT_ASSERT(hmap.bucket_count() < 30);
}

void THashTest::TestHMMapHas() {
    using mmap = THashMultiMap<char, int, THash<char>, TEqualTo<char>>;
    mmap m;
    m.insert(std::pair<const char, int>('X', 10));
    m.insert(std::pair<const char, int>('X', 20));
    m.insert(std::pair<const char, int>('Y', 32));
    UNIT_ASSERT(m.contains('X'));
    UNIT_ASSERT(m.contains('Y'));
    UNIT_ASSERT(!m.contains('Z'));
}

void THashTest::TestHSetConstructorsAndAssignments() {
    using container = THashSet<int>;

    container c1;
    c1.insert(100);
    c1.insert(200);

    container c2(c1);

    UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
    UNIT_ASSERT_VALUES_EQUAL(2, c2.size());
    UNIT_ASSERT(c1.contains(100));
    UNIT_ASSERT(c2.contains(200));

    container c3(std::move(c1));

    UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
    UNIT_ASSERT_VALUES_EQUAL(2, c3.size());
    UNIT_ASSERT(c3.contains(100));

    c2.insert(300);
    c3 = c2;

    UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(3, c3.size());
    UNIT_ASSERT(c3.contains(300));

    c2.insert(400);
    c3 = std::move(c2);

    UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
    UNIT_ASSERT(c3.contains(400));

    container c4 = {1, 2, 3};
    UNIT_ASSERT_VALUES_EQUAL(c4.size(), 3);
    UNIT_ASSERT(c4.contains(1));
    UNIT_ASSERT(c4.contains(2));
    UNIT_ASSERT(c4.contains(3));
}

void THashTest::TestHSetSize() {
    using container = THashSet<int>;

    container c;
    c.insert(100);
    c.insert(200);

    UNIT_ASSERT_VALUES_EQUAL(2, c.size());

    c.insert(200);

    UNIT_ASSERT_VALUES_EQUAL(2, c.size());
}

void THashTest::TestHSet2() {
    THashSet<int, THash<int>, TEqualTo<int>> s;
    auto p = s.insert(42);
    UNIT_ASSERT(p.second);
    UNIT_ASSERT(*(p.first) == 42);

    p = s.insert(42);
    UNIT_ASSERT(!p.second);
}

void THashTest::TestHSetEqualityOperator() {
    using container = THashSet<int>;

    container base;
    base.insert(1);
    base.insert(2);

    container c1(base);
    UNIT_ASSERT(c1 == base);

    c1.insert(1);
    UNIT_ASSERT(c1 == base);

    c1.insert(3);
    UNIT_ASSERT(c1 != base);

    container c2;
    c2.insert(2);
    c2.insert(1);
    UNIT_ASSERT(c2 == base);

    container c3;
    c3.insert(1);
    UNIT_ASSERT(c3 != base);
}

void THashTest::TestHSetEqualityOperatorWithCustomPred() {
    using container = THashSet<TString, THash<TCaseInsensitiveStringBuf>, TEqualTo<TCaseInsensitiveStringBuf>>;
    using containerCI = THashSet<TCaseInsensitiveString, THash<TCaseInsensitiveStringBuf>, TEqualTo<TCaseInsensitiveStringBuf>>;

    TestStringEquality<container>("idenitity", true, {"string1", "string2"}, {"string1", "string2"});
    TestStringEquality<container>("not_inserted", true, {"string1", "string2"}, {"string1", "string2", "STRING1"});
    TestStringEquality<container>("different_values", false, {"string1", "string2"}, {"STRING1", "string2"});
    TestStringEquality<containerCI>("same_values", true, {"string1", "string2"}, {"STRING1", "string2"});
}

void THashTest::TestHMSetConstructorsAndAssignments() {
    using container = THashMultiSet<int>;

    container c1;
    c1.insert(100);
    c1.insert(200);

    container c2(c1);

    UNIT_ASSERT_VALUES_EQUAL(2, c1.size());
    UNIT_ASSERT_VALUES_EQUAL(2, c2.size());
    UNIT_ASSERT(c1.find(100) != c1.end());
    UNIT_ASSERT(c2.find(200) != c2.end());

    container c3(std::move(c1));

    UNIT_ASSERT_VALUES_EQUAL(0, c1.size());
    UNIT_ASSERT_VALUES_EQUAL(2, c3.size());
    UNIT_ASSERT(c3.find(100) != c3.end());

    c2.insert(300);
    c3 = c2;

    UNIT_ASSERT_VALUES_EQUAL(3, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(3, c3.size());
    UNIT_ASSERT(c3.find(300) != c3.end());

    c2.insert(400);
    c3 = std::move(c2);

    UNIT_ASSERT_VALUES_EQUAL(0, c2.size());
    UNIT_ASSERT_VALUES_EQUAL(4, c3.size());
    UNIT_ASSERT(c3.find(400) != c3.end());
}

void THashTest::TestHMSetSize() {
    using container = THashMultiSet<int>;

    container c;
    c.insert(100);
    c.insert(200);

    UNIT_ASSERT_VALUES_EQUAL(2, c.size());

    c.insert(200);

    UNIT_ASSERT_VALUES_EQUAL(3, c.size());
}

void THashTest::TestHMSet1() {
    hmset s;
    UNIT_ASSERT(s.count(star) == 0);
    s.insert(star);
    UNIT_ASSERT(s.count(star) == 1);
    s.insert(star);
    UNIT_ASSERT(s.count(star) == 2);
    auto i = s.find(char(40));
    UNIT_ASSERT(i == s.end());

    i = s.find(star);
    UNIT_ASSERT(i != s.end());
    UNIT_ASSERT(*i == '*');
    UNIT_ASSERT(s.erase(star) == 2);
}

void THashTest::TestHMSetEqualityOperator() {
    using container = THashMultiSet<int>;

    container base;
    base.insert(1);
    base.insert(1);
    base.insert(2);

    container c1(base);
    UNIT_ASSERT(c1 == base);

    c1.insert(1);
    UNIT_ASSERT(!(c1 == base));

    container c2;
    c2.insert(2);
    c2.insert(1);
    c2.insert(1);
    UNIT_ASSERT(c2 == base);

    container c3;
    c3.insert(1);
    c3.insert(2);
    UNIT_ASSERT(!(c3 == base));

    c3.insert(1);
    UNIT_ASSERT(c3 == base);

    c3.insert(3);
    UNIT_ASSERT(!(c3 == base));
}

void THashTest::TestHMSetEqualityOperatorWithCustomPred() {
    using container = THashMultiSet<TString, THash<TCaseInsensitiveStringBuf>, TEqualTo<TCaseInsensitiveStringBuf>>;
    using containerCI = THashMultiSet<TCaseInsensitiveString, THash<TCaseInsensitiveStringBuf>, TEqualTo<TCaseInsensitiveStringBuf>>;

    TestStringEquality<container>("idenitity", true, {"string1", "string2", "string2"}, {"string1", "string2", "string2"});
    TestStringEquality<container>("diff_cardinality_1", false, {"string1", "string2", "string2"}, {"string1", "string2", "string2", "string2"});
    TestStringEquality<container>("diff_cardinality_2", false, {"string1", "string2", "string2"}, {"string1", "string2", "string1"});
    TestStringEquality<container>("different_values_1", false, {"string1", "string2", "string2"}, {"STRING1", "string2", "string2"});
    TestStringEquality<container>("different_values_2", false, {"string1", "string2", "string2"}, {"string1", "STRING2", "string2"});
    TestStringEquality<containerCI>("same_values", true, {"string1", "string2", "string2"}, {"string1", "STRING2", "string2"});
    TestStringEquality<container>("permutation", true, {"string1", "string2", "STRING2", "STRING2"}, {"string1", "STRING2", "string2", "STRING2"});
}

void THashTest::TestHMSetEmplace() {
    class TKey: public NTesting::TProbe {
    public:
        TKey(NTesting::TProbeState* state, int key)
            : TProbe(state)
            , Key_(key)
        {
        }

        operator size_t() const {
            return THash<int>()(Key_);
        }

        bool operator==(const TKey& other) const {
            return Key_ == other.Key_;
        }

    private:
        int Key_;
    };

    NTesting::TProbeState state;

    {
        THashMultiSet<TKey> c;
        c.emplace(&state, 1);
        c.emplace(&state, 1);
        c.emplace(&state, 2);

        UNIT_ASSERT_EQUAL(state.CopyAssignments, 0);
        UNIT_ASSERT_EQUAL(state.MoveAssignments, 0);
        UNIT_ASSERT_EQUAL(state.Constructors, 3);
        UNIT_ASSERT_EQUAL(state.MoveConstructors, 0);

        UNIT_ASSERT_EQUAL(c.count(TKey(&state, 1)), 2);
        UNIT_ASSERT_EQUAL(c.count(TKey(&state, 2)), 1);
        UNIT_ASSERT_EQUAL(c.count(TKey(&state, 3)), 0);

        UNIT_ASSERT_EQUAL(state.Constructors, 6);
        UNIT_ASSERT_EQUAL(state.Destructors, 3);
    }

    UNIT_ASSERT_EQUAL(state.CopyAssignments, 0);
    UNIT_ASSERT_EQUAL(state.MoveAssignments, 0);
    UNIT_ASSERT_EQUAL(state.CopyConstructors, 0);
    UNIT_ASSERT_EQUAL(state.MoveConstructors, 0);
    UNIT_ASSERT_EQUAL(state.Constructors, 6);
    UNIT_ASSERT_EQUAL(state.Destructors, 6);
}

void THashTest::TestInsertErase() {
    using hmap = THashMap<TString, size_t, THash<TString>, TEqualTo<TString>>;
    using val_type = hmap::value_type;

    {
        hmap values;

        UNIT_ASSERT(values.insert(val_type("foo", 0)).second);
        UNIT_ASSERT(values.insert(val_type("bar", 0)).second);
        UNIT_ASSERT(values.insert(val_type("abc", 0)).second);

        UNIT_ASSERT(values.erase("foo") == 1);
        UNIT_ASSERT(values.erase("bar") == 1);
        UNIT_ASSERT(values.erase("abc") == 1);
    }

    {
        hmap values;

        UNIT_ASSERT(values.insert(val_type("foo", 0)).second);
        UNIT_ASSERT(values.insert(val_type("bar", 0)).second);
        UNIT_ASSERT(values.insert(val_type("abc", 0)).second);

        UNIT_ASSERT(values.erase("abc") == 1);
        UNIT_ASSERT(values.erase("bar") == 1);
        UNIT_ASSERT(values.erase("foo") == 1);
    }
}

namespace {
    struct TItem: public TSimpleRefCount<TItem> {
        const TString Key;
        const TString Value;

        TItem(const TString& key, const TString& value)
            : Key(key)
            , Value(value)
        {
        }
    };

    using TItemPtr = TIntrusivePtr<TItem>;

    struct TSelectKey {
        const TString& operator()(const TItemPtr& item) const {
            return item->Key;
        }
    };

    using TItemMapBase = THashTable<
        TItemPtr,
        TString,
        THash<TString>,
        TSelectKey,
        TEqualTo<TString>,
        std::allocator<TItemPtr>>;

    struct TItemMap: public TItemMapBase {
        TItemMap()
            : TItemMapBase(1, THash<TString>(), TEqualTo<TString>())
        {
        }

        TItem& Add(const TString& key, const TString& value) {
            insert_ctx ins;
            iterator it = find_i(key, ins);
            if (it == end()) {
                it = insert_direct(new TItem(key, value), ins);
            }
            return **it;
        }
    };
} // namespace

void THashTest::TestResizeOnInsertSmartPtrBug() {
    TItemMap map;
    map.Add("key1", "value1");
    map.Add("key2", "value2");
    map.Add("key3", "value3");
    map.Add("key4", "value4");
    map.Add("key5", "value5");
    map.Add("key6", "value6");
    map.Add("key7", "value7");
    TItem& item = map.Add("key8", "value8");
    UNIT_ASSERT_EQUAL(item.Key, "key8");
    UNIT_ASSERT_EQUAL(item.Value, "value8");
}

template <typename T>
static void EmptyAndInsertTest(typename T::value_type v) {
    T c;
    UNIT_ASSERT(!c);
    c.insert(v);
    UNIT_ASSERT(c);
}

void THashTest::TestEmpty() {
    EmptyAndInsertTest<THashSet<int>>(1);
    EmptyAndInsertTest<THashMap<int, int>>(std::pair<int, int>(1, 2));
    EmptyAndInsertTest<THashMultiMap<int, int>>(std::pair<int, int>(1, 2));
}

void THashTest::TestDefaultConstructor() {
    THashSet<int> set;

    UNIT_ASSERT(set.begin() == set.end());

    UNIT_ASSERT(set.find(0) == set.end());

    auto range = set.equal_range(0);
    UNIT_ASSERT(range.first == range.second);
}

void THashTest::TestSizeOf() {
    /* This test checks that we don't waste memory when all functors passed to
     * THashTable are empty. It does rely on knowledge of THashTable internals,
     * so if those change, the test will have to be adjusted accordingly. */

    size_t expectedSize = sizeof(uintptr_t) + 3 * sizeof(size_t);

    UNIT_ASSERT_VALUES_EQUAL(sizeof(THashMap<int, int>), expectedSize);
    UNIT_ASSERT_VALUES_EQUAL(sizeof(THashMap<std::pair<int, int>, std::pair<int, int>>), expectedSize);
}

void THashTest::TestInvariants() {
    std::set<int> reference_set;
    THashSet<int> set;

    for (int i = 0; i < 1000; i++) {
        set.insert(i);
        reference_set.insert(i);
    }
    UNIT_ASSERT_VALUES_EQUAL(set.size(), 1000);

    int count0 = 0;
    for (int i = 0; i < 1000; i++) {
        count0 += (set.find(i) != set.end()) ? 1 : 0;
    }
    UNIT_ASSERT_VALUES_EQUAL(count0, 1000);

    int count1 = 0;
    for (auto pos = set.begin(); pos != set.end(); pos++) {
        ++count1;
    }
    UNIT_ASSERT_VALUES_EQUAL(count1, 1000);

    int count2 = 0;
    for (const int& value : set) {
        count2 += (reference_set.find(value) != reference_set.end()) ? 1 : 0;
    }
    UNIT_ASSERT_VALUES_EQUAL(count2, 1000);
}

struct TAllocatorCounters {
    TAllocatorCounters()
        : Allocations(0)
        , Deallocations(0)
    {
    }

    ~TAllocatorCounters() {
        std::allocator<char> allocator;

        /* Release whatever was (intentionally) leaked. */
        for (const auto& chunk : Chunks) {
            allocator.deallocate(static_cast<char*>(chunk.first), chunk.second);
        }
    }

    size_t Allocations;
    size_t Deallocations;
    TSet<std::pair<void*, size_t>> Chunks;
};

template <class T>
class TCountingAllocator: public std::allocator<T> {
    using base_type = std::allocator<T>;

public:
    using size_type = typename base_type::size_type;

    template <class Other>
    struct rebind {
        using other = TCountingAllocator<Other>;
    };

    TCountingAllocator()
        : Counters_(nullptr)
    {
    }

    TCountingAllocator(TAllocatorCounters* counters)
        : Counters_(counters)
    {
        Y_ASSERT(counters);
    }

    template <class Other>
    TCountingAllocator(const TCountingAllocator<Other>& other)
        : Counters_(other.Counters)
    {
    }

    T* allocate(size_type n) {
        auto result = base_type::allocate(n);

        if (Counters_) {
            ++Counters_->Allocations;
            Counters_->Chunks.emplace(result, n * sizeof(T));
        }

        return result;
    }

    void deallocate(T* p, size_type n) {
        if (Counters_) {
            ++Counters_->Deallocations;
            Counters_->Chunks.erase(std::make_pair(p, n * sizeof(T)));
        }

        base_type::deallocate(p, n);
    }

private:
    TAllocatorCounters* Counters_;
};

void THashTest::TestAllocation() {
    TAllocatorCounters counters;

    using int_set = THashSet<int, THash<int>, TEqualTo<int>, TCountingAllocator<int>>;

    {
        int_set set0(&counters);
        int_set set1(set0);
        set0.clear();
        int_set set2(&counters);
        set2 = set1;
        UNIT_ASSERT_VALUES_EQUAL(counters.Allocations, 0); /* Copying around null sets should not trigger allocations. */

        set0.insert(0);
        UNIT_ASSERT_VALUES_EQUAL(counters.Allocations, 2); /* One for buckets array, one for a new node. */

        set0.clear();
        set1 = set0;
        int_set set3(set0);
        UNIT_ASSERT_VALUES_EQUAL(counters.Allocations, 2); /* Copying from an empty set with allocated buckets should not trigger allocations. */

        for (int i = 0; i < 1000; i++) {
            set0.insert(i);
        }
        size_t allocations = counters.Allocations;
        set0.clear();
        UNIT_ASSERT_VALUES_EQUAL(counters.Allocations, allocations); /* clear() should not trigger allocations. */
    }

    UNIT_ASSERT_VALUES_EQUAL(counters.Allocations, counters.Deallocations);
}

template <int Value>
class TNonCopyableInt {
public:
    explicit TNonCopyableInt(int) {
    }

    TNonCopyableInt() = delete;
    TNonCopyableInt(const TNonCopyableInt&) = delete;
    TNonCopyableInt(TNonCopyable&&) = delete;
    TNonCopyableInt& operator=(const TNonCopyable&) = delete;
    TNonCopyableInt& operator=(TNonCopyable&&) = delete;

    operator int() const {
        return Value;
    }
};

void THashTest::TestInsertCopy() {
    THashMap<int, int> hash;

    /* Insertion should not make copies of the provided key. */
    hash[TNonCopyableInt<0>(0)] = 0;
}

void THashTest::TestEmplace() {
    using hash_t = THashMap<int, TNonCopyableInt<0>>;
    hash_t hash;
    hash.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(0));
    auto it = hash.find(1);
    UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(it->second), 0);
}

void THashTest::TestEmplaceNoresize() {
    using hash_t = THashMap<int, TNonCopyableInt<0>>;
    hash_t hash;
    hash.reserve(1);
    hash.emplace_noresize(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(0));
    auto it = hash.find(1);
    UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(it->second), 0);
}

void THashTest::TestEmplaceDirect() {
    using hash_t = THashMap<int, TNonCopyableInt<0>>;
    hash_t hash;
    hash_t::insert_ctx ins;
    hash.find(1, ins);
    hash.emplace_direct(ins, std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(0));
    auto it = hash.find(1);
    UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(it->second), 0);
}

void THashTest::TestTryEmplace() {
    static unsigned counter = 0u;

    struct TCountConstruct {
        explicit TCountConstruct(int v)
            : value(v)
        {
            ++counter;
        }
        TCountConstruct(const TCountConstruct&) = delete;
        int value;
    };

    THashMap<int, TCountConstruct> hash;
    {
        // try_emplace does not copy key if key is rvalue
        auto r = hash.try_emplace(TNonCopyableInt<0>(0), 1);
        UNIT_ASSERT(r.second);
        UNIT_ASSERT_VALUES_EQUAL(1, counter);
        UNIT_ASSERT_VALUES_EQUAL(1, r.first->second.value);
    }
    {
        auto r = hash.try_emplace(0, 2);
        UNIT_ASSERT(!r.second);
        UNIT_ASSERT_VALUES_EQUAL(1, counter);
        UNIT_ASSERT_VALUES_EQUAL(1, r.first->second.value);
    }
}

void THashTest::TestTryEmplaceCopyKey() {
    static unsigned counter = 0u;

    struct TCountCopy {
        explicit TCountCopy(int i)
            : Value(i)
        {
        }
        TCountCopy(const TCountCopy& other)
            : Value(other.Value)
        {
            ++counter;
        }

        operator int() const {
            return Value;
        }

        int Value;
    };

    THashMap<TCountCopy, TNonCopyableInt<0>> hash;
    TCountCopy key(1);
    {
        // try_emplace copy key if key is lvalue
        auto r = hash.try_emplace(key, 1);
        UNIT_ASSERT(r.second);
        UNIT_ASSERT_VALUES_EQUAL(1, counter);
    }
    {
        // no insert - no copy
        auto r = hash.try_emplace(key, 2);
        UNIT_ASSERT(!r.second);
        UNIT_ASSERT_VALUES_EQUAL(1, counter);
    }
}

void THashTest::TestInsertOrAssign() {
    static int constructorCounter = 0;
    static int assignmentCounter = 0;

    struct TCountConstruct {
        explicit TCountConstruct(int v)
            : Value(v)
        {
            ++constructorCounter;
        }

        TCountConstruct& operator=(int v) {
            Value = v;
            ++assignmentCounter;
            return *this;
        }

        TCountConstruct(const TCountConstruct&) = delete;
        int Value;
    };

    THashMap<int, TCountConstruct> hash;
    {
        auto r = hash.insert_or_assign(TNonCopyableInt<4>(4), 1);
        UNIT_ASSERT(r.second);
        UNIT_ASSERT_VALUES_EQUAL(1, hash.size());
        UNIT_ASSERT_VALUES_EQUAL(1, constructorCounter);
        UNIT_ASSERT_VALUES_EQUAL(0, assignmentCounter);
        UNIT_ASSERT_VALUES_EQUAL(1, r.first->second.Value);
    }
    {
        auto r = hash.insert_or_assign(TNonCopyableInt<4>(4), 5);
        UNIT_ASSERT(!r.second);
        UNIT_ASSERT_VALUES_EQUAL(1, hash.size());
        UNIT_ASSERT_VALUES_EQUAL(1, constructorCounter);
        UNIT_ASSERT_VALUES_EQUAL(1, assignmentCounter);
        UNIT_ASSERT_VALUES_EQUAL(5, r.first->second.Value);
    }
    {
        constexpr int iterations = 200;
        for (int iteration = 0; iteration < iterations; ++iteration) {
            hash.insert_or_assign(iteration, iteration);
        }
        UNIT_ASSERT_VALUES_EQUAL(iterations, hash.size());
        UNIT_ASSERT_VALUES_EQUAL(iterations, constructorCounter);
        UNIT_ASSERT_VALUES_EQUAL(2, assignmentCounter);
        UNIT_ASSERT_VALUES_EQUAL(4, hash.at(4).Value);
        UNIT_ASSERT_VALUES_EQUAL(44, hash.at(44).Value);
    }
}

void THashTest::TestHMMapEmplace() {
    using hash_t = THashMultiMap<int, TNonCopyableInt<0>>;
    hash_t hash;
    hash.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(0));
    auto it = hash.find(1);
    UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(it->second), 0);
}

void THashTest::TestHMMapEmplaceNoresize() {
    using hash_t = THashMultiMap<int, TNonCopyableInt<0>>;
    hash_t hash;
    hash.reserve(1);
    hash.emplace_noresize(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(0));
    auto it = hash.find(1);
    UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(it->second), 0);
}

void THashTest::TestHMMapEmplaceDirect() {
    using hash_t = THashMultiMap<int, TNonCopyableInt<0>>;
    hash_t hash;
    hash_t::insert_ctx ins;
    hash.find(1, ins);
    hash.emplace_direct(ins, std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(0));
    auto it = hash.find(1);
    UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(it->second), 0);
}

void THashTest::TestHSetEmplace() {
    using hash_t = THashSet<TNonCopyableInt<0>, THash<int>, TEqualTo<int>>;
    hash_t hash;
    UNIT_ASSERT(!hash.contains(0));
    hash.emplace(0);
    UNIT_ASSERT(hash.contains(0));
    UNIT_ASSERT(!hash.contains(1));
}

void THashTest::TestHSetEmplaceNoresize() {
    using hash_t = THashSet<TNonCopyableInt<0>, THash<int>, TEqualTo<int>>;
    hash_t hash;
    hash.reserve(1);
    UNIT_ASSERT(!hash.contains(0));
    hash.emplace_noresize(0);
    UNIT_ASSERT(hash.contains(0));
    UNIT_ASSERT(!hash.contains(1));
}

void THashTest::TestHSetEmplaceDirect() {
    using hash_t = THashSet<TNonCopyableInt<0>, THash<int>, TEqualTo<int>>;
    hash_t hash;
    UNIT_ASSERT(!hash.contains(0));
    hash_t::insert_ctx ins;
    hash.find(0, ins);
    hash.emplace_direct(ins, 1);
    UNIT_ASSERT(hash.contains(0));
    UNIT_ASSERT(!hash.contains(1));
}

void THashTest::TestNonCopyable() {
    struct TValue: public TNonCopyable {
        int value;
        TValue(int _value = 0)
            : value(_value)
        {
        }
        operator int() {
            return value;
        }
    };

    THashMap<int, TValue> hash;
    hash.emplace(std::piecewise_construct, std::forward_as_tuple(1), std::forward_as_tuple(5));
    auto&& value = hash[1];
    UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(value), 5);
    auto&& not_inserted = hash[2];
    UNIT_ASSERT_VALUES_EQUAL(static_cast<int>(not_inserted), 0);
}

void THashTest::TestValueInitialization() {
    THashMap<int, int> hash;

    int& value = hash[0];

    /* Implicitly inserted values should be value-initialized. */
    UNIT_ASSERT_VALUES_EQUAL(value, 0);
}

void THashTest::TestAssignmentClear() {
    /* This one tests that assigning an empty hash resets the buckets array.
     * See operator= for details. */

    THashMap<int, int> hash;
    size_t emptyBucketCount = hash.bucket_count();

    for (int i = 0; i < 100; i++) {
        hash[i] = i;
    }

    hash = THashMap<int, int>();

    UNIT_ASSERT_VALUES_EQUAL(hash.bucket_count(), emptyBucketCount);
}

void THashTest::TestReleaseNodes() {
    TAllocatorCounters counters;
    using TIntSet = THashSet<int, THash<int>, TEqualTo<int>, TCountingAllocator<int>>;

    TIntSet set(&counters);
    for (int i = 0; i < 3; i++) {
        set.insert(i);
    }
    UNIT_ASSERT_VALUES_EQUAL(counters.Allocations, 4);

    set.release_nodes();
    UNIT_ASSERT_VALUES_EQUAL(counters.Allocations, 4);
    UNIT_ASSERT_VALUES_EQUAL(set.size(), 0);

    for (int i = 10; i < 13; i++) {
        set.insert(i);
    }
    UNIT_ASSERT_VALUES_EQUAL(counters.Allocations, 7);
    UNIT_ASSERT(set.contains(10));
    UNIT_ASSERT(!set.contains(0));

    set.basic_clear();
    UNIT_ASSERT_VALUES_EQUAL(counters.Deallocations, 3);

    TIntSet set2;
    set2.release_nodes();
    set2.insert(1);
    UNIT_ASSERT_VALUES_EQUAL(set2.size(), 1);
}

void THashTest::TestAt() {
#define TEST_AT_THROWN_EXCEPTION(SRC_TYPE, DST_TYPE, KEY_TYPE, KEY, MESSAGE)                                                                               \
    {                                                                                                                                                      \
        THashMap<SRC_TYPE, DST_TYPE> testMap;                                                                                                              \
        try {                                                                                                                                              \
            KEY_TYPE testKey = KEY;                                                                                                                        \
            testMap.at(testKey);                                                                                                                           \
            UNIT_ASSERT_C(false, "THashMap::at(\"" << KEY << "\") should throw");                                                                          \
        } catch (const yexception& e) {                                                                                                                    \
            UNIT_ASSERT_C(e.AsStrBuf().Contains(MESSAGE), "Incorrect exception description: got \"" << e.what() << "\", expected: \"" << MESSAGE << "\""); \
        } catch (...) {                                                                                                                                    \
            UNIT_ASSERT_C(false, "THashMap::at(\"" << KEY << "\") should throw yexception");                                                               \
        }                                                                                                                                                  \
    }

    TEST_AT_THROWN_EXCEPTION(TString, TString, TString, "111", "111");
    TEST_AT_THROWN_EXCEPTION(TString, TString, const TString, "111", "111");
    TEST_AT_THROWN_EXCEPTION(TString, TString, TStringBuf, "111", "111");
    TEST_AT_THROWN_EXCEPTION(TString, TString, const TStringBuf, "111", "111");
    TEST_AT_THROWN_EXCEPTION(TStringBuf, TStringBuf, const char*, "111", "111");
    TEST_AT_THROWN_EXCEPTION(int, int, short, 11, "11");
    TEST_AT_THROWN_EXCEPTION(int, int, int, -1, "-1");
    TEST_AT_THROWN_EXCEPTION(int, int, long, 111, "111");
    TEST_AT_THROWN_EXCEPTION(int, int, long long, -1000000000000ll, "-1000000000000");
    TEST_AT_THROWN_EXCEPTION(int, int, unsigned short, 11, "11");
    TEST_AT_THROWN_EXCEPTION(int, int, unsigned int, 2, "2");
    TEST_AT_THROWN_EXCEPTION(int, int, unsigned long, 131, "131");
    TEST_AT_THROWN_EXCEPTION(int, int, unsigned long long, 1000000000000ll, "1000000000000");

    char key[] = {11, 12, 0, 1, 2, 11, 0};
    TEST_AT_THROWN_EXCEPTION(TString, TString, char*, key, "\\x0B\\x0C");
    TEST_AT_THROWN_EXCEPTION(TString, TString, TStringBuf, TStringBuf(key, sizeof(key) - 1), "\\x0B\\x0C\\0\\1\\2\\x0B");

#undef TEST_AT_THROWN_EXCEPTION
}

void THashTest::TestHMapInitializerList() {
    THashMap<TString, TString> h1 = {{"foo", "bar"}, {"bar", "baz"}, {"baz", "qux"}};
    THashMap<TString, TString> h2;
    h2.insert(std::pair<TString, TString>("foo", "bar"));
    h2.insert(std::pair<TString, TString>("bar", "baz"));
    h2.insert(std::pair<TString, TString>("baz", "qux"));
    UNIT_ASSERT_EQUAL(h1, h2);
}

void THashTest::TestHMapStringKeyTransparency() {
    auto test = [](auto m) {
        // Check transparency (is_transparent).
        m["foo"] = "bar";
        m[TStringBuf{"foo"}] = "bar";
        m[std::string_view{"foo"}] = "bar";
        m[std::string{"foo"}] = "bar";

        UNIT_ASSERT_EQUAL(m["foo"], "bar");
    };

    test(THashMap<TString, TString>());
    test(THashMap<std::string, std::string>());
}

void THashTest::TestHMMapInitializerList() {
    THashMultiMap<TString, TString> h1 = {
        {"foo", "bar"},
        {"foo", "baz"},
        {"baz", "qux"}};
    THashMultiMap<TString, TString> h2;
    h2.insert(std::pair<TString, TString>("foo", "bar"));
    h2.insert(std::pair<TString, TString>("foo", "baz"));
    h2.insert(std::pair<TString, TString>("baz", "qux"));
    UNIT_ASSERT_EQUAL(h1, h2);
}

void THashTest::TestHSetInitializerList() {
    THashSet<TString> h1 = {"foo", "bar", "baz"};
    THashSet<TString> h2;
    h2.insert("foo");
    h2.insert("bar");
    h2.insert("baz");
    UNIT_ASSERT_EQUAL(h1, h2);
}

void THashTest::TestHMSetInitializerList() {
    THashMultiSet<TString> h1 = {"foo", "foo", "bar", "baz"};
    THashMultiSet<TString> h2;
    h2.insert("foo");
    h2.insert("foo");
    h2.insert("bar");
    h2.insert("baz");
    UNIT_ASSERT_EQUAL(h1, h2);
}

namespace {
    struct TFoo {
        int A;
        int B;

        bool operator==(const TFoo& o) const {
            return A == o.A && B == o.B;
        }
    };
} // namespace

template <>
struct THash<TFoo> {
    size_t operator()(const TFoo& v) const {
        return v.A ^ v.B;
    }
};

template <>
void Out<TFoo>(IOutputStream& o, const TFoo& v) {
    o << '{' << v.A << ';' << v.B << '}';
}

void THashTest::TestHSetInsertInitializerList() {
    {
        const THashSet<int> x = {1};
        THashSet<int> y;
        y.insert({1});
        UNIT_ASSERT_VALUES_EQUAL(x, y);
    }
    {
        const THashSet<int> x = {1, 2};
        THashSet<int> y;
        y.insert({1, 2});
        UNIT_ASSERT_VALUES_EQUAL(x, y);
    }
    {
        const THashSet<int> x = {1, 2, 3, 4, 5};
        THashSet<int> y;
        y.insert({
            1,
            2,
            3,
            4,
            5,
        });
        UNIT_ASSERT_VALUES_EQUAL(x, y);
    }
    {
        const THashSet<TFoo> x = {{1, 2}};
        THashSet<TFoo> y;
        y.insert({{1, 2}});
        UNIT_ASSERT_VALUES_EQUAL(x, y);
    }
    {
        const THashSet<TFoo> x = {{1, 2}, {3, 4}};
        THashSet<TFoo> y;
        y.insert({{1, 2}, {3, 4}});
        UNIT_ASSERT_VALUES_EQUAL(x, y);
    }
}

/*
 * Sequence for MultiHash is reversed as it calculates hash as
 * f(head:tail) = f(tail)xHash(head)
 */
void THashTest::TestTupleHash() {
    std::tuple<int, int> tuple{1, 3};
    UNIT_ASSERT_VALUES_EQUAL(THash<decltype(tuple)>()(tuple), MultiHash(3, 1));

    /*
     * This thing checks that we didn't break STL code
     * See https://a.yandex-team.ru/arc/commit/2864838#comment-401
     * for example
     */
    struct A {
        A Foo(const std::tuple<A, float>& v) {
            return std::get<A>(v);
        }
    };
}

void THashTest::TestStringHash() {
    // Make sure that different THash<> variants behave in the same way
    const size_t expected = ComputeHash(TString("hehe"));
    UNIT_ASSERT_VALUES_EQUAL(ComputeHash("hehe"), expected);              // char[5]
    UNIT_ASSERT_VALUES_EQUAL(ComputeHash("hehe"sv), expected);            // std::string_view
    UNIT_ASSERT_VALUES_EQUAL(ComputeHash(TStringBuf("hehe")), expected);  // TStringBuf
    UNIT_ASSERT_VALUES_EQUAL(ComputeHash<const char*>("hehe"), expected); // const char*
}

template <class TFloat>
static void TestFloatingPointHashImpl() {
    const TFloat f = 0;
    Y_ASSERT(f == -f);
    THashSet<TFloat> set;
    set.insert(f);
    UNIT_ASSERT_C(set.contains(-f), TypeName<TFloat>());
    UNIT_ASSERT_VALUES_EQUAL_C(ComputeHash(f), ComputeHash(-f), TypeName<TFloat>());
    for (int i = 0; i < 5; ++i) {
        set.insert(-TFloat(i));
        set.insert(+TFloat(i));
    }
    UNIT_ASSERT_VALUES_EQUAL_C(set.size(), 9, TypeName<TFloat>());
}

void THashTest::TestFloatingPointHash() {
    TestFloatingPointHashImpl<float>();
    TestFloatingPointHashImpl<double>();
    // TestFloatingPointHashImpl<long double>();
}

// Copy-accounting value type: every special member of key/value objects is
// counted globally, so a test can assert exactly how many copies, moves and
// constructions a given container operation performs.
namespace NHmOpCounts {
    struct TOpStats {
        size_t DefaultCtors = 0;
        size_t CopyCtors = 0;
        size_t MoveCtors = 0;
        size_t CopyAssigns = 0;
        size_t MoveAssigns = 0;
        size_t Dtors = 0;

        void Reset() {
            *this = TOpStats{};
        }

        size_t AllCtors() const {
            return DefaultCtors + CopyCtors + MoveCtors;
        }

        size_t AllCopies() const {
            return CopyCtors + CopyAssigns;
        }
    };

    TOpStats& Stats() {
        static TOpStats stats;
        return stats;
    }

    struct TCounted {
        int Value = 0;

        TCounted() {
            ++Stats().DefaultCtors;
        }

        explicit TCounted(int value)
            : Value(value)
        {
            ++Stats().DefaultCtors;
        }

        TCounted(const TCounted& other)
            : Value(other.Value)
        {
            ++Stats().CopyCtors;
        }

        TCounted(TCounted&& other) noexcept
            : Value(other.Value)
        {
            ++Stats().MoveCtors;
        }

        TCounted& operator=(const TCounted& other) {
            Value = other.Value;
            ++Stats().CopyAssigns;
            return *this;
        }

        TCounted& operator=(TCounted&& other) noexcept {
            Value = other.Value;
            ++Stats().MoveAssigns;
            return *this;
        }

        ~TCounted() {
            ++Stats().Dtors;
        }

        bool operator==(const TCounted& other) const {
            return Value == other.Value;
        }
    };

    struct TCountedHash {
        size_t operator()(const TCounted& value) const {
            return THash<int>{}(value.Value);
        }
    };

    using TCountedMap = THashMap<TCounted, TCounted, TCountedHash, TEqualTo<TCounted>>;
    using TCountedSet = THashSet<TCounted, TCountedHash, TEqualTo<TCounted>>;
    using TCountedMultiMap = THashMultiMap<TCounted, TCounted, TCountedHash, TEqualTo<TCounted>>;
    using TCountedMultiSet = THashMultiSet<TCounted, TCountedHash, TEqualTo<TCounted>>;
} // namespace NHmOpCounts

void THashTest::TestTryEmplaceIsCopyFree() {
    using namespace NHmOpCounts;
    TCountedMap hash;

    // Fresh insertion of an rvalue key and value: both are moved, nothing is copied.
    {
        TCounted key(1);
        TCounted value(2);
        Stats().Reset();
        auto res = hash.try_emplace(std::move(key), std::move(value));
        UNIT_ASSERT(res.second);
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 2);
        UNIT_ASSERT_VALUES_EQUAL(Stats().Dtors, 0);
    }

    // Fresh insertion of an lvalue key and value: exactly one copy each, no extra.
    {
        TCounted key(3);
        TCounted value(4);
        Stats().Reset();
        auto res = hash.try_emplace(key, value);
        UNIT_ASSERT(res.second);
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 2);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().Dtors, 0);
    }

    // A duplicate constructs nothing, touches no arguments and keeps the stored value.
    {
        TCounted dupKey(1);
        TCounted dupValue(5);
        Stats().Reset();
        auto res = hash.try_emplace(std::move(dupKey), std::move(dupValue));
        UNIT_ASSERT(!res.second);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCtors(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().Dtors, 0);
        // try_emplace must leave the arguments intact when the key already exists
        UNIT_ASSERT_VALUES_EQUAL(dupKey.Value, 1);
        UNIT_ASSERT_VALUES_EQUAL(dupValue.Value, 5);
        // and the stored pair is not overwritten
        UNIT_ASSERT_VALUES_EQUAL(res.first->second.Value, 2);
    }
}

void THashTest::TestOperatorSquareIsCopyFree() {
    using namespace NHmOpCounts;
    TCountedMap hash;

    TCounted key(1);
    TCounted value(2);
    UNIT_ASSERT(hash.try_emplace(std::move(key), std::move(value)).second);

    // A duplicate lookup constructs nothing.
    {
        TCounted probe(1);
        Stats().Reset();
        TCounted& mapped = hash[probe];
        UNIT_ASSERT_VALUES_EQUAL(mapped.Value, 2);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCtors(), 0);
    }

    // Fresh insertion of an rvalue key: the key must be moved, not copied.
    {
        TCounted rvalueKey(3);
        Stats().Reset();
        TCounted& mapped = hash[std::move(rvalueKey)];
        UNIT_ASSERT_VALUES_EQUAL(mapped.Value, 0);
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
    }

    // Fresh insertion of an lvalue key: exactly one key copy, which is unavoidable.
    {
        TCounted lvalueKey(4);
        Stats().Reset();
        TCounted& mapped = hash[lvalueKey];
        UNIT_ASSERT_VALUES_EQUAL(mapped.Value, 0);
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 1);
        UNIT_ASSERT_VALUES_EQUAL(lvalueKey.Value, 4);
    }
}

void THashTest::TestHeterogeneousLookup() {
    THashMap<TString, ui64> m;

    // operator[] with a heterogeneous key creates...
    UNIT_ASSERT_VALUES_EQUAL(m["aaa"], 0);
    // ...and then every lookup flavor finds the element by a literal or a TStringBuf
    UNIT_ASSERT(m.find("aaa") != m.end());
    UNIT_ASSERT(m.find(TStringBuf("aaa")) != m.end());
    UNIT_ASSERT(m.contains(TStringBuf("aaa")));
    UNIT_ASSERT(m.contains("aaa"));
    UNIT_ASSERT_VALUES_EQUAL(m.count(TStringBuf("aaa")), 1);
    UNIT_ASSERT_VALUES_EQUAL(m.at("aaa"), 0);
    UNIT_ASSERT_VALUES_EQUAL(m.at(TStringBuf("aaa")), 0);

    // try_emplace: a fresh heterogeneous key is inserted, a duplicate is not,
    // and the stored value survives the duplicate attempt
    auto r1 = m.try_emplace("bbb", 2);
    UNIT_ASSERT(r1.second);
    auto r2 = m.try_emplace(TStringBuf("bbb"), 3);
    UNIT_ASSERT(!r2.second);
    UNIT_ASSERT_VALUES_EQUAL(m["bbb"], 2);
    UNIT_ASSERT_VALUES_EQUAL(m.size(), 2);

    // operator[] with a heterogeneous key also finds an existing element
    m[TStringBuf("aaa")] = 9;
    UNIT_ASSERT_VALUES_EQUAL(m["aaa"], 9);

    // erase by a heterogeneous key
    UNIT_ASSERT_VALUES_EQUAL(m.erase(TStringBuf("bbb")), 1);
    UNIT_ASSERT_VALUES_EQUAL(m.size(), 1);

    // heterogeneous lookup in sets
    THashSet<TString> s;
    s.insert(TString("x"));
    UNIT_ASSERT(s.find(TStringBuf("x")) != s.end());
    UNIT_ASSERT(s.contains("x"));
    UNIT_ASSERT_VALUES_EQUAL(s.count(TStringBuf("x")), 1);
}

void THashTest::TestMapInsertRvalueIsCopyFree() {
    using namespace NHmOpCounts;
    TCountedMap hash;

    // Inserting an rvalue pair: the const key cannot be moved and is copied once,
    // the mapped value must be moved, not copied.
    {
        TCountedMap::value_type elem(1, 2);
        Stats().Reset();
        auto res = hash.insert(std::move(elem));
        UNIT_ASSERT(res.second);
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().Dtors, 0);
    }

    // A duplicate is detected by probing first: nothing is constructed or copied.
    {
        TCountedMap::value_type elem(1, 2);
        Stats().Reset();
        auto res = hash.insert(std::move(elem));
        UNIT_ASSERT(!res.second);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCtors(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().Dtors, 0);
    }

    // Inserting an lvalue pair: exactly one copy of each member, and the
    // source pair is untouched.
    {
        TCountedMap::value_type elem(3, 4);
        Stats().Reset();
        auto res = hash.insert(elem);
        UNIT_ASSERT(res.second);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 2);
        UNIT_ASSERT_VALUES_EQUAL(elem.first.Value, 3);
        UNIT_ASSERT_VALUES_EQUAL(elem.second.Value, 4);
    }

    // insert_noresize keeps the same guarantees for rvalues.
    {
        TCountedMap::value_type elem(5, 6);
        Stats().Reset();
        auto res = hash.insert_noresize(std::move(elem));
        UNIT_ASSERT(res.second);
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
    }
}

void THashTest::TestSetInsertRvalueIsCopyFree() {
    using namespace NHmOpCounts;
    TCountedSet set;

    // Inserting an rvalue element: it must be moved, not copied.
    {
        TCounted elem(1);
        Stats().Reset();
        auto res = set.insert(std::move(elem));
        UNIT_ASSERT(res.second);
        UNIT_ASSERT_VALUES_EQUAL(set.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
    }

    // A duplicate is detected by probing first: nothing is constructed or copied.
    {
        TCounted elem(1);
        Stats().Reset();
        auto res = set.insert(std::move(elem));
        UNIT_ASSERT(!res.second);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCtors(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
    }

    // Inserting an lvalue element: exactly one copy, source untouched.
    {
        TCounted elem(2);
        Stats().Reset();
        auto res = set.insert(elem);
        UNIT_ASSERT(res.second);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 1);
        UNIT_ASSERT_VALUES_EQUAL(elem.Value, 2);
    }

    // insert_noresize and the hint overload keep the same guarantees for rvalues.
    {
        TCounted elem(3);
        Stats().Reset();
        auto res = set.insert_noresize(std::move(elem));
        UNIT_ASSERT(res.second);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
    }
    {
        TCounted elem(4);
        Stats().Reset();
        set.insert(set.cend(), std::move(elem));
        UNIT_ASSERT_VALUES_EQUAL(set.size(), 4);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
    }
}

void THashTest::TestMultiMapInsertRvalueIsCopyFree() {
    using namespace NHmOpCounts;
    TCountedMultiMap hash;

    // Inserting an rvalue pair: the const key is copied once, the value is moved.
    {
        TCountedMultiMap::value_type elem(1, 2);
        Stats().Reset();
        auto it = hash.insert(std::move(elem));
        UNIT_ASSERT(it != hash.end());
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
    }

    // Inserting an lvalue pair: exactly one copy of each member, source untouched.
    {
        TCountedMultiMap::value_type elem(1, 2);
        Stats().Reset();
        auto it = hash.insert(elem);
        UNIT_ASSERT(it != hash.end());
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 2);
        UNIT_ASSERT_VALUES_EQUAL(elem.first.Value, 1);
        UNIT_ASSERT_VALUES_EQUAL(elem.second.Value, 2);
    }

    // insert_noresize keeps the same guarantees for rvalues.
    {
        TCountedMultiMap::value_type elem(2, 3);
        Stats().Reset();
        auto it = hash.insert_noresize(std::move(elem));
        UNIT_ASSERT(it != hash.end());
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
    }
}

void THashTest::TestMultiSetInsertRvalueIsCopyFree() {
    using namespace NHmOpCounts;
    TCountedMultiSet hash;

    // Inserting an rvalue element: it must be moved, not copied.
    {
        TCounted elem(1);
        Stats().Reset();
        auto it = hash.insert(std::move(elem));
        UNIT_ASSERT(it != hash.end());
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 1);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
    }

    // Inserting an lvalue element: exactly one copy, source untouched.
    {
        TCounted elem(1);
        Stats().Reset();
        auto it = hash.insert(elem);
        UNIT_ASSERT(it != hash.end());
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 2);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 1);
        UNIT_ASSERT_VALUES_EQUAL(elem.Value, 1);
    }

    // insert_noresize keeps the same guarantees for rvalues.
    {
        TCounted elem(2);
        Stats().Reset();
        auto it = hash.insert_noresize(std::move(elem));
        UNIT_ASSERT(it != hash.end());
        UNIT_ASSERT_VALUES_EQUAL(hash.size(), 3);
        UNIT_ASSERT_VALUES_EQUAL(Stats().AllCopies(), 0);
        UNIT_ASSERT_VALUES_EQUAL(Stats().MoveCtors, 1);
    }
}

void THashTest::TestBitFieldKey() {
    struct TBitFieldCarrier {
        ui32 GlobalId : 32;
        ui32 SmallId : 3;
    };

    const TBitFieldCarrier carrier{15, 5};

    // A bit-field cannot bind to a reference parameter, so operator[] must
    // accept it through the const key_type& overload (a temporary key is
    // materialized from the bit-field).
    THashMap<ui32, ui32> hashMap;
    hashMap[carrier.GlobalId] = 1;
    UNIT_ASSERT_VALUES_EQUAL(hashMap.size(), 1);
    UNIT_ASSERT_VALUES_EQUAL(hashMap[carrier.GlobalId], 1);
    hashMap[carrier.GlobalId] += 1;
    UNIT_ASSERT_VALUES_EQUAL(hashMap[carrier.GlobalId], 2);
    UNIT_ASSERT_VALUES_EQUAL(hashMap[carrier.SmallId], 0);
    UNIT_ASSERT_VALUES_EQUAL(hashMap.size(), 2);

    // insert() must accept bit-fields as well.
    THashSet<ui32> hashSet;
    UNIT_ASSERT(hashSet.insert(carrier.GlobalId).second);
    UNIT_ASSERT(!hashSet.insert(carrier.GlobalId).second);
    UNIT_ASSERT(hashSet.insert(carrier.SmallId).second);
    UNIT_ASSERT_VALUES_EQUAL(hashSet.size(), 2);

    THashMultiSet<ui32> hashMultiSet;
    hashMultiSet.insert(carrier.GlobalId);
    hashMultiSet.insert(carrier.GlobalId);
    UNIT_ASSERT_VALUES_EQUAL(hashMultiSet.count(15u), 2);
}

void THashTest::TestConvertibleKeyType() {
    // A key type may expose separate conversion operators for lvalue and rvalue
    // contexts (like TStrongKeyType with its operator const T&() const& and
    // operator T&&() &&). With only insert(const value_type&) and
    // insert(value_type&&) the call would be ambiguous: each overload is
    // reachable only through a different user-defined conversion. The
    // constrained insert(TValue&&) overload takes over such arguments.
    struct TAdapter {
        ui64 Value = 0;

        operator const ui64&() const& {
            return Value;
        }

        operator ui64&&() && {
            return static_cast<ui64&&>(Value);
        }
    };

    THashSet<ui64> hashSet;
    UNIT_ASSERT(hashSet.insert(TAdapter{1}).second);
    UNIT_ASSERT(!hashSet.insert(TAdapter{1}).second);
    UNIT_ASSERT_VALUES_EQUAL(hashSet.size(), 1);

    THashMultiSet<ui64> hashMultiSet;
    hashMultiSet.insert(TAdapter{2});
    hashMultiSet.insert(TAdapter{2});
    UNIT_ASSERT_VALUES_EQUAL(hashMultiSet.count(2u), 2);

    THashMap<ui64, ui64> hashMap;
    hashMap[TAdapter{3}] = 4;
    UNIT_ASSERT_VALUES_EQUAL(hashMap.size(), 1);
    UNIT_ASSERT_VALUES_EQUAL(hashMap[TAdapter{3}], 4);
    TAdapter lvalue{5};
    hashMap[lvalue] = 6;
    UNIT_ASSERT_VALUES_EQUAL(hashMap[5u], 6);

    // A pair convertible into value_type is another user-defined conversion
    // case; it goes through the same overload (std's insert(P&&) equivalent).
    THashMap<ui64, ui64> pairMap;
    UNIT_ASSERT(pairMap.insert(std::make_pair(ui64(7), ui64(8))).second);
    UNIT_ASSERT_VALUES_EQUAL(pairMap[7u], 8);

    THashSet<TString> stringSet;
    stringSet.insert("literal");
    UNIT_ASSERT(stringSet.contains(TStringBuf("literal")));
}

namespace {
    // A value type constructible from anything (e.g. via a variadic constructor):
    // it must not make range insert ambiguous with the hint overload.
    struct TGreedy {
        int Value = 0;

        TGreedy() = default;

        template <typename T, typename... TArgs>
        TGreedy(const T&, const TArgs&...) {
        }

        operator size_t() const {
            return Value;
        }

        bool operator==(const TGreedy& other) const {
            return Value == other.Value;
        }
    };
} // namespace

void THashTest::TestRangeInsert() {
    TGreedy first;
    first.Value = 1;
    TGreedy second;
    second.Value = 2;

    THashSet<TGreedy> source;
    source.insert(first);
    source.insert(second);

    THashSet<TGreedy> target;
    target.insert(source.begin(), source.end());
    UNIT_ASSERT_VALUES_EQUAL(target.size(), 2);

    target.insert(target.cend(), TGreedy{});
    UNIT_ASSERT_VALUES_EQUAL(target.size(), 3);
}
