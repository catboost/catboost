#include <util/random/shuffle.h>
#include <library/cpp/testing/unittest/registar.h>

#include <util/stream/output.h>
#include <utility>

#include <util/charset/wide.h>
#include <util/generic/algorithm.h>
#include <util/generic/buffer.h>
#include <util/generic/map.h>
#include <util/generic/vector.h>
#include <util/generic/ptr.h>
#include <util/generic/ylimits.h>

#include <util/folder/dirut.h>

#include <util/random/random.h>
#include <util/random/fast.h>

#include <util/string/hex.h>
#include <util/string/cast.h>

#include "comptrie.h"
#include "set.h"
#include "first_symbol_iterator.h"
#include "search_iterator.h"
#include "pattern_searcher.h"

#include <array>
#include <iterator>


class TCompactTrieTest: public TTestBase {
private:
    UNIT_TEST_SUITE(TCompactTrieTest);
    UNIT_TEST(TestTrie8);
    UNIT_TEST(TestTrie16);
    UNIT_TEST(TestTrie32);

    UNIT_TEST(TestFastTrie8);
    UNIT_TEST(TestFastTrie16);
    UNIT_TEST(TestFastTrie32);

    UNIT_TEST(TestMinimizedTrie8);
    UNIT_TEST(TestMinimizedTrie16);
    UNIT_TEST(TestMinimizedTrie32);

    UNIT_TEST(TestFastMinimizedTrie8);
    UNIT_TEST(TestFastMinimizedTrie16);
    UNIT_TEST(TestFastMinimizedTrie32);

    UNIT_TEST(TestTrieIterator8);
    UNIT_TEST(TestTrieIterator16);
    UNIT_TEST(TestTrieIterator32);

    UNIT_TEST(TestMinimizedTrieIterator8);
    UNIT_TEST(TestMinimizedTrieIterator16);
    UNIT_TEST(TestMinimizedTrieIterator32);

    UNIT_TEST(TestPhraseSearch);
    UNIT_TEST(TestAddGet);
    UNIT_TEST(TestEmpty);
    UNIT_TEST(TestUninitializedNonEmpty);
    UNIT_TEST(TestRandom);
    UNIT_TEST(TestFindTails);
    UNIT_TEST(TestPrefixGrouped);
    UNIT_TEST(CrashTestPrefixGrouped);
    UNIT_TEST(TestMergeFromFile);
    UNIT_TEST(TestMergeFromBuffer);
    UNIT_TEST(TestUnique);
    UNIT_TEST(TestAddRetValue);
    UNIT_TEST(TestClear);

    UNIT_TEST(TestIterateEmptyKey);

    UNIT_TEST(TestTrieSet);

    UNIT_TEST(TestTrieForVectorInt64);
    UNIT_TEST(TestTrieForListInt64);
    UNIT_TEST(TestTrieForSetInt64);

    UNIT_TEST(TestTrieForVectorStroka);
    UNIT_TEST(TestTrieForListStroka);
    UNIT_TEST(TestTrieForSetStroka);

    UNIT_TEST(TestTrieForVectorWtroka);
    UNIT_TEST(TestTrieForVectorFloat);
    UNIT_TEST(TestTrieForVectorDouble);

    UNIT_TEST(TestTrieForListVectorInt64);
    UNIT_TEST(TestTrieForPairWtrokaVectorInt64);

    UNIT_TEST(TestEmptyValueOutOfOrder);
    UNIT_TEST(TestFindLongestPrefixWithEmptyValue);

    UNIT_TEST(TestSearchIterChar);
    UNIT_TEST(TestSearchIterWchar);
    UNIT_TEST(TestSearchIterWchar32)

    UNIT_TEST(TestCopyAndAssignment);

    UNIT_TEST(TestFirstSymbolIterator8);
    UNIT_TEST(TestFirstSymbolIterator16);
    UNIT_TEST(TestFirstSymbolIterator32);
    UNIT_TEST(TestFirstSymbolIteratorChar32);

    UNIT_TEST(TestArrayPacker);

    UNIT_TEST(TestBuilderFindLongestPrefix);
    UNIT_TEST(TestBuilderFindLongestPrefixWithEmptyValue);

    UNIT_TEST(TestPatternSearcherEmpty);
    UNIT_TEST(TestPatternSearcherSimple);
    UNIT_TEST(TestPatternSearcherRandom);

    UNIT_TEST_SUITE_END();

    static const char* SampleData[];

    template <class T>
    void CreateTrie(IOutputStream& out, bool minimize, bool useFastLayout);

    template <class T>
    void CheckData(const char* src, size_t len);

    template <class T>
    void CheckUpperBound(const char* src, size_t len);

    template <class T>
    void CheckIterator(const char* src, size_t len);

    template <class T>
    void TestTrie(bool minimize, bool useFastLayout);

    template <class T>
    void TestTrieIterator(bool minimize);

    template <class T, bool minimize>
    void TestRandom(const size_t n, const size_t maxKeySize);

    void TestFindTailsImpl(const TString& prefix);

    void TestUniqueImpl(bool isPrefixGrouped);

    TVector<TUtf16String> GetSampleKeys(size_t nKeys) const;
    template <class TContainer>
    TVector<TContainer> GetSampleVectorData(size_t nValues);
    template <class TContainer>
    TVector<TContainer> GetSampleTextVectorData(size_t nValues);
    template <class T>
    void CheckEquality(const T& value1, const T& value2) const;
    template <class TContainer>
    void TestTrieWithContainers(const TVector<TUtf16String>& keys, const TVector<TContainer>& sampleData, TString methodName);

    template <typename TChar>
    void TestSearchIterImpl();

    template <class TTrie>
    void TestFirstSymbolIteratorForTrie(const TTrie& trie, const TStringBuf& narrowAnswers);

    template <typename TSymbol>
    void TestFirstSymbolIterator();

    template <class T>
    class TIntPacker;
    template <class T>
    class TDummyPacker;
    class TStrokaPacker;

public:
    void TestPackers();

    void TestTrie8();
    void TestTrie16();
    void TestTrie32();

    void TestFastTrie8();
    void TestFastTrie16();
    void TestFastTrie32();

    void TestMinimizedTrie8();
    void TestMinimizedTrie16();
    void TestMinimizedTrie32();

    void TestFastMinimizedTrie8();
    void TestFastMinimizedTrie16();
    void TestFastMinimizedTrie32();

    void TestTrieIterator8();
    void TestTrieIterator16();
    void TestTrieIterator32();

    void TestMinimizedTrieIterator8();
    void TestMinimizedTrieIterator16();
    void TestMinimizedTrieIterator32();

    void TestPhraseSearch();
    void TestAddGet();
    void TestEmpty();
    void TestUninitializedNonEmpty();
    void TestRandom();
    void TestFindTails();
    void TestPrefixGrouped();
    void CrashTestPrefixGrouped();
    void TestMergeFromFile();
    void TestMergeFromBuffer();
    void TestUnique();
    void TestAddRetValue();
    void TestClear();

    void TestIterateEmptyKey();

    void TestTrieSet();

    void TestTrieForVectorInt64();
    void TestTrieForListInt64();
    void TestTrieForSetInt64();

    void TestTrieForVectorStroka();
    void TestTrieForListStroka();
    void TestTrieForSetStroka();

    void TestTrieForVectorWtroka();
    void TestTrieForVectorFloat();
    void TestTrieForVectorDouble();

    void TestTrieForListVectorInt64();
    void TestTrieForPairWtrokaVectorInt64();

    void TestEmptyValueOutOfOrder();
    void TestFindLongestPrefixWithEmptyValue();

    void TestSearchIterChar();
    void TestSearchIterWchar();
    void TestSearchIterWchar32();

    void TestCopyAndAssignment();

    void TestFirstSymbolIterator8();
    void TestFirstSymbolIterator16();
    void TestFirstSymbolIterator32();
    void TestFirstSymbolIteratorChar32();

    void TestArrayPacker();

    void TestBuilderFindLongestPrefix();
    void TestBuilderFindLongestPrefix(size_t keysCount, double branchProbability, bool isPrefixGrouped, bool hasEmptyKey);
    void TestBuilderFindLongestPrefixWithEmptyValue();

    void TestPatternSearcherOnDataset(
        const TVector<TString>& patterns,
        const TVector<TString>& samples
    );
    void TestPatternSearcherEmpty();
    void TestPatternSearcherSimple();
    void TestPatternSearcherRandom(
        size_t patternsNum,
        size_t patternMaxLength,
        size_t strMaxLength,
        int maxChar,
        TFastRng<ui64>& rng
    );
    void TestPatternSearcherRandom();
};

UNIT_TEST_SUITE_REGISTRATION(TCompactTrieTest);

const char* TCompactTrieTest::SampleData[] = {
    "",
    "a", "b", "c", "d",
    "aa", "ab", "ac", "ad",
    "aaa", "aab", "aac", "aad",
    "aba", "abb", "abc", "abd",
    "fba", "fbb", "fbc", "fbd",
    "fbbaa",
    "c\x85\xA4\xBF" // Just something outside ASCII.
};

template <class T>
typename TCompactTrie<T>::TKey MakeWideKey(const char* str, size_t len) {
    typename TCompactTrie<T>::TKey buffer;
    for (size_t i = 0; i < len; i++) {
        unsigned int ch = (str[i] & 0xFF);
        buffer.push_back((T)(ch | ch << 8 | ch << 16 | ch << 24));
    }
    return buffer;
}

template <class T>
typename TCompactTrie<T>::TKey MakeWideKey(const TString& str) {
    return MakeWideKey<T>(str.c_str(), str.length());
}

template <class T>
typename TCompactTrie<T>::TKey MakeWideKey(const TStringBuf& buf) {
    return MakeWideKey<T>(buf.data(), buf.length());
}

template <class T>
void TCompactTrieTest::CreateTrie(IOutputStream& out, bool minimize, bool useFastLayout) {
    TCompactTrieBuilder<T> builder;

    for (auto& i : SampleData) {
        size_t len = strlen(i);

        builder.Add(MakeWideKey<T>(i, len), len * 2);
    }

    TBufferOutput tmp2;
    IOutputStream& currentOutput = useFastLayout ? tmp2 : out;
    if (minimize) {
        TBufferOutput buftmp;
        builder.Save(buftmp);
        CompactTrieMinimize<TCompactTriePacker<ui64>>(currentOutput, buftmp.Buffer().Data(), buftmp.Buffer().Size(), false);
    } else {
        builder.Save(currentOutput);
    }
    if (useFastLayout) {
        CompactTrieMakeFastLayout<TCompactTriePacker<T>>(out, tmp2.Buffer().Data(), tmp2.Buffer().Size(), false);
    }
}

// Iterates over all strings of length <= 4 made of letters a-g.
static bool LexicographicStep(TString& s) {
    if (s.length() < 4) {
        s += "a";
        return true;
    }
    while (!s.empty() && s.back() == 'g')
        s.pop_back();
    if (s.empty())
        return false;
    char last = s.back();
    last++;
    s.pop_back();
    s.push_back(last);
    return true;
}

template <class T>
void TCompactTrieTest::CheckUpperBound(const char* data, size_t datalen) {
    TCompactTrie<T> trie(data, datalen);
    typedef typename TCompactTrie<T>::TKey TKey;
    typedef typename TCompactTrie<T>::TData TData;

    TString key;
    do {
        const TKey wideKey = MakeWideKey<T>(key);
        typename TCompactTrie<T>::TConstIterator it = trie.UpperBound(wideKey);
        UNIT_ASSERT_C(it == trie.End() || it.GetKey() >= wideKey, "key=" + key);
        TData data;
        const bool found = trie.Find(wideKey, &data);
        if (found)
            UNIT_ASSERT_C(it.GetKey() == wideKey && it.GetValue() == data, "key=" + key);
        if (it != trie.Begin())
            UNIT_ASSERT_C((--it).GetKey() < wideKey, "key=" + key);
    } while (LexicographicStep(key));
}

template <class T>
void TCompactTrieTest::CheckData(const char* data, size_t datalen) {
    TCompactTrie<T> trie(data, datalen);

    UNIT_ASSERT_VALUES_EQUAL(Y_ARRAY_SIZE(SampleData), trie.Size());

    for (auto& i : SampleData) {
        size_t len = strlen(i);
        ui64 value = 0;
        size_t prefixLen = 0;

        typename TCompactTrie<T>::TKey key = MakeWideKey<T>(i, len);
        UNIT_ASSERT(trie.Find(key, &value));
        UNIT_ASSERT_EQUAL(len * 2, value);
        UNIT_ASSERT(trie.FindLongestPrefix(key, &prefixLen, &value));
        UNIT_ASSERT_EQUAL(len, prefixLen);
        UNIT_ASSERT_EQUAL(len * 2, value);

        TString badkey("bb");
        badkey += i;
        key = MakeWideKey<T>(badkey);
        UNIT_ASSERT(!trie.Find(key));
        value = 123;
        UNIT_ASSERT(!trie.Find(key, &value));
        UNIT_ASSERT_EQUAL(123, value);
        UNIT_ASSERT(trie.FindLongestPrefix(key, &prefixLen, &value));
        UNIT_ASSERT_EQUAL(1, prefixLen);
        UNIT_ASSERT_EQUAL(2, value);

        badkey = i;
        badkey += "x";
        key = MakeWideKey<T>(badkey);
        UNIT_ASSERT(!trie.Find(key));
        value = 1234;
        UNIT_ASSERT(!trie.Find(key, &value));
        UNIT_ASSERT_EQUAL(1234, value);
        UNIT_ASSERT(trie.FindLongestPrefix(key, &prefixLen, &value));
        UNIT_ASSERT_EQUAL(len, prefixLen);
        UNIT_ASSERT_EQUAL(len * 2, value);
        UNIT_ASSERT(trie.FindLongestPrefix(key, &prefixLen, nullptr));
        UNIT_ASSERT_EQUAL(len, prefixLen);
    }

    TString testkey("fbbaa");
    typename TCompactTrie<T>::TKey key = MakeWideKey<T>(testkey);
    ui64 value = 0;
    size_t prefixLen = 0;
    UNIT_ASSERT(trie.FindLongestPrefix(key.data(), testkey.length() - 1, &prefixLen, &value));
    UNIT_ASSERT_EQUAL(prefixLen, 3);
    UNIT_ASSERT_EQUAL(6, value);

    testkey = "fbbax";
    key = MakeWideKey<T>(testkey);
    UNIT_ASSERT(trie.FindLongestPrefix(key, &prefixLen, &value));
    UNIT_ASSERT_EQUAL(prefixLen, 3);
    UNIT_ASSERT_EQUAL(6, value);

    value = 12345678;
    UNIT_ASSERT(!trie.Find(key, &value));
    UNIT_ASSERT_EQUAL(12345678, value); //Failed Find() should not change value
}

template <class T>
void TCompactTrieTest::CheckIterator(const char* data, size_t datalen) {
    typedef typename TCompactTrie<T>::TKey TKey;
    typedef typename TCompactTrie<T>::TValueType TValue;
    TMap<TKey, ui64> stored;

    for (auto& i : SampleData) {
        size_t len = strlen(i);

        stored[MakeWideKey<T>(i, len)] = len * 2;
    }

    TCompactTrie<T> trie(data, datalen);
    TVector<TValue> items;
    typename TCompactTrie<T>::TConstIterator it = trie.Begin();
    size_t entry_count = 0;
    TMap<TKey, ui64> received;
    while (it != trie.End()) {
        UNIT_ASSERT_VALUES_EQUAL(it.GetKeySize(), it.GetKey().size());
        received.insert(*it);
        items.push_back(*it);
        entry_count++;
        it++;
    }
    TMap<TKey, ui64> received2;
    for (std::pair<TKey, ui64> x : trie) {
        received2.insert(x);
    }
    UNIT_ASSERT(entry_count == stored.size());
    UNIT_ASSERT(received == stored);
    UNIT_ASSERT(received2 == stored);

    std::reverse(items.begin(), items.end());
    typename TCompactTrie<T>::TConstIterator revIt = trie.End();
    typename TCompactTrie<T>::TConstIterator const begin = trie.Begin();
    typename TCompactTrie<T>::TConstIterator emptyIt;
    size_t pos = 0;
    while (revIt != begin) {
        revIt--;
        UNIT_ASSERT(*revIt == items[pos]);
        pos++;
    }
    // Checking the assignment operator.
    revIt = begin;
    UNIT_ASSERT(revIt == trie.Begin());
    UNIT_ASSERT(!revIt.IsEmpty());
    UNIT_ASSERT(revIt != emptyIt);
    UNIT_ASSERT(revIt != trie.End());
    ++revIt; // Call a method that uses Skipper.
    revIt = emptyIt;
    UNIT_ASSERT(revIt == emptyIt);
    UNIT_ASSERT(revIt.IsEmpty());
    UNIT_ASSERT(revIt != trie.End());
    // Checking the move assignment operator.
    revIt = trie.Begin();
    UNIT_ASSERT(revIt == trie.Begin());
    UNIT_ASSERT(!revIt.IsEmpty());
    UNIT_ASSERT(revIt != emptyIt);
    UNIT_ASSERT(revIt != trie.End());
    ++revIt; // Call a method that uses Skipper.
    revIt = typename TCompactTrie<T>::TConstIterator();
    UNIT_ASSERT(revIt == emptyIt);
    UNIT_ASSERT(revIt.IsEmpty());
    UNIT_ASSERT(revIt != trie.End());
}

template <class T>
void TCompactTrieTest::TestTrie(bool minimize, bool useFastLayout) {
    TBufferOutput bufout;
    CreateTrie<T>(bufout, minimize, useFastLayout);
    CheckData<T>(bufout.Buffer().Data(), bufout.Buffer().Size());
    CheckUpperBound<T>(bufout.Buffer().Data(), bufout.Buffer().Size());
}

template <class T>
void TCompactTrieTest::TestTrieIterator(bool minimize) {
    TBufferOutput bufout;
    CreateTrie<T>(bufout, minimize, false);
    CheckIterator<T>(bufout.Buffer().Data(), bufout.Buffer().Size());
}

void TCompactTrieTest::TestTrie8() {
    TestTrie<char>(false, false);
}
void TCompactTrieTest::TestTrie16() {
    TestTrie<wchar16>(false, false);
}
void TCompactTrieTest::TestTrie32() {
    TestTrie<wchar32>(false, false);
}

void TCompactTrieTest::TestFastTrie8() {
    TestTrie<char>(false, true);
}
void TCompactTrieTest::TestFastTrie16() {
    TestTrie<wchar16>(false, true);
}
void TCompactTrieTest::TestFastTrie32() {
    TestTrie<wchar32>(false, true);
}

void TCompactTrieTest::TestMinimizedTrie8() {
    TestTrie<char>(true, false);
}
void TCompactTrieTest::TestMinimizedTrie16() {
    TestTrie<wchar16>(true, false);
}
void TCompactTrieTest::TestMinimizedTrie32() {
    TestTrie<wchar32>(true, false);
}

void TCompactTrieTest::TestFastMinimizedTrie8() {
    TestTrie<char>(true, true);
}
void TCompactTrieTest::TestFastMinimizedTrie16() {
    TestTrie<wchar16>(true, true);
}
void TCompactTrieTest::TestFastMinimizedTrie32() {
    TestTrie<wchar32>(true, true);
}

void TCompactTrieTest::TestTrieIterator8() {
    TestTrieIterator<char>(false);
}
void TCompactTrieTest::TestTrieIterator16() {
    TestTrieIterator<wchar16>(false);
}
void TCompactTrieTest::TestTrieIterator32() {
    TestTrieIterator<wchar32>(false);
}

void TCompactTrieTest::TestMinimizedTrieIterator8() {
    TestTrieIterator<char>(true);
}
void TCompactTrieTest::TestMinimizedTrieIterator16() {
    TestTrieIterator<wchar16>(true);
}
void TCompactTrieTest::TestMinimizedTrieIterator32() {
    TestTrieIterator<wchar32>(true);
}

void TCompactTrieTest::TestPhraseSearch() {
    static const char* phrases[] = {"ab", "ab cd", "ab cd ef"};
    static const char* const goodphrase = "ab cd ef gh";
    static const char* const badphrase = "cd ef gh ab";
    TBufferOutput bufout;

    TCompactTrieBuilder<char> builder;
    for (size_t i = 0; i < Y_ARRAY_SIZE(phrases); i++) {
        builder.Add(phrases[i], strlen(phrases[i]), i);
    }
    builder.Save(bufout);

    TCompactTrie<char> trie(bufout.Buffer().Data(), bufout.Buffer().Size());
    TVector<TCompactTrie<char>::TPhraseMatch> matches;
    trie.FindPhrases(goodphrase, strlen(goodphrase), matches);

    UNIT_ASSERT(matches.size() == Y_ARRAY_SIZE(phrases));
    for (size_t i = 0; i < Y_ARRAY_SIZE(phrases); i++) {
        UNIT_ASSERT(matches[i].first == strlen(phrases[i]));
        UNIT_ASSERT(matches[i].second == i);
    }

    trie.FindPhrases(badphrase, strlen(badphrase), matches);
    UNIT_ASSERT(matches.size() == 0);
}

void TCompactTrieTest::TestAddGet() {
    TCompactTrieBuilder<char> builder;
    builder.Add("abcd", 4, 1);
    builder.Add("acde", 4, 2);
    ui64 dummy;
    UNIT_ASSERT(builder.Find("abcd", 4, &dummy));
    UNIT_ASSERT(1 == dummy);
    UNIT_ASSERT(builder.Find("acde", 4, &dummy));
    UNIT_ASSERT(2 == dummy);
    UNIT_ASSERT(!builder.Find("fgdgfacde", 9, &dummy));
    UNIT_ASSERT(!builder.Find("ab", 2, &dummy));
}

void TCompactTrieTest::TestEmpty() {
    TCompactTrieBuilder<char> builder;
    ui64 dummy = 12345;
    size_t prefixLen;
    UNIT_ASSERT(!builder.Find("abc", 3, &dummy));
    TBufferOutput bufout;
    builder.Save(bufout);

    TCompactTrie<char> trie(bufout.Buffer().Data(), bufout.Buffer().Size());
    UNIT_ASSERT(!trie.Find("abc", 3, &dummy));
    UNIT_ASSERT(!trie.Find("", 0, &dummy));
    UNIT_ASSERT(!trie.FindLongestPrefix("abc", 3, &prefixLen, &dummy));
    UNIT_ASSERT(!trie.FindLongestPrefix("", 0, &prefixLen, &dummy));
    UNIT_ASSERT_EQUAL(12345, dummy);

    UNIT_ASSERT(trie.Begin() == trie.End());

    TCompactTrie<> trieNull;

    UNIT_ASSERT(!trieNull.Find(" ", 1));

    TCompactTrie<>::TPhraseMatchVector matches;
    trieNull.FindPhrases(" ", 1, matches); // just to be sure it doesn't crash

    UNIT_ASSERT(trieNull.Begin() == trieNull.End());
}

void TCompactTrieTest::TestUninitializedNonEmpty() {
    TBufferOutput bufout;
    CreateTrie<char>(bufout, false, false);
    TCompactTrie<char> trie(bufout.Buffer().Data(), bufout.Buffer().Size());
    typedef TCompactTrie<char>::TKey TKey;
    typedef TCompactTrie<char>::TConstIterator TIter;

    TCompactTrie<char> tails = trie.FindTails("abd", 3); // A trie that has empty value and no data.
    UNIT_ASSERT(!tails.IsEmpty());
    UNIT_ASSERT(!tails.IsInitialized());
    const TKey wideKey = MakeWideKey<char>("c", 1);
    TIter it = tails.UpperBound(wideKey);
    UNIT_ASSERT(it == tails.End());
    UNIT_ASSERT(it != tails.Begin());
    --it;
    UNIT_ASSERT(it == tails.Begin());
    ++it;
    UNIT_ASSERT(it == tails.End());
}

static char RandChar() {
    return char(RandomNumber<size_t>() % 256);
}

static TString RandStr(const size_t max) {
    size_t len = RandomNumber<size_t>() % max;
    TString key;
    for (size_t j = 0; j < len; ++j)
        key += RandChar();
    return key;
}

template <class T, bool minimize>
void TCompactTrieTest::TestRandom(const size_t n, const size_t maxKeySize) {
    const TStringBuf EMPTY_KEY = TStringBuf("", 1);
    TCompactTrieBuilder<char, typename T::TData, T> builder;
    typedef TMap<TString, typename T::TData> TKeys;
    TKeys keys;

    typename T::TData dummy;
    for (size_t i = 0; i < n; ++i) {
        const TString key = RandStr(maxKeySize);
        if (key != EMPTY_KEY && keys.find(key) == keys.end()) {
            const typename T::TData val = T::Data(key);
            keys[key] = val;
            UNIT_ASSERT_C(!builder.Find(key.data(), key.size(), &dummy), "key = " << HexEncode(TString(key)));
            builder.Add(key.data(), key.size(), val);
            UNIT_ASSERT_C(builder.Find(key.data(), key.size(), &dummy), "key = " << HexEncode(TString(key)));
            UNIT_ASSERT(dummy == val);
        }
    }

    TBufferStream stream;
    size_t len = builder.Save(stream);
    TCompactTrie<char, typename T::TData, T> trie(stream.Buffer().Data(), len);

    TBufferStream buftmp;
    if (minimize) {
        CompactTrieMinimize<T>(buftmp, stream.Buffer().Data(), len, false);
    }
    TCompactTrie<char, typename T::TData, T> trieMin(buftmp.Buffer().Data(), buftmp.Buffer().Size());

    TCompactTrieBuilder<char, typename T::TData, T> prefixGroupedBuilder(CTBF_PREFIX_GROUPED);

    for (typename TKeys::const_iterator i = keys.begin(), mi = keys.end(); i != mi; ++i) {
        UNIT_ASSERT(!prefixGroupedBuilder.Find(i->first.c_str(), i->first.size(), &dummy));
        UNIT_ASSERT(trie.Find(i->first.c_str(), i->first.size(), &dummy));
        UNIT_ASSERT(dummy == i->second);
        if (minimize) {
            UNIT_ASSERT(trieMin.Find(i->first.c_str(), i->first.size(), &dummy));
            UNIT_ASSERT(dummy == i->second);
        }

        prefixGroupedBuilder.Add(i->first.c_str(), i->first.size(), dummy);
        UNIT_ASSERT(prefixGroupedBuilder.Find(i->first.c_str(), i->first.size(), &dummy));

        for (typename TKeys::const_iterator j = keys.begin(), end = keys.end(); j != end; ++j) {
            typename T::TData valFound;
            if (j->first <= i->first) {
                UNIT_ASSERT(prefixGroupedBuilder.Find(j->first.c_str(), j->first.size(), &valFound));
                UNIT_ASSERT_VALUES_EQUAL(j->second, valFound);
            } else {
                UNIT_ASSERT(!prefixGroupedBuilder.Find(j->first.c_str(), j->first.size(), &valFound));
            }
        }
    }

    TBufferStream prefixGroupedBuffer;
    prefixGroupedBuilder.Save(prefixGroupedBuffer);

    UNIT_ASSERT_VALUES_EQUAL(stream.Buffer().Size(), prefixGroupedBuffer.Buffer().Size());
    UNIT_ASSERT(0 == memcmp(stream.Buffer().Data(), prefixGroupedBuffer.Buffer().Data(), stream.Buffer().Size()));
}

void TCompactTrieTest::TestRandom() {
    TestRandom<TIntPacker<ui64>, true>(1000, 1000);
    TestRandom<TIntPacker<int>, true>(100, 100);
    TestRandom<TDummyPacker<ui64>, true>(0, 0);
    TestRandom<TDummyPacker<ui64>, true>(100, 3);
    TestRandom<TDummyPacker<ui64>, true>(100, 100);
    TestRandom<TStrokaPacker, true>(100, 100);
}

void TCompactTrieTest::TestFindTailsImpl(const TString& prefix) {
    TCompactTrieBuilder<> builder;

    TMap<TString, ui64> input;

    for (auto& i : SampleData) {
        TString temp = i;
        ui64 val = temp.size() * 2;
        builder.Add(temp.data(), temp.size(), val);
        if (temp.StartsWith(prefix)) {
            input[temp.substr(prefix.size())] = val;
        }
    }

    typedef TCompactTrie<> TTrie;

    TBufferStream stream;
    size_t len = builder.Save(stream);
    TTrie trie(stream.Buffer().Data(), len);

    TTrie subtrie = trie.FindTails(prefix.data(), prefix.size());

    TMap<TString, ui64> output;

    for (TTrie::TConstIterator i = subtrie.Begin(), mi = subtrie.End(); i != mi; ++i) {
        TTrie::TValueType val = *i;
        output[TString(val.first.data(), val.first.size())] = val.second;
    }
    UNIT_ASSERT(input.size() == output.size());
    UNIT_ASSERT(input == output);

    TBufferStream buftmp;
    CompactTrieMinimize<TTrie::TPacker>(buftmp, stream.Buffer().Data(), len, false);
    TTrie trieMin(buftmp.Buffer().Data(), buftmp.Buffer().Size());

    subtrie = trieMin.FindTails(prefix.data(), prefix.size());
    output.clear();

    for (TTrie::TConstIterator i = subtrie.Begin(), mi = subtrie.End(); i != mi; ++i) {
        TTrie::TValueType val = *i;
        output[TString(val.first.data(), val.first.size())] = val.second;
    }
    UNIT_ASSERT(input.size() == output.size());
    UNIT_ASSERT(input == output);
}

void TCompactTrieTest::TestPrefixGrouped() {
    TBuffer b1b;
    TCompactTrieBuilder<char, ui32> b1(CTBF_PREFIX_GROUPED);
    const char* data[] = {
        "Kazan",
        "Moscow",
        "Monino",
        "Murmansk",
        "Fryanovo",
        "Fryazino",
        "Fryazevo",
        "Tumen",
    };

    for (size_t i = 0; i < Y_ARRAY_SIZE(data); ++i) {
        ui32 val = strlen(data[i]) + 1;
        b1.Add(data[i], strlen(data[i]), val);
        for (size_t j = 0; j < Y_ARRAY_SIZE(data); ++j) {
            ui32 mustHave = strlen(data[j]) + 1;
            ui32 found = 0;
            if (j <= i) {
                UNIT_ASSERT(b1.Find(data[j], strlen(data[j]), &found));
                UNIT_ASSERT_VALUES_EQUAL(mustHave, found);
            } else {
                UNIT_ASSERT(!b1.Find(data[j], strlen(data[j]), &found));
            }
        }
    }

    {
        TBufferOutput b1bo(b1b);
        b1.Save(b1bo);
    }

    TCompactTrie<char, ui32> t1(TBlob::FromBuffer(b1b));

    //t1.Print(Cerr);

    for (auto& i : data) {
        ui32 v;
        UNIT_ASSERT(t1.Find(i, strlen(i), &v));
        UNIT_ASSERT_VALUES_EQUAL(strlen(i) + 1, v);
    }
}

void TCompactTrieTest::CrashTestPrefixGrouped() {
    TCompactTrieBuilder<char, ui32> builder(CTBF_PREFIX_GROUPED);
    const char* data[] = {
        "Fryazino",
        "Fryanovo",
        "Monino",
        "",
        "Fryazevo",
    };
    bool wasException = false;
    try {
        for (size_t i = 0; i < Y_ARRAY_SIZE(data); ++i) {
            builder.Add(data[i], strlen(data[i]), i + 1);
        }
    } catch (const yexception& e) {
        wasException = true;
        UNIT_ASSERT(strstr(e.what(), "Bad input order - expected input strings to be prefix-grouped."));
    }
    UNIT_ASSERT_C(wasException, "CrashTestPrefixGrouped");
}

void TCompactTrieTest::TestMergeFromFile() {
    {
        TCompactTrieBuilder<> b;
        b.Add("yandex", 12);
        b.Add("google", 13);
        b.Add("mail", 14);
        TUnbufferedFileOutput out(GetSystemTempDir() + "/TCompactTrieTest-TestMerge-ru");
        b.Save(out);
    }

    {
        TCompactTrieBuilder<> b;
        b.Add("yandex", 112);
        b.Add("google", 113);
        b.Add("yahoo", 114);
        TUnbufferedFileOutput out(GetSystemTempDir() + "/TCompactTrieTest-TestMerge-com");
        b.Save(out);
    }

    {
        TCompactTrieBuilder<> b;
        UNIT_ASSERT(b.AddSubtreeInFile("com.", GetSystemTempDir() + "/TCompactTrieTest-TestMerge-com"));
        UNIT_ASSERT(b.Add("org.kernel", 22));
        UNIT_ASSERT(b.AddSubtreeInFile("ru.", GetSystemTempDir() + "/TCompactTrieTest-TestMerge-ru"));
        TUnbufferedFileOutput out(GetSystemTempDir() + "/TCompactTrieTest-TestMerge-res");
        b.Save(out);
    }

    TCompactTrie<> trie(TBlob::FromFileSingleThreaded(GetSystemTempDir() + "/TCompactTrieTest-TestMerge-res"));
    UNIT_ASSERT_VALUES_EQUAL(12u, trie.Get("ru.yandex"));
    UNIT_ASSERT_VALUES_EQUAL(13u, trie.Get("ru.google"));
    UNIT_ASSERT_VALUES_EQUAL(14u, trie.Get("ru.mail"));
    UNIT_ASSERT_VALUES_EQUAL(22u, trie.Get("org.kernel"));
    UNIT_ASSERT_VALUES_EQUAL(112u, trie.Get("com.yandex"));
    UNIT_ASSERT_VALUES_EQUAL(113u, trie.Get("com.google"));
    UNIT_ASSERT_VALUES_EQUAL(114u, trie.Get("com.yahoo"));

    unlink((GetSystemTempDir() + "/TCompactTrieTest-TestMerge-res").data());
    unlink((GetSystemTempDir() + "/TCompactTrieTest-TestMerge-com").data());
    unlink((GetSystemTempDir() + "/TCompactTrieTest-TestMerge-ru").data());
}

void TCompactTrieTest::TestMergeFromBuffer() {
    TArrayWithSizeHolder<char> buffer1;
    {
        TCompactTrieBuilder<> b;
        b.Add("aaaaa", 1);
        b.Add("bbbbb", 2);
        b.Add("ccccc", 3);
        buffer1.Resize(b.MeasureByteSize());
        TMemoryOutput out(buffer1.Get(), buffer1.Size());
        b.Save(out);
    }

    TArrayWithSizeHolder<char> buffer2;
    {
        TCompactTrieBuilder<> b;
        b.Add("aaaaa", 10);
        b.Add("bbbbb", 20);
        b.Add("ccccc", 30);
        b.Add("xxxxx", 40);
        b.Add("yyyyy", 50);
        buffer2.Resize(b.MeasureByteSize());
        TMemoryOutput out(buffer2.Get(), buffer2.Size());
        b.Save(out);
    }

    {
        TCompactTrieBuilder<> b;
        UNIT_ASSERT(b.AddSubtreeInBuffer("com.", std::move(buffer1)));
        UNIT_ASSERT(b.Add("org.upyachka", 42));
        UNIT_ASSERT(b.AddSubtreeInBuffer("ru.", std::move(buffer2)));
        TUnbufferedFileOutput out(GetSystemTempDir() + "/TCompactTrieTest-TestMergeFromBuffer-res");
        b.Save(out);
    }

    TCompactTrie<> trie(TBlob::FromFileSingleThreaded(GetSystemTempDir() + "/TCompactTrieTest-TestMergeFromBuffer-res"));
    UNIT_ASSERT_VALUES_EQUAL(10u, trie.Get("ru.aaaaa"));
    UNIT_ASSERT_VALUES_EQUAL(20u, trie.Get("ru.bbbbb"));
    UNIT_ASSERT_VALUES_EQUAL(40u, trie.Get("ru.xxxxx"));
    UNIT_ASSERT_VALUES_EQUAL(42u, trie.Get("org.upyachka"));
    UNIT_ASSERT_VALUES_EQUAL(1u, trie.Get("com.aaaaa"));
    UNIT_ASSERT_VALUES_EQUAL(2u, trie.Get("com.bbbbb"));
    UNIT_ASSERT_VALUES_EQUAL(3u, trie.Get("com.ccccc"));

    unlink((GetSystemTempDir() + "/TCompactTrieTest-TestMergeFromBuffer-res").data());
}

void TCompactTrieTest::TestUnique() {
    TestUniqueImpl(false);
    TestUniqueImpl(true);
}

void TCompactTrieTest::TestUniqueImpl(bool isPrefixGrouped) {
    TCompactTrieBuilder<char, ui32> builder(CTBF_UNIQUE | (isPrefixGrouped ? CTBF_PREFIX_GROUPED : CTBF_NONE));
    const char* data[] = {
        "Kazan",
        "Moscow",
        "Monino",
        "Murmansk",
        "Fryanovo",
        "Fryazino",
        "Fryazevo",
        "Fry",
        "Tumen",
    };
    for (size_t i = 0; i < Y_ARRAY_SIZE(data); ++i) {
        UNIT_ASSERT_C(builder.Add(data[i], strlen(data[i]), i + 1), i);
    }
    bool wasException = false;
    try {
        builder.Add(data[4], strlen(data[4]), 20);
    } catch (const yexception& e) {
        wasException = true;
        UNIT_ASSERT(strstr(e.what(), "Duplicate key"));
    }
    UNIT_ASSERT_C(wasException, "TestUnique");
}

void TCompactTrieTest::TestAddRetValue() {
    TCompactTrieBuilder<char, ui32> builder;
    const char* data[] = {
        "Kazan",
        "Moscow",
        "Monino",
        "Murmansk",
        "Fryanovo",
        "Fryazino",
        "Fryazevo",
        "Fry",
        "Tumen",
    };
    for (size_t i = 0; i < Y_ARRAY_SIZE(data); ++i) {
        UNIT_ASSERT(builder.Add(data[i], strlen(data[i]), i + 1));
        UNIT_ASSERT(!builder.Add(data[i], strlen(data[i]), i + 2));
        ui32 value;
        UNIT_ASSERT(builder.Find(data[i], strlen(data[i]), &value));
        UNIT_ASSERT(value == i + 2);
    }
}

void TCompactTrieTest::TestClear() {
    TCompactTrieBuilder<char, ui32> builder;
    const char* data[] = {
        "Kazan",
        "Moscow",
        "Monino",
        "Murmansk",
        "Fryanovo",
        "Fryazino",
        "Fryazevo",
        "Fry",
        "Tumen",
    };
    for (size_t i = 0; i < Y_ARRAY_SIZE(data); ++i) {
        builder.Add(data[i], strlen(data[i]), i + 1);
    }
    UNIT_ASSERT(builder.GetEntryCount() == Y_ARRAY_SIZE(data));
    builder.Clear();
    UNIT_ASSERT(builder.GetEntryCount() == 0);
    UNIT_ASSERT(builder.GetNodeCount() == 1);
}

void TCompactTrieTest::TestFindTails() {
    TestFindTailsImpl("aa");
    TestFindTailsImpl("bb");
    TestFindTailsImpl("fb");
    TestFindTailsImpl("fbc");
    TestFindTailsImpl("fbbaa");
}

template <class T>
class TCompactTrieTest::TDummyPacker: public TNullPacker<T> {
public:
    static T Data(const TString&) {
        T data;
        TNullPacker<T>().UnpackLeaf(nullptr, data);
        return data;
    }

    typedef T TData;
};

class TCompactTrieTest::TStrokaPacker: public TCompactTriePacker<TString> {
public:
    typedef TString TData;

    static TString Data(const TString& str) {
        return str;
    }
};

template <class T>
class TCompactTrieTest::TIntPacker: public TCompactTriePacker<T> {
public:
    typedef T TData;

    static TData Data(const TString&) {
        return RandomNumber<std::make_unsigned_t<T>>();
    }
};

void TCompactTrieTest::TestIterateEmptyKey() {
    TBuffer trieBuffer;
    {
        TCompactTrieBuilder<char, ui32> builder;
        UNIT_ASSERT(builder.Add("", 1));
        TBufferStream trieBufferO(trieBuffer);
        builder.Save(trieBufferO);
    }
    TCompactTrie<char, ui32> trie(TBlob::FromBuffer(trieBuffer));
    ui32 val;
    UNIT_ASSERT(trie.Find("", &val));
    UNIT_ASSERT(val == 1);
    TCompactTrie<char, ui32>::TConstIterator it = trie.Begin();
    UNIT_ASSERT(it.GetKey().empty());
    UNIT_ASSERT(it.GetValue() == 1);
}

void TCompactTrieTest::TestTrieSet() {
    TBuffer buffer;
    {
        TCompactTrieSet<char>::TBuilder builder;
        UNIT_ASSERT(builder.Add("a", 0));
        UNIT_ASSERT(builder.Add("ab", 1));
        UNIT_ASSERT(builder.Add("abc", 1));
        UNIT_ASSERT(builder.Add("abcd", 0));
        UNIT_ASSERT(!builder.Add("abcd", 1));

        TBufferStream stream(buffer);
        builder.Save(stream);
    }

    TCompactTrieSet<char> set(TBlob::FromBuffer(buffer));
    UNIT_ASSERT(set.Has("a"));
    UNIT_ASSERT(set.Has("ab"));
    UNIT_ASSERT(set.Has("abc"));
    UNIT_ASSERT(set.Has("abcd"));
    UNIT_ASSERT(!set.Has("abcde"));
    UNIT_ASSERT(!set.Has("aa"));
    UNIT_ASSERT(!set.Has("b"));
    UNIT_ASSERT(!set.Has(""));

    TCompactTrieSet<char> tails;
    UNIT_ASSERT(set.FindTails("a", tails));
    UNIT_ASSERT(tails.Has("b"));
    UNIT_ASSERT(tails.Has("bcd"));
    UNIT_ASSERT(!tails.Has("ab"));
    UNIT_ASSERT(!set.Has(""));

    TCompactTrieSet<char> empty;
    UNIT_ASSERT(set.FindTails("abcd", empty));
    UNIT_ASSERT(!empty.Has("a"));
    UNIT_ASSERT(!empty.Has("b"));
    UNIT_ASSERT(!empty.Has("c"));
    UNIT_ASSERT(!empty.Has("d"));
    UNIT_ASSERT(!empty.Has("d"));

    UNIT_ASSERT(empty.Has("")); // contains only empty string
}

// Tests for trie with vector (list, set) values

TVector<TUtf16String> TCompactTrieTest::GetSampleKeys(size_t nKeys) const {
    Y_ASSERT(nKeys <= 10);
    TString sampleKeys[] = {"a", "b", "ac", "bd", "abe", "bcf", "deg", "ah", "xy", "abc"};
    TVector<TUtf16String> result;
    for (size_t i = 0; i < nKeys; i++)
        result.push_back(ASCIIToWide(sampleKeys[i]));
    return result;
}

template <class TContainer>
TVector<TContainer> TCompactTrieTest::GetSampleVectorData(size_t nValues) {
    TVector<TContainer> data;
    for (size_t i = 0; i < nValues; i++) {
        data.push_back(TContainer());
        for (size_t j = 0; j < i; j++)
            data[i].insert(data[i].end(), (typename TContainer::value_type)((j == 3) ? 0 : (1 << (j * 5))));
    }
    return data;
}

template <class TContainer>
TVector<TContainer> TCompactTrieTest::GetSampleTextVectorData(size_t nValues) {
    TVector<TContainer> data;
    for (size_t i = 0; i < nValues; i++) {
        data.push_back(TContainer());
        for (size_t j = 0; j < i; j++)
            data[i].insert(data[i].end(), TString("abc") + ToString<size_t>(j));
    }
    return data;
}

template <class T>
void TCompactTrieTest::CheckEquality(const T& value1, const T& value2) const {
    UNIT_ASSERT_VALUES_EQUAL(value1, value2);
}

template <>
void TCompactTrieTest::CheckEquality<TVector<i64>>(const TVector<i64>& value1, const TVector<i64>& value2) const {
    UNIT_ASSERT_VALUES_EQUAL(value1.size(), value2.size());
    for (size_t i = 0; i < value1.size(); i++)
        UNIT_ASSERT_VALUES_EQUAL(value1[i], value2[i]);
}

template <class TContainer>
void TCompactTrieTest::TestTrieWithContainers(const TVector<TUtf16String>& keys, const TVector<TContainer>& sampleData, TString methodName) {
    TString fileName = GetSystemTempDir() + "/TCompactTrieTest-TestTrieWithContainers-" + methodName;

    TCompactTrieBuilder<wchar16, TContainer> b;
    for (size_t i = 0; i < keys.size(); i++) {
        b.Add(keys[i], sampleData[i]);
    }
    TUnbufferedFileOutput out(fileName);
    b.Save(out);

    TCompactTrie<wchar16, TContainer> trie(TBlob::FromFileSingleThreaded(fileName));
    for (size_t i = 0; i < keys.size(); i++) {
        TContainer value = trie.Get(keys[i]);
        UNIT_ASSERT_VALUES_EQUAL(value.size(), sampleData[i].size());
        typename TContainer::const_iterator p = value.begin();
        typename TContainer::const_iterator p1 = sampleData[i].begin();
        for (; p != value.end(); p++, p1++)
            CheckEquality<typename TContainer::value_type>(*p, *p1);
    }

    unlink(fileName.data());
}

template <>
void TCompactTrieTest::TestTrieWithContainers<std::pair<TUtf16String, TVector<i64>>>(const TVector<TUtf16String>& keys, const TVector<std::pair<TUtf16String, TVector<i64>>>& sampleData, TString methodName) {
    typedef std::pair<TUtf16String, TVector<i64>> TContainer;
    TString fileName = GetSystemTempDir() + "/TCompactTrieTest-TestTrieWithContainers-" + methodName;

    TCompactTrieBuilder<wchar16, TContainer> b;
    for (size_t i = 0; i < keys.size(); i++) {
        b.Add(keys[i], sampleData[i]);
    }
    TUnbufferedFileOutput out(fileName);
    b.Save(out);

    TCompactTrie<wchar16, TContainer> trie(TBlob::FromFileSingleThreaded(fileName));
    for (size_t i = 0; i < keys.size(); i++) {
        TContainer value = trie.Get(keys[i]);
        CheckEquality<TContainer::first_type>(value.first, sampleData[i].first);
        CheckEquality<TContainer::second_type>(value.second, sampleData[i].second);
    }

    unlink(fileName.data());
}

void TCompactTrieTest::TestTrieForVectorInt64() {
    TestTrieWithContainers<TVector<i64>>(GetSampleKeys(10), GetSampleVectorData<TVector<i64>>(10), "v-i64");
}

void TCompactTrieTest::TestTrieForListInt64() {
    TestTrieWithContainers<TList<i64>>(GetSampleKeys(10), GetSampleVectorData<TList<i64>>(10), "l-i64");
}

void TCompactTrieTest::TestTrieForSetInt64() {
    TestTrieWithContainers<TSet<i64>>(GetSampleKeys(10), GetSampleVectorData<TSet<i64>>(10), "s-i64");
}

void TCompactTrieTest::TestTrieForVectorStroka() {
    TestTrieWithContainers<TVector<TString>>(GetSampleKeys(10), GetSampleTextVectorData<TVector<TString>>(10), "v-str");
}

void TCompactTrieTest::TestTrieForListStroka() {
    TestTrieWithContainers<TList<TString>>(GetSampleKeys(10), GetSampleTextVectorData<TList<TString>>(10), "l-str");
}

void TCompactTrieTest::TestTrieForSetStroka() {
    TestTrieWithContainers<TSet<TString>>(GetSampleKeys(10), GetSampleTextVectorData<TSet<TString>>(10), "s-str");
}

void TCompactTrieTest::TestTrieForVectorWtroka() {
    TVector<TVector<TString>> data = GetSampleTextVectorData<TVector<TString>>(10);
    TVector<TVector<TUtf16String>> wData;
    for (size_t i = 0; i < data.size(); i++) {
        wData.push_back(TVector<TUtf16String>());
        for (size_t j = 0; j < data[i].size(); j++)
            wData[i].push_back(UTF8ToWide(data[i][j]));
    }
    TestTrieWithContainers<TVector<TUtf16String>>(GetSampleKeys(10), wData, "v-wtr");
}

void TCompactTrieTest::TestTrieForVectorFloat() {
    TestTrieWithContainers<TVector<float>>(GetSampleKeys(10), GetSampleVectorData<TVector<float>>(10), "v-float");
}

void TCompactTrieTest::TestTrieForVectorDouble() {
    TestTrieWithContainers<TVector<double>>(GetSampleKeys(10), GetSampleVectorData<TVector<double>>(10), "v-double");
}

void TCompactTrieTest::TestTrieForListVectorInt64() {
    TVector<i64> tmp;
    tmp.push_back(0);
    TList<TVector<i64>> dataElement(5, tmp);
    TVector<TList<TVector<i64>>> data(10, dataElement);
    TestTrieWithContainers<TList<TVector<i64>>>(GetSampleKeys(10), data, "l-v-i64");
}

void TCompactTrieTest::TestTrieForPairWtrokaVectorInt64() {
    TVector<TUtf16String> keys = GetSampleKeys(10);
    TVector<TVector<i64>> values = GetSampleVectorData<TVector<i64>>(10);
    TVector<std::pair<TUtf16String, TVector<i64>>> data;
    for (size_t i = 0; i < 10; i++)
        data.push_back(std::pair<TUtf16String, TVector<i64>>(keys[i] + u"_v", values[i]));
    TestTrieWithContainers<std::pair<TUtf16String, TVector<i64>>>(keys, data, "pair-str-v-i64");
}

void TCompactTrieTest::TestEmptyValueOutOfOrder() {
    TBufferOutput buffer;
    using TSymbol = ui32;
    {
        TCompactTrieBuilder<TSymbol, ui32> builder;
        TSymbol key = 1;
        builder.Add(&key, 1, 10);
        builder.Add(nullptr, 0, 14);
        builder.Save(buffer);
    }
    {
        TCompactTrie<TSymbol, ui32> trie(buffer.Buffer().Data(), buffer.Buffer().Size());
        UNIT_ASSERT(trie.Find(nullptr, 0));
    }
}

void TCompactTrieTest::TestFindLongestPrefixWithEmptyValue() {
    TBufferOutput buffer;
    {
        TCompactTrieBuilder<wchar16, ui32> builder;
        builder.Add(u"", 42);
        builder.Add(u"yandex", 271828);
        builder.Add(u"ya", 31415);
        builder.Save(buffer);
    }
    {
        TCompactTrie<wchar16, ui32> trie(buffer.Buffer().Data(), buffer.Buffer().Size());
        size_t prefixLen = 123;
        ui32 value = 0;

        UNIT_ASSERT(trie.FindLongestPrefix(u"google", &prefixLen, &value));
        UNIT_ASSERT(prefixLen == 0);
        UNIT_ASSERT(value == 42);

        UNIT_ASSERT(trie.FindLongestPrefix(u"yahoo", &prefixLen, &value));
        UNIT_ASSERT(prefixLen == 2);
        UNIT_ASSERT(value == 31415);
    }
}

template <typename TChar>
struct TConvertKey {
    static inline TString Convert(const TStringBuf& key) {
        return ToString(key);
    }
};

template <>
struct TConvertKey<wchar16> {
    static inline TUtf16String Convert(const TStringBuf& key) {
        return UTF8ToWide(key);
    }
};

template <>
struct TConvertKey<wchar32> {
    static inline TUtf32String Convert(const TStringBuf& key) {
        return TUtf32String::FromUtf8(key);
    }
};

template <class TSearchIter, class TKeyBuf>
static void MoveIter(TSearchIter& iter, const TKeyBuf& key) {
    for (size_t i = 0; i < key.length(); ++i) {
        UNIT_ASSERT(iter.Advance(key[i]));
    }
}

template <typename TChar>
void TCompactTrieTest::TestSearchIterImpl() {
    TBufferOutput buffer;
    {
        TCompactTrieBuilder<TChar, ui32> builder;
        TStringBuf data[] = {
            TStringBuf("abaab"),
            TStringBuf("abcdef"),
            TStringBuf("abbbc"),
            TStringBuf("bdfaa"),
        };
        for (size_t i = 0; i < Y_ARRAY_SIZE(data); ++i) {
            builder.Add(TConvertKey<TChar>::Convert(data[i]), i + 1);
        }
        builder.Save(buffer);
    }

    TCompactTrie<TChar, ui32> trie(buffer.Buffer().Data(), buffer.Buffer().Size());
    ui32 value = 0;
    auto iter(MakeSearchIterator(trie));
    MoveIter(iter, TConvertKey<TChar>::Convert(TStringBuf("abc")));
    UNIT_ASSERT(!iter.GetValue(&value));

    iter = MakeSearchIterator(trie);
    MoveIter(iter, TConvertKey<TChar>::Convert(TStringBuf("abbbc")));
    UNIT_ASSERT(iter.GetValue(&value));
    UNIT_ASSERT_EQUAL(value, 3);

    iter = MakeSearchIterator(trie);
    UNIT_ASSERT(iter.Advance(TConvertKey<TChar>::Convert(TStringBuf("bdfa"))));
    UNIT_ASSERT(!iter.GetValue(&value));

    iter = MakeSearchIterator(trie);
    UNIT_ASSERT(iter.Advance(TConvertKey<TChar>::Convert(TStringBuf("bdfaa"))));
    UNIT_ASSERT(iter.GetValue(&value));
    UNIT_ASSERT_EQUAL(value, 4);

    UNIT_ASSERT(!MakeSearchIterator(trie).Advance(TChar('z')));
    UNIT_ASSERT(!MakeSearchIterator(trie).Advance(TConvertKey<TChar>::Convert(TStringBuf("cdf"))));
    UNIT_ASSERT(!MakeSearchIterator(trie).Advance(TConvertKey<TChar>::Convert(TStringBuf("abca"))));
}

void TCompactTrieTest::TestSearchIterChar() {
    TestSearchIterImpl<char>();
}

void TCompactTrieTest::TestSearchIterWchar() {
    TestSearchIterImpl<wchar16>();
}

void TCompactTrieTest::TestSearchIterWchar32() {
    TestSearchIterImpl<wchar32>();
}

void TCompactTrieTest::TestCopyAndAssignment() {
    TBufferOutput bufout;
    typedef TCompactTrie<> TTrie;
    CreateTrie<char>(bufout, false, false);
    TTrie trie(bufout.Buffer().Data(), bufout.Buffer().Size());
    TTrie copy(trie);
    UNIT_ASSERT(copy.HasCorrectSkipper());
    TTrie assign;
    assign = trie;
    UNIT_ASSERT(assign.HasCorrectSkipper());
    TTrie move(std::move(trie));
    UNIT_ASSERT(move.HasCorrectSkipper());
    TTrie moveAssign;
    moveAssign = TTrie(bufout.Buffer().Data(), bufout.Buffer().Size());
    UNIT_ASSERT(moveAssign.HasCorrectSkipper());
}

template <class TTrie>
void TCompactTrieTest::TestFirstSymbolIteratorForTrie(const TTrie& trie, const TStringBuf& narrowAnswers) {
    NCompactTrie::TFirstSymbolIterator<TTrie> it;
    it.SetTrie(trie, trie.GetSkipper());
    typename TTrie::TKey answers = MakeWideKey<typename TTrie::TSymbol>(narrowAnswers);
    auto answer = answers.begin();
    for (; !it.AtEnd(); it.MakeStep(), ++answer) {
        UNIT_ASSERT(answer != answers.end());
        UNIT_ASSERT(it.GetKey() == *answer);
    }
    UNIT_ASSERT(answer == answers.end());
}

template <class TSymbol>
void TCompactTrieTest::TestFirstSymbolIterator() {
    TBufferOutput bufout;
    typedef TCompactTrie<TSymbol> TTrie;
    CreateTrie<TSymbol>(bufout, false, false);
    TTrie trie(bufout.Buffer().Data(), bufout.Buffer().Size());
    TStringBuf rootAnswers = "abcdf";
    TestFirstSymbolIteratorForTrie(trie, rootAnswers);
    TStringBuf aAnswers = "abcd";
    TestFirstSymbolIteratorForTrie(trie.FindTails(MakeWideKey<TSymbol>("a", 1)), aAnswers);
}

void TCompactTrieTest::TestFirstSymbolIterator8() {
    TestFirstSymbolIterator<char>();
}

void TCompactTrieTest::TestFirstSymbolIterator16() {
    TestFirstSymbolIterator<wchar16>();
}

void TCompactTrieTest::TestFirstSymbolIterator32() {
    TestFirstSymbolIterator<ui32>();
}

void TCompactTrieTest::TestFirstSymbolIteratorChar32() {
    TestFirstSymbolIterator<wchar32>();
}


void TCompactTrieTest::TestArrayPacker() {
    using TDataInt = std::array<int, 2>;
    const std::pair<TString, TDataInt> dataXxx{"xxx", {{15, 16}}};
    const std::pair<TString, TDataInt> dataYyy{"yyy", {{20, 30}}};

    TCompactTrieBuilder<char, TDataInt> trieBuilderOne;
    trieBuilderOne.Add(dataXxx.first, dataXxx.second);
    trieBuilderOne.Add(dataYyy.first, dataYyy.second);

    TBufferOutput bufferOne;
    trieBuilderOne.Save(bufferOne);

    const TCompactTrie<char, TDataInt> trieOne(bufferOne.Buffer().Data(), bufferOne.Buffer().Size());
    UNIT_ASSERT_VALUES_EQUAL(dataXxx.second, trieOne.Get(dataXxx.first));
    UNIT_ASSERT_VALUES_EQUAL(dataYyy.second, trieOne.Get(dataYyy.first));

    using TDataStroka = std::array<TString, 2>;
    const std::pair<TString, TDataStroka> dataZzz{"zzz", {{"hello", "there"}}};
    const std::pair<TString, TDataStroka> dataWww{"www", {{"half", "life"}}};

    TCompactTrieBuilder<char, TDataStroka> trieBuilderTwo;
    trieBuilderTwo.Add(dataZzz.first, dataZzz.second);
    trieBuilderTwo.Add(dataWww.first, dataWww.second);

    TBufferOutput bufferTwo;
    trieBuilderTwo.Save(bufferTwo);

    const TCompactTrie<char, TDataStroka> trieTwo(bufferTwo.Buffer().Data(), bufferTwo.Buffer().Size());
    UNIT_ASSERT_VALUES_EQUAL(dataZzz.second, trieTwo.Get(dataZzz.first));
    UNIT_ASSERT_VALUES_EQUAL(dataWww.second, trieTwo.Get(dataWww.first));
}

void TCompactTrieTest::TestBuilderFindLongestPrefix() {
    const size_t sizes[] = {10, 100};
    const double branchProbabilities[] = {0.01, 0.1, 0.5, 0.9, 0.99};
    for (size_t size : sizes) {
        for (double branchProbability : branchProbabilities) {
            TestBuilderFindLongestPrefix(size, branchProbability, false, false);
            TestBuilderFindLongestPrefix(size, branchProbability, false, true);
            TestBuilderFindLongestPrefix(size, branchProbability, true, false);
            TestBuilderFindLongestPrefix(size, branchProbability, true, true);
        }
    }
}

void TCompactTrieTest::TestBuilderFindLongestPrefix(size_t keysCount, double branchProbability, bool isPrefixGrouped, bool hasEmptyKey) {
    TVector<TString> keys;
    TString keyToAdd;
    for (size_t i = 0; i < keysCount; ++i) {
        const size_t prevKeyLen = keyToAdd.size();
        // add two random chars to prev key
        keyToAdd += RandChar();
        keyToAdd += RandChar();
        const bool changeBranch = prevKeyLen && RandomNumber<double>() < branchProbability;
        if (changeBranch) {
            const size_t branchPlace = RandomNumber<size_t>(prevKeyLen + 1); // random place in [0, prevKeyLen]
            *(keyToAdd.begin() + branchPlace) = RandChar();
        }
        keys.push_back(keyToAdd);
    }

    if (isPrefixGrouped)
        Sort(keys.begin(), keys.end());
    else
        Shuffle(keys.begin(), keys.end());

    TCompactTrieBuilder<char, TString> builder(isPrefixGrouped ? CTBF_PREFIX_GROUPED : CTBF_NONE);
    const TString EMPTY_VALUE = "empty";
    if (hasEmptyKey)
        builder.Add(nullptr, 0, EMPTY_VALUE);

    for (size_t i = 0; i < keysCount; ++i) {
        const TString& key = keys[i];

        for (size_t j = 0; j < keysCount; ++j) {
            const TString& otherKey = keys[j];
            const bool exists = j < i;
            size_t expectedSize = 0;
            if (exists) {
                expectedSize = otherKey.size();
            } else {
                size_t max = 0;
                for (size_t k = 0; k < i; ++k)
                    if (keys[k].size() < otherKey.size() && keys[k].size() > max && otherKey.StartsWith(keys[k]))
                        max = keys[k].size();
                expectedSize = max;
            }

            size_t prefixSize = 0xfcfcfc;
            TString value = "abcd";
            const bool expectedResult = hasEmptyKey || expectedSize != 0;
            UNIT_ASSERT_VALUES_EQUAL_C(expectedResult, builder.FindLongestPrefix(otherKey.data(), otherKey.size(), &prefixSize, &value), "otherKey = " << HexEncode(otherKey));
            if (expectedResult) {
                UNIT_ASSERT_VALUES_EQUAL(expectedSize, prefixSize);
                if (expectedSize) {
                    UNIT_ASSERT_VALUES_EQUAL(TStringBuf(otherKey).SubStr(0, prefixSize), value);
                } else {
                    UNIT_ASSERT_VALUES_EQUAL(EMPTY_VALUE, value);
                }
            } else {
                UNIT_ASSERT_VALUES_EQUAL("abcd", value);
                UNIT_ASSERT_VALUES_EQUAL(0xfcfcfc, prefixSize);
            }

            for (int c = 0; c < 10; ++c) {
                TString extendedKey = otherKey;
                extendedKey += RandChar();
                size_t extendedPrefixSize = 0xdddddd;
                TString extendedValue = "dcba";
                UNIT_ASSERT_VALUES_EQUAL(expectedResult, builder.FindLongestPrefix(extendedKey.data(), extendedKey.size(), &extendedPrefixSize, &extendedValue));
                if (expectedResult) {
                    UNIT_ASSERT_VALUES_EQUAL(value, extendedValue);
                    UNIT_ASSERT_VALUES_EQUAL(prefixSize, extendedPrefixSize);
                } else {
                    UNIT_ASSERT_VALUES_EQUAL("dcba", extendedValue);
                    UNIT_ASSERT_VALUES_EQUAL(0xdddddd, extendedPrefixSize);
                }
            }
        }
        builder.Add(key.data(), key.size(), key);
    }

    TBufferOutput buffer;
    builder.Save(buffer);
}

void TCompactTrieTest::TestBuilderFindLongestPrefixWithEmptyValue() {
    TCompactTrieBuilder<wchar16, ui32> builder;
    builder.Add(u"", 42);
    builder.Add(u"yandex", 271828);
    builder.Add(u"ya", 31415);

    size_t prefixLen = 123;
    ui32 value = 0;

    UNIT_ASSERT(builder.FindLongestPrefix(u"google", &prefixLen, &value));
    UNIT_ASSERT_VALUES_EQUAL(prefixLen, 0);
    UNIT_ASSERT_VALUES_EQUAL(value, 42);

    UNIT_ASSERT(builder.FindLongestPrefix(u"yahoo", &prefixLen, &value));
    UNIT_ASSERT_VALUES_EQUAL(prefixLen, 2);
    UNIT_ASSERT_VALUES_EQUAL(value, 31415);

    TBufferOutput buffer;
    builder.Save(buffer);
}

void TCompactTrieTest::TestPatternSearcherEmpty() {
    TCompactPatternSearcherBuilder<char, ui32> builder;

    TBufferOutput searcherData;
    builder.Save(searcherData);

    TCompactPatternSearcher<char, ui32> searcher(
        searcherData.Buffer().Data(),
        searcherData.Buffer().Size()
    );

    UNIT_ASSERT(searcher.SearchMatches("a").empty());
    UNIT_ASSERT(searcher.SearchMatches("").empty());
    UNIT_ASSERT(searcher.SearchMatches("abc").empty());
}

void TCompactTrieTest::TestPatternSearcherOnDataset(
    const TVector<TString>& patterns,
    const TVector<TString>& samples
) {
    TCompactPatternSearcherBuilder<char, ui32> builder;

    for (size_t patternIdx = 0; patternIdx < patterns.size(); ++patternIdx) {
        builder.Add(patterns[patternIdx], patternIdx);
    }

    TBufferOutput searcherData;
    builder.Save(searcherData);

    TCompactPatternSearcher<char, ui32> searcher(
        searcherData.Buffer().Data(),
        searcherData.Buffer().Size()
    );

    for (const auto& sample : samples) {
        const auto matches = searcher.SearchMatches(sample);

        size_t matchesNum = 0;
        THashSet<TString> processedPatterns;
        for (const auto& pattern : patterns) {
            if (pattern.empty() || processedPatterns.contains(pattern)) {
                continue;
            }
            for (size_t start = 0; start + pattern.size() <= sample.size(); ++start) {
                matchesNum += (pattern == sample.substr(start, pattern.size()));
            }
            processedPatterns.insert(pattern);
        }
        UNIT_ASSERT_VALUES_EQUAL(matchesNum, matches.size());


        TSet<std::pair<size_t, ui32>> foundMatches;
        for (const auto& match : matches) {
            std::pair<size_t, ui32> matchParams(match.End, match.Data);
            UNIT_ASSERT(!foundMatches.contains(matchParams));
            foundMatches.insert(matchParams);

            const auto& pattern = patterns[match.Data];
            UNIT_ASSERT_VALUES_EQUAL(
                sample.substr(match.End - pattern.size() + 1, pattern.size()),
                pattern
            );
        }
    }
}

void TCompactTrieTest::TestPatternSearcherSimple() {
    TestPatternSearcherOnDataset(
        { // patterns
            "abcd",
            "abc",
            "ab",
            "a",
            ""
        },
        { // samples
            "abcde",
            "abcd",
            "abc",
            "ab",
            "a",
            ""
        }
    );
    TestPatternSearcherOnDataset(
        { // patterns
            "a"
            "ab",
            "abcd",
        },
        { // samples
            "abcde",
            "abcd",
            "abc",
            "ab",
            "a",
            ""
        }
    );
    TestPatternSearcherOnDataset(
        { // patterns
            "aaaa",
            "aaa",
            "aa",
            "a",
        },
        { // samples
            "aaaaaaaaaaaa"
        }
    );
    TestPatternSearcherOnDataset(
        { // patterns
            "aa", "ab", "ac", "ad", "ae", "af",
            "ba", "bb", "bc", "bd", "be", "bf",
            "ca", "cb", "cc", "cd", "ce", "cf",
            "da", "db", "dc", "dd", "de", "df",
            "ea", "eb", "ec", "ed", "ee", "ef",
            "fa", "fb", "fc", "fd", "fe", "ff"
        },
        { // samples
            "dcabafeebfdcbacddacadbaabecdbaeffecdbfabcdcabcfaefecdfebacfedacefbdcacfeb",
            "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefancdefancdef",
            "fedcbafedcbafedcbafedcbafedcbafedcbafedcbafedcbafedcbafedcbafedcbafedcba",
            "",
            "a", "b", "c", "d", "e", "f",
            "aa", "ab", "ac", "ad", "ae", "af",
            "ba", "bb", "bc", "bd", "be", "bf",
            "ca", "cb", "cc", "cd", "ce", "cf",
            "da", "db", "dc", "dd", "de", "df",
            "ea", "eb", "ec", "ed", "ee", "ef",
            "fa", "fb", "fc", "fd", "fe", "ff"
        }
    );
}

static char RandChar(
    TFastRng<ui64>& rng,
    int maxChar
) {
    return static_cast<char>(rng.GenRand() % (maxChar + 1));
}

static TString RandStr(
    TFastRng<ui64>& rng,
    size_t maxLength,
    int maxChar,
    bool nonEmpty = false
) {
    Y_ASSERT(maxLength > 0);

    size_t length;
    if (nonEmpty) {
        length = rng.GenRand() % maxLength + 1;
    } else {
        length = rng.GenRand() % (maxLength + 1);
    }

    TString result;
    while (result.size() < length) {
        result += RandChar(rng, maxChar);
    }

    return result;
}

void TCompactTrieTest::TestPatternSearcherRandom(
    size_t patternsNum,
    size_t patternMaxLength,
    size_t strMaxLength,
    int maxChar,
    TFastRng<ui64>& rng
) {
    auto patternToSearch = RandStr(rng, patternMaxLength, maxChar, /*nonEmpty*/true);

    TVector<TString> patterns = {patternToSearch};
    while (patterns.size() < patternsNum) {
        patterns.push_back(RandStr(rng, patternMaxLength, maxChar, /*nonEmpty*/true));
    }

    auto filler = RandStr(rng, strMaxLength - patternToSearch.size() + 1, maxChar);
    size_t leftFillerSize = rng.GenRand() % (filler.size() + 1);
    auto leftFiller = filler.substr(0, leftFillerSize);
    auto rightFiller = filler.substr(leftFillerSize, filler.size() - leftFillerSize);
    auto sample = leftFiller + patternToSearch + rightFiller;

    TestPatternSearcherOnDataset(patterns, {sample});
}

void TCompactTrieTest::TestPatternSearcherRandom() {
    TFastRng<ui64> rng(0);
    for (size_t patternMaxLen : {1, 2, 10}) {
        for (size_t strMaxLen : TVector<size_t>{patternMaxLen, 2 * patternMaxLen, 10}) {
            for (int maxChar : {0, 1, 5, 255}) {
                for (size_t patternsNum : {1, 10}) {
                    for (size_t testIdx = 0; testIdx < 3; ++testIdx) {
                        TestPatternSearcherRandom(
                            patternsNum,
                            patternMaxLen,
                            strMaxLen,
                            maxChar,
                            rng
                        );
                    }
                }
            }
        }
    }
}
