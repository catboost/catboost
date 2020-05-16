#include <library/cpp/unittest/registar.h>
#include <library/cpp/containers/comptrie/comptrie.h>
#include <library/cpp/containers/comptrie/loader/loader.h>

using TDummyTrie = TCompactTrie<char, i32>;

namespace {
    const unsigned char DATA[] = {
#include "data.inc"
    };
}

Y_UNIT_TEST_SUITE(ArchiveLoaderTests) {
    Y_UNIT_TEST(BaseTest) {
        TDummyTrie trie = LoadTrieFromArchive<TDummyTrie>("/dummy.trie", DATA, true);
        UNIT_ASSERT_EQUAL(trie.Size(), 3);

        const TString TrieKyes[3] = {
            "zero", "one", "two"};
        i32 val = -1;
        for (i32 i = 0; i < 3; ++i) {
            UNIT_ASSERT(trie.Find(TrieKyes[i].data(), TrieKyes[i].size(), &val));
            UNIT_ASSERT_EQUAL(i, val);
        }

        UNIT_CHECK_GENERATED_EXCEPTION(
            LoadTrieFromArchive<TDummyTrie>("/noname.trie", DATA),
            yexception);
    }
}
