#include "join.h"
#include "subst.h"
#include <string>

#include <library/unittest/registar.h>

SIMPLE_UNIT_TEST_SUITE(TStringSubst) {
    static const size_t MIN_FROM_CTX = 4;
    static const yvector<TString> ALL_FROM{TString("F"), TString("FF")};
    static const yvector<TString> ALL_TO{TString(""), TString("T"), TString("TT"), TString("TTT")};

    static void AssertSubstGlobal(const TString& sFrom, const TString& sTo, const TString& from, const TString& to, const size_t fromPos, const size_t numSubst) {
        TString s = sFrom;
        size_t res = SubstGlobal(s, from, to, fromPos);
        UNIT_ASSERT_VALUES_EQUAL_C(res, numSubst,
                                   TStringBuilder() << "numSubst=" << numSubst << ", fromPos=" << fromPos << ", " << sFrom << " -> " << sTo);
        if (numSubst) {
            UNIT_ASSERT_STRINGS_EQUAL_C(s, sTo,
                                        TStringBuilder() << "numSubst=" << numSubst << ", fromPos=" << fromPos << ", " << sFrom << " -> " << sTo);
        } else {
            // ensure s didn't trigger copy-on-write
            UNIT_ASSERT_VALUES_EQUAL_C(s.c_str(), sFrom.c_str(),
                                       TStringBuilder() << "numSubst=" << numSubst << ", fromPos=" << fromPos << ", " << sFrom << " -> " << sTo);
        }
    }

    SIMPLE_UNIT_TEST(TestSubstGlobalNoSubstA) {
        for (const auto& from : ALL_FROM) {
            const size_t fromSz = +from;
            const size_t minSz = fromSz;
            const size_t maxSz = fromSz + MIN_FROM_CTX;
            for (size_t sz = minSz; sz <= maxSz; ++sz) {
                for (size_t fromPos = 0; fromPos < sz; ++fromPos) {
                    TString s{sz, '.'};
                    for (const auto& to : ALL_TO) {
                        AssertSubstGlobal(s, s, from, to, fromPos, 0);
                    }
                }
            }
        }
    }

    SIMPLE_UNIT_TEST(TestSubstGlobalNoSubstB) {
        for (const auto& from : ALL_FROM) {
            const size_t fromSz = +from;
            const size_t minSz = fromSz;
            const size_t maxSz = fromSz + MIN_FROM_CTX;
            for (size_t sz = minSz; sz <= maxSz; ++sz) {
                for (size_t fromPos = 0; fromPos <= sz - fromSz; ++fromPos) {
                    for (size_t fromBeg = 0; fromBeg < fromPos; ++fromBeg) {
                        const auto parts = {
                            TString{fromBeg, '.'},
                            TString{sz - fromSz - fromBeg, '.'}};
                        TString s = JoinSeq(from, parts);
                        for (const auto& to : ALL_TO) {
                            AssertSubstGlobal(s, s, from, to, fromPos, 0);
                        }
                    }
                }
            }
        }
    }

    static void DoTestSubstGlobal(yvector<TString> & parts, const size_t minBeg, const size_t sz,
                                  const TString& from, const size_t fromPos, const size_t numSubst) {
        const size_t numLeft = numSubst - parts.size();
        for (size_t fromBeg = minBeg; fromBeg <= sz - numLeft * +from; ++fromBeg) {
            if (parts.empty()) {
                parts.emplace_back(fromBeg, '.');
            } else {
                parts.emplace_back(fromBeg - minBeg, '.');
            }

            if (numLeft == 1) {
                parts.emplace_back(sz - fromBeg - +from, '.');
                TString sFrom = JoinSeq(from, parts);
                UNIT_ASSERT_VALUES_EQUAL_C(+sFrom, sz, sFrom);
                for (const auto& to : ALL_TO) {
                    TString sTo = JoinSeq(to, parts);
                    AssertSubstGlobal(sFrom, sTo, from, to, fromPos, numSubst);
                }
                parts.pop_back();
            } else {
                DoTestSubstGlobal(parts, fromBeg + +from, sz, from, fromPos, numSubst);
            }

            parts.pop_back();
        }
    }

    static void DoTestSubstGlobal(size_t numSubst) {
        yvector<TString> parts;
        for (const auto& from : ALL_FROM) {
            const size_t fromSz = +from;
            const size_t minSz = numSubst * fromSz;
            const size_t maxSz = numSubst * (fromSz + MIN_FROM_CTX);
            for (size_t sz = minSz; sz <= maxSz; ++sz) {
                const size_t maxPos = sz - numSubst * fromSz;
                for (size_t fromPos = 0; fromPos <= maxPos; ++fromPos) {
                    DoTestSubstGlobal(parts, fromPos, sz, from, fromPos, numSubst);
                }
            }
        }
    }

    SIMPLE_UNIT_TEST(TestSubstGlobalSubst1) {
        DoTestSubstGlobal(1);
    }

    SIMPLE_UNIT_TEST(TestSubstGlobalSubst2) {
        DoTestSubstGlobal(2);
    }

    SIMPLE_UNIT_TEST(TestSubstGlobalSubst3) {
        DoTestSubstGlobal(3);
    }

    SIMPLE_UNIT_TEST(TestSubstGlobalSubst4) {
        DoTestSubstGlobal(4);
    }

    SIMPLE_UNIT_TEST(TestSubstGlobalOld) {
        TString s;
        s = "aaa";
        SubstGlobal(s, "a", "bb");
        UNIT_ASSERT_EQUAL(s, TString("bbbbbb"));
        s = "aaa";
        SubstGlobal(s, "a", "b");
        UNIT_ASSERT_EQUAL(s, TString("bbb"));
        s = "aaa";
        SubstGlobal(s, "a", "");
        UNIT_ASSERT_EQUAL(s, TString(""));
        s = "abcdefbcbcdfb";
        SubstGlobal(s, "bc", "bbc", 2);
        UNIT_ASSERT_EQUAL(s, TString("abcdefbbcbbcdfb"));
    }

    SIMPLE_UNIT_TEST(TestSubstCharGlobal) {
        TUtf16String w = TUtf16String::FromAscii("abcdabcd");
        SubstGlobal(w, TChar('b'), TChar('B'), 3);
        UNIT_ASSERT_EQUAL(w, TUtf16String::FromAscii("abcdaBcd"));

        TString s = "aaa";
        SubstGlobal(s, 'a', 'b', 1);
        UNIT_ASSERT_EQUAL(s, TString("abb"));
    }

    SIMPLE_UNIT_TEST(TestSubstStdString) {
        std::string s = "aaa";
        SubstGlobal(s, "a", "b", 1);
        UNIT_ASSERT_EQUAL(s, "abb");
    }

    SIMPLE_UNIT_TEST(TestSubstGlobalChar) {
        {
            const TString s = "a";
            const TString st = "b";
            TString ss = s;
            UNIT_ASSERT_VALUES_EQUAL(s.size(), SubstGlobal(ss, 'a', 'b'));
            UNIT_ASSERT_VALUES_EQUAL(st, ss);
        }
        {
            const TString s = "aa";
            const TString st = "bb";
            TString ss = s;
            UNIT_ASSERT_VALUES_EQUAL(s.size(), SubstGlobal(ss, 'a', 'b'));
            UNIT_ASSERT_VALUES_EQUAL(st, ss);
        }
        {
            const TString s = "aaa";
            const TString st = "bbb";
            TString ss = s;
            UNIT_ASSERT_VALUES_EQUAL(s.size(), SubstGlobal(ss, 'a', 'b'));
            UNIT_ASSERT_VALUES_EQUAL(st, ss);
        }
        {
            const TString s = "aaaa";
            const TString st = "bbbb";
            TString ss = s;
            UNIT_ASSERT_VALUES_EQUAL(s.size(), SubstGlobal(ss, 'a', 'b'));
            UNIT_ASSERT_VALUES_EQUAL(st, ss);
        }
        {
            const TString s = "aaaaa";
            const TString st = "bbbbb";
            TString ss = s;
            UNIT_ASSERT_VALUES_EQUAL(s.size(), SubstGlobal(ss, 'a', 'b'));
            UNIT_ASSERT_VALUES_EQUAL(st, ss);
        }
        {
            const TString s = "aaaaaa";
            const TString st = "bbbbbb";
            TString ss = s;
            UNIT_ASSERT_VALUES_EQUAL(s.size(), SubstGlobal(ss, 'a', 'b'));
            UNIT_ASSERT_VALUES_EQUAL(st, ss);
        }
        {
            const TString s = "aaaaaaa";
            const TString st = "bbbbbbb";
            TString ss = s;
            UNIT_ASSERT_VALUES_EQUAL(s.size(), SubstGlobal(ss, 'a', 'b'));
            UNIT_ASSERT_VALUES_EQUAL(st, ss);
        }
        {
            const TString s = "aaaaaaaa";
            const TString st = "bbbbbbbb";
            TString ss = s;
            UNIT_ASSERT_VALUES_EQUAL(s.size(), SubstGlobal(ss, 'a', 'b'));
            UNIT_ASSERT_VALUES_EQUAL(st, ss);
        }
    }
}
