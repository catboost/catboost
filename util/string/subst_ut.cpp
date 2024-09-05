#include "join.h"
#include "subst.h"
#include <string>

#include <library/cpp/testing/unittest/registar.h>

Y_UNIT_TEST_SUITE(TStringSubst) {
    static const size_t MIN_FROM_CTX = 4;
    static const TVector<TString> ALL_FROM{TString("F"), TString("FF")};
    static const TVector<TString> ALL_TO{TString(""), TString("T"), TString("TT"), TString("TTT")};

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

    Y_UNIT_TEST(TestSubstGlobalNoSubstA) {
        for (const auto& from : ALL_FROM) {
            const size_t fromSz = from.size();
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

    Y_UNIT_TEST(TestSubstGlobalNoSubstB) {
        for (const auto& from : ALL_FROM) {
            const size_t fromSz = from.size();
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

    static void DoTestSubstGlobal(TVector<TString>& parts, const size_t minBeg, const size_t sz,
                                  const TString& from, const size_t fromPos, const size_t numSubst) {
        const size_t numLeft = numSubst - parts.size();
        for (size_t fromBeg = minBeg; fromBeg <= sz - numLeft * from.size(); ++fromBeg) {
            if (parts.empty()) {
                parts.emplace_back(fromBeg, '.');
            } else {
                parts.emplace_back(fromBeg - minBeg, '.');
            }

            if (numLeft == 1) {
                parts.emplace_back(sz - fromBeg - from.size(), '.');
                TString sFrom = JoinSeq(from, parts);
                UNIT_ASSERT_VALUES_EQUAL_C(sFrom.size(), sz, sFrom);
                for (const auto& to : ALL_TO) {
                    TString sTo = JoinSeq(to, parts);
                    AssertSubstGlobal(sFrom, sTo, from, to, fromPos, numSubst);
                }
                parts.pop_back();
            } else {
                DoTestSubstGlobal(parts, fromBeg + from.size(), sz, from, fromPos, numSubst);
            }

            parts.pop_back();
        }
    }

    static void DoTestSubstGlobal(size_t numSubst) {
        TVector<TString> parts;
        for (const auto& from : ALL_FROM) {
            const size_t fromSz = from.size();
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

    Y_UNIT_TEST(TestSubstGlobalSubst1) {
        DoTestSubstGlobal(1);
    }

    Y_UNIT_TEST(TestSubstGlobalSubst2) {
        DoTestSubstGlobal(2);
    }

    Y_UNIT_TEST(TestSubstGlobalSubst3) {
        DoTestSubstGlobal(3);
    }

    Y_UNIT_TEST(TestSubstGlobalSubst4) {
        DoTestSubstGlobal(4);
    }

    Y_UNIT_TEST(TestSubstGlobalOld) {
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
        s = "Москва ~ Париж";
        SubstGlobal(s, " ~ ", " ");
        UNIT_ASSERT_EQUAL(s, TString("Москва Париж"));
    }

    Y_UNIT_TEST(TestSubstGlobalOldRet) {
        const TString s1 = "aaa";
        const TString s2 = SubstGlobalCopy(s1, "a", "bb");
        UNIT_ASSERT_EQUAL(s2, TString("bbbbbb"));

        const TString s3 = "aaa";
        const TString s4 = SubstGlobalCopy(s3, "a", "b");
        UNIT_ASSERT_EQUAL(s4, TString("bbb"));

        const TString s5 = "aaa";
        const TString s6 = SubstGlobalCopy(s5, "a", "");
        UNIT_ASSERT_EQUAL(s6, TString(""));

        const TString s7 = "abcdefbcbcdfb";
        const TString s8 = SubstGlobalCopy(s7, "bc", "bbc", 2);
        UNIT_ASSERT_EQUAL(s8, TString("abcdefbbcbbcdfb"));

        const TString s9 = "Москва ~ Париж";
        const TString s10 = SubstGlobalCopy(s9, " ~ ", " ");
        UNIT_ASSERT_EQUAL(s10, TString("Москва Париж"));
    }

    Y_UNIT_TEST(TestSubstCharGlobal) {
        TUtf16String w = u"abcdabcd";
        SubstGlobal(w, wchar16('b'), wchar16('B'), 3);
        UNIT_ASSERT_EQUAL(w, u"abcdaBcd");

        TString s = "aaa";
        SubstGlobal(s, 'a', 'b', 1);
        UNIT_ASSERT_EQUAL(s, TString("abb"));
    }

    Y_UNIT_TEST(TestSubstCharGlobalRet) {
        const TUtf16String w1 = u"abcdabcd";
        const TUtf16String w2 = SubstGlobalCopy(w1, wchar16('b'), wchar16('B'), 3);
        UNIT_ASSERT_EQUAL(w2, u"abcdaBcd");

        const TString s1 = "aaa";
        const TString s2 = SubstGlobalCopy(s1, 'a', 'b', 1);
        UNIT_ASSERT_EQUAL(s2, TString("abb"));
    }

    Y_UNIT_TEST(TestSubstStdString) {
        std::string s = "aaa";
        SubstGlobal(s, "a", "b", 1);
        UNIT_ASSERT_EQUAL(s, "abb");
    }

    Y_UNIT_TEST(TestSubstStdStringRet) {
        const std::string s1 = "aaa";
        const std::string s2 = SubstGlobalCopy(s1, "a", "b", 1);
        UNIT_ASSERT_EQUAL(s2, "abb");
    }

    Y_UNIT_TEST(TestSubstGlobalChar) {
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
} // Y_UNIT_TEST_SUITE(TStringSubst)
