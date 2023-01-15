#include <library/cpp/unittest/registar.h>
#include <library/cpp/token/serialization/serialize_char_span.h>
#include <library/cpp/token/serialization/protos/char_span.pb.h>
#include <library/cpp/token/token_structure.h>

Y_UNIT_TEST_SUITE(CharSpanSaveTest) {
    TCharSpan GetCharSpan() {
        TCharSpan sp;
        sp.Type = TOKEN_NUMBER;
        sp.TokenDelim = TOKDELIM_PLUS;
        sp.Pos = 4;
        sp.Len = 5;
        sp.PrefixLen = 6;
        sp.SuffixLen = 7;
        return sp;
    }

    Y_UNIT_TEST(TestProto) {
        TCharSpan sp0 = GetCharSpan();
        TCharSpan sp1;
        UNIT_ASSERT_UNEQUAL(sp0, sp1);
        {
            NProto::TCharSpan psp;
            SerializeCharSpan(sp0, psp);
            DeserializeCharSpan(sp1, psp);
        }
        UNIT_ASSERT_EQUAL(sp0, sp1);
    }
}
