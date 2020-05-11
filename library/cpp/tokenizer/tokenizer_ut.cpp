#include <library/charset/wide.h>
#include <library/cpp/unittest/registar.h>
#include <util/stream/output.h>
#include <util/string/escape.h>
#include "tokenizer.h"
#include "nlpparser.h"

static TString ReplaceControlCharacters(const char* p);

class TTokenizerTest: public TTestBase {
    UNIT_TEST_SUITE(TTokenizerTest);
    UNIT_TEST(TestVeryLongMultitoken);
    UNIT_TEST(TestSentenceBreak);
    UNIT_TEST(TestParagraphBreak);
    UNIT_TEST(TestPositions);
    UNIT_TEST(TestHyphenations);
    UNIT_TEST(TestPrefixes);
    UNIT_TEST(TestSuffixes);
    UNIT_TEST(TestCompositeMultitokens);
    UNIT_TEST(TestPercentEncodedText);
    UNIT_TEST(TestOrigLens);
    UNIT_TEST(TestSurrogatePairs);
    UNIT_TEST(TestUTF8SurrogatePairs);
    UNIT_TEST(TestIdeographs);
    UNIT_TEST(TestAccents);
    UNIT_TEST(TestEmptyString);
    UNIT_TEST(TestWordBreaks);
    UNIT_TEST(TestTwitterUserNames);
    UNIT_TEST(TestCharClasses);
    UNIT_TEST(TestReversible);
    UNIT_TEST_SUITE_END();

public:
    void TestVeryLongMultitoken();
    void TestSentenceBreak();
    void TestParagraphBreak();
    void TestPositions();
    void TestHyphenations();
    void TestPrefixes();
    void TestSuffixes();
    void TestCompositeMultitokens();
    void TestPercentEncodedText();
    void TestOrigLens();
    void TestSurrogatePairs();
    void TestUTF8SurrogatePairs();
    void TestIdeographs();
    void TestAccents();
    void TestEmptyString();
    void TestWordBreaks();
    void TestTwitterUserNames();
    void TestCharClasses();
    void TestReversible();

private:
    void TestCase(const wchar16* s1, const char* s2, bool backwardCompatible = true, bool spacePreserve = false, bool urlDecode = true, const TString* origLensStr = nullptr) {
        TestCase(s1, TCharTraits<wchar16>::GetLength(s1), s2, backwardCompatible, spacePreserve, urlDecode, origLensStr);
    }
    void TestCase(const wchar16* s1, size_t n1, const char* s2, bool backwardCompatible = true, bool spacePreserve = false, bool urlDecode = true, const TString* origLensStr = nullptr);
};

UNIT_TEST_SUITE_REGISTRATION(TTokenizerTest);

namespace {
    class TTestTokenHandler: public ITokenHandler {
        TString OutputText;
        TStringOutput OutputStream;
        TString OrigLensStr;

    public:
        TTestTokenHandler()
            : OutputStream(OutputText)
        {
        }
        void OnToken(const TWideToken& token, size_t origleng, NLP_TYPE type) override {
            //            Cout << "TOK: '" << TUtf16String(token.Token, token.Leng) << "'" << Endl;
            OrigLensStr += ToString(origleng) + " ";
            if (type == NLP_WORD || type == NLP_INTEGER || type == NLP_MARK || type == NLP_FLOAT) {
                UNIT_ASSERT(TCharTraits<wchar16>::Find(token.Token, 0, token.Leng) == nullptr);
                const size_t n = token.SubTokens.size();
                if (n > 1)
                    OutputStream << '[';
                else {
                    // assert as in portionbuilder.h, but added a line with NLP_WORD:
                    // TPortionBuilder::StoreMultiToken() { .. if (tok.SubTokens.size() == 1) .. }
                    Y_ASSERT(n == 1 && ((type == NLP_WORD && token.SubTokens[0].Type == TOKEN_WORD) ||
                                        (type == NLP_INTEGER && token.SubTokens[0].Type == TOKEN_NUMBER) ||
                                        (type == NLP_MARK && token.SubTokens[0].Type == TOKEN_MARK) ||
                                        (type == NLP_FLOAT && token.SubTokens[0].Type == TOKEN_FLOAT)));
                }
                for (size_t i = 0; i < n; ++i) {
                    const TCharSpan& s = token.SubTokens[i];
                    OutputStream << '[' << TUtf16String(token.Token + s.Pos - s.PrefixLen, s.Len + s.SuffixLen + s.PrefixLen) << ']';
                    switch (s.Type) {
                        case TOKEN_WORD:
                            OutputStream << "W";
                            break;
                        case TOKEN_NUMBER:
                            OutputStream << "N";
                            break;
                        case TOKEN_MARK:
                            OutputStream << "M";
                            break;
                        case TOKEN_FLOAT:
                            OutputStream << "F";
                            break;
                        default:
                            OutputStream << "?";
                            break;
                    }
                    if (s.TokenDelim != TOKDELIM_NULL)
                        OutputStream << GetTokenDelimChar(s.TokenDelim);
                }
                if (n > 1)
                    OutputStream << ']';
            } else if (type == NLP_SENTBREAK || type == NLP_MISCTEXT) {
                const wchar16* p = token.Token;
                const wchar16* const e = token.Token + token.Leng;
                while (p != e) {
                    if (*p)
                        OutputStream << TUtf16String(p, 1);
                    else
                        OutputStream << ' ';
                    ++p;
                }
                if (type == NLP_SENTBREAK)
                    OutputStream << "<S>";
            } else if (type == NLP_PARABREAK) {
                UNIT_ASSERT(TCharTraits<wchar16>::Find(token.Token, 0, token.Leng) == nullptr);
                OutputStream << TUtf16String(token.Token, token.Leng) << "<P>";
            } else if (type == NLP_END) {
                UNIT_ASSERT(token.Leng == 1 && *token.Token == 0);
                OutputStream << " ";
            } else
                UNIT_FAIL("invalid NLP type");
        }
        const TString& GetOutputText() const {
            return OutputText;
        }
        const TString& GetOrigLensStr() const {
            return OrigLensStr;
        }
    };

    class TJoinAllTokenHandler: public ITokenHandler {
        TString OutputText;
        TStringOutput OutputStream = TStringOutput(OutputText);
    public:
        void OnToken(const TWideToken& token, size_t, NLP_TYPE) override {
            OutputStream << token.Text();
        }
        const TString& GetResult() {
            OutputStream.Flush();
            return OutputText;
        }
        void Reset() {
            OutputStream.Flush();
            OutputText = "";
        }
    };
}

void TTokenizerTest::TestCase(const wchar16* s1, size_t n1, const char* s2, bool backwardCompatible, bool spacePreserve, bool urlDecode, const TString* origLensStr) {
    TTestTokenHandler handler;
    TNlpTokenizer tokenizer(handler, backwardCompatible);
    TTokenizerOptions opts { spacePreserve, TLangMask(), urlDecode };
    tokenizer.Tokenize(s1, n1, opts);
    //    Cout << "OUTPUT: " << EscapeC(handler.GetOutputText()) << Endl;
    UNIT_ASSERT_STRINGS_EQUAL(ReplaceControlCharacters(handler.GetOutputText().c_str()).c_str(), ReplaceControlCharacters(s2).c_str());
    if (origLensStr) {
        UNIT_ASSERT_STRINGS_EQUAL(handler.GetOrigLensStr(), *origLensStr);
    }
}

void TTokenizerTest::TestVeryLongMultitoken() {
    // TODO: to add test for hyphens (soft/ordinary) on the boundary of tokens: s1+s2+...+s63+HYPHEN+s64+...
    {
        const TUtf16String text(
            u"01+02+03+04+05+06+07+08+09+10+11+12+13+14+15+16+17+18+19+20+21+22+23+24+25+26+27+28+29+30+"
            u"31+32+33+34+35+36+37+38+39+40+41+42+43+44+45+46+47+48+49+50+51+52+53+54+55+56+57+58+59+60+"
            u"61+62+63+64+65/###1");
        TestCase(text.c_str(), "[[01]N+[02]N+[03]N+[04]N+[05]N+[06]N+[07]N+[08]N+[09]N+[10]N+[11]N+[12]N+[13]N+[14]N+[15]N+[16]N+[17]N+[18]N+[19]N+[20]N+[21]N+[22]N+[23]N+[24]N+[25]N+[26]N+[27]N+[28]N+[29]N+[30]N+[31]N+[32]N+[33]N+[34]N+[35]N+[36]N+[37]N+[38]N+[39]N+[40]N+[41]N+[42]N+[43]N+[44]N+[45]N+[46]N+[47]N+[48]N+[49]N+[50]N+[51]N+[52]N+[53]N+[54]N+[55]N+[56]N+[57]N+[58]N+[59]N+[60]N+[61]N+[62+]N][[63]N+[64]N+[65]N]/##[#1]N", false);
    }
    {
        const TUtf16String text(
            u"01@02@03@04@05@06@07@08@09@10@11@12@13@14@15@16@17@18@19@20@21@22@23@24@25@26@27@28@29@30@"
            u"31@32@33@34@35@36@37@38@39@40@41@42@43@44@45@46@47@48@49@50@51@52@53@54@55@56@57@58@59@60@"
            u"61@62@63@64@65/###1");
        TestCase(text.c_str(), "[[01]N@[02]N@[03]N@[04]N@[05]N@[06]N@[07]N@[08]N@[09]N@[10]N@[11]N@[12]N@[13]N@[14]N@[15]N@[16]N@[17]N@[18]N@[19]N@[20]N@[21]N@[22]N@[23]N@[24]N@[25]N@[26]N@[27]N@[28]N@[29]N@[30]N@[31]N@[32]N@[33]N@[34]N@[35]N@[36]N@[37]N@[38]N@[39]N@[40]N@[41]N@[42]N@[43]N@[44]N@[45]N@[46]N@[47]N@[48]N@[49]N@[50]N@[51]N@[52]N@[53]N@[54]N@[55]N@[56]N@[57]N@[58]N@[59]N@[60]N@[61]N@[62]N][[@63]N@[64]N@[65]N]/##[#1]N", false);
    }
    {
        const TUtf16String text(u"H4sIAAAAAAAEAO29B2AcSZYlJi9tynt/SvVK1+B0oQiAYBMk2JBAEOzBiM3mkuwdaUcjKasqgcplVmVdZhZAzO2dvPfee++999577733ujudTif33/8/XGZkAWz2zkrayZ4hgKrIHz9+fB8/Iv7vX/M3nf2mv26yfW/30wcP7t97cP83nf02v9a5/efX+nX431+b//013b/48tc5/01/m1/rF/yGv//vetbmi5NqvWx/rd/st0nxzW9w/pv+1r/mrzGjpr/1r/3r7X766d69g9/t9MHu/t6D3Sfbpw8f7m7vnxzsbB/s7T/Z3jvevXdv5+TZ03v7D+mVX+s3/a1/nfd55dfcpZd+7d/0t/61ui/9/njp98dLvz9e+v39l2Y8iBiW946PT47v37u3/fTTJ0+29/ef3t8+OL53sr3/7ODBvd0nu0+e7p52sLzNK7/mXgdL89Lvj5d+f7z0++Ol399/Cd3EsTx4trd7srtDvR0cf0qEOXm4fXz6dG97/9N7+w8enj7bOd7d62B5m1d+zXsdLM1Lvz9e+v3x0u+Pl35//6UZM0cMy/t7z57tPX1I5Lh/TB2dPru//fDB3gn9uX//2b1Pd3ae7ex2sLzNK7/mfgdL89Lvj5d+f7z0++Ol399/acaMHMNy58H9vSc7p/e3Tw/297f3nzx4sv3w6cMH26efnnz65MGDp6e7Dx50sLzNK7/m/Q6W5qXfHy/9/njp98dLv7//0uzX+nUHsNy/9+Dk2cHuPZLT04fEWg92qcunD7fvn+wc33/6gGbpwU4Hy9u88mt+2sHSvPT746XfHy/9/njp9/dfmv1av94QlicPTx/c3z3Yvv/s02OaPiLMk/v3d7Z39+7v7u3tPTi5d3Cvi+UtXvk1H3Sx1Jd+f7z0++Ol3x8v/f7+S7Nf69cfwPLJw6f7T08e7mw/e7C/s71/f2dn++G9g5Ptk5N7O0/2SPJOnnU10W1e+TUPOlial35/vPT746XfHy/9/v5Ls1/rNxjSRHtPTp9+SpO2vwNR2N3b3z44uXe6vX98n7TY/f39/Xv7HSxv88qv+bCDpXnp98dLvz9e+v3x0u/vvzT7tX5sCMsHZDL2iL8+fXpC07fz6afbBxDY4wcPHux9unf87OT+0y6Wt3jl19rd6aKpb/3+eOv3x1u/P976/f23Zr9WMoTmwenx02enO9vEww+2959+urd98On+p9ukaR9++ow+O97tEfMWr/xau13rY976/fHW74+3fn+89fv7b81+rd9wUMqfPT2+t7e3TUaKFMvus6fbBw9JLO4/JLG7T1bh3sOuxrzNK7/Wbtf8mLd+f7z1++Ot3x9v/f7+W7Nf6zcaQPMBKdrTB/eoz2fPiDQ7Tx9sHxzsPdneefLs4P7O/dMnp3td83ObV36t3a79MW/9/njr98dbvz/e+v39t2a/1m9s0DyYVmVVP0p/fIefwzZ/127P8mlVZ21RLR+ly2qZH6bn1bLdvsqLi3n7KJ1U5eywwyy79x8++PTJ0+3jTw/IXt472N9+sn9AenTn9GTndOf4dO/hzgd0FTDZLbr6tXb3P6C3gDm1t98fvf3+6O33R2+/v9/b7Nf6TQZm/dNne59+ekJmjhwo0i5Pn5AH8fBkb/s+OQh7u7v7T49PnnRm/Tav/Fq7XUtp3vr98dbvj7d+f7z1+/tvzX6t33QAzd0Hnz472Lt/un2PLCoZ54f728c7D3a2T58enDzduXf/0+PjTzto3uaVX2u3ayrNW78/3vr98dbvj7d+f/8toPbb/Fq/y69hHONf52s6xr/mrrissfdv53+KMxl7/3aeobh5sfdv57OJAxZ7/3belLhGsfdv5+eI0xJ9/1YeiLgTsfdv5xuIoY/O362stpjg6Pu3s6diHKMAbmfpxGzFZ+BWNkgMSgzALa2DqHoG8AEa8X1V/a8Fi/2bDGB+Sw0n6ioG4Ja6Z/b/APpmlm2oDwAA");
        TestCase(text.c_str(),
                 "[[H]W[4]N[sIAAAAAAAEAO]W[29]N[B]W[2]N[AcSZYlJi]W[9]N[tynt]W/[SvVK]W[1]N+[B]W[0]N[oQiAYBMk]W[2]N[JBAEOzBiM]W[3]N[mkuwdaUcjKasqgcplVmVdZhZAzO]W[2]N[dvPfee+]W+[999577733]N[ujudTif]W[33]N/[8]N/[XGZkAWz]W[2]N[zkrayZ]W[4]N[hgKrIHz]W[9]N+[fB]W[8]N/[Iv]W[7]N[vX]W/[M]W[3]N[nf]W[2]N[mv]W[26]N[yfW]W/[30]N[wcP]W[7]N[t]W[97]N[cP]W[83]N[nf]W[02]N[v]W[9]N[a]W[5]N/[efX]W+[nX]W[431]N+[b]W]//[[013]N[b]W/[48]N[tc]W[5]N/[01]N/[m]W[1]N/[rF]W/[yGv]W]//[[vetbmi]W[5]N[NqvWx]W/[rd]W/[st]W[0]N[nxzW]W[9]N[w]W/[pv]W+[1]N[r]W/[mrzGjpr]W/[1]N[r]W/[3]N[r]W[7]N[X]W[766]N[d]W[69]N[g]W[9]N/[t]W[9]N[MHu]W/[t]W[6]N[D]W[3]N[Sfbpw]W[8]N[f]W[7]N[m]W[7]N[vnxzsbB]W/[s]W[7]N[T]W/[Z]W[3]N[jvevXdv]W[5]N+[TZ]W[03]N[v]W[7]N[D]W+[mVX]W+[s]W[3]N/[a]W[1]N/[nfd]W[55]N[dfcpZd]W+[7]N[d]W/[0]N[t]W]/[[61]N[ui]W/[9]N/[njp]W[98]N[dLvz]W[9]N[e]W+[v]W[39]N[l]W[2]N[Y]W[8]N[iBiW]W[946]N[PT]W[47]N[v]W[37]N[u]W[3]N/[fTTJ]W[0]N+[29]N/[ef]W[3]N[t]W[8]N+[OL]W[53]N[sr]W[3]N/[7]N[ODBvd]W[0]N[nu]W[0]N+[e]W[7]N[p]W[52]N[sLzNK]W[7]N/[mXgdL]W[89]N[Lvj]W[5]N[d]W+[f]W[7]N[z]W[0+]N+[Ol]W[399]N/[Cd]W[3]N[EsTx]W[4]N[trd]W[7]N[srtDvR]W][[0]N[cf]W[0]N[qEOXm]W[4]N[fXz]W[6]N[dG]W[97]N/[9]N[N]W[7]N+[w]W[8]N[enj]W[7]N[bOd]W[7]N[d]W[62]N[B]W[5]N[m]W[1]N[d]W+[zXsdLM]W[1]N[Lvz]W[9]N[e]W+[v]W[3]N[x]W[0]N[u]W+[Pl]W[35]N]//[[6]N[UZM]W[0]N[cMy]W/[t]W[7]N[z]W[57]N[tPX]W[1]N[I]W[5]N[Lh]W/[TB]W[2]N[dPru]W]//[[fDB]W[3]N[gn]W[9]N[uX]W]//[[2]N[b]W[1]N[Pd]W[3]N[ae]W[7]N[ex]W[2]N[sLzNK]W[7]N/[mfgdL]W[89]N[Lvj]W[5]N[d]W+[f]W[7]N[z]W[0+]N+[Ol]W[399]N/[acaMHMNy]W[58]N[H]W[9]N[vSc]W[7]N[p]W/[e]W[3]N[Tw]W/[297]N[f]W[3]N[nzx]W[4]N[sv]W[3]N[w]W[6]N[cMH]W[26]N[efnnz]W[65]N[MGDp]W[6]N[e]W[7]N[Dx]W[50]N[sLzNK]W[7]N/[m]W/[Q]W[6]N[W]W[5]N[qXfHy]W/[9]N/[njp]W[98]N][[dLv]W[7]N]//[[0]N[uzX]W+[nUHsNy]W/[9]N+[Dk]W[2]N[cHuPZLT]W[04]N[fEWg]W[92]N[qcunD]W[7]N[fvn]W+[wc]W[33]N/[6]N[gGbpwU]W[4]N[Hy]W[9]N[u]W[88]N[mt]W+[2]N[sHSvPT]W[746]N[XfHy]W/[9]N/[njp]W[9]N/[dfmv]W[1]N[av]W[94]N[QlicPTx]W/[c]W[3]N[z]W[3]N[Yvv]W/[s]W[02]N[OaPiLMk]W/[v]W[3]N[d]W[7]N[Z]W[39]N+[7]N[v]W[7]N[u]W[3]N[tPTi]W[5]N[d]W[3]N[Cvi]W+[UtXvk]W[1]N[H]W][[3]N[Sx]W[1]N[Jd]W+[f]W[7]N[z]W[0+]N+[Ol]W[3]N[x]W[8]N[v]W/[f]W[7]N+[S]W[7]N[Nf]W[69]N[cfwPLJw]W[6]N[f]W[7]N[T]W[08]N[e]W[7]N[mw]W/[e]W[7]N[C]W/[s]W[71]N/[f]W[2]N[dn+]W+[G]W[9]N[g]W[5]N[Ptk]W[5]N[N]W[7]N[O]W[0]N/[2]N[SPJOnnU]W[10]N[W]W[1]N[e]W+[TUPOlial]W[35]N/[vPT]W[746]N[XfHy]W/[9]N/[v]W[5]N[Ls]W[1]N]/[[rNxjSRHtPTp]W[9]N+[SpO]W[2]N[vwNR]W[2]N[N]W[3]N[b]W[3]N[z]W[44]N[uXe]W[6]N[vX]W[98]N[n]W[7]N[TY]W/[f]W[39]N/[Xv]W[7]N[HSxv]W[88]N[qv]W+[bCDpXnp]W[98]N[dLvz]W[9]N[e]W+[v]W[3]N[x]W[0]N[u]W/[vvzT]W[7]N[tX]W[5]N[sCMsHZDL]W[2]N[iL]W[8]N+[fXpC]W[07]N[fz]W[6]N[afbBxDY]W[4]N[wcPHux]W[9]N[unf]W[87]N[OT]W+[0]N[y]W[6]N[Wt]W[3]N[jl]W[19]N][[rd]W[6]N[aKpb]W/[3]N+[eOv]W[3]N[x]W[1]N[u]W/[P]W[976]N/[f]W[23]N[Zr]W[9]N[WMoTmwenx]W[02]N[enO]W[9]N[vEww]W+[2959]N+[urd]W[98]N[On]W+[p]W[9]N[ukaR]W[9+]N+[ow]W+[O]W[97]N[tEfMWr]W/[xau]W[13]N[rY]W[976]N/[fHW]W[74]N+[3]N[fn]W+[89]N[fv]W[7]N[b]W[81]N+[rd]W[9]N[wUMqfPT]W[2]N+[t]W[7]N[e]W[3]N[TUaKFMvus]W[6]N[fbBw]W[9]N[JLO]W[4]N/[JLG]W[7]N[T]W][[1]N[bh]W[3]N[sOuxrzNK]W[7]N/[Wbtf]W[8]N[mLd]W+[f]W[7]N[z]W[1+]N+[Ot]W[3]N[x]W[9]N[v]W/[f]W[7]N+[W]W[7]N[Nf]W[6]N[zcaQPMBKdrTB]W/[eoz]W[2]N[fPiDQ]W[7]N[Tx]W[9]N[sHxzsPdneefLs]W[4]N[P]W[7]N[O]W/[dMnp]W[3]N[td]W[83]N[ObV]W[36]N[t]W[3]N[a]W[79]N[MW]W/[9]N/[njr]W[98]N[dbvz]W/[e]W+[v]W[39]N[t]W[2]N[a]W/[1]N[m]W[9]N[s]W[0]N[DyYVmVVP]W][[0]N[p]W/[fIefwzZ]W/[127]N[P]W[8]N[mlVZ]W[21]N[RLR]W+[ly]W[2]N[qZH]W[6]N[bn]W[1]N[bLdvsqLi]W[3]N[n]W[7]N[KJ]W[1]N[U]W[5]N[eywwyy]W[79]N[x]W[8+]N+[PTJ]W[0]N+[3]N[jTw]W/[IXt]W[472]N[N]W[9]N+[sn]W[9]N[AenTn]W[9]N[GTndOf]W[4]N[dO]W/[hzgd]W[0]N[FTDZLbr]W[6]N[tXb]W[3]N[P]W[6]N[C]W[3]N[gDm]W[1]N[t]W[98]N[fvf]W[3]N+[6]N[O]W[33]N[R]W][[2+]N/[v]W[9]N/[b]W[7]N[Nf]W[6]N[TQZm]W/[dNne]W[59]N+[ekJmjhwo]W[0]N[i]W[5]N[Pn]W[5]N[AH]W[8]N[fBkb]W/[s]W+[OQh]W[7]N[u]W[7]N[v]W[7]N[T]W[49]N[PnnRm]W/[Tav]W/[Fq]W[7]N[XUtp]W[3]N[vr]W[98]N[dbvj]W[7]N[d]W+[f]W[7]N[z]W[1+]N/[tvzX]W[6]N[t]W[33]N[QAzd]W[0]N[Hnz]W[472]N[Lt]W/[un]W[2]N[PLCoZ]W[54]N[f]W[728]N[c]W[7]N[D]W[3]N][[a]W[2]N[T]W[58]N[enDzduXf]W/[0]N+[PjTzto]W[3]N[uaVX]W[2]N[u]W[3]N[ayrNW]W[78]N/[3]N[vr]W[98]N[dbvj]W[7]N[d]W+[f]W/[8]N[toPbb]W/[Fq]W/[y]W[69]N[hHONf]W[52]N[s]W[6]N[xr]W/[mrrissfdv]W[53]N+[KMxl]W[7]N/[3]N[aeobh]W[5]N[sfdv]W[57]N[OJAxZ]W[7]N/[3]N[belLhGsfdv]W[5]N+[eI]W[0]N[xJ]W[9]N/[1]N[YeiLgTsfdv]W[5]N[xuIoY]W/[O]W[362]N[stpjg]W[6]N[Pu]W[3]N[s]W[6]N[diHKMAbmfpxGzFZ+]W][[BWNkgMSgzALa]W[2]N[DqHoG]W[8]N[AEa]W[8]N[X]W[1]N[V]W/[a]W[8]N[Fi]W/[2]N[bDGB]W+[Sw]W[0]N[n]W[6]N[ioG]W[4]N[Ja]W[6]N[Z]W/[b]W/[APpmlm]W[2]N[oDwAA]W]", false);
    }
    {
        const TUtf16String text(u"MSDTRTETDSFGPLEVPTDKYWGAQTQRSIMNFPIGWEKQPVAIVRALGVIKKACAMANKASGKMEDRIADAVIAAAGEVIEGKFDDNFPLVVWQTGSGTQSNMNSNEVIANRAIEMLGGVIGSKDPVHPNDHCNMGQSSNDTFPTAMHIATAMSVRDVLLPGLEKLAKGLENKSEEFKDIIKIGRTHTQDATPLTLGQEFGGYAHQIRQGIARVELAMPGIYELAQGGTAVGTGLNTQKGWSEEVAANMAEITDLPFVTAPNKFEALAAHDAMVFLSGALATIAGSCYKIASDIRFLGSGPRSGLGELILPENEPGSSIMPGKVNPTQAEALTQVAAHVMGNDAAIKFAGSQGHFELNVYNPMMSYNLLQSIQLLGDATDSFTERMLNGIQANEPRIDKLMKESLMLVTALAPTIGYDNATKVAKTAHKNGTTLKEEAIALGFVDEATFDAVVRPEQMIGP-\n240139412");
        TestCase(text.c_str(), "[MSDTRTETDSFGPLEVPTDKYWGAQTQRSIMNFPIGWEKQPVAIVRALGVIKKACAMANKASGKMEDRIADAVIAAAGEVIEGKFDDNFPLVVWQTGSGTQSNMNSNEVIANRAIEMLGGVIGSKDPVHPNDHCNMGQSSNDTFPTAMHIATAMSVRDVLLPGLEKLAKGLENKSEEFKDIIKIGRTHTQDATPLTLGQEFGGYAHQIRQGIARVELAMPGIYELAQGGTAVGTGLNTQKGWSEEVAANMAEITD]W", false, true);
    }
    {
        TUtf16String text(u"a-b++"
                          "ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc-ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        text.replace(1, 1, 1, 0x00b7); // U+00B7 MIDDLE DOT, it must be replaced with '-'
        TestCase(text.c_str(), "[[a]W-[b+]W]+["
                               "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc]W");
    }
    {
        TUtf16String text(u"a-b++"
                          u"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc++ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd");
        text.replace(1, 1, 1, 0x00b7); // U+00B7 MIDDLE DOT, it must be replaced with '-'
        TestCase(text.c_str(), "[[a]W-[b+]W]+["
                               "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc]W");
    }
    {
        TUtf16String text(u"a-b++"
                          u"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc++");
        text.replace(1, 1, 1, 0x00b7); // U+00B7 MIDDLE DOT, it must be replaced with '-'
        TestCase(text.c_str(), "[[a]W-[b+]W]+["
                               "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc]W");
    }
    {
        TUtf16String text(u"a-b++"
                          u"cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc++");
        text.replace(1, 1, 1, 0x00b7); // U+00B7 MIDDLE DOT, it must be replaced with '-'
        TestCase(text.c_str(), "[[a]W-[b+]W]+["
                               "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc]W");
    }
    {
        TUtf16String text(u"aaa-\nbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb++");
        TestCase(text.c_str(), "[aaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb++]W", false, true);
    }
    {
        TUtf16String text(u"aaa-bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb++");
        text.replace(3, 1, 1, 0x00AD); // SOFT HYPHEN
        TestCase(text.c_str(), "[aaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb++]W");
    }
    {
        TUtf16String text(u"a-b++c++ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd-eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee");
        text.replace(1, 1, 1, 0x00b7); // U+00B7 MIDDLE DOT, it must be replaced with '-'
        TestCase(text.c_str(), "[[a]W-[b+]W]+[c+]W+[ddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd]W");
    }
}

void TTokenizerTest::TestSentenceBreak() {
    TestCase(u"Abc. Xyz.", "[Abc]W. <S>[Xyz]W.");
    TestCase(u"Abc...   Xyz.", "[Abc]W...   <S>[Xyz]W.");
    TestCase(u"Abc...)))\r\nXyz.", "[Abc]W...)))\r\n<S>[Xyz]W.");
    TestCase(u"Abc... -)\r\r (Xyz.", "[Abc]W... -)\r\r <S>([Xyz]W.");
    TestCase(u"Abc...))) \n\n(((Xyz.", "[Abc]W...))) \n\n((<S>([Xyz]W.");
    TestCase(u"Abc. - Xyz.", "[Abc]W. <S>- [Xyz]W.");
    TestCase(u"Abc. ((-\nXyz.", "[Abc]W. ((<S>-\n[Xyz]W.");
    TUtf16String text(
        u"Copyright ° 2006г. ООО Зелёный Ковёр тел: (495) 797-55-58; (499) 133-76-65; (499) 133-67-28.\n"
        u"Кошмар на улице Вязов-5. Дитя снов.");
    TestCase(text.c_str(), "[Copyright]W [°]W [[2006]N[г]W]. <S>"
                           "[ООО]W [Зелёный]W [Ковёр]W [тел]W: ([495]N) [[797]N-[55]N-[58]N]; ([499]N) [[133]N-[76]N-[65]N]; ([499]N) [[133]N-[67]N-[28]N].\n<S>"
                           "[Кошмар]W [на]W [улице]W [[Вязов]W-[5]N]. <S>[Дитя]W [снов]W.",
             false);
    TestCase(text.c_str(), "[Copyright]W [°]W [2006г]M. <S>"
                           "[ООО]W [Зелёный]W [Ковёр]W [тел]W: ([495]N) [797]N-[55]N-[58]N; ([499]N) [133]N-[76]N-[65]N; ([499]N) [133]N-[67]N-[28]N.\n<S>"
                           "[Кошмар]W [на]W [улице]W [Вязов]W-[5]N. [Дитя]W [снов]W.");

    // prefix must be part of the first word in the second sentence:
    TestCase(u"First sentence. #Second #sentence.", "[First]W [sentence]W. <S>[#Second]W [#sentence]W.", false);
    TestCase(u"For ex. № 1 is not a new sentence.", "[For]W [ex]W. [№]W [1]N [is]W [not]W [a]W [new]W [sentence]W.", false);
    TestCase(u"Third sentence./-+\\;*%{|}^()! Fourth sentence.", "[Third]W [sentence]W./-+\\;*%{|}^()! <S>[Fourth]W [sentence]W.", false);

    // before removing @change_sentence_break, SentenceBreak pointer was after "-)"
    // now it is before "-)" because the condition of the ragel machine was relaxed to be able breaking sentences more flexibly
    TestCase(u"Sentence one. -) Sentence two.", "[Sentence]W [one]W. <S>-) [Sentence]W [two]W.", false);
}

void TTokenizerTest::TestParagraphBreak() {
    TestCase(u"abc.\n\nxyz.", "[abc]W.\n\n<P>[xyz]W.", true, true);
    TestCase(u"Abc.\n\nXyz.", "[Abc]W.\n\n<S>[Xyz]W.", true, true);
    TestCase(u"abc.\n$%&\nxyz.", "[abc]W.\n$%&\n<P>[xyz]W.", true, true);
    TestCase(u"Abc.\n$%&\nXyz.", "[Abc]W.\n$%&\n<P>[Xyz]W.", true, true);
}

void TTokenizerTest::TestPositions() {
    TestCase(u"a.bcdef-ghijk", "[a]W.[[bcdef]W-[ghijk]W]");
}

// Example of a HTML-document with hyphenations:
// <html><head>
// <meta http-equiv="content-type" content="text/html;charset=utf-8">
// <title>a test document</title>
// </head><body>
//
// This is not a hyphe-
// nation.
//
// <pre>
// This is a hyphe-
// nation.
// </pre>
//
// This is NOT a hyphe-<br>nation.
//
// </body></html>
void TTokenizerTest::TestHyphenations() {
    TestCase(u"This is a hyphe-\nnation.", "[This]W [is]W [a]W [hyphenation]W.", false, true);                   // SpacePreserve == true
    TestCase(u"This is NOT a hyphe-\nnation.", "[This]W [is]W [NOT]W [a]W [hyphe]W-\n[nation]W.", false, false); // SpacePreserve == false

    {
        const wchar16 text[] = {'a', 0x00AD, 'b', 0x00AD, 'c', 0x00AD, 'd', 0};
        TestCase(text, "[abcd]W", false);
    }

    // yc_8F = 0x00AD; soft hyphen
    {
        const wchar16 text[] = {'a', 'a', '-', 'b', 0x00AD, 'b', '-', 'c', 'c', 0};
        TestCase(text, "[[aa]W-[bb]W-[cc]W]");
    }
    {
        const wchar16 text[] = {'a', 'a', '-', 'b', 0x00AD, 'b', '-', 'c', 'c', 0};
        TestCase(text, "[[aa]W-[bb]W-[cc]W]");
    }
    {
        const wchar16 text[] = {'e', 'x', 'a', 0x00AD, 0x0301, 'm', 'p', 'l', 'e', 0};
        TestCase(text, "[exa\xCC\x81mple]W");
    }
    {
        TUtf16String text(7, 'a');
        text.append(0x00AD);
        text.append(38, 'b');
        text.append(2, '1');
        text.append(114, 'c');
        text.append(3, '2');
        text.append(100, 'd');
        TestCase(text.c_str(), "[aaaaaaabbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb11cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc222]M");
    }
    // hyphenation
    TestCase(u"aa-b-\nb-cc", "[[aa]W-[bb]W-[cc]W]", false, true);
    TestCase(u"a1-b-\n2-c3", "[a1]M-[b]W-[2]N-[c3]M", true, true);
    TestCase(u"aa b-\nb cc", "[aa]W [bb]W [cc]W", false, true);
    TestCase(u"aa-b-\nb-cc-dd-ee-ff-gg-hh-ii-jj", "[[aa]W-[bb]W-[cc]W-[dd]W-[ee]W-[ff]W]-[[gg]W-[hh]W-[ii]W-[jj]W]", false, true);
    TestCase(u"aa-bb-cc-dd-e-\ne-ff-gg-hh-ii-jj-kk-ll-mm-nn",
             "[[aa]W-[bb]W-[cc]W-[dd]W-[ee]W-[ff]W-[gg]W-[hh]W-[ii]W]-" // maximum 9 subtokens in multitoken
             "[[jj]W-[kk]W-[ll]W-[mm]W-[nn]W]",
             false, true);
    TestCase(u"aa-bb-cc-dd-ee-f-\nf-gg-hh-ii-jj-kk-ll-mm-nn",
             "[[aa]W-[bb]W-[cc]W-[dd]W-[ee]W]-[[ff]W-[gg]W-[hh]W-[ii]W-[jj]W]-[[kk]W-[ll]W-[mm]W-[nn]W]", false, true);
    TestCase(u"aa-bb-cc-dd-ee-f-\nf-gg-hh+ii-jj-kk-ll-mm-nn",
             "[[aa]W-[bb]W-[cc]W-[dd]W-[ee]W]-[[ff]W-[gg]W-[hh+]W][[ii]W-[jj]W-[kk]W-[ll]W-[mm]W]-[nn]W", false, true);
}

void TTokenizerTest::TestPrefixes() {
    TestCase(u"#1 place $100, #2 place $50", "[#1]N [place]W [$100]N, [#2]N [place]W [$50]N", false);
    TestCase(u"#1 place $100, #2 place $50", "#[1]N [place]W $[100]N, #[2]N [place]W $[50]N", true);

    TestCase(u"8+$D$1$", "[[8]N+[$D]W][$1]N$", false);
    TestCase(u"8++$D$1$", "[[8+]N+[$D]W][$1]N$", false);
    TestCase(u"8+++$D$1$", "[[8++]N+[$D]W][$1]N$", false);

    TestCase(u"a+@b", "[[a]W+[@b]W]", false);
    TestCase(u"a++@b", "[[a+]W+[@b]W]", false);
    TestCase(u"a#@b", "[[a#]W@[b]W]", false);
    TestCase(u"a##b", "[a#]W[#b]W", false);
    TestCase(u"a++/b", "[[a++]W/[b]W]", false);
    TestCase(u"a+@@b", "[[a+]W@[@b]W]", false);

    TestCase(u"perkins+@cmu.edu", "[perkins+]W@[cmu]W.[edu]W", true);
    TestCase(u"c++@ya.ru", "[c++]W@[ya]W.[ru]W", true);

    TestCase(u"D$8+", "[D]W[$8+]N", false);
    TestCase(u"D$8+", "[D]W$[8+]N", true);

    TestCase(u"8+$D$1$", "[8+]N$[D]W$[1]N$", true);
    TestCase(u"8+#D$1$", "[8+]N#[D]W$[1]N$", true);
    TestCase(u"8+@D$1$", "[8+]N@[D]W$[1]N$", true); // for integers that are a part of multitoken suffix is removed
    TestCase(u"8++$D$1$", "[8++]N$[D]W$[1]N$", true);
    TestCase(u"8++#D$1$", "[8++]N#[D]W$[1]N$", true);
    TestCase(u"8++@D$1$", "[8++]N@[D]W$[1]N$", true); // for integers that are a part of multitoken suffix is removed
    TestCase(u"8+++$D$1$", "[8++]N+$[D]W$[1]N$", true);
    TestCase(u"8+++#D$1$", "[8++]N+#[D]W$[1]N$", true);
    TestCase(u"8+++@D$1$", "[8++]N+@[D]W$[1]N$", true);

    TestCase(u"https://www.google.com/#hl=en&sclient=psy-ab&q=topcoder+prize+%2415000&oq=topcoder+prize+$15000",
             "[https]W://[[www]W.[google]W.[com]W/[#hl]W]=[en]W&[sclient]W=[[psy]W-[ab]W]&[q]W=[[topcoder]W+[prize+]W]%[2415000]N&[oq]W=[[topcoder]W+[prize]W+[$15000]N]", false);
    TestCase(u"https://www.google.com/#hl=en&sclient=psy-ab&q=topcoder+prize+%2415000&oq=topcoder+prize+$15000",
             "[https]W://[www]W.[google]W.[com]W/#[hl]W=[en]W&[sclient]W=[[psy]W-[ab]W]&[q]W=[topcoder]W+[prize+]W%[2415000]N&[oq]W=[topcoder]W+[prize+]W$[15000]N", true);
    TestCase(u"$100+$200+/10=$120+", "[[$100]N+[$200+]N/[10]N]=[$120+]N", false);
    TestCase(u"$100+$200+/10=$120+", "$[100+]N$[200]N+/[10]N=$[120+]N", true);
    TestCase(u"$$$abc.def $$$abc.$$$def ", "$$[[$abc]W.[def]W] $$[$abc]W.$$[$def]W ", false);
    TestCase(u"abc def@#$ghi jkl#@$mno pqr", "[abc]W [def]W@#[$ghi]W [[jkl#]W@[$mno]W] [pqr]W", false);

    TestCase(u"@a_b_@c_d #e_f_#g_h", "[[@a]W_[b]W_[@c]W_[d]W] [[#e]W_[f]W_[#g]W_[h]W]", false);
    TestCase(u"@a_b_@c_d #e_f_#g_h", "@[a]W_[b]W_@[c]W_[d]W #[e]W_[f]W_#[g]W_[h]W", true);

    TestCase(u"I_have_@one_thousand_dollars_$1000_a-lot-of-money", "[[I]W_[have]W_[@one]W_[thousand]W_[dollars]W_[$1000]N_[a]W-[lot]W-[of]W-[money]W]", false);

    // yc_8F = 0x00AD; soft hyphen
    {
        const wchar16 text[] = {'a', '-', '#', 'b', 0x00AD, 'b', '-', '$', 'c', 0};
        TestCase(text, "[a]W-[#bb]W-[$c]W", false);
    }
    {
        const wchar16 text[] = {'a', '-', 'b', 0x00AD, '#', 'b', '-', '$', 'c', 0};
        TestCase(text, "[[a]W-[b]W]­[#b]W-[$c]W", false);
    }
    {
        const wchar16 text[] = {'a', '-', 'b', '#', 0x00AD, 'b', '-', '$', 'c', 0};
        TestCase(text, "[[a]W-[b#]W]­[b]W-[$c]W", false);
    }
    // accents
    {
        const wchar16 text[] = {'o', ' ', '#', 0x301, 'p', '-', 0x301, '$', 'q', ' ', 'r', 0};
        TestCase(text, "[o]W [#\xCC\x81p]W-\xCC\x81[$q]W [r]W", false);
    }
    {
        const wchar16 text[] = {'o', ' ', 0x301, '#', 'p', '-', '$', 0x301, 'q', ' ', 'r', 0};
        TestCase(text, "[o]W \xCC\x81[#p]W-[$\xCC\x81q]W [r]W", false);
    }
}

void TTokenizerTest::TestSuffixes() {
    TestCase(u"a+ b++ c#", "[a+]W [b++]W [c#]W");
    TestCase(u"a+-b++-c#", "[a+]W-[b++]W-[c#]W");
    TestCase(u"a-x+ b-y++ c-z#", "[[a]W-[x+]W] [[b]W-[y++]W] [[c]W-[z#]W]");
    TestCase(u"a1+ b2++ c3#", "[a1]M+ [b2]M++ [c3]M#");
    TestCase(u"1a+ 2b++ 3c#", "[1a]M+ [2b]M++ [3c]M#");
    TestCase(u"1+ 2++ 3#", "[1+]N [2++]N [3#]N");
    TestCase(u"1.4+ 2.5++ 3.6#", "[1.4]F+ [2.5]F++ [3.6]F#");
    TestCase(u"18+ 16+/18+", "[18+]N [16]N+/[18]N+");
}

void TTokenizerTest::TestCompositeMultitokens() {
    TestCase(u"a-b-c-d-e-f-g/h+i_j.k@l'm-n-o-p++q/r/s+t@u_v-w-x#y-z+",
             "[[a]W-[b]W-[c]W-[d]W-[e]W]-[[f]W-[g]W]/[h]W+[i]W_[j]W.[k]W@[[l]W'[m]W-[n]W-[o]W-[p+]W]+[q]W/[r]W/[s]W+[t]W@[u]W_[[v]W-[w]W-[x#]W][[y]W-[z+]W]");
    TestCase(u"a-b'c_d-e'f_j-h'i_j-k'l", "[[a]W-[b]W'[c]W_[d]W-[e]W'[f]W_[j]W-[h]W'[i]W_[j]W-[k]W'[l]W]", false);
    TestCase(u"a-b'c_d-e'f_j-h'i_j-k'l", "[[a]W-[b]W'[c]W]_[[d]W-[e]W'[f]W]_[[j]W-[h]W'[i]W]_[[j]W-[k]W'[l]W]", true);
    TestCase(u"a_1_b_2_c_3_d_4_e_5_f_6_g_7_h_8", "[[a]W_[1]N_[b]W_[2]N_[c]W_[3]N_[d]W_[4]N_[e]W_[5]N_[f]W_[6]N_[g]W_[7]N_[h]W_[8]N]", false);
    TestCase(u"a_1_b_2_c_3_d_4_e_5_f_6_g_7_h_8", "[a]W_[1]N_[b]W_[2]N_[c]W_[3]N_[d]W_[4]N_[e]W_[5]N_[f]W_[6]N_[g]W_[7]N_[h]W_[8]N", true);

    TestCase(u"1black-and-white2Mother's3+", "[[1]N[black]W-[and]W-[white]W[2]N[Mother]W'[s]W[3+]N]", false);
}

void TTokenizerTest::TestPercentEncodedText() {
    // good UTF8:
    TestCase(u"ru.wikipedia.org/wiki/%D0%A0%D0%BE%D1%81%D1%82%D0%BE%D0%B2_"
             u"%28%D1%84%D1%83%D1%82%D0%B1%D0%BE%D0%BB%D1%8C%D0%BD%D1%8B%D0%B9_%D0%BA%D0%BB%D1%83%D0%B1%29",
             "[[ru]W.[wikipedia]W.[org]W/[wiki]W/[Ростов]W]_([[футбольный]W_[клуб]W])", false);
    TestCase(u"http://ru.wikipedia.org/wiki/%D0%92%D0%B5%D1%80%D1%85%D0%BE%D0%B2%D0%BD%D1%8B%D0%B9_"
             u"%D0%A1%D0%BE%D0%B2%D0%B5%D1%82_%D0%A1%D0%A1%D0%A1%D0%A0",
             "[http]W://[[ru]W.[wikipedia]W.[org]W/[wiki]W/[Верховный]W_[Совет]W_[СССР]W]", false);
    // bad UTF8:
    TestCase(u"http://ru.wikipedia.org/wiki/%D5%E0%E1%F0%E0%F5%E0%E1%F0",
             "[http]W://[[ru]W.[wikipedia]W.[org]W/[wiki]W]/"
             "%[[D]W[5]N]%[[E]W[0]N]%[[E]W[1]N]%[[F]W[0]N]%[[E]W[0]N]%[[F]W[5]N]%[[E]W[0]N]%[[E]W[1]N]%[[F]W[0]N]",
             false);
    TestCase(u"vse-uroki.ru/tags/%F1%EA%E0%F7%E0%F2%FC+depositfiles.com+"
             u"%C2%E8%E4%E5%EE%EA%F3%F0%F1+%AB%D1%E5%EA%F0%E5%F2%FB+Web+%C4%E8%E7%E0%E9%ED%E0+%F1+%EF%EE%EC%EE%F9%FC%FE+Photoshop%BB/",
             "[[vse]W-[uroki]W.[ru]W/[tags]W]/%[[F]W[1]N]%[EA]W%[[E]W[0]N]%[[F]W[7]N]%[[E]W[0]N]%[[F]W[2]N]%[[FC]W+[depositfiles]W.[com+]W]"
             "%[[C]W[2]N]%[[E]W[8]N]%[[E]W[4]N]%[[E]W[5]N]%[EE]W%[EA]W%[[F]W[3]N]%[[F]W[0]N]%[[F]W[1+]N]%[AB]W%[[D]W[1]N]%[[E]W[5]N]%[EA]W"
             "%[[F]W[0]N]%[[E]W[5]N]%[[F]W[2]N]%[[FB]W+[Web+]W]%[[C]W[4]N]%[[E]W[8]N]%[[E]W[7]N]%[[E]W[0]N]%[[E]W[9]N]%[ED]W%[[E]W[0+]N]"
             "%[[F]W[1+]N]%[EF]W%[EE]W%[EC]W%[EE]W%[[F]W[9]N]%[FC]W%[[FE]W+[Photoshop]W]%[BB]W/",
             false);
    // double percent encoded text: no recursive conversion
    TestCase(u"a-b %61%2D%62 %25%36%31%25%32%44%25%36%32", "[[a]W-[b]W] [[a]W-[b]W] %[61]N%[[2]N[D]W]%[62]N", false);
    // disabled url decoder
    TestCase(u"%D1%82%D0%B5%D0%BA%D1%81%D1%82", "%[[D]W[1]N]%[82]N%[[D]W[0]N]%[[B]W[5]N]%[[D]W[0]N]%[BA]W%[[D]W[1]N]%[81]N%[[D]W[1]N]%[82]N",
             false, false, false);
}
void TTokenizerTest::TestOrigLens() {
    {
        const TString lens = "3 1 2 1 2 ";
        TestCase(u"abc de fg", "[abc]W [de]W [fg]W", true, false, true, &lens);
    }
    {
        //soft hyphen
        const TString lens = "4 1 2 1 2 ";
        TestCase(u"ab\u00AD"
                 u"c de fg",
                 "[abc]W [de]W [fg]W", true, false, true, &lens);
    }
    {
        const TString lens = "3 1 18 1 2 ";
        TestCase(u"abc %D0%BA%D1%83%D0%B1 de", "[abc]W [куб]W [de]W", true, false, true, &lens);
    }
    {
        const TString lens = "3 1 18 3 12 1 2 ";
        TestCase(u"abc %D0%BA%D1%83%D0%B1%20%D0%B1%D1%83 de", "[abc]W [куб]W [бу]W [de]W", true, false, true, &lens);
    }
}

void TTokenizerTest::TestSurrogatePairs() {
    TestCase(u"\U00013348", "[\xF0\x93\x8D\x88]W");
    TestCase(u"𠚼𠝹𠠝𠤖𠪴𠮨𠳭", "[𠚼]W[𠝹]W[𠠝]W[𠤖]W[𠪴]W[𠮨]W[𠳭]W");
    TestCase(u"𠚼  𠝹abc𠠝\n\n𠤖\r\n 𠪴!\n𠮨 \r \n \r 𠳭", "[𠚼]W  [𠝹]W[abc]W[𠠝]W\n\n[𠤖]W\r\n [𠪴]W!\n[𠮨]W \r \n \r [𠳭]W");
    TestCase(u"  𠚼 ?𠝹...𠠝: 𠤖 ::𠪴\n!\n𠮨.\r𠳭 xyz", "  [𠚼]W ?[𠝹]W...[𠠝]W: [𠤖]W ::[𠪴]W\n!\n[𠮨]W.\r[𠳭]W [xyz]W");
    TestCase(u"...𠚼 ?? 𠝹-abc'𠠝 -𠤖: 𠪴-!\n𠮨\r::𠳭  ", "...[𠚼]W ?? [𠝹]W-[abc]W'[𠠝]W -[𠤖]W: [𠪴]W-!\n[𠮨]W\r::[𠳭]W  ");
    TestCase(u"-𠚼  𠝹???𠠝𠤖 ---𠪴!:\n𠮨 \r\r\r𠳭...", "-[𠚼]W  [𠝹]W???[𠠝]W[𠤖]W ---[𠪴]W!:\n[𠮨]W \r\r\r[𠳭]W...");
    TestCase(u" ?𠚼\n\n\n  𠝹  \n\n\n𠠝𠤖\r\r\r 𠪴\n!𠮨 \n𠳭?\n", " ?[𠚼]W\n\n\n  [𠝹]W  \n\n\n[𠠝]W[𠤖]W\r\r\r [𠪴]W\n![𠮨]W \n[𠳭]W?\n");

    const wchar16 text1[] = {'a', 0xDB00, 0xDB00, 'b', 0xDB00, 0}; // surrogate leads
    TestCase(text1, "[a]W\xEF\xBF\xBD\xEF\xBF\xBD[b]W\xEF\xBF\xBD");
    const wchar16 text2[] = {'x', 0xDC00, 0xDC00, 'y', 0xDC00, 0}; // surrogate tails
    TestCase(text2, "[x]W\xEF\xBF\xBD\xEF\xBF\xBD[y]W\xEF\xBF\xBD");
}

void TTokenizerTest::TestUTF8SurrogatePairs() {
    TestCase(u"%f0%a0%9a%bc %f0%a0%9d%b9 %f0%a0%a0%9d %f0%a0%a4%96 %f0%a0%aa%b4", "[𠚼]W [𠝹]W [𠠝]W [𠤖]W [𠪴]W");
}

void TTokenizerTest::TestIdeographs() {
    TestCase(u"IU \uC544\uC774\uC720 Complete Album 2008-2011",
             "[IU]W [아이유]W [Complete]W [Album]W [[2008]N-[2011]N]", false);
    TestCase(u"このようなことが起きているかいないかは検出可能である",
             "[こ]W[の]W[よ]W[う]W[な]W[こ]W[と]W[が]W[起]W[き]W[て]W[い]W[る]W[か]W[い]W[な]W[い]W[か]W[は]W[検]W[出]W[可]W[能]W[で]W[あ]W[る]W");
    TestCase(u"こ の よ う な こ と が 起 き て い る か い な い か は 検 出 可 能 で あ る",
             "[こ]W [の]W [よ]W [う]W [な]W [こ]W [と]W [が]W [起]W [き]W [て]W [い]W [る]W [か]W [い]W [な]W [い]W [か]W [は]W [検]W [出]W [可]W [能]W [で]W [あ]W [る]W");
    TestCase(u"検  出abc可\n\n能\r\n で!\nあ \r \n \r る", "[検]W  [出]W[abc]W[可]W\n\n[能]W\r\n [で]W!\n[あ]W \r \n \r [る]W");
    TestCase(u"  検 ?出...可: 能 ::で\n!\nあ.\rる xyz", "  [検]W ?[出]W...[可]W: [能]W ::[で]W\n!\n[あ]W.\r[る]W [xyz]W");
    TestCase(u"...検 ?? 出-abc'可 -能: で-!\nあ\r::る  ", "...[検]W ?? [出]W-[abc]W'[可]W -[能]W: [で]W-!\n[あ]W\r::[る]W  ");
    TestCase(u"-検  出???可能 ---で!:\nあ \r\r\rる...", "-[検]W  [出]W???[可]W[能]W ---[で]W!:\n[あ]W \r\r\r[る]W...");
    TestCase(u" ?検\n\n\n  出  \n\n\n可能\r\r\r で\n!あ \nる?\n", " ?[検]W\n\n\n  [出]W  \n\n\n[可]W[能]W\r\r\r [で]W\n![あ]W \n[る]W?\n");
}

void TTokenizerTest::TestAccents() {
    {
        // accent (0x0301) in the front - it was cause of std::bad_alloc in backward compatible mode
        TestCase(u"\u0301%F0%F1%F2%F3",
                 "\xCC\x81"
                 "%[F0]M%[F1]M%[F2]M%[F3]M",
                 true); // backward compatible
    }
    {
        const wchar16 text[] = {0x652, '2', '5', '0', '1', 0};
        TestCase(text, "[\xD9\x92"
                       "2501]N");
    }
    {
        const wchar16 text[] = {0x301, '1', '2', 0x301, '3', '4', 0x301, 0};
        TestCase(text, "[\xCC\x81"
                       "12\xCC\x81"
                       "34\xCC\x81]N");
    }
    {
        const wchar16 text[] = {0x301, 'o', 'p', 0x301, 'q', 'r', 0x301, 0};
        TestCase(text, "[\xCC\x81op\xCC\x81qr\xCC\x81]W");
    }
    {
        const wchar16 text[] = {' ', 0x301, 'o', 'p', 0x301, 'q', 'r', 0x301, 0};
        TestCase(text, " \xCC\x81[op\xCC\x81qr\xCC\x81]W");
    }
    {
        const wchar16 text[] = {'o', ' ', 0x301, 'p', 'q', 'r', ' ', 's', 0};
        TestCase(text, "[o]W \xCC\x81[pqr]W [s]W");
    }
    {
        const wchar16 text[] = {'o', ' ', 0x301, 'p', 'q', 'r', '-', 0x301, 's', 't', 'u', ' ', 'v', 0};
        TestCase(text, "[o]W \xCC\x81[[pqr]W-[\xCC\x81stu]W] [v]W");
    }
    {
        const wchar16 text[] = {0x301, 'a', 'b', 'c', 0x00AD, 0x301, '1', '2', '3', 0x301, 0};
        TestCase(text, "[\xCC\x81"
                       "abc\xCC\x81"
                       "123\xCC\x81]M");
    }
    {
        const wchar16 text[] = {0x301, 'a', 'b', 'c', 0x00AD, 0x0301, 'd', 'e', 'f', 0x301, 0};
        TestCase(text, "[\xCC\x81"
                       "abc\xCC\x81"
                       "def\xCC\x81]W");
    }
    {
        const wchar16 text[] = {0x301, '1', '2', '3', 0x00AD, 0x0301, 'a', 'b', 'c', 0x301, 0};
        TestCase(text, "[\xCC\x81"
                       "123\xCC\x81"
                       "abc\xCC\x81]M");
    }
    {
        const wchar16 text[] = {0x301, '1', '2', '3', 0x00AD, 0x0301, '4', '5', '6', 0x301, 0};
        TestCase(text, "[\xCC\x81"
                       "123\xCC\x81"
                       "456\xCC\x81]N");
    }
    //////////////////////////////////////////////////////////////////////////
    {
        const wchar16 text[] = {0x301, 'a', 'b', 'c', 0x00AD, 0x301, '1', '2', '3', 0x301, 0};
        TestCase(text, "[[\xCC\x81"
                       "abc]W[\xCC\x81"
                       "123\xCC\x81]N]",
                 false);
    }
    {
        const wchar16 text[] = {0x301, '1', '2', '3', 0x00AD, 0x0301, 'a', 'b', 'c', 0x301, 0};
        TestCase(text, "[[\xCC\x81"
                       "123]N[\xCC\x81"
                       "abc\xCC\x81]W]",
                 false);
    }
#ifndef CATBOOST_OPENSOURCE
    { // #1
        TUtf16String text(u"abc");
        text.insert((size_t)0, (size_t)5, (wchar16)0x301);
        text.append(5ul, 0x301u);

        TTestTokenHandler handler;
        TNlpTokenizer tokenizer(handler, false);
        tokenizer.Tokenize(text.c_str(), text.size(), false);
        UNIT_ASSERT_STRINGS_EQUAL(WideToChar(UTF8ToWide(handler.GetOutputText()), CODES_YANDEX).c_str(), "[\x80\x80\x80\x80\200abc\x80\x80\x80\x80\x80]W");
    }
    { // #2
        TUtf16String text(u"abc");
        text.insert((size_t)0, (size_t)10, (size_t)0x301);
        text.append(300ul, 0x301u);

        TTestTokenHandler handler;
        TNlpTokenizer tokenizer(handler, false);
        tokenizer.Tokenize(text.c_str(), text.size(), false);
        UNIT_ASSERT_STRINGS_EQUAL(WideToChar(UTF8ToWide(handler.GetOutputText()), CODES_YANDEX).c_str(), "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80[abc\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80]W");
    }
    { // #3
        TUtf16String text(u"abc");
        text.insert(size_t(0), size_t(300), wchar16(0x301));
        text.append(size_t(10), wchar16(0x301));

        TTestTokenHandler handler;
        TNlpTokenizer tokenizer(handler, false);
        tokenizer.Tokenize(text.c_str(), text.size(), false);
        UNIT_ASSERT_STRINGS_EQUAL(WideToChar(UTF8ToWide(handler.GetOutputText()), CODES_YANDEX).c_str(), "\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80[abc\x80\x80\x80\x80\x80\x80\x80\x80\x80\x80]W");
    }
#endif
}

void TTokenizerTest::TestEmptyString() {
    const wchar16 text[] = {0};
    TestCase(text, "");
}

void TTokenizerTest::TestWordBreaks() {
    const TStringBuf text1 = AsStringBuf("These\0words\0are\0separated\0with\0zeros.\0 The next sentence.\0The last sentence.");
    const TUtf16String s1(UTF8ToWide(text1));
    TestCase(s1.c_str(), s1.size(), "[These]W [words]W [are]W [separated]W [with]W [zeros]W.  <S>[The]W [next]W [sentence]W. <S>[The]W [last]W [sentence]W.");
}

TString ReplaceControlCharacters(const char* p) {
    TString s;
    while (*p) {
        if (*p == '\n')
            s += "\\n";
        else if (*p == '\r')
            s += "\\r";
        else if (*p == '\t')
            s += "\\t";
        else
            s += *p;
        ++p;
    }
    return s;
}

void TTokenizerTest::TestTwitterUserNames() {
    TestCase(u"@_jackster", "@_[jackster]W", false);
    TestCase(u"@hs_girl_probz_", "[[@hs]W_[girl]W_[probz]W]_", false);
    TestCase(u"@MikeEpps__", "[@MikeEpps]W__", false);
    TestCase(u"@_Billy__Madison", "@_[Billy]W__[Madison]W", false);
    TestCase(u"@sandyhuricane_", "[@sandyhuricane]W_", false);
    TestCase(u"@_getl0w", "@_[[getl]W[0]N[w]W]", false);
    TestCase(u"@8_Semesters", "[[@8]N_[Semesters]W]", false);
    TestCase(u"@savannah__rose", "[@savannah]W__[rose]W", false);
}

void TTokenizerTest::TestCharClasses() {
    UNIT_ASSERT_VALUES_EQUAL(TNlpParser::GetCharClass(','), (int)TNlpParser::CC_COMMA);
    //    UNIT_ASSERT_VALUES_EQUAL(TNlpParser::GetCharClass(';'), (int)TNlpParser::CC_TERM_PUNCT);
    UNIT_ASSERT_VALUES_EQUAL(TNlpParser::GetCharClass(0x2116), (int)TNlpParser::CC_NUMERO_SIGN); // №
    UNIT_ASSERT_VALUES_EQUAL(TNlpParser::GetCharClass(0x00A9), (int)TNlpParser::CC_COPYRIGHT_SIGN);
    UNIT_ASSERT_VALUES_EQUAL(TNlpParser::GetCharClass('*'), (int)TNlpParser::CC_ASTERISK);
    UNIT_ASSERT_VALUES_EQUAL(TNlpParser::GetCharClass(0xD855), (int)TNlpParser::CC_SURROGATE_LEAD);
    UNIT_ASSERT_VALUES_EQUAL(TNlpParser::GetCharClass(0xDC6C), (int)TNlpParser::CC_SURROGATE_TAIL);
}

void TTokenizerTest::TestReversible() {
    TVector<TString> textsUtf8{
        "мама мыла раму",
        "#test",
        "qwe#asd",
        "https://yandex.ru/yandsearch?text=qwe",
        "https://yandex.ru/yandsearch?text=%D0%BC%D0%B0%D0%BC%D0%B0%20%D0%BC%D1%8B%D0%BB%D0%B0%20%D1%80%D0%B0%D0%BC%D1%83&lr=213",
        "c++ европа+",
        "check\ni++\n<=>",
        "c++d",
        "<html>\n\t<head></head>\n\t<body>\n\t\t<a href=\"http://yandex.ru/\" />\n\t</body>\n</html>"
    };

    TJoinAllTokenHandler handler;
    TNlpTokenizer tokenizer(handler, false);
    for (size_t version = 2; version <= 4; ++version) {
        for (const TString& s : textsUtf8) {
            handler.Reset();
            TTokenizerOptions opts;
            opts.Version = version;
            opts.UrlDecode = false;
            TUtf16String sUtf16 = UTF8ToWide(s);
            tokenizer.Tokenize(sUtf16.c_str(), sUtf16.size(), opts);
            UNIT_ASSERT_VALUES_EQUAL(handler.GetResult(), s);
        }
    }
}
