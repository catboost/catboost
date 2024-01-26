#include <util/generic/utility.h>
#include <util/string/util.h>
#include <library/cpp/testing/unittest/registar.h>
#include "lcs_via_lis.h"

class TLCSTest: public TTestBase {
    UNIT_TEST_SUITE(TLCSTest);
    UNIT_TEST(LCSTest);
    UNIT_TEST_SUITE_END();

private:
    size_t Length(TStringBuf s1, TStringBuf s2) {
        TVector<TVector<size_t>> c;
        c.resize(s1.size() + 1);

        for (size_t i = 0; i < c.size(); ++i) {
            c[i].resize(s2.size() + 1);
        }

        for (size_t i = 0; i < s1.size(); ++i) {
            for (size_t j = 0; j < s2.size(); ++j) {
                if (s1[i] == s2[j])
                    c[i + 1][j + 1] = c[i][j] + 1;
                else
                    c[i + 1][j + 1] = Max(c[i + 1][j], c[i][j + 1]);
            }
        }

        return c[s1.size()][s2.size()];
    }

    void CheckLCSLength(TStringBuf s1, TStringBuf s2, size_t size) {
        size_t len = NLCS::MeasureLCS<char>(s1, s2);

        UNIT_ASSERT_VALUES_EQUAL(len, Length(s1, s2));
        UNIT_ASSERT_VALUES_EQUAL(len, size);
    }

    void CheckLCSString(TStringBuf s1, TStringBuf s2, TStringBuf reflcs) {
        TString lcs;
        size_t len = NLCS::MakeLCS<char>(s1, s2, &lcs);
        const TString comment = Sprintf("%s & %s = %s", s1.data(), s2.data(), reflcs.data());

        UNIT_ASSERT_VALUES_EQUAL_C(Length(s1, s2), len, comment);
        UNIT_ASSERT_VALUES_EQUAL_C(lcs.size(), len, comment);
        UNIT_ASSERT_VALUES_EQUAL_C(NLCS::MeasureLCS<char>(s1, s2), len, comment);
        UNIT_ASSERT_VALUES_EQUAL_C(reflcs, TStringBuf(lcs), comment);
    }

    void LCSTest() {
        CheckLCSString("abacx", "baabca", "bac");
        const char* m = "mama_myla_ramu";
        const char* n = "papa_lubil_mamu";
        const char* s = "aa_l_amu";
        CheckLCSString(m, n, s);
        CheckLCSString(n, m, s);
        CheckLCSString(m, m, m);
        CheckLCSString(m, "", "");
        CheckLCSString("", m, "");
        CheckLCSString("", "", "");
        {
            const char* s1 =
                "atmwuaoccmgirveexxtbkkmioclarskvicyesddirlrgflnietcagkswbvsdnxksipfndmusnuee"
                "tojygyjyobdfiutsbruuspvibmywhokxsarwbyirsqqnxxnbtkmucmdafaogwmobuuhejspgurpe"
                "ceokhgdubqewnyqtwwqfibydbcbxhofvcjsftamhvnbdwxdqnphcdiowplcmguxmnpcufrahjagv"
                "cpjqsyqurrhyghkasnodqqktldyfomgabrxdxpubenfscamoodgpocilewqtbsncggwvkkpuasdl"
                "cltxywovqjligwhjxnmmtgeuujphgtdjrnxcnmwwbgxnpiotgnhyrxuvxkxdlgmpfeyocaviuhec"
                "ydecqmttjictnpptoblxqgcrsojrymakgdjvcnppinymdrlgdfpyuykwpmdeifqwupojsgmruyxs"
                "qijwnxngaxnppxgusyfnaelccytxwrkhxxirhnsastdlyejslrivkrpovnhbwxcdmpqbnjthjtlj"
                "wnoayakfnpcwdnlgnrghjhiianhsncbjehlsoldjykymduytyiygrdreicjvghdibyjsmxnwvrqb"
                "jjkjfrtlonfarbwhovladadlbyeygvuxcutxjqhosuxbemtwsqjlvvyegsfgeladiovecfxfymct"
                "ulofkcogguantmicfrhpnauetejktkhamfuwirjvlyfgjrobywbbitbnckiegbiwbtmapqrbbqws"
                "wviuplyprwwqoekiuxsmwgwyuwgeurvxempguwmgtadvrkxykffjxfdyxmsibmdlqhldlfbiaegt"
                "kswcmfidnrhaibuscoyukwhdtoqwlpwnfgamvfqjpfklgurcwvgsluyoyiuumrwwsqgxatxnxhil"
                "ywpkeaugfaahmchjruavepvnygcmcjdnvulwcuhlolsfxcsrjciwajbhdahpldcfggubcxalqxrl"
                "coaiyeawxyxujtynodhnxbhs";
            const char* s2 =
                "yjufxfeiifhrmydpmsqqgjwtpcxbhqmfpnvsvsapqvsmqmugpqehbdsiiqhcrasxuhcvswcwanwb"
                "knesbuhtbaitcwebsdbbxwyubjoroekjxweeqnqmydbdbgbnfymcermhpbikocpsfccwjemxjpmc"
                "hkhtfaoqgvvtpipujsxesiglgnpsdwfjhcawkfpffuyltqqhdkeqwkfpqjhnjdsnxlevyvydikbr"
                "hnicihnevsofgouxjcnjjknxwwgaaaikdcxmhxfowadqudrapvwlcuwatrmiweijljdehxiwqrnq"
                "tnhgukbwmadcjpfnxtswhmwnvvnwsllkyshfobrdmukswpgwunysxpnnlmccolvqyjsvagulpcda"
                "ctsnyjleqgttcgpnhlnagxenuknpxiramgeshhjyoesupkcfcvvpwyweuvcwrawsgvfshppijuug"
                "hdnujdqjtcdissmlnjgibdljjxntxrgytxlbgvsrrusatqelspeoyvndjifjqxqrpduwbyojjbhi"
                "tmondbbnuuhpkglmfykeheddwkxjyapfniqoic";
            CheckLCSLength(s1, s2, 247);
        }
        {
            const char* s1 =
                "ssyrtvllktbhijkyckviokukaodexiykxcvlvninoasblpuujnqnilnhmacsulaulskphkccnfop"
                "jlhejqhemdpnihasouinuceugkfkaifdepvffntbkfivsddtxyslnlwiyfbbpnrwlkryetncahih"
                "vqcrdvvehvrxgnitghbomkprdngrqdswypwfvpdrvqnavxnprakkgoibsxybngvenlbfcghcyldn"
                "kssfuxvpvfhaawqiandxpsrkyqtiltmojmmwygevhodvsuvdojvwpvvbwpbbnerriufrwgwcjlgx"
                "jcjchsfiomtkowihdtcyewknlvdeankladmdhwqxokmunttgaqdsbuyhekkxatpydfgquyxuucda"
                "dllepofxoirmaablfyyibcnqkdbnsaygkqkbvupdhajfshouofnokwlbdtglrbklpgknyuiedppl"
                "chxbnnmbumdtrsgwitjlmkkdxysvmsvcdulmanmsdeqkmwgfthmntdbthdthdodnubqajpfyssea"
                "hwxymiyubkhhxlbmjptujiemrdljqkskdkuokvimencavihwqdaqtcljrgwvxpuegnoecobfllwu"
                "upsfhjrrpiqtjlwigjkpltwfpoqxsdrojtawpaximslojqtadsactemuhpnshkgyoyouldanktcg"
                "dhxdpwawabxwjhnjdmewrwtciquuiqnwdsbdvnuvjyewmjppkwvcotptmyrsqaovmaysjuvtenuy"
                "orvdsssgjgcgksdwaaladocotgveuscwauawdhqlkqsjtmltvkkcfkgwpdtormkefkigkppwpvsy"
                "fpblccsbyboouahotiifixsjuxlvqpqmpmkcgbvsefkqasearilhlxuqdfqqxxohhbdyiudqsree"
                "btwfxdtxlaynrusgbffhltthkejitseuqyeqvhqgyobijwmspwbsisghsarji";
            const char* s2 =
                "lcnygawhsrprxafwnwxnrxurpnqwwtwuciuexxvswckwopnmhhmhvdwmxtgceitqofivvavnqlmo"
                "hbieyaxcysagyqcvijxyowiognsntxcdlmueoafqvqwafyhgyoewhwxvqswvwfkqohtutawwnmhr"
                "pjmhyiygdekonlxhkbeaheqvnqbwhambhrrhkyfcehjfblgriumapavbcxxdqytiuylxmhtjjloa"
                "fvnyhqsgalacknksxxxilgfcankxreaburvmxukmfprfpmfqgqhmirlnltkjxslumhtamtndaffh"
                "ybfywiwjlafnsgsvywflsfkeeidwptigidvhnqdjiwvgkfynncyujodsjiglubptsdofftfxayno"
                "txoykiivbvvipdpcgjadvbanaeljdbxxynlqqpdphirrbjqfqtnxabitncafatbosjnlimxfxlrh"
                "thlfsdyfbhtwywewqubjcvskmwxwjyhesqbviflihnjfejcgjtkgicgmmcurynmdaoifojmedcqb"
                "aeqalxnljvglnvyverrtosfeeuajyxkmmpjgcioqxnnqpjokxaenfoudondtahjduymwsyxpbvrf"
                "kpgiavuihdliphwojgjobafhjshjnmhufciepexevtfwcybtripqmnekkirxmljumyxjpqbvmftt"
                "xmfwokjskummmkaksbeoowfgjwiumpbkdujexectnghqjjuotofwuwvcgtluephymdkrscbfiaeg"
                "aaypnclkstfqimisanikjxocmhcrotkntprwjbuudswuyuujfawjucwgifhqgepxeidmvcwqsqrv"
                "karuvpnxhmrvdcocidgtuxasdqkwsdvijmnpmayhfiva";
            CheckLCSLength(s1, s2, 288);
        }
    }
};
UNIT_TEST_SUITE_REGISTRATION(TLCSTest)
