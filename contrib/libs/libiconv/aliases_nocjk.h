/* ANSI-C code produced by gperf version 3.0.4 */
/* Command-line: gperf -m10 aliases_nocjk.gperf  */
/* Computed positions: -k'1-2,4-11,$' */

#if !((' ' == 32) && ('!' == 33) && ('"' == 34) && ('#' == 35) \
      && ('%' == 37) && ('&' == 38) && ('\'' == 39) && ('(' == 40) \
      && (')' == 41) && ('*' == 42) && ('+' == 43) && (',' == 44) \
      && ('-' == 45) && ('.' == 46) && ('/' == 47) && ('0' == 48) \
      && ('1' == 49) && ('2' == 50) && ('3' == 51) && ('4' == 52) \
      && ('5' == 53) && ('6' == 54) && ('7' == 55) && ('8' == 56) \
      && ('9' == 57) && (':' == 58) && (';' == 59) && ('<' == 60) \
      && ('=' == 61) && ('>' == 62) && ('?' == 63) && ('A' == 65) \
      && ('B' == 66) && ('C' == 67) && ('D' == 68) && ('E' == 69) \
      && ('F' == 70) && ('G' == 71) && ('H' == 72) && ('I' == 73) \
      && ('J' == 74) && ('K' == 75) && ('L' == 76) && ('M' == 77) \
      && ('N' == 78) && ('O' == 79) && ('P' == 80) && ('Q' == 81) \
      && ('R' == 82) && ('S' == 83) && ('T' == 84) && ('U' == 85) \
      && ('V' == 86) && ('W' == 87) && ('X' == 88) && ('Y' == 89) \
      && ('Z' == 90) && ('[' == 91) && ('\\' == 92) && (']' == 93) \
      && ('^' == 94) && ('_' == 95) && ('a' == 97) && ('b' == 98) \
      && ('c' == 99) && ('d' == 100) && ('e' == 101) && ('f' == 102) \
      && ('g' == 103) && ('h' == 104) && ('i' == 105) && ('j' == 106) \
      && ('k' == 107) && ('l' == 108) && ('m' == 109) && ('n' == 110) \
      && ('o' == 111) && ('p' == 112) && ('q' == 113) && ('r' == 114) \
      && ('s' == 115) && ('t' == 116) && ('u' == 117) && ('v' == 118) \
      && ('w' == 119) && ('x' == 120) && ('y' == 121) && ('z' == 122) \
      && ('{' == 123) && ('|' == 124) && ('}' == 125) && ('~' == 126))
/* The character set is not based on ISO-646.  */
#error "gperf generated tables don't work with this execution character set. Please report a bug to <bug-gnu-gperf@gnu.org>."
#endif

#line 1 "aliases_nocjk.gperf"
struct alias { int name; unsigned int encoding_index; };

#define TOTAL_KEYWORDS 253
#define MIN_WORD_LENGTH 2
#define MAX_WORD_LENGTH 19
#define MIN_HASH_VALUE 6
#define MAX_HASH_VALUE 622
/* maximum key range = 617, duplicates = 0 */

#ifdef __GNUC__
__inline
#else
#ifdef __cplusplus
inline
#endif
#endif
static unsigned int
aliases_hash (const char *str, unsigned int len)
{
  static const unsigned short asso_values[] =
    {
      623, 623, 623, 623, 623, 623, 623, 623, 623, 623,
      623, 623, 623, 623, 623, 623, 623, 623, 623, 623,
      623, 623, 623, 623, 623, 623, 623, 623, 623, 623,
      623, 623, 623, 623, 623, 623, 623, 623, 623, 623,
      623, 623, 623, 623, 623,   0,   8, 623,  73,   2,
        0,  87,   9,  12,   5,  82,  15,  19, 203, 623,
      623, 623, 623, 623, 623,   1, 228,   0,  83,   1,
      623,   9,  23,   0,   2,  52, 116,  12,  14,  43,
        7, 623,  93,   0,  82, 141,  13,  68,   8,  22,
        1, 623, 623, 623, 623,   1, 623, 623, 623, 623,
      623, 623, 623, 623, 623, 623, 623, 623, 623, 623,
      623, 623, 623, 623, 623, 623, 623, 623, 623, 623,
      623, 623, 623, 623, 623, 623, 623, 623
    };
  int hval = len;

  switch (hval)
    {
      default:
        hval += asso_values[(unsigned char)str[10]];
      /*FALLTHROUGH*/
      case 10:
        hval += asso_values[(unsigned char)str[9]];
      /*FALLTHROUGH*/
      case 9:
        hval += asso_values[(unsigned char)str[8]];
      /*FALLTHROUGH*/
      case 8:
        hval += asso_values[(unsigned char)str[7]];
      /*FALLTHROUGH*/
      case 7:
        hval += asso_values[(unsigned char)str[6]];
      /*FALLTHROUGH*/
      case 6:
        hval += asso_values[(unsigned char)str[5]];
      /*FALLTHROUGH*/
      case 5:
        hval += asso_values[(unsigned char)str[4]];
      /*FALLTHROUGH*/
      case 4:
        hval += asso_values[(unsigned char)str[3]];
      /*FALLTHROUGH*/
      case 3:
      case 2:
        hval += asso_values[(unsigned char)str[1]];
      /*FALLTHROUGH*/
      case 1:
        hval += asso_values[(unsigned char)str[0]];
        break;
    }
  return hval + asso_values[(unsigned char)str[len - 1]];
}

struct stringpool_t
  {
    char stringpool_str6[sizeof("ASCII")];
    char stringpool_str7[sizeof("CSASCII")];
    char stringpool_str8[sizeof("CSVISCII")];
    char stringpool_str9[sizeof("JAVA")];
    char stringpool_str16[sizeof("MAC")];
    char stringpool_str17[sizeof("CP862")];
    char stringpool_str19[sizeof("VISCII")];
    char stringpool_str20[sizeof("MS-EE")];
    char stringpool_str23[sizeof("862")];
    char stringpool_str24[sizeof("CSUCS4")];
    char stringpool_str25[sizeof("CP1252")];
    char stringpool_str27[sizeof("CP866")];
    char stringpool_str28[sizeof("866")];
    char stringpool_str29[sizeof("CP1251")];
    char stringpool_str32[sizeof("ECMA-114")];
    char stringpool_str34[sizeof("MS-ANSI")];
    char stringpool_str35[sizeof("CP1256")];
    char stringpool_str40[sizeof("VISCII1.1-1")];
    char stringpool_str41[sizeof("C99")];
    char stringpool_str42[sizeof("CP154")];
    char stringpool_str43[sizeof("CP1254")];
    char stringpool_str44[sizeof("ECMA-118")];
    char stringpool_str49[sizeof("CP1255")];
    char stringpool_str52[sizeof("CP819")];
    char stringpool_str55[sizeof("CP1258")];
    char stringpool_str70[sizeof("ISO8859-2")];
    char stringpool_str71[sizeof("ISO-8859-2")];
    char stringpool_str72[sizeof("ISO_8859-2")];
    char stringpool_str74[sizeof("ISO8859-1")];
    char stringpool_str75[sizeof("ISO-8859-1")];
    char stringpool_str76[sizeof("ISO_8859-1")];
    char stringpool_str77[sizeof("ISO8859-11")];
    char stringpool_str78[sizeof("ISO-8859-11")];
    char stringpool_str79[sizeof("ISO_8859-11")];
    char stringpool_str80[sizeof("ISO8859-6")];
    char stringpool_str81[sizeof("ISO-8859-6")];
    char stringpool_str82[sizeof("ISO_8859-6")];
    char stringpool_str83[sizeof("ISO8859-16")];
    char stringpool_str84[sizeof("ISO-8859-16")];
    char stringpool_str85[sizeof("ISO_8859-16")];
    char stringpool_str87[sizeof("ISO_8859-16:2001")];
    char stringpool_str88[sizeof("ISO8859-4")];
    char stringpool_str89[sizeof("ISO-8859-4")];
    char stringpool_str90[sizeof("ISO_8859-4")];
    char stringpool_str91[sizeof("ISO8859-14")];
    char stringpool_str92[sizeof("ISO-8859-14")];
    char stringpool_str93[sizeof("ISO_8859-14")];
    char stringpool_str94[sizeof("ISO8859-5")];
    char stringpool_str95[sizeof("ISO-8859-5")];
    char stringpool_str96[sizeof("ISO_8859-5")];
    char stringpool_str97[sizeof("ISO8859-15")];
    char stringpool_str98[sizeof("ISO-8859-15")];
    char stringpool_str99[sizeof("ISO_8859-15")];
    char stringpool_str100[sizeof("ISO8859-8")];
    char stringpool_str101[sizeof("ISO-8859-8")];
    char stringpool_str102[sizeof("ISO_8859-8")];
    char stringpool_str103[sizeof("850")];
    char stringpool_str104[sizeof("ISO_8859-14:1998")];
    char stringpool_str106[sizeof("CP1131")];
    char stringpool_str107[sizeof("ISO_8859-15:1998")];
    char stringpool_str108[sizeof("ISO8859-9")];
    char stringpool_str109[sizeof("ISO-8859-9")];
    char stringpool_str110[sizeof("ISO_8859-9")];
    char stringpool_str111[sizeof("ISO-IR-6")];
    char stringpool_str112[sizeof("CP874")];
    char stringpool_str113[sizeof("ISO-IR-226")];
    char stringpool_str114[sizeof("TCVN")];
    char stringpool_str115[sizeof("ISO-IR-126")];
    char stringpool_str118[sizeof("L2")];
    char stringpool_str120[sizeof("ISO-IR-166")];
    char stringpool_str122[sizeof("L1")];
    char stringpool_str123[sizeof("CSKZ1048")];
    char stringpool_str124[sizeof("PT154")];
    char stringpool_str125[sizeof("R8")];
    char stringpool_str126[sizeof("MACTHAI")];
    char stringpool_str128[sizeof("L6")];
    char stringpool_str130[sizeof("CSPTCP154")];
    char stringpool_str132[sizeof("ISO-IR-144")];
    char stringpool_str133[sizeof("ARMSCII-8")];
    char stringpool_str135[sizeof("PTCP154")];
    char stringpool_str136[sizeof("L4")];
    char stringpool_str137[sizeof("LATIN2")];
    char stringpool_str141[sizeof("LATIN1")];
    char stringpool_str142[sizeof("L5")];
    char stringpool_str143[sizeof("US")];
    char stringpool_str144[sizeof("ISO-IR-148")];
    char stringpool_str145[sizeof("GEORGIAN-PS")];
    char stringpool_str146[sizeof("UCS-2")];
    char stringpool_str147[sizeof("LATIN6")];
    char stringpool_str148[sizeof("L8")];
    char stringpool_str149[sizeof("ANSI_X3.4-1986")];
    char stringpool_str150[sizeof("US-ASCII")];
    char stringpool_str151[sizeof("CSUNICODE")];
    char stringpool_str152[sizeof("ISO_646.IRV:1991")];
    char stringpool_str153[sizeof("ISO_8859-10:1992")];
    char stringpool_str155[sizeof("LATIN4")];
    char stringpool_str158[sizeof("CSUNICODE11")];
    char stringpool_str159[sizeof("ANSI_X3.4-1968")];
    char stringpool_str161[sizeof("LATIN5")];
    char stringpool_str162[sizeof("ISO-IR-199")];
    char stringpool_str164[sizeof("UCS-4")];
    char stringpool_str166[sizeof("GEORGIAN-ACADEMY")];
    char stringpool_str167[sizeof("LATIN8")];
    char stringpool_str169[sizeof("ISO646-US")];
    char stringpool_str170[sizeof("CP850")];
    char stringpool_str171[sizeof("CP1250")];
    char stringpool_str174[sizeof("KZ-1048")];
    char stringpool_str176[sizeof("LATIN-9")];
    char stringpool_str181[sizeof("CP367")];
    char stringpool_str182[sizeof("ISO-IR-101")];
    char stringpool_str187[sizeof("ROMAN8")];
    char stringpool_str188[sizeof("MACROMANIA")];
    char stringpool_str189[sizeof("CP1257")];
    char stringpool_str191[sizeof("GREEK8")];
    char stringpool_str194[sizeof("L10")];
    char stringpool_str197[sizeof("CSMACINTOSH")];
    char stringpool_str198[sizeof("MACROMAN")];
    char stringpool_str199[sizeof("CP1253")];
    char stringpool_str201[sizeof("TCVN-5712")];
    char stringpool_str202[sizeof("NEXTSTEP")];
    char stringpool_str206[sizeof("TCVN5712-1")];
    char stringpool_str207[sizeof("MACINTOSH")];
    char stringpool_str209[sizeof("ISO-CELTIC")];
    char stringpool_str210[sizeof("CSHPROMAN8")];
    char stringpool_str212[sizeof("GREEK")];
    char stringpool_str213[sizeof("CHAR")];
    char stringpool_str214[sizeof("TIS620.2529-1")];
    char stringpool_str216[sizeof("ISO-IR-109")];
    char stringpool_str219[sizeof("ISO8859-10")];
    char stringpool_str220[sizeof("ISO-8859-10")];
    char stringpool_str221[sizeof("ISO_8859-10")];
    char stringpool_str222[sizeof("ISO-IR-138")];
    char stringpool_str225[sizeof("ISO-IR-179")];
    char stringpool_str228[sizeof("MS-GREEK")];
    char stringpool_str229[sizeof("MACGREEK")];
    char stringpool_str232[sizeof("HP-ROMAN8")];
    char stringpool_str234[sizeof("ISO8859-7")];
    char stringpool_str235[sizeof("ISO-8859-7")];
    char stringpool_str236[sizeof("ISO_8859-7")];
    char stringpool_str237[sizeof("ASMO-708")];
    char stringpool_str239[sizeof("TIS620")];
    char stringpool_str240[sizeof("TIS-620")];
    char stringpool_str241[sizeof("UTF-16")];
    char stringpool_str242[sizeof("CSUNICODE11UTF7")];
    char stringpool_str244[sizeof("ISO8859-3")];
    char stringpool_str245[sizeof("ISO-8859-3")];
    char stringpool_str246[sizeof("ISO_8859-3")];
    char stringpool_str247[sizeof("ISO8859-13")];
    char stringpool_str248[sizeof("ISO-8859-13")];
    char stringpool_str249[sizeof("ISO_8859-13")];
    char stringpool_str250[sizeof("ISO-10646-UCS-2")];
    char stringpool_str251[sizeof("CSKOI8R")];
    char stringpool_str253[sizeof("ISO-IR-110")];
    char stringpool_str254[sizeof("IBM862")];
    char stringpool_str257[sizeof("ELOT_928")];
    char stringpool_str258[sizeof("UTF-8")];
    char stringpool_str259[sizeof("ISO-10646-UCS-4")];
    char stringpool_str260[sizeof("HEBREW")];
    char stringpool_str262[sizeof("CYRILLIC")];
    char stringpool_str263[sizeof("RK1048")];
    char stringpool_str264[sizeof("IBM866")];
    char stringpool_str266[sizeof("UCS-2LE")];
    char stringpool_str267[sizeof("CSISOLATIN2")];
    char stringpool_str269[sizeof("ISO-IR-127")];
    char stringpool_str271[sizeof("CSISOLATIN1")];
    char stringpool_str272[sizeof("MACCROATIAN")];
    char stringpool_str273[sizeof("CSISOLATINARABIC")];
    char stringpool_str274[sizeof("CSISOLATINCYRILLIC")];
    char stringpool_str275[sizeof("UCS-4LE")];
    char stringpool_str276[sizeof("CP1133")];
    char stringpool_str277[sizeof("CSISOLATIN6")];
    char stringpool_str278[sizeof("CSIBM866")];
    char stringpool_str280[sizeof("KOI8-T")];
    char stringpool_str281[sizeof("ISO-IR-157")];
    char stringpool_str282[sizeof("L7")];
    char stringpool_str283[sizeof("CYRILLIC-ASIAN")];
    char stringpool_str285[sizeof("CSISOLATIN4")];
    char stringpool_str286[sizeof("LATIN10")];
    char stringpool_str288[sizeof("WINDOWS-1252")];
    char stringpool_str289[sizeof("IBM819")];
    char stringpool_str290[sizeof("WINDOWS-1251")];
    char stringpool_str291[sizeof("CSISOLATIN5")];
    char stringpool_str292[sizeof("L3")];
    char stringpool_str293[sizeof("WINDOWS-1256")];
    char stringpool_str297[sizeof("WINDOWS-1254")];
    char stringpool_str299[sizeof("UNICODE-1-1")];
    char stringpool_str300[sizeof("WINDOWS-1255")];
    char stringpool_str301[sizeof("LATIN7")];
    char stringpool_str302[sizeof("KOI8-R")];
    char stringpool_str303[sizeof("WINDOWS-1258")];
    char stringpool_str304[sizeof("ISO_8859-4:1988")];
    char stringpool_str305[sizeof("CSPC862LATINHEBREW")];
    char stringpool_str307[sizeof("ISO_8859-5:1988")];
    char stringpool_str310[sizeof("ISO_8859-8:1988")];
    char stringpool_str311[sizeof("LATIN3")];
    char stringpool_str314[sizeof("TIS620-0")];
    char stringpool_str316[sizeof("UTF-32")];
    char stringpool_str318[sizeof("ISO_8859-9:1989")];
    char stringpool_str319[sizeof("STRK1048-2002")];
    char stringpool_str320[sizeof("UCS-2-SWAPPED")];
    char stringpool_str321[sizeof("MACICELAND")];
    char stringpool_str324[sizeof("ISO-IR-100")];
    char stringpool_str326[sizeof("MACUKRAINE")];
    char stringpool_str327[sizeof("MULELAO-1")];
    char stringpool_str328[sizeof("ARABIC")];
    char stringpool_str329[sizeof("UCS-4-SWAPPED")];
    char stringpool_str332[sizeof("CSISOLATINGREEK")];
    char stringpool_str334[sizeof("WCHAR_T")];
    char stringpool_str338[sizeof("MACCENTRALEUROPE")];
    char stringpool_str345[sizeof("MACARABIC")];
    char stringpool_str350[sizeof("ISO-IR-203")];
    char stringpool_str356[sizeof("UTF-16LE")];
    char stringpool_str361[sizeof("WINDOWS-1250")];
    char stringpool_str362[sizeof("ISO_8859-2:1987")];
    char stringpool_str363[sizeof("CSISOLATINHEBREW")];
    char stringpool_str364[sizeof("ISO_8859-1:1987")];
    char stringpool_str366[sizeof("MS-CYRL")];
    char stringpool_str367[sizeof("ISO_8859-6:1987")];
    char stringpool_str369[sizeof("TIS620.2533-1")];
    char stringpool_str370[sizeof("WINDOWS-1257")];
    char stringpool_str371[sizeof("MACCYRILLIC")];
    char stringpool_str372[sizeof("MAC-CYRILLIC")];
    char stringpool_str375[sizeof("WINDOWS-1253")];
    char stringpool_str378[sizeof("UCS-2BE")];
    char stringpool_str382[sizeof("ISO_8859-3:1988")];
    char stringpool_str385[sizeof("UNICODE-1-1-UTF-7")];
    char stringpool_str387[sizeof("UCS-4BE")];
    char stringpool_str388[sizeof("WINDOWS-874")];
    char stringpool_str392[sizeof("UTF-7")];
    char stringpool_str398[sizeof("KOI8-U")];
    char stringpool_str407[sizeof("IBM850")];
    char stringpool_str436[sizeof("UTF-32LE")];
    char stringpool_str437[sizeof("MACTURKISH")];
    char stringpool_str439[sizeof("MS-TURK")];
    char stringpool_str440[sizeof("TIS620.2533-0")];
    char stringpool_str441[sizeof("CSISOLATIN3")];
    char stringpool_str444[sizeof("ISO_8859-7:1987")];
    char stringpool_str449[sizeof("ISO_8859-7:2003")];
    char stringpool_str457[sizeof("MS-HEBR")];
    char stringpool_str461[sizeof("UCS-2-INTERNAL")];
    char stringpool_str468[sizeof("UTF-16BE")];
    char stringpool_str470[sizeof("UCS-4-INTERNAL")];
    char stringpool_str490[sizeof("IBM367")];
    char stringpool_str492[sizeof("KOI8-RU")];
    char stringpool_str499[sizeof("TCVN5712-1:1993")];
    char stringpool_str504[sizeof("MACHEBREW")];
    char stringpool_str510[sizeof("IBM-CP1133")];
    char stringpool_str538[sizeof("UNICODEBIG")];
    char stringpool_str548[sizeof("UTF-32BE")];
    char stringpool_str570[sizeof("MS-ARAB")];
    char stringpool_str576[sizeof("UNICODELITTLE")];
    char stringpool_str586[sizeof("CSPC850MULTILINGUAL")];
    char stringpool_str622[sizeof("WINBALTRIM")];
  };
static const struct stringpool_t stringpool_contents =
  {
    "ASCII",
    "CSASCII",
    "CSVISCII",
    "JAVA",
    "MAC",
    "CP862",
    "VISCII",
    "MS-EE",
    "862",
    "CSUCS4",
    "CP1252",
    "CP866",
    "866",
    "CP1251",
    "ECMA-114",
    "MS-ANSI",
    "CP1256",
    "VISCII1.1-1",
    "C99",
    "CP154",
    "CP1254",
    "ECMA-118",
    "CP1255",
    "CP819",
    "CP1258",
    "ISO8859-2",
    "ISO-8859-2",
    "ISO_8859-2",
    "ISO8859-1",
    "ISO-8859-1",
    "ISO_8859-1",
    "ISO8859-11",
    "ISO-8859-11",
    "ISO_8859-11",
    "ISO8859-6",
    "ISO-8859-6",
    "ISO_8859-6",
    "ISO8859-16",
    "ISO-8859-16",
    "ISO_8859-16",
    "ISO_8859-16:2001",
    "ISO8859-4",
    "ISO-8859-4",
    "ISO_8859-4",
    "ISO8859-14",
    "ISO-8859-14",
    "ISO_8859-14",
    "ISO8859-5",
    "ISO-8859-5",
    "ISO_8859-5",
    "ISO8859-15",
    "ISO-8859-15",
    "ISO_8859-15",
    "ISO8859-8",
    "ISO-8859-8",
    "ISO_8859-8",
    "850",
    "ISO_8859-14:1998",
    "CP1131",
    "ISO_8859-15:1998",
    "ISO8859-9",
    "ISO-8859-9",
    "ISO_8859-9",
    "ISO-IR-6",
    "CP874",
    "ISO-IR-226",
    "TCVN",
    "ISO-IR-126",
    "L2",
    "ISO-IR-166",
    "L1",
    "CSKZ1048",
    "PT154",
    "R8",
    "MACTHAI",
    "L6",
    "CSPTCP154",
    "ISO-IR-144",
    "ARMSCII-8",
    "PTCP154",
    "L4",
    "LATIN2",
    "LATIN1",
    "L5",
    "US",
    "ISO-IR-148",
    "GEORGIAN-PS",
    "UCS-2",
    "LATIN6",
    "L8",
    "ANSI_X3.4-1986",
    "US-ASCII",
    "CSUNICODE",
    "ISO_646.IRV:1991",
    "ISO_8859-10:1992",
    "LATIN4",
    "CSUNICODE11",
    "ANSI_X3.4-1968",
    "LATIN5",
    "ISO-IR-199",
    "UCS-4",
    "GEORGIAN-ACADEMY",
    "LATIN8",
    "ISO646-US",
    "CP850",
    "CP1250",
    "KZ-1048",
    "LATIN-9",
    "CP367",
    "ISO-IR-101",
    "ROMAN8",
    "MACROMANIA",
    "CP1257",
    "GREEK8",
    "L10",
    "CSMACINTOSH",
    "MACROMAN",
    "CP1253",
    "TCVN-5712",
    "NEXTSTEP",
    "TCVN5712-1",
    "MACINTOSH",
    "ISO-CELTIC",
    "CSHPROMAN8",
    "GREEK",
    "CHAR",
    "TIS620.2529-1",
    "ISO-IR-109",
    "ISO8859-10",
    "ISO-8859-10",
    "ISO_8859-10",
    "ISO-IR-138",
    "ISO-IR-179",
    "MS-GREEK",
    "MACGREEK",
    "HP-ROMAN8",
    "ISO8859-7",
    "ISO-8859-7",
    "ISO_8859-7",
    "ASMO-708",
    "TIS620",
    "TIS-620",
    "UTF-16",
    "CSUNICODE11UTF7",
    "ISO8859-3",
    "ISO-8859-3",
    "ISO_8859-3",
    "ISO8859-13",
    "ISO-8859-13",
    "ISO_8859-13",
    "ISO-10646-UCS-2",
    "CSKOI8R",
    "ISO-IR-110",
    "IBM862",
    "ELOT_928",
    "UTF-8",
    "ISO-10646-UCS-4",
    "HEBREW",
    "CYRILLIC",
    "RK1048",
    "IBM866",
    "UCS-2LE",
    "CSISOLATIN2",
    "ISO-IR-127",
    "CSISOLATIN1",
    "MACCROATIAN",
    "CSISOLATINARABIC",
    "CSISOLATINCYRILLIC",
    "UCS-4LE",
    "CP1133",
    "CSISOLATIN6",
    "CSIBM866",
    "KOI8-T",
    "ISO-IR-157",
    "L7",
    "CYRILLIC-ASIAN",
    "CSISOLATIN4",
    "LATIN10",
    "WINDOWS-1252",
    "IBM819",
    "WINDOWS-1251",
    "CSISOLATIN5",
    "L3",
    "WINDOWS-1256",
    "WINDOWS-1254",
    "UNICODE-1-1",
    "WINDOWS-1255",
    "LATIN7",
    "KOI8-R",
    "WINDOWS-1258",
    "ISO_8859-4:1988",
    "CSPC862LATINHEBREW",
    "ISO_8859-5:1988",
    "ISO_8859-8:1988",
    "LATIN3",
    "TIS620-0",
    "UTF-32",
    "ISO_8859-9:1989",
    "STRK1048-2002",
    "UCS-2-SWAPPED",
    "MACICELAND",
    "ISO-IR-100",
    "MACUKRAINE",
    "MULELAO-1",
    "ARABIC",
    "UCS-4-SWAPPED",
    "CSISOLATINGREEK",
    "WCHAR_T",
    "MACCENTRALEUROPE",
    "MACARABIC",
    "ISO-IR-203",
    "UTF-16LE",
    "WINDOWS-1250",
    "ISO_8859-2:1987",
    "CSISOLATINHEBREW",
    "ISO_8859-1:1987",
    "MS-CYRL",
    "ISO_8859-6:1987",
    "TIS620.2533-1",
    "WINDOWS-1257",
    "MACCYRILLIC",
    "MAC-CYRILLIC",
    "WINDOWS-1253",
    "UCS-2BE",
    "ISO_8859-3:1988",
    "UNICODE-1-1-UTF-7",
    "UCS-4BE",
    "WINDOWS-874",
    "UTF-7",
    "KOI8-U",
    "IBM850",
    "UTF-32LE",
    "MACTURKISH",
    "MS-TURK",
    "TIS620.2533-0",
    "CSISOLATIN3",
    "ISO_8859-7:1987",
    "ISO_8859-7:2003",
    "MS-HEBR",
    "UCS-2-INTERNAL",
    "UTF-16BE",
    "UCS-4-INTERNAL",
    "IBM367",
    "KOI8-RU",
    "TCVN5712-1:1993",
    "MACHEBREW",
    "IBM-CP1133",
    "UNICODEBIG",
    "UTF-32BE",
    "MS-ARAB",
    "UNICODELITTLE",
    "CSPC850MULTILINGUAL",
    "WINBALTRIM"
  };
#define stringpool ((const char *) &stringpool_contents)

static const struct alias aliases[] =
  {
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 13 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str6, ei_ascii},
#line 22 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str7, ei_ascii},
#line 258 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str8, ei_viscii},
#line 52 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str9, ei_java},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 212 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str16, ei_mac_roman},
#line 201 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str17, ei_cp862},
    {-1},
#line 256 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str19, ei_viscii},
#line 173 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str20, ei_cp1250},
    {-1}, {-1},
#line 203 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str23, ei_cp862},
#line 35 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str24, ei_ucs4},
#line 177 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str25, ei_cp1252},
    {-1},
#line 205 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str27, ei_cp866},
#line 207 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str28, ei_cp866},
#line 174 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str29, ei_cp1251},
    {-1}, {-1},
#line 98 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str32, ei_iso8859_6},
    {-1},
#line 179 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str34, ei_cp1252},
#line 189 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str35, ei_cp1256},
    {-1}, {-1}, {-1}, {-1},
#line 257 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str40, ei_viscii},
#line 51 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str41, ei_c99},
#line 237 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str42, ei_pt154},
#line 183 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str43, ei_cp1254},
#line 108 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str44, ei_iso8859_7},
    {-1}, {-1}, {-1}, {-1},
#line 186 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str49, ei_cp1255},
    {-1}, {-1},
#line 57 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str52, ei_iso8859_1},
    {-1}, {-1},
#line 195 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str55, ei_cp1258},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1},
#line 70 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str70, ei_iso8859_2},
#line 63 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str71, ei_iso8859_2},
#line 64 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str72, ei_iso8859_2},
    {-1},
#line 62 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str74, ei_iso8859_1},
#line 53 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str75, ei_iso8859_1},
#line 54 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str76, ei_iso8859_1},
#line 139 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str77, ei_iso8859_11},
#line 137 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str78, ei_iso8859_11},
#line 138 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str79, ei_iso8859_11},
#line 102 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str80, ei_iso8859_6},
#line 94 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str81, ei_iso8859_6},
#line 95 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str82, ei_iso8859_6},
#line 166 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str83, ei_iso8859_16},
#line 160 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str84, ei_iso8859_16},
#line 161 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str85, ei_iso8859_16},
    {-1},
#line 162 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str87, ei_iso8859_16},
#line 86 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str88, ei_iso8859_4},
#line 79 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str89, ei_iso8859_4},
#line 80 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str90, ei_iso8859_4},
#line 153 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str91, ei_iso8859_14},
#line 146 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str92, ei_iso8859_14},
#line 147 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str93, ei_iso8859_14},
#line 93 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str94, ei_iso8859_5},
#line 87 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str95, ei_iso8859_5},
#line 88 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str96, ei_iso8859_5},
#line 159 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str97, ei_iso8859_15},
#line 154 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str98, ei_iso8859_15},
#line 155 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str99, ei_iso8859_15},
#line 120 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str100, ei_iso8859_8},
#line 114 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str101, ei_iso8859_8},
#line 115 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str102, ei_iso8859_8},
#line 199 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str103, ei_cp850},
#line 148 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str104, ei_iso8859_14},
    {-1},
#line 209 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str106, ei_cp1131},
#line 156 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str107, ei_iso8859_15},
#line 128 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str108, ei_iso8859_9},
#line 121 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str109, ei_iso8859_9},
#line 122 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str110, ei_iso8859_9},
#line 16 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str111, ei_ascii},
#line 254 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str112, ei_cp874},
#line 163 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str113, ei_iso8859_16},
#line 259 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str114, ei_tcvn},
#line 107 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str115, ei_iso8859_7},
    {-1}, {-1},
#line 68 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str118, ei_iso8859_2},
    {-1},
#line 253 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str120, ei_tis620},
    {-1},
#line 60 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str122, ei_iso8859_1},
#line 243 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str123, ei_rk1048},
#line 235 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str124, ei_pt154},
#line 228 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str125, ei_hp_roman8},
#line 225 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str126, ei_mac_thai},
    {-1},
#line 134 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str128, ei_iso8859_10},
    {-1},
#line 239 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str130, ei_pt154},
    {-1},
#line 90 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str132, ei_iso8859_5},
#line 231 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str133, ei_armscii_8},
    {-1},
#line 236 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str135, ei_pt154},
#line 84 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str136, ei_iso8859_4},
#line 67 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str137, ei_iso8859_2},
    {-1}, {-1}, {-1},
#line 59 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str141, ei_iso8859_1},
#line 126 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str142, ei_iso8859_9},
#line 21 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str143, ei_ascii},
#line 124 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str144, ei_iso8859_9},
#line 233 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str145, ei_georgian_ps},
#line 24 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str146, ei_ucs2},
#line 133 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str147, ei_iso8859_10},
#line 151 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str148, ei_iso8859_14},
#line 18 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str149, ei_ascii},
#line 12 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str150, ei_ascii},
#line 26 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str151, ei_ucs2},
#line 15 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str152, ei_ascii},
#line 131 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str153, ei_iso8859_10},
    {-1},
#line 83 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str155, ei_iso8859_4},
    {-1}, {-1},
#line 30 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str158, ei_ucs2be},
#line 17 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str159, ei_ascii},
    {-1},
#line 125 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str161, ei_iso8859_9},
#line 149 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str162, ei_iso8859_14},
    {-1},
#line 33 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str164, ei_ucs4},
    {-1},
#line 232 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str166, ei_georgian_academy},
#line 150 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str167, ei_iso8859_14},
    {-1},
#line 14 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str169, ei_ascii},
#line 197 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str170, ei_cp850},
#line 171 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str171, ei_cp1250},
    {-1}, {-1},
#line 242 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str174, ei_rk1048},
    {-1},
#line 158 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str176, ei_iso8859_15},
    {-1}, {-1}, {-1}, {-1},
#line 19 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str181, ei_ascii},
#line 66 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str182, ei_iso8859_2},
    {-1}, {-1}, {-1}, {-1},
#line 227 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str187, ei_hp_roman8},
#line 217 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str188, ei_mac_romania},
#line 192 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str189, ei_cp1257},
    {-1},
#line 110 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str191, ei_iso8859_7},
    {-1}, {-1},
#line 165 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str194, ei_iso8859_16},
    {-1}, {-1},
#line 213 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str197, ei_mac_roman},
#line 210 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str198, ei_mac_roman},
#line 180 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str199, ei_cp1253},
    {-1},
#line 260 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str201, ei_tcvn},
#line 230 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str202, ei_nextstep},
    {-1}, {-1}, {-1},
#line 261 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str206, ei_tcvn},
#line 211 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str207, ei_mac_roman},
    {-1},
#line 152 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str209, ei_iso8859_14},
#line 229 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str210, ei_hp_roman8},
    {-1},
#line 111 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str212, ei_iso8859_7},
#line 263 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str213, ei_local_char},
#line 250 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str214, ei_tis620},
    {-1},
#line 74 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str216, ei_iso8859_3},
    {-1}, {-1},
#line 136 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str219, ei_iso8859_10},
#line 129 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str220, ei_iso8859_10},
#line 130 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str221, ei_iso8859_10},
#line 117 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str222, ei_iso8859_8},
    {-1}, {-1},
#line 142 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str225, ei_iso8859_13},
    {-1}, {-1},
#line 182 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str228, ei_cp1253},
#line 221 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str229, ei_mac_greek},
    {-1}, {-1},
#line 226 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str232, ei_hp_roman8},
    {-1},
#line 113 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str234, ei_iso8859_7},
#line 103 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str235, ei_iso8859_7},
#line 104 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str236, ei_iso8859_7},
#line 99 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str237, ei_iso8859_6},
    {-1},
#line 248 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str239, ei_tis620},
#line 247 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str240, ei_tis620},
#line 38 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str241, ei_utf16},
#line 46 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str242, ei_utf7},
    {-1},
#line 78 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str244, ei_iso8859_3},
#line 71 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str245, ei_iso8859_3},
#line 72 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str246, ei_iso8859_3},
#line 145 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str247, ei_iso8859_13},
#line 140 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str248, ei_iso8859_13},
#line 141 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str249, ei_iso8859_13},
#line 25 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str250, ei_ucs2},
#line 168 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str251, ei_koi8_r},
    {-1},
#line 82 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str253, ei_iso8859_4},
#line 202 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str254, ei_cp862},
    {-1}, {-1},
#line 109 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str257, ei_iso8859_7},
#line 23 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str258, ei_utf8},
#line 34 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str259, ei_ucs4},
#line 118 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str260, ei_iso8859_8},
    {-1},
#line 91 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str262, ei_iso8859_5},
#line 240 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str263, ei_rk1048},
#line 206 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str264, ei_cp866},
    {-1},
#line 31 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str266, ei_ucs2le},
#line 69 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str267, ei_iso8859_2},
    {-1},
#line 97 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str269, ei_iso8859_6},
    {-1},
#line 61 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str271, ei_iso8859_1},
#line 216 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str272, ei_mac_croatian},
#line 101 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str273, ei_iso8859_6},
#line 92 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str274, ei_iso8859_5},
#line 37 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str275, ei_ucs4le},
#line 245 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str276, ei_cp1133},
#line 135 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str277, ei_iso8859_10},
#line 208 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str278, ei_cp866},
    {-1},
#line 234 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str280, ei_koi8_t},
#line 132 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str281, ei_iso8859_10},
#line 144 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str282, ei_iso8859_13},
#line 238 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str283, ei_pt154},
    {-1},
#line 85 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str285, ei_iso8859_4},
#line 164 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str286, ei_iso8859_16},
    {-1},
#line 178 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str288, ei_cp1252},
#line 58 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str289, ei_iso8859_1},
#line 175 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str290, ei_cp1251},
#line 127 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str291, ei_iso8859_9},
#line 76 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str292, ei_iso8859_3},
#line 190 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str293, ei_cp1256},
    {-1}, {-1}, {-1},
#line 184 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str297, ei_cp1254},
    {-1},
#line 29 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str299, ei_ucs2be},
#line 187 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str300, ei_cp1255},
#line 143 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str301, ei_iso8859_13},
#line 167 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str302, ei_koi8_r},
#line 196 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str303, ei_cp1258},
#line 81 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str304, ei_iso8859_4},
#line 204 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str305, ei_cp862},
    {-1},
#line 89 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str307, ei_iso8859_5},
    {-1}, {-1},
#line 116 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str310, ei_iso8859_8},
#line 75 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str311, ei_iso8859_3},
    {-1}, {-1},
#line 249 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str314, ei_tis620},
    {-1},
#line 41 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str316, ei_utf32},
    {-1},
#line 123 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str318, ei_iso8859_9},
#line 241 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str319, ei_rk1048},
#line 48 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str320, ei_ucs2swapped},
#line 215 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str321, ei_mac_iceland},
    {-1}, {-1},
#line 56 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str324, ei_iso8859_1},
    {-1},
#line 220 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str326, ei_mac_ukraine},
#line 244 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str327, ei_mulelao},
#line 100 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str328, ei_iso8859_6},
#line 50 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str329, ei_ucs4swapped},
    {-1}, {-1},
#line 112 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str332, ei_iso8859_7},
    {-1},
#line 264 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str334, ei_local_wchar_t},
    {-1}, {-1}, {-1},
#line 214 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str338, ei_mac_centraleurope},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 224 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str345, ei_mac_arabic},
    {-1}, {-1}, {-1}, {-1},
#line 157 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str350, ei_iso8859_15},
    {-1}, {-1}, {-1}, {-1}, {-1},
#line 40 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str356, ei_utf16le},
    {-1}, {-1}, {-1}, {-1},
#line 172 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str361, ei_cp1250},
#line 65 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str362, ei_iso8859_2},
#line 119 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str363, ei_iso8859_8},
#line 55 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str364, ei_iso8859_1},
    {-1},
#line 176 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str366, ei_cp1251},
#line 96 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str367, ei_iso8859_6},
    {-1},
#line 252 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str369, ei_tis620},
#line 193 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str370, ei_cp1257},
#line 218 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str371, ei_mac_cyrillic},
#line 219 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str372, ei_mac_cyrillic},
    {-1}, {-1},
#line 181 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str375, ei_cp1253},
    {-1}, {-1},
#line 27 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str378, ei_ucs2be},
    {-1}, {-1}, {-1},
#line 73 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str382, ei_iso8859_3},
    {-1}, {-1},
#line 45 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str385, ei_utf7},
    {-1},
#line 36 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str387, ei_ucs4be},
#line 255 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str388, ei_cp874},
    {-1}, {-1}, {-1},
#line 44 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str392, ei_utf7},
    {-1}, {-1}, {-1}, {-1}, {-1},
#line 169 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str398, ei_koi8_u},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 198 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str407, ei_cp850},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1},
#line 43 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str436, ei_utf32le},
#line 222 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str437, ei_mac_turkish},
    {-1},
#line 185 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str439, ei_cp1254},
#line 251 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str440, ei_tis620},
#line 77 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str441, ei_iso8859_3},
    {-1}, {-1},
#line 105 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str444, ei_iso8859_7},
    {-1}, {-1}, {-1}, {-1},
#line 106 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str449, ei_iso8859_7},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 188 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str457, ei_cp1255},
    {-1}, {-1}, {-1},
#line 47 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str461, ei_ucs2internal},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 39 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str468, ei_utf16be},
    {-1},
#line 49 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str470, ei_ucs4internal},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1},
#line 20 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str490, ei_ascii},
    {-1},
#line 170 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str492, ei_koi8_ru},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 262 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str499, ei_tcvn},
    {-1}, {-1}, {-1}, {-1},
#line 223 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str504, ei_mac_hebrew},
    {-1}, {-1}, {-1}, {-1}, {-1},
#line 246 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str510, ei_cp1133},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 28 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str538, ei_ucs2be},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 42 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str548, ei_utf32be},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1},
#line 191 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str570, ei_cp1256},
    {-1}, {-1}, {-1}, {-1}, {-1},
#line 32 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str576, ei_ucs2le},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 200 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str586, ei_cp850},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
    {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1}, {-1},
#line 194 "aliases_nocjk.gperf"
    {(int)(long)&((struct stringpool_t *)0)->stringpool_str622, ei_cp1257}
  };

#ifdef __GNUC__
__inline
#if defined __GNUC_STDC_INLINE__ || defined __GNUC_GNU_INLINE__
__attribute__ ((__gnu_inline__))
#endif
#endif
const struct alias *
aliases_lookup (const char *str, unsigned int len)
{
  if (len <= MAX_WORD_LENGTH && len >= MIN_WORD_LENGTH)
    {
      int key = aliases_hash (str, len);

      if (key <= MAX_HASH_VALUE && key >= 0)
        {
          int o = aliases[key].name;
          if (o >= 0)
            {
              const char *s = o + stringpool;

              if (*str == *s && !strcmp (str + 1, s + 1))
                return &aliases[key];
            }
        }
    }
  return 0;
}
