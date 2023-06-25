#define	SEOS	1
#define	SCOMMENT	2
#define	SLABEL	3
#define	SUNKNOWN	4
#define	SHOLLERITH	5
#define	SICON	6
#define	SRCON	7
#define	SDCON	8
#define	SBITCON	9
#define	SOCTCON	10
#define	SHEXCON	11
#define	STRUE	12
#define	SFALSE	13
#define	SNAME	14
#define	SNAMEEQ	15
#define	SFIELD	16
#define	SSCALE	17
#define	SINCLUDE	18
#define	SLET	19
#define	SASSIGN	20
#define	SAUTOMATIC	21
#define	SBACKSPACE	22
#define	SBLOCK	23
#define	SCALL	24
#define	SCHARACTER	25
#define	SCLOSE	26
#define	SCOMMON	27
#define	SCOMPLEX	28
#define	SCONTINUE	29
#define	SDATA	30
#define	SDCOMPLEX	31
#define	SDIMENSION	32
#define	SDO	33
#define	SDOUBLE	34
#define	SELSE	35
#define	SELSEIF	36
#define	SEND	37
#define	SENDFILE	38
#define	SENDIF	39
#define	SENTRY	40
#define	SEQUIV	41
#define	SEXTERNAL	42
#define	SFORMAT	43
#define	SFUNCTION	44
#define	SGOTO	45
#define	SASGOTO	46
#define	SCOMPGOTO	47
#define	SARITHIF	48
#define	SLOGIF	49
#define	SIMPLICIT	50
#define	SINQUIRE	51
#define	SINTEGER	52
#define	SINTRINSIC	53
#define	SLOGICAL	54
#define	SNAMELIST	55
#define	SOPEN	56
#define	SPARAM	57
#define	SPAUSE	58
#define	SPRINT	59
#define	SPROGRAM	60
#define	SPUNCH	61
#define	SREAD	62
#define	SREAL	63
#define	SRETURN	64
#define	SREWIND	65
#define	SSAVE	66
#define	SSTATIC	67
#define	SSTOP	68
#define	SSUBROUTINE	69
#define	STHEN	70
#define	STO	71
#define	SUNDEFINED	72
#define	SWRITE	73
#define	SLPAR	74
#define	SRPAR	75
#define	SEQUALS	76
#define	SCOLON	77
#define	SCOMMA	78
#define	SCURRENCY	79
#define	SPLUS	80
#define	SMINUS	81
#define	SSTAR	82
#define	SSLASH	83
#define	SPOWER	84
#define	SCONCAT	85
#define	SAND	86
#define	SOR	87
#define	SNEQV	88
#define	SEQV	89
#define	SNOT	90
#define	SEQ	91
#define	SLT	92
#define	SGT	93
#define	SLE	94
#define	SGE	95
#define	SNE	96
#define	SENDDO	97
#define	SWHILE	98
#define	SSLASHD	99
#define	SBYTE	100

/* #line	125	"/n/bopp/v5/dmg/f2c/gram.in" */
#include "defs.h"
#include "p1defs.h"

static int nstars;			/* Number of labels in an
					   alternate return CALL */
static int datagripe;
static int ndim;
static int vartype;
int new_dcl;
static ftnint varleng;
static struct Dims dims[MAXDIM+1];
extern struct Labelblock **labarray;	/* Labels in an alternate
						   return CALL */
extern int maxlablist;

/* The next two variables are used to verify that each statement might be reached
   during runtime.   lastwasbranch   is tested only in the defintion of the
   stat:   nonterminal. */

int lastwasbranch = NO;
static int thiswasbranch = NO;
extern ftnint yystno;
extern flag intonly;
static chainp datastack;
extern long laststfcn, thisstno;
extern int can_include;	/* for netlib */
extern void endcheck Argdcl((void));
extern struct Primblock *primchk Argdcl((expptr));

#define ESNULL (Extsym *)0
#define NPNULL (Namep)0
#define LBNULL (struct Listblock *)0

 static void
pop_datastack(Void) {
	chainp d0 = datastack;
	if (d0->datap)
		curdtp = (chainp)d0->datap;
	datastack = d0->nextp;
	d0->nextp = 0;
	frchain(&d0);
	}


/* #line	172	"/n/bopp/v5/dmg/f2c/gram.in" */
typedef union 	{
	int ival;
	ftnint lval;
	char *charpval;
	chainp chval;
	tagptr tagval;
	expptr expval;
	struct Labelblock *labval;
	struct Nameblock *namval;
	struct Eqvchain *eqvval;
	Extsym *extval;
	} YYSTYPE;
extern	int	yyerrflag;
#ifndef	YYMAXDEPTH
#define	YYMAXDEPTH	150
#endif
YYSTYPE	yylval;
YYSTYPE	yyval;
#define YYEOFCODE 1
#define YYERRCODE 2
short	yyexca[] =
{-1, 1,
	1, -1,
	-2, 0,
-1, 20,
	4, 38,
	-2, 231,
-1, 24,
	4, 42,
	-2, 231,
-1, 151,
	4, 247,
	-2, 189,
-1, 175,
	4, 269,
	81, 269,
	-2, 189,
-1, 225,
	80, 174,
	-2, 140,
-1, 246,
	77, 231,
	-2, 228,
-1, 273,
	4, 290,
	-2, 144,
-1, 277,
	4, 299,
	81, 299,
	-2, 146,
-1, 330,
	80, 175,
	-2, 142,
-1, 360,
	4, 271,
	17, 271,
	77, 271,
	81, 271,
	-2, 190,
-1, 439,
	94, 0,
	95, 0,
	96, 0,
	97, 0,
	98, 0,
	99, 0,
	-2, 154,
-1, 456,
	4, 293,
	81, 293,
	-2, 144,
-1, 458,
	4, 295,
	81, 295,
	-2, 144,
-1, 460,
	4, 297,
	81, 297,
	-2, 144,
-1, 462,
	4, 300,
	81, 300,
	-2, 145,
-1, 506,
	81, 293,
	-2, 144,
};
#define	YYNPROD	305
#define	YYPRIVATE 57344
#define	YYLAST	1455
short	yyact[] =
{
 239, 359, 474, 306, 416, 427, 299, 389, 473, 267,
 315, 231, 400, 358, 318, 415, 328, 253, 319, 100,
 224, 297, 294, 280, 402, 401, 305, 117, 185, 265,
  17, 122, 204, 275, 196, 191, 202, 203, 119, 129,
 107, 271, 200, 184, 112, 104, 338, 102, 166, 167,
 336, 337, 338, 344, 343, 342, 121, 157, 120, 345,
 347, 346, 349, 348, 350, 261, 276, 336, 337, 338,
 131, 132, 133, 134, 104, 136, 539, 158, 399, 158,
 313, 166, 167, 336, 337, 338, 344, 343, 342, 341,
 340, 311, 345, 347, 346, 349, 348, 350, 399, 398,
 105, 514, 115, 537, 166, 167, 336, 337, 338, 344,
 343, 342, 341, 340, 238, 345, 347, 346, 349, 348,
 350, 106, 130, 104, 478, 211, 187, 188, 412, 320,
 259, 260, 261, 411,  95, 166, 167, 336, 337, 338,
 186, 213, 296, 212, 194, 486, 195, 542, 245,  96,
  97,  98, 527, 104, 529, 158, 523, 449, 258, 158,
 241, 243, 484, 101, 487, 485, 216, 274, 471, 222,
 217, 472, 221, 158, 483, 465, 430, 220, 166, 167,
 259, 260, 261, 262, 158, 166, 167, 336, 337, 338,
 344, 156, 121, 156, 120, 464, 345, 347, 346, 349,
 348, 350, 463, 373, 281, 282, 283, 236, 104, 232,
 242, 242, 249, 101, 292, 301, 263, 468, 290, 302,
 279, 296, 291, 288, 289, 166, 167, 259, 260, 261,
 264, 317, 455, 335, 189, 351, 312, 310, 446, 453,
 431, 284, 425, 335, 166, 167, 259, 260, 261, 262,
 258, 466, 325, 158, 467, 450, 380,  99, 449, 158,
 158, 158, 158, 158, 258, 258, 357, 379, 269, 156,
 234, 420, 266, 156, 421, 409, 393, 335, 410, 394,
 361, 333, 323, 362, 334, 258, 378, 156, 270, 208,
 326, 101, 330, 178, 113, 332, 374, 111, 156, 375,
 376, 403, 352, 110, 109, 108, 354, 355, 385, 386,
 363, 356, 384, 225, 377, 425, 367, 368, 369, 370,
 371, 422, 223, 364, 335, 538, 391, 335, 534, 533,
 532, 335, 423, 335, 372, 413, 408, 395, 390, 166,
 167, 259, 260, 261, 262, 381, 434, 528, 531, 526,
 494, 429, 237, 335, 496, 335, 335, 335, 104, 104,
 490, 298, 138, 158, 258, 335, 448, 156, 258, 258,
 258, 258, 258, 156, 156, 156, 156, 156, 251, 192,
 451, 103, 335, 454, 309, 277, 277, 360, 287, 426,
 118, 352, 166, 167, 259, 260, 261, 262, 137, 387,
 403, 232, 435, 436, 437, 438, 439, 440, 441, 442,
 443, 444, 477, 247, 469, 406, 482, 470, 308, 269,
 452, 166, 167, 336, 337, 338, 344, 335, 479, 155,
 244, 155, 488, 228, 225, 499, 335, 335, 335, 335,
 335, 335, 335, 335, 335, 335, 383, 497, 273, 273,
 495, 502, 201, 258, 150, 151, 214, 175, 103, 103,
 103, 103, 501, 190, 475, 454, 210, 172, 193, 142,
 503, 197, 198, 199, 504, 510, 335, 156, 207, 403,
 277, 513, 507, 508, 509, 331, 277, 482, 517, 489,
 335, 520, 492, 335, 197, 218, 219, 242, 498, 335,
 525, 519, 518, 516, 515, 524, 353, 155, 404, 512,
 246, 155, 248, 104, 406, 417,  30, 535, 406, 511,
 390, 209, 213, 335, 227, 155, 268,  93,   6, 541,
 250, 335, 171, 173, 177,  82, 155, 335,   4, 475,
  81, 335,   5, 273, 543,  80, 457, 459, 461, 382,
 124,  79, 103, 174, 304, 295, 307, 522,  78,  77,
  76,  60,  49, 242,  48,  45, 424, 322,  33, 114,
 530, 118, 206, 316, 414, 321, 205, 397, 396, 300,
 197, 536, 481, 135, 215, 392, 277, 277, 277, 314,
 540, 116,  26, 406,  25, 353,  24,  23,  22,  21,
 388, 286,   9,   8,   7, 155,   2, 404, 303,  20,
 165, 155, 155, 155, 155, 155,  51, 491, 293, 268,
 230, 329, 268, 268, 166, 167, 336, 337, 338, 344,
 343, 457, 459, 461, 327, 345, 347, 346, 349, 348,
 350, 418,  92, 256,  53, 339,  19,  55,  37, 456,
 458, 460, 226,   3,   1,   0,   0,   0,   0,   0,
   0, 307,   0, 405, 197,   0,   0,   0,   0,   0,
   0, 277, 277, 277, 419,   0,   0,   0, 353,   0,
 321,   0,   0,   0,   0,   0, 404,   0,   0,   0,
 493,   0,   0,   0, 432, 166, 167, 336, 337, 338,
 344, 343, 342, 341, 340,   0, 345, 347, 346, 349,
 348, 350,   0,   0,   0, 155,   0, 500,   0,   0,
   0,   0,   0,   0,   0,   0, 268,   0,   0,   0,
   0,   0, 462,   0, 506, 458, 460, 166, 167, 336,
 337, 338, 344, 343, 342, 341, 340,   0, 345, 347,
 346, 349, 348, 350,   0,   0,   0, 295,   0,   0,
   0,   0, 405, 480,   0, 307, 405,   0,   0, 447,
   0,   0,   0,   0, 166, 167, 336, 337, 338, 344,
 343, 342, 341, 340, 316, 345, 347, 346, 349, 348,
 350,   0,   0, 445,   0,   0,   0,   0, 166, 167,
 336, 337, 338, 344, 343, 342, 341, 340, 268, 345,
 347, 346, 349, 348, 350,   0,   0,   0, 505,   0,
   0,   0,   0,   0,   0,   0, 505, 505, 505,   0,
   0,   0,   0,   0,   0,   0, 307,  12,   0,   0,
   0, 405,   0,   0,   0,   0, 505,   0,   0,   0,
 521,  10,  56,  46,  73,  86,  14,  61,  70,  91,
  38,  66,  47,  42,  68,  72,  31,  67,  35,  34,
  11,  88,  36,  18,  41,  39,  28,  16,  57,  58,
  59,  50,  54,  43,  89,  64,  40,  69,  44,  90,
  29,  62,  85,  13,   0,  83,  65,  52,  87,  27,
  74,  63,  15, 433,   0,  71,  84,   0, 166, 167,
 336, 337, 338, 344, 343, 342, 341, 340,   0, 345,
 347, 346, 349, 348, 350,   0,   0,   0,   0,   0,
  32,   0,   0,  75, 166, 167, 336, 337, 338, 344,
 343, 342, 341, 340,   0, 345, 347, 346, 349, 348,
 350,  73,   0,   0,   0,  70,   0,   0,  66,   0,
   0,  68,  72,   0,  67, 161, 162, 163, 164, 170,
 169, 168, 159, 160, 104,   0,   0,   0,   0,   0,
   0,   0,  64,   0,  69,   0,   0,   0,   0,   0,
   0,   0,   0,  65,   0,   0,   0,  74,   0,   0,
   0,   0,  71, 161, 162, 163, 164, 170, 169, 168,
 159, 160, 104,   0, 161, 162, 163, 164, 170, 169,
 168, 159, 160, 104,   0,   0,   0,   0,   0,   0,
  75,   0,   0,   0, 235,   0,   0,   0,   0,   0,
 166, 167, 365,   0, 366,   0,   0,   0,   0,   0,
 240, 161, 162, 163, 164, 170, 169, 168, 159, 160,
 104,   0, 161, 162, 163, 164, 170, 169, 168, 159,
 160, 104, 235, 229,   0,   0,   0,   0, 166, 167,
 233,   0,   0, 235,   0,   0,   0,   0, 240, 166,
 167, 476,   0,   0,   0,   0,   0,   0,   0, 240,
 161, 162, 163, 164, 170, 169, 168, 159, 160, 104,
 161, 162, 163, 164, 170, 169, 168, 159, 160, 104,
 235,   0,   0,   0,   0,   0, 166, 167, 233,   0,
   0, 235,   0,   0,   0,   0, 240, 166, 167, 428,
   0,   0,   0,   0,   0,   0,   0, 240, 161, 162,
 163, 164, 170, 169, 168, 159, 160, 104,   0, 161,
 162, 163, 164, 170, 169, 168, 159, 160, 104, 278,
   0,   0,   0, 272,   0, 166, 167,   0,   0,   0,
   0,   0,   0,   0,   0, 240, 161, 162, 163, 164,
 170, 169, 168, 159, 160, 104,   0,   0,   0,   0,
   0,   0,   0,   0,   0,  94, 161, 162, 163, 164,
 170, 169, 168, 159, 160, 104, 257, 235,   0,   0,
   0,   0,   0, 166, 167,   0,   0,   0, 278,   0,
   0,   0,   0, 240, 166, 167,   0, 123,   0,   0,
 126, 127, 128,   0, 240,   0,   0,   0,   0,   0,
   0,   0, 139, 140,   0, 324, 141,   0, 143, 144,
 145, 166, 167, 146, 147, 148,   0, 149,   0,   0,
   0, 240,   0,   0,   0, 252,   0,   0,   0,   0,
   0, 166, 167, 254,   0, 255,   0, 179, 180, 181,
 182, 183, 161, 162, 163, 164, 170, 169, 168, 159,
 160, 104,   0, 161, 162, 163, 164, 170, 169, 168,
 159, 160, 104, 161, 162, 163, 164, 170, 169, 168,
 159, 160, 104, 161, 162, 163, 164, 170, 169, 168,
 159, 160, 104, 161, 162, 163, 164, 170, 169, 168,
 159, 160, 104,   0,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
   0, 154,   0,   0,   0,   0,   0, 166, 167, 152,
   0, 153, 252,   0,   0,   0,   0,   0, 166, 167,
 285,   0, 154,   0,   0,   0,   0,   0, 166, 167,
 176,   0, 407,   0,   0,   0,   0,   0, 166, 167,
  56,  46, 252,  86,   0,  61,   0,  91, 166, 167,
  47,   0,   0,   0,   0,   0,   0,   0,   0,  88,
   0,   0,   0,   0,   0,   0,  57,  58,  59,  50,
   0,   0,  89,   0,   0,   0,   0,  90,   0,  62,
  85,   0,   0,  83,   0,  52,  87,   0,   0,  63,
   0, 125,   0,   0,  84
};
short	yypact[] =
{
-1000, 536, 524, 830,-1000,-1000,-1000,-1000,-1000,-1000,
 519,-1000,-1000,-1000,-1000,-1000,-1000, 210, 496,  19,
 224, 223, 222, 216,  82, 213,  16, 106,-1000,-1000,
-1000,-1000,-1000,1378,-1000,-1000,-1000,  37,-1000,-1000,
-1000,-1000,-1000,-1000,-1000, 496,-1000,-1000,-1000,-1000,
-1000, 392,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,
-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,
-1000,-1000,-1000,-1000,-1000,-1000,1284, 390,1305, 390,
 212,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,
-1000,-1000,-1000,-1000,-1000, 496, 496, 496, 496,-1000,
 496,-1000, 302,-1000,-1000, 496,-1000, -30, 496, 496,
 496, 375,-1000,-1000,-1000, 496, 208,-1000,-1000,-1000,
-1000, 504, 389, 132,-1000,-1000, 379,-1000,-1000,-1000,
-1000, 106, 496, 496, 375,-1000,-1000, 243, 357, 515,
-1000, 356, 995,1140,1140, 353, 513, 496, 336, 496,
-1000,-1000,-1000,-1000,1198,-1000,-1000,  95,1325,-1000,
-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,-1000,
-1000,-1000,1198, 191, 207,-1000,-1000,1092,1151,-1000,
-1000,-1000,-1000,1295, 311,-1000,-1000, 302, 302, 496,
-1000,-1000, 136, 284,-1000,  82,-1000, 284,-1000,-1000,
-1000, 496,-1000, 341,-1000, 307, 927,   5, 106,  -6,
 496,  82,  28,-1000,-1000,1178,-1000, 496,-1000,-1000,
-1000,-1000,-1000,1140,-1000,1140, 411,-1000,1140,-1000,
 203,-1000, 851, 513,-1000,1140,-1000,-1000,-1000,1140,
1140,-1000, 851,-1000,1140,-1000,  82, 513,-1000, 309,
 202,-1000,1325,-1000,-1000,-1000, 957,-1000,1325,1325,
1325,1325,1325, -22, 256, 122, 342,-1000,-1000, 342,
 342,-1000,1151, 205, 186, 175, 851,-1000,1151,-1000,
-1000,-1000,-1000,-1000,  95,-1000,-1000, 321,-1000,-1000,
 302,-1000,-1000, 198,-1000,-1000,-1000,  37,-1000,  -3,
1315, 496,-1000, 197,-1000,  47,-1000,-1000, 341, 498,
-1000, 496,-1000,-1000, 193,-1000, 242,  28,-1000,-1000,
-1000, 163,1140, 851,1054,-1000, 851, 273,  96, 159,
 851, 496, 825,-1000,1043,1140,1140,1140,1140,1140,
1140,1140,1140,1140,1140,-1000,-1000,-1000,-1000,-1000,
-1000,-1000, 715, 157, -41, 102, 691, 289, 177,-1000,
-1000,-1000,1198, 161, 851,-1000,-1000,  45, -22, -22,
 -22, 142,-1000, 342, 122, 151, 122,-1000,1151,1151,
1151, 654, 121, 114,  94,-1000,-1000,-1000, 173,-1000,
 138,-1000, 284,-1000,  57,-1000,  90,1006,-1000,1315,
-1000,-1000,  39,1102,-1000,-1000,-1000,1140,-1000,-1000,
 496,-1000, 341,  93,  84,-1000,  61,-1000,  83,-1000,
-1000, 496,1140,-1000, 283,1140, 612,-1000, 272, 277,
1140,1140,-1000, 513,-1000, -18, -41, -41, -41, 338,
 -35, -35, 541, 102,  52,-1000,1140,-1000, 513, 513,
  82,-1000,  95,-1000,-1000, 342,-1000,-1000,-1000,-1000,
-1000,-1000,-1000,1151,1151,1151,-1000, 503, 502,  37,
-1000,-1000,1006,-1000,-1000,  21,-1000,-1000,1315,-1000,
-1000,-1000,-1000, 341,-1000, 498, 498, 496,-1000, 851,
1140,  75, 851, 432,-1000,-1000,1140, 271, 851,  71,
 269,  76,-1000,1140, 270, 236, 269, 252, 251, 250,
-1000,-1000,-1000,-1000,1006,-1000,-1000,  17, 247,-1000,
-1000,-1000,  -2,1140,-1000,-1000,-1000, 513,-1000,-1000,
 851,-1000,-1000,-1000,-1000,-1000, 851,-1000,-1000,-1000,
 851,  66, 513,-1000
};
short	yypgo[] =
{
   0, 654, 653,   1, 652, 167,   9,  30, 648, 647,
 646,   4,   0, 645, 644, 643,  39, 642,   3,  26,
 641, 634, 621,  18,  14, 620,  35, 618, 617,  29,
  41,  33,  20, 362,  22, 616,  34, 352,  66, 270,
  16,  57, 378,   2,  24,  25,  11, 207, 114, 610,
 609,  38,  28,  43, 608, 606, 604, 603, 602,1205,
 134, 601, 600,   7, 599, 598, 597, 596, 594, 592,
 591,  31, 589,  19, 585,  21,  37,   6, 584,   5,
  42, 583,  36, 582, 579,  12,  27,  10, 578, 577,
   8,  13,  32, 576, 574, 572,  15, 569, 516, 568,
 567, 566, 565, 564, 562, 561, 560, 454, 559, 558,
 553, 551, 545, 540,  23, 535, 530,  17
};
short	yyr1[] =
{
   0,   1,   1,  55,  55,  55,  55,  55,  55,  55,
   2,  56,  56,  56,  56,  56,  56,  56,  60,  52,
  33,  53,  53,  61,  61,  62,  62,  63,  63,  26,
  26,  26,  27,  27,  34,  34,  17,  57,  57,  57,
  57,  57,  57,  57,  57,  57,  57,  57,  57,  10,
  10,  10,  74,   7,   8,   9,   9,   9,   9,   9,
   9,   9,   9,   9,   9,   9,   9,  16,  16,  16,
  50,  50,  50,  50,  51,  51,  64,  64,  65,  65,
  66,  66,  80,  54,  54,  67,  67,  81,  82,  76,
  83,  84,  77,  77,  85,  85,  45,  45,  45,  70,
  70,  86,  86,  72,  72,  87,  36,  18,  18,  19,
  19,  75,  75,  89,  88,  88,  90,  90,  43,  43,
  91,  91,   3,  68,  68,  92,  92,  95,  93,  94,
  94,  96,  96,  11,  69,  69,  97,  20,  20,  71,
  21,  21,  22,  22,  38,  38,  38,  39,  39,  39,
  39,  39,  39,  39,  39,  39,  39,  39,  39,  39,
  39,  12,  12,  13,  13,  13,  13,  13,  13,  37,
  37,  37,  37,  32,  40,  40,  44,  44,  48,  48,
  48,  48,  48,  48,  48,  47,  49,  49,  49,  41,
  41,  42,  42,  42,  42,  42,  42,  42,  42,  58,
  58,  58,  58,  58,  58, 100,  58,  58,  58,  99,
  23,  24, 101,  24,  98,  98,  98,  98,  98,  98,
  98,  98,  98,  98,  98,   4, 102, 103, 103, 103,
 103,  73,  73,  35,  25,  25,  46,  46,  14,  14,
  28,  28,  59,  78,  79, 104, 105, 105, 105, 105,
 105, 105, 105, 105, 105, 105, 105, 105, 105, 105,
 105, 106, 113, 113, 113, 108, 115, 115, 115, 110,
 110, 107, 107, 116, 116, 117, 117, 117, 117, 117,
 117,  15, 109, 111, 112, 112,  29,  29,   6,   6,
  30,  30,  30,  31,  31,  31,  31,  31,  31,   5,
   5,   5,   5,   5, 114
};
short	yyr2[] =
{
   0,   0,   3,   2,   2,   2,   3,   3,   2,   1,
   1,   3,   4,   3,   4,   4,   5,   3,   0,   1,
   1,   0,   1,   2,   3,   1,   3,   1,   3,   0,
   2,   3,   1,   3,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   2,   1,   5,   7,
   5,   5,   0,   2,   1,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   1,   1,   0,   4,   6,
   3,   4,   5,   3,   1,   3,   3,   3,   3,   3,
   3,   3,   3,   1,   3,   3,   3,   0,   6,   0,
   0,   0,   2,   3,   1,   3,   1,   2,   1,   1,
   3,   1,   1,   1,   3,   3,   2,   1,   5,   1,
   3,   0,   3,   0,   2,   3,   1,   3,   1,   1,
   1,   3,   1,   3,   3,   4,   1,   0,   2,   1,
   3,   1,   3,   1,   1,   2,   4,   1,   3,   0,
   0,   1,   1,   3,   1,   3,   1,   1,   1,   3,
   3,   3,   3,   2,   3,   3,   3,   3,   3,   2,
   3,   1,   1,   1,   1,   1,   1,   1,   1,   1,
   2,   4,   5,   5,   0,   1,   1,   1,   1,   1,
   1,   1,   1,   1,   1,   5,   1,   1,   1,   1,
   3,   1,   1,   3,   3,   3,   3,   2,   3,   1,
   5,   4,   1,   2,   2,   0,   7,   2,   2,   5,
   3,   1,   0,   5,   4,   5,   2,   1,   1,  10,
   1,   3,   4,   3,   3,   1,   1,   3,   3,   7,
   7,   0,   1,   3,   1,   3,   1,   2,   1,   1,
   1,   3,   0,   0,   0,   1,   2,   2,   2,   2,
   2,   2,   2,   3,   4,   4,   2,   3,   4,   1,
   3,   3,   1,   1,   1,   3,   1,   1,   1,   1,
   1,   3,   3,   1,   3,   1,   1,   1,   2,   2,
   2,   1,   3,   3,   4,   4,   1,   3,   1,   5,
   1,   1,   1,   3,   3,   3,   3,   3,   3,   1,
   3,   5,   5,   5,   0
};
short	yychk[] =
{
-1000,  -1, -55,  -2,   2,   6,   4, -56, -57, -58,
  21,  40,   7,  63,  26,  72,  47,  -7,  43, -10,
 -50, -64, -65, -66, -67, -68, -69,  69,  46,  60,
 -98,  36, 100, -99,  39,  38,  42,  -8,  30,  45,
  56,  44,  33,  53,  58,-102,  23,  32,-103,-104,
  51, -35,  67, -14,  52,  -9,  22,  48,  49,  50,
-105,  27,  61,  71,  55,  66,  31,  37,  34,  57,
  28,  75,  35,  24,  70, 103,-106,-108,-109,-111,
-112,-113,-115,  65,  76,  62,  25,  68,  41,  54,
  59,  29, -17,   8, -59, -60, -60, -60, -60,  47,
 -73,  81, -52, -33,  17,  81, 102, -73,  81,  81,
  81,  81, -73,  81, -97,  86, -70, -86, -33, -51,
  88,  86, -71, -59, -98,  73, -59, -59, -59, -16,
  85, -71, -71, -71, -71, -81, -71, -37, -33, -59,
 -59, -59,  77, -59, -59, -59, -59, -59, -59, -59,
-107, -42,  85,  87,  77, -37, -48, -41, -12,  15,
  16,   8,   9,  10,  11, -49,  83,  84,  14,  13,
  12,-107,  77,-107,-110, -42,  85,-107,  81, -59,
 -59, -59, -59, -59, -53, -52, -53, -52, -52, -60,
 -33, -26,  77, -33, -76, -51, -36, -33, -33, -33,
 -80,  77, -82, -76, -92, -93, -95, -33,  81,  17,
  77,  -3, -73,   9,  77, -78, -36, -51, -33, -33,
 -80, -82, -92,  79, -32,  77,  -4,   9,  77,  78,
 -25, -46, -38,  85, -39,  77, -47, -37, -48, -12,
  93, -40, -38, -40,  77,  -3, -33,  77, -33, -41,
-116, -42,  77,-117,  85,  87, -15,  18, -12,  85,
  86,  87,  88, -41, -41, -29,  81,  -6, -37,  77,
  81, -30,  81, -39,  -5, -31, -38, -47,  77, -30,
-114,-114,-114,-114, -41,  85, -61,  77, -26, -26,
 -52, -71,  78, -27, -34, -33,  85, -75,  77, -77,
 -84, -73, -75, -54, -37, -19, -18, -37,  77,  77,
  -7,  86, -86,  86, -72, -87, -33, -73, -24, -23,
 101, -33,-100, -38,  77, -36, -38, -21, -40, -22,
 -38,  74, -38,  78,  81, -12,  85,  86,  87, -13,
  92,  91,  90,  89,  88,  94,  96,  95,  98,  97,
  99,  -3, -38, -39, -38, -38, -38, -73, -91,  -3,
  78,  78,  81, -41, -38,  85,  87, -41, -41, -41,
 -41, -41,  78,  81, -29, -29, -29, -30,  81,  81,
  81, -38, -39,  -5, -31,-114,-114,  78, -62, -63,
  17, -26, -74,  78,  81, -16, -88, -89, 102,  81,
 -85, -45, -44, -12, -47, -33, -48,  77, -36,  78,
  81,  86,  81, -19, -94, -96, -11,  17, -20, -33,
  78,  81,  79, -24,-101,  79, -38, -79,  85,  78,
  80,  81, -33,  78, -46, -38, -38, -38, -38, -38,
 -38, -38, -38, -38, -38,  78,  81,  78,  77,  81,
  78,-117, -41,  78,  -6,  81, -39,  -5, -39,  -5,
 -39,  -5,  78,  81,  81,  81,  78,  81,  79, -75,
 -34,  78,  81, -90, -43, -38,  85, -85,  85, -44,
 -37, -83, -18,  81,  78,  81,  84,  81, -87, -38,
  77, -28, -38,  78,  78, -32,  77, -40, -38,  -3,
 -39, -91,  -3, -73, -23, -33, -39, -23, -23, -23,
 -63,  17, -16, -90,  80, -45, -44, -77, -23, -96,
 -11, -33, -38,  81,  73, -79,  78,  81,  78,  78,
 -38,  78,  78,  78,  78, -43, -38,  86,  78,  78,
 -38,  -3,  81,  -3
};
short	yydef[] =
{
   1,  -2,   0,   0,   9,  10,   2,   3,   4,   5,
   0, 242,   8,  18,  18,  18,  18, 231,   0,  37,
  -2,  39,  40,  41,  -2,  43,  44,  45,  47, 139,
 199, 242, 202,   0, 242, 242, 242,  67, 139, 139,
 139, 139,  87, 139, 134,   0, 242, 242, 217, 218,
 242, 220, 242, 242, 242,  54, 226, 242, 242, 242,
 245, 242, 238, 239,  55,  56,  57,  58,  59,  60,
  61,  62,  63,  64,  65,  66,   0,   0,   0,   0,
 259, 242, 242, 242, 242, 242, 262, 263, 264, 266,
 267, 268,   6,  36,   7,  21,  21,   0,   0,  18,
   0, 232,  29,  19,  20,   0,  89,   0, 232,   0,
   0,   0,  89, 127, 135,   0,  46,  99, 101, 102,
  74,   0,   0, 231, 203, 204,   0, 207, 208,  53,
 243,   0,   0,   0,   0,  89, 127,   0, 169,   0,
 216,   0,   0, 174, 174,   0,   0,   0,   0,   0,
 246,  -2, 248, 249,   0, 191, 192,   0,   0, 178,
 179, 180, 181, 182, 183, 184, 161, 162, 186, 187,
 188, 250,   0, 251, 252,  -2, 270, 256,   0, 304,
 304, 304, 304,   0,  11,  22,  13,  29,  29,   0,
 139,  17,   0, 111,  91, 231,  73, 111,  77,  79,
  81,   0,  86,   0, 124, 126,   0,   0,   0,   0,
   0, 231,   0, 122, 205,   0,  70,   0,  76,  78,
  80,  85, 123,   0, 170,  -2,   0, 225,   0, 221,
   0, 234, 236,   0, 144,   0, 146, 147, 148,   0,
   0, 223, 175, 224,   0, 227,  -2,   0, 233, 275,
   0, 189,   0, 273, 276, 277,   0, 281,   0,   0,
   0,   0,   0, 197, 275, 253,   0, 286, 288,   0,
   0, 257,   0,  -2, 291, 292,   0,  -2,   0, 260,
 261, 265, 282, 283, 304, 304,  12,   0,  14,  15,
  29,  52,  30,   0,  32,  34,  35,  67, 113,   0,
   0,   0, 106,   0,  83,   0, 109, 107,   0,   0,
 128,   0, 100,  75,   0, 103,   0,   0, 201, 211,
 212,   0,   0, 244,   0,  71, 214,   0,   0, 141,
  -2,   0,   0, 222,   0,   0,   0,   0,   0,   0,
   0,   0,   0,   0,   0, 163, 164, 165, 166, 167,
 168, 237,   0, 144, 153, 159,   0,   0,   0, 120,
  -2, 272,   0,   0, 278, 279, 280, 193, 194, 195,
 196, 198, 271,   0, 255,   0, 254, 258,   0,   0,
   0,   0, 144,   0,   0, 284, 285,  23,   0,  25,
  27,  16, 111,  31,   0,  50,   0,   0,  51,   0,
  92,  94,  96,   0,  98, 176, 177,   0,  72,  82,
   0,  90,   0,   0,   0, 129, 131, 133, 136, 137,
  48,   0,   0, 200,   0,   0,   0,  68,   0, 171,
 174,   0, 215,   0, 235, 149, 150, 151, 152,  -2,
 155, 156, 157, 158, 160, 145,   0, 209,   0,   0,
 231, 274, 275, 190, 287,   0,  -2, 294,  -2, 296,
  -2, 298,  -2,   0,   0,   0,  24,   0,   0,  67,
  33, 112,   0, 114, 116, 119, 118,  93,   0,  97,
  84,  91, 110,   0, 125,   0,   0,   0, 104, 105,
   0, 210, 240,   0, 244, 172, 174,   0, 143,   0,
 144,   0, 121,   0,   0, 169,  -2,   0,   0,   0,
  26,  28,  49, 115,   0,  95,  96,   0,   0, 130,
 132, 138,   0,   0, 206,  69, 173,   0, 185, 229,
 230, 289, 301, 302, 303, 117, 119,  88, 108, 213,
 241,   0,   0, 219
};
short	yytok1[] =
{
   1,   4,   5,   6,   7,   8,   9,  10,  11,  12,
  13,  14,  15,  16,  17,  18,  19,  20,  21,  22,
  23,  24,  25,  26,  27,  28,  29,  30,  31,  32,
  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,
  43,  44,  45,  46,  47,  48,  49,  50,  51,  52,
  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,
  63,  64,  65,  66,  67,  68,  69,  70,  71,  72,
  73,  74,  75,  76,  77,  78,  79,  80,  81,  82,
  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,
  93,  94,  95,  96,  97,  98,  99, 100, 101, 102,
 103
};
short	yytok2[] =
{
   2,   3
};
long	yytok3[] =
{
   0
};
#define YYFLAG 		-1000
#define YYERROR		goto yyerrlab
#define YYACCEPT	return(0)
#define YYABORT		return(1)
#define	yyclearin	yychar = -1
#define	yyerrok		yyerrflag = 0

#ifdef	yydebug
//#include	"y.debug"
#else
#define	yydebug		0
char*	yytoknames[1];		/* for debugging */
char*	yystates[1];		/* for debugging */
#endif

/*	parser for yacc output	*/

int	yynerrs = 0;		/* number of errors */
int	yyerrflag = 0;		/* error recovery flag */

extern	int	fprint(int, char*, ...);
extern	int	sprint(char*, char*, ...);

char*
yytokname(int yyc)
{
	static char x[10];

	if(yyc > 0 && yyc <= sizeof(yytoknames)/sizeof(yytoknames[0]))
	if(yytoknames[yyc-1])
		return yytoknames[yyc-1];
	sprintf(x, "<%d>", yyc);
	return x;
}

char*
yystatname(int yys)
{
	static char x[10];

	if(yys >= 0 && yys < sizeof(yystates)/sizeof(yystates[0]))
	if(yystates[yys])
		return yystates[yys];
	sprintf(x, "<%d>\n", yys);
	return x;
}

long
yylex1(void)
{
	long yychar;
	long *t3p;
	int c;

	yychar = yylex();
	if(yychar <= 0) {
		c = yytok1[0];
		goto out;
	}
	if(yychar < sizeof(yytok1)/sizeof(yytok1[0])) {
		c = yytok1[yychar];
		goto out;
	}
	if(yychar >= YYPRIVATE)
		if(yychar < YYPRIVATE+sizeof(yytok2)/sizeof(yytok2[0])) {
			c = yytok2[yychar-YYPRIVATE];
			goto out;
		}
	for(t3p=yytok3;; t3p+=2) {
		c = t3p[0];
		if(c == yychar) {
			c = t3p[1];
			goto out;
		}
		if(c == 0)
			break;
	}
	c = 0;

out:
	if(c == 0)
		c = yytok2[1];	/* unknown char */
	if(yydebug >= 3)
		printf("lex %.4lX %s\n", yychar, yytokname(c));
	return c;
}

int
yyparse(void)
{
	struct
	{
		YYSTYPE	yyv;
		int	yys;
	} yys[YYMAXDEPTH], *yyp, *yypt;
	short *yyxi;
	int yyj, yym, yystate, yyn, yyg;
	YYSTYPE save1, save2;
	int save3, save4;
	long yychar;

	save1 = yylval;
	save2 = yyval;
	save3 = yynerrs;
	save4 = yyerrflag;

	yystate = 0;
	yychar = -1;
	yynerrs = 0;
	yyerrflag = 0;
	yyp = &yys[-1];
	goto yystack;

ret0:
	yyn = 0;
	goto ret;

ret1:
	yyn = 1;
	goto ret;

ret:
	yylval = save1;
	yyval = save2;
	yynerrs = save3;
	yyerrflag = save4;
	return yyn;

yystack:
	/* put a state and value onto the stack */
	if(yydebug >= 4)
		printf("char %s in %s", yytokname(yychar), yystatname(yystate));

	yyp++;
	if(yyp >= &yys[YYMAXDEPTH]) { 
		yyerror("yacc stack overflow"); 
		goto ret1; 
	}
	yyp->yys = yystate;
	yyp->yyv = yyval;

yynewstate:
	yyn = yypact[yystate];
	if(yyn <= YYFLAG)
		goto yydefault; /* simple state */
	if(yychar < 0)
		yychar = yylex1();
	yyn += yychar;
	if(yyn < 0 || yyn >= YYLAST)
		goto yydefault;
	yyn = yyact[yyn];
	if(yychk[yyn] == yychar) { /* valid shift */
		yychar = -1;
		yyval = yylval;
		yystate = yyn;
		if(yyerrflag > 0)
			yyerrflag--;
		goto yystack;
	}

yydefault:
	/* default state action */
	yyn = yydef[yystate];
	if(yyn == -2) {
		if(yychar < 0)
			yychar = yylex1();

		/* look through exception table */
		for(yyxi=yyexca;; yyxi+=2)
			if(yyxi[0] == -1 && yyxi[1] == yystate)
				break;
		for(yyxi += 2;; yyxi += 2) {
			yyn = yyxi[0];
			if(yyn < 0 || yyn == yychar)
				break;
		}
		yyn = yyxi[1];
		if(yyn < 0)
			goto ret0;
	}
	if(yyn == 0) {
		/* error ... attempt to resume parsing */
		switch(yyerrflag) {
		case 0:   /* brand new error */
			yyerror("syntax error");
			if(yydebug >= 1) {
				printf("%s", yystatname(yystate));
				printf("saw %s\n", yytokname(yychar));
			}
yyerrlab:
			yynerrs++;

		case 1:
		case 2: /* incompletely recovered error ... try again */
			yyerrflag = 3;

			/* find a state where "error" is a legal shift action */
			while(yyp >= yys) {
				yyn = yypact[yyp->yys] + YYERRCODE;
				if(yyn >= 0 && yyn < YYLAST) {
					yystate = yyact[yyn];  /* simulate a shift of "error" */
					if(yychk[yystate] == YYERRCODE)
						goto yystack;
				}

				/* the current yyp has no shift onn "error", pop stack */
				if(yydebug >= 2)
					printf("error recovery pops state %d, uncovers %d\n",
						yyp->yys, (yyp-1)->yys );
				yyp--;
			}
			/* there is no state on the stack with an error shift ... abort */
			goto ret1;

		case 3:  /* no shift yet; clobber input char */
			if(yydebug >= YYEOFCODE)
				printf("error recovery discards %s\n", yytokname(yychar));
			if(yychar == YYEOFCODE)
				goto ret1;
			yychar = -1;
			goto yynewstate;   /* try again in the same state */
		}
	}

	/* reduction by production yyn */
	if(yydebug >= 2)
		printf("reduce %d in:\n\t%s", yyn, yystatname(yystate));

	yypt = yyp;
	yyp -= yyr2[yyn];
	yyval = (yyp+1)->yyv;
	yym = yyn;

	/* consult goto table to find next state */
	yyn = yyr1[yyn];
	yyg = yypgo[yyn];
	yyj = yyg + yyp->yys + 1;

	if(yyj >= YYLAST || yychk[yystate=yyact[yyj]] != -yyn)
		yystate = yyact[yyg];
	switch(yym) {
		
case 3:
/* #line	220	"/n/bopp/v5/dmg/f2c/gram.in" */
{
/* stat:   is the nonterminal for Fortran statements */

		  lastwasbranch = NO; } break;
case 5:
/* #line	226	"/n/bopp/v5/dmg/f2c/gram.in" */
{ /* forbid further statement function definitions... */
		  if (parstate == INDATA && laststfcn != thisstno)
			parstate = INEXEC;
		  thisstno++;
		  if(yypt[-1].yyv.labval && (yypt[-1].yyv.labval->labelno==dorange))
			enddo(yypt[-1].yyv.labval->labelno);
		  if(lastwasbranch && thislabel==NULL)
			warn("statement cannot be reached");
		  lastwasbranch = thiswasbranch;
		  thiswasbranch = NO;
		  if(yypt[-1].yyv.labval)
			{
			if(yypt[-1].yyv.labval->labtype == LABFORMAT)
				err("label already that of a format");
			else
				yypt[-1].yyv.labval->labtype = LABEXEC;
			}
		  freetemps();
		} break;
case 6:
/* #line	246	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if (can_include)
			doinclude( yypt[-0].yyv.charpval );
		  else {
			fprintf(diagfile, "Cannot open file %s\n", yypt[-0].yyv.charpval);
			done(1);
			}
		} break;
case 7:
/* #line	254	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if (yypt[-2].yyv.labval)
			lastwasbranch = NO;
		  endcheck();
		  endproc(); /* lastwasbranch = NO; -- set in endproc() */
		} break;
case 8:
/* #line	260	"/n/bopp/v5/dmg/f2c/gram.in" */
{ unclassifiable();

/* flline flushes the current line, ignoring the rest of the text there */

		  flline(); } break;
case 9:
/* #line	266	"/n/bopp/v5/dmg/f2c/gram.in" */
{ flline();  needkwd = NO;  inioctl = NO;
		  yyerrok; yyclearin; } break;
case 10:
/* #line	271	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		if(yystno != 0)
			{
			yyval.labval = thislabel =  mklabel(yystno);
			if( ! headerdone ) {
				if (procclass == CLUNKNOWN)
					procclass = CLMAIN;
				puthead(CNULL, procclass);
				}
			if(thislabel->labdefined)
				execerr("label %s already defined",
					convic(thislabel->stateno) );
			else	{
				if(thislabel->blklevel!=0 && thislabel->blklevel<blklevel
				    && thislabel->labtype!=LABFORMAT)
					warn1("there is a branch to label %s from outside block",
					      convic( (ftnint) (thislabel->stateno) ) );
				thislabel->blklevel = blklevel;
				thislabel->labdefined = YES;
				if(thislabel->labtype != LABFORMAT)
					p1_label((long)(thislabel - labeltab));
				}
			}
		else    yyval.labval = thislabel = NULL;
		} break;
case 11:
/* #line	299	"/n/bopp/v5/dmg/f2c/gram.in" */
{startproc(yypt[-0].yyv.extval, CLMAIN); } break;
case 12:
/* #line	301	"/n/bopp/v5/dmg/f2c/gram.in" */
{	warn("ignoring arguments to main program");
			/* hashclear(); */
			startproc(yypt[-1].yyv.extval, CLMAIN); } break;
case 13:
/* #line	305	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(yypt[-0].yyv.extval) NO66("named BLOCKDATA");
		  startproc(yypt[-0].yyv.extval, CLBLOCK); } break;
case 14:
/* #line	308	"/n/bopp/v5/dmg/f2c/gram.in" */
{ entrypt(CLPROC, TYSUBR, (ftnint) 0,  yypt[-1].yyv.extval, yypt[-0].yyv.chval); } break;
case 15:
/* #line	310	"/n/bopp/v5/dmg/f2c/gram.in" */
{ entrypt(CLPROC, TYUNKNOWN, (ftnint) 0, yypt[-1].yyv.extval, yypt[-0].yyv.chval); } break;
case 16:
/* #line	312	"/n/bopp/v5/dmg/f2c/gram.in" */
{ entrypt(CLPROC, yypt[-4].yyv.ival, varleng, yypt[-1].yyv.extval, yypt[-0].yyv.chval); } break;
case 17:
/* #line	314	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(parstate==OUTSIDE || procclass==CLMAIN
			|| procclass==CLBLOCK)
				execerr("misplaced entry statement", CNULL);
		  entrypt(CLENTRY, 0, (ftnint) 0, yypt[-1].yyv.extval, yypt[-0].yyv.chval);
		} break;
case 18:
/* #line	322	"/n/bopp/v5/dmg/f2c/gram.in" */
{ newproc(); } break;
case 19:
/* #line	326	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.extval = newentry(yypt[-0].yyv.namval, 1); } break;
case 20:
/* #line	330	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.namval = mkname(token); } break;
case 21:
/* #line	333	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.extval = NULL; } break;
case 29:
/* #line	351	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = 0; } break;
case 30:
/* #line	353	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66(" () argument list");
		  yyval.chval = 0; } break;
case 31:
/* #line	356	"/n/bopp/v5/dmg/f2c/gram.in" */
{yyval.chval = yypt[-1].yyv.chval; } break;
case 32:
/* #line	360	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = (yypt[-0].yyv.namval ? mkchain((char *)yypt[-0].yyv.namval,CHNULL) : CHNULL ); } break;
case 33:
/* #line	362	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(yypt[-0].yyv.namval) yypt[-2].yyv.chval = yyval.chval = mkchain((char *)yypt[-0].yyv.namval, yypt[-2].yyv.chval); } break;
case 34:
/* #line	366	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(yypt[-0].yyv.namval->vstg!=STGUNKNOWN && yypt[-0].yyv.namval->vstg!=STGARG)
			dclerr("name declared as argument after use", yypt[-0].yyv.namval);
		  yypt[-0].yyv.namval->vstg = STGARG;
		} break;
case 35:
/* #line	371	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("altenate return argument");

/* substars   means that '*'ed formal parameters should be replaced.
   This is used to specify alternate return labels; in theory, only
   parameter slots which have '*' should accept the statement labels.
   This compiler chooses to ignore the '*'s in the formal declaration, and
   always return the proper value anyway.

   This variable is only referred to in   proc.c   */

		  yyval.namval = 0;  substars = YES; } break;
case 36:
/* #line	387	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		char *s;
		s = copyn(toklen+1, token);
		s[toklen] = '\0';
		yyval.charpval = s;
		} break;
case 45:
/* #line	403	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("SAVE statement");
		  saveall = YES; } break;
case 46:
/* #line	406	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("SAVE statement"); } break;
case 47:
/* #line	408	"/n/bopp/v5/dmg/f2c/gram.in" */
{ fmtstmt(thislabel); setfmt(thislabel); } break;
case 48:
/* #line	410	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("PARAMETER statement"); } break;
case 49:
/* #line	414	"/n/bopp/v5/dmg/f2c/gram.in" */
{ settype(yypt[-4].yyv.namval, yypt[-6].yyv.ival, yypt[-0].yyv.lval);
		  if(ndim>0) setbound(yypt[-4].yyv.namval,ndim,dims);
		} break;
case 50:
/* #line	418	"/n/bopp/v5/dmg/f2c/gram.in" */
{ settype(yypt[-2].yyv.namval, yypt[-4].yyv.ival, yypt[-0].yyv.lval);
		  if(ndim>0) setbound(yypt[-2].yyv.namval,ndim,dims);
		} break;
case 51:
/* #line	422	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if (new_dcl == 2) {
			err("attempt to give DATA in type-declaration");
			new_dcl = 1;
			}
		} break;
case 52:
/* #line	429	"/n/bopp/v5/dmg/f2c/gram.in" */
{ new_dcl = 2; } break;
case 53:
/* #line	432	"/n/bopp/v5/dmg/f2c/gram.in" */
{ varleng = yypt[-0].yyv.lval; } break;
case 54:
/* #line	436	"/n/bopp/v5/dmg/f2c/gram.in" */
{ varleng = (yypt[-0].yyv.ival<0 || ONEOF(yypt[-0].yyv.ival,M(TYLOGICAL)|M(TYLONG))
				? 0 : typesize[yypt[-0].yyv.ival]);
		  vartype = yypt[-0].yyv.ival; } break;
case 55:
/* #line	441	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = TYLONG; } break;
case 56:
/* #line	442	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = tyreal; } break;
case 57:
/* #line	443	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ++complex_seen; yyval.ival = tycomplex; } break;
case 58:
/* #line	444	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = TYDREAL; } break;
case 59:
/* #line	445	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ++dcomplex_seen; NOEXT("DOUBLE COMPLEX statement"); yyval.ival = TYDCOMPLEX; } break;
case 60:
/* #line	446	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = TYLOGICAL; } break;
case 61:
/* #line	447	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("CHARACTER statement"); yyval.ival = TYCHAR; } break;
case 62:
/* #line	448	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = TYUNKNOWN; } break;
case 63:
/* #line	449	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = TYUNKNOWN; } break;
case 64:
/* #line	450	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NOEXT("AUTOMATIC statement"); yyval.ival = - STGAUTO; } break;
case 65:
/* #line	451	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NOEXT("STATIC statement"); yyval.ival = - STGBSS; } break;
case 66:
/* #line	452	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = TYINT1; } break;
case 67:
/* #line	456	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.lval = varleng; } break;
case 68:
/* #line	458	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		expptr p;
		p = yypt[-1].yyv.expval;
		NO66("length specification *n");
		if( ! ISICON(p) || p->constblock.Const.ci <= 0 )
			{
			yyval.lval = 0;
			dclerr("length must be a positive integer constant",
				NPNULL);
			}
		else {
			if (vartype == TYCHAR)
				yyval.lval = p->constblock.Const.ci;
			else switch((int)p->constblock.Const.ci) {
				case 1:	yyval.lval = 1; break;
				case 2: yyval.lval = typesize[TYSHORT];	break;
				case 4: yyval.lval = typesize[TYLONG];	break;
				case 8: yyval.lval = typesize[TYDREAL];	break;
				case 16: yyval.lval = typesize[TYDCOMPLEX]; break;
				default:
					dclerr("invalid length",NPNULL);
					yyval.lval = varleng;
				}
			}
		} break;
case 69:
/* #line	484	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("length specification *(*)"); yyval.lval = -1; } break;
case 70:
/* #line	488	"/n/bopp/v5/dmg/f2c/gram.in" */
{ incomm( yyval.extval = comblock("") , yypt[-0].yyv.namval ); } break;
case 71:
/* #line	490	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.extval = yypt[-1].yyv.extval;  incomm(yypt[-1].yyv.extval, yypt[-0].yyv.namval); } break;
case 72:
/* #line	492	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.extval = yypt[-2].yyv.extval;  incomm(yypt[-2].yyv.extval, yypt[-0].yyv.namval); } break;
case 73:
/* #line	494	"/n/bopp/v5/dmg/f2c/gram.in" */
{ incomm(yypt[-2].yyv.extval, yypt[-0].yyv.namval); } break;
case 74:
/* #line	498	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.extval = comblock(""); } break;
case 75:
/* #line	500	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.extval = comblock(token); } break;
case 76:
/* #line	504	"/n/bopp/v5/dmg/f2c/gram.in" */
{ setext(yypt[-0].yyv.namval); } break;
case 77:
/* #line	506	"/n/bopp/v5/dmg/f2c/gram.in" */
{ setext(yypt[-0].yyv.namval); } break;
case 78:
/* #line	510	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("INTRINSIC statement"); setintr(yypt[-0].yyv.namval); } break;
case 79:
/* #line	512	"/n/bopp/v5/dmg/f2c/gram.in" */
{ setintr(yypt[-0].yyv.namval); } break;
case 82:
/* #line	520	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		struct Equivblock *p;
		if(nequiv >= maxequiv)
			many("equivalences", 'q', maxequiv);
		p  =  & eqvclass[nequiv++];
		p->eqvinit = NO;
		p->eqvbottom = 0;
		p->eqvtop = 0;
		p->equivs = yypt[-1].yyv.eqvval;
		} break;
case 83:
/* #line	533	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.eqvval=ALLOC(Eqvchain);
		  yyval.eqvval->eqvitem.eqvlhs = primchk(yypt[-0].yyv.expval);
		} break;
case 84:
/* #line	537	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.eqvval=ALLOC(Eqvchain);
		  yyval.eqvval->eqvitem.eqvlhs = primchk(yypt[-0].yyv.expval);
		  yyval.eqvval->eqvnextp = yypt[-2].yyv.eqvval;
		} break;
case 87:
/* #line	548	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(parstate == OUTSIDE)
			{
			newproc();
			startproc(ESNULL, CLMAIN);
			}
		  if(parstate < INDATA)
			{
			enddcl();
			parstate = INDATA;
			datagripe = 1;
			}
		} break;
case 88:
/* #line	563	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ftnint junk;
		  if(nextdata(&junk) != NULL)
			err("too few initializers");
		  frdata(yypt[-4].yyv.chval);
		  frrpl();
		} break;
case 89:
/* #line	571	"/n/bopp/v5/dmg/f2c/gram.in" */
{ frchain(&datastack); curdtp = 0; } break;
case 90:
/* #line	573	"/n/bopp/v5/dmg/f2c/gram.in" */
{ pop_datastack(); } break;
case 91:
/* #line	575	"/n/bopp/v5/dmg/f2c/gram.in" */
{ toomanyinit = NO; } break;
case 94:
/* #line	580	"/n/bopp/v5/dmg/f2c/gram.in" */
{ dataval(ENULL, yypt[-0].yyv.expval); } break;
case 95:
/* #line	582	"/n/bopp/v5/dmg/f2c/gram.in" */
{ dataval(yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 97:
/* #line	587	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if( yypt[-1].yyv.ival==OPMINUS && ISCONST(yypt[-0].yyv.expval) )
			consnegop((Constp)yypt[-0].yyv.expval);
		  yyval.expval = yypt[-0].yyv.expval;
		} break;
case 101:
/* #line	599	"/n/bopp/v5/dmg/f2c/gram.in" */
{ int k;
		  yypt[-0].yyv.namval->vsave = YES;
		  k = yypt[-0].yyv.namval->vstg;
		if( ! ONEOF(k, M(STGUNKNOWN)|M(STGBSS)|M(STGINIT)) )
			dclerr("can only save static variables", yypt[-0].yyv.namval);
		} break;
case 105:
/* #line	613	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(yypt[-2].yyv.namval->vclass == CLUNKNOWN)
			make_param((struct Paramblock *)yypt[-2].yyv.namval, yypt[-0].yyv.expval);
		  else dclerr("cannot make into parameter", yypt[-2].yyv.namval);
		} break;
case 106:
/* #line	620	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(ndim>0) setbound(yypt[-1].yyv.namval, ndim, dims); } break;
case 107:
/* #line	624	"/n/bopp/v5/dmg/f2c/gram.in" */
{ Namep np;
		  struct Primblock *pp = (struct Primblock *)yypt[-0].yyv.expval;
		  int tt = yypt[-0].yyv.expval->tag;
		  if (tt != TPRIM) {
			if (tt == TCONST)
				err("parameter in data statement");
			else
				erri("tag %d in data statement",tt);
			yyval.chval = 0;
			err_lineno = lineno;
			break;
			}
		  np = pp -> namep;
		  vardcl(np);
		  if ((pp->fcharp || pp->lcharp)
		   && (np->vtype != TYCHAR || np->vdim && !pp->argsp))
			sserr(np);
		  if(np->vstg == STGCOMMON)
			extsymtab[np->vardesc.varno].extinit = YES;
		  else if(np->vstg==STGEQUIV)
			eqvclass[np->vardesc.varno].eqvinit = YES;
		  else if(np->vstg!=STGINIT && np->vstg!=STGBSS) {
			errstr(np->vstg == STGARG
				? "Dummy argument \"%.60s\" in data statement."
				: "Cannot give data to \"%.75s\"",
				np->fvarname);
			yyval.chval = 0;
			err_lineno = lineno;
			break;
			}
		  yyval.chval = mkchain((char *)yypt[-0].yyv.expval, CHNULL);
		} break;
case 108:
/* #line	657	"/n/bopp/v5/dmg/f2c/gram.in" */
{ chainp p; struct Impldoblock *q;
		pop_datastack();
		q = ALLOC(Impldoblock);
		q->tag = TIMPLDO;
		(q->varnp = (Namep) (yypt[-1].yyv.chval->datap))->vimpldovar = 1;
		p = yypt[-1].yyv.chval->nextp;
		if(p)  { q->implb = (expptr)(p->datap); p = p->nextp; }
		if(p)  { q->impub = (expptr)(p->datap); p = p->nextp; }
		if(p)  { q->impstep = (expptr)(p->datap); }
		frchain( & (yypt[-1].yyv.chval) );
		yyval.chval = mkchain((char *)q, CHNULL);
		q->datalist = hookup(yypt[-3].yyv.chval, yyval.chval);
		} break;
case 109:
/* #line	673	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if (!datastack)
			curdtp = 0;
		  datastack = mkchain((char *)curdtp, datastack);
		  curdtp = yypt[-0].yyv.chval; curdtelt = 0;
		  } break;
case 110:
/* #line	679	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = hookup(yypt[-2].yyv.chval, yypt[-0].yyv.chval); } break;
case 111:
/* #line	683	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ndim = 0; } break;
case 113:
/* #line	687	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ndim = 0; } break;
case 116:
/* #line	692	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		  if(ndim == maxdim)
			err("too many dimensions");
		  else if(ndim < maxdim)
			{ dims[ndim].lb = 0;
			  dims[ndim].ub = yypt[-0].yyv.expval;
			}
		  ++ndim;
		} break;
case 117:
/* #line	702	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		  if(ndim == maxdim)
			err("too many dimensions");
		  else if(ndim < maxdim)
			{ dims[ndim].lb = yypt[-2].yyv.expval;
			  dims[ndim].ub = yypt[-0].yyv.expval;
			}
		  ++ndim;
		} break;
case 118:
/* #line	714	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = 0; } break;
case 120:
/* #line	719	"/n/bopp/v5/dmg/f2c/gram.in" */
{ nstars = 1; labarray[0] = yypt[-0].yyv.labval; } break;
case 121:
/* #line	721	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(nstars < maxlablist)  labarray[nstars++] = yypt[-0].yyv.labval; } break;
case 122:
/* #line	725	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.labval = execlab( convci(toklen, token) ); } break;
case 123:
/* #line	729	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("IMPLICIT statement"); } break;
case 126:
/* #line	735	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if (vartype != TYUNKNOWN)
			dclerr("-- expected letter range",NPNULL);
		  setimpl(vartype, varleng, 'a', 'z'); } break;
case 127:
/* #line	740	"/n/bopp/v5/dmg/f2c/gram.in" */
{ needkwd = 1; } break;
case 131:
/* #line	749	"/n/bopp/v5/dmg/f2c/gram.in" */
{ setimpl(vartype, varleng, yypt[-0].yyv.ival, yypt[-0].yyv.ival); } break;
case 132:
/* #line	751	"/n/bopp/v5/dmg/f2c/gram.in" */
{ setimpl(vartype, varleng, yypt[-2].yyv.ival, yypt[-0].yyv.ival); } break;
case 133:
/* #line	755	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(toklen!=1 || token[0]<'a' || token[0]>'z')
			{
			dclerr("implicit item must be single letter", NPNULL);
			yyval.ival = 0;
			}
		  else yyval.ival = token[0];
		} break;
case 136:
/* #line	769	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		if(yypt[-2].yyv.namval->vclass == CLUNKNOWN)
			{
			yypt[-2].yyv.namval->vclass = CLNAMELIST;
			yypt[-2].yyv.namval->vtype = TYINT;
			yypt[-2].yyv.namval->vstg = STGBSS;
			yypt[-2].yyv.namval->varxptr.namelist = yypt[-0].yyv.chval;
			yypt[-2].yyv.namval->vardesc.varno = ++lastvarno;
			}
		else dclerr("cannot be a namelist name", yypt[-2].yyv.namval);
		} break;
case 137:
/* #line	783	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.namval, CHNULL); } break;
case 138:
/* #line	785	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = hookup(yypt[-2].yyv.chval, mkchain((char *)yypt[-0].yyv.namval, CHNULL)); } break;
case 139:
/* #line	789	"/n/bopp/v5/dmg/f2c/gram.in" */
{ switch(parstate)
			{
			case OUTSIDE:	newproc();
					startproc(ESNULL, CLMAIN);
			case INSIDE:	parstate = INDCL;
			case INDCL:	break;

			case INDATA:
				if (datagripe) {
					errstr(
				"Statement order error: declaration after DATA",
						CNULL);
					datagripe = 0;
					}
				break;

			default:
				dclerr("declaration among executables", NPNULL);
			}
		} break;
case 140:
/* #line	811	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = 0; } break;
case 141:
/* #line	813	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = revchain(yypt[-0].yyv.chval); } break;
case 142:
/* #line	817	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.expval, CHNULL); } break;
case 143:
/* #line	819	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.expval, yypt[-2].yyv.chval); } break;
case 145:
/* #line	824	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = yypt[-1].yyv.expval; if (yyval.expval->tag == TPRIM)
					paren_used(&yyval.expval->primblock); } break;
case 149:
/* #line	832	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(yypt[-1].yyv.ival, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 150:
/* #line	834	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPSTAR, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 151:
/* #line	836	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPSLASH, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 152:
/* #line	838	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPPOWER, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 153:
/* #line	840	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(yypt[-1].yyv.ival == OPMINUS)
			yyval.expval = mkexpr(OPNEG, yypt[-0].yyv.expval, ENULL);
		  else {
			yyval.expval = yypt[-0].yyv.expval;
			if (yyval.expval->tag == TPRIM)
				paren_used(&yyval.expval->primblock);
			}
		} break;
case 154:
/* #line	849	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(yypt[-1].yyv.ival, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 155:
/* #line	851	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66(".EQV. operator");
		  yyval.expval = mkexpr(OPEQV, yypt[-2].yyv.expval,yypt[-0].yyv.expval); } break;
case 156:
/* #line	854	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66(".NEQV. operator");
		  yyval.expval = mkexpr(OPNEQV, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 157:
/* #line	857	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPOR, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 158:
/* #line	859	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPAND, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 159:
/* #line	861	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPNOT, yypt[-0].yyv.expval, ENULL); } break;
case 160:
/* #line	863	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("concatenation operator //");
		  yyval.expval = mkexpr(OPCONCAT, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 161:
/* #line	867	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = OPPLUS; } break;
case 162:
/* #line	868	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = OPMINUS; } break;
case 163:
/* #line	871	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = OPEQ; } break;
case 164:
/* #line	872	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = OPGT; } break;
case 165:
/* #line	873	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = OPLT; } break;
case 166:
/* #line	874	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = OPGE; } break;
case 167:
/* #line	875	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = OPLE; } break;
case 168:
/* #line	876	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = OPNE; } break;
case 169:
/* #line	880	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkprim(yypt[-0].yyv.namval, LBNULL, CHNULL); } break;
case 170:
/* #line	882	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("substring operator :");
		  yyval.expval = mkprim(yypt[-1].yyv.namval, LBNULL, yypt[-0].yyv.chval); } break;
case 171:
/* #line	885	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkprim(yypt[-3].yyv.namval, mklist(yypt[-1].yyv.chval), CHNULL); } break;
case 172:
/* #line	887	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("substring operator :");
		  yyval.expval = mkprim(yypt[-4].yyv.namval, mklist(yypt[-2].yyv.chval), yypt[-0].yyv.chval); } break;
case 173:
/* #line	892	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-3].yyv.expval, mkchain((char *)yypt[-1].yyv.expval,CHNULL)); } break;
case 174:
/* #line	896	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = 0; } break;
case 176:
/* #line	901	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(yypt[-0].yyv.namval->vclass == CLPARAM)
			yyval.expval = (expptr) cpexpr(
				( (struct Paramblock *) (yypt[-0].yyv.namval) ) -> paramval);
		} break;
case 178:
/* #line	908	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mklogcon(1); } break;
case 179:
/* #line	909	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mklogcon(0); } break;
case 180:
/* #line	910	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkstrcon(toklen, token); } break;
case 181:
/* #line	911	"/n/bopp/v5/dmg/f2c/gram.in" */
 { yyval.expval = mkintqcon(toklen, token); } break;
case 182:
/* #line	912	"/n/bopp/v5/dmg/f2c/gram.in" */
 { yyval.expval = mkrealcon(tyreal, token); } break;
case 183:
/* #line	913	"/n/bopp/v5/dmg/f2c/gram.in" */
 { yyval.expval = mkrealcon(TYDREAL, token); } break;
case 185:
/* #line	918	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkcxcon(yypt[-3].yyv.expval,yypt[-1].yyv.expval); } break;
case 186:
/* #line	922	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NOEXT("hex constant");
		  yyval.expval = mkbitcon(4, toklen, token); } break;
case 187:
/* #line	925	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NOEXT("octal constant");
		  yyval.expval = mkbitcon(3, toklen, token); } break;
case 188:
/* #line	928	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NOEXT("binary constant");
		  yyval.expval = mkbitcon(1, toklen, token); } break;
case 190:
/* #line	934	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = yypt[-1].yyv.expval; } break;
case 193:
/* #line	940	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(yypt[-1].yyv.ival, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 194:
/* #line	942	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPSTAR, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 195:
/* #line	944	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPSLASH, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 196:
/* #line	946	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.expval = mkexpr(OPPOWER, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 197:
/* #line	948	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(yypt[-1].yyv.ival == OPMINUS)
			yyval.expval = mkexpr(OPNEG, yypt[-0].yyv.expval, ENULL);
		  else yyval.expval = yypt[-0].yyv.expval;
		} break;
case 198:
/* #line	953	"/n/bopp/v5/dmg/f2c/gram.in" */
{ NO66("concatenation operator //");
		  yyval.expval = mkexpr(OPCONCAT, yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 200:
/* #line	958	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		if(yypt[-2].yyv.labval->labdefined)
			execerr("no backward DO loops", CNULL);
		yypt[-2].yyv.labval->blklevel = blklevel+1;
		exdo(yypt[-2].yyv.labval->labelno, NPNULL, yypt[-0].yyv.chval);
		} break;
case 201:
/* #line	965	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		exdo((int)(ctls - ctlstack - 2), NPNULL, yypt[-0].yyv.chval);
		NOEXT("DO without label");
		} break;
case 202:
/* #line	970	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exenddo(NPNULL); } break;
case 203:
/* #line	972	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exendif();  thiswasbranch = NO; } break;
case 205:
/* #line	974	"/n/bopp/v5/dmg/f2c/gram.in" */
{westart(1);} break;
case 206:
/* #line	975	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exelif(yypt[-2].yyv.expval); lastwasbranch = NO; } break;
case 207:
/* #line	977	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exelse(); lastwasbranch = NO; } break;
case 208:
/* #line	979	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exendif(); lastwasbranch = NO; } break;
case 209:
/* #line	983	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exif(yypt[-1].yyv.expval); } break;
case 210:
/* #line	987	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-2].yyv.namval, yypt[-0].yyv.chval); } break;
case 212:
/* #line	991	"/n/bopp/v5/dmg/f2c/gram.in" */
{westart(0);} break;
case 213:
/* #line	992	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain(CNULL, (chainp)yypt[-1].yyv.expval); } break;
case 214:
/* #line	996	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exequals((struct Primblock *)yypt[-2].yyv.expval, yypt[-0].yyv.expval); } break;
case 215:
/* #line	998	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exassign(yypt[-0].yyv.namval, yypt[-2].yyv.labval); } break;
case 218:
/* #line	1002	"/n/bopp/v5/dmg/f2c/gram.in" */
{ inioctl = NO; } break;
case 219:
/* #line	1004	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exarif(yypt[-6].yyv.expval, yypt[-4].yyv.labval, yypt[-2].yyv.labval, yypt[-0].yyv.labval);  thiswasbranch = YES; } break;
case 220:
/* #line	1006	"/n/bopp/v5/dmg/f2c/gram.in" */
{ excall(yypt[-0].yyv.namval, LBNULL, 0, labarray); } break;
case 221:
/* #line	1008	"/n/bopp/v5/dmg/f2c/gram.in" */
{ excall(yypt[-2].yyv.namval, LBNULL, 0, labarray); } break;
case 222:
/* #line	1010	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(nstars < maxlablist)
			excall(yypt[-3].yyv.namval, mklist(revchain(yypt[-1].yyv.chval)), nstars, labarray);
		  else
			many("alternate returns", 'l', maxlablist);
		} break;
case 223:
/* #line	1016	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exreturn(yypt[-0].yyv.expval);  thiswasbranch = YES; } break;
case 224:
/* #line	1018	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exstop(yypt[-2].yyv.ival, yypt[-0].yyv.expval);  thiswasbranch = yypt[-2].yyv.ival; } break;
case 225:
/* #line	1022	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.labval = mklabel( convci(toklen, token) ); } break;
case 226:
/* #line	1026	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(parstate == OUTSIDE)
			{
			newproc();
			startproc(ESNULL, CLMAIN);
			}
		} break;
case 227:
/* #line	1035	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exgoto(yypt[-0].yyv.labval);  thiswasbranch = YES; } break;
case 228:
/* #line	1037	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exasgoto(yypt[-0].yyv.namval);  thiswasbranch = YES; } break;
case 229:
/* #line	1039	"/n/bopp/v5/dmg/f2c/gram.in" */
{ exasgoto(yypt[-4].yyv.namval);  thiswasbranch = YES; } break;
case 230:
/* #line	1041	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(nstars < maxlablist)
			putcmgo(putx(fixtype(yypt[-0].yyv.expval)), nstars, labarray);
		  else
			many("labels in computed GOTO list", 'l', maxlablist);
		} break;
case 233:
/* #line	1053	"/n/bopp/v5/dmg/f2c/gram.in" */
{ nstars = 0; yyval.namval = yypt[-0].yyv.namval; } break;
case 234:
/* #line	1057	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = yypt[-0].yyv.expval ? mkchain((char *)yypt[-0].yyv.expval,CHNULL) : CHNULL; } break;
case 235:
/* #line	1059	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = yypt[-0].yyv.expval ? mkchain((char *)yypt[-0].yyv.expval, yypt[-2].yyv.chval) : yypt[-2].yyv.chval; } break;
case 237:
/* #line	1064	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(nstars < maxlablist) labarray[nstars++] = yypt[-0].yyv.labval; yyval.expval = 0; } break;
case 238:
/* #line	1068	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = 0; } break;
case 239:
/* #line	1070	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = 2; } break;
case 240:
/* #line	1074	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.expval, CHNULL); } break;
case 241:
/* #line	1076	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = hookup(yypt[-2].yyv.chval, mkchain((char *)yypt[-0].yyv.expval,CHNULL) ); } break;
case 242:
/* #line	1080	"/n/bopp/v5/dmg/f2c/gram.in" */
{ if(parstate == OUTSIDE)
			{
			newproc();
			startproc(ESNULL, CLMAIN);
			}

/* This next statement depends on the ordering of the state table encoding */

		  if(parstate < INDATA) enddcl();
		} break;
case 243:
/* #line	1093	"/n/bopp/v5/dmg/f2c/gram.in" */
{ intonly = YES; } break;
case 244:
/* #line	1097	"/n/bopp/v5/dmg/f2c/gram.in" */
{ intonly = NO; } break;
case 245:
/* #line	1102	"/n/bopp/v5/dmg/f2c/gram.in" */
{ endio(); } break;
case 247:
/* #line	1107	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(IOSUNIT, yypt[-0].yyv.expval); endioctl(); } break;
case 248:
/* #line	1109	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(IOSUNIT, ENULL); endioctl(); } break;
case 249:
/* #line	1111	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(IOSUNIT, IOSTDERR); endioctl(); } break;
case 251:
/* #line	1114	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(CHNULL); } break;
case 252:
/* #line	1116	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(CHNULL); } break;
case 253:
/* #line	1118	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(revchain(yypt[-0].yyv.chval)); } break;
case 254:
/* #line	1120	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(revchain(yypt[-0].yyv.chval)); } break;
case 255:
/* #line	1122	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(revchain(yypt[-0].yyv.chval)); } break;
case 256:
/* #line	1124	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(CHNULL); } break;
case 257:
/* #line	1126	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(revchain(yypt[-0].yyv.chval)); } break;
case 258:
/* #line	1128	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(revchain(yypt[-0].yyv.chval)); } break;
case 259:
/* #line	1130	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(CHNULL); } break;
case 260:
/* #line	1132	"/n/bopp/v5/dmg/f2c/gram.in" */
{ doio(revchain(yypt[-0].yyv.chval)); } break;
case 262:
/* #line	1139	"/n/bopp/v5/dmg/f2c/gram.in" */
{ iostmt = IOBACKSPACE; } break;
case 263:
/* #line	1141	"/n/bopp/v5/dmg/f2c/gram.in" */
{ iostmt = IOREWIND; } break;
case 264:
/* #line	1143	"/n/bopp/v5/dmg/f2c/gram.in" */
{ iostmt = IOENDFILE; } break;
case 266:
/* #line	1150	"/n/bopp/v5/dmg/f2c/gram.in" */
{ iostmt = IOINQUIRE; } break;
case 267:
/* #line	1152	"/n/bopp/v5/dmg/f2c/gram.in" */
{ iostmt = IOOPEN; } break;
case 268:
/* #line	1154	"/n/bopp/v5/dmg/f2c/gram.in" */
{ iostmt = IOCLOSE; } break;
case 269:
/* #line	1158	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		ioclause(IOSUNIT, ENULL);
		ioclause(IOSFMT, yypt[-0].yyv.expval);
		endioctl();
		} break;
case 270:
/* #line	1164	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		ioclause(IOSUNIT, ENULL);
		ioclause(IOSFMT, ENULL);
		endioctl();
		} break;
case 271:
/* #line	1172	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		  ioclause(IOSUNIT, yypt[-1].yyv.expval);
		  endioctl();
		} break;
case 272:
/* #line	1177	"/n/bopp/v5/dmg/f2c/gram.in" */
{ endioctl(); } break;
case 275:
/* #line	1185	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(IOSPOSITIONAL, yypt[-0].yyv.expval); } break;
case 276:
/* #line	1187	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(IOSPOSITIONAL, ENULL); } break;
case 277:
/* #line	1189	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(IOSPOSITIONAL, IOSTDERR); } break;
case 278:
/* #line	1191	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(yypt[-1].yyv.ival, yypt[-0].yyv.expval); } break;
case 279:
/* #line	1193	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(yypt[-1].yyv.ival, ENULL); } break;
case 280:
/* #line	1195	"/n/bopp/v5/dmg/f2c/gram.in" */
{ ioclause(yypt[-1].yyv.ival, IOSTDERR); } break;
case 281:
/* #line	1199	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.ival = iocname(); } break;
case 282:
/* #line	1203	"/n/bopp/v5/dmg/f2c/gram.in" */
{ iostmt = IOREAD; } break;
case 283:
/* #line	1207	"/n/bopp/v5/dmg/f2c/gram.in" */
{ iostmt = IOWRITE; } break;
case 284:
/* #line	1211	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		iostmt = IOWRITE;
		ioclause(IOSUNIT, ENULL);
		ioclause(IOSFMT, yypt[-1].yyv.expval);
		endioctl();
		} break;
case 285:
/* #line	1218	"/n/bopp/v5/dmg/f2c/gram.in" */
{
		iostmt = IOWRITE;
		ioclause(IOSUNIT, ENULL);
		ioclause(IOSFMT, ENULL);
		endioctl();
		} break;
case 286:
/* #line	1227	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.tagval, CHNULL); } break;
case 287:
/* #line	1229	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.tagval, yypt[-2].yyv.chval); } break;
case 288:
/* #line	1233	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.tagval = (tagptr) yypt[-0].yyv.expval; } break;
case 289:
/* #line	1235	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.tagval = (tagptr) mkiodo(yypt[-1].yyv.chval,revchain(yypt[-3].yyv.chval)); } break;
case 290:
/* #line	1239	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.expval, CHNULL); } break;
case 291:
/* #line	1241	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.tagval, CHNULL); } break;
case 293:
/* #line	1246	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.expval, mkchain((char *)yypt[-2].yyv.expval, CHNULL) ); } break;
case 294:
/* #line	1248	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.tagval, mkchain((char *)yypt[-2].yyv.expval, CHNULL) ); } break;
case 295:
/* #line	1250	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.expval, mkchain((char *)yypt[-2].yyv.tagval, CHNULL) ); } break;
case 296:
/* #line	1252	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.tagval, mkchain((char *)yypt[-2].yyv.tagval, CHNULL) ); } break;
case 297:
/* #line	1254	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.expval, yypt[-2].yyv.chval); } break;
case 298:
/* #line	1256	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.chval = mkchain((char *)yypt[-0].yyv.tagval, yypt[-2].yyv.chval); } break;
case 299:
/* #line	1260	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.tagval = (tagptr) yypt[-0].yyv.expval; } break;
case 300:
/* #line	1262	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.tagval = (tagptr) yypt[-1].yyv.expval; } break;
case 301:
/* #line	1264	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.tagval = (tagptr) mkiodo(yypt[-1].yyv.chval, mkchain((char *)yypt[-3].yyv.expval, CHNULL) ); } break;
case 302:
/* #line	1266	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.tagval = (tagptr) mkiodo(yypt[-1].yyv.chval, mkchain((char *)yypt[-3].yyv.tagval, CHNULL) ); } break;
case 303:
/* #line	1268	"/n/bopp/v5/dmg/f2c/gram.in" */
{ yyval.tagval = (tagptr) mkiodo(yypt[-1].yyv.chval, revchain(yypt[-3].yyv.chval)); } break;
case 304:
/* #line	1272	"/n/bopp/v5/dmg/f2c/gram.in" */
{ startioctl(); } break;
	}
	goto yystack;  /* stack new state and value */
}
