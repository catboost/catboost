#2!/usr/bin/python

import yatest.common
import hashlib
import shutil
import re
import glob
import os

# Based heavily on Makefile.am from upstream

# Helper routines

_TJ_ROOT = "contrib/libs/libjpeg-turbo/"
_TESTIMAGES_ROOT = "contrib/libs/libjpeg-turbo/testimages/"
_TESTIMAGES_RE = re.compile(r'^\$\(srcdir\)/testimages/(.*)$')
_TESTORIG = "testorig.jpg"

def cp_testimage(name, newname):
    shutil.copyfile(yatest.common.source_path(_TESTIMAGES_ROOT + name), newname)

def run(command):
    command = command.split()
    ut = ""
    command[0] = yatest.common.binary_path(_TJ_ROOT + ut + command[0] + "/" + command[0])
    for i in range(1, len(command)):
        m = _TESTIMAGES_RE.match(command[i])
        if m:
            name = m.group(1)
            if name == "$(TESTORIG)":
                name = _TESTORIG
            command[i] = yatest.common.source_path(_TESTIMAGES_ROOT + name)
    yatest.common.execute(command)

def md5cmp(hexdigest, filename):
    with open(filename, 'rb') as f:
        assert(hexdigest == hashlib.md5(f.read()).hexdigest())

def rm_f(patterns):
    for pattern in patterns.split():
        for fn in glob.glob(pattern):
            if "/" in fn:
                raise Exception("/ seems supsicious in a file name to remove")
            os.unlink(fn)

# Makefile.am mimic
MD5_JPEG_RGB_ISLOW = "768e970dd57b340ff1b83c9d3d47c77b"
MD5_PPM_RGB_ISLOW = "00a257f5393fef8821f2b88ac7421291"
MD5_BMP_RGB_ISLOW_565 = "f07d2e75073e4bb10f6c6f4d36e2e3be"
MD5_BMP_RGB_ISLOW_565D = "4cfa0928ef3e6bb626d7728c924cfda4"
MD5_JPEG_422_IFAST_OPT = "2540287b79d913f91665e660303ab2c8"
MD5_PPM_422_IFAST = "35bd6b3f833bad23de82acea847129fa"
MD5_PPM_422M_IFAST = "8dbc65323d62cca7c91ba02dd1cfa81d"
MD5_BMP_422M_IFAST_565 = "3294bd4d9a1f2b3d08ea6020d0db7065"
MD5_BMP_422M_IFAST_565D = "da98c9c7b6039511be4a79a878a9abc1"
MD5_JPEG_420_IFAST_Q100_PROG = "990cbe0329c882420a2094da7e5adade"
MD5_PPM_420_Q100_IFAST = "5a732542015c278ff43635e473a8a294"
MD5_PPM_420M_Q100_IFAST = "ff692ee9323a3b424894862557c092f1"
MD5_JPEG_GRAY_ISLOW = "72b51f894b8f4a10b3ee3066770aa38d"
MD5_PPM_GRAY_ISLOW = "8d3596c56eace32f205deccc229aa5ed"
MD5_PPM_GRAY_ISLOW_RGB = "116424ac07b79e5e801f00508eab48ec"
MD5_BMP_GRAY_ISLOW_565 = "12f78118e56a2f48b966f792fedf23cc"
MD5_BMP_GRAY_ISLOW_565D = "bdbbd616441a24354c98553df5dc82db"
MD5_JPEG_420S_IFAST_OPT = "388708217ac46273ca33086b22827ed8"
# See README.md for more details on why this next bit is necessary.
MD5_JPEG_3x2_FLOAT_PROG_SSE = "343e3f8caf8af5986ebaf0bdc13b5c71"
MD5_PPM_3x2_FLOAT_SSE = "1a75f36e5904d6fc3a85a43da9ad89bb"
MD5_JPEG_3x2_FLOAT_PROG_32BIT = "9bca803d2042bd1eb03819e2bf92b3e5"
MD5_PPM_3x2_FLOAT_32BIT = "f6bfab038438ed8f5522fbd33595dcdc"
MD5_PPM_3x2_FLOAT_64BIT = "0e917a34193ef976b679a6b069b1be26"
MD5_JPEG_3x2_IFAST_PROG = "1ee5d2c1a77f2da495f993c8c7cceca5"
MD5_PPM_3x2_IFAST = "fd283664b3b49127984af0a7f118fccd"
MD5_JPEG_420_ISLOW_ARI = "e986fb0a637a8d833d96e8a6d6d84ea1"
MD5_JPEG_444_ISLOW_PROGARI = "0a8f1c8f66e113c3cf635df0a475a617"
MD5_PPM_420M_IFAST_ARI = "72b59a99bcf1de24c5b27d151bde2437"
MD5_JPEG_420_ISLOW = "9a68f56bc76e466aa7e52f415d0f4a5f"
MD5_PPM_420M_ISLOW_2_1 = "9f9de8c0612f8d06869b960b05abf9c9"
MD5_PPM_420M_ISLOW_15_8 = "b6875bc070720b899566cc06459b63b7"
MD5_PPM_420M_ISLOW_13_8 = "bc3452573c8152f6ae552939ee19f82f"
MD5_PPM_420M_ISLOW_11_8 = "d8cc73c0aaacd4556569b59437ba00a5"
MD5_PPM_420M_ISLOW_9_8 = "d25e61bc7eac0002f5b393aa223747b6"
MD5_PPM_420M_ISLOW_7_8 = "ddb564b7c74a09494016d6cd7502a946"
MD5_PPM_420M_ISLOW_3_4 = "8ed8e68808c3fbc4ea764fc9d2968646"
MD5_PPM_420M_ISLOW_5_8 = "a3363274999da2366a024efae6d16c9b"
MD5_PPM_420M_ISLOW_1_2 = "e692a315cea26b988c8e8b29a5dbcd81"
MD5_PPM_420M_ISLOW_3_8 = "79eca9175652ced755155c90e785a996"
MD5_PPM_420M_ISLOW_1_4 = "79cd778f8bf1a117690052cacdd54eca"
MD5_PPM_420M_ISLOW_1_8 = "391b3d4aca640c8567d6f8745eb2142f"
MD5_BMP_420_ISLOW_256 = "4980185e3776e89bd931736e1cddeee6"
MD5_BMP_420_ISLOW_565 = "bf9d13e16c4923b92e1faa604d7922cb"
MD5_BMP_420_ISLOW_565D = "6bde71526acc44bcff76f696df8638d2"
MD5_BMP_420M_ISLOW_565 = "8dc0185245353cfa32ad97027342216f"
MD5_BMP_420M_ISLOW_565D = "ce034037d212bc403330df6f915c161b"
MD5_PPM_420_ISLOW_SKIP15_31 = "c4c65c1e43d7275cd50328a61e6534f0"
MD5_PPM_420_ISLOW_ARI_SKIP16_139 = "087c6b123db16ac00cb88c5b590bb74a"
MD5_PPM_420_ISLOW_PROG_CROP62x62_71_71 = "26eb36ccc7d1f0cb80cdabb0ac8b5d99"
MD5_PPM_420_ISLOW_ARI_CROP53x53_4_4 = "886c6775af22370257122f8b16207e6d"
MD5_PPM_444_ISLOW_SKIP1_6 = "5606f86874cf26b8fcee1117a0a436a6"
MD5_PPM_444_ISLOW_PROG_CROP98x98_13_13 = "db87dc7ce26bcdc7a6b56239ce2b9d6c"
MD5_PPM_444_ISLOW_ARI_CROP37x37_0_0 = "cb57b32bd6d03e35432362f7bf184b6d"
MD5_JPEG_CROP = "b4197f377e621c4e9b1d20471432610d"


def test_bittest():

# These tests are carefully crafted to provide full coverage of as many of the
# underlying algorithms as possible (including all of the SIMD-accelerated
# ones.)

# CC: null  SAMP: fullsize  FDCT: islow  ENT: huff
    run("cjpeg -rgb -dct int -outfile testout_rgb_islow.jpg $(srcdir)/testimages/testorig.ppm")
    md5cmp(MD5_JPEG_RGB_ISLOW, "testout_rgb_islow.jpg")
# CC: null  SAMP: fullsize  IDCT: islow  ENT: huff
    run("djpeg -dct int -ppm -outfile testout_rgb_islow.ppm testout_rgb_islow.jpg")
    md5cmp(MD5_PPM_RGB_ISLOW, "testout_rgb_islow.ppm")
    rm_f("testout_rgb_islow.ppm")
#if WITH_12BIT
#   rm_f("testout_rgb_islow.jpg")
#else
# CC: RGB->RGB565  SAMP: fullsize  IDCT: islow  ENT: huff
    run("djpeg -dct int -rgb565 -dither none -bmp -outfile testout_rgb_islow_565.bmp testout_rgb_islow.jpg")
    md5cmp(MD5_BMP_RGB_ISLOW_565, "testout_rgb_islow_565.bmp")
    rm_f("testout_rgb_islow_565.bmp")
# CC: RGB->RGB565 (dithered)  SAMP: fullsize  IDCT: islow  ENT: huff
    run("djpeg -dct int -rgb565 -bmp -outfile testout_rgb_islow_565D.bmp testout_rgb_islow.jpg")
    md5cmp(MD5_BMP_RGB_ISLOW_565D, "testout_rgb_islow_565D.bmp")
    rm_f("testout_rgb_islow_565D.bmp testout_rgb_islow.jpg")
#endif

# CC: RGB->YCC  SAMP: fullsize/h2v1  FDCT: ifast  ENT: 2-pass huff
    run("cjpeg -sample 2x1 -dct fast -opt -outfile testout_422_ifast_opt.jpg $(srcdir)/testimages/testorig.ppm")
    md5cmp(MD5_JPEG_422_IFAST_OPT, "testout_422_ifast_opt.jpg")
# CC: YCC->RGB  SAMP: fullsize/h2v1 fancy  IDCT: ifast  ENT: huff
    run("djpeg -dct fast -outfile testout_422_ifast.ppm testout_422_ifast_opt.jpg")
    md5cmp(MD5_PPM_422_IFAST, "testout_422_ifast.ppm")
    rm_f("testout_422_ifast.ppm")
# CC: YCC->RGB  SAMP: h2v1 merged  IDCT: ifast  ENT: huff
    run("djpeg -dct fast -nosmooth -outfile testout_422m_ifast.ppm testout_422_ifast_opt.jpg")
    md5cmp(MD5_PPM_422M_IFAST, "testout_422m_ifast.ppm")
    rm_f("testout_422m_ifast.ppm")
#if WITH_12BIT
#   rm_f("testout_422_ifast_opt.jpg")
#else
# CC: YCC->RGB565  SAMP: h2v1 merged  IDCT: ifast  ENT: huff
    run("djpeg -dct int -nosmooth -rgb565 -dither none -bmp -outfile testout_422m_ifast_565.bmp testout_422_ifast_opt.jpg")
    md5cmp(MD5_BMP_422M_IFAST_565, "testout_422m_ifast_565.bmp")
    rm_f("testout_422m_ifast_565.bmp")
# CC: YCC->RGB565 (dithered)  SAMP: h2v1 merged  IDCT: ifast  ENT: huff
    run("djpeg -dct int -nosmooth -rgb565 -bmp -outfile testout_422m_ifast_565D.bmp testout_422_ifast_opt.jpg")
    md5cmp(MD5_BMP_422M_IFAST_565D, "testout_422m_ifast_565D.bmp")
    rm_f("testout_422m_ifast_565D.bmp testout_422_ifast_opt.jpg")
#endif

# CC: RGB->YCC  SAMP: fullsize/h2v2  FDCT: ifast  ENT: prog huff
    run("cjpeg -sample 2x2 -quality 100 -dct fast -prog -outfile testout_420_q100_ifast_prog.jpg $(srcdir)/testimages/testorig.ppm")
    md5cmp(MD5_JPEG_420_IFAST_Q100_PROG, "testout_420_q100_ifast_prog.jpg")
# CC: YCC->RGB  SAMP: fullsize/h2v2 fancy  IDCT: ifast  ENT: prog huff
    run("djpeg -dct fast -outfile testout_420_q100_ifast.ppm testout_420_q100_ifast_prog.jpg")
    md5cmp(MD5_PPM_420_Q100_IFAST, "testout_420_q100_ifast.ppm")
    rm_f("testout_420_q100_ifast.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: ifast  ENT: prog huff
    run("djpeg -dct fast -nosmooth -outfile testout_420m_q100_ifast.ppm testout_420_q100_ifast_prog.jpg")
    md5cmp(MD5_PPM_420M_Q100_IFAST, "testout_420m_q100_ifast.ppm")
    rm_f("testout_420m_q100_ifast.ppm testout_420_q100_ifast_prog.jpg")

# CC: RGB->Gray  SAMP: fullsize  FDCT: islow  ENT: huff
    run("cjpeg -gray -dct int -outfile testout_gray_islow.jpg $(srcdir)/testimages/testorig.ppm")
    md5cmp(MD5_JPEG_GRAY_ISLOW, "testout_gray_islow.jpg")
# CC: Gray->Gray  SAMP: fullsize  IDCT: islow  ENT: huff
    run("djpeg -dct int -outfile testout_gray_islow.ppm testout_gray_islow.jpg")
    md5cmp(MD5_PPM_GRAY_ISLOW, "testout_gray_islow.ppm")
    rm_f("testout_gray_islow.ppm")
# CC: Gray->RGB  SAMP: fullsize  IDCT: islow  ENT: huff
    run("djpeg -dct int -rgb -outfile testout_gray_islow_rgb.ppm testout_gray_islow.jpg")
    md5cmp(MD5_PPM_GRAY_ISLOW_RGB, "testout_gray_islow_rgb.ppm")
    rm_f("testout_gray_islow_rgb.ppm")
#if WITH_12BIT
#   rm_f("testout_gray_islow.jpg")
#else
# CC: Gray->RGB565  SAMP: fullsize  IDCT: islow  ENT: huff
    run("djpeg -dct int -rgb565 -dither none -bmp -outfile testout_gray_islow_565.bmp testout_gray_islow.jpg")
    md5cmp(MD5_BMP_GRAY_ISLOW_565, "testout_gray_islow_565.bmp")
    rm_f("testout_gray_islow_565.bmp")
# CC: Gray->RGB565 (dithered)  SAMP: fullsize  IDCT: islow  ENT: huff
    run("djpeg -dct int -rgb565 -bmp -outfile testout_gray_islow_565D.bmp testout_gray_islow.jpg")
    md5cmp(MD5_BMP_GRAY_ISLOW_565D, "testout_gray_islow_565D.bmp")
    rm_f("testout_gray_islow_565D.bmp testout_gray_islow.jpg")
#endif

# CC: RGB->YCC  SAMP: fullsize smooth/h2v2 smooth  FDCT: islow
# ENT: 2-pass huff
    run("cjpeg -sample 2x2 -smooth 1 -dct int -opt -outfile testout_420s_ifast_opt.jpg $(srcdir)/testimages/testorig.ppm")
    md5cmp(MD5_JPEG_420S_IFAST_OPT, "testout_420s_ifast_opt.jpg")
    rm_f("testout_420s_ifast_opt.jpg")

# The output of the floating point tests is not validated by default, because
# the output differs depending on the type of floating point math used, and
# this is only deterministic if the DCT/IDCT are implemented using SIMD
# instructions on a particular platform.  Pass one of the following on the make
# command line to validate the floating point tests against one of the expected
# results:
#
# FLOATTEST=sse  validate against the expected results from the libjpeg-turbo
#                SSE SIMD extensions
# FLOATTEST=32bit  validate against the expected results from the C code
#                  when running on a 32-bit FPU (or when SSE is being used for
#                  floating point math, which is generally the default with
#                  x86-64 compilers)
# FLOATTEST=64bit  validate against the exepected results from the C code
#                  when running on a 64-bit FPU

# CC: RGB->YCC  SAMP: fullsize/int  FDCT: float  ENT: prog huff
#    run("cjpeg -sample 3x2 -dct float -prog -outfile testout_3x2_float_prog.jpg $(srcdir)/testimages/testorig.ppm")
#    if [ "${FLOATTEST}" = "sse" ]; then \
#        md5cmp(MD5_JPEG_3x2_FLOAT_PROG_SSE, "testout_3x2_float_prog.jpg"); \
#    elif [ "${FLOATTEST}" = "32bit" -o "${FLOATTEST}" = "64bit" ]; then \
#        md5cmp(MD5_JPEG_3x2_FLOAT_PROG_32BIT, "testout_3x2_float_prog.jpg"); \
#    fi
# CC: YCC->RGB  SAMP: fullsize/int  IDCT: float  ENT: prog huff
#    run("djpeg -dct float -outfile testout_3x2_float.ppm testout_3x2_float_prog.jpg")
#    if [ "${FLOATTEST}" = "sse" ]; then \
#        md5cmp(MD5_PPM_3x2_FLOAT_SSE, "testout_3x2_float.ppm"); \
#    elif [ "${FLOATTEST}" = "32bit" ]; then \
#        md5cmp(MD5_PPM_3x2_FLOAT_32BIT, "testout_3x2_float.ppm"); \
#    elif [ "${FLOATTEST}" = "64bit" ]; then \
#        md5cmp(MD5_PPM_3x2_FLOAT_64BIT, "testout_3x2_float.ppm"); \
#    fi
#    rm_f("testout_3x2_float.ppm testout_3x2_float_prog.jpg")

# CC: RGB->YCC  SAMP: fullsize/int  FDCT: ifast  ENT: prog huff
    run("cjpeg -sample 3x2 -dct fast -prog -outfile testout_3x2_ifast_prog.jpg $(srcdir)/testimages/testorig.ppm")
    md5cmp(MD5_JPEG_3x2_IFAST_PROG, "testout_3x2_ifast_prog.jpg")
# CC: YCC->RGB  SAMP: fullsize/int  IDCT: ifast  ENT: prog huff
    run("djpeg -dct fast -outfile testout_3x2_ifast.ppm testout_3x2_ifast_prog.jpg")
    md5cmp(MD5_PPM_3x2_IFAST, "testout_3x2_ifast.ppm")
    rm_f("testout_3x2_ifast.ppm testout_3x2_ifast_prog.jpg")

#if WITH_ARITH_ENC
# CC: YCC->RGB  SAMP: fullsize/h2v2  FDCT: islow  ENT: arith
    run("cjpeg -dct int -arithmetic -outfile testout_420_islow_ari.jpg $(srcdir)/testimages/testorig.ppm")
    md5cmp(MD5_JPEG_420_ISLOW_ARI, "testout_420_islow_ari.jpg")
    rm_f("testout_420_islow_ari.jpg")
    run("jpegtran -arithmetic -outfile testout_420_islow_ari.jpg $(srcdir)/testimages/testimgint.jpg")
    md5cmp(MD5_JPEG_420_ISLOW_ARI, "testout_420_islow_ari.jpg")
    rm_f("testout_420_islow_ari.jpg")
# CC: YCC->RGB  SAMP: fullsize  FDCT: islow  ENT: prog arith
    run("cjpeg -sample 1x1 -dct int -prog -arithmetic -outfile testout_444_islow_progari.jpg $(srcdir)/testimages/testorig.ppm")
    md5cmp(MD5_JPEG_444_ISLOW_PROGARI, "testout_444_islow_progari.jpg")
    rm_f("testout_444_islow_progari.jpg")
#endif
#if WITH_ARITH_DEC
# CC: RGB->YCC  SAMP: h2v2 merged  IDCT: ifast  ENT: arith
    run("djpeg -fast -ppm -outfile testout_420m_ifast_ari.ppm $(srcdir)/testimages/testimgari.jpg")
    md5cmp(MD5_PPM_420M_IFAST_ARI, "testout_420m_ifast_ari.ppm")
    rm_f("testout_420m_ifast_ari.ppm")
    run("jpegtran -outfile testout_420_islow.jpg $(srcdir)/testimages/testimgari.jpg")
    md5cmp(MD5_JPEG_420_ISLOW, "testout_420_islow.jpg")
    rm_f("testout_420_islow.jpg")
#endif

# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 16x16 islow  ENT: huff
    run("djpeg -dct int -scale 2/1 -nosmooth -ppm -outfile testout_420m_islow_2_1.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_2_1, "testout_420m_islow_2_1.ppm")
    rm_f("testout_420m_islow_2_1.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 15x15 islow  ENT: huff
    run("djpeg -dct int -scale 15/8 -nosmooth -ppm -outfile testout_420m_islow_15_8.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_15_8, "testout_420m_islow_15_8.ppm")
    rm_f("testout_420m_islow_15_8.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 13x13 islow  ENT: huff
    run("djpeg -dct int -scale 13/8 -nosmooth -ppm -outfile testout_420m_islow_13_8.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_13_8, "testout_420m_islow_13_8.ppm")
    rm_f("testout_420m_islow_13_8.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 11x11 islow  ENT: huff
    run("djpeg -dct int -scale 11/8 -nosmooth -ppm -outfile testout_420m_islow_11_8.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_11_8, "testout_420m_islow_11_8.ppm")
    rm_f("testout_420m_islow_11_8.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 9x9 islow  ENT: huff
    run("djpeg -dct int -scale 9/8 -nosmooth -ppm -outfile testout_420m_islow_9_8.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_9_8, "testout_420m_islow_9_8.ppm")
    rm_f("testout_420m_islow_9_8.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 7x7 islow/14x14 islow  ENT: huff
    run("djpeg -dct int -scale 7/8 -nosmooth -ppm -outfile testout_420m_islow_7_8.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_7_8, "testout_420m_islow_7_8.ppm")
    rm_f("testout_420m_islow_7_8.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 6x6 islow/12x12 islow  ENT: huff
    run("djpeg -dct int -scale 3/4 -nosmooth -ppm -outfile testout_420m_islow_3_4.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_3_4, "testout_420m_islow_3_4.ppm")
    rm_f("testout_420m_islow_3_4.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 5x5 islow/10x10 islow  ENT: huff
    run("djpeg -dct int -scale 5/8 -nosmooth -ppm -outfile testout_420m_islow_5_8.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_5_8, "testout_420m_islow_5_8.ppm")
    rm_f("testout_420m_islow_5_8.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 4x4 islow/8x8 islow  ENT: huff
    run("djpeg -dct int -scale 1/2 -nosmooth -ppm -outfile testout_420m_islow_1_2.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_1_2, "testout_420m_islow_1_2.ppm")
    rm_f("testout_420m_islow_1_2.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 3x3 islow/6x6 islow  ENT: huff
    run("djpeg -dct int -scale 3/8 -nosmooth -ppm -outfile testout_420m_islow_3_8.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_3_8, "testout_420m_islow_3_8.ppm")
    rm_f("testout_420m_islow_3_8.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 2x2 islow/4x4 islow  ENT: huff
    run("djpeg -dct int -scale 1/4 -nosmooth -ppm -outfile testout_420m_islow_1_4.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_1_4, "testout_420m_islow_1_4.ppm")
    rm_f("testout_420m_islow_1_4.ppm")
# CC: YCC->RGB  SAMP: h2v2 merged  IDCT: 1x1 islow/2x2 islow  ENT: huff
    run("djpeg -dct int -scale 1/8 -nosmooth -ppm -outfile testout_420m_islow_1_8.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420M_ISLOW_1_8, "testout_420m_islow_1_8.ppm")
    rm_f("testout_420m_islow_1_8.ppm")
#if WITH_12BIT
#else
# CC: YCC->RGB (dithered)  SAMP: h2v2 fancy  IDCT: islow  ENT: huff
    run("djpeg -dct int -colors 256 -bmp -outfile testout_420_islow_256.bmp $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_BMP_420_ISLOW_256, "testout_420_islow_256.bmp")
    rm_f("testout_420_islow_256.bmp")
# CC: YCC->RGB565  SAMP: h2v2 fancy  IDCT: islow  ENT: huff
    run("djpeg -dct int -rgb565 -dither none -bmp -outfile testout_420_islow_565.bmp $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_BMP_420_ISLOW_565, "testout_420_islow_565.bmp")
    rm_f("testout_420_islow_565.bmp")
# CC: YCC->RGB565 (dithered)  SAMP: h2v2 fancy  IDCT: islow  ENT: huff
    run("djpeg -dct int -rgb565 -bmp -outfile testout_420_islow_565D.bmp $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_BMP_420_ISLOW_565D, "testout_420_islow_565D.bmp")
    rm_f("testout_420_islow_565D.bmp")
# CC: YCC->RGB565  SAMP: h2v2 merged  IDCT: islow  ENT: huff
    run("djpeg -dct int -nosmooth -rgb565 -dither none -bmp -outfile testout_420m_islow_565.bmp $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_BMP_420M_ISLOW_565, "testout_420m_islow_565.bmp")
    rm_f("testout_420m_islow_565.bmp")
# CC: YCC->RGB565 (dithered)  SAMP: h2v2 merged  IDCT: islow  ENT: huff
    run("djpeg -dct int -nosmooth -rgb565 -bmp -outfile testout_420m_islow_565D.bmp $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_BMP_420M_ISLOW_565D, "testout_420m_islow_565D.bmp")
    rm_f("testout_420m_islow_565D.bmp")
#endif

# Partial decode tests.  These tests are designed to cover all of the possible
# code paths in jpeg_skip_scanlines().

# Context rows: Yes  Intra-iMCU row: Yes  iMCU row prefetch: No   ENT: huff
    run("djpeg -dct int -skip 15,31 -ppm -outfile testout_420_islow_skip15,31.ppm $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_PPM_420_ISLOW_SKIP15_31, "testout_420_islow_skip15,31.ppm")
    rm_f("testout_420_islow_skip15,31.ppm")
# Context rows: Yes  Intra-iMCU row: No   iMCU row prefetch: Yes  ENT: arith
#if WITH_ARITH_DEC
    run("djpeg -dct int -skip 16,139 -ppm -outfile testout_420_islow_ari_skip16,139.ppm $(srcdir)/testimages/testimgari.jpg")
    md5cmp(MD5_PPM_420_ISLOW_ARI_SKIP16_139, "testout_420_islow_ari_skip16,139.ppm")
    rm_f("testout_420_islow_ari_skip16,139.ppm")
#endif
# Context rows: Yes  Intra-iMCU row: No   iMCU row prefetch: No   ENT: prog huff
    run("cjpeg -dct int -prog -outfile testout_420_islow_prog.jpg $(srcdir)/testimages/testorig.ppm")
    run("djpeg -dct int -crop 62x62+71+71 -ppm -outfile testout_420_islow_prog_crop62x62,71,71.ppm testout_420_islow_prog.jpg")
    md5cmp(MD5_PPM_420_ISLOW_PROG_CROP62x62_71_71, "testout_420_islow_prog_crop62x62,71,71.ppm")
    rm_f("testout_420_islow_prog_crop62x62,71,71.ppm testout_420_islow_prog.jpg")
# Context rows: Yes  Intra-iMCU row: No   iMCU row prefetch: No   ENT: arith
#if WITH_ARITH_DEC
    run("djpeg -dct int -crop 53x53+4+4 -ppm -outfile testout_420_islow_ari_crop53x53,4,4.ppm $(srcdir)/testimages/testimgari.jpg")
    md5cmp(MD5_PPM_420_ISLOW_ARI_CROP53x53_4_4, "testout_420_islow_ari_crop53x53,4,4.ppm")
    rm_f("testout_420_islow_ari_crop53x53,4,4.ppm")
#endif
# Context rows: No   Intra-iMCU row: Yes  ENT: huff
    run("cjpeg -dct int -sample 1x1 -outfile testout_444_islow.jpg $(srcdir)/testimages/testorig.ppm")
    run("djpeg -dct int -skip 1,6 -ppm -outfile testout_444_islow_skip1,6.ppm testout_444_islow.jpg")
    md5cmp(MD5_PPM_444_ISLOW_SKIP1_6, "testout_444_islow_skip1,6.ppm")
    rm_f("testout_444_islow_skip1,6.ppm testout_444_islow.jpg")
# Context rows: No   Intra-iMCU row: No   ENT: prog huff
    run("cjpeg -dct int -prog -sample 1x1 -outfile testout_444_islow_prog.jpg $(srcdir)/testimages/testorig.ppm")
    run("djpeg -dct int -crop 98x98+13+13 -ppm -outfile testout_444_islow_prog_crop98x98,13,13.ppm testout_444_islow_prog.jpg")
    md5cmp(MD5_PPM_444_ISLOW_PROG_CROP98x98_13_13, "testout_444_islow_prog_crop98x98,13,13.ppm")
    rm_f("testout_444_islow_prog_crop98x98,13,13.ppm testout_444_islow_prog.jpg")
# Context rows: No   Intra-iMCU row: No   ENT: arith
#if WITH_ARITH_ENC
    run("cjpeg -dct int -arithmetic -sample 1x1 -outfile testout_444_islow_ari.jpg $(srcdir)/testimages/testorig.ppm")
#if WITH_ARITH_DEC
    run("djpeg -dct int -crop 37x37+0+0 -ppm -outfile testout_444_islow_ari_crop37x37,0,0.ppm testout_444_islow_ari.jpg")
    md5cmp(MD5_PPM_444_ISLOW_ARI_CROP37x37_0_0, "testout_444_islow_ari_crop37x37,0,0.ppm")
    rm_f("testout_444_islow_ari_crop37x37,0,0.ppm")
#endif
    rm_f("testout_444_islow_ari.jpg")
#endif

    run("jpegtran -crop 120x90+20+50 -transpose -perfect -outfile testout_crop.jpg $(srcdir)/testimages/$(TESTORIG)")
    md5cmp(MD5_JPEG_CROP, "testout_crop.jpg")
    rm_f("testout_crop.jpg")
