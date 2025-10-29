//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_PREPROCESSOR_H
#define __CCCL_PREPROCESSOR_H

// warn when MSVC is used with the traditional preprocessor
#if defined(_MSC_VER) && !defined(__clang__)
#  if (!defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL == 1) \
    && !defined(CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING)
#    pragma message(                                                                                               \
      "MSVC/cl.exe with traditional preprocessor is used. This may lead to unexpected compilation errors. Please " \
      "switch to the standard conforming preprocessor by passing `/Zc:preprocessor` to cl.exe. You can define "    \
      "CCCL_IGNORE_MSVC_TRADITIONAL_PREPROCESSOR_WARNING to suppress this warning.")
#  endif // !defined(_MSVC_TRADITIONAL) || _MSVC_TRADITIONAL == 1
#endif // defined(_MSC_VER) && !defined(__clang__)

#ifdef __has_include
#  define _CCCL_HAS_INCLUDE(_X) __has_include(_X)
#else
#  define _CCCL_HAS_INCLUDE(_X) 0
#endif

#ifdef __COUNTER__
#  define _CCCL_COUNTER() __COUNTER__
#else
#  define _CCCL_COUNTER() __LINE__
#endif

// Convert parameter to string
#define _CCCL_TO_STRING2(_STR) #_STR
#define _CCCL_TO_STRING(_STR)  _CCCL_TO_STRING2(_STR)

#define _CCCL_PP_FIRST(first, ...)      first
#define _CCCL_PP_SECOND(_, second, ...) second
#define _CCCL_PP_THIRD(_1, _2, third)   third

#define _CCCL_PP_EXPAND(...) __VA_ARGS__
#define _CCCL_PP_EAT(...)

#define _CCCL_PP_CAT_(_Xp, ...) _Xp##__VA_ARGS__
#define _CCCL_PP_CAT(_Xp, ...)  _CCCL_PP_CAT_(_Xp, __VA_ARGS__)

#define _CCCL_PP_CAT2_(_Xp, ...) _Xp##__VA_ARGS__
#define _CCCL_PP_CAT2(_Xp, ...)  _CCCL_PP_CAT2_(_Xp, __VA_ARGS__)

#define _CCCL_PP_CAT3_(_Xp, ...) _Xp##__VA_ARGS__
#define _CCCL_PP_CAT3(_Xp, ...)  _CCCL_PP_CAT3_(_Xp, __VA_ARGS__)

#define _CCCL_PP_CAT4_(_Xp, ...) _Xp##__VA_ARGS__
#define _CCCL_PP_CAT4(_Xp, ...)  _CCCL_PP_CAT4_(_Xp, __VA_ARGS__)

#define _CCCL_PP_EVAL_(_Xp, _ARGS) _Xp _ARGS
#define _CCCL_PP_EVAL(_Xp, ...)    _CCCL_PP_EVAL_(_Xp, (__VA_ARGS__))

#define _CCCL_PP_EVAL2_(_Xp, _ARGS) _Xp _ARGS
#define _CCCL_PP_EVAL2(_Xp, ...)    _CCCL_PP_EVAL2_(_Xp, (__VA_ARGS__))

#define _CCCL_PP_CHECK(...)              _CCCL_PP_EXPAND(_CCCL_PP_CHECK_N(__VA_ARGS__, 0, ))
#define _CCCL_PP_CHECK_N(_Xp, _Num, ...) _Num
#define _CCCL_PP_PROBE(_Xp)              _Xp, 1,
#define _CCCL_PP_PROBE_N(_Xp, _Num)      _Xp, _Num,

#define _CCCL_PP_IS_PAREN(_Xp)       _CCCL_PP_CHECK(_CCCL_PP_IS_PAREN_PROBE _Xp)
#define _CCCL_PP_IS_PAREN_PROBE(...) _CCCL_PP_PROBE(~)

#define _CCCL_PP_IIF(_BIT)         _CCCL_PP_CAT_(_CCCL_PP_IIF_, _BIT)
#define _CCCL_PP_IIF_0(_TRUE, ...) __VA_ARGS__
#define _CCCL_PP_IIF_1(_TRUE, ...) _TRUE

#define _CCCL_PP_LPAREN (
#define _CCCL_PP_RPAREN )

#define _CCCL_PP_NOT(_BIT) _CCCL_PP_CAT_(_CCCL_PP_NOT_, _BIT)
#define _CCCL_PP_NOT_0     1
#define _CCCL_PP_NOT_1     0

#define _CCCL_PP_EMPTY()
#define _CCCL_PP_COMMA()        ,
#define _CCCL_PP_LBRACE()       {
#define _CCCL_PP_RBRACE()       }
#define _CCCL_PP_COMMA_IIF(_Xp) _CCCL_PP_IIF(_Xp)(_CCCL_PP_EMPTY, _CCCL_PP_COMMA)() /**/

#define _CCCL_PP_FOR_EACH(_Mp, ...)                          _CCCL_PP_FOR_EACH_N(_CCCL_PP_COUNT(__VA_ARGS__), _Mp, __VA_ARGS__)
#define _CCCL_PP_FOR_EACH_N(_Np, _Mp, ...)                   _CCCL_PP_CAT2(_CCCL_PP_FOR_EACH_, _Np)(_Mp, __VA_ARGS__)
#define _CCCL_PP_FOR_EACH_1(_Mp, _1)                         _Mp(_1)
#define _CCCL_PP_FOR_EACH_2(_Mp, _1, _2)                     _Mp(_1) _Mp(_2)
#define _CCCL_PP_FOR_EACH_3(_Mp, _1, _2, _3)                 _Mp(_1) _Mp(_2) _Mp(_3)
#define _CCCL_PP_FOR_EACH_4(_Mp, _1, _2, _3, _4)             _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4)
#define _CCCL_PP_FOR_EACH_5(_Mp, _1, _2, _3, _4, _5)         _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5)
#define _CCCL_PP_FOR_EACH_6(_Mp, _1, _2, _3, _4, _5, _6)     _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6)
#define _CCCL_PP_FOR_EACH_7(_Mp, _1, _2, _3, _4, _5, _6, _7) _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7)
#define _CCCL_PP_FOR_EACH_8(_Mp, _1, _2, _3, _4, _5, _6, _7, _8) \
  _Mp(_1) _Mp(_2) _Mp(_3) _Mp(_4) _Mp(_5) _Mp(_6) _Mp(_7) _Mp(_8)

#define _CCCL_PP_PROBE_EMPTY_PROBE__CCCL_PP_PROBE_EMPTY _CCCL_PP_PROBE(~)

#define _CCCL_PP_PROBE_EMPTY()
#define _CCCL_PP_IS_NOT_EMPTY(...)                                                                             \
  _CCCL_PP_EVAL(_CCCL_PP_CHECK, _CCCL_PP_CAT(_CCCL_PP_PROBE_EMPTY_PROBE_, _CCCL_PP_PROBE_EMPTY __VA_ARGS__())) \
  /**/

#define _CCCL_PP_TAIL(_, ...) __VA_ARGS__

///////////////////////////////////////////////////////////////////////////////

// Count the number of arguments. There must be at least one argument and fewer
// than 126 arguments.
// clang-format off
#define _CCCL_PP_COUNT_IMPL(                                                                      \
  _125, _124, _123, _122, _121, _120, _119, _118, _117, _116, _115, _114, _113, _112, _111, _110, \
  _109, _108, _107, _106, _105, _104, _103, _102, _101, _100, _99, _98, _97, _96, _95, _94,       \
  _93, _92, _91, _90, _89, _88, _87, _86, _85, _84, _83, _82, _81, _80, _79, _78,                 \
  _77, _76, _75, _74, _73, _72, _71, _70, _69, _68, _67, _66, _65, _64, _63, _62,                 \
  _61, _60, _59, _58, _57, _56, _55, _54, _53, _52, _51, _50, _49, _48, _47, _46,                 \
  _45, _44, _43, _42, _41, _40, _39, _38, _37, _36, _35, _34, _33, _32, _31, _30,                 \
  _29, _28, _27, _26, _25, _24, _23, _22, _21, _20, _19, _18, _17, _16, _15, _14,                 \
  _13, _12, _11, _10, _9, _8, _7, _6, _5, _4, _3, _2, _1, _0, ...) _0

#define _CCCL_PP_COUNT(...)                                                         \
  _CCCL_PP_EXPAND(_CCCL_PP_COUNT_IMPL( __VA_ARGS__,                                 \
    125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114, 113, 112, 111, 110, \
    109, 108, 107, 106, 105, 104, 103, 102, 101, 100, 99, 98, 97, 96, 95, 94,       \
    93, 92, 91, 90, 89, 88, 87, 86, 85, 84, 83, 82, 81, 80, 79, 78,                 \
    77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62,                 \
    61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46,                 \
    45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30,                 \
    29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,                 \
    13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))
// clang-format on

///////////////////////////////////////////////////////////////////////////////

#define _CCCL_PP_INC(_X) _CCCL_PP_INC_IMPL0(_X)

#define _CCCL_PP_INC_IMPL0(_X) _CCCL_PP_CAT(_CCCL_PP_INC_IMPL_TAG, _X)

#define _CCCL_PP_INC_IMPL_TAG0   1
#define _CCCL_PP_INC_IMPL_TAG1   2
#define _CCCL_PP_INC_IMPL_TAG2   3
#define _CCCL_PP_INC_IMPL_TAG3   4
#define _CCCL_PP_INC_IMPL_TAG4   5
#define _CCCL_PP_INC_IMPL_TAG5   6
#define _CCCL_PP_INC_IMPL_TAG6   7
#define _CCCL_PP_INC_IMPL_TAG7   8
#define _CCCL_PP_INC_IMPL_TAG8   9
#define _CCCL_PP_INC_IMPL_TAG9   10
#define _CCCL_PP_INC_IMPL_TAG10  11
#define _CCCL_PP_INC_IMPL_TAG11  12
#define _CCCL_PP_INC_IMPL_TAG12  13
#define _CCCL_PP_INC_IMPL_TAG13  14
#define _CCCL_PP_INC_IMPL_TAG14  15
#define _CCCL_PP_INC_IMPL_TAG15  16
#define _CCCL_PP_INC_IMPL_TAG16  17
#define _CCCL_PP_INC_IMPL_TAG17  18
#define _CCCL_PP_INC_IMPL_TAG18  19
#define _CCCL_PP_INC_IMPL_TAG19  20
#define _CCCL_PP_INC_IMPL_TAG20  21
#define _CCCL_PP_INC_IMPL_TAG21  22
#define _CCCL_PP_INC_IMPL_TAG22  23
#define _CCCL_PP_INC_IMPL_TAG23  24
#define _CCCL_PP_INC_IMPL_TAG24  25
#define _CCCL_PP_INC_IMPL_TAG25  26
#define _CCCL_PP_INC_IMPL_TAG26  27
#define _CCCL_PP_INC_IMPL_TAG27  28
#define _CCCL_PP_INC_IMPL_TAG28  29
#define _CCCL_PP_INC_IMPL_TAG29  30
#define _CCCL_PP_INC_IMPL_TAG30  31
#define _CCCL_PP_INC_IMPL_TAG31  32
#define _CCCL_PP_INC_IMPL_TAG32  33
#define _CCCL_PP_INC_IMPL_TAG33  34
#define _CCCL_PP_INC_IMPL_TAG34  35
#define _CCCL_PP_INC_IMPL_TAG35  36
#define _CCCL_PP_INC_IMPL_TAG36  37
#define _CCCL_PP_INC_IMPL_TAG37  38
#define _CCCL_PP_INC_IMPL_TAG38  39
#define _CCCL_PP_INC_IMPL_TAG39  40
#define _CCCL_PP_INC_IMPL_TAG40  41
#define _CCCL_PP_INC_IMPL_TAG41  42
#define _CCCL_PP_INC_IMPL_TAG42  43
#define _CCCL_PP_INC_IMPL_TAG43  44
#define _CCCL_PP_INC_IMPL_TAG44  45
#define _CCCL_PP_INC_IMPL_TAG45  46
#define _CCCL_PP_INC_IMPL_TAG46  47
#define _CCCL_PP_INC_IMPL_TAG47  48
#define _CCCL_PP_INC_IMPL_TAG48  49
#define _CCCL_PP_INC_IMPL_TAG49  50
#define _CCCL_PP_INC_IMPL_TAG50  51
#define _CCCL_PP_INC_IMPL_TAG51  52
#define _CCCL_PP_INC_IMPL_TAG52  53
#define _CCCL_PP_INC_IMPL_TAG53  54
#define _CCCL_PP_INC_IMPL_TAG54  55
#define _CCCL_PP_INC_IMPL_TAG55  56
#define _CCCL_PP_INC_IMPL_TAG56  57
#define _CCCL_PP_INC_IMPL_TAG57  58
#define _CCCL_PP_INC_IMPL_TAG58  59
#define _CCCL_PP_INC_IMPL_TAG59  60
#define _CCCL_PP_INC_IMPL_TAG60  61
#define _CCCL_PP_INC_IMPL_TAG61  62
#define _CCCL_PP_INC_IMPL_TAG62  63
#define _CCCL_PP_INC_IMPL_TAG63  64
#define _CCCL_PP_INC_IMPL_TAG64  65
#define _CCCL_PP_INC_IMPL_TAG65  66
#define _CCCL_PP_INC_IMPL_TAG66  67
#define _CCCL_PP_INC_IMPL_TAG67  68
#define _CCCL_PP_INC_IMPL_TAG68  69
#define _CCCL_PP_INC_IMPL_TAG69  70
#define _CCCL_PP_INC_IMPL_TAG70  71
#define _CCCL_PP_INC_IMPL_TAG71  72
#define _CCCL_PP_INC_IMPL_TAG72  73
#define _CCCL_PP_INC_IMPL_TAG73  74
#define _CCCL_PP_INC_IMPL_TAG74  75
#define _CCCL_PP_INC_IMPL_TAG75  76
#define _CCCL_PP_INC_IMPL_TAG76  77
#define _CCCL_PP_INC_IMPL_TAG77  78
#define _CCCL_PP_INC_IMPL_TAG78  79
#define _CCCL_PP_INC_IMPL_TAG79  80
#define _CCCL_PP_INC_IMPL_TAG80  81
#define _CCCL_PP_INC_IMPL_TAG81  82
#define _CCCL_PP_INC_IMPL_TAG82  83
#define _CCCL_PP_INC_IMPL_TAG83  84
#define _CCCL_PP_INC_IMPL_TAG84  85
#define _CCCL_PP_INC_IMPL_TAG85  86
#define _CCCL_PP_INC_IMPL_TAG86  87
#define _CCCL_PP_INC_IMPL_TAG87  88
#define _CCCL_PP_INC_IMPL_TAG88  89
#define _CCCL_PP_INC_IMPL_TAG89  90
#define _CCCL_PP_INC_IMPL_TAG90  91
#define _CCCL_PP_INC_IMPL_TAG91  92
#define _CCCL_PP_INC_IMPL_TAG92  93
#define _CCCL_PP_INC_IMPL_TAG93  94
#define _CCCL_PP_INC_IMPL_TAG94  95
#define _CCCL_PP_INC_IMPL_TAG95  96
#define _CCCL_PP_INC_IMPL_TAG96  97
#define _CCCL_PP_INC_IMPL_TAG97  98
#define _CCCL_PP_INC_IMPL_TAG98  99
#define _CCCL_PP_INC_IMPL_TAG99  100
#define _CCCL_PP_INC_IMPL_TAG100 101
#define _CCCL_PP_INC_IMPL_TAG101 102
#define _CCCL_PP_INC_IMPL_TAG102 103
#define _CCCL_PP_INC_IMPL_TAG103 104
#define _CCCL_PP_INC_IMPL_TAG104 105
#define _CCCL_PP_INC_IMPL_TAG105 106
#define _CCCL_PP_INC_IMPL_TAG106 107
#define _CCCL_PP_INC_IMPL_TAG107 108
#define _CCCL_PP_INC_IMPL_TAG108 109
#define _CCCL_PP_INC_IMPL_TAG109 110
#define _CCCL_PP_INC_IMPL_TAG110 111
#define _CCCL_PP_INC_IMPL_TAG111 112
#define _CCCL_PP_INC_IMPL_TAG112 113
#define _CCCL_PP_INC_IMPL_TAG113 114
#define _CCCL_PP_INC_IMPL_TAG114 115
#define _CCCL_PP_INC_IMPL_TAG115 116
#define _CCCL_PP_INC_IMPL_TAG116 117
#define _CCCL_PP_INC_IMPL_TAG117 118
#define _CCCL_PP_INC_IMPL_TAG118 119
#define _CCCL_PP_INC_IMPL_TAG119 120
#define _CCCL_PP_INC_IMPL_TAG120 121
#define _CCCL_PP_INC_IMPL_TAG121 122
#define _CCCL_PP_INC_IMPL_TAG122 123
#define _CCCL_PP_INC_IMPL_TAG123 124
#define _CCCL_PP_INC_IMPL_TAG124 125
#define _CCCL_PP_INC_IMPL_TAG125 126
#define _CCCL_PP_INC_IMPL_TAG126 127
#define _CCCL_PP_INC_IMPL_TAG127 128
#define _CCCL_PP_INC_IMPL_TAG128 129
#define _CCCL_PP_INC_IMPL_TAG129 130
#define _CCCL_PP_INC_IMPL_TAG130 131
#define _CCCL_PP_INC_IMPL_TAG131 132
#define _CCCL_PP_INC_IMPL_TAG132 133
#define _CCCL_PP_INC_IMPL_TAG133 134
#define _CCCL_PP_INC_IMPL_TAG134 135
#define _CCCL_PP_INC_IMPL_TAG135 136
#define _CCCL_PP_INC_IMPL_TAG136 137
#define _CCCL_PP_INC_IMPL_TAG137 138
#define _CCCL_PP_INC_IMPL_TAG138 139
#define _CCCL_PP_INC_IMPL_TAG139 140
#define _CCCL_PP_INC_IMPL_TAG140 141
#define _CCCL_PP_INC_IMPL_TAG141 142
#define _CCCL_PP_INC_IMPL_TAG142 143
#define _CCCL_PP_INC_IMPL_TAG143 144
#define _CCCL_PP_INC_IMPL_TAG144 145
#define _CCCL_PP_INC_IMPL_TAG145 146
#define _CCCL_PP_INC_IMPL_TAG146 147
#define _CCCL_PP_INC_IMPL_TAG147 148
#define _CCCL_PP_INC_IMPL_TAG148 149
#define _CCCL_PP_INC_IMPL_TAG149 150
#define _CCCL_PP_INC_IMPL_TAG150 151
#define _CCCL_PP_INC_IMPL_TAG151 152
#define _CCCL_PP_INC_IMPL_TAG152 153
#define _CCCL_PP_INC_IMPL_TAG153 154
#define _CCCL_PP_INC_IMPL_TAG154 155
#define _CCCL_PP_INC_IMPL_TAG155 156
#define _CCCL_PP_INC_IMPL_TAG156 157
#define _CCCL_PP_INC_IMPL_TAG157 158
#define _CCCL_PP_INC_IMPL_TAG158 159
#define _CCCL_PP_INC_IMPL_TAG159 160
#define _CCCL_PP_INC_IMPL_TAG160 161
#define _CCCL_PP_INC_IMPL_TAG161 162
#define _CCCL_PP_INC_IMPL_TAG162 163
#define _CCCL_PP_INC_IMPL_TAG163 164
#define _CCCL_PP_INC_IMPL_TAG164 165
#define _CCCL_PP_INC_IMPL_TAG165 166
#define _CCCL_PP_INC_IMPL_TAG166 167
#define _CCCL_PP_INC_IMPL_TAG167 168
#define _CCCL_PP_INC_IMPL_TAG168 169
#define _CCCL_PP_INC_IMPL_TAG169 170
#define _CCCL_PP_INC_IMPL_TAG170 171
#define _CCCL_PP_INC_IMPL_TAG171 172
#define _CCCL_PP_INC_IMPL_TAG172 173
#define _CCCL_PP_INC_IMPL_TAG173 174
#define _CCCL_PP_INC_IMPL_TAG174 175
#define _CCCL_PP_INC_IMPL_TAG175 176
#define _CCCL_PP_INC_IMPL_TAG176 177
#define _CCCL_PP_INC_IMPL_TAG177 178
#define _CCCL_PP_INC_IMPL_TAG178 179
#define _CCCL_PP_INC_IMPL_TAG179 180
#define _CCCL_PP_INC_IMPL_TAG180 181
#define _CCCL_PP_INC_IMPL_TAG181 182
#define _CCCL_PP_INC_IMPL_TAG182 183
#define _CCCL_PP_INC_IMPL_TAG183 184
#define _CCCL_PP_INC_IMPL_TAG184 185
#define _CCCL_PP_INC_IMPL_TAG185 186
#define _CCCL_PP_INC_IMPL_TAG186 187
#define _CCCL_PP_INC_IMPL_TAG187 188
#define _CCCL_PP_INC_IMPL_TAG188 189
#define _CCCL_PP_INC_IMPL_TAG189 190
#define _CCCL_PP_INC_IMPL_TAG190 191
#define _CCCL_PP_INC_IMPL_TAG191 192
#define _CCCL_PP_INC_IMPL_TAG192 193
#define _CCCL_PP_INC_IMPL_TAG193 194
#define _CCCL_PP_INC_IMPL_TAG194 195
#define _CCCL_PP_INC_IMPL_TAG195 196
#define _CCCL_PP_INC_IMPL_TAG196 197
#define _CCCL_PP_INC_IMPL_TAG197 198
#define _CCCL_PP_INC_IMPL_TAG198 199
#define _CCCL_PP_INC_IMPL_TAG199 200
#define _CCCL_PP_INC_IMPL_TAG200 201
#define _CCCL_PP_INC_IMPL_TAG201 202
#define _CCCL_PP_INC_IMPL_TAG202 203
#define _CCCL_PP_INC_IMPL_TAG203 204
#define _CCCL_PP_INC_IMPL_TAG204 205
#define _CCCL_PP_INC_IMPL_TAG205 206
#define _CCCL_PP_INC_IMPL_TAG206 207
#define _CCCL_PP_INC_IMPL_TAG207 208
#define _CCCL_PP_INC_IMPL_TAG208 209
#define _CCCL_PP_INC_IMPL_TAG209 210
#define _CCCL_PP_INC_IMPL_TAG210 211
#define _CCCL_PP_INC_IMPL_TAG211 212
#define _CCCL_PP_INC_IMPL_TAG212 213
#define _CCCL_PP_INC_IMPL_TAG213 214
#define _CCCL_PP_INC_IMPL_TAG214 215
#define _CCCL_PP_INC_IMPL_TAG215 216
#define _CCCL_PP_INC_IMPL_TAG216 217
#define _CCCL_PP_INC_IMPL_TAG217 218
#define _CCCL_PP_INC_IMPL_TAG218 219
#define _CCCL_PP_INC_IMPL_TAG219 220
#define _CCCL_PP_INC_IMPL_TAG220 221
#define _CCCL_PP_INC_IMPL_TAG221 222
#define _CCCL_PP_INC_IMPL_TAG222 223
#define _CCCL_PP_INC_IMPL_TAG223 224
#define _CCCL_PP_INC_IMPL_TAG224 225
#define _CCCL_PP_INC_IMPL_TAG225 226
#define _CCCL_PP_INC_IMPL_TAG226 227
#define _CCCL_PP_INC_IMPL_TAG227 228
#define _CCCL_PP_INC_IMPL_TAG228 229
#define _CCCL_PP_INC_IMPL_TAG229 230
#define _CCCL_PP_INC_IMPL_TAG230 231
#define _CCCL_PP_INC_IMPL_TAG231 232
#define _CCCL_PP_INC_IMPL_TAG232 233
#define _CCCL_PP_INC_IMPL_TAG233 234
#define _CCCL_PP_INC_IMPL_TAG234 235
#define _CCCL_PP_INC_IMPL_TAG235 236
#define _CCCL_PP_INC_IMPL_TAG236 237
#define _CCCL_PP_INC_IMPL_TAG237 238
#define _CCCL_PP_INC_IMPL_TAG238 239
#define _CCCL_PP_INC_IMPL_TAG239 240
#define _CCCL_PP_INC_IMPL_TAG240 241
#define _CCCL_PP_INC_IMPL_TAG241 242
#define _CCCL_PP_INC_IMPL_TAG242 243
#define _CCCL_PP_INC_IMPL_TAG243 244
#define _CCCL_PP_INC_IMPL_TAG244 245
#define _CCCL_PP_INC_IMPL_TAG245 246
#define _CCCL_PP_INC_IMPL_TAG246 247
#define _CCCL_PP_INC_IMPL_TAG247 248
#define _CCCL_PP_INC_IMPL_TAG248 249
#define _CCCL_PP_INC_IMPL_TAG249 250
#define _CCCL_PP_INC_IMPL_TAG250 251
#define _CCCL_PP_INC_IMPL_TAG251 252
#define _CCCL_PP_INC_IMPL_TAG252 253
#define _CCCL_PP_INC_IMPL_TAG253 254
#define _CCCL_PP_INC_IMPL_TAG254 255
#define _CCCL_PP_INC_IMPL_TAG255 256
#define _CCCL_PP_INC_IMPL_TAG256 257

#define _CCCL_PP_DEC(_X) _CCCL_PP_DEC_IMPL0(_X)

#define _CCCL_PP_DEC_IMPL0(_X) _CCCL_PP_CAT(_CCCL_PP_DEC_IMPL_TAG, _X)

#define _CCCL_PP_DEC_IMPL_TAG0   ~##~ // This will generate a syntax error
#define _CCCL_PP_DEC_IMPL_TAG1   0
#define _CCCL_PP_DEC_IMPL_TAG2   1
#define _CCCL_PP_DEC_IMPL_TAG3   2
#define _CCCL_PP_DEC_IMPL_TAG4   3
#define _CCCL_PP_DEC_IMPL_TAG5   4
#define _CCCL_PP_DEC_IMPL_TAG6   5
#define _CCCL_PP_DEC_IMPL_TAG7   6
#define _CCCL_PP_DEC_IMPL_TAG8   7
#define _CCCL_PP_DEC_IMPL_TAG9   8
#define _CCCL_PP_DEC_IMPL_TAG10  9
#define _CCCL_PP_DEC_IMPL_TAG11  10
#define _CCCL_PP_DEC_IMPL_TAG12  11
#define _CCCL_PP_DEC_IMPL_TAG13  12
#define _CCCL_PP_DEC_IMPL_TAG14  13
#define _CCCL_PP_DEC_IMPL_TAG15  14
#define _CCCL_PP_DEC_IMPL_TAG16  15
#define _CCCL_PP_DEC_IMPL_TAG17  16
#define _CCCL_PP_DEC_IMPL_TAG18  17
#define _CCCL_PP_DEC_IMPL_TAG19  18
#define _CCCL_PP_DEC_IMPL_TAG20  19
#define _CCCL_PP_DEC_IMPL_TAG21  20
#define _CCCL_PP_DEC_IMPL_TAG22  21
#define _CCCL_PP_DEC_IMPL_TAG23  22
#define _CCCL_PP_DEC_IMPL_TAG24  23
#define _CCCL_PP_DEC_IMPL_TAG25  24
#define _CCCL_PP_DEC_IMPL_TAG26  25
#define _CCCL_PP_DEC_IMPL_TAG27  26
#define _CCCL_PP_DEC_IMPL_TAG28  27
#define _CCCL_PP_DEC_IMPL_TAG29  28
#define _CCCL_PP_DEC_IMPL_TAG30  29
#define _CCCL_PP_DEC_IMPL_TAG31  30
#define _CCCL_PP_DEC_IMPL_TAG32  31
#define _CCCL_PP_DEC_IMPL_TAG33  32
#define _CCCL_PP_DEC_IMPL_TAG34  33
#define _CCCL_PP_DEC_IMPL_TAG35  34
#define _CCCL_PP_DEC_IMPL_TAG36  35
#define _CCCL_PP_DEC_IMPL_TAG37  36
#define _CCCL_PP_DEC_IMPL_TAG38  37
#define _CCCL_PP_DEC_IMPL_TAG39  38
#define _CCCL_PP_DEC_IMPL_TAG40  39
#define _CCCL_PP_DEC_IMPL_TAG41  40
#define _CCCL_PP_DEC_IMPL_TAG42  41
#define _CCCL_PP_DEC_IMPL_TAG43  42
#define _CCCL_PP_DEC_IMPL_TAG44  43
#define _CCCL_PP_DEC_IMPL_TAG45  44
#define _CCCL_PP_DEC_IMPL_TAG46  45
#define _CCCL_PP_DEC_IMPL_TAG47  46
#define _CCCL_PP_DEC_IMPL_TAG48  47
#define _CCCL_PP_DEC_IMPL_TAG49  48
#define _CCCL_PP_DEC_IMPL_TAG50  49
#define _CCCL_PP_DEC_IMPL_TAG51  50
#define _CCCL_PP_DEC_IMPL_TAG52  51
#define _CCCL_PP_DEC_IMPL_TAG53  52
#define _CCCL_PP_DEC_IMPL_TAG54  53
#define _CCCL_PP_DEC_IMPL_TAG55  54
#define _CCCL_PP_DEC_IMPL_TAG56  55
#define _CCCL_PP_DEC_IMPL_TAG57  56
#define _CCCL_PP_DEC_IMPL_TAG58  57
#define _CCCL_PP_DEC_IMPL_TAG59  58
#define _CCCL_PP_DEC_IMPL_TAG60  59
#define _CCCL_PP_DEC_IMPL_TAG61  60
#define _CCCL_PP_DEC_IMPL_TAG62  61
#define _CCCL_PP_DEC_IMPL_TAG63  62
#define _CCCL_PP_DEC_IMPL_TAG64  63
#define _CCCL_PP_DEC_IMPL_TAG65  64
#define _CCCL_PP_DEC_IMPL_TAG66  65
#define _CCCL_PP_DEC_IMPL_TAG67  66
#define _CCCL_PP_DEC_IMPL_TAG68  67
#define _CCCL_PP_DEC_IMPL_TAG69  68
#define _CCCL_PP_DEC_IMPL_TAG70  69
#define _CCCL_PP_DEC_IMPL_TAG71  70
#define _CCCL_PP_DEC_IMPL_TAG72  71
#define _CCCL_PP_DEC_IMPL_TAG73  72
#define _CCCL_PP_DEC_IMPL_TAG74  73
#define _CCCL_PP_DEC_IMPL_TAG75  74
#define _CCCL_PP_DEC_IMPL_TAG76  75
#define _CCCL_PP_DEC_IMPL_TAG77  76
#define _CCCL_PP_DEC_IMPL_TAG78  77
#define _CCCL_PP_DEC_IMPL_TAG79  78
#define _CCCL_PP_DEC_IMPL_TAG80  79
#define _CCCL_PP_DEC_IMPL_TAG81  80
#define _CCCL_PP_DEC_IMPL_TAG82  81
#define _CCCL_PP_DEC_IMPL_TAG83  82
#define _CCCL_PP_DEC_IMPL_TAG84  83
#define _CCCL_PP_DEC_IMPL_TAG85  84
#define _CCCL_PP_DEC_IMPL_TAG86  85
#define _CCCL_PP_DEC_IMPL_TAG87  86
#define _CCCL_PP_DEC_IMPL_TAG88  87
#define _CCCL_PP_DEC_IMPL_TAG89  88
#define _CCCL_PP_DEC_IMPL_TAG90  89
#define _CCCL_PP_DEC_IMPL_TAG91  90
#define _CCCL_PP_DEC_IMPL_TAG92  91
#define _CCCL_PP_DEC_IMPL_TAG93  92
#define _CCCL_PP_DEC_IMPL_TAG94  93
#define _CCCL_PP_DEC_IMPL_TAG95  94
#define _CCCL_PP_DEC_IMPL_TAG96  95
#define _CCCL_PP_DEC_IMPL_TAG97  96
#define _CCCL_PP_DEC_IMPL_TAG98  97
#define _CCCL_PP_DEC_IMPL_TAG99  98
#define _CCCL_PP_DEC_IMPL_TAG100 99
#define _CCCL_PP_DEC_IMPL_TAG101 100
#define _CCCL_PP_DEC_IMPL_TAG102 101
#define _CCCL_PP_DEC_IMPL_TAG103 102
#define _CCCL_PP_DEC_IMPL_TAG104 103
#define _CCCL_PP_DEC_IMPL_TAG105 104
#define _CCCL_PP_DEC_IMPL_TAG106 105
#define _CCCL_PP_DEC_IMPL_TAG107 106
#define _CCCL_PP_DEC_IMPL_TAG108 107
#define _CCCL_PP_DEC_IMPL_TAG109 108
#define _CCCL_PP_DEC_IMPL_TAG110 109
#define _CCCL_PP_DEC_IMPL_TAG111 110
#define _CCCL_PP_DEC_IMPL_TAG112 111
#define _CCCL_PP_DEC_IMPL_TAG113 112
#define _CCCL_PP_DEC_IMPL_TAG114 113
#define _CCCL_PP_DEC_IMPL_TAG115 114
#define _CCCL_PP_DEC_IMPL_TAG116 115
#define _CCCL_PP_DEC_IMPL_TAG117 116
#define _CCCL_PP_DEC_IMPL_TAG118 117
#define _CCCL_PP_DEC_IMPL_TAG119 118
#define _CCCL_PP_DEC_IMPL_TAG120 119
#define _CCCL_PP_DEC_IMPL_TAG121 120
#define _CCCL_PP_DEC_IMPL_TAG122 121
#define _CCCL_PP_DEC_IMPL_TAG123 122
#define _CCCL_PP_DEC_IMPL_TAG124 123
#define _CCCL_PP_DEC_IMPL_TAG125 124
#define _CCCL_PP_DEC_IMPL_TAG126 125
#define _CCCL_PP_DEC_IMPL_TAG127 126
#define _CCCL_PP_DEC_IMPL_TAG128 127
#define _CCCL_PP_DEC_IMPL_TAG129 128
#define _CCCL_PP_DEC_IMPL_TAG130 129
#define _CCCL_PP_DEC_IMPL_TAG131 130
#define _CCCL_PP_DEC_IMPL_TAG132 131
#define _CCCL_PP_DEC_IMPL_TAG133 132
#define _CCCL_PP_DEC_IMPL_TAG134 133
#define _CCCL_PP_DEC_IMPL_TAG135 134
#define _CCCL_PP_DEC_IMPL_TAG136 135
#define _CCCL_PP_DEC_IMPL_TAG137 136
#define _CCCL_PP_DEC_IMPL_TAG138 137
#define _CCCL_PP_DEC_IMPL_TAG139 138
#define _CCCL_PP_DEC_IMPL_TAG140 139
#define _CCCL_PP_DEC_IMPL_TAG141 140
#define _CCCL_PP_DEC_IMPL_TAG142 141
#define _CCCL_PP_DEC_IMPL_TAG143 142
#define _CCCL_PP_DEC_IMPL_TAG144 143
#define _CCCL_PP_DEC_IMPL_TAG145 144
#define _CCCL_PP_DEC_IMPL_TAG146 145
#define _CCCL_PP_DEC_IMPL_TAG147 146
#define _CCCL_PP_DEC_IMPL_TAG148 147
#define _CCCL_PP_DEC_IMPL_TAG149 148
#define _CCCL_PP_DEC_IMPL_TAG150 149
#define _CCCL_PP_DEC_IMPL_TAG151 150
#define _CCCL_PP_DEC_IMPL_TAG152 151
#define _CCCL_PP_DEC_IMPL_TAG153 152
#define _CCCL_PP_DEC_IMPL_TAG154 153
#define _CCCL_PP_DEC_IMPL_TAG155 154
#define _CCCL_PP_DEC_IMPL_TAG156 155
#define _CCCL_PP_DEC_IMPL_TAG157 156
#define _CCCL_PP_DEC_IMPL_TAG158 157
#define _CCCL_PP_DEC_IMPL_TAG159 158
#define _CCCL_PP_DEC_IMPL_TAG160 159
#define _CCCL_PP_DEC_IMPL_TAG161 160
#define _CCCL_PP_DEC_IMPL_TAG162 161
#define _CCCL_PP_DEC_IMPL_TAG163 162
#define _CCCL_PP_DEC_IMPL_TAG164 163
#define _CCCL_PP_DEC_IMPL_TAG165 164
#define _CCCL_PP_DEC_IMPL_TAG166 165
#define _CCCL_PP_DEC_IMPL_TAG167 166
#define _CCCL_PP_DEC_IMPL_TAG168 167
#define _CCCL_PP_DEC_IMPL_TAG169 168
#define _CCCL_PP_DEC_IMPL_TAG170 169
#define _CCCL_PP_DEC_IMPL_TAG171 170
#define _CCCL_PP_DEC_IMPL_TAG172 171
#define _CCCL_PP_DEC_IMPL_TAG173 172
#define _CCCL_PP_DEC_IMPL_TAG174 173
#define _CCCL_PP_DEC_IMPL_TAG175 174
#define _CCCL_PP_DEC_IMPL_TAG176 175
#define _CCCL_PP_DEC_IMPL_TAG177 176
#define _CCCL_PP_DEC_IMPL_TAG178 177
#define _CCCL_PP_DEC_IMPL_TAG179 178
#define _CCCL_PP_DEC_IMPL_TAG180 179
#define _CCCL_PP_DEC_IMPL_TAG181 180
#define _CCCL_PP_DEC_IMPL_TAG182 181
#define _CCCL_PP_DEC_IMPL_TAG183 182
#define _CCCL_PP_DEC_IMPL_TAG184 183
#define _CCCL_PP_DEC_IMPL_TAG185 184
#define _CCCL_PP_DEC_IMPL_TAG186 185
#define _CCCL_PP_DEC_IMPL_TAG187 186
#define _CCCL_PP_DEC_IMPL_TAG188 187
#define _CCCL_PP_DEC_IMPL_TAG189 188
#define _CCCL_PP_DEC_IMPL_TAG190 189
#define _CCCL_PP_DEC_IMPL_TAG191 190
#define _CCCL_PP_DEC_IMPL_TAG192 191
#define _CCCL_PP_DEC_IMPL_TAG193 192
#define _CCCL_PP_DEC_IMPL_TAG194 193
#define _CCCL_PP_DEC_IMPL_TAG195 194
#define _CCCL_PP_DEC_IMPL_TAG196 195
#define _CCCL_PP_DEC_IMPL_TAG197 196
#define _CCCL_PP_DEC_IMPL_TAG198 197
#define _CCCL_PP_DEC_IMPL_TAG199 198
#define _CCCL_PP_DEC_IMPL_TAG200 199
#define _CCCL_PP_DEC_IMPL_TAG201 200
#define _CCCL_PP_DEC_IMPL_TAG202 201
#define _CCCL_PP_DEC_IMPL_TAG203 202
#define _CCCL_PP_DEC_IMPL_TAG204 203
#define _CCCL_PP_DEC_IMPL_TAG205 204
#define _CCCL_PP_DEC_IMPL_TAG206 205
#define _CCCL_PP_DEC_IMPL_TAG207 206
#define _CCCL_PP_DEC_IMPL_TAG208 207
#define _CCCL_PP_DEC_IMPL_TAG209 208
#define _CCCL_PP_DEC_IMPL_TAG210 209
#define _CCCL_PP_DEC_IMPL_TAG211 210
#define _CCCL_PP_DEC_IMPL_TAG212 211
#define _CCCL_PP_DEC_IMPL_TAG213 212
#define _CCCL_PP_DEC_IMPL_TAG214 213
#define _CCCL_PP_DEC_IMPL_TAG215 214
#define _CCCL_PP_DEC_IMPL_TAG216 215
#define _CCCL_PP_DEC_IMPL_TAG217 216
#define _CCCL_PP_DEC_IMPL_TAG218 217
#define _CCCL_PP_DEC_IMPL_TAG219 218
#define _CCCL_PP_DEC_IMPL_TAG220 219
#define _CCCL_PP_DEC_IMPL_TAG221 220
#define _CCCL_PP_DEC_IMPL_TAG222 221
#define _CCCL_PP_DEC_IMPL_TAG223 222
#define _CCCL_PP_DEC_IMPL_TAG224 223
#define _CCCL_PP_DEC_IMPL_TAG225 224
#define _CCCL_PP_DEC_IMPL_TAG226 225
#define _CCCL_PP_DEC_IMPL_TAG227 226
#define _CCCL_PP_DEC_IMPL_TAG228 227
#define _CCCL_PP_DEC_IMPL_TAG229 228
#define _CCCL_PP_DEC_IMPL_TAG230 229
#define _CCCL_PP_DEC_IMPL_TAG231 230
#define _CCCL_PP_DEC_IMPL_TAG232 231
#define _CCCL_PP_DEC_IMPL_TAG233 232
#define _CCCL_PP_DEC_IMPL_TAG234 233
#define _CCCL_PP_DEC_IMPL_TAG235 234
#define _CCCL_PP_DEC_IMPL_TAG236 235
#define _CCCL_PP_DEC_IMPL_TAG237 236
#define _CCCL_PP_DEC_IMPL_TAG238 237
#define _CCCL_PP_DEC_IMPL_TAG239 238
#define _CCCL_PP_DEC_IMPL_TAG240 239
#define _CCCL_PP_DEC_IMPL_TAG241 240
#define _CCCL_PP_DEC_IMPL_TAG242 241
#define _CCCL_PP_DEC_IMPL_TAG243 242
#define _CCCL_PP_DEC_IMPL_TAG244 243
#define _CCCL_PP_DEC_IMPL_TAG245 244
#define _CCCL_PP_DEC_IMPL_TAG246 245
#define _CCCL_PP_DEC_IMPL_TAG247 246
#define _CCCL_PP_DEC_IMPL_TAG248 247
#define _CCCL_PP_DEC_IMPL_TAG249 248
#define _CCCL_PP_DEC_IMPL_TAG250 249
#define _CCCL_PP_DEC_IMPL_TAG251 250
#define _CCCL_PP_DEC_IMPL_TAG252 251
#define _CCCL_PP_DEC_IMPL_TAG253 252
#define _CCCL_PP_DEC_IMPL_TAG254 253
#define _CCCL_PP_DEC_IMPL_TAG255 254
#define _CCCL_PP_DEC_IMPL_TAG256 255
#define _CCCL_PP_DEC_IMPL_TAG257 256

////////////////////////////////////////////////////////////////////////////////

// _CCCL_PP_REPEAT(COUNT, MACRO, STATE, INCREMENT)
//
// Expands to: MACRO(STATE) MACRO(INCREMENT(STATE)) ... MACRO(INCREMENT(INCREMENT(INCREMENT(...))))
// STATE defaults to 0, INCREMENT defaults to _CCCL_PP_INC
#define _CCCL_PP_REPEAT_AUX1(_N, _M)         _CCCL_PP_CAT(_CCCL_PP_REPEAT, _N)(_M, 0, _CCCL_PP_INC)
#define _CCCL_PP_REPEAT_AUX2(_N, _M, _S)     _CCCL_PP_CAT(_CCCL_PP_REPEAT, _N)(_M, _S, _CCCL_PP_INC)
#define _CCCL_PP_REPEAT_AUX3(_N, _M, _S, _F) _CCCL_PP_CAT(_CCCL_PP_REPEAT, _N)(_M, _S, _F)

#define _CCCL_PP_REPEAT_AUX(_C, _N, ...) _CCCL_PP_CAT(_CCCL_PP_REPEAT_AUX, _C)(_N, __VA_ARGS__)
#define _CCCL_PP_REPEAT(_N, ...)         _CCCL_PP_REPEAT_AUX(_CCCL_PP_COUNT(__VA_ARGS__), _N, __VA_ARGS__)

#define _CCCL_PP_REPEAT0(_M, _S, _F)
#define _CCCL_PP_REPEAT1(_M, _S, _F)   _M(_S)
#define _CCCL_PP_REPEAT2(_M, _S, _F)   _M(_S) _CCCL_PP_REPEAT1(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT3(_M, _S, _F)   _M(_S) _CCCL_PP_REPEAT2(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT4(_M, _S, _F)   _M(_S) _CCCL_PP_REPEAT3(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT5(_M, _S, _F)   _M(_S) _CCCL_PP_REPEAT4(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT6(_M, _S, _F)   _M(_S) _CCCL_PP_REPEAT5(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT7(_M, _S, _F)   _M(_S) _CCCL_PP_REPEAT6(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT8(_M, _S, _F)   _M(_S) _CCCL_PP_REPEAT7(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT9(_M, _S, _F)   _M(_S) _CCCL_PP_REPEAT8(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT10(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT9(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT11(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT10(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT12(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT11(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT13(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT12(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT14(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT13(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT15(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT14(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT16(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT15(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT17(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT16(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT18(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT17(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT19(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT18(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT20(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT19(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT21(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT20(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT22(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT21(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT23(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT22(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT24(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT23(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT25(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT24(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT26(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT25(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT27(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT26(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT28(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT27(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT29(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT28(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT30(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT29(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT31(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT30(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT32(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT31(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT33(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT32(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT34(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT33(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT35(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT34(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT36(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT35(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT37(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT36(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT38(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT37(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT39(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT38(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT40(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT39(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT41(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT40(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT42(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT41(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT43(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT42(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT44(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT43(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT45(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT44(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT46(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT45(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT47(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT46(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT48(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT47(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT49(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT48(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT50(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT49(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT51(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT50(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT52(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT51(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT53(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT52(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT54(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT53(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT55(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT54(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT56(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT55(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT57(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT56(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT58(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT57(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT59(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT58(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT60(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT59(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT61(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT60(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT62(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT61(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT63(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT62(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT64(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT63(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT65(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT64(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT66(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT65(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT67(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT66(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT68(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT67(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT69(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT68(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT70(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT69(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT71(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT70(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT72(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT71(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT73(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT72(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT74(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT73(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT75(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT74(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT76(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT75(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT77(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT76(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT78(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT77(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT79(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT78(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT80(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT79(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT81(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT80(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT82(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT81(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT83(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT82(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT84(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT83(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT85(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT84(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT86(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT85(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT87(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT86(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT88(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT87(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT89(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT88(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT90(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT89(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT91(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT90(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT92(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT91(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT93(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT92(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT94(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT93(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT95(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT94(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT96(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT95(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT97(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT96(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT98(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT97(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT99(_M, _S, _F)  _M(_S) _CCCL_PP_REPEAT98(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT100(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT99(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT101(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT100(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT102(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT101(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT103(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT102(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT104(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT103(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT105(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT104(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT106(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT105(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT107(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT106(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT108(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT107(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT109(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT108(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT110(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT109(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT111(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT110(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT112(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT111(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT113(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT112(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT114(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT113(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT115(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT114(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT116(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT115(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT117(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT116(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT118(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT117(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT119(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT118(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT120(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT119(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT121(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT120(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT122(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT121(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT123(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT122(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT124(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT123(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT125(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT124(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT126(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT125(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT127(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT126(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT128(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT127(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT129(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT128(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT130(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT129(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT131(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT130(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT132(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT131(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT133(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT132(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT134(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT133(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT135(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT134(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT136(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT135(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT137(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT136(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT138(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT137(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT139(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT138(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT140(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT139(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT141(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT140(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT142(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT141(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT143(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT142(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT144(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT143(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT145(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT144(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT146(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT145(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT147(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT146(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT148(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT147(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT149(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT148(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT150(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT149(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT151(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT150(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT152(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT151(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT153(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT152(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT154(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT153(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT155(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT154(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT156(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT155(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT157(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT156(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT158(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT157(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT159(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT158(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT160(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT159(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT161(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT160(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT162(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT161(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT163(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT162(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT164(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT163(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT165(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT164(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT166(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT165(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT167(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT166(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT168(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT167(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT169(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT168(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT170(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT169(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT171(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT170(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT172(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT171(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT173(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT172(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT174(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT173(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT175(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT174(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT176(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT175(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT177(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT176(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT178(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT177(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT179(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT178(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT180(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT179(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT181(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT180(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT182(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT181(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT183(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT182(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT184(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT183(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT185(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT184(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT186(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT185(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT187(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT186(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT188(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT187(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT189(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT188(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT190(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT189(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT191(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT190(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT192(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT191(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT193(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT192(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT194(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT193(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT195(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT194(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT196(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT195(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT197(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT196(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT198(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT197(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT199(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT198(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT200(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT199(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT201(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT200(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT202(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT201(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT203(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT202(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT204(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT203(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT205(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT204(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT206(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT205(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT207(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT206(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT208(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT207(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT209(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT208(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT210(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT209(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT211(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT210(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT212(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT211(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT213(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT212(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT214(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT213(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT215(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT214(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT216(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT215(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT217(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT216(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT218(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT217(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT219(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT218(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT220(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT219(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT221(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT220(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT222(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT221(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT223(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT222(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT224(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT223(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT225(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT224(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT226(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT225(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT227(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT226(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT228(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT227(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT229(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT228(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT230(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT229(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT231(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT230(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT232(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT231(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT233(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT232(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT234(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT233(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT235(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT234(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT236(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT235(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT237(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT236(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT238(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT237(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT239(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT238(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT240(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT239(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT241(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT240(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT242(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT241(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT243(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT242(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT244(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT243(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT245(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT244(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT246(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT245(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT247(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT246(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT248(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT247(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT249(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT248(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT250(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT249(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT251(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT250(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT252(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT251(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT253(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT252(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT254(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT253(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT255(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT254(_M, _F(_S), _F)
#define _CCCL_PP_REPEAT256(_M, _S, _F) _M(_S) _CCCL_PP_REPEAT255(_M, _F(_S), _F)

////////////////////////////////////////////////////////////////////////////////

// _CCCL_PP_REPEAT_REVERSE(COUNT, MACRO, STATE, INCREMENT)
//
// Expands to: MACRO(INCREMENT(INCREMENT(INCREMENT(...)))) ... MACRO(INCREMENT(STATE)) MACRO(STATE)
// STATE defaults to 0, INCREMENT defaults to _CCCL_PP_INC
#define _CCCL_PP_REPEAT_REVERSE_AUX1(_N, _M)         _CCCL_PP_CAT(_CCCL_PP_REPEAT_REVERSE, _N)(_M, 0, _CCCL_PP_INC)
#define _CCCL_PP_REPEAT_REVERSE_AUX2(_N, _M, _S)     _CCCL_PP_CAT(_CCCL_PP_REPEAT_REVERSE, _N)(_M, _S, _CCCL_PP_INC)
#define _CCCL_PP_REPEAT_REVERSE_AUX3(_N, _M, _S, _F) _CCCL_PP_CAT(_CCCL_PP_REPEAT_REVERSE, _N)(_M, _S, _F)

#define _CCCL_PP_REPEAT_REVERSE_AUX(_C, _N, ...) _CCCL_PP_CAT(_CCCL_PP_REPEAT_REVERSE_AUX, _C)(_N, __VA_ARGS__)
#define _CCCL_PP_REPEAT_REVERSE(_N, ...)         _CCCL_PP_REPEAT_REVERSE_AUX(_CCCL_PP_COUNT(__VA_ARGS__), _N, __VA_ARGS__)

#define _CCCL_PP_REPEAT_REVERSE0(_M, _S, _F)
#define _CCCL_PP_REPEAT_REVERSE1(_M, _S, _F)   _M(_S)
#define _CCCL_PP_REPEAT_REVERSE2(_M, _S, _F)   _CCCL_PP_REPEAT_REVERSE1(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE3(_M, _S, _F)   _CCCL_PP_REPEAT_REVERSE2(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE4(_M, _S, _F)   _CCCL_PP_REPEAT_REVERSE3(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE5(_M, _S, _F)   _CCCL_PP_REPEAT_REVERSE4(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE6(_M, _S, _F)   _CCCL_PP_REPEAT_REVERSE5(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE7(_M, _S, _F)   _CCCL_PP_REPEAT_REVERSE6(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE8(_M, _S, _F)   _CCCL_PP_REPEAT_REVERSE7(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE9(_M, _S, _F)   _CCCL_PP_REPEAT_REVERSE8(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE10(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE9(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE11(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE10(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE12(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE11(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE13(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE12(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE14(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE13(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE15(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE14(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE16(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE15(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE17(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE16(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE18(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE17(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE19(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE18(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE20(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE19(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE21(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE20(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE22(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE21(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE23(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE22(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE24(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE23(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE25(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE24(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE26(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE25(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE27(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE26(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE28(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE27(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE29(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE28(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE30(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE29(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE31(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE30(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE32(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE31(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE33(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE32(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE34(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE33(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE35(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE34(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE36(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE35(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE37(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE36(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE38(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE37(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE39(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE38(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE40(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE39(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE41(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE40(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE42(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE41(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE43(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE42(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE44(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE43(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE45(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE44(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE46(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE45(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE47(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE46(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE48(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE47(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE49(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE48(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE50(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE49(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE51(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE50(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE52(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE51(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE53(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE52(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE54(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE53(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE55(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE54(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE56(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE55(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE57(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE56(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE58(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE57(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE59(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE58(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE60(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE59(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE61(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE60(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE62(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE61(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE63(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE62(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE64(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE63(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE65(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE64(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE66(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE65(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE67(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE66(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE68(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE67(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE69(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE68(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE70(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE69(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE71(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE70(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE72(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE71(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE73(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE72(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE74(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE73(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE75(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE74(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE76(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE75(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE77(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE76(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE78(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE77(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE79(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE78(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE80(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE79(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE81(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE80(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE82(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE81(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE83(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE82(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE84(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE83(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE85(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE84(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE86(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE85(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE87(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE86(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE88(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE87(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE89(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE88(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE90(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE89(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE91(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE90(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE92(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE91(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE93(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE92(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE94(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE93(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE95(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE94(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE96(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE95(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE97(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE96(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE98(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE97(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE99(_M, _S, _F)  _CCCL_PP_REPEAT_REVERSE98(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE100(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE99(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE101(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE100(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE102(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE101(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE103(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE102(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE104(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE103(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE105(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE104(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE106(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE105(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE107(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE106(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE108(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE107(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE109(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE108(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE110(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE109(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE111(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE110(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE112(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE111(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE113(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE112(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE114(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE113(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE115(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE114(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE116(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE115(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE117(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE116(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE118(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE117(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE119(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE118(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE120(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE119(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE121(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE120(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE122(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE121(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE123(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE122(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE124(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE123(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE125(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE124(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE126(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE125(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE127(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE126(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE128(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE127(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE129(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE128(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE130(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE129(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE131(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE130(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE132(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE131(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE133(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE132(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE134(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE133(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE135(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE134(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE136(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE135(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE137(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE136(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE138(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE137(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE139(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE138(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE140(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE139(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE141(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE140(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE142(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE141(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE143(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE142(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE144(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE143(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE145(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE144(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE146(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE145(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE147(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE146(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE148(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE147(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE149(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE148(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE150(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE149(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE151(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE150(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE152(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE151(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE153(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE152(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE154(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE153(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE155(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE154(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE156(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE155(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE157(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE156(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE158(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE157(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE159(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE158(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE160(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE159(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE161(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE160(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE162(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE161(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE163(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE162(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE164(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE163(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE165(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE164(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE166(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE165(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE167(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE166(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE168(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE167(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE169(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE168(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE170(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE169(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE171(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE170(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE172(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE171(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE173(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE172(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE174(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE173(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE175(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE174(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE176(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE175(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE177(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE176(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE178(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE177(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE179(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE178(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE180(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE179(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE181(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE180(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE182(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE181(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE183(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE182(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE184(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE183(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE185(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE184(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE186(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE185(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE187(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE186(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE188(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE187(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE189(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE188(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE190(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE189(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE191(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE190(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE192(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE191(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE193(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE192(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE194(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE193(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE195(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE194(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE196(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE195(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE197(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE196(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE198(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE197(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE199(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE198(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE200(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE199(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE201(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE200(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE202(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE201(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE203(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE202(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE204(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE203(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE205(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE204(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE206(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE205(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE207(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE206(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE208(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE207(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE209(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE208(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE210(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE209(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE211(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE210(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE212(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE211(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE213(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE212(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE214(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE213(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE215(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE214(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE216(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE215(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE217(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE216(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE218(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE217(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE219(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE218(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE220(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE219(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE221(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE220(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE222(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE221(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE223(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE222(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE224(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE223(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE225(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE224(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE226(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE225(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE227(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE226(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE228(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE227(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE229(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE228(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE230(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE229(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE231(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE230(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE232(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE231(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE233(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE232(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE234(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE233(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE235(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE234(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE236(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE235(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE237(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE236(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE238(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE237(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE239(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE238(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE240(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE239(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE241(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE240(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE242(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE241(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE243(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE242(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE244(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE243(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE245(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE244(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE246(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE245(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE247(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE246(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE248(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE247(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE249(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE248(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE250(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE249(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE251(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE250(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE252(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE251(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE253(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE252(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE254(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE253(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE255(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE254(_M, _F(_S), _F) _M(_S)
#define _CCCL_PP_REPEAT_REVERSE256(_M, _S, _F) _CCCL_PP_REPEAT_REVERSE255(_M, _F(_S), _F) _M(_S)

#define _CCCL_PP_SPLICE_WITH_IMPL1(SEP, P1)       P1
#define _CCCL_PP_SPLICE_WITH_IMPL2(SEP, P1, ...)  _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL1(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL3(SEP, P1, ...)  _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL2(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL4(SEP, P1, ...)  _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL3(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL5(SEP, P1, ...)  _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL4(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL6(SEP, P1, ...)  _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL5(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL7(SEP, P1, ...)  _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL6(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL8(SEP, P1, ...)  _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL7(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL9(SEP, P1, ...)  _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL8(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL10(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL9(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL11(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL10(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL12(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL11(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL13(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL12(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL14(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL13(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL15(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL14(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL16(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL15(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL17(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL16(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL18(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL17(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL19(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL18(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL20(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL19(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL21(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL20(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL22(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL21(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL23(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL22(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL24(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL23(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL25(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL24(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL26(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL25(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL27(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL26(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL28(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL27(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL29(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL28(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL30(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL29(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL31(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL30(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL32(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL31(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL33(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL32(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL34(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL33(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL35(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL34(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL36(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL35(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL37(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL36(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL38(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL37(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL39(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL38(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL40(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL39(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL41(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL40(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL42(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL41(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL43(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL42(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL44(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL43(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL45(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL44(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL46(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL45(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL47(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL46(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL48(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL47(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL49(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL48(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL50(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL49(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL51(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL50(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL52(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL51(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL53(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL52(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL54(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL53(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL55(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL54(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL56(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL55(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL57(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL56(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL58(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL57(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL59(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL58(SEP, __VA_ARGS__))
#define _CCCL_PP_SPLICE_WITH_IMPL60(SEP, P1, ...) _CCCL_PP_CAT(P1##SEP, _CCCL_PP_SPLICE_WITH_IMPL59(SEP, __VA_ARGS__))

#define _CCCL_PP_SPLICE_WITH_IMPL_DISPATCH(N) _CCCL_PP_SPLICE_WITH_IMPL##N

// Splices a pack of arguments into a single token, separated by SEP
// E.g., _CCCL_PP_SPLICE_WITH(_, A, B, C) will evaluate to A_B_C
#define _CCCL_PP_SPLICE_WITH(SEP, ...) \
  _CCCL_PP_EXPAND(_CCCL_PP_EVAL(_CCCL_PP_SPLICE_WITH_IMPL_DISPATCH, _CCCL_PP_COUNT(__VA_ARGS__))(SEP, __VA_ARGS__))

#endif // __CCCL_PREPROCESSOR_H
