/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * swigwarn.h
 *
 * SWIG warning message numbers
 * This file serves as the main registry of warning message numbers.  Some of these
 * numbers are used internally in the C/C++ source code of SWIG.   However, some
 * of the numbers are used in SWIG configuration files (swig.swg and others).
 *
 * The numbers are roughly organized into a few different classes by functionality.
 *
 * Even though symbolic constants are used in the SWIG source, this is
 * not always the case in SWIG interface files.  Do not change the
 * numbers in this file.
 *
 * This file is used as the input for generating Lib/swigwarn.swg.
 * ----------------------------------------------------------------------------- */

#ifndef SWIG_SWIGWARN_H
#define SWIG_SWIGWARN_H

#define WARN_NONE                     0

/* -- Deprecated features -- */

/* Unused since 4.2.0: #define WARN_DEPRECATED_EXTERN        101 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_VAL           102 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_OUT           103 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_DISABLEDOC    104 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_ENABLEDOC     105 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_DOCONLY       106 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_STYLE         107 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_LOCALSTYLE    108 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_TITLE         109 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_SECTION       110 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_SUBSECTION    111 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_SUBSUBSECTION 112 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_ADDMETHODS    113 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_READONLY      114 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_READWRITE     115 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_EXCEPT        116 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_NEW           117 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_EXCEPT_TM     118 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_IGNORE_TM     119 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_OPTC          120 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_NAME          121 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_NOEXTERN      122 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_NODEFAULT     123 */
/* Unused since 4.1.0: #define WARN_DEPRECATED_TYPEMAP_LANG  124 */
/* Unused since 4.2.0: #define WARN_DEPRECATED_INPUT_FILE    125 */
/* Unused since 4.3.0: #define WARN_DEPRECATED_NESTED_WORKAROUND 126 */

/* -- Preprocessor -- */

#define WARN_PP_MISSING_FILE          201
#define WARN_PP_EVALUATION            202
#define WARN_PP_INCLUDEALL_IMPORTALL  203
#define WARN_PP_CPP_WARNING           204
#define WARN_PP_CPP_ERROR             205
#define WARN_PP_UNEXPECTED_TOKENS     206

/* -- C/C++ Parser -- */

#define WARN_PARSE_CLASS_KEYWORD      301
#define WARN_PARSE_REDEFINED          302
#define WARN_PARSE_EXTEND_UNDEF       303
#define WARN_PARSE_UNSUPPORTED_VALUE  304
#define WARN_PARSE_BAD_VALUE          305
/* Unused since 1.3.32: #define WARN_PARSE_PRIVATE            306 */
/* Unused since 4.2.0: #define WARN_PARSE_BAD_DEFAULT        307 */
#define WARN_PARSE_NAMESPACE_ALIAS    308
#define WARN_PARSE_PRIVATE_INHERIT    309
/* Unused since 1.3.18: #define WARN_PARSE_TEMPLATE_REPEAT    310 */
/* Unused since 1.3.18: #define WARN_PARSE_TEMPLATE_PARTIAL   311 */
#define WARN_PARSE_UNNAMED_NESTED_CLASS 312
#define WARN_PARSE_UNDEFINED_EXTERN   313
#define WARN_PARSE_KEYWORD            314
#define WARN_PARSE_USING_UNDEF        315
/* Unused since 1.3.18: #define WARN_PARSE_MODULE_REPEAT      316 */
#define WARN_PARSE_TEMPLATE_SP_UNDEF  317
#define WARN_PARSE_TEMPLATE_AMBIG     318
#define WARN_PARSE_NO_ACCESS          319
#define WARN_PARSE_EXPLICIT_TEMPLATE  320
#define WARN_PARSE_BUILTIN_NAME       321
#define WARN_PARSE_REDUNDANT          322
#define WARN_PARSE_REC_INHERITANCE    323
#define WARN_PARSE_NESTED_TEMPLATE    324
#define WARN_PARSE_NAMED_NESTED_CLASS 325
#define WARN_PARSE_EXTEND_NAME        326
#define WARN_PARSE_EXTERN_TEMPLATE    327
#define WARN_PARSE_ASSIGNED_VALUE     328
#define WARN_PARSE_USING_CONSTRUCTOR  329
#define WARN_PARSE_TEMPLATE_FORWARD   330
#define WARN_PARSE_TEMPLATE_NESTED    331

#define WARN_CPP11_LAMBDA             340
/* Unused since 3.0.11: #define WARN_CPP11_ALIAS_DECLARATION  341 */
/* Unused since 3.0.11: #define WARN_CPP11_ALIAS_TEMPLATE     342 */
/* Unused since 4.2.0: #define WARN_CPP11_VARIADIC_TEMPLATE  343 */
#define WARN_CPP11_DECLTYPE           344
#define WARN_CPP14_AUTO               345
#define WARN_CPP11_AUTO               346

#define WARN_IGNORE_OPERATOR_NEW        350	/* new */
#define WARN_IGNORE_OPERATOR_DELETE     351	/* delete */
#define WARN_IGNORE_OPERATOR_PLUS       352	/* + */
#define WARN_IGNORE_OPERATOR_MINUS      353	/* - */
#define WARN_IGNORE_OPERATOR_MUL        354	/* * */
#define WARN_IGNORE_OPERATOR_DIV        355	/* / */
#define WARN_IGNORE_OPERATOR_MOD        356	/* % */
#define WARN_IGNORE_OPERATOR_XOR        357	/* ^ */
#define WARN_IGNORE_OPERATOR_AND        358	/* & */
#define WARN_IGNORE_OPERATOR_OR         359	/* | */
#define WARN_IGNORE_OPERATOR_NOT        360	/* ~ */
#define WARN_IGNORE_OPERATOR_LNOT       361	/* ! */
#define WARN_IGNORE_OPERATOR_EQ         362	/* = */
#define WARN_IGNORE_OPERATOR_LT         363	/* < */
#define WARN_IGNORE_OPERATOR_GT         364	/* > */
#define WARN_IGNORE_OPERATOR_PLUSEQ     365	/* += */
#define WARN_IGNORE_OPERATOR_MINUSEQ    366	/* -= */
#define WARN_IGNORE_OPERATOR_MULEQ      367	/* *= */
#define WARN_IGNORE_OPERATOR_DIVEQ      368	/* /= */
#define WARN_IGNORE_OPERATOR_MODEQ      369	/* %= */
#define WARN_IGNORE_OPERATOR_XOREQ      370	/* ^= */
#define WARN_IGNORE_OPERATOR_ANDEQ      371	/* &= */
#define WARN_IGNORE_OPERATOR_OREQ       372	/* |= */
#define WARN_IGNORE_OPERATOR_LSHIFT     373	/* << */
#define WARN_IGNORE_OPERATOR_RSHIFT     374	/* >> */
#define WARN_IGNORE_OPERATOR_LSHIFTEQ   375	/* <<= */
#define WARN_IGNORE_OPERATOR_RSHIFTEQ   376	/* >>= */
#define WARN_IGNORE_OPERATOR_EQUALTO    377	/* == */
#define WARN_IGNORE_OPERATOR_NOTEQUAL   378	/* != */
#define WARN_IGNORE_OPERATOR_LTEQUAL    379	/* <= */
#define WARN_IGNORE_OPERATOR_GTEQUAL    380	/* >= */
#define WARN_IGNORE_OPERATOR_LAND       381	/* && */
#define WARN_IGNORE_OPERATOR_LOR        382	/* || */
#define WARN_IGNORE_OPERATOR_PLUSPLUS   383	/* ++ */
#define WARN_IGNORE_OPERATOR_MINUSMINUS 384	/* -- */
#define WARN_IGNORE_OPERATOR_COMMA      385	/* , */
#define WARN_IGNORE_OPERATOR_ARROWSTAR  386	/* ->* */
#define WARN_IGNORE_OPERATOR_ARROW      387	/* -> */
#define WARN_IGNORE_OPERATOR_CALL       388	/* () */
#define WARN_IGNORE_OPERATOR_INDEX      389	/* [] */
#define WARN_IGNORE_OPERATOR_UPLUS      390	/* + */
#define WARN_IGNORE_OPERATOR_UMINUS     391	/* - */
#define WARN_IGNORE_OPERATOR_UMUL       392	/* * */
#define WARN_IGNORE_OPERATOR_UAND       393	/* & */
#define WARN_IGNORE_OPERATOR_NEWARR     394	/* new [] */
#define WARN_IGNORE_OPERATOR_DELARR     395	/* delete [] */
#define WARN_IGNORE_OPERATOR_REF        396	/* operator *() */
#define WARN_IGNORE_OPERATOR_LTEQUALGT  397	/* <=> */

/* please leave 350-399 free for WARN_IGNORE_OPERATOR_* */

/* -- Type system and typemaps -- */

#define WARN_TYPE_UNDEFINED_CLASS     401
#define WARN_TYPE_INCOMPLETE          402
#define WARN_TYPE_ABSTRACT            403
#define WARN_TYPE_REDEFINED           404
#define WARN_TYPE_RVALUE_REF_QUALIFIER_IGNORED 405
#define WARN_TYPE_NSPACE_SETTING      406

/* Unused since 4.1.0: #define WARN_TYPEMAP_SOURCETARGET     450 */
#define WARN_TYPEMAP_CHARLEAK         451
/* Unused since 1.3.32: #define WARN_TYPEMAP_SWIGTYPE         452 */
#define WARN_TYPEMAP_APPLY_UNDEF      453
#define WARN_TYPEMAP_SWIGTYPELEAK     454
#define WARN_TYPEMAP_WCHARLEAK        455

#define WARN_TYPEMAP_IN_UNDEF         460
#define WARN_TYPEMAP_OUT_UNDEF        461
#define WARN_TYPEMAP_VARIN_UNDEF      462
#define WARN_TYPEMAP_VAROUT_UNDEF     463
#define WARN_TYPEMAP_CONST_UNDEF      464
#define WARN_TYPEMAP_UNDEF            465
#define WARN_TYPEMAP_VAR_UNDEF        466
#define WARN_TYPEMAP_TYPECHECK        467
#define WARN_TYPEMAP_THROW            468
#define WARN_TYPEMAP_DIRECTORIN_UNDEF  469
#define WARN_TYPEMAP_THREAD_UNSAFE     470	/* mostly used in directorout typemaps */
#define WARN_TYPEMAP_DIRECTOROUT_UNDEF 471
#define WARN_TYPEMAP_TYPECHECK_UNDEF   472
#define WARN_TYPEMAP_DIRECTOROUT_PTR   473
#define WARN_TYPEMAP_OUT_OPTIMAL_IGNORED  474
#define WARN_TYPEMAP_OUT_OPTIMAL_MULTIPLE 475
#define WARN_TYPEMAP_INITIALIZER_LIST  476
#define WARN_TYPEMAP_DIRECTORTHROWS_UNDEF 477

/* -- Fragments -- */
#define WARN_FRAGMENT_NOT_FOUND       490

/* -- General code generation -- */

#define WARN_LANG_OVERLOAD_DECL       501
#define WARN_LANG_OVERLOAD_CONSTRUCT  502
#define WARN_LANG_IDENTIFIER          503
#define WARN_LANG_RETURN_TYPE         504
#define WARN_LANG_VARARGS             505
#define WARN_LANG_VARARGS_KEYWORD     506
#define WARN_LANG_NATIVE_UNIMPL       507
#define WARN_LANG_DEREF_SHADOW        508
#define WARN_LANG_OVERLOAD_SHADOW     509
#define WARN_LANG_FRIEND_IGNORE       510 /* No longer issued */
#define WARN_LANG_OVERLOAD_KEYWORD    511
#define WARN_LANG_OVERLOAD_CONST      512
#define WARN_LANG_CLASS_UNNAMED       513
#define WARN_LANG_DIRECTOR_VDESTRUCT  514
#define WARN_LANG_DISCARD_CONST       515
#define WARN_LANG_OVERLOAD_IGNORED    516
#define WARN_LANG_DIRECTOR_ABSTRACT   517
#define WARN_LANG_PORTABILITY_FILENAME 518
#define WARN_LANG_TEMPLATE_METHOD_IGNORE 519
#define WARN_LANG_SMARTPTR_MISSING    520
#define WARN_LANG_ILLEGAL_DESTRUCTOR  521
#define WARN_LANG_EXTEND_CONSTRUCTOR  522
#define WARN_LANG_EXTEND_DESTRUCTOR   523
#define WARN_LANG_EXPERIMENTAL        524
#define WARN_LANG_DIRECTOR_FINAL      525
#define WARN_LANG_USING_NAME_DIFFERENT 526
#define WARN_LANG_DEPRECATED          527

/* -- Doxygen comments -- */

#define WARN_DOXYGEN_UNKNOWN_COMMAND          560
#define WARN_DOXYGEN_UNEXPECTED_END_OF_COMMENT  561
#define WARN_DOXYGEN_COMMAND_EXPECTED         562
#define WARN_DOXYGEN_HTML_ERROR               563
#define WARN_DOXYGEN_COMMAND_ERROR            564
#define WARN_DOXYGEN_UNKNOWN_CHARACTER        565
#define WARN_DOXYGEN_UNEXPECTED_ITERATOR_VALUE  566

/* -- Reserved (600-699) -- */

/* -- Language module specific warnings (700 - 899) -- */

/* Feel free to claim any number in this space that's not currently being used. Just make sure you
   add an entry here */

#define WARN_D_TYPEMAP_CTYPE_UNDEF            700
#define WARN_D_TYPEMAP_IMTYPE_UNDEF           701
#define WARN_D_TYPEMAP_DTYPE_UNDEF            702
#define WARN_D_MULTIPLE_INHERITANCE           703
#define WARN_D_TYPEMAP_CLASSMOD_UNDEF         704
#define WARN_D_TYPEMAP_DBODY_UNDEF            705
#define WARN_D_TYPEMAP_DOUT_UNDEF             706
#define WARN_D_TYPEMAP_DIN_UNDEF              707
#define WARN_D_TYPEMAP_DDIRECTORIN_UNDEF      708
#define WARN_D_TYPEMAP_DCONSTRUCTOR_UNDEF     709
#define WARN_D_EXCODE_MISSING                 710
#define WARN_D_CANTHROW_MISSING               711
#define WARN_D_NO_DIRECTORCONNECT_ATTR        712
#define WARN_D_NAME_COLLISION                 713

/* please leave 700-719 free for D */

#define WARN_SCILAB_TRUNCATED_NAME            720

/* please leave 720-739 free for Scilab */

#define WARN_PYTHON_INDENT_MISMATCH           740

/* please leave 740-749 free for Python */

/* Unused since 4.2.0: #define WARN_R_MISSING_RTYPECHECK_TYPEMAP     750 */
#define WARN_R_TYPEMAP_RTYPECHECK_UNDEF       751

/* please leave 750-759 free for R */

#define WARN_C_TYPEMAP_CTYPE_UNDEF            760
#define WARN_C_UNSUPPORTTED                   761

/* please leave 760-779 free for C */

#define WARN_RUBY_WRONG_NAME                  801
#define WARN_RUBY_MULTIPLE_INHERITANCE        802

/* please leave 800-809 free for Ruby */

#define WARN_JAVA_TYPEMAP_JNI_UNDEF           810
#define WARN_JAVA_TYPEMAP_JTYPE_UNDEF         811
#define WARN_JAVA_TYPEMAP_JSTYPE_UNDEF        812
#define WARN_JAVA_MULTIPLE_INHERITANCE        813
#define WARN_JAVA_TYPEMAP_GETCPTR_UNDEF       814
#define WARN_JAVA_TYPEMAP_CLASSMOD_UNDEF      815
#define WARN_JAVA_TYPEMAP_JAVABODY_UNDEF      816
#define WARN_JAVA_TYPEMAP_JAVAOUT_UNDEF       817
#define WARN_JAVA_TYPEMAP_JAVAIN_UNDEF        818
#define WARN_JAVA_TYPEMAP_JAVADIRECTORIN_UNDEF    819
#define WARN_JAVA_TYPEMAP_JAVADIRECTOROUT_UNDEF   820
#define WARN_JAVA_TYPEMAP_INTERFACECODE_UNDEF 821
#define WARN_JAVA_COVARIANT_RET               822
#define WARN_JAVA_TYPEMAP_JAVACONSTRUCT_UNDEF 823
#define WARN_JAVA_TYPEMAP_DIRECTORIN_NODESC   824
#define WARN_JAVA_NO_DIRECTORCONNECT_ATTR     825
#define WARN_JAVA_NSPACE_WITHOUT_PACKAGE      826
#define WARN_JAVA_TYPEMAP_INTERFACEMODIFIERS_UNDEF 827

/* please leave 810-829 free for Java */

#define WARN_CSHARP_TYPEMAP_CTYPE_UNDEF       830
#define WARN_CSHARP_TYPEMAP_CSTYPE_UNDEF      831
#define WARN_CSHARP_TYPEMAP_CSWTYPE_UNDEF     832
#define WARN_CSHARP_MULTIPLE_INHERITANCE      833
#define WARN_CSHARP_TYPEMAP_GETCPTR_UNDEF     834
#define WARN_CSHARP_TYPEMAP_CLASSMOD_UNDEF    835
#define WARN_CSHARP_TYPEMAP_CSBODY_UNDEF      836
#define WARN_CSHARP_TYPEMAP_CSOUT_UNDEF       837
#define WARN_CSHARP_TYPEMAP_CSIN_UNDEF        838
#define WARN_CSHARP_TYPEMAP_CSDIRECTORIN_UNDEF    839
#define WARN_CSHARP_TYPEMAP_CSDIRECTOROUT_UNDEF   840
#define WARN_CSHARP_TYPEMAP_INTERFACECODE_UNDEF   841
#define WARN_CSHARP_COVARIANT_RET             842
#define WARN_CSHARP_TYPEMAP_CSCONSTRUCT_UNDEF 843
#define WARN_CSHARP_EXCODE                    844
#define WARN_CSHARP_CANTHROW                  845
#define WARN_CSHARP_NO_DIRECTORCONNECT_ATTR   846
#define WARN_CSHARP_TYPEMAP_INTERFACEMODIFIERS_UNDEF 847

/* please leave 830-849 free for C# */

/* 850-860 were used by Modula 3 (removed in SWIG 4.1.0) - avoid reusing for now */

#define WARN_PHP_MULTIPLE_INHERITANCE         870
#define WARN_PHP_UNKNOWN_PRAGMA               871
/* Unused since 4.1.0: define WARN_PHP_PUBLIC_BASE                  872 */

/* please leave 870-889 free for PHP */

#define WARN_GO_NAME_CONFLICT                 890

/* please leave 890-899 free for Go */

/* -- User defined warnings (900 - 999) -- */

#endif
