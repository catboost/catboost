/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * cparse.h
 *
 * SWIG parser module.
 * ----------------------------------------------------------------------------- */

#ifndef SWIG_CPARSE_H
#define SWIG_CPARSE_H

#include "swig.h"
#include "swigwarn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* cscanner.c */
  extern String *cparse_file;
  extern int cparse_line;
  extern int cparse_cplusplus;
  extern int cparse_cplusplusout;
  extern int cparse_start_line;
  extern String *cparse_unknown_directive;
  extern int scan_doxygen_comments;

  extern void Swig_cparse_cplusplus(int);
  extern void Swig_cparse_cplusplusout(int);
  extern void scanner_file(File *);
  extern void scanner_next_token(int);
  extern int skip_balanced(int startchar, int endchar);
  extern String *get_raw_text_balanced(int startchar, int endchar);
  extern void skip_decl(void);
  extern void scanner_last_id(int);
  extern void scanner_clear_rename(void);
  extern void scanner_set_location(String *file, int line);
  extern void scanner_set_main_input_file(String *file);
  extern String *scanner_get_main_input_file(void);
  extern void Swig_cparse_follow_locators(int);
  extern void scanner_start_inline(String *, int);
  extern String *scanner_ccode;
  extern int yylex(void);

/* parser.y */
  extern SwigType *Swig_cparse_type(String *);
  extern Node *Swig_cparse(File *);
  extern Hash *Swig_cparse_features(void);
  extern void SWIG_cparse_set_compact_default_args(int defargs);
  extern int SWIG_cparse_template_reduce(int treduce);

/* util.c */
  extern void Swig_cparse_replace_descriptor(String *s);
  extern SwigType *Swig_cparse_smartptr(Node *n);
  extern void cparse_normalize_void(Node *);
  extern Parm *Swig_cparse_parm(String *s);
  extern ParmList *Swig_cparse_parms(String *s, Node *file_line_node);
  extern Node *Swig_cparse_new_node(const_String_or_char_ptr tag);

/* templ.c */
  extern int Swig_cparse_template_expand(Node *n, String *rname, ParmList *tparms, Symtab *tscope);
  extern Node *Swig_cparse_template_locate(String *name, ParmList *tparms, String *symname, Symtab *tscope);
  extern void Swig_cparse_debug_templates(int);
  extern ParmList *Swig_cparse_template_parms_expand(ParmList *instantiated_parameters, Node *primary, Node *templ);
  extern ParmList *Swig_cparse_template_partialargs_expand(ParmList *partially_specialized_parms, Node *primary, ParmList *templateparms);

#ifdef __cplusplus
}
#endif
#define SWIG_WARN_NODE_BEGIN(Node) \
 { \
  String *wrnfilter = Node ? Getattr(Node,"feature:warnfilter") : 0; \
  if (wrnfilter) Swig_warnfilter(wrnfilter,1)
#define SWIG_WARN_NODE_END(Node) \
  if (wrnfilter) Swig_warnfilter(wrnfilter,0); \
 }

#endif
