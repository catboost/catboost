/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * parser.y
 *
 * Bison parser for SWIG.   The grammar is a somewhat broken subset of C/C++.
 * This file is a bit of a mess and probably needs to be rewritten at
 * some point.  Beware.
 * ----------------------------------------------------------------------------- */

%require "3.5"

/* There are a small number of known shift-reduce conflicts in this file, fail
   compilation if any more are introduced.

   Please don't increase the number of the conflicts if at all possible. And if
   you really have no choice but to do it, make sure you clearly document each
   new conflict in this file.
 */
%expect 7

/* Make the internal token numbers the same as the external token numbers
 * which saves Bison generating a lookup table to map between them, giving
 * a smaller and faster generated parser.
 */
%define api.token.raw

%{
/* doh.h uses #pragma GCC poison with GCC to prevent direct calls to certain
 * standard C library functions being introduced, but those cause errors due
 * to checks like `#if defined YYMALLOC || defined malloc` in the bison
 * template code.  We can't easily arrange to include headers after that
 * template code, so instead we disable the problematic poisoning for this
 * file.
 */
#define DOH_NO_POISON_MALLOC_FREE

#include "swig.h"
#include "cparse.h"
#include "preprocessor.h"
#include <ctype.h>

#define YYMALLOC Malloc
#define YYFREE Free

/* -----------------------------------------------------------------------------
 *                               Externals
 * ----------------------------------------------------------------------------- */

int  yyparse(void);

/* NEW Variables */

static void    *top = 0;      /* Top of the generated parse tree */
static int      unnamed = 0;  /* Unnamed datatype counter */
static Hash    *classes = 0;        /* Hash table of classes */
static Hash    *classes_typedefs = 0; /* Hash table of typedef classes: typedef struct X {...} Y; */
static Symtab  *prev_symtab = 0;
static Node    *current_class = 0;
String  *ModuleName = 0;
static Node    *module_node = 0;
static String  *Classprefix = 0;  
static String  *Namespaceprefix = 0;
static int      inclass = 0;
static Node    *currentOuterClass = 0; /* for nested classes */
static String  *last_cpptype = 0;
static int      inherit_list = 0;
static Parm    *template_parameters = 0;
static int      parsing_template_declaration = 0;
static int      extendmode   = 0;
static int      compact_default_args = 0;
static int      template_reduce = 0;
static int      cparse_externc = 0;
int		ignore_nested_classes = 0;
int		kwargs_supported = 0;

/* -----------------------------------------------------------------------------
 *                            Doxygen Comment Globals
 * ----------------------------------------------------------------------------- */
static String *currentDeclComment = NULL; /* Comment of C/C++ declaration. */

/* -----------------------------------------------------------------------------
 *                            Assist Functions
 * ----------------------------------------------------------------------------- */


 
/* Called by the parser (yyparse) when an error is found.*/
static void yyerror (const char *e) {
  (void)e;
}

static Node *new_node(const_String_or_char_ptr tag) {
  Node *n = Swig_cparse_new_node(tag);
  return n;
}

/* Copies a node.  Does not copy tree links or symbol table data (except for
   sym:name) */

static Node *copy_node(Node *n) {
  Node *nn;
  Iterator k;
  nn = NewHash();
  Setfile(nn,Getfile(n));
  Setline(nn,Getline(n));
  for (k = First(n); k.key; k = Next(k)) {
    String *ci;
    String *key = k.key;
    char *ckey = Char(key);
    if ((strcmp(ckey,"nextSibling") == 0) ||
	(strcmp(ckey,"previousSibling") == 0) ||
	(strcmp(ckey,"parentNode") == 0) ||
	(strcmp(ckey,"lastChild") == 0)) {
      continue;
    }
    if (Strncmp(key,"csym:",5) == 0) continue;
    /* We do copy sym:name.  For templates */
    if ((strcmp(ckey,"sym:name") == 0) || 
	(strcmp(ckey,"sym:weak") == 0) ||
	(strcmp(ckey,"sym:typename") == 0)) {
      String *ci = Copy(k.item);
      Setattr(nn,key, ci);
      Delete(ci);
      continue;
    }
    if (strcmp(ckey,"sym:symtab") == 0) {
      Setattr(nn,"sym:needs_symtab", "1");
    }
    /* We don't copy any other symbol table attributes */
    if (strncmp(ckey,"sym:",4) == 0) {
      continue;
    }
    /* If children.  We copy them recursively using this function */
    if (strcmp(ckey,"firstChild") == 0) {
      /* Copy children */
      Node *cn = k.item;
      while (cn) {
	Node *copy = copy_node(cn);
	appendChild(nn,copy);
	Delete(copy);
	cn = nextSibling(cn);
      }
      continue;
    }
    /* We don't copy the symbol table.  But we drop an attribute 
       requires_symtab so that functions know it needs to be built */

    if (strcmp(ckey,"symtab") == 0) {
      /* Node defined a symbol table. */
      Setattr(nn,"requires_symtab","1");
      continue;
    }
    /* Can't copy nodes */
    if (strcmp(ckey,"node") == 0) {
      continue;
    }
    if ((strcmp(ckey,"parms") == 0) || (strcmp(ckey,"pattern") == 0) || (strcmp(ckey,"throws") == 0)
	|| (strcmp(ckey,"kwargs") == 0)) {
      ParmList *pl = CopyParmList(k.item);
      Setattr(nn,key,pl);
      Delete(pl);
      continue;
    }
    if (strcmp(ckey,"nested:outer") == 0) { /* don't copy outer classes links, they will be updated later */
      Setattr(nn, key, k.item);
      continue;
    }
    /* defaultargs will be patched back in later in update_defaultargs() */
    if (strcmp(ckey,"defaultargs") == 0) {
      Setattr(nn, "needs_defaultargs", "1");
      continue;
    }
    /* same for abstracts, which contains pointers to the source node children, and so will need to be patch too */
    if (strcmp(ckey,"abstracts") == 0) {
      SetFlag(nn, "needs_abstracts");
      continue;
    }
    /* Looks okay.  Just copy the data using Copy */
    ci = Copy(k.item);
    Setattr(nn, key, ci);
    Delete(ci);
  }
  return nn;
}

static void set_comment(Node *n, String *comment) {
  String *name;
  Parm *p;
  if (!n || !comment)
    return;

  if (Getattr(n, "doxygen"))
    Append(Getattr(n, "doxygen"), comment);
  else {
    Setattr(n, "doxygen", comment);
    /* This is the first comment, populate it with @params, if any */
    p = Getattr(n, "parms");
    while (p) {
      if (Getattr(p, "doxygen"))
	Printv(comment, "\n@param ", Getattr(p, "name"), Getattr(p, "doxygen"), NIL);
      p=nextSibling(p);
    }
  }
  
  /* Append same comment to every generated overload */
  name = Getattr(n, "name");
  if (!name)
    return;
  n = nextSibling(n);
  while (n && Getattr(n, "name") && Strcmp(Getattr(n, "name"), name) == 0) {
    Setattr(n, "doxygen", comment);
    n = nextSibling(n);
  }
}

/* -----------------------------------------------------------------------------
 *                              Variables
 * ----------------------------------------------------------------------------- */

static int cplus_mode  = 0;

/* C++ modes */

#define  CPLUS_PUBLIC    1
#define  CPLUS_PRIVATE   2
#define  CPLUS_PROTECTED 3

/* storage classes */

#define SWIG_STORAGE_CLASS_EXTERNC	0x0001
#define SWIG_STORAGE_CLASS_EXTERNCPP	0x0002
#define SWIG_STORAGE_CLASS_EXTERN	0x0004
#define SWIG_STORAGE_CLASS_STATIC	0x0008
#define SWIG_STORAGE_CLASS_TYPEDEF	0x0010
#define SWIG_STORAGE_CLASS_VIRTUAL	0x0020
#define SWIG_STORAGE_CLASS_FRIEND	0x0040
#define SWIG_STORAGE_CLASS_EXPLICIT	0x0080
#define SWIG_STORAGE_CLASS_CONSTEXPR	0x0100
#define SWIG_STORAGE_CLASS_THREAD_LOCAL	0x0200

/* Test if multiple bits are set in x. */
static int multiple_bits_set(unsigned x) { return (x & (x - 1)) != 0; }

static const char* storage_class_string(int c) {
  switch (c) {
    case SWIG_STORAGE_CLASS_EXTERNC:
      return "extern \"C\"";
    case SWIG_STORAGE_CLASS_EXTERNCPP:
      return "extern \"C++\"";
    case SWIG_STORAGE_CLASS_EXTERN:
      return "extern";
    case SWIG_STORAGE_CLASS_STATIC:
      return "static";
    case SWIG_STORAGE_CLASS_TYPEDEF:
      return "typedef";
    case SWIG_STORAGE_CLASS_VIRTUAL:
      return "virtual";
    case SWIG_STORAGE_CLASS_FRIEND:
      return "friend";
    case SWIG_STORAGE_CLASS_EXPLICIT:
      return "explicit";
    case SWIG_STORAGE_CLASS_CONSTEXPR:
      return "constexpr";
    case SWIG_STORAGE_CLASS_THREAD_LOCAL:
      return "thread_local";
  }
  assert(0);
  return "<unknown>";
}

/* include types */
static int   import_mode = 0;

void SWIG_cparse_set_compact_default_args(int defargs) {
  compact_default_args = defargs;
}

int SWIG_cparse_template_reduce(int treduce) {
  template_reduce = treduce;
  return treduce;  
}

/* -----------------------------------------------------------------------------
 *                           Assist functions
 * ----------------------------------------------------------------------------- */

static int promote_type(int t) {
  if (t <= T_UCHAR || t == T_CHAR || t == T_WCHAR) return T_INT;
  return t;
}

/* Perform type-promotion for binary operators */
static int promote(int t1, int t2) {
  t1 = promote_type(t1);
  t2 = promote_type(t2);
  return t1 > t2 ? t1 : t2;
}

static String *yyrename = 0;

/* Forward renaming operator */

static String *resolve_create_node_scope(String *cname, int is_class_definition, int *errored);


Hash *Swig_cparse_features(void) {
  static Hash   *features_hash = 0;
  if (!features_hash) features_hash = NewHash();
  return features_hash;
}

/* -----------------------------------------------------------------------------
 * feature_identifier_fix()
 *
 * If a template, return template with all template parameters fully resolved.
 *
 * This is a copy and modification of typemap_identifier_fix.
 * ----------------------------------------------------------------------------- */

static String *feature_identifier_fix(String *s) {
  String *tp = SwigType_istemplate_templateprefix(s);
  if (tp) {
    String *ts, *ta, *tq;
    ts = SwigType_templatesuffix(s);
    ta = SwigType_templateargs(s);
    tq = Swig_symbol_type_qualify(ta,0);
    Append(tp,tq);
    Append(tp,ts);
    Delete(ts);
    Delete(ta);
    Delete(tq);
    return tp;
  } else {
    return NewString(s);
  }
}

static void set_access_mode(Node *n) {
  if (cplus_mode == CPLUS_PUBLIC)
    Setattr(n, "access", "public");
  else if (cplus_mode == CPLUS_PROTECTED)
    Setattr(n, "access", "protected");
  else
    Setattr(n, "access", "private");
}

static void restore_access_mode(Node *n) {
  String *mode = Getattr(n, "access");
  if (Strcmp(mode, "private") == 0)
    cplus_mode = CPLUS_PRIVATE;
  else if (Strcmp(mode, "protected") == 0)
    cplus_mode = CPLUS_PROTECTED;
  else
    cplus_mode = CPLUS_PUBLIC;
}

/* Generate the symbol table name for an object */
/* This is a bit of a mess. Need to clean up */
static String *add_oldname = 0;



static String *make_name(Node *n, String *name,SwigType *decl) {
  String *made_name = 0;
  int destructor = name && (*(Char(name)) == '~');

  if (yyrename) {
    String *s = NewString(yyrename);
    Delete(yyrename);
    yyrename = 0;
    if (destructor  && (*(Char(s)) != '~')) {
      Insert(s,0,"~");
    }
    return s;
  }

  if (!name) return 0;

  if (parsing_template_declaration)
    SetFlag(n, "parsing_template_declaration");
  made_name = Swig_name_make(n, Namespaceprefix, name, decl, add_oldname);
  Delattr(n, "parsing_template_declaration");

  return made_name;
}

/* Generate an unnamed identifier */
static String *make_unnamed(void) {
  unnamed++;
  return NewStringf("$unnamed%d$",unnamed);
}

static int is_operator(String *name) {
  return Strncmp(name,"operator ", 9) == 0;
}

/* Add declaration list to symbol table */
static int  add_only_one = 0;

static void add_symbols(Node *n) {
  String *decl;
  String *wrn = 0;

  if (inclass && n) {
    cparse_normalize_void(n);
  }
  while (n) {
    String *symname = 0;
    String *old_prefix = 0;
    Symtab *old_scope = 0;
    int isfriend = inclass && Strstr(Getattr(n, "storage"), "friend") != NULL;
    int iscdecl = Cmp(nodeType(n),"cdecl") == 0;
    int only_csymbol = 0;
    
    if (inclass) {
      String *name = Getattr(n, "name");
      if (isfriend) {
	/* Friends methods in a class are declared in the namespace enclosing the class (outer most class if a nested class) */
	String *prefix = name ? Swig_scopename_prefix(name) : 0;
	Node *outer = currentOuterClass;
	Symtab *namespace_symtab;
	old_prefix = Namespaceprefix;
	old_scope = Swig_symbol_current();

	assert(outer);
	while (Getattr(outer, "nested:outer")) {
	  outer = Getattr(outer, "nested:outer");
	}
	namespace_symtab = Getattr(outer, "sym:symtab");
	if (!namespace_symtab)
	  namespace_symtab = Getattr(outer, "unofficial:symtab");
	Swig_symbol_setscope(namespace_symtab);
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);

	if (!prefix) {
	  /* To check - this should probably apply to operators too */
	  if (name && !is_operator(name) && Namespaceprefix) {
	    String *friendusing = NewStringf("using namespace %s;", Namespaceprefix);
	    Setattr(n, "friendusing", friendusing);
	    Delete(friendusing);
	  }
	} else {
	  /* Qualified friend declarations should not be possible as they are ignored in the parse tree */
	  assert(0);
	}
      } else if (Equal(nodeType(n), "using")) {
	String *uname = Getattr(n, "uname");
	Node *cls = currentOuterClass;
	String *nprefix = 0;
	String *nlast = 0;
	Swig_scopename_split(uname, &nprefix, &nlast);
	if (Swig_item_in_list(Getattr(cls, "baselist"), nprefix) || Swig_item_in_list(Getattr(cls, "protectedbaselist"), nprefix) || Swig_item_in_list(Getattr(cls, "privatebaselist"), nprefix)) {
	  String *plain_name = SwigType_istemplate(nprefix) ? SwigType_templateprefix(nprefix) : nprefix;
	  if (Equal(nlast, plain_name)) {
	    /* Using declaration looks like it is using a constructor in an immediate base class - change the constructor name for this class.
	     * C++11 requires using declarations for inheriting base constructors to be in the immediate base class.
	     * Note that we don't try and look up the constructor in the base class as the constructor may be an implicit/implied constructor and hence not exist. */
	    Symtab *stab = Swig_symbol_current();
	    String *nname = Getattr(stab, "name");
	    Setattr(n, "name", nname);
	    SetFlag(n, "usingctor");
	  }
	}
      } else {
	/* for member functions, we need to remove the redundant
	   class scope if provided, as in
	   
	   struct Foo {
	   int Foo::method(int a);
	   };
	   
	*/
	String *prefix = name ? Swig_scopename_prefix(name) : 0;
	if (prefix) {
	  if (Classprefix && (Equal(prefix,Classprefix))) {
	    String *base = Swig_scopename_last(name);
	    Setattr(n,"name",base);
	    Delete(base);
	  }
	  Delete(prefix);
	}
      }
    }

    if (!isfriend && (inclass || extendmode)) {
      Setattr(n,"ismember","1");
    }

    if (extendmode) {
      if (!Getattr(n, "template"))
        SetFlag(n,"isextendmember");
    }

    if (!isfriend && inclass) {
      if ((cplus_mode != CPLUS_PUBLIC)) {
	only_csymbol = 1;
	if (cplus_mode == CPLUS_PROTECTED) {
	  Setattr(n,"access", "protected");
	  only_csymbol = !Swig_need_protected(n);
	} else {
	  Setattr(n,"access", "private");
	  /* private are needed only when they are pure virtuals - why? */
	  if ((Cmp(Getattr(n,"storage"),"virtual") == 0) && (Cmp(Getattr(n,"value"),"0") == 0)) {
	    only_csymbol = 0;
	  }
	  if (Cmp(nodeType(n),"destructor") == 0) {
	    /* Needed for "unref" feature */
	    only_csymbol = 0;
	  }
	}
      } else {
	Setattr(n, "access", "public");
      }
    } else if (extendmode && !inclass) {
      Setattr(n, "access", "public");
    }

    if (Getattr(n,"sym:name")) {
      n = nextSibling(n);
      continue;
    }
    decl = Getattr(n,"decl");
    if (!SwigType_isfunction(decl)) {
      String *name = Getattr(n,"name");
      String *makename = Getattr(n,"parser:makename");
      if (iscdecl) {	
	String *storage = Getattr(n, "storage");
	if (Cmp(storage,"typedef") == 0) {
	  Setattr(n,"kind","typedef");
	} else {
	  SwigType *type = Getattr(n,"type");
	  String *value = Getattr(n,"value");
	  Setattr(n,"kind","variable");
	  if (value && Len(value)) {
	    Setattr(n,"hasvalue","1");
	  }
	  if (type) {
	    SwigType *ty;
	    SwigType *tmp = 0;
	    if (decl) {
	      ty = tmp = Copy(type);
	      SwigType_push(ty,decl);
	    } else {
	      ty = type;
	    }
	    if (storage && (Strstr(storage, "constexpr") || (Strstr(storage, "static") && !SwigType_ismutable(ty)))) {
	      SetFlag(n, "hasconsttype");
	    }
	    Delete(tmp);
	  }
	  if (!type) {
	    Printf(stderr,"notype name %s\n", name);
	  }
	}
      }
      Swig_features_get(Swig_cparse_features(), Namespaceprefix, name, 0, n);
      if (makename) {
	symname = make_name(n, makename,0);
        Delattr(n,"parser:makename"); /* temporary information, don't leave it hanging around */
      } else {
        makename = name;
	symname = make_name(n, makename,0);
      }
      
      if (!symname) {
	symname = Copy(Getattr(n,"unnamed"));
      }
      if (symname) {
	if (parsing_template_declaration)
	  SetFlag(n, "parsing_template_declaration");
	wrn = Swig_name_warning(n, Namespaceprefix, symname,0);
	Delattr(n, "parsing_template_declaration");
      }
    } else {
      String *name = Getattr(n,"name");
      SwigType *fdecl = Copy(decl);
      SwigType *fun = SwigType_pop_function(fdecl);
      if (iscdecl) {	
	Setattr(n,"kind","function");
      }
      
      Swig_features_get(Swig_cparse_features(),Namespaceprefix,name,fun,n);

      symname = make_name(n, name,fun);
      if (parsing_template_declaration)
	SetFlag(n, "parsing_template_declaration");
      wrn = Swig_name_warning(n, Namespaceprefix,symname,fun);
      Delattr(n, "parsing_template_declaration");
      
      Delete(fdecl);
      Delete(fun);
      
    }
    if (!symname) {
      n = nextSibling(n);
      continue;
    }

    if (GetFlag(n, "valueignored")) {
      SWIG_WARN_NODE_BEGIN(n);
      Swig_warning(WARN_PARSE_ASSIGNED_VALUE, Getfile(n), Getline(n), "Value assigned to %s not used due to limited parsing implementation.\n", SwigType_namestr(Getattr(n, "name")));
      SWIG_WARN_NODE_END(n);
    }

    if (cparse_cplusplus) {
      String *value = Getattr(n, "value");
      if (value && Strcmp(value, "delete") == 0) {
	/* C++11 deleted definition / deleted function */
        SetFlag(n,"deleted");
        SetFlag(n,"feature:ignore");
      }
      if (SwigType_isrvalue_reference(Getattr(n, "refqualifier"))) {
	/* Ignore rvalue ref-qualifiers by default
	 * Use Getattr instead of GetFlag to handle explicit ignore and explicit not ignore */
	if (!(Getattr(n, "feature:ignore") || Strncmp(symname, "$ignore", 7) == 0)) {
	  SWIG_WARN_NODE_BEGIN(n);
	  Swig_warning(WARN_TYPE_RVALUE_REF_QUALIFIER_IGNORED, Getfile(n), Getline(n),
	      "Method with rvalue ref-qualifier %s ignored.\n", Swig_name_decl(n));
	  SWIG_WARN_NODE_END(n);
	  SetFlag(n, "feature:ignore");
	}
      }
      if (Equal(Getattr(n, "type"), "auto")) {
	/* Ignore functions with an auto return type and no trailing return type
	 * Use Getattr instead of GetFlag to handle explicit ignore and explicit not ignore */
	if (!(Getattr(n, "feature:ignore") || Strncmp(symname, "$ignore", 7) == 0)) {
	  SWIG_WARN_NODE_BEGIN(n);
	  if (SwigType_isfunction(Getattr(n, "decl")))
	    Swig_warning(WARN_CPP14_AUTO, Getfile(n), Getline(n), "Unable to deduce auto return type for '%s' (ignored).\n", Swig_name_decl(n));
	  else
	    Swig_warning(WARN_CPP11_AUTO, Getfile(n), Getline(n), "Unable to deduce auto type for variable '%s' (ignored).\n", Swig_name_decl(n));
	  SWIG_WARN_NODE_END(n);
	  SetFlag(n, "feature:ignore");
	}
      }
    }
    if (only_csymbol || GetFlag(n, "feature:ignore") || Strncmp(symname, "$ignore", 7) == 0) {
      /* Only add to C symbol table and continue */
      Swig_symbol_add(0, n);
      if (!only_csymbol && !GetFlag(n, "feature:ignore")) {
	/* Print the warning attached to $ignore name, if any */
        char *c = Char(symname) + 7;
	if (strlen(c)) {
	  SWIG_WARN_NODE_BEGIN(n);
	  Swig_warning(0,Getfile(n), Getline(n), "%s\n",c+1);
	  SWIG_WARN_NODE_END(n);
	}
	/* If the symbol was ignored via "rename" and is visible, set also feature:ignore*/
	SetFlag(n, "feature:ignore");
      }
      if (!GetFlag(n, "feature:ignore") && Strcmp(symname,"$ignore") == 0) {
	/* Add feature:ignore if the symbol was explicitly ignored, regardless of visibility */
	SetFlag(n, "feature:ignore");
      }
    } else {
      Node *c;
      if ((wrn) && (Len(wrn))) {
	String *metaname = symname;
	if (!Getmeta(metaname,"already_warned")) {
	  SWIG_WARN_NODE_BEGIN(n);
	  Swig_warning(0,Getfile(n),Getline(n), "%s\n", wrn);
	  SWIG_WARN_NODE_END(n);
	  Setmeta(metaname,"already_warned","1");
	}
      }
      c = Swig_symbol_add(symname,n);

      if (c != n) {
	/* symbol conflict attempting to add in the new symbol */
	if (Getattr(n,"sym:weak")) {
	  Setattr(n,"sym:name",symname);
	} else {
	  Swig_symbol_conflict_warn(n, c, symname, inclass);
        }
      }
    }
    /* restore the class scope if needed */
    if (isfriend) {
      Swig_symbol_setscope(old_scope);
      if (old_prefix) {
	Delete(Namespaceprefix);
	Namespaceprefix = old_prefix;
      }
    }
    Delete(symname);

    if (add_only_one) return;
    n = nextSibling(n);
  }
}


/* add symbols a parse tree node copy */

static void add_symbols_copy(Node *n) {
  int    emode = 0;
  while (n) {
    if (Equal(nodeType(n), "access")) {
      String *kind = Getattr(n,"kind");
      if (Strcmp(kind,"public") == 0) {
	cplus_mode = CPLUS_PUBLIC;
      } else if (Strcmp(kind,"private") == 0) {
	cplus_mode = CPLUS_PRIVATE;
      } else if (Strcmp(kind,"protected") == 0) {
	cplus_mode = CPLUS_PROTECTED;
      }
      n = nextSibling(n);
      continue;
    }

    add_oldname = Getattr(n,"sym:name");
    if ((add_oldname) || (Getattr(n,"sym:needs_symtab"))) {
      int old_inclass = -1;
      Node *oldCurrentOuterClass = 0;
      if (add_oldname) {
	DohIncref(add_oldname);
	/*  Disable this, it prevents %rename to work with templates */
	/* If already renamed, we used that name  */
	/*
	if (Strcmp(add_oldname, Getattr(n,"name")) != 0) {
	  Delete(yyrename);
	  yyrename = Copy(add_oldname);
	}
	*/
      }
      Delattr(n,"sym:needs_symtab");
      Delattr(n,"sym:name");

      add_only_one = 1;
      add_symbols(n);

      if (Getattr(n,"partialargs")) {
	Swig_symbol_cadd(Getattr(n,"partialargs"),n);
      }
      add_only_one = 0;
      if (Equal(nodeType(n), "class")) {
	/* add_symbols() above sets "sym:symtab", so "unofficial:symtab" is not required */
	old_inclass = inclass;
	oldCurrentOuterClass = currentOuterClass;
	inclass = 1;
	currentOuterClass = n;
	if (Strcmp(Getattr(n,"kind"),"class") == 0) {
	  cplus_mode = CPLUS_PRIVATE;
	} else {
	  cplus_mode = CPLUS_PUBLIC;
	}
      }
      if (Equal(nodeType(n), "extend")) {
	emode = cplus_mode;
	cplus_mode = CPLUS_PUBLIC;
      }

      if (Getattr(n, "requires_symtab")) {
	Swig_symbol_newscope();
	Swig_symbol_setscopename(Getattr(n, "name"));
	Delete(Namespaceprefix);
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);
      }

      add_symbols_copy(firstChild(n));

      if (Equal(nodeType(n), "extend")) {
	cplus_mode = emode;
      }
      if (Getattr(n,"requires_symtab")) {
	Setattr(n,"symtab", Swig_symbol_popscope());
	Delattr(n,"requires_symtab");
	Delete(Namespaceprefix);
	Namespaceprefix = Swig_symbol_qualifiedscopename(0);
      }
      if (add_oldname) {
	Delete(add_oldname);
	add_oldname = 0;
      }
      if (Equal(nodeType(n), "class")) {
	inclass = old_inclass;
	currentOuterClass = oldCurrentOuterClass;
      }
    } else {
      if (Equal(nodeType(n), "extend")) {
	emode = cplus_mode;
	cplus_mode = CPLUS_PUBLIC;
      }
      add_symbols_copy(firstChild(n));
      if (Equal(nodeType(n), "extend")) {
	cplus_mode = emode;
      }
    }
    n = nextSibling(n);
  }
}

/* Add in the "defaultargs" attribute for functions in instantiated templates.
 * n should be any instantiated template (class or start of linked list of functions). */
static void update_defaultargs(Node *n) {
  if (n) {
    Node *firstdefaultargs = n;
    update_defaultargs(firstChild(n));
    n = nextSibling(n);
    /* recursively loop through nodes of all types, but all we really need are the overloaded functions */
    while (n) {
      update_defaultargs(firstChild(n));
      if (!Getattr(n, "defaultargs")) {
	if (Getattr(n, "needs_defaultargs")) {
	  Setattr(n, "defaultargs", firstdefaultargs);
	  Delattr(n, "needs_defaultargs");
	} else {
	  firstdefaultargs = n;
	}
      } else {
	/* Functions added in with %extend (for specialized template classes) will already have default args patched up */
	assert(Getattr(n, "defaultargs") == firstdefaultargs);
      }
      n = nextSibling(n);
    }
  }
}

/* Check a set of declarations to see if any are pure-abstract */

static List *pure_abstracts(Node *n) {
  List *abstracts = 0;
  while (n) {
    if (Cmp(nodeType(n),"cdecl") == 0) {
      String *decl = Getattr(n,"decl");
      if (SwigType_isfunction(decl)) {
	String *init = Getattr(n,"value");
	if (Cmp(init,"0") == 0) {
	  if (!abstracts) {
	    abstracts = NewList();
	  }
	  Append(abstracts,n);
	  SetFlag(n,"abstract");
	}
      }
    } else if (Cmp(nodeType(n),"destructor") == 0) {
      if (Cmp(Getattr(n,"value"),"0") == 0) {
	if (!abstracts) {
	  abstracts = NewList();
	}
	Append(abstracts,n);
	SetFlag(n,"abstract");
      }
    }
    n = nextSibling(n);
  }
  return abstracts;
}

/* Recompute the "abstracts" attribute for the classes in instantiated templates, similarly to update_defaultargs() above. */
static void update_abstracts(Node *n) {
  for (; n; n = nextSibling(n)) {
    Node* const child = firstChild(n);
    if (!child)
      continue;

    update_abstracts(child);

    if (Getattr(n, "needs_abstracts")) {
      Setattr(n, "abstracts", pure_abstracts(child));
      Delattr(n, "needs_abstracts");
    }
  }
}

/* Make a classname */

static String *make_class_name(String *name) {
  String *nname = 0;
  String *prefix;
  if (Namespaceprefix) {
    nname= NewStringf("%s::%s", Namespaceprefix, name);
  } else {
    nname = NewString(name);
  }
  prefix = SwigType_istemplate_templateprefix(nname);
  if (prefix) {
    String *args, *qargs;
    args   = SwigType_templateargs(nname);
    qargs  = Swig_symbol_type_qualify(args,0);
    Append(prefix,qargs);
    Delete(nname);
    Delete(args);
    Delete(qargs);
    nname = prefix;
  }
  return nname;
}

/* Use typedef name as class name */

static void add_typedef_name(Node *n, Node *declnode, String *oldName, Symtab *cscope, String *scpname) {
  String *class_rename = 0;
  SwigType *decl = Getattr(declnode, "decl");
  if (!decl || !Len(decl)) {
    String *cname;
    String *tdscopename;
    String *class_scope = Swig_symbol_qualifiedscopename(cscope);
    String *name = Getattr(declnode, "name");
    cname = Copy(name);
    Setattr(n, "tdname", cname);
    tdscopename = class_scope ? NewStringf("%s::%s", class_scope, name) : Copy(name);
    class_rename = Getattr(n, "class_rename");
    if (class_rename && (Strcmp(class_rename, oldName) == 0))
      Setattr(n, "class_rename", NewString(name));
    if (!classes_typedefs) classes_typedefs = NewHash();
    if (!Equal(scpname, tdscopename) && !Getattr(classes_typedefs, tdscopename)) {
      Setattr(classes_typedefs, tdscopename, n);
    }
    Setattr(n, "decl", decl);
    Delete(class_scope);
    Delete(cname);
    Delete(tdscopename);
  }
}

/* If the class name is qualified.  We need to create or lookup namespace entries */

static Symtab *set_scope_to_global(void) {
  Symtab *symtab = Swig_symbol_global_scope();
  Swig_symbol_setscope(symtab);
  return symtab;
}
 
/* Remove the block braces, { and }, if the 'noblock' attribute is set.
 * Node *kw can be either a Hash or Parmlist. */
static String *remove_block(Node *kw, const String *inputcode) {
  String *modified_code = 0;
  while (kw) {
   String *name = Getattr(kw,"name");
   if (name && (Cmp(name,"noblock") == 0)) {
     char *cstr = Char(inputcode);
     int len = Len(inputcode);
     if (len && cstr[0] == '{') {
       --len; ++cstr; 
       if (len && cstr[len - 1] == '}') { --len; }
       /* we now remove the extra spaces */
       while (len && isspace((int)cstr[0])) { --len; ++cstr; }
       while (len && isspace((int)cstr[len - 1])) { --len; }
       modified_code = NewStringWithSize(cstr, len);
       break;
     }
   }
   kw = nextSibling(kw);
  }
  return modified_code;
}

/*
#define RESOLVE_DEBUG 1
*/
static Node *nscope = 0;
static Node *nscope_inner = 0;

/* Remove the scope prefix from cname and return the base name without the prefix.
 * The scopes required for the symbol name are resolved and/or created, if required.
 * For example AA::BB::CC as input returns CC and creates the namespace AA then inner 
 * namespace BB in the current scope. */
static String *resolve_create_node_scope(String *cname_in, int is_class_definition, int *errored) {
  Symtab *gscope = 0;
  Node *cname_node = 0;
  String *cname = cname_in;
  String *last = Swig_scopename_last(cname);
  nscope = 0;
  nscope_inner = 0;  
  *errored = 0;

  if (Strncmp(cname, "::", 2) == 0) {
    if (is_class_definition) {
      Swig_error(cparse_file, cparse_line, "Using the unary scope operator :: in class definition '%s' is invalid.\n", SwigType_namestr(cname));
      *errored = 1;
      return last;
    }
    cname = NewString(Char(cname) + 2);
  }
  if (is_class_definition) {
    /* Only lookup symbols which are in scope via a using declaration but not via a using directive.
       For example find y via 'using x::y' but not y via a 'using namespace x'. */
    cname_node = Swig_symbol_clookup_no_inherit(cname, 0);
    if (!cname_node) {
      Node *full_lookup_node = Swig_symbol_clookup(cname, 0);
      if (full_lookup_node) {
       /* This finds a symbol brought into scope via both a using directive and a using declaration. */
	Node *last_node = Swig_symbol_clookup_no_inherit(last, 0);
	if (last_node == full_lookup_node)
	  cname_node = last_node;
      }
    }
  } else {
    /* For %template, the template needs to be in scope via any means. */
    cname_node = Swig_symbol_clookup(cname, 0);
  }
#if RESOLVE_DEBUG
  if (!cname_node)
    Printf(stdout, "symbol does not yet exist (%d): [%s]\n", is_class_definition, cname_in);
  else
    Printf(stdout, "symbol does exist (%d): [%s]\n", is_class_definition, cname_in);
#endif

  if (cname_node) {
    /* The symbol has been defined already or is in another scope.
       If it is a weak symbol, it needs replacing and if it was brought into the current scope,
       the scope needs adjusting appropriately for the new symbol.
       Similarly for defined templates. */
    Symtab *symtab = Getattr(cname_node, "sym:symtab");
    Node *sym_weak = Getattr(cname_node, "sym:weak");
    if ((symtab && sym_weak) || Equal(nodeType(cname_node), "template")) {
      /* Check if the scope is the current scope */
      String *current_scopename = Swig_symbol_qualifiedscopename(0);
      String *found_scopename = Swig_symbol_qualifiedscopename(symtab);
      if (!current_scopename)
	current_scopename = NewString("");
      if (!found_scopename)
	found_scopename = NewString("");

      {
	int fail = 1;
	List *current_scopes = Swig_scopename_tolist(current_scopename);
	List *found_scopes = Swig_scopename_tolist(found_scopename);
        Iterator cit = First(current_scopes);
	Iterator fit = First(found_scopes);
#if RESOLVE_DEBUG
Printf(stdout, "comparing current: [%s] found: [%s]\n", current_scopename, found_scopename);
#endif
	for (; fit.item && cit.item; fit = Next(fit), cit = Next(cit)) {
	  String *current = cit.item;
	  String *found = fit.item;
#if RESOLVE_DEBUG
	  Printf(stdout, "  looping %s %s\n", current, found);
#endif
	  if (Strcmp(current, found) != 0)
	    break;
	}

	if (!cit.item) {
	  String *subscope = NewString("");
	  for (; fit.item; fit = Next(fit)) {
	    if (Len(subscope) > 0)
	      Append(subscope, "::");
	    Append(subscope, fit.item);
	  }
	  if (Len(subscope) > 0)
	    cname = NewStringf("%s::%s", subscope, last);
	  else
	    cname = Copy(last);
#if RESOLVE_DEBUG
	  Printf(stdout, "subscope to create: [%s] cname: [%s]\n", subscope, cname);
#endif
	  fail = 0;
	  Delete(subscope);
	} else {
	  if (is_class_definition) {
	    if (!fit.item) {
	      /* It is valid to define a new class with the same name as one forward declared in a parent scope */
	      fail = 0;
	    } else if (Swig_scopename_check(cname)) {
	      /* Classes defined with scope qualifiers must have a matching forward declaration in matching scope */
	      fail = 1;
	    } else {
	      /* This may let through some invalid cases */
	      fail = 0;
	    }
#if RESOLVE_DEBUG
	    Printf(stdout, "scope for class definition, fail: %d\n", fail);
#endif
	  } else {
#if RESOLVE_DEBUG
	    Printf(stdout, "no matching base scope for template\n");
#endif
	    fail = 1;
	  }
	}

	Delete(found_scopes);
	Delete(current_scopes);

	if (fail) {
	  String *cname_resolved = NewStringf("%s::%s", found_scopename, last);
	  Swig_error(cparse_file, cparse_line, "'%s' resolves to '%s' and was incorrectly instantiated in scope '%s' instead of within scope '%s'.\n",
	    SwigType_namestr(cname_in), SwigType_namestr(cname_resolved), SwigType_namestr(current_scopename), SwigType_namestr(found_scopename));
	  *errored = 1;
	  Delete(cname_resolved);
	}
      }

      Delete(current_scopename);
      Delete(found_scopename);
    }
  } else if (!is_class_definition) {
    /* A template instantiation requires a template to be found in scope */
    Swig_error(cparse_file, cparse_line, "Template '%s' undefined.\n", SwigType_namestr(cname_in));
    *errored = 1;
  }

  if (*errored)
    return last;

  if (Swig_scopename_check(cname) && !*errored) {
    Node   *ns;
    String *prefix = Swig_scopename_prefix(cname);
    if (Len(prefix) == 0) {
      String *base = Copy(last);
      /* Use the global scope, but we need to add a 'global' namespace.  */
      if (!gscope) gscope = set_scope_to_global();
      /* note that this namespace is not the "unnamed" one,
	 and we don't use Setattr(nscope,"name", ""),
	 because the unnamed namespace is private */
      nscope = new_node("namespace");
      Setattr(nscope,"symtab", gscope);;
      nscope_inner = nscope;
      Delete(last);
      return base;
    }
    /* Try to locate the scope */
    ns = Swig_symbol_clookup(prefix,0);
    if (!ns) {
      Swig_error(cparse_file, cparse_line, "Undefined scope '%s'\n", SwigType_namestr(prefix));
      *errored = 1;
    } else {
      Symtab *nstab = Getattr(ns,"symtab");
      if (!nstab) {
	Swig_error(cparse_file, cparse_line, "'%s' is not defined as a valid scope.\n", SwigType_namestr(prefix));
	*errored = 1;
	ns = 0;
      } else {
	/* Check if the node scope is the current scope */
	String *tname = Swig_symbol_qualifiedscopename(0);
	String *nname = Swig_symbol_qualifiedscopename(nstab);
	if (tname && (Strcmp(tname,nname) == 0)) {
	  ns = 0;
	  cname = Copy(last);
	}
	Delete(tname);
	Delete(nname);
      }
      if (ns) {
	/* we will try to create a new node using the namespaces we
	   can find in the scope name */
	List *scopes = Swig_scopename_tolist(prefix);
	String *sname;
	Iterator si;

	for (si = First(scopes); si.item; si = Next(si)) {
	  Node *ns1,*ns2;
	  sname = si.item;
	  ns1 = Swig_symbol_clookup(sname,0);
	  assert(ns1);
	  if (Strcmp(nodeType(ns1),"namespace") == 0) {
	    if (Getattr(ns1,"alias")) {
	      ns1 = Getattr(ns1,"namespace");
	    }
	  } else {
	    /* now this last part is a class */
	    si = Next(si);
	    /*  or a nested class tree, which is unrolled here */
	    for (; si.item; si = Next(si)) {
	      if (si.item) {
		Printf(sname,"::%s",si.item);
	      }
	    }
	    /* we get the 'inner' class */
	    nscope_inner = Swig_symbol_clookup(sname,0);
	    /* set the scope to the inner class */
	    Swig_symbol_setscope(Getattr(nscope_inner,"symtab"));
	    /* save the last namespace prefix */
	    Delete(Namespaceprefix);
	    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	    /* and return the node name, including the inner class prefix */
	    break;
	  }
	  /* here we just populate the namespace tree as usual */
	  ns2 = new_node("namespace");
	  Setattr(ns2,"name",sname);
	  Setattr(ns2,"symtab", Getattr(ns1,"symtab"));
	  add_symbols(ns2);
	  Swig_symbol_setscope(Getattr(ns1,"symtab"));
	  Delete(Namespaceprefix);
	  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	  if (nscope_inner) {
	    if (Getattr(nscope_inner,"symtab") != Getattr(ns2,"symtab")) {
	      appendChild(nscope_inner,ns2);
	      Delete(ns2);
	    }
	  }
	  nscope_inner = ns2;
	  if (!nscope) nscope = ns2;
	}
	cname = Copy(last);
	Delete(scopes);
      }
    }
    Delete(prefix);
  }
  Delete(last);

  return cname;
}
 
/* look for simple typedef name in typedef list */
static String *try_to_find_a_name_for_unnamed_structure(const String *storage, Node *decls) {
  String *name = 0;
  Node *n = decls;
  if (storage && Equal(storage, "typedef")) {
    for (; n; n = nextSibling(n)) {
      if (!Len(Getattr(n, "decl"))) {
	name = Copy(Getattr(n, "name"));
	break;
      }
    }
  }
  return name;
}

/* traverse copied tree segment, and update outer class links*/
static void update_nested_classes(Node *n)
{
  Node *c = firstChild(n);
  while (c) {
    if (Getattr(c, "nested:outer"))
      Setattr(c, "nested:outer", n);
    update_nested_classes(c);
    c = nextSibling(c);
  }
}

/* -----------------------------------------------------------------------------
 * nested_forward_declaration()
 * 
 * Nested struct handling for C++ code if the nested classes are disabled.
 * Create the nested class/struct/union as a forward declaration.
 * ----------------------------------------------------------------------------- */

static Node *nested_forward_declaration(const String *storage, const String *kind, String *sname, String *name, Node *cpp_opt_declarators) {
  Node *nn = 0;

  if (sname) {
    /* Add forward declaration of the nested type */
    Node *n = new_node("classforward");
    Setattr(n, "kind", kind);
    Setattr(n, "name", sname);
    Setattr(n, "storage", storage);
    Setattr(n, "sym:weak", "1");
    SetFlag(n, "nested:forward");
    add_symbols(n);
    nn = n;
  }

  /* Add any variable instances. Also add in any further typedefs of the nested type.
     Note that anonymous typedefs (eg typedef struct {...} a, b;) are treated as class forward declarations */
  if (cpp_opt_declarators) {
    int storage_typedef = (storage && Equal(storage, "typedef"));
    int variable_of_anonymous_type = !sname && !storage_typedef;
    if (!variable_of_anonymous_type) {
      int anonymous_typedef = !sname && storage_typedef;
      Node *n = cpp_opt_declarators;
      SwigType *type = name;
      while (n) {
	Setattr(n, "type", type);
	Setattr(n, "storage", storage);
	if (anonymous_typedef) {
	  Setattr(n, "nodeType", "classforward");
	  Setattr(n, "sym:weak", "1");
	}
	n = nextSibling(n);
      }
      add_symbols(cpp_opt_declarators);

      if (nn) {
	set_nextSibling(nn, cpp_opt_declarators);
      } else {
	nn = cpp_opt_declarators;
      }
    }
  }

  if (!currentOuterClass || !GetFlag(currentOuterClass, "nested")) {
    if (nn && Equal(nodeType(nn), "classforward")) {
      Node *n = nn;
      if (!GetFlag(n, "feature:ignore")) {
	SWIG_WARN_NODE_BEGIN(n);
	Swig_warning(WARN_PARSE_NAMED_NESTED_CLASS, cparse_file, cparse_line, "Nested %s not currently supported (%s ignored)\n", kind, SwigType_namestr(sname ? sname : name));
	SWIG_WARN_NODE_END(n);
      }
    } else {
      Swig_warning(WARN_PARSE_UNNAMED_NESTED_CLASS, cparse_file, cparse_line, "Nested %s not currently supported (ignored).\n", kind);
    }
  }

  return nn;
}


Node *Swig_cparse(File *f) {
  scanner_file(f);
  top = 0;
  if (yyparse() == 2) {
    Swig_error(cparse_file, cparse_line, "Parser exceeded stack depth or ran out of memory\n");
    Exit(EXIT_FAILURE);
  }
  return (Node *)top;
}

static void single_new_feature(const char *featurename, String *val, Hash *featureattribs, char *declaratorid, SwigType *type, ParmList *declaratorparms, String *qualifier) {
  String *fname;
  String *name;
  String *fixname;
  SwigType *t = Copy(type);

  /* Printf(stdout, "single_new_feature: [%s] [%s] [%s] [%s] [%s] [%s]\n", featurename, val, declaratorid, t, ParmList_str_defaultargs(declaratorparms), qualifier); */

  fname = NewStringf("feature:%s",featurename);
  if (declaratorid) {
    fixname = feature_identifier_fix(declaratorid);
  } else {
    fixname = NewStringEmpty();
  }
  if (Namespaceprefix) {
    name = NewStringf("%s::%s",Namespaceprefix, fixname);
  } else {
    name = fixname;
  }

  if (declaratorparms) Setmeta(val,"parms",declaratorparms);
  if (!Len(t)) t = 0;
  if (t) {
    if (qualifier) SwigType_push(t,qualifier);
    if (SwigType_isfunction(t)) {
      SwigType *decl = SwigType_pop_function(t);
      if (SwigType_ispointer(t)) {
	String *nname = NewStringf("*%s",name);
	Swig_feature_set(Swig_cparse_features(), nname, decl, fname, val, featureattribs);
	Delete(nname);
      } else {
	Swig_feature_set(Swig_cparse_features(), name, decl, fname, val, featureattribs);
      }
      Delete(decl);
    } else if (SwigType_ispointer(t)) {
      String *nname = NewStringf("*%s",name);
      Swig_feature_set(Swig_cparse_features(),nname,0,fname,val, featureattribs);
      Delete(nname);
    }
  } else {
    /* Global feature, that is, feature not associated with any particular symbol */
    Swig_feature_set(Swig_cparse_features(),name,0,fname,val, featureattribs);
  }
  Delete(fname);
  Delete(name);
}

/* Add a new feature to the Hash. Additional features are added if the feature has a parameter list (declaratorparms)
 * and one or more of the parameters have a default argument. An extra feature is added for each defaulted parameter,
 * simulating the equivalent overloaded method. */
static void new_feature(const char *featurename, String *val, Hash *featureattribs, char *declaratorid, SwigType *type, ParmList *declaratorparms, String *qualifier) {

  ParmList *declparms = declaratorparms;

  /* remove the { and } braces if the noblock attribute is set */
  String *newval = remove_block(featureattribs, val);
  val = newval ? newval : val;

  /* Add the feature */
  single_new_feature(featurename, val, featureattribs, declaratorid, type, declaratorparms, qualifier);

  /* Add extra features if there are default parameters in the parameter list */
  if (type) {
    while (declparms) {
      if (ParmList_has_defaultargs(declparms)) {

        /* Create a parameter list for the new feature by copying all
           but the last (defaulted) parameter */
        ParmList* newparms = CopyParmListMax(declparms, ParmList_len(declparms)-1);

        /* Create new declaration - with the last parameter removed */
        SwigType *newtype = Copy(type);
        Delete(SwigType_pop_function(newtype)); /* remove the old parameter list from newtype */
        SwigType_add_function(newtype,newparms);

        single_new_feature(featurename, Copy(val), featureattribs, declaratorid, newtype, newparms, qualifier);
        declparms = newparms;
      } else {
        declparms = 0;
      }
    }
  }
}

/* check if a function declaration is a plain C object */
static int is_cfunction(Node *n) {
  if (!cparse_cplusplus || cparse_externc)
    return 1;
  if (Swig_storage_isexternc(n)) {
    return 1;
  }
  return 0;
}

/* If the Node is a function with parameters, check to see if any of the parameters
 * have default arguments. If so create a new function for each defaulted argument. 
 * The additional functions form a linked list of nodes with the head being the original Node n. */
static void default_arguments(Node *n) {
  Node *function = n;

  if (function) {
    ParmList *varargs = Getattr(function,"feature:varargs");
    if (varargs) {
      /* Handles the %varargs directive by looking for "feature:varargs" and 
       * substituting ... with an alternative set of arguments.  */
      Parm     *p = Getattr(function,"parms");
      Parm     *pp = 0;
      while (p) {
	SwigType *t = Getattr(p,"type");
	if (Strcmp(t,"v(...)") == 0) {
	  if (pp) {
	    ParmList *cv = Copy(varargs);
	    set_nextSibling(pp,cv);
	    Delete(cv);
	  } else {
	    ParmList *cv =  Copy(varargs);
	    Setattr(function,"parms", cv);
	    Delete(cv);
	  }
	  break;
	}
	pp = p;
	p = nextSibling(p);
      }
    }

    /* Do not add in functions if kwargs is being used or if user wants old default argument wrapping
       (one wrapped method per function irrespective of number of default arguments) */
    if (compact_default_args 
	|| is_cfunction(function) 
	|| GetFlag(function,"feature:compactdefaultargs") 
	|| (GetFlag(function,"feature:kwargs") && kwargs_supported)) {
      ParmList *p = Getattr(function,"parms");
      if (p) 
        Setattr(p,"compactdefargs", "1"); /* mark parameters for special handling */
      function = 0; /* don't add in extra methods */
    }
  }

  while (function) {
    ParmList *parms = Getattr(function,"parms");
    if (ParmList_has_defaultargs(parms)) {

      /* Create a parameter list for the new function by copying all
         but the last (defaulted) parameter */
      ParmList* newparms = CopyParmListMax(parms,ParmList_len(parms)-1);

      /* Create new function and add to symbol table */
      {
	SwigType *ntype = Copy(nodeType(function));
	char *cntype = Char(ntype);
        Node *new_function = new_node(ntype);
        SwigType *decl = Copy(Getattr(function,"decl"));
        int constqualifier = SwigType_isconst(decl);
	String *ccode = Copy(Getattr(function,"code"));
	String *cstorage = Copy(Getattr(function,"storage"));
	String *cvalue = Copy(Getattr(function,"value"));
	SwigType *ctype = Copy(Getattr(function,"type"));
	String *cthrow = Copy(Getattr(function,"throw"));

        Delete(SwigType_pop_function(decl)); /* remove the old parameter list from decl */
        SwigType_add_function(decl,newparms);
        if (constqualifier)
          SwigType_add_qualifier(decl,"const");

        Setattr(new_function,"name", Getattr(function,"name"));
        Setattr(new_function,"code", ccode);
        Setattr(new_function,"decl", decl);
        Setattr(new_function,"parms", newparms);
        Setattr(new_function,"storage", cstorage);
        Setattr(new_function,"value", cvalue);
        Setattr(new_function,"type", ctype);
        Setattr(new_function,"throw", cthrow);

	Delete(ccode);
	Delete(cstorage);
	Delete(cvalue);
	Delete(ctype);
	Delete(cthrow);
	Delete(decl);

        {
          Node *throws = Getattr(function,"throws");
	  ParmList *pl = CopyParmList(throws);
          if (throws) Setattr(new_function,"throws",pl);
	  Delete(pl);
        }

        /* copy specific attributes for global (or in a namespace) template functions - these are not class template methods */
        if (strcmp(cntype,"template") == 0) {
          Node *templatetype = Getattr(function,"templatetype");
          Node *symtypename = Getattr(function,"sym:typename");
          Parm *templateparms = Getattr(function,"templateparms");
          if (templatetype) {
	    Node *tmp = Copy(templatetype);
	    Setattr(new_function,"templatetype",tmp);
	    Delete(tmp);
	  }
          if (symtypename) {
	    Node *tmp = Copy(symtypename);
	    Setattr(new_function,"sym:typename",tmp);
	    Delete(tmp);
	  }
          if (templateparms) {
	    Parm *tmp = CopyParmList(templateparms);
	    Setattr(new_function,"templateparms",tmp);
	    Delete(tmp);
	  }
        } else if (strcmp(cntype,"constructor") == 0) {
          /* only copied for constructors as this is not a user defined feature - it is hard coded in the parser */
          if (GetFlag(function,"feature:new")) SetFlag(new_function,"feature:new");
        }

        add_symbols(new_function);
        /* mark added functions as ones with overloaded parameters and point to the parsed method */
        Setattr(new_function,"defaultargs", n);

        /* Point to the new function, extending the linked list */
        set_nextSibling(function, new_function);
	Delete(new_function);
        function = new_function;
	
	Delete(ntype);
      }
    } else {
      function = 0;
    }
  }
}

/* -----------------------------------------------------------------------------
 * mark_nodes_as_extend()
 *
 * Used by the %extend to mark subtypes with "feature:extend".
 * template instances declared within %extend are skipped
 * ----------------------------------------------------------------------------- */

static void mark_nodes_as_extend(Node *n) {
  for (; n; n = nextSibling(n)) {
    if (Getattr(n, "template") && Strcmp(nodeType(n), "class") == 0)
      continue;
    /* Fix me: extend is not a feature. Replace with isextendmember? */
    Setattr(n, "feature:extend", "1");
    mark_nodes_as_extend(firstChild(n));
  }
}

/* -----------------------------------------------------------------------------
 * add_qualifier_to_declarator()
 *
 * Normally the qualifier is pushed on to the front of the type.
 * Adding a qualifier to a pointer to member function is a special case.
 * For example       : typedef double (Cls::*pmf)(void) const;
 * The qualifier is  : q(const).
 * The declarator is : m(Cls).f(void).
 * We need           : m(Cls).q(const).f(void).
 * ----------------------------------------------------------------------------- */

static String *add_qualifier_to_declarator(SwigType *type, SwigType *qualifier) {
  int is_pointer_to_member_function = 0;
  String *decl = Copy(type);
  String *poppedtype = NewString("");
  assert(qualifier);

  while (decl) {
    if (SwigType_ismemberpointer(decl)) {
      String *memberptr = SwigType_pop(decl);
      if (SwigType_isfunction(decl)) {
	is_pointer_to_member_function = 1;
	SwigType_push(decl, qualifier);
	SwigType_push(decl, memberptr);
	Insert(decl, 0, poppedtype);
	Delete(memberptr);
	break;
      } else {
	Append(poppedtype, memberptr);
      }
      Delete(memberptr);
    } else {
      String *popped = SwigType_pop(decl);
      if (!popped)
	break;
      Append(poppedtype, popped);
      Delete(popped);
    }
  }

  if (!is_pointer_to_member_function) {
    Delete(decl);
    decl = Copy(type);
    SwigType_push(decl, qualifier);
  }

  Delete(poppedtype);
  return decl;
}

%}

%union {
  const char  *id;
  List  *bases;
  struct Define {
    // The value of the expression as C/C++ code.
    String *val;
    // If type is a string or char type, this is the actual value of that
    // string or char type as a String (in cases where SWIG can determine
    // it - currently that means for literals).  This is useful in cases where
    // we want to emit a string or character literal in the target language -
    // we could just try emitting the C/C++ code for the literal, but that
    // won't always be correct in most target languages.
    //
    // SWIG's scanner reads the string or character literal in the source code
    // and interprets quoting and escape sequences.  Concatenation of adjacent
    // string literals is currently handled here in the parser (though
    // technically it should happen before parsing).
    //
    // stringval holds the actual value of the string (from the scanner,
    // taking into account concatenation of adjacent string literals).
    // Then val is created by escaping stringval using SWIG's %(escape)s
    // Printf specification, and adding the appropriate quotes (and
    // an L-prefix for wide literals).  So val is also a C/C++ source
    // representation of the string, but may not be the same representation
    // as in the source code (it should be equivalent though).
    //
    // Some examples:
    //
    // C/C++ source  stringval   val       Notes
    // ------------- ----------- --------- -------
    // "bar"         bar         "bar"
    // "b\x61r"      bar         "bar"
    // "b\141r"      bar         "bar"
    // "b" "ar"      bar         "bar"
    // u8"bar"       bar         "bar"     C++11
    // R"bar"        bar         "bar"     C++11
    // "\228\22"     "8"         "\"8\""
    // "\\\"\'"      \"'         "\\\"\'"
    // R"(\"')"      \"'         "\\\"\'"  C++11
    // L"bar"        bar         L"bar"
    // L"b" L"ar"    bar         L"bar"
    // L"b" "ar"     bar         L"bar"    C++11
    // "b" L"ar"     bar         L"bar"    C++11
    // 'x'           x           'x'
    // '\"'          "           '\"'
    // '\42'         "           '\"'
    // '\042'        "           '\"'
    // '\x22'        "           '\"'
    //
    // Zero bytes are allowed in stringval (DOH's String can hold a string
    // with embedded zero bytes), but handling may currently be buggy in
    // places.
    String *stringval;
    // If type is an integer or boolean type, this is the actual value of that
    // type as a base 10 integer in a String (in cases where SWIG can determine
    // this value - currently that means for literals).  This is useful in
    // cases where we want to emit an integer or boolean literal in the target
    // language - we could just try emitting the C/C++ code for the literal,
    // but that won't always be correct in most target languages.
    //
    // SWIG doesn't attempt to evaluate constant expressions, except that it
    // can handle unary - (because a negative integer literal is actually
    // syntactically unary minus applied to a positive integer literal),
    // unary + (for consistency with unary -) and parentheses (because
    // literals in #define are often in parentheses).  These operators are
    // handled in the parser so whitespace is also handled within such
    // expressions.
    //
    // Some examples:
    //
    // C/C++ source  numval      val       Notes
    // ------------- ----------- --------- -------
    // 123           123         123
    // 0x7b          123         0x7b
    // 0x7B          123         0x7B
    // 0173          123         0173
    // 0b1111011     123         0b1111011 C++14
    // -10           -10         -10	   numval not set for unsigned type
    // -0x00a        -10         -0x00a    numval not set for unsigned type
    // -012          -10         -012      numval not set for unsigned type
    // -0b1010       -10         -0b1010   C++14; numval not set for unsigned
    // (42)          42          (42)
    // +42           42          +42
    // +(42)         42          +(42)
    // -(42)         -42         -(42)     numval not set for unsigned type
    // (-(42))       -42         (-(42))   numval not set for unsigned type
    // false         0           false
    // (false)       0           (false)
    // true          1           true
    // (true)        1           (true)
    String *numval;
    int     type;
    /* The type code for the argument when the top level operator is unary.
     * This is useful because our grammar parses cases such as (7)*6 as a
     * cast applied to an unary operator.
     */
    int	    unary_arg_type;
    String *qualifier;
    String *refqualifier;
    String *bitfield;
    Parm   *throws;
    String *throwf;
    String *nexcept;
    String *final;
  } dtype;
  struct {
    String *filename;
    int   line;
  } loc;
  struct Decl {
    char      *id;
    SwigType  *type;
    String    *defarg;
    String    *stringdefarg;
    String    *numdefarg;
    ParmList  *parms;
    short      have_parms;
    ParmList  *throws;
    String    *throwf;
    String    *nexcept;
    String    *final;
  } decl;
  Parm         *tparms;
  struct {
    String     *method;
    Hash       *kwargs;
  } tmap;
  struct {
    String     *type;
    String     *us;
  } ptype;
  SwigType     *type;
  String       *str;
  Parm         *p;
  ParmList     *pl;
  int           intvalue;
  enum { INCLUDE_INCLUDE, INCLUDE_IMPORT } includetype;
  Node         *node;
  struct {
    Parm       *parms;
    Parm       *last;
  } pbuilder;
  struct {
    Node       *node;
    Node       *last;
  } nodebuilder;
};

/* Define special token END for end of input. */
%token END 0

%token <id> ID
%token <str> HBLOCK
%token <id> POUND 
%token <str> STRING WSTRING
%token INCLUDE IMPORT INSERT
%token <str> CHARCONST WCHARCONST
%token <dtype> NUM_INT NUM_DOUBLE NUM_FLOAT NUM_LONGDOUBLE NUM_UNSIGNED NUM_LONG NUM_ULONG NUM_LONGLONG NUM_ULONGLONG NUM_BOOL
%token TYPEDEF
%token <type> TYPE_INT TYPE_UNSIGNED TYPE_SHORT TYPE_LONG TYPE_FLOAT TYPE_DOUBLE TYPE_CHAR TYPE_WCHAR TYPE_VOID TYPE_SIGNED TYPE_BOOL TYPE_COMPLEX TYPE_RAW TYPE_NON_ISO_INT8 TYPE_NON_ISO_INT16 TYPE_NON_ISO_INT32 TYPE_NON_ISO_INT64
%token LPAREN RPAREN COMMA SEMI EXTERN LBRACE RBRACE PERIOD ELLIPSIS
%token CONST_QUAL VOLATILE REGISTER STRUCT UNION EQUAL SIZEOF ALIGNOF MODULE LBRACKET RBRACKET
%token BEGINFILE ENDOFFILE
%token CONSTANT
%token RENAME NAMEWARN EXTEND PRAGMA FEATURE VARARGS
%token ENUM
%token CLASS TYPENAME PRIVATE PUBLIC PROTECTED COLON STATIC VIRTUAL FRIEND THROW CATCH EXPLICIT
%token STATIC_ASSERT CONSTEXPR THREAD_LOCAL DECLTYPE AUTO NOEXCEPT /* C++11 keywords */
%token OVERRIDE FINAL /* C++11 identifiers with special meaning */
%token USING
%token NAMESPACE
%token NATIVE INLINE
%token TYPEMAP ECHO APPLY CLEAR SWIGTEMPLATE FRAGMENT
%token WARN 
%token LESSTHAN GREATERTHAN DELETE_KW DEFAULT
%token LESSTHANOREQUALTO GREATERTHANOREQUALTO EQUALTO NOTEQUALTO LESSEQUALGREATER
%token ARROW
%token QUESTIONMARK
%token TYPES PARMS
%token NONID DSTAR DCNOT
%token TEMPLATE
%token <str> OPERATOR
%token <str> CONVERSIONOPERATOR
%token PARSETYPE PARSEPARM PARSEPARMS

%token <str> DOXYGENSTRING
%token <str> DOXYGENPOSTSTRING

%precedence CAST
%left  QUESTIONMARK
%left  LOR
%left  LAND
%left  OR
%left  XOR
%left  AND
%left  EQUALTO NOTEQUALTO
/* We don't currently allow < and > in any context where the associativity or
 * precedence matters and Bison warns about that.
 */
%left  /* GREATERTHAN LESSTHAN */ GREATERTHANOREQUALTO LESSTHANOREQUALTO
%left  LESSEQUALGREATER
%left  LSHIFT RSHIFT
%left  PLUS MINUS
%left  STAR SLASH MODULO
%precedence UMINUS NOT LNOT
%token DCOLON

%type <node>     program interface declaration swig_directive ;

/* SWIG directives */
%type <node>     extend_directive apply_directive clear_directive constant_directive ;
%type <node>     echo_directive fragment_directive include_directive inline_directive ;
%type <node>     insert_directive module_directive native_directive ;
%type <node>     pragma_directive rename_directive feature_directive varargs_directive typemap_directive ;
%type <node>     types_directive template_directive warn_directive ;

/* C declarations */
%type <node>     c_declaration c_decl c_decl_tail c_enum_decl c_enum_forward_decl c_constructor_decl;
%type <type>     c_enum_inherit;
%type <node>     enumlist enumlist_item edecl_with_dox edecl;
%type <id>       c_enum_key;

/* C++ declarations */
%type <node>     cpp_declaration cpp_class_decl cpp_forward_class_decl cpp_template_decl;
%type <node>     cpp_members cpp_member cpp_member_no_dox;
%type <nodebuilder> cpp_members_builder;
%type <node>     cpp_constructor_decl cpp_destructor_decl cpp_protection_decl cpp_conversion_operator cpp_static_assert;
%type <node>     cpp_swig_directive cpp_template_possible cpp_opt_declarators ;
%type <node>     cpp_using_decl cpp_namespace_decl cpp_catch_decl cpp_lambda_decl;
%type <node>     kwargs options;
%type <nodebuilder> kwargs_builder;

/* Misc */
%type <id>       identifier;
%type <dtype>    initializer cpp_const exception_specification cv_ref_qualifier qualifiers_exception_specification;
%type <str>      storage_class;
%type <intvalue> storage_class_raw storage_class_list;
%type <pl>       parms rawparms varargs_parms ;
%type <p>        parm_no_dox parm valparm valparms;
%type <pbuilder> valparms_builder;
%type <p>        typemap_parm tm_list;
%type <pbuilder> tm_list_builder;
%type <p>        templateparameter ;
%type <type>     templcpptype cpptype;
%type            classkey classkeyopt;
%type <id>       access_specifier;
%type <node>     base_specifier;
%type <intvalue> variadic_opt;
%type <type>     type rawtype type_right anon_bitfield_type decltype decltypeexpr cpp_alternate_rettype;
%type <bases>    base_list inherit raw_inherit;
%type <dtype>    definetype def_args etype default_delete deleted_definition explicit_default;
%type <dtype>    expr exprnum exprsimple exprcompound valexpr exprmem;
%type <id>       ename ;
%type <str>      less_valparms_greater;
%type <str>      type_qualifier;
%type <str>      ref_qualifier;
%type <id>       type_qualifier_raw;
%type <id>       idstring idstringopt;
%type <id>       pragma_lang;
%type <str>      pragma_arg;
%type <includetype> includetype;
%type <type>     pointer primitive_type;
%type <decl>     declarator direct_declarator notso_direct_declarator parameter_declarator plain_declarator;
%type <decl>     abstract_declarator direct_abstract_declarator ctor_end;
%type <tmap>     typemap_type;
%type <str>      idcolon idcolontail idcolonnt idcolontailnt idtemplate idtemplatetemplate stringbrace stringbracesemi;
%type <str>      string stringnum wstring;
%type <tparms>   template_parms;
%type <pbuilder> template_parms_builder;
%type <dtype>    cpp_vend;
%type <intvalue> rename_namewarn;
%type <ptype>    type_specifier primitive_type_list ;
%type <node>     fname stringtype;
%type <node>     featattr;
%type            lambda_introducer lambda_body lambda_template lambda_tail;
%type <str>      virt_specifier_seq virt_specifier_seq_opt;
%type <str>      class_virt_specifier_opt;

%{

// Default-initialised instances of token types to avoid uninitialised fields.
// The compiler will initialise all fields to zero or NULL for us.

static const struct Decl default_decl;
static const struct Define default_dtype;

/* C++ decltype/auto type deduction. */
static SwigType *deduce_type(const struct Define *dtype) {
  Node *n;
  if (!dtype->val) return NULL;
  n = Swig_symbol_clookup(dtype->val, 0);
  if (n) {
    if (Strcmp(nodeType(n),"enumitem") == 0) {
      /* For an enumitem, the "type" attribute gives us the underlying integer
       * type - we want the "type" attribute from the enum itself, which is
       * "parentNode".
       */
      n = Getattr(n, "parentNode");
    }
    return Getattr(n, "type");
  } else if (dtype->type != T_AUTO && dtype->type != T_UNKNOWN) {
    /* Try to deduce the type from the T_* type code. */
    String *deduced_type = NewSwigType(dtype->type);
    if (Len(deduced_type) > 0) return deduced_type;
    Delete(deduced_type);
  }
  return NULL;
}

static Node *new_enum_node(SwigType *enum_base_type) {
  Node *n = new_node("enum");
  if (enum_base_type) {
    switch (SwigType_type(enum_base_type)) {
      case T_USER:
	// We get T_USER if the underlying type is a typedef.  Unfortunately we
	// aren't able to resolve a typedef at this point, so we have to assume
	// it's a typedef to an integral or boolean type.
	break;
      case T_BOOL:
      case T_SCHAR:
      case T_UCHAR:
      case T_SHORT:
      case T_USHORT:
      case T_INT:
      case T_UINT:
      case T_LONG:
      case T_ULONG:
      case T_LONGLONG:
      case T_ULONGLONG:
      case T_CHAR:
      case T_WCHAR:
	break;
      default:
	Swig_error(cparse_file, cparse_line, "Underlying type of enum must be an integral type\n");
    }
    Setattr(n, "enumbase", enum_base_type);
  }
  return n;
}

%}

%%

/* ======================================================================
 *                          High-level Interface file
 *
 * An interface is just a sequence of declarations which may be SWIG directives
 * or normal C declarations.
 * ====================================================================== */

program        :  interface {
                   if (!classes) classes = NewHash();
		   Setattr($interface,"classes",classes); 
		   Setattr($interface,"name",ModuleName);
		   
		   if ((!module_node) && ModuleName) {
		     module_node = new_node("module");
		     Setattr(module_node,"name",ModuleName);
		   }
		   Setattr($interface,"module",module_node);
	           top = $interface;
               }
               | PARSETYPE parm END {
                 top = Copy(Getattr($parm,"type"));
		 Delete($parm);
               }
               | PARSETYPE error {
                 top = 0;
               }
               | PARSEPARM parm END {
                 top = $parm;
               }
               | PARSEPARM error {
                 top = 0;
               }
               | PARSEPARMS LPAREN parms RPAREN END {
                 top = $parms;
               }
               | PARSEPARMS error {
                 top = 0;
               }
               ;

interface      : interface[in] declaration {  
                   /* add declaration to end of linked list (the declaration isn't always a single declaration, sometimes it is a linked list itself) */
                   if (currentDeclComment != NULL) {
		     set_comment($declaration, currentDeclComment);
		     currentDeclComment = NULL;
                   }                                      
                   appendChild($in,$declaration);
                   $$ = $in;
               }
               | interface[in] DOXYGENSTRING {
		   Delete(currentDeclComment);
                   currentDeclComment = $DOXYGENSTRING; 
                   $$ = $in;
               }
               | interface[in] DOXYGENPOSTSTRING {
                   Node *node = lastChild($in);
                   if (node) {
                     set_comment(node, $DOXYGENPOSTSTRING);
		   } else {
		     Delete($DOXYGENPOSTSTRING);
		   }
                   $$ = $in;
               }
               | %empty {
                   $$ = new_node("top");
               }
               ;

declaration    : swig_directive
               | c_declaration
               | cpp_declaration
               | SEMI { $$ = 0; }
               | error {
		  if (cparse_unknown_directive) {
		      Swig_error(cparse_file, cparse_line, "Unknown directive '%s'.\n", cparse_unknown_directive);
		  } else {
		      Swig_error(cparse_file, cparse_line, "Syntax error in input(1).\n");
		  }
		  Exit(EXIT_FAILURE);
               }
/* Out of class constructor/destructor declarations */
               | c_constructor_decl { 
                  if ($$) {
   		      add_symbols($$);
                  }
                  $$ = $c_constructor_decl; 
	       }              

/* Out of class conversion operator.  For example:
     inline A::operator char *() const { ... }.

   This is nearly impossible to parse normally.  We just let the
   first part generate a syntax error and then resynchronize on the
   CONVERSIONOPERATOR token---discarding the rest of the definition. Ugh.

 */

               | error CONVERSIONOPERATOR {
                  $$ = 0;
		  Delete($CONVERSIONOPERATOR);
                  skip_decl();
               }
               ;

/* ======================================================================
 *                           SWIG DIRECTIVES 
 * ====================================================================== */
  
swig_directive : extend_directive
               | apply_directive
 	       | clear_directive
               | constant_directive
               | echo_directive
               | fragment_directive
               | include_directive
               | inline_directive
               | insert_directive
               | module_directive
               | native_directive
               | pragma_directive
               | rename_directive
               | feature_directive
               | varargs_directive
               | typemap_directive
               | types_directive
               | template_directive
               | warn_directive
               ;

/* ------------------------------------------------------------
   %extend classname { ... } 
   ------------------------------------------------------------ */

extend_directive : EXTEND options classkeyopt idcolon LBRACE {
               Node *cls;
	       String *clsname;
	       extendmode = 1;
	       cplus_mode = CPLUS_PUBLIC;
	       if (!classes) classes = NewHash();
	       if (!classes_typedefs) classes_typedefs = NewHash();
	       clsname = make_class_name($idcolon);
	       cls = Getattr(classes,clsname);
	       if (!cls) {
	         cls = Getattr(classes_typedefs, clsname);
		 if (!cls) {
		   /* No previous definition. Create a new scope */
		   Node *am = Getattr(Swig_extend_hash(),clsname);
		   if (!am) {
		     Swig_symbol_newscope();
		     Swig_symbol_setscopename($idcolon);
		     prev_symtab = 0;
		   } else {
		     prev_symtab = Swig_symbol_setscope(Getattr(am,"symtab"));
		   }
		   current_class = 0;
		 } else {
		   /* Previous typedef class definition.  Use its symbol table.
		      Deprecated, just the real name should be used. 
		      Note that %extend before the class typedef never worked, only %extend after the class typedef. */
		   prev_symtab = Swig_symbol_setscope(Getattr(cls, "symtab"));
		   current_class = cls;
		   SWIG_WARN_NODE_BEGIN(cls);
		   Swig_warning(WARN_PARSE_EXTEND_NAME, cparse_file, cparse_line, "Deprecated %%extend name used - the %s name '%s' should be used instead of the typedef name '%s'.\n", Getattr(cls, "kind"), SwigType_namestr(Getattr(cls, "name")), $idcolon);
		   SWIG_WARN_NODE_END(cls);
		 }
	       } else {
		 /* Previous class definition.  Use its symbol table */
		 prev_symtab = Swig_symbol_setscope(Getattr(cls,"symtab"));
		 current_class = cls;
	       }
	       Classprefix = NewString($idcolon);
	       Namespaceprefix= Swig_symbol_qualifiedscopename(0);
	       Delete(clsname);
	     } cpp_members RBRACE {
               String *clsname;
	       extendmode = 0;
               $$ = new_node("extend");
	       Setattr($$,"symtab",Swig_symbol_popscope());
	       if (prev_symtab) {
		 Swig_symbol_setscope(prev_symtab);
	       }
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
               clsname = make_class_name($idcolon);
	       Setattr($$,"name",clsname);

	       mark_nodes_as_extend($cpp_members);
	       if (current_class) {
		 /* We add the extension to the previously defined class */
		 appendChild($$, $cpp_members);
		 appendChild(current_class,$$);
	       } else {
		 /* We store the extensions in the extensions hash */
		 Node *am = Getattr(Swig_extend_hash(),clsname);
		 if (am) {
		   /* Append the members to the previous extend methods */
		   appendChild(am, $cpp_members);
		 } else {
		   appendChild($$, $cpp_members);
		   Setattr(Swig_extend_hash(),clsname,$$);
		 }
	       }
	       current_class = 0;
	       Delete(Classprefix);
	       Delete(clsname);
	       Classprefix = 0;
	       prev_symtab = 0;
	       $$ = 0;

	     }
             ;

/* ------------------------------------------------------------
   %apply
   ------------------------------------------------------------ */

apply_directive : APPLY typemap_parm LBRACE tm_list RBRACE {
                    $$ = new_node("apply");
                    Setattr($$,"pattern",Getattr($typemap_parm,"pattern"));
		    appendChild($$,$tm_list);
               };

/* ------------------------------------------------------------
   %clear
   ------------------------------------------------------------ */

clear_directive : CLEAR tm_list SEMI {
		 $$ = new_node("clear");
		 appendChild($$,$tm_list);
               }
               ;

/* ------------------------------------------------------------
   %constant name = value;
   %constant type name = value;

   Note: Source/Preprocessor/cpp.c injects `%constant X = Y;` for
   each `#define X Y` so that's handled here too.
   ------------------------------------------------------------ */

constant_directive :  CONSTANT identifier EQUAL definetype SEMI {
		 SwigType *type = NewSwigType($definetype.type);
		 if (Len(type) > 0) {
		   $$ = new_node("constant");
		   Setattr($$, "name", $identifier);
		   Setattr($$, "type", type);
		   Setattr($$, "value", $definetype.val);
		   if ($definetype.stringval) Setattr($$, "stringval", $definetype.stringval);
		   if ($definetype.numval) Setattr($$, "numval", $definetype.numval);
		   Setattr($$, "storage", "%constant");
		   SetFlag($$, "feature:immutable");
		   add_symbols($$);
		   Delete(type);
		 } else {
		   Swig_warning(WARN_PARSE_UNSUPPORTED_VALUE, cparse_file, cparse_line, "Unsupported constant value (ignored)\n");
		   $$ = 0;
		 }
	       }
               | CONSTANT type declarator def_args SEMI {
		 SwigType_push($type, $declarator.type);
		 /* Sneaky callback function trick */
		 if (SwigType_isfunction($type)) {
		   SwigType_add_pointer($type);
		 }
		 $$ = new_node("constant");
		 Setattr($$, "name", $declarator.id);
		 Setattr($$, "type", $type);
		 Setattr($$, "value", $def_args.val);
		 if ($def_args.stringval) Setattr($$, "stringval", $def_args.stringval);
		 if ($def_args.numval) Setattr($$, "numval", $def_args.numval);
		 Setattr($$, "storage", "%constant");
		 SetFlag($$, "feature:immutable");
		 add_symbols($$);
               }
	       /* Member function pointers with qualifiers. eg.
	         %constant short (Funcs::*pmf)(bool) const = &Funcs::F; */
	       | CONSTANT type direct_declarator LPAREN parms RPAREN cv_ref_qualifier def_args SEMI {
		 SwigType_add_function($type, $parms);
		 SwigType_push($type, $cv_ref_qualifier.qualifier);
		 SwigType_push($type, $direct_declarator.type);
		 /* Sneaky callback function trick */
		 if (SwigType_isfunction($type)) {
		   SwigType_add_pointer($type);
		 }
		 $$ = new_node("constant");
		 Setattr($$, "name", $direct_declarator.id);
		 Setattr($$, "type", $type);
		 Setattr($$, "value", $def_args.val);
		 if ($def_args.stringval) Setattr($$, "stringval", $def_args.stringval);
		 if ($def_args.numval) Setattr($$, "numval", $def_args.numval);
		 Setattr($$, "storage", "%constant");
		 SetFlag($$, "feature:immutable");
		 add_symbols($$);
	       }
               | CONSTANT error SEMI {
		 Swig_warning(WARN_PARSE_BAD_VALUE,cparse_file,cparse_line,"Bad constant value (ignored).\n");
		 $$ = 0;
	       }
	       | CONSTANT error END {
		 Swig_error(cparse_file,cparse_line,"Missing semicolon (';') after %%constant.\n");
		 Exit(EXIT_FAILURE);
	       }
               ;

/* ------------------------------------------------------------
   %echo "text"
   %echo %{ ... %}
   ------------------------------------------------------------ */

echo_directive : ECHO HBLOCK {
		 char temp[64];
		 Replace($HBLOCK,"$file",cparse_file, DOH_REPLACE_ANY);
		 sprintf(temp,"%d", cparse_line);
		 Replace($HBLOCK,"$line",temp,DOH_REPLACE_ANY);
		 Printf(stderr,"%s\n", $HBLOCK);
		 Delete($HBLOCK);
                 $$ = 0;
	       }
               | ECHO string {
		 char temp[64];
		 String *s = $string;
		 Replace(s,"$file",cparse_file, DOH_REPLACE_ANY);
		 sprintf(temp,"%d", cparse_line);
		 Replace(s,"$line",temp,DOH_REPLACE_ANY);
		 Printf(stderr,"%s\n", s);
		 Delete(s);
                 $$ = 0;
               }
               ;

/* fragment keyword arguments */
stringtype    : string LBRACE parm RBRACE {		 
                 $$ = NewHash();
                 Setattr($$,"value",$string);
		 Setattr($$,"type",Getattr($parm,"type"));
               }
               ;

fname         : string {
                 $$ = NewHash();
                 Setattr($$,"value",$string);
              }
              | stringtype
              ;

/* ------------------------------------------------------------
   %fragment(name, section) %{ ... %}
   %fragment("name" {type}, "section") %{ ... %}
   %fragment("name", "section", fragment="fragment1", fragment="fragment2") %{ ... %}
   Also as above but using { ... }
   %fragment("name");
   ------------------------------------------------------------ */

fragment_directive: FRAGMENT LPAREN fname COMMA kwargs RPAREN HBLOCK {
                   Hash *p = $kwargs;
		   $$ = new_node("fragment");
		   Setattr($$,"value",Getattr($fname,"value"));
		   Setattr($$,"type",Getattr($fname,"type"));
		   Setattr($$,"section",Getattr(p,"name"));
		   Setattr($$,"kwargs",nextSibling(p));
		   Setattr($$,"code",$HBLOCK);
		   Delete($HBLOCK);
                 }
                 | FRAGMENT LPAREN fname COMMA kwargs RPAREN LBRACE {
		   Hash *p = $kwargs;
		   String *code;
		   if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		   $$ = new_node("fragment");
		   Setattr($$,"value",Getattr($fname,"value"));
		   Setattr($$,"type",Getattr($fname,"type"));
		   Setattr($$,"section",Getattr(p,"name"));
		   Setattr($$,"kwargs",nextSibling(p));
		   Delitem(scanner_ccode,0);
		   Delitem(scanner_ccode,DOH_END);
		   code = Copy(scanner_ccode);
		   Setattr($$,"code",code);
		   Delete(code);
                 }
                 | FRAGMENT LPAREN fname RPAREN SEMI {
		   $$ = new_node("fragment");
		   Setattr($$,"value",Getattr($fname,"value"));
		   Setattr($$,"type",Getattr($fname,"type"));
		   Setattr($$,"emitonly","1");
		 }
                 ;

/* ------------------------------------------------------------
   %includefile(option1="xyz", ...) "filename" [ declarations ] 
   %importfile(option1="xyz", ...) "filename" [ declarations ]
   ------------------------------------------------------------ */

include_directive: includetype options string BEGINFILE <loc>{
		     $$.filename = Copy(cparse_file);
		     $$.line = cparse_line;
		     scanner_set_location($string,1);
                     if ($options) { 
		       String *maininput = Getattr($options, "maininput");
		       if (maininput)
		         scanner_set_main_input_file(NewString(maininput));
		     }
               }[loc] interface ENDOFFILE {
                     String *mname = 0;
                     $$ = $interface;
		     scanner_set_location($loc.filename, $loc.line + 1);
		     Delete($loc.filename);
		     switch ($includetype) {
		       case INCLUDE_INCLUDE:
			 set_nodeType($$, "include");
			 break;
		       case INCLUDE_IMPORT:
			 mname = $options ? Getattr($options, "module") : 0;
			 set_nodeType($$, "import");
			 if (import_mode) --import_mode;
			 break;
		     }
		     
		     Setattr($$,"name",$string);
		     /* Search for the module (if any) */
		     {
			 Node *n = firstChild($$);
			 while (n) {
			     if (Strcmp(nodeType(n),"module") == 0) {
			         if (mname) {
				   Setattr(n,"name", mname);
				   mname = 0;
				 }
				 Setattr($$,"module",Getattr(n,"name"));
				 break;
			     }
			     n = nextSibling(n);
			 }
			 if (mname) {
			   /* There is no module node in the import
			      node, ie, you imported a .h file
			      directly.  We are forced then to create
			      a new import node with a module node.
			   */			      
			   Node *nint = new_node("import");
			   Node *mnode = new_node("module");
			   Setattr(mnode,"name", mname);
                           Setattr(mnode,"options",$options);
			   appendChild(nint,mnode);
			   Delete(mnode);
			   appendChild(nint,firstChild($$));
			   $$ = nint;
			   Setattr($$,"module",mname);
			 }
		     }
		     Setattr($$,"options",$options);
               }
               ;

includetype    : INCLUDE { $$ = INCLUDE_INCLUDE; }
	       | IMPORT  { $$ = INCLUDE_IMPORT; ++import_mode;}
	       ;

/* ------------------------------------------------------------
   %inline %{ ... %}
   ------------------------------------------------------------ */

inline_directive : INLINE HBLOCK {
                 String *cpps;
		 if (Namespaceprefix) {
		   Swig_error(cparse_file, cparse_start_line, "%%inline directive inside a namespace is disallowed.\n");
		   $$ = 0;
		 } else {
		   $$ = new_node("insert");
		   Setattr($$,"code",$HBLOCK);
		   /* Need to run through the preprocessor */
		   Seek($HBLOCK,0,SEEK_SET);
		   Setline($HBLOCK,cparse_start_line);
		   Setfile($HBLOCK,cparse_file);
		   cpps = Preprocessor_parse($HBLOCK);
		   scanner_start_inline(cpps, cparse_start_line);
		   Delete(cpps);
		 }
		 Delete($HBLOCK);
	       }
               | INLINE LBRACE {
                 String *cpps;
		 int start_line = cparse_line;
		 if (Namespaceprefix) {
		   Swig_error(cparse_file, cparse_start_line, "%%inline directive inside a namespace is disallowed.\n");
		 }
		 if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		 if (Namespaceprefix) {
		   $$ = 0;
		 } else {
		   String *code;
                   $$ = new_node("insert");
		   Delitem(scanner_ccode,0);
		   Delitem(scanner_ccode,DOH_END);
		   code = Copy(scanner_ccode);
		   Setattr($$,"code", code);
		   Delete(code);		   
		   cpps=Copy(scanner_ccode);
		   scanner_start_inline(cpps, start_line);
		   Delete(cpps);
		 }
               }
                ;

/* ------------------------------------------------------------
   %{ ... %}
   %insert(section) "filename"
   %insert("section") "filename"
   %insert(section) %{ ... %}
   %insert("section") %{ ... %}
   %insert(section) { ... }
   %insert("section") { ... }
   ------------------------------------------------------------ */

insert_directive : HBLOCK {
                 $$ = new_node("insert");
		 Setattr($$,"code",$HBLOCK);
		 Delete($HBLOCK);
	       }
               | INSERT LPAREN idstring RPAREN string {
		 String *code = NewStringEmpty();
		 $$ = new_node("insert");
		 Setattr($$,"section",$idstring);
		 Setattr($$,"code",code);
		 if (Swig_insert_file($string,code) < 0) {
		   Swig_error(cparse_file, cparse_line, "Couldn't find '%s'.\n", $string);
		   $$ = 0;
		 } 
               }
               | INSERT LPAREN idstring RPAREN HBLOCK {
		 $$ = new_node("insert");
		 Setattr($$,"section",$idstring);
		 Setattr($$,"code",$HBLOCK);
		 Delete($HBLOCK);
               }
               | INSERT LPAREN idstring RPAREN LBRACE {
		 String *code;
		 if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		 $$ = new_node("insert");
		 Setattr($$,"section",$idstring);
		 Delitem(scanner_ccode,0);
		 Delitem(scanner_ccode,DOH_END);
		 code = Copy(scanner_ccode);
		 Setattr($$,"code", code);
		 Delete(code);
	       }
               ;
      
/* ------------------------------------------------------------
    %module modname
    %module "modname"
   ------------------------------------------------------------ */

module_directive: MODULE options idstring {
                 $$ = new_node("module");
		 if ($options) {
		   Setattr($$,"options",$options);
		   if (Getattr($options,"directors")) {
		     Wrapper_director_mode_set(1);
		     if (!cparse_cplusplus) {
		       Swig_error(cparse_file, cparse_line, "Directors are not supported for C code and require the -c++ option\n");
		     }
		   } 
		   if (Getattr($options,"dirprot")) {
		     Wrapper_director_protected_mode_set(1);
		   } 
		   if (Getattr($options,"allprotected")) {
		     Wrapper_all_protected_mode_set(1);
		   } 
		   if (Getattr($options,"templatereduce")) {
		     template_reduce = 1;
		   }
		   if (Getattr($options,"notemplatereduce")) {
		     template_reduce = 0;
		   }
		 }
		 if (!ModuleName) ModuleName = NewString($idstring);
		 if (!import_mode) {
		   /* first module included, we apply global
		      ModuleName, which can be modify by -module */
		   String *mname = Copy(ModuleName);
		   Setattr($$,"name",mname);
		   Delete(mname);
		 } else { 
		   /* import mode, we just pass the idstring */
		   Setattr($$,"name",$idstring);   
		 }		 
		 if (!module_node) module_node = $$;
	       }
               ;

/* ------------------------------------------------------------
   %native(scriptname) name;
   %native(scriptname) type name (parms);
   ------------------------------------------------------------ */

native_directive : NATIVE LPAREN identifier[name] RPAREN storage_class identifier[wrap_name] SEMI {
                 $$ = new_node("native");
		 Setattr($$,"name",$name);
		 Setattr($$,"wrap:name",$wrap_name);
		 Delete($storage_class);
	         add_symbols($$);
	       }
               | NATIVE LPAREN identifier RPAREN storage_class type declarator SEMI {
		 if (!SwigType_isfunction($declarator.type)) {
		   Swig_error(cparse_file,cparse_line,"%%native declaration '%s' is not a function.\n", $declarator.id);
		   $$ = 0;
		 } else {
		     Delete(SwigType_pop_function($declarator.type));
		     /* Need check for function here */
		     SwigType_push($type,$declarator.type);
		     $$ = new_node("native");
	             Setattr($$,"name",$identifier);
		     Setattr($$,"wrap:name",$declarator.id);
		     Setattr($$,"type",$type);
		     Setattr($$,"parms",$declarator.parms);
		     Setattr($$,"decl",$declarator.type);
		 }
		 Delete($storage_class);
	         add_symbols($$);
	       }
               ;

/* ------------------------------------------------------------
   %pragma(lang) name=value
   %pragma(lang) name
   %pragma name = value
   %pragma name
   ------------------------------------------------------------ */

pragma_directive : PRAGMA pragma_lang identifier EQUAL pragma_arg {
                 $$ = new_node("pragma");
		 Setattr($$,"lang",$pragma_lang);
		 Setattr($$,"name",$identifier);
		 Setattr($$,"value",$pragma_arg);
	       }
              | PRAGMA pragma_lang identifier {
		$$ = new_node("pragma");
		Setattr($$,"lang",$pragma_lang);
		Setattr($$,"name",$identifier);
	      }
              ;

pragma_arg    : string
              | HBLOCK
              ;

pragma_lang   : LPAREN identifier RPAREN { $$ = $identifier; }
              | %empty { $$ = "swig"; }
              ;

/* ------------------------------------------------------------
   %rename(newname) identifier;
   ------------------------------------------------------------ */

rename_directive : rename_namewarn declarator idstring SEMI {
                SwigType *t = $declarator.type;
		Hash *kws = NewHash();
		String *fixname;
		fixname = feature_identifier_fix($declarator.id);
		Setattr(kws,"name",$idstring);
		if (!Len(t)) t = 0;
		/* Special declarator check */
		if (t) {
		  if (SwigType_isfunction(t)) {
		    SwigType *decl = SwigType_pop_function(t);
		    if (SwigType_ispointer(t)) {
		      String *nname = NewStringf("*%s",fixname);
		      if ($rename_namewarn) {
			Swig_name_rename_add(Namespaceprefix, nname,decl,kws,$declarator.parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,nname,decl,kws);
		      }
		      Delete(nname);
		    } else {
		      if ($rename_namewarn) {
			Swig_name_rename_add(Namespaceprefix,(fixname),decl,kws,$declarator.parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,(fixname),decl,kws);
		      }
		    }
		    Delete(decl);
		  } else if (SwigType_ispointer(t)) {
		    String *nname = NewStringf("*%s",fixname);
		    if ($rename_namewarn) {
		      Swig_name_rename_add(Namespaceprefix,(nname),0,kws,$declarator.parms);
		    } else {
		      Swig_name_namewarn_add(Namespaceprefix,(nname),0,kws);
		    }
		    Delete(nname);
		  }
		} else {
		  if ($rename_namewarn) {
		    Swig_name_rename_add(Namespaceprefix,(fixname),0,kws,$declarator.parms);
		  } else {
		    Swig_name_namewarn_add(Namespaceprefix,(fixname),0,kws);
		  }
		}
                $$ = 0;
		scanner_clear_rename();
              }
              | rename_namewarn LPAREN kwargs RPAREN declarator cpp_const SEMI {
		String *fixname;
		Hash *kws = $kwargs;
		SwigType *t = $declarator.type;
		fixname = feature_identifier_fix($declarator.id);
		if (!Len(t)) t = 0;
		/* Special declarator check */
		if (t) {
		  if ($cpp_const.qualifier) SwigType_push(t,$cpp_const.qualifier);
		  if (SwigType_isfunction(t)) {
		    SwigType *decl = SwigType_pop_function(t);
		    if (SwigType_ispointer(t)) {
		      String *nname = NewStringf("*%s",fixname);
		      if ($rename_namewarn) {
			Swig_name_rename_add(Namespaceprefix, nname,decl,kws,$declarator.parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,nname,decl,kws);
		      }
		      Delete(nname);
		    } else {
		      if ($rename_namewarn) {
			Swig_name_rename_add(Namespaceprefix,(fixname),decl,kws,$declarator.parms);
		      } else {
			Swig_name_namewarn_add(Namespaceprefix,(fixname),decl,kws);
		      }
		    }
		    Delete(decl);
		  } else if (SwigType_ispointer(t)) {
		    String *nname = NewStringf("*%s",fixname);
		    if ($rename_namewarn) {
		      Swig_name_rename_add(Namespaceprefix,(nname),0,kws,$declarator.parms);
		    } else {
		      Swig_name_namewarn_add(Namespaceprefix,(nname),0,kws);
		    }
		    Delete(nname);
		  }
		} else {
		  if ($rename_namewarn) {
		    Swig_name_rename_add(Namespaceprefix,(fixname),0,kws,$declarator.parms);
		  } else {
		    Swig_name_namewarn_add(Namespaceprefix,(fixname),0,kws);
		  }
		}
                $$ = 0;
		scanner_clear_rename();
              }
              | rename_namewarn LPAREN kwargs RPAREN string SEMI {
		if ($rename_namewarn) {
		  Swig_name_rename_add(Namespaceprefix,$string,0,$kwargs,0);
		} else {
		  Swig_name_namewarn_add(Namespaceprefix,$string,0,$kwargs);
		}
		$$ = 0;
		scanner_clear_rename();
              }
              ;

rename_namewarn : RENAME {
		    $$ = 1;
                } 
                | NAMEWARN {
                    $$ = 0;
                };


/* ------------------------------------------------------------
   Feature targeting a symbol name (non-global feature):

     %feature(featurename) name "val";
     %feature(featurename, val) name;

   where "val" could instead be the other bracket types, that is,
   { val } or %{ val %} or indeed omitted whereupon it defaults to "1".
   Or, the global feature which does not target a symbol name:

     %feature(featurename) "val";
     %feature(featurename, val);

   An empty val (empty string) clears the feature.
   Any number of feature attributes can optionally be added, for example
   a non-global feature with 2 attributes:

     %feature(featurename, attrib1="attribval1", attrib2="attribval2") name "val";
     %feature(featurename, val, attrib1="attribval1", attrib2="attribval2") name;
   ------------------------------------------------------------ */

                  /* Non-global feature */
feature_directive : FEATURE LPAREN idstring featattr RPAREN declarator cpp_const stringbracesemi {
                    String *val = $stringbracesemi ? NewString($stringbracesemi) : NewString("1");
                    new_feature($idstring, val, $featattr, $declarator.id, $declarator.type, $declarator.parms, $cpp_const.qualifier);
                    $$ = 0;
                    scanner_clear_rename();
                  }
                  | FEATURE LPAREN idstring COMMA stringnum featattr RPAREN declarator cpp_const SEMI {
                    String *val = Len($stringnum) ? $stringnum : 0;
                    new_feature($idstring, val, $featattr, $declarator.id, $declarator.type, $declarator.parms, $cpp_const.qualifier);
                    $$ = 0;
                    scanner_clear_rename();
                  }

                  /* Global feature */
                  | FEATURE LPAREN idstring featattr RPAREN stringbracesemi {
                    String *val = $stringbracesemi ? NewString($stringbracesemi) : NewString("1");
                    new_feature($idstring, val, $featattr, 0, 0, 0, 0);
                    $$ = 0;
                    scanner_clear_rename();
                  }
                  | FEATURE LPAREN idstring COMMA stringnum featattr RPAREN SEMI {
                    String *val = Len($stringnum) ? $stringnum : 0;
                    new_feature($idstring, val, $featattr, 0, 0, 0, 0);
                    $$ = 0;
                    scanner_clear_rename();
                  }
                  ;

stringbracesemi : stringbrace
                | SEMI { $$ = 0; }
                | PARMS LPAREN parms RPAREN SEMI { $$ = $parms; } 
                ;

featattr        : COMMA idstring EQUAL stringnum featattr[in] {
		  $$ = NewHash();
		  Setattr($$,"name",$idstring);
		  Setattr($$,"value",$stringnum);
		  if ($in) set_nextSibling($$, $in);
		}
		| %empty {
		  $$ = 0;
		}
		;

/* %varargs() directive. */

varargs_directive : VARARGS LPAREN varargs_parms RPAREN declarator cpp_const SEMI {
                 Parm *val;
		 String *name;
		 SwigType *t;
		 if (Namespaceprefix) name = NewStringf("%s::%s", Namespaceprefix, $declarator.id);
		 else name = NewString($declarator.id);
		 val = $varargs_parms;
		 if ($declarator.parms) {
		   Setmeta(val,"parms",$declarator.parms);
		 }
		 t = $declarator.type;
		 if (!Len(t)) t = 0;
		 if (t) {
		   if ($cpp_const.qualifier) SwigType_push(t,$cpp_const.qualifier);
		   if (SwigType_isfunction(t)) {
		     SwigType *decl = SwigType_pop_function(t);
		     if (SwigType_ispointer(t)) {
		       String *nname = NewStringf("*%s",name);
		       Swig_feature_set(Swig_cparse_features(), nname, decl, "feature:varargs", val, 0);
		       Delete(nname);
		     } else {
		       Swig_feature_set(Swig_cparse_features(), name, decl, "feature:varargs", val, 0);
		     }
		     Delete(decl);
		   } else if (SwigType_ispointer(t)) {
		     String *nname = NewStringf("*%s",name);
		     Swig_feature_set(Swig_cparse_features(),nname,0,"feature:varargs",val, 0);
		     Delete(nname);
		   }
		 } else {
		   Swig_feature_set(Swig_cparse_features(),name,0,"feature:varargs",val, 0);
		 }
		 Delete(name);
		 $$ = 0;
              };

varargs_parms   : parms
                | NUM_INT COMMA parm { 
		  int i;
		  int n;
		  Parm *p;
		  n = atoi(Char($NUM_INT.val));
		  if (n <= 0) {
		    Swig_error(cparse_file, cparse_line,"Argument count in %%varargs must be positive.\n");
		    $$ = 0;
		  } else {
		    String *name = Getattr($parm, "name");
		    $$ = Copy($parm);
		    if (name)
		      Setattr($$, "name", NewStringf("%s%d", name, n));
		    for (i = 1; i < n; i++) {
		      p = Copy($parm);
		      name = Getattr(p, "name");
		      if (name)
		        Setattr(p, "name", NewStringf("%s%d", name, n-i));
		      set_nextSibling(p,$$);
		      Delete($$);
		      $$ = p;
		    }
		  }
                }
               ;


/* ------------------------------------------------------------
   %typemap(method) type { ... }
   %typemap(method) type "..."
   %typemap(method) type;    - typemap deletion
   %typemap(method) type1,type2,... = type;    - typemap copy
   ------------------------------------------------------------ */

typemap_directive :  TYPEMAP LPAREN typemap_type RPAREN tm_list stringbrace {
		   $$ = 0;
		   if ($typemap_type.method) {
		     String *code = 0;
		     $$ = new_node("typemap");
		     Setattr($$,"method",$typemap_type.method);
		     if ($typemap_type.kwargs) {
		       ParmList *kw = $typemap_type.kwargs;
                       code = remove_block(kw, $stringbrace);
		       Setattr($$,"kwargs", $typemap_type.kwargs);
		     }
		     code = code ? code : NewString($stringbrace);
		     Setattr($$,"code", code);
		     Delete(code);
		     appendChild($$,$tm_list);
		     Delete($typemap_type.kwargs);
		     Delete($typemap_type.method);
		   }
	       }
               | TYPEMAP LPAREN typemap_type RPAREN tm_list SEMI {
		 $$ = 0;
		 if ($typemap_type.method) {
		   $$ = new_node("typemap");
		   Setattr($$,"method",$typemap_type.method);
		   appendChild($$,$tm_list);
		   Delete($typemap_type.method);
		 }
		 Delete($typemap_type.kwargs);
	       }
               | TYPEMAP LPAREN typemap_type RPAREN tm_list EQUAL typemap_parm SEMI {
		   $$ = 0;
		   if ($typemap_type.method) {
		     $$ = new_node("typemapcopy");
		     Setattr($$,"method",$typemap_type.method);
		     Setattr($$,"pattern", Getattr($typemap_parm,"pattern"));
		     appendChild($$,$tm_list);
		     Delete($typemap_type.method);
		   }
		   Delete($typemap_type.kwargs);
	       }
               ;

/* typemap method and optional kwargs */

typemap_type   : kwargs {
		 String *name = Getattr($kwargs, "name");
		 Hash *p = nextSibling($kwargs);
		 $$.method = name;
		 $$.kwargs = p;
		 if (Getattr($kwargs, "value")) {
		   Swig_error(cparse_file, cparse_line,
			      "%%typemap method shouldn't have a value specified.\n");
		 }
		 while (p) {
		   if (!Getattr(p, "value")) {
		     Swig_error(cparse_file, cparse_line,
				"%%typemap attribute '%s' is missing its value.  If this is specifying the target language, that's no longer supported: use #ifdef SWIG<LANG> instead.\n",
				Getattr(p, "name"));
		     /* Set to empty value to avoid segfaults later. */
		     Setattr(p, "value", NewStringEmpty());
		   }
		   p = nextSibling(p);
		 }
                }
               ;

tm_list        : tm_list_builder {
		 $$ = $tm_list_builder.parms;
	       }
               ;

tm_list_builder: typemap_parm {
                 $$.parms = $$.last = $typemap_parm;
	       }
	       | tm_list_builder[in] COMMA typemap_parm {
		 // Build a linked list in the order specified, but avoiding
		 // a right recursion rule because "Right recursion uses up
		 // space on the Bison stack in proportion to the number of
		 // elements in the sequence".
		 set_nextSibling($in.last, $typemap_parm);
		 $$.parms = $in.parms;
		 $$.last = $typemap_parm;
	       }
               ;

typemap_parm   : type plain_declarator {
                  Parm *parm;
		  SwigType_push($type,$plain_declarator.type);
		  $$ = new_node("typemapitem");
		  parm = NewParmWithoutFileLineInfo($type,$plain_declarator.id);
		  Setattr($$,"pattern",parm);
		  Setattr($$,"parms", $plain_declarator.parms);
		  Delete(parm);
		  /*		  $$ = NewParmWithoutFileLineInfo($type,$plain_declarator.id);
				  Setattr($$,"parms",$plain_declarator.parms); */
                }
               | LPAREN parms RPAREN {
                  $$ = new_node("typemapitem");
		  Setattr($$,"pattern",$parms);
		  /*		  Setattr($$,"multitype",$parms); */
               }
               | LPAREN parms[pattern] RPAREN LPAREN parms[in] RPAREN {
		 $$ = new_node("typemapitem");
		 Setattr($$,"pattern", $pattern);
		 /*                 Setattr($$,"multitype",$in); */
		 Setattr($$,"parms",$in);
               }
               ;

/* ------------------------------------------------------------
   %types(parmlist); 
   %types(parmlist) %{ ... %}
   ------------------------------------------------------------ */

types_directive : TYPES LPAREN parms RPAREN stringbracesemi {
                   $$ = new_node("types");
		   Setattr($$,"parms",$parms);
                   if ($stringbracesemi)
		     Setattr($$,"convcode",NewString($stringbracesemi));
               }
               ;

/* ------------------------------------------------------------
   %template(name) tname<args>;
   ------------------------------------------------------------ */

template_directive: SWIGTEMPLATE LPAREN idstringopt RPAREN idcolonnt LESSTHAN valparms GREATERTHAN SEMI {
                  Parm *p;
		  Node *n = 0;
		  Node *outer_class = currentOuterClass;
		  Symtab *tscope = 0;
		  String *symname = $idstringopt ? NewString($idstringopt) : 0;
		  int errored_flag = 0;
		  String *idcolonnt;

		  $$ = 0;

		  tscope = Swig_symbol_current();          /* Get the current scope */

		  /* If the class name is qualified, we need to create or lookup namespace entries */
		  idcolonnt = resolve_create_node_scope($idcolonnt, 0, &errored_flag);

		  if (!errored_flag) {
		    if (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0)
		      outer_class = nscope_inner;

		    /*
		      We use the new namespace entry 'nscope' only to
		      emit the template node. The template parameters are
		      resolved in the current 'tscope'.

		      This is closer to the C++ (typedef) behavior.
		    */
		    n = Swig_cparse_template_locate(idcolonnt, $valparms, symname, tscope);
		  }

		  /* Patch the argument types to respect namespaces */
		  p = $valparms;
		  while (p) {
		    SwigType *value = Getattr(p,"value");
		    if (!value) {
		      SwigType *ty = Getattr(p,"type");
		      if (ty) {
			SwigType *rty = 0;
			int reduce = template_reduce;
			if (reduce || !SwigType_ispointer(ty)) {
			  rty = Swig_symbol_typedef_reduce(ty,tscope);
			  if (!reduce) reduce = SwigType_ispointer(rty);
			}
			ty = reduce ? Swig_symbol_type_qualify(rty,tscope) : Swig_symbol_type_qualify(ty,tscope);
			Setattr(p,"type",ty);
			Delete(ty);
			Delete(rty);
		      }
		    } else {
		      value = Swig_symbol_type_qualify(value,tscope);
		      Setattr(p,"value",value);
		      Delete(value);
		    }

		    p = nextSibling(p);
		  }

		  /* Look for the template */
		  {
                    Node *nn = n;
                    Node *linklistend = 0;
                    Node *linkliststart = 0;
                    while (nn) {
                      Node *templnode = 0;
                      if (GetFlag(nn, "instantiate")) {
			Delattr(nn, "instantiate");
			{
			  int nnisclass = (Strcmp(Getattr(nn, "templatetype"), "class") == 0); /* class template not a classforward nor function template */
			  Parm *tparms = Getattr(nn, "templateparms");
			  int specialized = !tparms; /* fully specialized (an explicit specialization) */
			  String *tname = Copy(idcolonnt);
			  Node *primary_template = Swig_symbol_clookup(tname, 0);

			  /* Expand the template */
			  ParmList *temparms = Swig_cparse_template_parms_expand($valparms, primary_template, nn);

                          templnode = copy_node(nn);
			  update_nested_classes(templnode); /* update classes nested within template */
                          /* We need to set the node name based on name used to instantiate */
                          Setattr(templnode,"name",tname);
			  Delete(tname);
                          if (!specialized) {
                            Delattr(templnode,"sym:typename");
                          } else {
                            Setattr(templnode,"sym:typename","1");
                          }
			  /* for now, nested %template is allowed only in the same scope as the template declaration */
                          if (symname && !(nnisclass && ((outer_class && (outer_class != Getattr(nn, "nested:outer")))
			    ||(extendmode && current_class && (current_class != Getattr(nn, "nested:outer")))))) {
			    /*
			       Comment this out for 1.3.28. We need to
			       re-enable it later but first we need to
			       move %ignore from using %rename to use
			       %feature(ignore).

			       String *symname = Swig_name_make(templnode, 0, symname, 0, 0);
			    */
                            Swig_cparse_template_expand(templnode, symname, temparms, tscope);
                            Setattr(templnode, "sym:name", symname);
                          } else {
                            static int cnt = 0;
                            String *nname = NewStringf("__dummy_%d__", cnt++);
                            Swig_cparse_template_expand(templnode,nname,temparms,tscope);
                            Setattr(templnode,"sym:name",nname);
                            SetFlag(templnode,"hidden");
			    Delete(nname);
                            Setattr(templnode,"feature:onlychildren", "typemap,typemapitem,typemapcopy,typedef,types,fragment,apply");
			    if (symname) {
			      Swig_warning(WARN_PARSE_NESTED_TEMPLATE, cparse_file, cparse_line, "Named nested template instantiations not supported. Processing as if no name was given to %%template().\n");
			    }
                          }
                          Delattr(templnode,"templatetype");
                          Setattr(templnode,"template",nn);
                          Setfile(templnode,cparse_file);
                          Setline(templnode,cparse_line);
                          Delete(temparms);
			  if (outer_class && nnisclass) {
			    SetFlag(templnode, "nested");
			    Setattr(templnode, "nested:outer", outer_class);
			  }
                          add_symbols_copy(templnode);

			  if (Equal(nodeType(templnode), "classforward") && !(GetFlag(templnode, "feature:ignore") || GetFlag(templnode, "hidden"))) {
			    SWIG_WARN_NODE_BEGIN(templnode);
			    /* A full template class definition is required in order to wrap a template class as a proxy class so this %template is ineffective. */
			    if (GetFlag(templnode, "nested:forward"))
			      Swig_warning(WARN_PARSE_TEMPLATE_NESTED, cparse_file, cparse_line, "Unsupported template nested class '%s' cannot be used to instantiate a full template class with name '%s'.\n", Swig_name_decl(templnode), Getattr(templnode, "sym:name"));
			    else
			      Swig_warning(WARN_PARSE_TEMPLATE_FORWARD, cparse_file, cparse_line, "Template forward class '%s' cannot be used to instantiate a full template class with name '%s'.\n", Swig_name_decl(templnode), Getattr(templnode, "sym:name"));
			    SWIG_WARN_NODE_END(templnode);
			  }

                          if (Strcmp(nodeType(templnode),"class") == 0) {

                            /* Identify pure abstract methods */
                            Setattr(templnode,"abstracts", pure_abstracts(firstChild(templnode)));

                            /* Set up inheritance in symbol table */
                            {
                              Symtab  *csyms;
                              List *baselist = Getattr(templnode,"baselist");
                              csyms = Swig_symbol_current();
                              Swig_symbol_setscope(Getattr(templnode,"symtab"));
                              if (baselist) {
                                List *bases = Swig_make_inherit_list(Getattr(templnode,"name"),baselist, Namespaceprefix);
                                if (bases) {
                                  Iterator s;
                                  for (s = First(bases); s.item; s = Next(s)) {
                                    Symtab *st = Getattr(s.item,"symtab");
                                    if (st) {
				      Setfile(st,Getfile(s.item));
				      Setline(st,Getline(s.item));
                                      Swig_symbol_inherit(st);
                                    }
                                  }
				  Delete(bases);
                                }
                              }
                              Swig_symbol_setscope(csyms);
                            }

                            /* Merge in %extend methods for this class.
			       This only merges methods within %extend for a template specialized class such as
			         template<typename T> class K {}; %extend K<int> { ... }
			       The copy_node() call above has already added in the generic %extend methods such as
			         template<typename T> class K {}; %extend K { ... } */

			    /* !!! This may be broken.  We may have to add the
			       %extend methods at the beginning of the class */
                            {
                              String *stmp = 0;
                              String *clsname;
                              Node *am;
                              if (Namespaceprefix) {
                                clsname = stmp = NewStringf("%s::%s", Namespaceprefix, Getattr(templnode,"name"));
                              } else {
                                clsname = Getattr(templnode,"name");
                              }
                              am = Getattr(Swig_extend_hash(),clsname);
                              if (am) {
                                Symtab *st = Swig_symbol_current();
                                Swig_symbol_setscope(Getattr(templnode,"symtab"));
                                /*			    Printf(stdout,"%s: %s %p %p\n", Getattr(templnode,"name"), clsname, Swig_symbol_current(), Getattr(templnode,"symtab")); */
                                Swig_extend_merge(templnode,am);
                                Swig_symbol_setscope(st);
				Swig_extend_append_previous(templnode,am);
                                Delattr(Swig_extend_hash(),clsname);
                              }
			      if (stmp) Delete(stmp);
                            }

                            /* Add to classes hash */
			    if (!classes)
			      classes = NewHash();

			    if (Namespaceprefix) {
			      String *temp = NewStringf("%s::%s", Namespaceprefix, Getattr(templnode,"name"));
			      Setattr(classes,temp,templnode);
			      Delete(temp);
			    } else {
			      String *qs = Swig_symbol_qualifiedscopename(templnode);
			      Setattr(classes, qs,templnode);
			      Delete(qs);
			    }
                          }
                        }

                        /* all the overloaded function templates are added into a linked list */
                        if (!linkliststart)
                          linkliststart = templnode;
                        if (nscope_inner) {
                          /* non-global namespace */
                          if (templnode) {
                            appendChild(nscope_inner,templnode);
			    Delete(templnode);
                            if (nscope) $$ = nscope;
                          }
                        } else {
                          /* global namespace */
                          if (!linklistend) {
                            $$ = templnode;
                          } else {
                            set_nextSibling(linklistend,templnode);
			    Delete(templnode);
                          }
                          linklistend = templnode;
                        }
                      }
                      nn = Getattr(nn,"sym:nextSibling"); /* repeat for overloaded function templates. If a class template there will never be a sibling. */
                    }
                    update_defaultargs(linkliststart);
                    update_abstracts(linkliststart);
		  }
	          Swig_symbol_setscope(tscope);
		  Delete(Namespaceprefix);
		  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  Delete(symname);
                }
               ;

/* ------------------------------------------------------------
   %warn "text"
   ------------------------------------------------------------ */

warn_directive : WARN string {
		  Swig_warning(0,cparse_file, cparse_line,"%s\n", $string);
		  $$ = 0;
               }
               ;

/* ======================================================================
 *                              C Parsing
 * ====================================================================== */

c_declaration   : c_decl {
                    $$ = $c_decl; 
                    if ($$) {
   		      add_symbols($$);
                      default_arguments($$);
   	            }
                }
                | c_enum_decl
                | c_enum_forward_decl

/* An extern C type declaration, disable cparse_cplusplus if needed. */

                | EXTERN string LBRACE {
		  if (Strcmp($string,"C") == 0) {
		    cparse_externc = 1;
		  }
		} interface RBRACE {
		  cparse_externc = 0;
		  if (Strcmp($string,"C") == 0) {
		    Node *n = firstChild($interface);
		    $$ = new_node("extern");
		    Setattr($$,"name",$string);
		    appendChild($$,n);
		    while (n) {
		      String *s = Getattr(n, "storage");
		      if (s) {
			if (Strstr(s, "thread_local")) {
			  Insert(s,0,"externc ");
			} else if (!Equal(s, "typedef")) {
			  Setattr(n,"storage","externc");
			}
		      } else {
			Setattr(n,"storage","externc");
		      }
		      n = nextSibling(n);
		    }
		  } else {
		    if (!Equal($string,"C++")) {
		      Swig_warning(WARN_PARSE_UNDEFINED_EXTERN,cparse_file, cparse_line,"Unrecognized extern type \"%s\".\n", $string);
		    }
		    $$ = new_node("extern");
		    Setattr($$,"name",$string);
		    appendChild($$,firstChild($interface));
		  }
                }
                | cpp_lambda_decl {
		  $$ = $cpp_lambda_decl;
		  SWIG_WARN_NODE_BEGIN($$);
		  Swig_warning(WARN_CPP11_LAMBDA, cparse_file, cparse_line, "Lambda expressions and closures are not fully supported yet.\n");
		  SWIG_WARN_NODE_END($$);
		}
                | USING idcolon EQUAL type plain_declarator SEMI {
		  /* Convert using statement to a typedef statement */
		  $$ = new_node("cdecl");
		  Setattr($$,"type",$type);
		  Setattr($$,"storage","typedef");
		  Setattr($$,"name",$idcolon);
		  Setattr($$,"decl",$plain_declarator.type);
		  SetFlag($$,"typealias");
		  add_symbols($$);
		}
                | TEMPLATE LESSTHAN template_parms GREATERTHAN USING idcolon EQUAL type plain_declarator SEMI {
		  /* Convert alias template to a "template" typedef statement */
		  $$ = new_node("template");
		  Setattr($$,"type",$type);
		  Setattr($$,"storage","typedef");
		  Setattr($$,"name",$idcolon);
		  Setattr($$,"decl",$plain_declarator.type);
		  Setattr($$,"templateparms",$template_parms);
		  Setattr($$,"templatetype","cdecl");
		  SetFlag($$,"aliastemplate");
		  add_symbols($$);
		}
                | cpp_static_assert
                ;

/* ------------------------------------------------------------
   A C global declaration of some kind (may be variable, function, typedef, etc.)
   ------------------------------------------------------------ */

c_decl  : storage_class type declarator cpp_const initializer c_decl_tail {
	      String *decl = $declarator.type;
              $$ = new_node("cdecl");
	      if ($cpp_const.qualifier)
	        decl = add_qualifier_to_declarator($declarator.type, $cpp_const.qualifier);
	      Setattr($$,"refqualifier",$cpp_const.refqualifier);
	      Setattr($$,"type",$type);
	      Setattr($$,"storage",$storage_class);
	      Setattr($$,"name",$declarator.id);
	      Setattr($$,"decl",decl);
	      Setattr($$,"parms",$declarator.parms);
	      Setattr($$,"value",$initializer.val);
	      if ($initializer.stringval) Setattr($$, "stringval", $initializer.stringval);
	      if ($initializer.numval) Setattr($$, "numval", $initializer.numval);
	      Setattr($$,"throws",$cpp_const.throws);
	      Setattr($$,"throw",$cpp_const.throwf);
	      Setattr($$,"noexcept",$cpp_const.nexcept);
	      Setattr($$,"final",$cpp_const.final);
	      if ($initializer.val && $initializer.type) {
		/* store initializer type as it might be different to the declared type */
		SwigType *valuetype = NewSwigType($initializer.type);
		if (Len(valuetype) > 0) {
		  Setattr($$, "valuetype", valuetype);
		} else {
		  /* If we can't determine the initializer type use the declared type. */
		  Setattr($$, "valuetype", $type);
		}
		Delete(valuetype);
	      }
	      if (!$c_decl_tail) {
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr($$,"code",code);
		  Delete(code);
		}
	      } else {
		Node *n = $c_decl_tail;
		/* Inherit attributes */
		while (n) {
		  String *type = Copy($type);
		  Setattr(n,"type",type);
		  Setattr(n,"storage",$storage_class);
		  n = nextSibling(n);
		  Delete(type);
		}
	      }
	      if ($initializer.bitfield) {
		Setattr($$,"bitfield", $initializer.bitfield);
	      }

	      if ($declarator.id) {
		/* Ignore all scoped declarations, could be 1. out of class function definition 2. friend function declaration 3. ... */
		String *p = Swig_scopename_prefix($declarator.id);
		if (p) {
		  /* This is a special case. If the scope name of the declaration exactly
		     matches that of the declaration, then we will allow it. Otherwise, delete. */
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last($declarator.id);
		    Setattr($$, "name", lstr);
		    Delete(lstr);
		    set_nextSibling($$, $c_decl_tail);
		  } else {
		    Delete($$);
		    $$ = $c_decl_tail;
		  }
		  Delete(p);
		} else if (Strncmp($declarator.id, "::", 2) == 0) {
		  /* global scope declaration/definition ignored */
		  Delete($$);
		  $$ = $c_decl_tail;
		} else {
		  set_nextSibling($$, $c_decl_tail);
		}
	      } else {
		Swig_error(cparse_file, cparse_line, "Missing symbol name for global declaration\n");
		$$ = 0;
	      }

	      if ($cpp_const.qualifier && $storage_class && Strstr($storage_class, "static"))
		Swig_error(cparse_file, cparse_line, "Static function %s cannot have a qualifier.\n", Swig_name_decl($$));
	      Delete($storage_class);
           }
	   | storage_class type declarator cpp_const EQUAL error SEMI {
	      String *decl = $declarator.type;
	      $$ = new_node("cdecl");
	      if ($cpp_const.qualifier)
	        decl = add_qualifier_to_declarator($declarator.type, $cpp_const.qualifier);
	      Setattr($$, "refqualifier", $cpp_const.refqualifier);
	      Setattr($$, "type", $type);
	      Setattr($$, "storage", $storage_class);
	      Setattr($$, "name", $declarator.id);
	      Setattr($$, "decl", decl);
	      Setattr($$, "parms", $declarator.parms);

	      /* Set dummy value to avoid adding in code for handling missing value in later stages */
	      Setattr($$, "value", "*parse error*");
	      SetFlag($$, "valueignored");

	      Setattr($$, "throws", $cpp_const.throws);
	      Setattr($$, "throw", $cpp_const.throwf);
	      Setattr($$, "noexcept", $cpp_const.nexcept);
	      Setattr($$, "final", $cpp_const.final);

	      if ($declarator.id) {
		/* Ignore all scoped declarations, could be 1. out of class function definition 2. friend function declaration 3. ... */
		String *p = Swig_scopename_prefix($declarator.id);
		if (p) {
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last($declarator.id);
		    Setattr($$, "name", lstr);
		    Delete(lstr);
		  } else {
		    Delete($$);
		    $$ = 0;
		  }
		  Delete(p);
		} else if (Strncmp($declarator.id, "::", 2) == 0) {
		  /* global scope declaration/definition ignored */
		  Delete($$);
		  $$ = 0;
		}
	      }

	      if ($cpp_const.qualifier && $storage_class && Strstr($storage_class, "static"))
		Swig_error(cparse_file, cparse_line, "Static function %s cannot have a qualifier.\n", Swig_name_decl($$));
	      Delete($storage_class);
	   }
           /* Alternate function syntax introduced in C++11:
              auto funcName(int x, int y) -> int; */
           | storage_class AUTO declarator cpp_const ARROW cpp_alternate_rettype virt_specifier_seq_opt initializer c_decl_tail {
              $$ = new_node("cdecl");
	      if ($cpp_const.qualifier) SwigType_push($declarator.type, $cpp_const.qualifier);
	      Setattr($$,"refqualifier",$cpp_const.refqualifier);
	      Setattr($$,"type",$cpp_alternate_rettype);
	      Setattr($$,"storage",$storage_class);
	      Setattr($$,"name",$declarator.id);
	      Setattr($$,"decl",$declarator.type);
	      Setattr($$,"parms",$declarator.parms);
	      Setattr($$,"throws",$cpp_const.throws);
	      Setattr($$,"throw",$cpp_const.throwf);
	      Setattr($$,"noexcept",$cpp_const.nexcept);
	      Setattr($$,"final",$cpp_const.final);
	      if (!$c_decl_tail) {
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr($$,"code",code);
		  Delete(code);
		}
	      } else {
		Node *n = $c_decl_tail;
		while (n) {
		  String *type = Copy($cpp_alternate_rettype);
		  Setattr(n,"type",type);
		  Setattr(n,"storage",$storage_class);
		  n = nextSibling(n);
		  Delete(type);
		}
	      }

	      if ($declarator.id) {
		/* Ignore all scoped declarations, could be 1. out of class function definition 2. friend function declaration 3. ... */
		String *p = Swig_scopename_prefix($declarator.id);
		if (p) {
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last($declarator.id);
		    Setattr($$,"name",lstr);
		    Delete(lstr);
		    set_nextSibling($$, $c_decl_tail);
		  } else {
		    Delete($$);
		    $$ = $c_decl_tail;
		  }
		  Delete(p);
		} else if (Strncmp($declarator.id, "::", 2) == 0) {
		  /* global scope declaration/definition ignored */
		  Delete($$);
		  $$ = $c_decl_tail;
		}
	      } else {
		set_nextSibling($$, $c_decl_tail);
	      }

	      if ($cpp_const.qualifier && $storage_class && Strstr($storage_class, "static"))
		Swig_error(cparse_file, cparse_line, "Static function %s cannot have a qualifier.\n", Swig_name_decl($$));
	      Delete($storage_class);
           }
           /* C++14 allows the trailing return type to be omitted.  It's
            * probably not feasible for SWIG to deduce it but we should
            * at least support parsing this so that the rest of an API
            * can be wrapped.  This also means you can provide declaration
            * with an explicit return type in the interface file for SWIG
            * to wrap.
            */
	   | storage_class AUTO declarator cpp_const LBRACE {
	      if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);

              $$ = new_node("cdecl");
	      if ($cpp_const.qualifier) SwigType_push($declarator.type, $cpp_const.qualifier);
	      Setattr($$, "refqualifier", $cpp_const.refqualifier);
	      Setattr($$, "type", NewString("auto"));
	      Setattr($$, "storage", $storage_class);
	      Setattr($$, "name", $declarator.id);
	      Setattr($$, "decl", $declarator.type);
	      Setattr($$, "parms", $declarator.parms);
	      Setattr($$, "throws", $cpp_const.throws);
	      Setattr($$, "throw", $cpp_const.throwf);
	      Setattr($$, "noexcept", $cpp_const.nexcept);
	      Setattr($$, "final", $cpp_const.final);

	      if ($declarator.id) {
		/* Ignore all scoped declarations, could be 1. out of class function definition 2. friend function declaration 3. ... */
		String *p = Swig_scopename_prefix($declarator.id);
		if (p) {
		  if ((Namespaceprefix && Strcmp(p, Namespaceprefix) == 0) ||
		      (Classprefix && Strcmp(p, Classprefix) == 0)) {
		    String *lstr = Swig_scopename_last($declarator.id);
		    Setattr($$, "name", lstr);
		    Delete(lstr);
		  } else {
		    Delete($$);
		    $$ = 0;
		  }
		  Delete(p);
		} else if (Strncmp($declarator.id, "::", 2) == 0) {
		  /* global scope declaration/definition ignored */
		  Delete($$);
		  $$ = 0;
		}
	      }

	      if ($cpp_const.qualifier && $storage_class && Strstr($storage_class, "static"))
		Swig_error(cparse_file, cparse_line, "Static function %s cannot have a qualifier.\n", Swig_name_decl($$));
	      Delete($storage_class);
	   }
	   /* C++11 auto variable declaration. */
	   | storage_class AUTO idcolon EQUAL definetype SEMI {
	      SwigType *type = deduce_type(&$definetype);
	      if (!type)
		type = NewString("auto");
	      $$ = new_node("cdecl");
	      Setattr($$, "type", type);
	      Setattr($$, "storage", $storage_class);
	      Setattr($$, "name", $idcolon);
	      Setattr($$, "decl", NewStringEmpty());
	      Setattr($$, "value", $definetype.val);
	      if ($definetype.stringval) Setattr($$, "stringval", $definetype.stringval);
	      if ($definetype.numval) Setattr($$, "numval", $definetype.numval);
	      Setattr($$, "valuetype", type);
	      Delete($storage_class);
	      Delete(type);
	   }
	   /* C++11 auto variable declaration for which we can't parse the initialiser. */
	   | storage_class AUTO idcolon EQUAL error SEMI {
	      SwigType *type = NewString("auto");
	      $$ = new_node("cdecl");
	      Setattr($$, "type", type);
	      Setattr($$, "storage", $storage_class);
	      Setattr($$, "name", $idcolon);
	      Setattr($$, "decl", NewStringEmpty());
	      Setattr($$, "valuetype", type);
	      Delete($storage_class);
	      Delete(type);
	   }
	   ;

/* Allow lists of variables and functions to be built up */

c_decl_tail    : SEMI { 
                   $$ = 0;
                   Clear(scanner_ccode); 
               }
               | COMMA declarator cpp_const initializer c_decl_tail[in] {
		 $$ = new_node("cdecl");
		 if ($cpp_const.qualifier) SwigType_push($declarator.type,$cpp_const.qualifier);
		 Setattr($$,"refqualifier",$cpp_const.refqualifier);
		 Setattr($$,"name",$declarator.id);
		 Setattr($$,"decl",$declarator.type);
		 Setattr($$,"parms",$declarator.parms);
		 Setattr($$,"value",$initializer.val);
		 if ($initializer.stringval) Setattr($$, "stringval", $initializer.stringval);
		 if ($initializer.numval) Setattr($$, "numval", $initializer.numval);
		 Setattr($$,"throws",$cpp_const.throws);
		 Setattr($$,"throw",$cpp_const.throwf);
		 Setattr($$,"noexcept",$cpp_const.nexcept);
		 Setattr($$,"final",$cpp_const.final);
		 if ($initializer.bitfield) {
		   Setattr($$,"bitfield", $initializer.bitfield);
		 }
		 if (!$in) {
		   if (Len(scanner_ccode)) {
		     String *code = Copy(scanner_ccode);
		     Setattr($$,"code",code);
		     Delete(code);
		   }
		 } else {
		   set_nextSibling($$, $in);
		 }
	       }
               | LBRACE { 
                   if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
                   $$ = 0;
               }
               | error {
		   $$ = 0;
		   if (yychar == RPAREN) {
		       Swig_error(cparse_file, cparse_line, "Unexpected closing parenthesis (')').\n");
		   } else {
		       Swig_error(cparse_file, cparse_line, "Syntax error - possibly a missing semicolon (';').\n");
		   }
		   Exit(EXIT_FAILURE);
               }
              ;

initializer   : def_args
	      | COLON expr {
		$$ = default_dtype;
		$$.bitfield = $expr.val;
	      }
              ;

cpp_alternate_rettype : primitive_type
              | TYPE_BOOL
              | TYPE_VOID
	      | c_enum_key idcolon {
		$$ = $idcolon;
		Insert($$, 0, "enum ");
	      }
              | TYPE_RAW
              | idcolon { $$ = $idcolon; }
              | idcolon AND {
                $$ = $idcolon;
                SwigType_add_reference($$);
              }
              | idcolon LAND {
                $$ = $idcolon;
                SwigType_add_rvalue_reference($$);
              }
              | CONST_QUAL idcolon AND {
                $$ = $idcolon;
                SwigType_add_qualifier($$, "const");
                SwigType_add_reference($$);
              }
              | CONST_QUAL idcolon LAND {
                $$ = $idcolon;
                SwigType_add_qualifier($$, "const");
                SwigType_add_rvalue_reference($$);
              }
              | decltype
              ;

/* ------------------------------------------------------------
   Lambda functions and expressions, such as:
   auto myFunc = [] { return something; };
   auto myFunc = [](int x, int y) { return x+y; };
   auto myFunc = [](int x, int y) -> int { return x+y; };
   auto myFunc = [](int x, int y) throw() -> int { return x+y; };
   auto six = [](int x, int y) { return x+y; }(4, 2);
   ------------------------------------------------------------ */
cpp_lambda_decl : storage_class AUTO idcolon EQUAL lambda_introducer lambda_template LPAREN parms RPAREN cpp_const lambda_body lambda_tail {
		  $$ = new_node("lambda");
		  Setattr($$,"name",$idcolon);
		  Delete($storage_class);
		  add_symbols($$);
	        }
                | storage_class AUTO idcolon EQUAL lambda_introducer lambda_template LPAREN parms RPAREN cpp_const ARROW type lambda_body lambda_tail {
		  $$ = new_node("lambda");
		  Setattr($$,"name",$idcolon);
		  Delete($storage_class);
		  add_symbols($$);
		}
                | storage_class AUTO idcolon EQUAL lambda_introducer lambda_template lambda_body lambda_tail {
		  $$ = new_node("lambda");
		  Setattr($$,"name",$idcolon);
		  Delete($storage_class);
		  add_symbols($$);
		}
                ;

lambda_introducer : LBRACKET {
		  if (skip_balanced('[',']') < 0) Exit(EXIT_FAILURE);
	        }
		;

lambda_template : LESSTHAN {
		  if (skip_balanced('<','>') < 0) Exit(EXIT_FAILURE);
		}
		| %empty
		;

lambda_body : LBRACE {
		  if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		}

lambda_tail :	SEMI
		| LPAREN {
		  if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
		} SEMI {
		}
		;

/* ------------------------------------------------------------
   enum
   or
   enum class
   ------------------------------------------------------------ */

c_enum_key : ENUM {
		   $$ = "enum";
	      }
	      | ENUM CLASS {
		   $$ = "enum class";
	      }
	      | ENUM STRUCT {
		   $$ = "enum struct";
	      }
	      ;

/* ------------------------------------------------------------
   base enum type (eg. unsigned short)
   ------------------------------------------------------------ */

c_enum_inherit : COLON type_right {
                   $$ = $type_right;
              }
              | %empty { $$ = 0; }
              ;
/* ------------------------------------------------------------
   enum [class] Name;
   enum [class] Name [: base_type];
   ------------------------------------------------------------ */

c_enum_forward_decl : storage_class c_enum_key ename c_enum_inherit SEMI {
		   SwigType *ty = 0;
		   int scopedenum = $ename && !Equal($c_enum_key, "enum");
		   $$ = new_node("enumforward");
		   ty = NewStringf("enum %s", $ename);
		   Setattr($$,"enumkey",$c_enum_key);
		   if (scopedenum)
		     SetFlag($$, "scopedenum");
		   Setattr($$,"name",$ename);
		   Setattr($$, "enumbase", $c_enum_inherit);
		   Setattr($$,"type",ty);
		   Setattr($$,"sym:weak", "1");
		   Delete($storage_class);
		   add_symbols($$);
	      }
              ;

/* ------------------------------------------------------------
   enum [class] Name [: base_type] { ... };
   or
   enum [class] Name [: base_type] { ... } MyEnum [= ...];
 * ------------------------------------------------------------ */

c_enum_decl :  storage_class c_enum_key ename c_enum_inherit LBRACE enumlist RBRACE SEMI {
		  SwigType *ty = 0;
		  int scopedenum = $ename && !Equal($c_enum_key, "enum");
		  $$ = new_enum_node($c_enum_inherit);
		  ty = NewStringf("enum %s", $ename);
		  Setattr($$,"enumkey",$c_enum_key);
		  if (scopedenum)
		    SetFlag($$, "scopedenum");
		  Setattr($$,"name",$ename);
		  Setattr($$,"type",ty);
		  appendChild($$,$enumlist);
		  add_symbols($$);      /* Add to tag space */

		  if (scopedenum) {
		    Swig_symbol_newscope();
		    Swig_symbol_setscopename($ename);
		    Delete(Namespaceprefix);
		    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  }

		  add_symbols($enumlist);      /* Add enum values to appropriate enum or enum class scope */

		  if (scopedenum) {
		    Setattr($$,"symtab", Swig_symbol_popscope());
		    Delete(Namespaceprefix);
		    Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  }
		  Delete($storage_class);
               }
	       | storage_class c_enum_key ename c_enum_inherit LBRACE enumlist RBRACE declarator cpp_const initializer c_decl_tail {
		 Node *n;
		 SwigType *ty = 0;
		 String   *unnamed = 0;
		 int       unnamedinstance = 0;
		 int scopedenum = $ename && !Equal($c_enum_key, "enum");

		 $$ = new_enum_node($c_enum_inherit);
		 Setattr($$,"enumkey",$c_enum_key);
		 if (scopedenum)
		   SetFlag($$, "scopedenum");
		 if ($ename) {
		   Setattr($$,"name",$ename);
		   ty = NewStringf("enum %s", $ename);
		 } else if ($declarator.id) {
		   unnamed = make_unnamed();
		   ty = NewStringf("enum %s", unnamed);
		   Setattr($$,"unnamed",unnamed);
                   /* name is not set for unnamed enum instances, e.g. enum { foo } Instance; */
		   if ($storage_class && Cmp($storage_class,"typedef") == 0) {
		     Setattr($$,"name",$declarator.id);
                   } else {
                     unnamedinstance = 1;
                   }
		   Setattr($$,"storage",$storage_class);
		 }
		 if ($declarator.id && Cmp($storage_class,"typedef") == 0) {
		   Setattr($$,"tdname",$declarator.id);
                   Setattr($$,"allows_typedef","1");
                 }
		 appendChild($$,$enumlist);
		 n = new_node("cdecl");
		 Setattr(n,"type",ty);
		 Setattr(n,"name",$declarator.id);
		 Setattr(n,"storage",$storage_class);
		 Setattr(n,"decl",$declarator.type);
		 Setattr(n,"parms",$declarator.parms);
		 Setattr(n,"unnamed",unnamed);

                 if (unnamedinstance) {
		   SwigType *cty = NewString("enum ");
		   Setattr($$,"type",cty);
		   SetFlag($$,"unnamedinstance");
		   SetFlag(n,"unnamedinstance");
		   Delete(cty);
                 }
		 if ($c_decl_tail) {
		   Node *p = $c_decl_tail;
		   set_nextSibling(n,p);
		   while (p) {
		     SwigType *cty = Copy(ty);
		     Setattr(p,"type",cty);
		     Setattr(p,"unnamed",unnamed);
		     Setattr(p,"storage",$storage_class);
		     Delete(cty);
		     p = nextSibling(p);
		   }
		 } else {
		   if (Len(scanner_ccode)) {
		     String *code = Copy(scanner_ccode);
		     Setattr(n,"code",code);
		     Delete(code);
		   }
		 }

                 /* Ensure that typedef enum ABC {foo} XYZ; uses XYZ for sym:name, like structs.
                  * Note that class_rename/yyrename are bit of a mess so used this simple approach to change the name. */
                 if ($declarator.id && $ename && Cmp($storage_class,"typedef") == 0) {
		   String *name = NewString($declarator.id);
                   Setattr($$, "parser:makename", name);
		   Delete(name);
                 }

		 add_symbols($$);       /* Add enum to tag space */
		 set_nextSibling($$,n);
		 Delete(n);

		 if (scopedenum) {
		   Swig_symbol_newscope();
		   Swig_symbol_setscopename($ename);
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 }

		 add_symbols($enumlist);      /* Add enum values to appropriate enum or enum class scope */

		 if (scopedenum) {
		   Setattr($$,"symtab", Swig_symbol_popscope());
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 }

	         add_symbols(n);
		 Delete($storage_class);
		 Delete(unnamed);
	       }
               ;

c_constructor_decl : storage_class type LPAREN parms RPAREN ctor_end {
                   /* This is a sick hack.  If the ctor_end has parameters,
                      and the parms parameter only has 1 parameter, this
                      could be a declaration of the form:

                         type (id)(parms)

			 Otherwise it's an error. */
                    int err = 0;
                    $$ = 0;

		    if ((ParmList_len($parms) == 1) && (!Swig_scopename_check($type))) {
		      SwigType *ty = Getattr($parms,"type");
		      String *name = Getattr($parms,"name");
		      err = 1;
		      if (!name) {
			$$ = new_node("cdecl");
			Setattr($$,"type",$type);
			Setattr($$,"storage",$storage_class);
			Setattr($$,"name",ty);

			if ($ctor_end.have_parms) {
			  SwigType *decl = NewStringEmpty();
			  SwigType_add_function(decl,$ctor_end.parms);
			  Setattr($$,"decl",decl);
			  Setattr($$,"parms",$ctor_end.parms);
			  if (Len(scanner_ccode)) {
			    String *code = Copy(scanner_ccode);
			    Setattr($$,"code",code);
			    Delete(code);
			  }
			}
			if ($ctor_end.defarg)
			  Setattr($$, "value", $ctor_end.defarg);
			if ($ctor_end.stringdefarg)
			  Setattr($$, "stringval", $ctor_end.stringdefarg);
			if ($ctor_end.numdefarg)
			  Setattr($$, "numval", $ctor_end.numdefarg);
			Setattr($$,"throws",$ctor_end.throws);
			Setattr($$,"throw",$ctor_end.throwf);
			Setattr($$,"noexcept",$ctor_end.nexcept);
			Setattr($$,"final",$ctor_end.final);
			err = 0;
		      }
		    }
		    Delete($storage_class);
		    if (err) {
		      Swig_error(cparse_file,cparse_line,"Syntax error in input(2).\n");
		      Exit(EXIT_FAILURE);
		    }
                }
                ;

/* ======================================================================
 *                       C++ Support
 * ====================================================================== */

cpp_declaration : cpp_class_decl
                | cpp_forward_class_decl
                | cpp_template_decl
                | cpp_using_decl
                | cpp_namespace_decl
                | cpp_catch_decl
                ;


/* A simple class/struct/union definition */

/* Note that class_virt_specifier_opt for supporting final classes introduces one shift-reduce conflict
   with C style variable declarations, such as: struct X final; */

cpp_class_decl: storage_class cpptype idcolon class_virt_specifier_opt inherit LBRACE <node>{
                   String *prefix;
                   List *bases = 0;
		   Node *scope = 0;
		   int errored_flag = 0;
		   String *code;
		   $$ = new_node("class");
		   Setattr($$,"kind",$cpptype);
		   if ($inherit) {
		     Setattr($$,"baselist", Getattr($inherit,"public"));
		     Setattr($$,"protectedbaselist", Getattr($inherit,"protected"));
		     Setattr($$,"privatebaselist", Getattr($inherit,"private"));
		   }
		   Setattr($$,"allows_typedef","1");

		   /* Temporary unofficial symtab for use until add_symbols() adds "sym:symtab" */
		   Setattr($$, "unofficial:symtab", Swig_symbol_current());
		  
		   /* If the class name is qualified.  We need to create or lookup namespace/scope entries */
		   scope = resolve_create_node_scope($idcolon, 1, &errored_flag);
		   /* save nscope_inner to the class - it may be overwritten in nested classes*/
		   Setattr($$, "nested:innerscope", nscope_inner);
		   Setattr($$, "nested:nscope", nscope);
		   Setfile(scope,cparse_file);
		   Setline(scope,cparse_line);
		   Setattr($$, "name", scope);

		   if (currentOuterClass) {
		     SetFlag($$, "nested");
		     Setattr($$, "nested:outer", currentOuterClass);
		     set_access_mode($$);
		   }
		   Swig_features_get(Swig_cparse_features(), Namespaceprefix, Getattr($$, "name"), 0, $$);
		   /* save yyrename to the class attribute, to be used later in add_symbols()*/
		   Setattr($$, "class_rename", make_name($$, scope, 0));
		   Setattr($$, "Classprefix", scope);
		   Classprefix = NewString(scope);
		   /* Deal with inheritance  */
		   if ($inherit)
		     bases = Swig_make_inherit_list(scope, Getattr($inherit, "public"), Namespaceprefix);
		   prefix = SwigType_istemplate_templateprefix(scope);
		   if (prefix) {
		     String *fbase, *tbase;
		     if (Namespaceprefix) {
		       fbase = NewStringf("%s::%s", Namespaceprefix, scope);
		       tbase = NewStringf("%s::%s", Namespaceprefix, prefix);
		     } else {
		       fbase = Copy(scope);
		       tbase = Copy(prefix);
		     }
		     Swig_name_inherit(tbase,fbase);
		     Delete(fbase);
		     Delete(tbase);
		   }
                   if (Strcmp($cpptype, "class") == 0) {
		     cplus_mode = CPLUS_PRIVATE;
		   } else {
		     cplus_mode = CPLUS_PUBLIC;
		   }
		   if (!cparse_cplusplus) {
		     set_scope_to_global();
		   }
		   Swig_symbol_newscope();
		   Swig_symbol_setscopename(scope);
		   Swig_inherit_base_symbols(bases);
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   cparse_start_line = cparse_line;

		   /* If there are active template parameters, we need to make sure they are
                      placed in the class symbol table so we can catch shadows */

		   if (template_parameters) {
		     Parm *tp = template_parameters;
		     while(tp) {
		       String *tpname = Copy(Getattr(tp,"name"));
		       Node *tn = new_node("templateparm");
		       Setattr(tn,"name",tpname);
		       Swig_symbol_cadd(tpname,tn);
		       tp = nextSibling(tp);
		       Delete(tpname);
		     }
		   }
		   Delete(prefix);
		   inclass = 1;
		   currentOuterClass = $$;
		   if (cparse_cplusplusout) {
		     /* save the structure declaration to declare it in global scope for C++ to see */
		     code = get_raw_text_balanced('{', '}');
		     Setattr($$, "code", code);
		     Delete(code);
		   }
               }[node] cpp_members RBRACE cpp_opt_declarators {
		   Node *p;
		   SwigType *ty;
		   Symtab *cscope;
		   Node *am = 0;
		   String *scpname = 0;
		   $$ = currentOuterClass;
		   currentOuterClass = Getattr($$, "nested:outer");
		   nscope_inner = Getattr($$, "nested:innerscope");
		   nscope = Getattr($$, "nested:nscope");
		   Delattr($$, "nested:innerscope");
		   Delattr($$, "nested:nscope");
		   if (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0) { /* actual parent class for this class */
		     Node* forward_declaration = Swig_symbol_clookup_no_inherit(Getattr($$,"name"), Getattr(nscope_inner, "symtab"));
		     if (forward_declaration) {
		       Setattr($$, "access", Getattr(forward_declaration, "access"));
		     }
		     Setattr($$, "nested:outer", nscope_inner);
		     SetFlag($$, "nested");
                   }
		   if (!currentOuterClass)
		     inclass = 0;
		   cscope = Getattr($$, "unofficial:symtab");
		   Delattr($$, "unofficial:symtab");
		   
		   /* Check for pure-abstract class */
		   Setattr($$,"abstracts", pure_abstracts($cpp_members));
		   
		   /* This bit of code merges in a previously defined %extend directive (if any) */
		   {
		     String *clsname = Swig_symbol_qualifiedscopename(0);
		     am = Getattr(Swig_extend_hash(), clsname);
		     if (am) {
		       Swig_extend_merge($$, am);
		       Delattr(Swig_extend_hash(), clsname);
		     }
		     Delete(clsname);
		   }
		   if (!classes) classes = NewHash();
		   scpname = Swig_symbol_qualifiedscopename(0);
		   Setattr(classes, scpname, $$);

		   appendChild($$, $cpp_members);
		   
		   if (am) 
		     Swig_extend_append_previous($$, am);

		   p = $cpp_opt_declarators;
		   if (p && !nscope_inner) {
		     if (!cparse_cplusplus && currentOuterClass)
		       appendChild(currentOuterClass, p);
		     else
		      appendSibling($$, p);
		   }
		   
		   if (nscope_inner) {
		     ty = NewString(scpname); /* if the class is declared out of scope, let the declarator use fully qualified type*/
		   } else if (cparse_cplusplus && !cparse_externc) {
		     ty = NewString(Getattr($node, "name"));
		   } else {
		     ty = NewStringf("%s %s", $cpptype, Getattr($node, "name"));
		   }
		   while (p) {
		     Setattr(p, "storage", $storage_class);
		     Setattr(p, "type" ,ty);
		     if (!cparse_cplusplus && currentOuterClass && (!Getattr(currentOuterClass, "name"))) {
		       SetFlag(p, "hasconsttype");
		     }
		     p = nextSibling(p);
		   }
		   if ($cpp_opt_declarators && Cmp($storage_class,"typedef") == 0)
		     add_typedef_name($$, $cpp_opt_declarators, Getattr($node, "name"), cscope, scpname);
		   Delete(scpname);

		   if (cplus_mode != CPLUS_PUBLIC) {
		   /* we 'open' the class at the end, to allow %template
		      to add new members */
		     Node *pa = new_node("access");
		     Setattr(pa, "kind", "public");
		     cplus_mode = CPLUS_PUBLIC;
		     appendChild($$, pa);
		     Delete(pa);
		   }
		   if (currentOuterClass)
		     restore_access_mode($$);
		   Setattr($$, "symtab", Swig_symbol_popscope());
		   Classprefix = Getattr($$, "Classprefix");
		   Delattr($$, "Classprefix");
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   if (cplus_mode == CPLUS_PRIVATE) {
		     $$ = 0; /* skip private nested classes */
		   } else if (cparse_cplusplus && currentOuterClass && ignore_nested_classes && !GetFlag($$, "feature:flatnested")) {
		     $$ = nested_forward_declaration($storage_class, $cpptype, Getattr($node, "name"), Copy(Getattr($node, "name")), $cpp_opt_declarators);
		   } else if (nscope_inner) {
		     /* this is tricky */
		     /* we add the declaration in the original namespace */
		     if (Strcmp(nodeType(nscope_inner), "class") == 0 && cparse_cplusplus && ignore_nested_classes && !GetFlag($$, "feature:flatnested"))
		       $$ = nested_forward_declaration($storage_class, $cpptype, Getattr($node, "name"), Copy(Getattr($node, "name")), $cpp_opt_declarators);
		     appendChild(nscope_inner, $$);
		     Swig_symbol_setscope(Getattr(nscope_inner, "symtab"));
		     Delete(Namespaceprefix);
		     Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		     yyrename = Copy(Getattr($$, "class_rename"));
		     add_symbols($$);
		     Delattr($$, "class_rename");
		     /* but the variable definition in the current scope */
		     Swig_symbol_setscope(cscope);
		     Delete(Namespaceprefix);
		     Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		     add_symbols($cpp_opt_declarators);
		     if (nscope) {
		       $$ = nscope; /* here we return recreated namespace tower instead of the class itself */
		       if ($cpp_opt_declarators) {
			 appendSibling($$, $cpp_opt_declarators);
		       }
		     } else if (!SwigType_istemplate(ty) && template_parameters == 0) { /* for template we need the class itself */
		       $$ = $cpp_opt_declarators;
		     }
		   } else {
		     Delete(yyrename);
		     yyrename = 0;
		     if (!cparse_cplusplus && currentOuterClass) { /* nested C structs go into global scope*/
		       Node *outer = currentOuterClass;
		       while (Getattr(outer, "nested:outer"))
			 outer = Getattr(outer, "nested:outer");
		       appendSibling(outer, $$);
		       Swig_symbol_setscope(cscope); /* declaration goes in the parent scope */
		       add_symbols($cpp_opt_declarators);
		       set_scope_to_global();
		       Delete(Namespaceprefix);
		       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		       yyrename = Copy(Getattr($$, "class_rename"));
		       add_symbols($$);
		       if (!cparse_cplusplusout)
			 Delattr($$, "nested:outer");
		       Delattr($$, "class_rename");
		       $$ = 0;
		     } else {
		       yyrename = Copy(Getattr($$, "class_rename"));
		       add_symbols($$);
		       add_symbols($cpp_opt_declarators);
		       Delattr($$, "class_rename");
		     }
		   }
		   Delete(ty);
		   Swig_symbol_setscope(cscope);
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   Classprefix = currentOuterClass ? Getattr(currentOuterClass, "Classprefix") : 0;
		   Delete($storage_class);
	       }

/* An unnamed struct, possibly with a typedef */

             | storage_class cpptype inherit LBRACE <node>{
	       String *unnamed;
	       String *code;
	       unnamed = make_unnamed();
	       $$ = new_node("class");
	       Setattr($$,"kind",$cpptype);
	       if ($inherit) {
		 Setattr($$,"baselist", Getattr($inherit,"public"));
		 Setattr($$,"protectedbaselist", Getattr($inherit,"protected"));
		 Setattr($$,"privatebaselist", Getattr($inherit,"private"));
	       }
	       Setattr($$,"storage",$storage_class);
	       Setattr($$,"unnamed",unnamed);
	       Setattr($$,"allows_typedef","1");

	       /* Temporary unofficial symtab for use until add_symbols() adds "sym:symtab" */
	       Setattr($$, "unofficial:symtab", Swig_symbol_current());

	       if (currentOuterClass) {
		 SetFlag($$, "nested");
		 Setattr($$, "nested:outer", currentOuterClass);
		 set_access_mode($$);
	       }
	       Swig_features_get(Swig_cparse_features(), Namespaceprefix, 0, 0, $$);
	       /* save yyrename to the class attribute, to be used later in add_symbols()*/
	       Setattr($$, "class_rename", make_name($$,0,0));
	       if (Strcmp($cpptype, "class") == 0) {
		 cplus_mode = CPLUS_PRIVATE;
	       } else {
		 cplus_mode = CPLUS_PUBLIC;
	       }
	       Swig_symbol_newscope();
	       cparse_start_line = cparse_line;
	       currentOuterClass = $$;
	       inclass = 1;
	       Classprefix = 0;
	       Delete(Namespaceprefix);
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       /* save the structure declaration to make a typedef for it later*/
	       code = get_raw_text_balanced('{', '}');
	       Setattr($$, "code", code);
	       Delete(code);
	     }[node] cpp_members RBRACE cpp_opt_declarators {
	       String *unnamed;
               List *bases = 0;
	       String *name = 0;
	       Node *n;
	       Symtab *cscope;
	       Classprefix = 0;
	       (void)$node;
	       $$ = currentOuterClass;
	       currentOuterClass = Getattr($$, "nested:outer");
	       if (!currentOuterClass)
		 inclass = 0;
	       else
		 restore_access_mode($$);

	       cscope = Getattr($$, "unofficial:symtab");
	       Delattr($$, "unofficial:symtab");

	       unnamed = Getattr($$,"unnamed");
               /* Check for pure-abstract class */
	       Setattr($$,"abstracts", pure_abstracts($cpp_members));
	       n = $cpp_opt_declarators;
	       if (cparse_cplusplus && currentOuterClass && ignore_nested_classes && !GetFlag($$, "feature:flatnested")) {
		 String *name = n ? Copy(Getattr(n, "name")) : 0;
		 $$ = nested_forward_declaration($storage_class, $cpptype, 0, name, n);
	       } else if (n) {
	         appendSibling($$,n);
		 /* If a proper typedef name was given, we'll use it to set the scope name */
		 name = try_to_find_a_name_for_unnamed_structure($storage_class, n);
		 if (name) {
		   String *scpname = 0;
		   SwigType *ty;
		   Setattr($$,"tdname",name);
		   Setattr($$,"name",name);
		   Swig_symbol_setscopename(name);
		   if ($inherit)
		     bases = Swig_make_inherit_list(name,Getattr($inherit,"public"),Namespaceprefix);
		   Swig_inherit_base_symbols(bases);

		     /* If a proper name was given, we use that as the typedef, not unnamed */
		   Clear(unnamed);
		   Append(unnamed, name);
		   if (cparse_cplusplus && !cparse_externc) {
		     ty = NewString(name);
		   } else {
		     ty = NewStringf("%s %s", $cpptype,name);
		   }
		   while (n) {
		     Setattr(n,"storage",$storage_class);
		     Setattr(n, "type", ty);
		     if (!cparse_cplusplus && currentOuterClass && (!Getattr(currentOuterClass, "name"))) {
		       SetFlag(n,"hasconsttype");
		     }
		     n = nextSibling(n);
		   }
		   n = $cpp_opt_declarators;

		   /* Check for previous extensions */
		   {
		     String *clsname = Swig_symbol_qualifiedscopename(0);
		     Node *am = Getattr(Swig_extend_hash(),clsname);
		     if (am) {
		       /* Merge the extension into the symbol table */
		       Swig_extend_merge($$,am);
		       Swig_extend_append_previous($$,am);
		       Delattr(Swig_extend_hash(),clsname);
		     }
		     Delete(clsname);
		   }
		   if (!classes) classes = NewHash();
		   scpname = Swig_symbol_qualifiedscopename(0);
		   Setattr(classes,scpname,$$);
		   Delete(scpname);
		 } else { /* no suitable name was found for a struct */
		   Setattr($$, "nested:unnamed", Getattr(n, "name")); /* save the name of the first declarator for later use in name generation*/
		   while (n) { /* attach unnamed struct to the declarators, so that they would receive proper type later*/
		     Setattr(n, "nested:unnamedtype", $$);
		     Setattr(n, "storage", $storage_class);
		     n = nextSibling(n);
		   }
		   n = $cpp_opt_declarators;
		   Swig_symbol_setscopename("<unnamed>");
		 }
		 appendChild($$,$cpp_members);
		 /* Pop the scope */
		 Setattr($$,"symtab",Swig_symbol_popscope());
		 if (name) {
		   Delete(yyrename);
		   yyrename = Copy(Getattr($$, "class_rename"));
		   Delete(Namespaceprefix);
		   Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		   add_symbols($$);
		   add_symbols(n);
		   Delattr($$, "class_rename");
		 } else if (cparse_cplusplus)
		   $$ = 0; /* ignore unnamed structs for C++ */
		   Delete(unnamed);
	       } else { /* unnamed struct without declarator*/
		 Swig_symbol_popscope();
	         Delete(Namespaceprefix);
		 Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		 add_symbols($cpp_members);
		 Delete($$);
		 $$ = $cpp_members; /* pass member list to outer class/namespace (instead of self)*/
	       }
	       Swig_symbol_setscope(cscope);
	       Delete(Namespaceprefix);
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       Classprefix = currentOuterClass ? Getattr(currentOuterClass, "Classprefix") : 0;
	       Delete($storage_class);
              }
             ;

cpp_opt_declarators :  SEMI { $$ = 0; }
                    |  declarator cpp_const initializer c_decl_tail {
                        $$ = new_node("cdecl");
                        Setattr($$,"name",$declarator.id);
                        Setattr($$,"decl",$declarator.type);
                        Setattr($$,"parms",$declarator.parms);
			set_nextSibling($$, $c_decl_tail);
                    }
                    ;
/* ------------------------------------------------------------
   class Name;
   ------------------------------------------------------------ */

cpp_forward_class_decl : storage_class cpptype idcolon SEMI {
	      if ($storage_class && Strstr($storage_class, "friend")) {
		/* Ignore */
                $$ = 0; 
	      } else {
		$$ = new_node("classforward");
		Setattr($$,"kind",$cpptype);
		Setattr($$,"name",$idcolon);
		Setattr($$,"sym:weak", "1");
		add_symbols($$);
	      }
	      Delete($storage_class);
             }
             ;

/* ------------------------------------------------------------
   template<...> decl
   ------------------------------------------------------------ */

cpp_template_decl : TEMPLATE LESSTHAN template_parms GREATERTHAN { 
		    if (currentOuterClass)
		      Setattr(currentOuterClass, "template_parameters", template_parameters);
		    template_parameters = $template_parms; 
		    parsing_template_declaration = 1;
		  } cpp_template_possible {
			String *tname = 0;
			int     error = 0;

			/* check if we get a namespace node with a class declaration, and retrieve the class */
			Symtab *cscope = Swig_symbol_current();
			Symtab *sti = 0;
			Node *ntop = $cpp_template_possible;
			Node *ni = ntop;
			SwigType *ntype = ni ? nodeType(ni) : 0;
			while (ni && Strcmp(ntype,"namespace") == 0) {
			  sti = Getattr(ni,"symtab");
			  ni = firstChild(ni);
			  ntype = nodeType(ni);
			}
			if (sti) {
			  Swig_symbol_setscope(sti);
			  Delete(Namespaceprefix);
			  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
			  $$ = ni;
			} else {
			  $$ = $cpp_template_possible;
			}

			if ($$) tname = Getattr($$,"name");
			
			/* Check if the class is a template specialization */
			if (($$) && (Strchr(tname,'<')) && (!is_operator(tname))) {
			  /* If a specialization.  Check if defined. */
			  Node *tempn = 0;
			  {
			    String *tbase = SwigType_templateprefix(tname);
			    tempn = Swig_symbol_clookup_local(tbase,0);
			    if (!tempn || (Strcmp(nodeType(tempn),"template") != 0)) {
			      SWIG_WARN_NODE_BEGIN(tempn);
			      Swig_warning(WARN_PARSE_TEMPLATE_SP_UNDEF, Getfile($$),Getline($$),"Specialization of non-template '%s'.\n", tbase);
			      SWIG_WARN_NODE_END(tempn);
			      tempn = 0;
			      error = 1;
			    }
			    Delete(tbase);
			  }
			  Setattr($$,"specialization","1");
			  Setattr($$,"templatetype",nodeType($$));
			  set_nodeType($$,"template");
			  /* Template partial specialization */
			  if (tempn && ($template_parms) && ($$)) {
			    ParmList *primary_templateparms = Getattr(tempn, "templateparms");
			    String *targs = SwigType_templateargs(tname); /* tname contains name and specialized template parameters, for example: X<(p.T,TT)> */
			    List *tlist = SwigType_parmlist(targs);
			    int specialization_parms_len = Len(tlist);
			    if (!Getattr($$,"sym:weak")) {
			      Setattr($$,"sym:typename","1");
			    }
			    Setattr($$, "primarytemplate", tempn);
			    Setattr($$, "templateparms", $template_parms);
			    Delattr($$, "specialization");
			    Setattr($$, "partialspecialization", "1");
			    
			    if (specialization_parms_len > ParmList_len(primary_templateparms)) {
			      Swig_error(Getfile($$), Getline($$), "Template partial specialization has more arguments than primary template %d %d.\n", specialization_parms_len, ParmList_len(primary_templateparms));
			      
			    } else if (specialization_parms_len < ParmList_numrequired(primary_templateparms)) {
			      Swig_error(Getfile($$), Getline($$), "Template partial specialization has fewer arguments than primary template %d %d.\n", specialization_parms_len, ParmList_len(primary_templateparms));
			    } else {
			      /* Create a specialized name with template parameters replaced with $ variables, such as, X<(T1,p.T2) => X<($1,p.$2)> */
			      Parm *p = $template_parms;
			      String *fname = NewString(tname);
			      String *ffname = 0;
			      ParmList *partialparms = 0;

			      char   tmp[32];
			      int i = 0;
			      while (p) {
				String *name = Getattr(p,"name");
				++i;
				if (!name) {
				  p = nextSibling(p);
				  continue;
				}
				sprintf(tmp, "$%d", i);
				Replaceid(fname, name, tmp);
				p = nextSibling(p);
			      }
			      /* Patch argument names with typedef */
			      {
				Iterator tt;
				Parm *parm_current = 0;
				List *tparms = SwigType_parmlist(fname);
				ffname = SwigType_templateprefix(fname);
				Append(ffname,"<(");
				for (tt = First(tparms); tt.item; ) {
				  SwigType *rtt = Swig_symbol_typedef_reduce(tt.item,0);
				  SwigType *ttr = Swig_symbol_type_qualify(rtt,0);

				  Parm *newp = NewParmWithoutFileLineInfo(ttr, 0);
				  if (partialparms)
				    set_nextSibling(parm_current, newp);
				  else
				    partialparms = newp;
				  parm_current = newp;

				  Append(ffname,ttr);
				  tt = Next(tt);
				  if (tt.item) Putc(',',ffname);
				  Delete(rtt);
				  Delete(ttr);
				}
				Delete(tparms);
				Append(ffname,")>");
			      }
			      {
				/* Replace each primary template parameter's name and value with $ variables, such as, class Y,class T=Y => class $1,class $2=$1 */
				ParmList *primary_templateparms_copy = CopyParmList(primary_templateparms);
				p = primary_templateparms_copy;
				i = 0;
				while (p) {
				  String *name = Getattr(p, "name");
				  Parm *pp = nextSibling(p);
				  ++i;
				  sprintf(tmp, "$%d", i);
				  while (pp) {
				    Replaceid(Getattr(pp, "value"), name, tmp);
				    pp = nextSibling(pp);
				  }
				  Setattr(p, "name", NewString(tmp));
				  p = nextSibling(p);
				}
				/* Modify partialparms by adding in missing default values ($ variables) from primary template parameters */
				partialparms = Swig_cparse_template_partialargs_expand(partialparms, tempn, primary_templateparms_copy);
				Delete(primary_templateparms_copy);
			      }
			      {
				Node *new_partial = NewHash();
				String *partials = Getattr(tempn,"partials");
				if (!partials) {
				  partials = NewList();
				  Setattr(tempn,"partials",partials);
				  Delete(partials);
				}
				/*			      Printf(stdout,"partial: fname = '%s', '%s'\n", fname, Swig_symbol_typedef_reduce(fname,0)); */
				Setattr(new_partial, "partialparms", partialparms);
				Setattr(new_partial, "templcsymname", ffname);
				Append(partials, new_partial);
			      }
			      Setattr($$,"partialargs",ffname);
			      Swig_symbol_cadd(ffname,$$);
			    }
			    Delete(tlist);
			    Delete(targs);
			  } else {
			    /* An explicit template specialization */
			    /* add default args from primary (unspecialized) template */
			    String *ty = Swig_symbol_template_deftype(tname,0);
			    String *fname = Swig_symbol_type_qualify(ty,0);
			    Swig_symbol_cadd(fname,$$);
			    Delete(ty);
			    Delete(fname);
			  }
			} else if ($$) {
			  Setattr($$, "templatetype", nodeType($$));
			  set_nodeType($$,"template");
			  Setattr($$,"templateparms", $template_parms);
			  if (!Getattr($$,"sym:weak")) {
			    Setattr($$,"sym:typename","1");
			  }
			  add_symbols($$);
			  default_arguments($$);
			  /* We also place a fully parameterized version in the symbol table */
			  {
			    Parm *p;
			    String *fname = NewStringf("%s<(", Getattr($$,"name"));
			    p = $template_parms;
			    while (p) {
			      String *n = Getattr(p,"name");
			      if (!n) n = Getattr(p,"type");
			      Append(fname,n);
			      p = nextSibling(p);
			      if (p) Putc(',',fname);
			    }
			    Append(fname,")>");
			    Swig_symbol_cadd(fname,$$);
			  }
			}
			$$ = ntop;
			Swig_symbol_setscope(cscope);
			Delete(Namespaceprefix);
			Namespaceprefix = Swig_symbol_qualifiedscopename(0);
			if (error || (nscope_inner && Strcmp(nodeType(nscope_inner), "class") == 0)) {
			  $$ = 0;
			}
			if (currentOuterClass)
			  template_parameters = Getattr(currentOuterClass, "template_parameters");
			else
			  template_parameters = 0;
			parsing_template_declaration = 0;
                }

		/* Class template explicit instantiation definition */
                | TEMPLATE cpptype idcolon {
		  Swig_warning(WARN_PARSE_EXPLICIT_TEMPLATE, cparse_file, cparse_line, "Explicit template instantiation ignored.\n");
                  $$ = 0; 
		}

		/* Function template explicit instantiation definition */
		| TEMPLATE cpp_alternate_rettype idcolon LPAREN parms RPAREN {
			Swig_warning(WARN_PARSE_EXPLICIT_TEMPLATE, cparse_file, cparse_line, "Explicit template instantiation ignored.\n");
                  $$ = 0; 
		}

		/* Class template explicit instantiation declaration (extern template) */
		| EXTERN TEMPLATE cpptype idcolon {
		  Swig_warning(WARN_PARSE_EXTERN_TEMPLATE, cparse_file, cparse_line, "Extern template ignored.\n");
                  $$ = 0; 
                }

		/* Function template explicit instantiation declaration (extern template) */
		| EXTERN TEMPLATE cpp_alternate_rettype idcolon LPAREN parms RPAREN {
			Swig_warning(WARN_PARSE_EXTERN_TEMPLATE, cparse_file, cparse_line, "Extern template ignored.\n");
                  $$ = 0; 
		}
		;

cpp_template_possible:  c_decl
                | cpp_class_decl
                | cpp_constructor_decl
                | cpp_template_decl {
		  $$ = 0;
                }
                | cpp_forward_class_decl
                | cpp_conversion_operator
                ;

template_parms : template_parms_builder {
		 $$ = $template_parms_builder.parms;
	       }
	       | %empty {
		 $$ = 0;
	       }
	       ;

template_parms_builder : templateparameter {
		    $$.parms = $$.last = $templateparameter;
		  }
		  | template_parms_builder[in] COMMA templateparameter {
		    // Build a linked list in the order specified, but avoiding
		    // a right recursion rule because "Right recursion uses up
		    // space on the Bison stack in proportion to the number of
		    // elements in the sequence".
		    set_nextSibling($in.last, $templateparameter);
		    $$.parms = $in.parms;
		    $$.last = $templateparameter;
		  }
		  ;

templateparameter : templcpptype def_args {
		    $$ = NewParmWithoutFileLineInfo($templcpptype, 0);
		    Setfile($$, cparse_file);
		    Setline($$, cparse_line);
		    Setattr($$, "value", $def_args.val);
		    if ($def_args.stringval) Setattr($$, "stringval", $def_args.stringval);
		    if ($def_args.numval) Setattr($$, "numval", $def_args.numval);
		  }
		  | TEMPLATE LESSTHAN template_parms GREATERTHAN cpptype idcolon def_args {
		    $$ = NewParmWithoutFileLineInfo(NewStringf("template< %s > %s %s", ParmList_str_defaultargs($template_parms), $cpptype, $idcolon), $idcolon);
		    Setfile($$, cparse_file);
		    Setline($$, cparse_line);
		    if ($def_args.val) {
		      Setattr($$, "value", $def_args.val);
		    }
		  }
		  | TEMPLATE LESSTHAN template_parms GREATERTHAN cpptype def_args {
		    $$ = NewParmWithoutFileLineInfo(NewStringf("template< %s > %s", ParmList_str_defaultargs($template_parms), $cpptype), 0);
		    Setfile($$, cparse_file);
		    Setline($$, cparse_line);
		    if ($def_args.val) {
		      Setattr($$, "value", $def_args.val);
		    }
		  }
		  | parm {
		    Parm *p = $parm;
		    String *name = Getattr(p, "name");
		    $$ = $parm;

		    /* Correct the 'type name' parameter string, split into the appropriate "name" and "type" attributes */
		    if (!name) {
		      String *type = Getattr(p, "type");
		      if ((Strncmp(type, "class ", 6) == 0) || (Strncmp(type, "typename ", 9) == 0)) {
			/* A 'class T' parameter */
			const char *t = Strchr(type, ' ');
			Setattr(p, "name", t + 1);
			Setattr(p, "type", NewStringWithSize(type, (int)(t - Char(type))));
		      } else if ((Strncmp(type, "v.class ", 8) == 0) || (Strncmp(type, "v.typename ", 11) == 0)) {
			/* Variadic template args */
			const char *t = Strchr(type, ' ');
			Setattr(p, "name", t + 1);
			Setattr(p, "type", NewStringWithSize(type, (int)(t - Char(type))));
		      }
		    }
                  }
                  ;

/* Namespace support */

cpp_using_decl : USING idcolon SEMI {
                  String *uname = Swig_symbol_type_qualify($idcolon,0);
                  /* Possible TODO: In testcase using_member_multiple_inherit class Susing3, uname is "Susing1::usingmethod" instead of "Susing2::usingmethod" */
		  String *name = Swig_scopename_last($idcolon);
                  $$ = new_node("using");
		  Setattr($$,"uname",uname);
		  Setattr($$,"name", name);
		  Delete(uname);
		  Delete(name);
		  add_symbols($$);
             }
	     | USING TYPENAME idcolon SEMI {
		  String *uname = Swig_symbol_type_qualify($idcolon,0);
		  String *name = Swig_scopename_last($idcolon);
		  $$ = new_node("using");
		  Setattr($$,"uname",uname);
		  Setattr($$,"name", name);
		  Delete(uname);
		  Delete(name);
		  add_symbols($$);
	     }
             | USING NAMESPACE idcolon SEMI {
	       Node *n = Swig_symbol_clookup($idcolon,0);
	       if (!n) {
		 Swig_error(cparse_file, cparse_line, "Nothing known about namespace '%s'\n", SwigType_namestr($idcolon));
		 $$ = 0;
	       } else {

		 while (Strcmp(nodeType(n),"using") == 0) {
		   n = Getattr(n,"node");
		 }
		 if (n) {
		   if (Strcmp(nodeType(n),"namespace") == 0) {
		     Symtab *current = Swig_symbol_current();
		     Symtab *symtab = Getattr(n,"symtab");
		     $$ = new_node("using");
		     Setattr($$,"node",n);
		     Setattr($$,"namespace", $idcolon);
		     if (current != symtab) {
		       Swig_symbol_inherit(symtab);
		     }
		   } else {
		     Swig_error(cparse_file, cparse_line, "'%s' is not a namespace.\n", SwigType_namestr($idcolon));
		     $$ = 0;
		   }
		 } else {
		   $$ = 0;
		 }
	       }
             }
             ;

cpp_namespace_decl : NAMESPACE idcolon LBRACE <node>{
                Hash *h;
		Node *parent_ns = 0;
		List *scopes = Swig_scopename_tolist($idcolon);
		int ilen = Len(scopes);
		int i;

/*
Printf(stdout, "==== Namespace %s creation...\n", $idcolon);
*/
		$$ = 0;
		for (i = 0; i < ilen; i++) {
		  Node *ns = new_node("namespace");
		  Symtab *current_symtab = Swig_symbol_current();
		  String *scopename = Getitem(scopes, i);
		  Setattr(ns, "name", scopename);
		  $$ = ns;
		  if (parent_ns)
		    appendChild(parent_ns, ns);
		  parent_ns = ns;
		  h = Swig_symbol_clookup(scopename, 0);
		  if (h && (current_symtab == Getattr(h, "sym:symtab")) && (Strcmp(nodeType(h), "namespace") == 0)) {
/*
Printf(stdout, "  Scope %s [found C++17 style]\n", scopename);
*/
		    if (Getattr(h, "alias")) {
		      h = Getattr(h, "namespace");
		      Swig_warning(WARN_PARSE_NAMESPACE_ALIAS, cparse_file, cparse_line, "Namespace alias '%s' not allowed here. Assuming '%s'\n",
				   scopename, Getattr(h, "name"));
		      scopename = Getattr(h, "name");
		    }
		    Swig_symbol_setscope(Getattr(h, "symtab"));
		  } else {
/*
Printf(stdout, "  Scope %s [creating single scope C++17 style]\n", scopename);
*/
		    h = Swig_symbol_newscope();
		    Swig_symbol_setscopename(scopename);
		  }
		  Delete(Namespaceprefix);
		  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		}
		Delete(scopes);
             }[node] interface RBRACE {
		Node *n = $node;
		Node *top_ns = 0;
		do {
		  Setattr(n, "symtab", Swig_symbol_popscope());
		  Delete(Namespaceprefix);
		  Namespaceprefix = Swig_symbol_qualifiedscopename(0);
		  add_symbols(n);
		  top_ns = n;
		  n = parentNode(n);
		} while(n);
		appendChild($node, firstChild($interface));
		Delete($interface);
		$$ = top_ns;
             } 
             | NAMESPACE LBRACE <node>{
	       Hash *h;
	       $$ = Swig_symbol_current();
	       h = Swig_symbol_clookup("    ",0);
	       if (h && (Strcmp(nodeType(h),"namespace") == 0)) {
		 Swig_symbol_setscope(Getattr(h,"symtab"));
	       } else {
		 Swig_symbol_newscope();
		 /* we don't use "__unnamed__", but a long 'empty' name */
		 Swig_symbol_setscopename("    ");
	       }
	       Namespaceprefix = 0;
             }[node] interface RBRACE {
	       $$ = $interface;
	       set_nodeType($$,"namespace");
	       Setattr($$,"unnamed","1");
	       Setattr($$,"symtab", Swig_symbol_popscope());
	       Swig_symbol_setscope($node);
	       Delete(Namespaceprefix);
	       Namespaceprefix = Swig_symbol_qualifiedscopename(0);
	       add_symbols($$);
             }
             | NAMESPACE identifier EQUAL idcolon SEMI {
	       /* Namespace alias */
	       Node *n;
	       $$ = new_node("namespace");
	       Setattr($$,"name",$identifier);
	       Setattr($$,"alias",$idcolon);
	       n = Swig_symbol_clookup($idcolon,0);
	       if (!n) {
		 Swig_error(cparse_file, cparse_line, "Unknown namespace '%s'\n", SwigType_namestr($idcolon));
		 $$ = 0;
	       } else {
		 if (Strcmp(nodeType(n),"namespace") != 0) {
		   Swig_error(cparse_file, cparse_line, "'%s' is not a namespace\n", SwigType_namestr($idcolon));
		   $$ = 0;
		 } else {
		   while (Getattr(n,"alias")) {
		     n = Getattr(n,"namespace");
		   }
		   Setattr($$,"namespace",n);
		   add_symbols($$);
		   /* Set up a scope alias */
		   Swig_symbol_alias($identifier,Getattr(n,"symtab"));
		 }
	       }
             }
             ;

cpp_members : cpp_members_builder {
		 $$ = $cpp_members_builder.node;
	       }
	       | cpp_members_builder DOXYGENSTRING {
		 /* Quietly ignore misplaced doxygen string after a member, like Doxygen does */
		 $$ = $cpp_members_builder.node;
		 Delete($DOXYGENSTRING);
	       }
	       | %empty {
		 $$ = 0;
	       }
	       | DOXYGENSTRING {
		 /* Quietly ignore misplaced doxygen string in empty class, like Doxygen does */
		 $$ = 0;
		 Delete($DOXYGENSTRING);
	       }
	       | error {
		 Swig_error(cparse_file, cparse_line, "Syntax error in input(3).\n");
		 Exit(EXIT_FAILURE);
	       }
	       ;

cpp_members_builder : cpp_member {
	     $$.node = $$.last = $cpp_member;
	   }
	   | cpp_members_builder[in] cpp_member {
	     // Build a linked list in the order specified, but avoiding
	     // a right recursion rule because "Right recursion uses up
	     // space on the Bison stack in proportion to the number of
	     // elements in the sequence".
	     if ($cpp_member) {
	       if ($in.node) {
		 Node *last = $in.last;
		 /* Advance to the last sibling. */
		 for (Node *p = last; p; p = nextSibling(p)) {
		   last = p;
		 }
		 set_nextSibling(last, $cpp_member);
		 set_previousSibling($cpp_member, last);
		 $$.node = $in.node;
	       } else {
		 $$.node = $$.last = $cpp_member;
	       }
	     } else {
	       $$ = $in;
	     }
	   }
	   ;

/* ======================================================================
 *                         C++ Class members
 * ====================================================================== */

/* A class member.  May be data or a function. Static or virtual as well */

cpp_member_no_dox : c_declaration
             | cpp_constructor_decl { 
                 $$ = $cpp_constructor_decl; 
		 if (extendmode && current_class) {
		   String *symname;
		   symname= make_name($$,Getattr($$,"name"), Getattr($$,"decl"));
		   if (Strcmp(symname,Getattr($$,"name")) == 0) {
		     /* No renaming operation.  Set name to class name */
		     Delete(yyrename);
		     yyrename = NewString(Getattr(current_class,"sym:name"));
		   } else {
		     Delete(yyrename);
		     yyrename = symname;
		   }
		 }
		 add_symbols($$);
                 default_arguments($$);
             }
             | cpp_destructor_decl
             | cpp_protection_decl
             | cpp_swig_directive
             | cpp_conversion_operator
             | cpp_forward_class_decl
	     | cpp_class_decl
             | storage_class idcolon SEMI { $$ = 0; Delete($storage_class); }
             | cpp_using_decl
             | cpp_template_decl
             | cpp_catch_decl
	     | include_directive
             | template_directive
             | warn_directive
             | anonymous_bitfield { $$ = 0; }
             | fragment_directive
             | types_directive
             | SEMI { $$ = 0; }

cpp_member   : cpp_member_no_dox
             | DOXYGENSTRING cpp_member_no_dox {
	         $$ = $cpp_member_no_dox;
		 set_comment($cpp_member_no_dox, $DOXYGENSTRING);
	     }
             | cpp_member_no_dox DOXYGENPOSTSTRING {
	         $$ = $cpp_member_no_dox;
		 set_comment($cpp_member_no_dox, $DOXYGENPOSTSTRING);
	     }
	     | EXTEND LBRACE {
	       extendmode = 1;
	       if (cplus_mode != CPLUS_PUBLIC) {
		 Swig_error(cparse_file,cparse_line,"%%extend can only be used in a public section\n");
	       }
	     } cpp_members RBRACE {
	       extendmode = 0;
	       $$ = new_node("extend");
	       mark_nodes_as_extend($cpp_members);
	       appendChild($$, $cpp_members);
	     }
             ;

/* Possibly a constructor */
/* Note: the use of 'type' is here to resolve a shift-reduce conflict.  For example:
            typedef Foo ();
            typedef Foo (*ptr)();
*/
  
cpp_constructor_decl : storage_class type LPAREN parms RPAREN ctor_end {
	      /* Cannot be a constructor declaration/definition if parsed as a friend destructor/constructor
	         or a badly declared friend function without return type */
	      int isfriend = Strstr($storage_class, "friend") != NULL;
	      if (!isfriend && (inclass || extendmode)) {
	        String *name = SwigType_templateprefix($type); /* A constructor can optionally be declared with template parameters before C++20, strip these off */
		SwigType *decl = NewStringEmpty();
		$$ = new_node("constructor");
		Setattr($$,"storage",$storage_class);
		Setattr($$, "name", name);
		Setattr($$,"parms",$parms);
		SwigType_add_function(decl,$parms);
		Setattr($$,"decl",decl);
		Setattr($$,"throws",$ctor_end.throws);
		Setattr($$,"throw",$ctor_end.throwf);
		Setattr($$,"noexcept",$ctor_end.nexcept);
		Setattr($$,"final",$ctor_end.final);
		if (Len(scanner_ccode)) {
		  String *code = Copy(scanner_ccode);
		  Setattr($$,"code",code);
		  Delete(code);
		}
		SetFlag($$,"feature:new");
		if ($ctor_end.defarg)
		  Setattr($$, "value", $ctor_end.defarg);
		if ($ctor_end.stringdefarg)
		  Setattr($$, "stringval", $ctor_end.stringdefarg);
		if ($ctor_end.numdefarg)
		  Setattr($$, "numval", $ctor_end.numdefarg);
	      } else {
		$$ = 0;
              }
	      Delete($storage_class);
              }
              ;

/* A destructor */

cpp_destructor_decl : storage_class NOT idtemplate LPAREN parms RPAREN cpp_vend {
	       String *name = SwigType_templateprefix($idtemplate); /* A destructor can optionally be declared with template parameters before C++20, strip these off */
	       Insert(name, 0, "~");
	       $$ = new_node("destructor");
	       Setattr($$, "storage", $storage_class);
	       Setattr($$, "name", name);
	       Delete(name);
	       if (Len(scanner_ccode)) {
		 String *code = Copy(scanner_ccode);
		 Setattr($$, "code", code);
		 Delete(code);
	       }
	       {
		 String *decl = NewStringEmpty();
		 SwigType_add_function(decl, $parms);
		 Setattr($$, "decl", decl);
		 Delete(decl);
	       }
	       Setattr($$, "throws", $cpp_vend.throws);
	       Setattr($$, "throw", $cpp_vend.throwf);
	       Setattr($$, "noexcept", $cpp_vend.nexcept);
	       Setattr($$, "final", $cpp_vend.final);
	       if ($cpp_vend.val) {
		 if (Equal($cpp_vend.val, "0")) {
		   if (!Strstr($storage_class, "virtual"))
		     Swig_error(cparse_file, cparse_line, "Destructor %s uses a pure specifier but is not virtual.\n", Swig_name_decl($$));
		 } else if (!(Equal($cpp_vend.val, "delete") || Equal($cpp_vend.val, "default"))) {
		   Swig_error(cparse_file, cparse_line, "Destructor %s has an invalid pure specifier, only = 0 is allowed.\n", Swig_name_decl($$));
		 }
		 Setattr($$, "value", $cpp_vend.val);
	       }
	       /* TODO: check all storage decl-specifiers are valid */
	       if ($cpp_vend.qualifier)
		 Swig_error(cparse_file, cparse_line, "Destructor %s %s cannot have a qualifier.\n", Swig_name_decl($$), SwigType_str($cpp_vend.qualifier, 0));
	       add_symbols($$);
	       Delete($storage_class);
	      }
              ;


/* C++ type conversion operator */
cpp_conversion_operator : storage_class CONVERSIONOPERATOR type pointer LPAREN parms RPAREN cpp_vend {
                 $$ = new_node("cdecl");
                 Setattr($$,"type",$type);
		 Setattr($$,"name",$CONVERSIONOPERATOR);
		 Setattr($$,"storage",$storage_class);

		 SwigType_add_function($pointer,$parms);
		 if ($cpp_vend.qualifier) {
		   SwigType_push($pointer,$cpp_vend.qualifier);
		 }
		 if ($cpp_vend.val) {
		   Setattr($$,"value",$cpp_vend.val);
		 }
		 Setattr($$,"refqualifier",$cpp_vend.refqualifier);
		 Setattr($$,"decl",$pointer);
		 Setattr($$,"parms",$parms);
		 Setattr($$,"conversion_operator","1");
		 add_symbols($$);
		 Delete($CONVERSIONOPERATOR);
		 Delete($storage_class);
              }
               | storage_class CONVERSIONOPERATOR type AND LPAREN parms RPAREN cpp_vend {
		 SwigType *decl;
                 $$ = new_node("cdecl");
                 Setattr($$,"type",$type);
		 Setattr($$,"name",$CONVERSIONOPERATOR);
		 Setattr($$,"storage",$storage_class);
		 decl = NewStringEmpty();
		 SwigType_add_reference(decl);
		 SwigType_add_function(decl,$parms);
		 if ($cpp_vend.qualifier) {
		   SwigType_push(decl,$cpp_vend.qualifier);
		 }
		 if ($cpp_vend.val) {
		   Setattr($$,"value",$cpp_vend.val);
		 }
		 Setattr($$,"refqualifier",$cpp_vend.refqualifier);
		 Setattr($$,"decl",decl);
		 Setattr($$,"parms",$parms);
		 Setattr($$,"conversion_operator","1");
		 add_symbols($$);
		 Delete($CONVERSIONOPERATOR);
		 Delete($storage_class);
	       }
               | storage_class CONVERSIONOPERATOR type LAND LPAREN parms RPAREN cpp_vend {
		 SwigType *decl;
                 $$ = new_node("cdecl");
                 Setattr($$,"type",$type);
		 Setattr($$,"name",$CONVERSIONOPERATOR);
		 Setattr($$,"storage",$storage_class);
		 decl = NewStringEmpty();
		 SwigType_add_rvalue_reference(decl);
		 SwigType_add_function(decl,$parms);
		 if ($cpp_vend.qualifier) {
		   SwigType_push(decl,$cpp_vend.qualifier);
		 }
		 if ($cpp_vend.val) {
		   Setattr($$,"value",$cpp_vend.val);
		 }
		 Setattr($$,"refqualifier",$cpp_vend.refqualifier);
		 Setattr($$,"decl",decl);
		 Setattr($$,"parms",$parms);
		 Setattr($$,"conversion_operator","1");
		 add_symbols($$);
		 Delete($CONVERSIONOPERATOR);
		 Delete($storage_class);
	       }

               | storage_class CONVERSIONOPERATOR type pointer AND LPAREN parms RPAREN cpp_vend {
		 SwigType *decl;
                 $$ = new_node("cdecl");
                 Setattr($$,"type",$type);
		 Setattr($$,"name",$CONVERSIONOPERATOR);
		 Setattr($$,"storage",$storage_class);
		 decl = NewStringEmpty();
		 SwigType_add_pointer(decl);
		 SwigType_add_reference(decl);
		 SwigType_add_function(decl,$parms);
		 if ($cpp_vend.qualifier) {
		   SwigType_push(decl,$cpp_vend.qualifier);
		 }
		 if ($cpp_vend.val) {
		   Setattr($$,"value",$cpp_vend.val);
		 }
		 Setattr($$,"refqualifier",$cpp_vend.refqualifier);
		 Setattr($$,"decl",decl);
		 Setattr($$,"parms",$parms);
		 Setattr($$,"conversion_operator","1");
		 add_symbols($$);
		 Delete($CONVERSIONOPERATOR);
		 Delete($storage_class);
	       }

              | storage_class CONVERSIONOPERATOR type LPAREN parms RPAREN cpp_vend {
		String *t = NewStringEmpty();
		$$ = new_node("cdecl");
		Setattr($$,"type",$type);
		Setattr($$,"name",$CONVERSIONOPERATOR);
		 Setattr($$,"storage",$storage_class);
		SwigType_add_function(t,$parms);
		if ($cpp_vend.qualifier) {
		  SwigType_push(t,$cpp_vend.qualifier);
		}
		if ($cpp_vend.val) {
		  Setattr($$,"value",$cpp_vend.val);
		}
		Setattr($$,"refqualifier",$cpp_vend.refqualifier);
		Setattr($$,"decl",t);
		Setattr($$,"parms",$parms);
		Setattr($$,"conversion_operator","1");
		add_symbols($$);
		Delete($CONVERSIONOPERATOR);
		Delete($storage_class);
              }
              ;

/* isolated catch clause. */

cpp_catch_decl : CATCH LPAREN parms RPAREN LBRACE {
                 if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
                 $$ = 0;
               }
               ;

/* static_assert(bool, const char*); (C++11)
 * static_assert(bool); (C++17) */
cpp_static_assert : STATIC_ASSERT LPAREN {
                if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
                $$ = 0;
              }
              ;

/* public: */
cpp_protection_decl : PUBLIC COLON { 
                $$ = new_node("access");
		Setattr($$,"kind","public");
                cplus_mode = CPLUS_PUBLIC;
              }

/* private: */
              | PRIVATE COLON { 
                $$ = new_node("access");
                Setattr($$,"kind","private");
		cplus_mode = CPLUS_PRIVATE;
	      }

/* protected: */

              | PROTECTED COLON { 
		$$ = new_node("access");
		Setattr($$,"kind","protected");
		cplus_mode = CPLUS_PROTECTED;
	      }
              ;
/* These directives can be included inside a class definition */

cpp_swig_directive: pragma_directive

/* A constant (includes #defines) inside a class */
             | constant_directive

             | rename_directive
             | feature_directive
             | varargs_directive
             | insert_directive
             | typemap_directive
             | apply_directive
             | clear_directive
             | echo_directive
             ;

cpp_vend       : cpp_const SEMI { 
                     Clear(scanner_ccode);
                     $$ = $cpp_const;
               }
               | cpp_const EQUAL definetype SEMI { 
                     Clear(scanner_ccode);
                     $$ = $cpp_const;
                     $$.val = $definetype.val;
               }
               | cpp_const LBRACE { 
                     if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
                     $$ = $cpp_const;
               }
               ;


anonymous_bitfield :  storage_class anon_bitfield_type COLON expr SEMI { Delete($storage_class); };

/* Equals type_right without the ENUM keyword and cpptype (templates etc.): */
anon_bitfield_type : primitive_type
               | TYPE_BOOL
               | TYPE_VOID
               | TYPE_RAW

               | idcolon { $$ = $idcolon; }
               ;

/* ====================================================================== 
 *                       PRIMITIVES
 * ====================================================================== */
storage_class  : storage_class_list {
		 String *r = NewStringEmpty();

		 /* Check for invalid combinations. */
		 if (multiple_bits_set($storage_class_list & (SWIG_STORAGE_CLASS_EXTERN |
					     SWIG_STORAGE_CLASS_STATIC))) {
		   Swig_error(cparse_file, cparse_line, "Storage class can't be both 'static' and 'extern'");
		 }
		 if (multiple_bits_set($storage_class_list & (SWIG_STORAGE_CLASS_EXTERNC |
					     SWIG_STORAGE_CLASS_EXTERN |
					     SWIG_STORAGE_CLASS_EXTERNCPP))) {
		   Swig_error(cparse_file, cparse_line, "Declaration can only be one of 'extern', 'extern \"C\"' and 'extern \"C++\"'");
		 }

		 if ($storage_class_list & SWIG_STORAGE_CLASS_TYPEDEF) {
		   Append(r, "typedef ");
		 } else {
		   if ($storage_class_list & SWIG_STORAGE_CLASS_EXTERNC)
		     Append(r, "externc ");
		   if ($storage_class_list & (SWIG_STORAGE_CLASS_EXTERN|SWIG_STORAGE_CLASS_EXTERNCPP))
		     Append(r, "extern ");
		   if ($storage_class_list & SWIG_STORAGE_CLASS_STATIC)
		     Append(r, "static ");
		 }
		 if ($storage_class_list & SWIG_STORAGE_CLASS_VIRTUAL)
		   Append(r, "virtual ");
		 if ($storage_class_list & SWIG_STORAGE_CLASS_FRIEND)
		   Append(r, "friend ");
		 if ($storage_class_list & SWIG_STORAGE_CLASS_EXPLICIT)
		   Append(r, "explicit ");
		 if ($storage_class_list & SWIG_STORAGE_CLASS_CONSTEXPR)
		   Append(r, "constexpr ");
		 if ($storage_class_list & SWIG_STORAGE_CLASS_THREAD_LOCAL)
		   Append(r, "thread_local ");
		 if (Len(r) == 0) {
		   Delete(r);
		   $$ = 0;
		 } else {
		   Chop(r);
		   $$ = r;
		 }
	       }
	       | %empty { $$ = 0; }
	       ;

storage_class_list: storage_class_raw
	       | storage_class_list[in] storage_class_raw {
		  if ($in & $storage_class_raw) {
		    Swig_error(cparse_file, cparse_line, "Repeated storage class or type specifier '%s'\n", storage_class_string($storage_class_raw));
		  }
		  $$ = $in | $storage_class_raw;
	       }
	       ;

storage_class_raw  : EXTERN { $$ = SWIG_STORAGE_CLASS_EXTERN; }
	       | EXTERN string {
		   if (Strcmp($string,"C") == 0) {
		     $$ = SWIG_STORAGE_CLASS_EXTERNC;
		   } else if (Strcmp($string,"C++") == 0) {
		     $$ = SWIG_STORAGE_CLASS_EXTERNCPP;
		   } else {
		     Swig_warning(WARN_PARSE_UNDEFINED_EXTERN,cparse_file, cparse_line,"Unrecognized extern type \"%s\".\n", $string);
		     $$ = 0;
		   }
	       }
	       | STATIC { $$ = SWIG_STORAGE_CLASS_STATIC; }
	       | TYPEDEF { $$ = SWIG_STORAGE_CLASS_TYPEDEF; }
	       | VIRTUAL { $$ = SWIG_STORAGE_CLASS_VIRTUAL; }
	       | FRIEND { $$ = SWIG_STORAGE_CLASS_FRIEND; }
	       | EXPLICIT { $$ = SWIG_STORAGE_CLASS_EXPLICIT; }
	       | CONSTEXPR { $$ = SWIG_STORAGE_CLASS_CONSTEXPR; }
	       | THREAD_LOCAL { $$ = SWIG_STORAGE_CLASS_THREAD_LOCAL; }
	       ;

/* ------------------------------------------------------------------------------
   Function parameter lists
   ------------------------------------------------------------------------------ */

parms          : rawparms {
                 Parm *p;
		 $$ = $rawparms;
		 p = $rawparms;
                 while (p) {
		   Replace(Getattr(p,"type"),"typename ", "", DOH_REPLACE_ANY);
		   p = nextSibling(p);
                 }
               }
    	       ;

/* rawparms constructs parameter lists and deal with quirks of doxygen post strings (after the parameter's comma */
rawparms	: parm { $$ = $parm; }
		| parm DOXYGENPOSTSTRING {
		  set_comment($parm, $DOXYGENPOSTSTRING);
		  $$ = $parm;
		}
		| parm DOXYGENSTRING {
		  /* Misplaced doxygen string, attach it to previous parameter, like Doxygen does */
		  set_comment($parm, $DOXYGENSTRING);
		  $$ = $parm;
		}
		| parm COMMA parms {
		  if ($parms) {
		    set_nextSibling($parm, $parms);
		  }
		  $$ = $parm;
		}
		| parm DOXYGENPOSTSTRING COMMA parms {
		  if ($parms) {
		    set_nextSibling($parm, $parms);
		  }
		  set_comment($parm, $DOXYGENPOSTSTRING);
		  $$ = $parm;
		}
		| parm COMMA DOXYGENPOSTSTRING parms {
		  if ($parms) {
		    set_nextSibling($parm, $parms);
		  }
		  set_comment($parm, $DOXYGENPOSTSTRING);
		  $$ = $parm;
		}
		| %empty {
		  $$ = 0;
		}
		;

parm_no_dox	: rawtype parameter_declarator {
                   SwigType_push($rawtype,$parameter_declarator.type);
		   $$ = NewParmWithoutFileLineInfo($rawtype,$parameter_declarator.id);
		   Setfile($$,cparse_file);
		   Setline($$,cparse_line);
		   if ($parameter_declarator.defarg)
		     Setattr($$, "value", $parameter_declarator.defarg);
		   if ($parameter_declarator.stringdefarg)
		     Setattr($$, "stringval", $parameter_declarator.stringdefarg);
		   if ($parameter_declarator.numdefarg)
		     Setattr($$, "numval", $parameter_declarator.numdefarg);
		}
                | ELLIPSIS {
		  SwigType *t = NewString("v(...)");
		  $$ = NewParmWithoutFileLineInfo(t, 0);
		  Setfile($$,cparse_file);
		  Setline($$,cparse_line);
		}
		;

parm		: parm_no_dox
		| DOXYGENSTRING parm_no_dox {
		  $$ = $parm_no_dox;
		  set_comment($parm_no_dox, $DOXYGENSTRING);
		}
		;

valparms : valparms_builder {
		 $$ = $valparms_builder.parms;
                 for (Parm *p = $$; p; p = nextSibling(p)) {
		   if (Getattr(p,"type")) {
		     Replace(Getattr(p,"type"),"typename ", "", DOH_REPLACE_ANY);
		   }
                 }
	       }
	       | %empty {
		 $$ = 0;
	       }
	       ;

valparms_builder : valparm {
		    $$.parms = $$.last = $valparm;
		  }
		  | valparms_builder[in] COMMA valparm {
		    // Build a linked list in the order specified, but avoiding
		    // a right recursion rule because "Right recursion uses up
		    // space on the Bison stack in proportion to the number of
		    // elements in the sequence".
		    set_nextSibling($in.last, $valparm);
		    $$.parms = $in.parms;
		    $$.last = $valparm;
		  }
		  ;

valparm        : parm {
		  $$ = $parm;
		  {
		    /* We need to make a possible adjustment for integer parameters. */
		    SwigType *type;
		    Node     *n = 0;

		    while (!n) {
		      type = Getattr($parm,"type");
		      n = Swig_symbol_clookup(type,0);     /* See if we can find a node that matches the typename */
		      if ((n) && (Strcmp(nodeType(n),"cdecl") == 0)) {
			SwigType *decl = Getattr(n,"decl");
			if (!SwigType_isfunction(decl)) {
			  String *value = Getattr(n,"value");
			  if (value) {
			    String *v = Copy(value);
			    Setattr($parm,"type",v);
			    Delete(v);
			    n = 0;
			  }
			}
		      } else {
			break;
		      }
		    }
		  }

               }
               | valexpr {
                  $$ = NewParmWithoutFileLineInfo(0,0);
                  Setfile($$,cparse_file);
		  Setline($$,cparse_line);
		  Setattr($$,"value",$valexpr.val);
		  if ($valexpr.stringval) Setattr($$, "stringval", $valexpr.stringval);
		  if ($valexpr.numval) Setattr($$, "numval", $valexpr.numval);
               }
               ;

def_args       : EQUAL definetype { 
                 $$ = $definetype;
               }
	       | EQUAL definetype LBRACKET {
		 if (skip_balanced('[', ']') < 0) Exit(EXIT_FAILURE);
		 $$ = default_dtype;
		 $$.type = T_UNKNOWN;
		 $$.val = $definetype.val;
		 Append($$.val, scanner_ccode);
		 Clear(scanner_ccode);
               }
               | EQUAL LBRACE {
		 if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		 $$ = default_dtype;
		 $$.val = NewString(scanner_ccode);
		 $$.type = T_UNKNOWN;
	       }
               | %empty {
		 $$ = default_dtype;
                 $$.type = T_UNKNOWN;
               }
               ;

parameter_declarator : declarator def_args {
                 $$ = $declarator;
		 $$.defarg = $def_args.val;
		 $$.stringdefarg = $def_args.stringval;
		 $$.numdefarg = $def_args.numval;
            }
            | abstract_declarator def_args {
	      $$ = $abstract_declarator;
	      $$.defarg = $def_args.val;
	      $$.stringdefarg = $def_args.stringval;
	      $$.numdefarg = $def_args.numval;
            }
            | def_args {
	      $$ = default_decl;
	      $$.defarg = $def_args.val;
	      $$.stringdefarg = $def_args.stringval;
	      $$.numdefarg = $def_args.numval;
            }
	    /* Member function pointers with qualifiers. eg.
	      int f(short (Funcs::*parm)(bool) const); */
	    | direct_declarator LPAREN parms RPAREN qualifiers_exception_specification {
	      SwigType *t;
	      $$ = $direct_declarator;
	      t = NewStringEmpty();
	      SwigType_add_function(t,$parms);
	      if ($qualifiers_exception_specification.qualifier)
		SwigType_push(t, $qualifiers_exception_specification.qualifier);
	      if ($qualifiers_exception_specification.nexcept)
		SwigType_add_qualifier(t, "noexcept");
	      if (!$$.have_parms) {
		$$.parms = $parms;
		$$.have_parms = 1;
	      }
	      if (!$$.type) {
		$$.type = t;
	      } else {
		SwigType_push(t, $$.type);
		Delete($$.type);
		$$.type = t;
	      }
	    }
            ;

plain_declarator : declarator {
                 $$ = $declarator;
		 if (SwigType_isfunction($declarator.type)) {
		   Delete(SwigType_pop_function($declarator.type));
		 } else if (SwigType_isarray($declarator.type)) {
		   SwigType *ta = SwigType_pop_arrays($declarator.type);
		   if (SwigType_isfunction($declarator.type)) {
		     Delete(SwigType_pop_function($declarator.type));
		   } else {
		     $$.parms = 0;
		   }
		   SwigType_push($declarator.type,ta);
		   Delete(ta);
		 } else {
		   $$.parms = 0;
		 }
            }
            | abstract_declarator {
              $$ = $abstract_declarator;
	      if (SwigType_isfunction($abstract_declarator.type)) {
		Delete(SwigType_pop_function($abstract_declarator.type));
	      } else if (SwigType_isarray($abstract_declarator.type)) {
		SwigType *ta = SwigType_pop_arrays($abstract_declarator.type);
		if (SwigType_isfunction($abstract_declarator.type)) {
		  Delete(SwigType_pop_function($abstract_declarator.type));
		} else {
		  $$.parms = 0;
		}
		SwigType_push($abstract_declarator.type,ta);
		Delete(ta);
	      } else {
		$$.parms = 0;
	      }
            }
	    /* Member function pointers with qualifiers. eg.
	      int f(short (Funcs::*parm)(bool) const) */
	    | direct_declarator LPAREN parms RPAREN cv_ref_qualifier {
	      SwigType *t;
	      $$ = $direct_declarator;
	      t = NewStringEmpty();
	      SwigType_add_function(t, $parms);
	      if ($cv_ref_qualifier.qualifier)
	        SwigType_push(t, $cv_ref_qualifier.qualifier);
	      if (!$$.have_parms) {
		$$.parms = $parms;
		$$.have_parms = 1;
	      }
	      if (!$$.type) {
		$$.type = t;
	      } else {
		SwigType_push(t, $$.type);
		Delete($$.type);
		$$.type = t;
	      }
	    }
            | %empty {
	      $$ = default_decl;
	    }
            ;

declarator :  pointer notso_direct_declarator {
              $$ = $notso_direct_declarator;
	      if ($$.type) {
		SwigType_push($pointer,$$.type);
		Delete($$.type);
	      }
	      $$.type = $pointer;
           }
           | pointer AND notso_direct_declarator {
              $$ = $notso_direct_declarator;
	      SwigType_add_reference($pointer);
              if ($$.type) {
		SwigType_push($pointer,$$.type);
		Delete($$.type);
	      }
	      $$.type = $pointer;
           }
           | pointer LAND notso_direct_declarator {
              $$ = $notso_direct_declarator;
	      SwigType_add_rvalue_reference($pointer);
              if ($$.type) {
		SwigType_push($pointer,$$.type);
		Delete($$.type);
	      }
	      $$.type = $pointer;
           }
           | direct_declarator {
              $$ = $direct_declarator;
	      if (!$$.type) $$.type = NewStringEmpty();
           }
           | AND notso_direct_declarator {
	     $$ = $notso_direct_declarator;
	     $$.type = NewStringEmpty();
	     SwigType_add_reference($$.type);
	     if ($notso_direct_declarator.type) {
	       SwigType_push($$.type,$notso_direct_declarator.type);
	       Delete($notso_direct_declarator.type);
	     }
           }
           | LAND notso_direct_declarator {
	     /* Introduced in C++11, move operator && */
             /* Adds one S/R conflict */
	     $$ = $notso_direct_declarator;
	     $$.type = NewStringEmpty();
	     SwigType_add_rvalue_reference($$.type);
	     if ($notso_direct_declarator.type) {
	       SwigType_push($$.type,$notso_direct_declarator.type);
	       Delete($notso_direct_declarator.type);
	     }
           }
           | idcolon DSTAR notso_direct_declarator { 
	     SwigType *t = NewStringEmpty();

	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer(t,$idcolon);
	     if ($$.type) {
	       SwigType_push(t,$$.type);
	       Delete($$.type);
	     }
	     $$.type = t;
	     } 
           | pointer idcolon DSTAR notso_direct_declarator { 
	     SwigType *t = NewStringEmpty();
	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer(t,$idcolon);
	     SwigType_push($pointer,t);
	     if ($$.type) {
	       SwigType_push($pointer,$$.type);
	       Delete($$.type);
	     }
	     $$.type = $pointer;
	     Delete(t);
	   }
           | pointer idcolon DSTAR AND notso_direct_declarator { 
	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer($pointer,$idcolon);
	     SwigType_add_reference($pointer);
	     if ($$.type) {
	       SwigType_push($pointer,$$.type);
	       Delete($$.type);
	     }
	     $$.type = $pointer;
	   }
           | idcolon DSTAR AND notso_direct_declarator { 
	     SwigType *t = NewStringEmpty();
	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer(t,$idcolon);
	     SwigType_add_reference(t);
	     if ($$.type) {
	       SwigType_push(t,$$.type);
	       Delete($$.type);
	     } 
	     $$.type = t;
	   }
           
           /* Variadic versions eg. MyClasses&... myIds */
           
           |  pointer ELLIPSIS notso_direct_declarator {
              $$ = $notso_direct_declarator;
	      if ($$.type) {
		SwigType_push($pointer,$$.type);
		Delete($$.type);
	      }
	      $$.type = $pointer;
	      SwigType_add_variadic($$.type);
           }
           | pointer AND ELLIPSIS notso_direct_declarator {
              $$ = $notso_direct_declarator;
	      SwigType_add_reference($pointer);
              if ($$.type) {
		SwigType_push($pointer,$$.type);
		Delete($$.type);
	      }
	      $$.type = $pointer;
	      SwigType_add_variadic($$.type);
           }
           | pointer LAND ELLIPSIS notso_direct_declarator {
              $$ = $notso_direct_declarator;
	      SwigType_add_rvalue_reference($pointer);
              if ($$.type) {
		SwigType_push($pointer,$$.type);
		Delete($$.type);
	      }
	      $$.type = $pointer;
	      SwigType_add_variadic($$.type);
           }
           | AND ELLIPSIS notso_direct_declarator {
	     $$ = $notso_direct_declarator;
	     $$.type = NewStringEmpty();
	     SwigType_add_reference($$.type);
	     SwigType_add_variadic($$.type);
	     if ($notso_direct_declarator.type) {
	       SwigType_push($$.type,$notso_direct_declarator.type);
	       Delete($notso_direct_declarator.type);
	     }
           }
           | LAND ELLIPSIS notso_direct_declarator {
	     /* Introduced in C++11, move operator && */
             /* Adds one S/R conflict */
	     $$ = $notso_direct_declarator;
	     $$.type = NewStringEmpty();
	     SwigType_add_rvalue_reference($$.type);
	     SwigType_add_variadic($$.type);
	     if ($notso_direct_declarator.type) {
	       SwigType_push($$.type,$notso_direct_declarator.type);
	       Delete($notso_direct_declarator.type);
	     }
           }
           | idcolon DSTAR ELLIPSIS notso_direct_declarator {
	     SwigType *t = NewStringEmpty();

	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer(t,$idcolon);
	     SwigType_add_variadic(t);
	     if ($$.type) {
	       SwigType_push(t,$$.type);
	       Delete($$.type);
	     }
	     $$.type = t;
	     } 
           | pointer idcolon DSTAR ELLIPSIS notso_direct_declarator {
	     SwigType *t = NewStringEmpty();
	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer(t,$idcolon);
	     SwigType_add_variadic(t);
	     SwigType_push($pointer,t);
	     if ($$.type) {
	       SwigType_push($pointer,$$.type);
	       Delete($$.type);
	     }
	     $$.type = $pointer;
	     Delete(t);
	   }
           | pointer idcolon DSTAR AND ELLIPSIS notso_direct_declarator {
	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer($pointer,$idcolon);
	     SwigType_add_reference($pointer);
	     SwigType_add_variadic($pointer);
	     if ($$.type) {
	       SwigType_push($pointer,$$.type);
	       Delete($$.type);
	     }
	     $$.type = $pointer;
	   }
           | pointer idcolon DSTAR LAND ELLIPSIS notso_direct_declarator {
	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer($pointer,$idcolon);
	     SwigType_add_rvalue_reference($pointer);
	     SwigType_add_variadic($pointer);
	     if ($$.type) {
	       SwigType_push($pointer,$$.type);
	       Delete($$.type);
	     }
	     $$.type = $pointer;
	   }
           | idcolon DSTAR AND ELLIPSIS notso_direct_declarator {
	     SwigType *t = NewStringEmpty();
	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer(t,$idcolon);
	     SwigType_add_reference(t);
	     SwigType_add_variadic(t);
	     if ($$.type) {
	       SwigType_push(t,$$.type);
	       Delete($$.type);
	     } 
	     $$.type = t;
	   }
           | idcolon DSTAR LAND ELLIPSIS notso_direct_declarator {
	     SwigType *t = NewStringEmpty();
	     $$ = $notso_direct_declarator;
	     SwigType_add_memberpointer(t,$idcolon);
	     SwigType_add_rvalue_reference(t);
	     SwigType_add_variadic(t);
	     if ($$.type) {
	       SwigType_push(t,$$.type);
	       Delete($$.type);
	     } 
	     $$.type = t;
	   }
           ;

notso_direct_declarator : idcolon {
                /* Note: This is non-standard C.  Template declarator is allowed to follow an identifier */
		 $$ = default_decl;
                 $$.id = Char($idcolon);
                  }
                  | NOT idcolon {
		  $$ = default_decl;
                  $$.id = Char(NewStringf("~%s",$idcolon));
                  }

/* This generates a shift-reduce conflict with constructors */
                 | LPAREN idcolon RPAREN {
		  $$ = default_decl;
                  $$.id = Char($idcolon);
                  }

/*
                  | LPAREN AND idcolon RPAREN {
		     $$ = default_decl;
                     $$.id = Char($idcolon);
                  }
*/
/* Technically, this should be LPAREN declarator RPAREN, but we get reduce/reduce conflicts */
                  | LPAREN pointer notso_direct_declarator[in] RPAREN {
		    $$ = $in;
		    if ($$.type) {
		      SwigType_push($pointer,$$.type);
		      Delete($$.type);
		    }
		    $$.type = $pointer;
                  }
                  | LPAREN idcolon DSTAR notso_direct_declarator[in] RPAREN {
		    SwigType *t;
		    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t,$idcolon);
		    if ($$.type) {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
		    }
                  | notso_direct_declarator[in] LBRACKET RBRACKET { 
		    SwigType *t;
		    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ($$.type) {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
                  }
                  | notso_direct_declarator[in] LBRACKET expr RBRACKET { 
		    SwigType *t;
		    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_array(t,$expr.val);
		    if ($$.type) {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
                  }
                  | notso_direct_declarator[in] LPAREN parms RPAREN {
		    SwigType *t;
                    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_function(t,$parms);
		    if (!$$.have_parms) {
		      $$.parms = $parms;
		      $$.have_parms = 1;
		    }
		    if (!$$.type) {
		      $$.type = t;
		    } else {
		      SwigType_push(t, $$.type);
		      Delete($$.type);
		      $$.type = t;
		    }
		  }
                  ;

direct_declarator : idcolon {
                /* Note: This is non-standard C.  Template declarator is allowed to follow an identifier */
		 $$ = default_decl;
                 $$.id = Char($idcolon);
                  }
                  
                  | NOT idcolon {
		  $$ = default_decl;
                  $$.id = Char(NewStringf("~%s",$idcolon));
                  }

/* This generate a shift-reduce conflict with constructors */
/*
                  | LPAREN idcolon RPAREN {
		  $$ = default_decl;
                  $$.id = Char($idcolon);
                  }
*/
/* Technically, this should be LPAREN declarator RPAREN, but we get reduce/reduce conflicts */
                  | LPAREN pointer direct_declarator[in] RPAREN {
		    $$ = $in;
		    if ($$.type) {
		      SwigType_push($pointer,$$.type);
		      Delete($$.type);
		    }
		    $$.type = $pointer;
                  }
                  | LPAREN AND direct_declarator[in] RPAREN {
                    $$ = $in;
		    if (!$$.type) {
		      $$.type = NewStringEmpty();
		    }
		    SwigType_add_reference($$.type);
                  }
                  | LPAREN LAND direct_declarator[in] RPAREN {
                    $$ = $in;
		    if (!$$.type) {
		      $$.type = NewStringEmpty();
		    }
		    SwigType_add_rvalue_reference($$.type);
                  }
                  | LPAREN idcolon DSTAR declarator RPAREN {
		    SwigType *t;
		    $$ = $declarator;
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t,$idcolon);
		    if ($$.type) {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
		  }
                  | LPAREN idcolon DSTAR type_qualifier declarator RPAREN {
		    SwigType *t;
		    $$ = $declarator;
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t, $idcolon);
		    SwigType_push(t, $type_qualifier);
		    if ($$.type) {
		      SwigType_push(t, $$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
		  }
                  | LPAREN idcolon DSTAR abstract_declarator RPAREN {
		    SwigType *t;
		    $$ = $abstract_declarator;
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t, $idcolon);
		    if ($$.type) {
		      SwigType_push(t, $$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
		  }
                  | LPAREN idcolon DSTAR type_qualifier abstract_declarator RPAREN {
		    SwigType *t;
		    $$ = $abstract_declarator;
		    t = NewStringEmpty();
		    SwigType_add_memberpointer(t, $idcolon);
		    SwigType_push(t, $type_qualifier);
		    if ($$.type) {
		      SwigType_push(t, $$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
		  }
                  | direct_declarator[in] LBRACKET RBRACKET { 
		    SwigType *t;
		    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ($$.type) {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
                  }
                  | direct_declarator[in] LBRACKET expr RBRACKET { 
		    SwigType *t;
		    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_array(t,$expr.val);
		    if ($$.type) {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
                  }
                  | direct_declarator[in] LPAREN parms RPAREN {
		    SwigType *t;
                    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_function(t,$parms);
		    if (!$$.have_parms) {
		      $$.parms = $parms;
		      $$.have_parms = 1;
		    }
		    if (!$$.type) {
		      $$.type = t;
		    } else {
		      SwigType_push(t, $$.type);
		      Delete($$.type);
		      $$.type = t;
		    }
		  }
                 /* User-defined string literals. eg.
                    int operator"" _mySuffix(const char* val, int length) {...} */
		 /* This produces one S/R conflict. */
                 | OPERATOR ID LPAREN parms RPAREN {
		    $$ = default_decl;
		    SwigType *t;
                    Append($OPERATOR, " "); /* intervening space is mandatory */
		    Append($OPERATOR, $ID);
		    $$.id = Char($OPERATOR);
		    t = NewStringEmpty();
		    SwigType_add_function(t,$parms);
		    $$.parms = $parms;
		    $$.have_parms = 1;
		    $$.type = t;
		  }
                  ;

abstract_declarator : pointer variadic_opt {
		    $$ = default_decl;
		    $$.type = $pointer;
		    if ($variadic_opt) SwigType_add_variadic($$.type);
                  }
                  | pointer direct_abstract_declarator { 
                     $$ = $direct_abstract_declarator;
                     SwigType_push($pointer,$direct_abstract_declarator.type);
		     $$.type = $pointer;
		     Delete($direct_abstract_declarator.type);
                  }
                  | pointer AND variadic_opt {
		    $$ = default_decl;
		    $$.type = $pointer;
		    SwigType_add_reference($$.type);
		    if ($variadic_opt) SwigType_add_variadic($$.type);
		  }
                  | pointer LAND variadic_opt {
		    $$ = default_decl;
		    $$.type = $pointer;
		    SwigType_add_rvalue_reference($$.type);
		    if ($variadic_opt) SwigType_add_variadic($$.type);
		  }
                  | pointer AND direct_abstract_declarator {
		    $$ = $direct_abstract_declarator;
		    SwigType_add_reference($pointer);
		    if ($$.type) {
		      SwigType_push($pointer,$$.type);
		      Delete($$.type);
		    }
		    $$.type = $pointer;
                  }
                  | pointer LAND direct_abstract_declarator {
		    $$ = $direct_abstract_declarator;
		    SwigType_add_rvalue_reference($pointer);
		    if ($$.type) {
		      SwigType_push($pointer,$$.type);
		      Delete($$.type);
		    }
		    $$.type = $pointer;
                  }
                  | direct_abstract_declarator
                  | AND direct_abstract_declarator {
		    $$ = $direct_abstract_declarator;
		    $$.type = NewStringEmpty();
		    SwigType_add_reference($$.type);
		    if ($direct_abstract_declarator.type) {
		      SwigType_push($$.type,$direct_abstract_declarator.type);
		      Delete($direct_abstract_declarator.type);
		    }
                  }
                  | LAND direct_abstract_declarator {
		    $$ = $direct_abstract_declarator;
		    $$.type = NewStringEmpty();
		    SwigType_add_rvalue_reference($$.type);
		    if ($direct_abstract_declarator.type) {
		      SwigType_push($$.type,$direct_abstract_declarator.type);
		      Delete($direct_abstract_declarator.type);
		    }
                  }
                  | AND variadic_opt {
		    $$ = default_decl;
                    $$.type = NewStringEmpty();
		    SwigType_add_reference($$.type);
		    if ($variadic_opt) SwigType_add_variadic($$.type);
                  }
                  | LAND variadic_opt {
		    $$ = default_decl;
                    $$.type = NewStringEmpty();
		    SwigType_add_rvalue_reference($$.type);
		    if ($variadic_opt) SwigType_add_variadic($$.type);
                  }
                  | idcolon DSTAR { 
		    $$ = default_decl;
		    $$.type = NewStringEmpty();
                    SwigType_add_memberpointer($$.type,$idcolon);
      	          }
                  | idcolon DSTAR type_qualifier {
		    $$ = default_decl;
		    $$.type = NewStringEmpty();
		    SwigType_add_memberpointer($$.type, $idcolon);
		    SwigType_push($$.type, $type_qualifier);
		  }
                  | pointer idcolon DSTAR { 
		    $$ = default_decl;
		    SwigType *t = NewStringEmpty();
                    $$.type = $pointer;
		    SwigType_add_memberpointer(t,$idcolon);
		    SwigType_push($$.type,t);
		    Delete(t);
                  }
                  | pointer idcolon DSTAR direct_abstract_declarator { 
		    $$ = $direct_abstract_declarator;
		    SwigType_add_memberpointer($pointer,$idcolon);
		    if ($$.type) {
		      SwigType_push($pointer,$$.type);
		      Delete($$.type);
		    }
		    $$.type = $pointer;
                  }
                  ;

direct_abstract_declarator : direct_abstract_declarator[in] LBRACKET RBRACKET { 
		    SwigType *t;
		    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_array(t,"");
		    if ($$.type) {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
                  }
                  | direct_abstract_declarator[in] LBRACKET expr RBRACKET { 
		    SwigType *t;
		    $$ = $in;
		    t = NewStringEmpty();
		    SwigType_add_array(t,$expr.val);
		    if ($$.type) {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		    }
		    $$.type = t;
                  }
                  | LBRACKET RBRACKET { 
		    $$ = default_decl;
		    $$.type = NewStringEmpty();
		    SwigType_add_array($$.type,"");
                  }
                  | LBRACKET expr RBRACKET { 
		    $$ = default_decl;
		    $$.type = NewStringEmpty();
		    SwigType_add_array($$.type,$expr.val);
		  }
                  | LPAREN abstract_declarator RPAREN {
                    $$ = $abstract_declarator;
		  }
                  | direct_abstract_declarator[in] LPAREN parms RPAREN {
		    SwigType *t;
                    $$ = $in;
		    t = NewStringEmpty();
                    SwigType_add_function(t,$parms);
		    if (!$$.type) {
		      $$.type = t;
		    } else {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		      $$.type = t;
		    }
		    if (!$$.have_parms) {
		      $$.parms = $parms;
		      $$.have_parms = 1;
		    }
		  }
                  | direct_abstract_declarator[in] LPAREN parms RPAREN cv_ref_qualifier {
		    SwigType *t;
                    $$ = $in;
		    t = NewStringEmpty();
                    SwigType_add_function(t,$parms);
		    SwigType_push(t, $cv_ref_qualifier.qualifier);
		    if (!$$.type) {
		      $$.type = t;
		    } else {
		      SwigType_push(t,$$.type);
		      Delete($$.type);
		      $$.type = t;
		    }
		    if (!$$.have_parms) {
		      $$.parms = $parms;
		      $$.have_parms = 1;
		    }
		  }
                  | LPAREN parms RPAREN {
		    $$ = default_decl;
                    $$.type = NewStringEmpty();
                    SwigType_add_function($$.type,$parms);
		    $$.parms = $parms;
		    $$.have_parms = 1;
                  }
                  ;


pointer    : pointer[in] STAR type_qualifier {
	     $$ = $in;
             SwigType_add_pointer($$);
	     SwigType_push($$,$type_qualifier);
           }
	   | pointer[in] STAR {
	     $$ = $in;
	     SwigType_add_pointer($$);
	   }
	   | STAR type_qualifier {
	     $$ = NewStringEmpty();
	     SwigType_add_pointer($$);
	     SwigType_push($$,$type_qualifier);
           }
           | STAR {
	     $$ = NewStringEmpty();
	     SwigType_add_pointer($$);
           }
           ;

/* cv-qualifier plus C++11 ref-qualifier for non-static member functions */
cv_ref_qualifier : type_qualifier {
		  $$.qualifier = $type_qualifier;
	       }
	       | type_qualifier ref_qualifier {
		  $$.qualifier = $type_qualifier;
		  $$.refqualifier = $ref_qualifier;
		  SwigType_push($$.qualifier, $ref_qualifier);
	       }
	       | ref_qualifier {
		  $$.qualifier = NewStringEmpty();
		  $$.refqualifier = $ref_qualifier;
		  SwigType_push($$.qualifier, $ref_qualifier);
	       }
	       ;

ref_qualifier : AND {
	          $$ = NewStringEmpty();
	          SwigType_add_reference($$);
	       }
	       | LAND {
	          $$ = NewStringEmpty();
	          SwigType_add_rvalue_reference($$);
	       }
	       ;

type_qualifier : type_qualifier_raw {
	          $$ = NewStringEmpty();
	          if ($type_qualifier_raw) SwigType_add_qualifier($$,$type_qualifier_raw);
               }
               | type_qualifier[in] type_qualifier_raw {
		  $$ = $in;
	          if ($type_qualifier_raw) SwigType_add_qualifier($$,$type_qualifier_raw);
               }
               ;

type_qualifier_raw :  CONST_QUAL { $$ = "const"; }
                   |  VOLATILE { $$ = "volatile"; }
                   |  REGISTER { $$ = 0; }
                   ;

/* Data type must be a built in type or an identifier for user-defined types
   This type can be preceded by a modifier. */

type            : rawtype %expect 4 {
                   $$ = $rawtype;
                   Replace($$,"typename ","", DOH_REPLACE_ANY);
                }
                ;

rawtype        : type_qualifier type_right {
                   $$ = $type_right;
	           SwigType_push($$,$type_qualifier);
               }
	       | type_right
               | type_right type_qualifier {
		  $$ = $type_right;
	          SwigType_push($$,$type_qualifier);
	       }
               | type_qualifier[type_qualifier1] type_right type_qualifier[type_qualifier2] {
		  $$ = $type_right;
	          SwigType_push($$,$type_qualifier2);
	          SwigType_push($$,$type_qualifier1);
	       }
	       | rawtype[in] ELLIPSIS {
		  $$ = $in;
		  SwigType_add_variadic($$);
	       }
               ;

type_right     : primitive_type
               | TYPE_BOOL
               | TYPE_VOID
               | c_enum_key idcolon { $$ = NewStringf("enum %s", $idcolon); }
               | TYPE_RAW

               | idcolon %expect 1 {
		  $$ = $idcolon;
               }
               | cpptype idcolon %expect 1 {
		 $$ = NewStringf("%s %s", $cpptype, $idcolon);
               }
               | decltype
               ;

decltype       : DECLTYPE LPAREN <str>{
		 $$ = get_raw_text_balanced('(', ')');
	       }[expr] decltypeexpr {
		 String *expr = $expr;
		 if ($decltypeexpr) {
		   $$ = $decltypeexpr;
		 } else {
		   $$ = NewStringf("decltype%s", expr);
		   /* expr includes parentheses but don't include them in the warning message. */
		   Delitem(expr, 0);
		   Delitem(expr, DOH_END);
		   Swig_warning(WARN_CPP11_DECLTYPE, cparse_file, cparse_line, "Unable to deduce decltype for '%s'.\n", expr);
		 }
		 Delete(expr);
	       }
	       ;

decltypeexpr   : expr RPAREN {
		 $$ = deduce_type(&$expr);
	       }
	       | error RPAREN {
		 /* Avoid a parse error if we can't parse the expression
		  * decltype() is applied to.
		  *
		  * Set $$ to 0 here to trigger the decltype rule above to
		  * issue a warning.
		  */
		 $$ = 0;
		 if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
		 Clear(scanner_ccode);
	       }
	       ;

primitive_type : primitive_type_list {
		 String *type = $primitive_type_list.type;
		 if (!type) type = NewString("int");
		 if ($primitive_type_list.us) {
		   $$ = NewStringf("%s %s", $primitive_type_list.us, type);
		   Delete($primitive_type_list.us);
                   Delete(type);
		 } else {
                   $$ = type;
		 }
		 if (Cmp($$,"signed int") == 0) {
		   Delete($$);
		   $$ = NewString("int");
                 } else if (Cmp($$,"signed long") == 0) {
		   Delete($$);
                   $$ = NewString("long");
                 } else if (Cmp($$,"signed short") == 0) {
		   Delete($$);
		   $$ = NewString("short");
		 } else if (Cmp($$,"signed long long") == 0) {
		   Delete($$);
		   $$ = NewString("long long");
		 }
               }
               ;

primitive_type_list : type_specifier
               | type_specifier primitive_type_list[in] {
                    if ($type_specifier.us && $in.us) {
		      Swig_error(cparse_file, cparse_line, "Extra %s specifier.\n", $in.us);
		    }
                    $$ = $in;
                    if ($type_specifier.us) $$.us = $type_specifier.us;
		    if ($type_specifier.type) {
		      if (!$in.type) $$.type = $type_specifier.type;
		      else {
			int err = 0;
			if ((Cmp($type_specifier.type,"long") == 0)) {
			  if ((Cmp($in.type,"long") == 0) || (Strncmp($in.type,"double",6) == 0)) {
			    $$.type = NewStringf("long %s", $in.type);
			  } else if (Cmp($in.type,"int") == 0) {
			    $$.type = $type_specifier.type;
			  } else {
			    err = 1;
			  }
			} else if ((Cmp($type_specifier.type,"short")) == 0) {
			  if (Cmp($in.type,"int") == 0) {
			    $$.type = $type_specifier.type;
			  } else {
			    err = 1;
			  }
			} else if (Cmp($type_specifier.type,"int") == 0) {
			  $$.type = $in.type;
			} else if (Cmp($type_specifier.type,"double") == 0) {
			  if (Cmp($in.type,"long") == 0) {
			    $$.type = NewString("long double");
			  } else if (Cmp($in.type,"_Complex") == 0) {
			    $$.type = NewString("double _Complex");
			  } else {
			    err = 1;
			  }
			} else if (Cmp($type_specifier.type,"float") == 0) {
			  if (Cmp($in.type,"_Complex") == 0) {
			    $$.type = NewString("float _Complex");
			  } else {
			    err = 1;
			  }
			} else if (Cmp($type_specifier.type,"_Complex") == 0) {
			  $$.type = NewStringf("%s _Complex", $in.type);
			} else {
			  err = 1;
			}
			if (err) {
			  Swig_error(cparse_file, cparse_line, "Extra %s specifier.\n", $type_specifier.type);
			}
		      }
		    }
               }
               ; 


type_specifier : TYPE_INT { 
		    $$.type = NewString("int");
                    $$.us = 0;
               }
               | TYPE_SHORT { 
                    $$.type = NewString("short");
                    $$.us = 0;
                }
               | TYPE_LONG { 
                    $$.type = NewString("long");
                    $$.us = 0;
                }
               | TYPE_CHAR { 
                    $$.type = NewString("char");
                    $$.us = 0;
                }
               | TYPE_WCHAR { 
                    $$.type = NewString("wchar_t");
                    $$.us = 0;
                }
               | TYPE_FLOAT { 
                    $$.type = NewString("float");
                    $$.us = 0;
                }
               | TYPE_DOUBLE { 
                    $$.type = NewString("double");
                    $$.us = 0;
                }
               | TYPE_SIGNED { 
                    $$.us = NewString("signed");
                    $$.type = 0;
                }
               | TYPE_UNSIGNED { 
                    $$.us = NewString("unsigned");
                    $$.type = 0;
                }
               | TYPE_COMPLEX { 
                    $$.type = NewString("_Complex");
                    $$.us = 0;
                }
               | TYPE_NON_ISO_INT8 { 
                    $$.type = NewString("__int8");
                    $$.us = 0;
                }
               | TYPE_NON_ISO_INT16 { 
                    $$.type = NewString("__int16");
                    $$.us = 0;
                }
               | TYPE_NON_ISO_INT32 { 
                    $$.type = NewString("__int32");
                    $$.us = 0;
                }
               | TYPE_NON_ISO_INT64 { 
                    $$.type = NewString("__int64");
                    $$.us = 0;
                }
               ;

definetype     : expr
               | default_delete
               ;

default_delete : deleted_definition
                | explicit_default
		;

/* For C++ deleted definition '= delete' */
deleted_definition : DELETE_KW {
		  $$ = default_dtype;
		  $$.val = NewString("delete");
		  $$.type = T_STRING;
		}
		;

/* For C++ explicitly defaulted functions '= default' */
explicit_default : DEFAULT {
		  $$ = default_dtype;
		  $$.val = NewString("default");
		  $$.type = T_STRING;
		}
		;

/* Some stuff for handling enums */

ename          :  identifier
	       |  %empty { $$ = 0; }
	       ;

constant_directives : constant_directive
		| constant_directive constant_directives
		;

optional_ignored_defines
		: constant_directives
		| %empty
		;

/* Enum lists - any #define macros (constant directives) within the enum list are ignored. Trailing commas accepted. */

/*
   Note that "_last" attribute is not supposed to be set on the last enum element, as might be expected from its name, but on the _first_ one, and _only_ on it,
   so we propagate it back to the first item while parsing and reset it on all the subsequent ones.
 */

enumlist	: enumlist_item {
		  Setattr($enumlist_item,"_last",$enumlist_item);
		  $$ = $enumlist_item;
		}
		| enumlist_item DOXYGENPOSTSTRING {
		  Setattr($enumlist_item,"_last",$enumlist_item);
		  set_comment($enumlist_item, $DOXYGENPOSTSTRING);
		  $$ = $enumlist_item;
		}
		| enumlist_item DOXYGENSTRING {
		  Setattr($enumlist_item, "_last", $enumlist_item);
		  /* Misplaced doxygen string, attach it to previous parameter, like Doxygen does */
		  set_comment($enumlist_item, $DOXYGENSTRING);
		  $$ = $enumlist_item;
		}
		| enumlist_item COMMA enumlist[in] {
		  if ($in) {
		    set_nextSibling($enumlist_item, $in);
		    Setattr($enumlist_item,"_last",Getattr($in,"_last"));
		    Setattr($in,"_last",NULL);
		  } else {
		    Setattr($enumlist_item,"_last",$enumlist_item);
		  }
		  $$ = $enumlist_item;
		}
		| enumlist_item DOXYGENPOSTSTRING COMMA enumlist[in] {
		  if ($in) {
		    set_nextSibling($enumlist_item, $in);
		    Setattr($enumlist_item,"_last",Getattr($in,"_last"));
		    Setattr($in,"_last",NULL);
		  } else {
		    Setattr($enumlist_item,"_last",$enumlist_item);
		  }
		  set_comment($enumlist_item, $DOXYGENPOSTSTRING);
		  $$ = $enumlist_item;
		}
		| enumlist_item COMMA DOXYGENPOSTSTRING enumlist[in] {
		  if ($in) {
		    set_nextSibling($enumlist_item, $in);
		    Setattr($enumlist_item,"_last",Getattr($in,"_last"));
		    Setattr($in,"_last",NULL);
		  } else {
		    Setattr($enumlist_item,"_last",$enumlist_item);
		  }
		  set_comment($enumlist_item, $DOXYGENPOSTSTRING);
		  $$ = $enumlist_item;
		}
		| optional_ignored_defines {
		  $$ = 0;
		}
		;

enumlist_item	: optional_ignored_defines edecl_with_dox optional_ignored_defines {
		  $$ = $edecl_with_dox;
		}
		;

edecl_with_dox	: edecl
		| DOXYGENSTRING edecl {
		  $$ = $edecl;
		  set_comment($edecl, $DOXYGENSTRING);
		}
		;

edecl          :  identifier {
		   SwigType *type = NewSwigType(T_INT);
		   $$ = new_node("enumitem");
		   Setattr($$,"name",$identifier);
		   Setattr($$,"type",type);
		   SetFlag($$,"feature:immutable");
		   Delete(type);
		 }
                 | identifier EQUAL etype {
		   SwigType *type = NewSwigType($etype.type == T_BOOL ? T_BOOL : ($etype.type == T_CHAR ? T_CHAR : T_INT));
		   $$ = new_node("enumitem");
		   Setattr($$,"name",$identifier);
		   Setattr($$,"type",type);
		   SetFlag($$,"feature:immutable");
		   Setattr($$,"enumvalue", $etype.val);
		   if ($etype.stringval) {
		     Setattr($$, "enumstringval", $etype.stringval);
		   }
		   if ($etype.numval) {
		     Setattr($$, "enumnumval", $etype.numval);
		   }
		   Setattr($$,"value",$identifier);
		   Delete(type);
                 }
                 ;

etype            : expr {
                   $$ = $expr;
		   /* We get T_USER here for a typedef - unfortunately we can't
		    * currently resolve typedefs at this stage of parsing. */
		   if (($$.type != T_INT) && ($$.type != T_UINT) &&
		       ($$.type != T_LONG) && ($$.type != T_ULONG) &&
		       ($$.type != T_LONGLONG) && ($$.type != T_ULONGLONG) &&
		       ($$.type != T_SHORT) && ($$.type != T_USHORT) &&
		       ($$.type != T_SCHAR) && ($$.type != T_UCHAR) &&
		       ($$.type != T_CHAR) && ($$.type != T_BOOL) &&
		       ($$.type != T_UNKNOWN) && ($$.type != T_USER)) {
		     Swig_error(cparse_file,cparse_line,"Type error. Expecting an integral type\n");
		   }
                }
               ;

/* Arithmetic expressions.  Used for constants, C++ templates, and other cool stuff. */

expr           : valexpr
               | type {
		 Node *n;
		 $$ = default_dtype;
		 $$.val = $type;
		 $$.type = T_UNKNOWN;
		 /* Check if value is in scope */
		 n = Swig_symbol_clookup($type,0);
		 if (n) {
                   /* A band-aid for enum values used in expressions. */
                   if (Strcmp(nodeType(n),"enumitem") == 0) {
                     String *q = Swig_symbol_qualified(n);
                     if (q) {
                       $$.val = NewStringf("%s::%s", q, Getattr(n,"name"));
		       $$.type = SwigType_type(Getattr(n, "type"));
                       Delete(q);
                     }
		   } else {
		     SwigType *type = Getattr(n, "type");
		     if (type) {
		       $$.type = SwigType_type(type);
		     }
		   }
		 }
               }
	       ;

/* simple member access expressions */
exprmem        : ID[lhs] ARROW ID[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s->%s", $lhs, $rhs);
	       }
	       | ID[lhs] ARROW ID[rhs] LPAREN {
		 if (skip_balanced('(', ')') < 0) Exit(EXIT_FAILURE);
		 $$ = default_dtype;
		 $$.val = NewStringf("%s->%s%s", $lhs, $rhs, scanner_ccode);
		 Clear(scanner_ccode);
	       }
	       | exprmem[in] ARROW ID {
		 $$ = $in;
		 Printf($$.val, "->%s", $ID);
	       }
	       | exprmem[in] ARROW ID LPAREN {
		 if (skip_balanced('(', ')') < 0) Exit(EXIT_FAILURE);
		 $$ = $in;
		 Printf($$.val, "->%s%s", $ID, scanner_ccode);
		 Clear(scanner_ccode);
	       }
	       | ID[lhs] PERIOD ID[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s.%s", $lhs, $rhs);
	       }
	       | ID[lhs] PERIOD ID[rhs] LPAREN {
		 if (skip_balanced('(', ')') < 0) Exit(EXIT_FAILURE);
		 $$ = default_dtype;
		 $$.val = NewStringf("%s.%s%s", $lhs, $rhs, scanner_ccode);
		 Clear(scanner_ccode);
	       }
	       | exprmem[in] PERIOD ID {
		 $$ = $in;
		 Printf($$.val, ".%s", $ID);
	       }
	       | exprmem[in] PERIOD ID LPAREN {
		 if (skip_balanced('(', ')') < 0) Exit(EXIT_FAILURE);
		 $$ = $in;
		 Printf($$.val, ".%s%s", $ID, scanner_ccode);
		 Clear(scanner_ccode);
	       }
	       ;

/* Non-compound expression */
exprsimple     : exprnum
               | exprmem
               | string {
		  $$ = default_dtype;
		  $$.stringval = $string;
		  $$.val = NewStringf("\"%(escape)s\"", $string);
		  $$.type = T_STRING;
	       }
	       | wstring {
		  $$ = default_dtype;
		  $$.stringval = $wstring;
		  $$.val = NewStringf("L\"%(escape)s\"", $wstring);
		  $$.type = T_WSTRING;
	       }
	       | CHARCONST {
		  $$ = default_dtype;
		  $$.stringval = $CHARCONST;
		  $$.val = NewStringf("'%(escape)s'", $CHARCONST);
		  $$.type = T_CHAR;
	       }
	       | WCHARCONST {
		  $$ = default_dtype;
		  $$.stringval = $WCHARCONST;
		  $$.val = NewStringf("L'%(escape)s'", $WCHARCONST);
		  $$.type = T_WCHAR;
	       }

	       /* In sizeof(X) X can be a type or expression.  We don't actually
		* need to parse X as the type of sizeof is always size_t (which
		* SWIG handles as T_ULONG), so we just skip to the closing ')' and
		* grab the skipped text to use in the value of the expression.
		*/
	       | SIZEOF LPAREN {
		  if (skip_balanced('(', ')') < 0) Exit(EXIT_FAILURE);
		  $$ = default_dtype;
		  $$.val = NewStringf("sizeof%s", scanner_ccode);
		  Clear(scanner_ccode);
		  $$.type = T_ULONG;
               }
	       /* alignof(T) always has type size_t. */
	       | ALIGNOF LPAREN {
		  if (skip_balanced('(', ')') < 0) Exit(EXIT_FAILURE);
		  $$ = default_dtype;
		  $$.val = NewStringf("alignof%s", scanner_ccode);
		  Clear(scanner_ccode);
		  $$.type = T_ULONG;
	       }
	       /* noexcept(X) always has type bool. */
	       | NOEXCEPT LPAREN {
		  if (skip_balanced('(', ')') < 0) Exit(EXIT_FAILURE);
		  $$ = default_dtype;
		  $$.val = NewStringf("noexcept%s", scanner_ccode);
		  Clear(scanner_ccode);
		  $$.type = T_BOOL;
	       }
	       | SIZEOF ELLIPSIS LPAREN identifier RPAREN {
		  $$ = default_dtype;
		  $$.val = NewStringf("sizeof...(%s)", $identifier);
		  $$.type = T_ULONG;
               }
	       /* `sizeof expr` without parentheses is valid for an expression,
		* but not for a type.  This doesn't support `sizeof x` (or
		* `sizeof <unaryop> x` but that's unlikely to be seen in real
		* code).
		*/
	       | SIZEOF exprsimple[in] {
		  $$ = default_dtype;
		  $$.val = NewStringf("sizeof(%s)", $in.val);
		  $$.type = T_ULONG;
	       }
               ;

valexpr        : exprsimple
	       | exprcompound

/* grouping */
               |  LPAREN expr RPAREN %prec CAST {
	            $$ = default_dtype;
		    $$.val = NewStringf("(%s)",$expr.val);
		    $$.stringval = Copy($expr.stringval);
		    $$.numval = Copy($expr.numval);
		    $$.type = $expr.type;
	       }

/* A few common casting operations */

               | LPAREN expr[lhs] RPAREN expr[rhs] %prec CAST {
		 int cast_type_code = SwigType_type($lhs.val);
		 $$ = $rhs;
		 $$.unary_arg_type = 0;
		 if ($rhs.type != T_STRING) {
		   switch ($lhs.type) {
		     case T_FLOAT:
		     case T_DOUBLE:
		     case T_LONGDOUBLE:
		     case T_FLTCPLX:
		     case T_DBLCPLX:
		       $$.val = NewStringf("(%s)%s", $lhs.val, $rhs.val); /* SwigType_str and decimal points don't mix! */
		       break;
		     default:
		       $$.val = NewStringf("(%s) %s", SwigType_str($lhs.val,0), $rhs.val);
		       break;
		   }
		   $$.stringval = 0;
		   $$.numval = 0;
		 }
		 /* As well as C-style casts, this grammar rule currently also
		  * matches a binary operator with a LHS in parentheses for
		  * binary operators which also have an unary form, e.g.:
		  *
		  * (6)*7
		  * (6)&7
		  * (6)+7
		  * (6)-7
		  */
		 if (cast_type_code != T_USER && cast_type_code != T_UNKNOWN) {
		   /* $lhs is definitely a type so we know this is a cast. */
		   $$.type = cast_type_code;
		 } else if ($rhs.type == 0 || $rhs.unary_arg_type == 0) {
		   /* Not one of the cases above, so we know this is a cast. */
		   $$.type = cast_type_code;
		 } else {
		   $$.type = promote($lhs.type, $rhs.unary_arg_type);
		 }
 	       }
               | LPAREN expr[lhs] pointer RPAREN expr[rhs] %prec CAST {
                 $$ = $rhs;
		 $$.unary_arg_type = 0;
		 if ($rhs.type != T_STRING) {
		   SwigType_push($lhs.val,$pointer);
		   $$.val = NewStringf("(%s) %s", SwigType_str($lhs.val,0), $rhs.val);
		   $$.stringval = 0;
		   $$.numval = 0;
		 }
 	       }
               | LPAREN expr[lhs] AND RPAREN expr[rhs] %prec CAST {
                 $$ = $rhs;
		 $$.unary_arg_type = 0;
		 if ($rhs.type != T_STRING) {
		   SwigType_add_reference($lhs.val);
		   $$.val = NewStringf("(%s) %s", SwigType_str($lhs.val,0), $rhs.val);
		   $$.stringval = 0;
		   $$.numval = 0;
		 }
 	       }
               | LPAREN expr[lhs] LAND RPAREN expr[rhs] %prec CAST {
                 $$ = $rhs;
		 $$.unary_arg_type = 0;
		 if ($rhs.type != T_STRING) {
		   SwigType_add_rvalue_reference($lhs.val);
		   $$.val = NewStringf("(%s) %s", SwigType_str($lhs.val,0), $rhs.val);
		   $$.stringval = 0;
		   $$.numval = 0;
		 }
 	       }
               | LPAREN expr[lhs] pointer AND RPAREN expr[rhs] %prec CAST {
                 $$ = $rhs;
		 $$.unary_arg_type = 0;
		 if ($rhs.type != T_STRING) {
		   SwigType_push($lhs.val,$pointer);
		   SwigType_add_reference($lhs.val);
		   $$.val = NewStringf("(%s) %s", SwigType_str($lhs.val,0), $rhs.val);
		   $$.stringval = 0;
		   $$.numval = 0;
		 }
 	       }
               | LPAREN expr[lhs] pointer LAND RPAREN expr[rhs] %prec CAST {
                 $$ = $rhs;
		 $$.unary_arg_type = 0;
		 if ($rhs.type != T_STRING) {
		   SwigType_push($lhs.val,$pointer);
		   SwigType_add_rvalue_reference($lhs.val);
		   $$.val = NewStringf("(%s) %s", SwigType_str($lhs.val,0), $rhs.val);
		   $$.stringval = 0;
		   $$.numval = 0;
		 }
 	       }
               | AND expr {
		 $$ = $expr;
		 $$.val = NewStringf("&%s", $expr.val);
		 $$.stringval = 0;
		 $$.numval = 0;
		 /* Record the type code for expr so we can properly handle
		  * cases such as (6)&7 which get parsed using this rule then
		  * the rule for a C-style cast.
		  */
		 $$.unary_arg_type = $expr.type;
		 switch ($$.type) {
		   case T_CHAR:
		     $$.type = T_STRING;
		     break;
		   case T_WCHAR:
		     $$.type = T_WSTRING;
		     break;
		   default:
		     $$.type = T_POINTER;
		 }
	       }
               | STAR expr {
		 $$ = $expr;
		 $$.val = NewStringf("*%s", $expr.val);
		 $$.stringval = 0;
		 $$.numval = 0;
		 /* Record the type code for expr so we can properly handle
		  * cases such as (6)*7 which get parsed using this rule then
		  * the rule for a C-style cast.
		  */
		 $$.unary_arg_type = $expr.type;
		 switch ($$.type) {
		   case T_STRING:
		     $$.type = T_CHAR;
		     break;
		   case T_WSTRING:
		     $$.type = T_WCHAR;
		     break;
		   default:
		     $$.type = T_UNKNOWN;
		 }
	       }
	       ;

exprnum        :  NUM_INT
               |  NUM_DOUBLE
               |  NUM_FLOAT
               |  NUM_LONGDOUBLE
               |  NUM_UNSIGNED
               |  NUM_LONG
               |  NUM_ULONG
               |  NUM_LONGLONG
               |  NUM_ULONGLONG
               |  NUM_BOOL
               ;

exprcompound   : expr[lhs] PLUS expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s+%s", $lhs.val, $rhs.val);
		 $$.type = promote($lhs.type,$rhs.type);
	       }
               | expr[lhs] MINUS expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s-%s", $lhs.val, $rhs.val);
		 $$.type = promote($lhs.type,$rhs.type);
	       }
               | expr[lhs] STAR expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s*%s", $lhs.val, $rhs.val);
		 $$.type = promote($lhs.type,$rhs.type);
	       }
               | expr[lhs] SLASH expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s/%s", $lhs.val, $rhs.val);
		 $$.type = promote($lhs.type,$rhs.type);
	       }
               | expr[lhs] MODULO expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s%%%s", $lhs.val, $rhs.val);
		 $$.type = promote($lhs.type,$rhs.type);
	       }
               | expr[lhs] AND expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s&%s", $lhs.val, $rhs.val);
		 $$.type = promote($lhs.type,$rhs.type);
	       }
               | expr[lhs] OR expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s|%s", $lhs.val, $rhs.val);
		 $$.type = promote($lhs.type,$rhs.type);
	       }
               | expr[lhs] XOR expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s^%s", $lhs.val, $rhs.val);
		 $$.type = promote($lhs.type,$rhs.type);
	       }
               | expr[lhs] LSHIFT expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s << %s", $lhs.val, $rhs.val);
		 $$.type = promote_type($lhs.type);
	       }
               | expr[lhs] RSHIFT expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s >> %s", $lhs.val, $rhs.val);
		 $$.type = promote_type($lhs.type);
	       }
               | expr[lhs] LAND expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s&&%s", $lhs.val, $rhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
               | expr[lhs] LOR expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s||%s", $lhs.val, $rhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
               | expr[lhs] EQUALTO expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s==%s", $lhs.val, $rhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
               | expr[lhs] NOTEQUALTO expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s!=%s", $lhs.val, $rhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
	       /* Trying to parse `>` in the general case results in conflicts
		* in the parser, but all user-reported cases are actually inside
		* parentheses and we can handle that case.
		*/
	       | LPAREN expr[lhs] GREATERTHAN expr[rhs] RPAREN {
		 $$ = default_dtype;
		 $$.val = NewStringf("(%s > %s)", $lhs.val, $rhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }

	       /* Similarly for `<` except trying to handle exprcompound on the
		* left side gives a shift/reduce conflict, so also restrict
		* handling to non-compound subexpressions there.  Again this
		* covers all user-reported cases.
		*/
               | LPAREN exprsimple[lhs] LESSTHAN expr[rhs] RPAREN {
		 $$ = default_dtype;
		 $$.val = NewStringf("(%s < %s)", $lhs.val, $rhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
               | expr[lhs] GREATERTHANOREQUALTO expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s >= %s", $lhs.val, $rhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
               | expr[lhs] LESSTHANOREQUALTO expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s <= %s", $lhs.val, $rhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }

	       // C++17 fold expressions.
	       //
	       // We don't handle unary left fold currently, since the obvious
	       // approach introduces shift/reduce conflicts.  (Binary folds
	       // should be handled by composition of expressions.)
	       //
	       // Fold expressions using the following operators are not
	       // currently handled (because we don't actually seem to handle
	       // these operators in expressions at all!):
	       //
	       // = += -= *= /= %= ^= &= |= <<= >>= , .* ->*.
	       | expr[lhs] PLUS ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s+...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] MINUS ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s-...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] STAR ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s*...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] SLASH ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s/...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] MODULO ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s%%...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] AND ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s&...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] OR ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s|...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] XOR ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s^...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] LSHIFT ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s << ...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] RSHIFT ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s >> ...", $lhs.val);
		 $$.type = promote_type($lhs.type);
	       }
	       | expr[lhs] LAND ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s&&...", $lhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
	       | expr[lhs] LOR ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s||...", $lhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
	       | expr[lhs] EQUALTO ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s==...", $lhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
	       | expr[lhs] NOTEQUALTO ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s!=...", $lhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
	       /* Trying to parse `>` in the general case results in conflicts
		* in the parser, but all user-reported cases are actually inside
		* parentheses and we can handle that case.
		*/
	       | LPAREN expr[lhs] GREATERTHAN ELLIPSIS RPAREN {
		 $$ = default_dtype;
		 $$.val = NewStringf("(%s > ...)", $lhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
	       /* Similarly for `<` except trying to handle exprcompound on the
		* left side gives a shift/reduce conflict, so also restrict
		* handling to non-compound subexpressions there.  Again this
		* covers all user-reported cases.
		*/
	       | LPAREN exprsimple[lhs] LESSTHAN ELLIPSIS RPAREN {
		 $$ = default_dtype;
		 $$.val = NewStringf("(%s < %s)", $lhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
	       | expr[lhs] GREATERTHANOREQUALTO ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s >= ...", $lhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
	       | expr[lhs] LESSTHANOREQUALTO ELLIPSIS {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s <= ...", $lhs.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }

	       | expr[lhs] LESSEQUALGREATER expr[rhs] {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s <=> %s", $lhs.val, $rhs.val);
		 /* `<=>` returns one of `std::strong_ordering`,
		  * `std::partial_ordering` or `std::weak_ordering`.  The main
		  * thing to do with the return value in this context is to
		  * compare it with another ordering of the same type or
		  * with a literal 0.  We set .type = T_USER here which does
		  * what we want for the comparison operators, and also means
		  * that deduce_type() won't deduce a type for this (which is
		  * better than it deducing the wrong type).
		  */
		 $$.type = T_USER;
		 $$.unary_arg_type = 0;
	       }
	       | expr[expr1] QUESTIONMARK expr[expr2] COLON expr[expr3] %prec QUESTIONMARK {
		 $$ = default_dtype;
		 $$.val = NewStringf("%s?%s:%s", $expr1.val, $expr2.val, $expr3.val);
		 /* This may not be exactly right, but is probably good enough
		  * for the purposes of parsing constant expressions. */
		 $$.type = promote($expr2.type, $expr3.type);
	       }
               | MINUS expr[in] %prec UMINUS {
		 $$ = default_dtype;
		 $$.val = NewStringf("-%s",$in.val);
		 if ($in.numval) {
		   switch ($in.type) {
		     case T_CHAR: // Unsigned on some architectures.
		     case T_UCHAR:
		     case T_USHORT:
		     case T_UINT:
		     case T_ULONG:
		     case T_ULONGLONG:
		       // Avoid negative numval with an unsigned type.
		       break;
		     default:
		       $$.numval = NewStringf("-%s", $in.numval);
		       break;
		   }
		   Delete($in.numval);
		 }
		 $$.type = promote_type($in.type);
		 /* Record the type code for expr so we can properly handle
		  * cases such as (6)-7 which get parsed using this rule then
		  * the rule for a C-style cast.
		  */
		 $$.unary_arg_type = $in.type;
	       }
               | PLUS expr[in] %prec UMINUS {
		 $$ = default_dtype;
                 $$.val = NewStringf("+%s",$in.val);
		 $$.numval = $in.numval;
		 $$.type = promote_type($in.type);
		 /* Record the type code for expr so we can properly handle
		  * cases such as (6)+7 which get parsed using this rule then
		  * the rule for a C-style cast.
		  */
		 $$.unary_arg_type = $in.type;
	       }
               | NOT expr[in] {
		 $$ = default_dtype;
		 $$.val = NewStringf("~%s",$in.val);
		 $$.type = promote_type($in.type);
	       }
               | LNOT expr[in] {
		 $$ = default_dtype;
                 $$.val = NewStringf("!%s", $in.val);
		 $$.type = cparse_cplusplus ? T_BOOL : T_INT;
	       }
               | type LPAREN {
		 $$ = default_dtype;
		 String *qty;
		 if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
		 qty = Swig_symbol_type_qualify($type,0);
		 if (SwigType_istemplate(qty)) {
		   String *nstr = SwigType_namestr(qty);
		   Delete(qty);
		   qty = nstr;
		 }
		 $$.val = NewStringf("%s%s",qty,scanner_ccode);
		 Clear(scanner_ccode);
		 /* Try to deduce the type - this could be a C++ "constructor
		  * cast" such as `double(4)` or a function call such as
		  * `some_func()`.  In the latter case we get T_USER, but that
		  * is wrong so we map it to T_UNKNOWN until we can actually
		  * deduce the return type of a function call (which is
		  * complicated because the return type can vary between
		  * overloaded forms).
		  */
		 $$.type = SwigType_type(qty);
		 if ($$.type == T_USER) $$.type = T_UNKNOWN;
		 $$.unary_arg_type = 0;
		 Delete(qty);
               }
               ;

variadic_opt  : ELLIPSIS {
		$$ = 1;
	      }
	      | %empty {
	        $$ = 0;
	      }
	      ;

inherit        : raw_inherit
               ;

raw_inherit     : COLON { inherit_list = 1; } base_list { $$ = $base_list; inherit_list = 0; }
                | %empty { $$ = 0; }
                ;

base_list      : base_specifier {
		   Hash *list = NewHash();
		   Node *base = $base_specifier;
		   Node *name = Getattr(base,"name");
		   List *lpublic = NewList();
		   List *lprotected = NewList();
		   List *lprivate = NewList();
		   Setattr(list,"public",lpublic);
		   Setattr(list,"protected",lprotected);
		   Setattr(list,"private",lprivate);
		   Delete(lpublic);
		   Delete(lprotected);
		   Delete(lprivate);
		   Append(Getattr(list,Getattr(base,"access")),name);
	           $$ = list;
               }

               | base_list[in] COMMA base_specifier {
		   Hash *list = $in;
		   Node *base = $base_specifier;
		   Node *name = Getattr(base,"name");
		   Append(Getattr(list,Getattr(base,"access")),name);
                   $$ = list;
               }
               ;

base_specifier : opt_virtual <intvalue>{
		 $$ = cparse_line;
	       }[line] idcolon variadic_opt {
		 $$ = NewHash();
		 Setfile($$, cparse_file);
		 Setline($$, $line);
		 Setattr($$, "name", $idcolon);
		 Setfile($idcolon, cparse_file);
		 Setline($idcolon, $line);
                 if (last_cpptype && (Strcmp(last_cpptype,"struct") != 0)) {
		   Setattr($$,"access","private");
		   Swig_warning(WARN_PARSE_NO_ACCESS, Getfile($$), Getline($$), "No access specifier given for base class '%s' (ignored).\n", SwigType_namestr($idcolon));
                 } else {
		   Setattr($$,"access","public");
		 }
		 if ($variadic_opt) {
		   SwigType_add_variadic(Getattr($$, "name"));
		 }
               }
	       | opt_virtual access_specifier <intvalue>{
		 $$ = cparse_line;
	       }[line] opt_virtual idcolon variadic_opt {
		 $$ = NewHash();
		 Setfile($$, cparse_file);
		 Setline($$, $line);
		 Setattr($$, "name", $idcolon);
		 Setfile($idcolon, cparse_file);
		 Setline($idcolon, $line);
		 Setattr($$, "access", $access_specifier);
		 if (Strcmp($access_specifier, "public") != 0) {
		   Swig_warning(WARN_PARSE_PRIVATE_INHERIT, Getfile($$), Getline($$), "%s inheritance from base '%s' (ignored).\n", $access_specifier, SwigType_namestr($idcolon));
		 }
		 if ($variadic_opt) {
		   SwigType_add_variadic(Getattr($$, "name"));
		 }
               }
               ;

access_specifier :  PUBLIC { $$ = "public"; }
               | PRIVATE { $$ = "private"; }
               | PROTECTED { $$ = "protected"; }
               ;

templcpptype   : CLASS variadic_opt {
                   $$ = NewString("class");
		   if (!inherit_list) last_cpptype = $$;
		   if ($variadic_opt) SwigType_add_variadic($$);
               }
               | TYPENAME variadic_opt {
                   $$ = NewString("typename");
		   if (!inherit_list) last_cpptype = $$;
		   if ($variadic_opt) SwigType_add_variadic($$);
               }
               ;

cpptype        : templcpptype
               | STRUCT {
                   $$ = NewString("struct");
		   if (!inherit_list) last_cpptype = $$;
               }
               | UNION {
                   $$ = NewString("union");
		   if (!inherit_list) last_cpptype = $$;
               }
               ;

classkey       : CLASS {
		   if (!inherit_list) last_cpptype = NewString("class");
               }
               | STRUCT {
		   if (!inherit_list) last_cpptype = NewString("struct");
               }
               | UNION {
		   if (!inherit_list) last_cpptype = NewString("union");
               }
               ;

classkeyopt    : classkey
               | %empty
               ;

opt_virtual    : VIRTUAL
               | %empty
               ;

virt_specifier_seq : OVERRIDE {
                   $$ = 0;
	       }
	       | FINAL {
                   $$ = NewString("1");
	       }
	       | FINAL OVERRIDE {
                   $$ = NewString("1");
	       }
	       | OVERRIDE FINAL {
                   $$ = NewString("1");
	       }
               ;

virt_specifier_seq_opt : virt_specifier_seq
               | %empty {
                   $$ = 0;
               }
               ;

class_virt_specifier_opt : FINAL {
                   $$ = NewString("1");
               }
               | %empty {
                   $$ = 0;
               }
               ;

exception_specification : THROW LPAREN parms RPAREN {
		    $$ = default_dtype;
                    $$.throws = $parms;
                    $$.throwf = NewString("1");
	       }
	       | NOEXCEPT {
		    $$ = default_dtype;
                    $$.nexcept = NewString("true");
	       }
	       | virt_specifier_seq {
		    $$ = default_dtype;
                    $$.final = $virt_specifier_seq;
	       }
	       | THROW LPAREN parms RPAREN virt_specifier_seq {
		    $$ = default_dtype;
                    $$.throws = $parms;
                    $$.throwf = NewString("1");
                    $$.final = $virt_specifier_seq;
	       }
	       | NOEXCEPT virt_specifier_seq {
		    $$ = default_dtype;
                    $$.nexcept = NewString("true");
		    $$.final = $virt_specifier_seq;
	       }
	       | NOEXCEPT LPAREN expr RPAREN {
		    $$ = default_dtype;
                    $$.nexcept = $expr.val;
	       }
	       ;	

qualifiers_exception_specification : cv_ref_qualifier {
		    $$ = default_dtype;
                    $$.qualifier = $cv_ref_qualifier.qualifier;
                    $$.refqualifier = $cv_ref_qualifier.refqualifier;
               }
               | exception_specification {
		    $$ = $exception_specification;
               }
               | cv_ref_qualifier exception_specification {
		    $$ = $exception_specification;
                    $$.qualifier = $cv_ref_qualifier.qualifier;
                    $$.refqualifier = $cv_ref_qualifier.refqualifier;
               }
               ;

cpp_const      : qualifiers_exception_specification
               | %empty {
                 $$ = default_dtype;
               }
               ;

ctor_end       : cpp_const ctor_initializer SEMI { 
                    Clear(scanner_ccode); 
		    $$ = default_decl;
		    $$.throws = $cpp_const.throws;
		    $$.throwf = $cpp_const.throwf;
		    $$.nexcept = $cpp_const.nexcept;
		    $$.final = $cpp_const.final;
                    if ($cpp_const.qualifier)
                      Swig_error(cparse_file, cparse_line, "Constructor cannot have a qualifier.\n");
               }
               | cpp_const ctor_initializer LBRACE { 
                    if ($cpp_const.qualifier)
                      Swig_error(cparse_file, cparse_line, "Constructor cannot have a qualifier.\n");
                    if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		    $$ = default_decl;
                    $$.throws = $cpp_const.throws;
                    $$.throwf = $cpp_const.throwf;
                    $$.nexcept = $cpp_const.nexcept;
                    $$.final = $cpp_const.final;
               }
               | LPAREN parms RPAREN SEMI { 
                    Clear(scanner_ccode); 
		    $$ = default_decl;
                    $$.parms = $parms; 
                    $$.have_parms = 1; 
               }
               | LPAREN parms RPAREN LBRACE {
                    if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		    $$ = default_decl;
                    $$.parms = $parms; 
                    $$.have_parms = 1; 
               }
               | EQUAL definetype SEMI { 
		    $$ = default_decl;
                    $$.defarg = $definetype.val; 
		    $$.stringdefarg = $definetype.stringval;
		    $$.numdefarg = $definetype.numval;
               }
               | exception_specification EQUAL default_delete SEMI {
		    $$ = default_decl;
                    $$.defarg = $default_delete.val;
		    $$.stringdefarg = $default_delete.stringval;
		    $$.numdefarg = $default_delete.numval;
                    $$.throws = $exception_specification.throws;
                    $$.throwf = $exception_specification.throwf;
                    $$.nexcept = $exception_specification.nexcept;
                    $$.final = $exception_specification.final;
                    if ($exception_specification.qualifier)
                      Swig_error(cparse_file, cparse_line, "Constructor cannot have a qualifier.\n");
               }
               ;

ctor_initializer : COLON mem_initializer_list
               | %empty
               ;

mem_initializer_list : mem_initializer
               | mem_initializer_list COMMA mem_initializer
               | mem_initializer ELLIPSIS
               | mem_initializer_list COMMA mem_initializer ELLIPSIS
               ;

mem_initializer : idcolon LPAREN {
		  if (skip_balanced('(',')') < 0) Exit(EXIT_FAILURE);
		  Clear(scanner_ccode);
		}
                /* Uniform initialization in C++11.
		   Example:
                   struct MyStruct {
                     MyStruct(int x, double y) : x_{x}, y_{y} {}
                     int x_;
                     double y_;
                   };
                */
                | idcolon LBRACE {
		  if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		  Clear(scanner_ccode);
		}
                ;

less_valparms_greater : LESSTHAN valparms GREATERTHAN {
                     String *s = NewStringEmpty();
                     SwigType_add_template(s,$valparms);
		     $$ = s;
		     scanner_last_id(1);
                }
		;

/* Identifiers including the C++11 identifiers with special meaning */
identifier     : ID
	       | OVERRIDE { $$ = Swig_copy_string("override"); }
	       | FINAL { $$ = Swig_copy_string("final"); }
	       ;

idstring       : identifier
               | default_delete { $$ = Char($default_delete.val); }
               | string { $$ = Char($string); }
               ;

idstringopt    : idstring
               | %empty { $$ = 0; }
               ;

idcolon        : idtemplate idcolontail { 
		 $$ = NewStringf("%s%s", $idtemplate, $idcolontail);
		 Delete($idcolontail);
               }
               | NONID DCOLON idtemplatetemplate idcolontail {
		 $$ = NewStringf("::%s%s",$idtemplatetemplate,$idcolontail);
                 Delete($idcolontail);
               }
               | idtemplate {
		 $$ = NewString($idtemplate);
   	       }
               | NONID DCOLON idtemplatetemplate {
		 $$ = NewStringf("::%s",$idtemplatetemplate);
               }
               | OPERATOR %expect 1 {
		 $$ = $OPERATOR;
	       }
               | OPERATOR less_valparms_greater {
		 $$ = $OPERATOR;
		 Append($$, $less_valparms_greater);
		 Delete($less_valparms_greater);
	       }
               | NONID DCOLON OPERATOR {
		 $$ = $OPERATOR;
		 Insert($$, 0, "::");
               }
               ;

idcolontail    : DCOLON idtemplatetemplate idcolontail[in] {
                   $$ = NewStringf("::%s%s",$idtemplatetemplate,$in);
		   Delete($in);
               }
               | DCOLON idtemplatetemplate {
                   $$ = NewStringf("::%s",$idtemplatetemplate);
               }
               | DCOLON OPERATOR {
		   $$ = $OPERATOR;
		   Insert($$, 0, "::");
               }
               | DCNOT idtemplate {
		 $$ = NewStringf("::~%s",$idtemplate);
               }
               ;


idtemplate    : identifier {
		$$ = NewString($identifier);
	      }
	      | identifier less_valparms_greater {
		$$ = NewString($identifier);
		Append($$, $less_valparms_greater);
		Delete($less_valparms_greater);
	      }
              ;

idtemplatetemplate : idtemplate
	      | TEMPLATE identifier less_valparms_greater {
		$$ = NewString($identifier);
		Append($$, $less_valparms_greater);
		Delete($less_valparms_greater);
	      }
              ;

/* Identifier, but no templates */
idcolonnt     : identifier idcolontailnt {
		 $$ = NewStringf("%s%s", $identifier, $idcolontailnt);
		 Delete($idcolontailnt);
               }
               | NONID DCOLON identifier idcolontailnt {
		 $$ = NewStringf("::%s%s",$identifier,$idcolontailnt);
                 Delete($idcolontailnt);
               }
               | identifier {
		 $$ = NewString($identifier);
   	       }     
               | NONID DCOLON identifier {
		 $$ = NewStringf("::%s",$identifier);
               }
               | OPERATOR {
		 $$ = $OPERATOR;
	       }
               | NONID DCOLON OPERATOR {
		 $$ = $OPERATOR;
		 Insert($$, 0, "::");
               }
               ;

idcolontailnt   : DCOLON identifier idcolontailnt[in] {
                   $$ = NewStringf("::%s%s",$identifier,$in);
		   Delete($in);
               }
               | DCOLON identifier {
                   $$ = NewStringf("::%s",$identifier);
               }
               | DCOLON OPERATOR {
		   $$ = $OPERATOR;
		   Insert($$, 0, "::");
               }
               | DCNOT identifier {
		 $$ = NewStringf("::~%s",$identifier);
               }
               ;

/* Concatenated strings */
string	       : string[in] STRING { 
		   $$ = $in;
		   Append($$, $STRING);
		   Delete($STRING);
	       }
	       | STRING
	       ; 
wstring	       : wstring[in] WSTRING {
		   // Concatenated wide strings: L"str1" L"str2"
		   $$ = $in;
		   Append($$, $WSTRING);
		   Delete($WSTRING);
	       }
	       | wstring[in] STRING {
		   // Concatenated wide string and normal string literal: L"str1" "str2" (C++11).
		   $$ = $in;
		   Append($$, $STRING);
		   Delete($STRING);
	       }
	       | string[in] WSTRING {
		   // Concatenated normal string and wide string literal: "str1" L"str2" (C++11).
		   $$ = $in;
		   Append($$, $WSTRING);
		   Delete($WSTRING);
	       }
	       | WSTRING
	       ;

stringbrace    : string
               | LBRACE {
		  if (skip_balanced('{','}') < 0) Exit(EXIT_FAILURE);
		  $$ = NewString(scanner_ccode);
               }
              | HBLOCK
               ;

options        : LPAREN kwargs RPAREN {
                  Hash *n;
                  $$ = NewHash();
                  n = $kwargs;
                  while(n) {
                     String *name, *value;
                     name = Getattr(n,"name");
                     value = Getattr(n,"value");
		     if (!value) value = (String *) "1";
                     Setattr($$,name, value);
		     n = nextSibling(n);
		  }
               }   
               | %empty { $$ = 0; };

 
/* Keyword arguments */
kwargs	       : kwargs_builder {
		 $$ = $kwargs_builder.node;
	       }
	       ;

kwargs_builder : idstring EQUAL stringnum {
		 Node *n = NewHash();
		 Setattr(n, "name", $idstring);
		 Setattr(n, "value", $stringnum);
		 Delete($stringnum);
		 $$.node = $$.last = n;
	       }
	       | kwargs_builder[in] COMMA idstring EQUAL stringnum {
		 $$ = $in;
		 Node *n = NewHash();
		 Setattr(n, "name", $idstring);
		 Setattr(n, "value", $stringnum);
		 Delete($stringnum);
		 set_nextSibling($$.last, n);
		 $$.last = n;
	       }
	       | idstring {
		 Node *n = NewHash();
		 Setattr(n, "name", $idstring);
		 $$.node = $$.last = n;
	       }
	       | kwargs_builder[in] COMMA idstring {
		 $$ = $in;
		 Node *n = NewHash();
		 Setattr(n, "name", $idstring);
		 set_nextSibling($$.last, n);
		 $$.last = n;
	       }
	       | idstring EQUAL stringtype {
		 Node *n = $stringtype;
		 Setattr(n, "name", $idstring);
		 $$.node = $$.last = n;
	       }
	       | kwargs_builder[in] COMMA idstring EQUAL stringtype {
		 $$ = $in;
		 Node *n = $stringtype;
		 Setattr(n, "name", $idstring);
		 set_nextSibling($$.last, n);
		 $$.last = n;
	       }
	       ;

stringnum      : string
               | exprnum {
		 $$ = $exprnum.val;
               }
               ;

%%

SwigType *Swig_cparse_type(String *s) {
   String *ns;
   ns = NewString(s);
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSETYPE);
   if (yyparse() == 2) {
      Swig_error(cparse_file, cparse_line, "Parser exceeded stack depth or ran out of memory\n");
      Exit(EXIT_FAILURE);
   }
   /*   Printf(stdout,"typeparse: '%s' ---> '%s'\n", s, top); */
   return (SwigType *)top;
}


Parm *Swig_cparse_parm(String *s) {
   String *ns;
   ns = NewString(s);
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSEPARM);
   if (yyparse() == 2) {
      Swig_error(cparse_file, cparse_line, "Parser exceeded stack depth or ran out of memory\n");
      Exit(EXIT_FAILURE);
   }
   /*   Printf(stdout,"parmparse: '%s' ---> '%s'\n", s, top); */
   Delete(ns);
   return (Parm *)top;
}


ParmList *Swig_cparse_parms(String *s, Node *file_line_node) {
   String *ns;
   char *cs = Char(s);
   if (cs && cs[0] != '(') {
     ns = NewStringf("(%s)",s);
   } else {
     ns = NewString(s);
   }
   Setfile(ns, Getfile(file_line_node));
   Setline(ns, Getline(file_line_node));
   Seek(ns,0,SEEK_SET);
   scanner_file(ns);
   top = 0;
   scanner_next_token(PARSEPARMS);
   if (yyparse() == 2) {
      Swig_error(cparse_file, cparse_line, "Parser exceeded stack depth or ran out of memory\n");
      Exit(EXIT_FAILURE);
   }
   /*   Printf(stdout,"parmsparse: '%s' ---> '%s'\n", s, top); */
   return (ParmList *)top;
}

