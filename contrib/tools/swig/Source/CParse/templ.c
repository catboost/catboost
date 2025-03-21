/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * templ.c
 *
 * Expands a template into a specialized version.   
 * ----------------------------------------------------------------------------- */

#include "swig.h"
#include "cparse.h"
#include <ctype.h>

static int template_debug = 0;


const char *baselists[3];

void SwigType_template_init(void) {
  baselists[0] = "baselist";
  baselists[1] = "protectedbaselist";
  baselists[2] = "privatebaselist";
}

void Swig_cparse_debug_templates(int x) {
  template_debug = x;
}

/* -----------------------------------------------------------------------------
 * add_parms()
 *
 * Add the value and type of each parameter into patchlist and typelist
 * (List of String/SwigType) for later template parameter substitutions.
 * ----------------------------------------------------------------------------- */

static void add_parms(ParmList *p, List *patchlist, List *typelist, int is_pattern) {
  while (p) {
    SwigType *ty = Getattr(p, "type");
    SwigType *val = Getattr(p, "value");
    Append(typelist, ty);
    if (is_pattern) {
      /* Typemap patterns are not simple parameter lists.
       * Output style ("out", "ret" etc) typemap names can be
       * qualified names and so may need template expansion */
      SwigType *name = Getattr(p, "name");
      Append(typelist, name);
    }
    Append(patchlist, val);
    p = nextSibling(p);
  }
}

/* -----------------------------------------------------------------------------
 * expand_variadic_parms()
 *
 * Expand variadic parameter in the parameter list stored as attribute in n. For example:
 *   template <typename... T> struct X : { X(T&... tt); }
 *   %template(XABC) X<A,B,C>;
 * inputs for the constructor parameter list will be for attribute = "parms":
 *   Getattr(n, attribute)   : v.r.T tt
 *   unexpanded_variadic_parm: v.typename T
 *   expanded_variadic_parms : A,B,C
 * results in:
 *   Getattr(n, attribute)   : r.A,r.B,r.C
 * that is, template is expanded as: struct XABC : { X(A&,B&,C&); }
 * Note that there are no parameter names are in the expanded parameter list.
 * Nothing happens if the parameter list has no variadic parameters.
 * ----------------------------------------------------------------------------- */

static void expand_variadic_parms(Node *n, const char *attribute, Parm *unexpanded_variadic_parm, ParmList *expanded_variadic_parms) {
  ParmList *p = Getattr(n, attribute);
  if (unexpanded_variadic_parm) {
    Parm *variadic = ParmList_variadic_parm(p);
    if (variadic) {
      SwigType *type = Getattr(variadic, "type");
      String *name = Getattr(variadic, "name");
      String *unexpanded_name = Getattr(unexpanded_variadic_parm, "name");
      ParmList *expanded = CopyParmList(expanded_variadic_parms);
      Parm *ep = expanded;
      int i = 0;
      while (ep) {
	SwigType *newtype = Copy(type);
	SwigType_del_variadic(newtype);
	Replaceid(newtype, unexpanded_name, Getattr(ep, "type"));
	Setattr(ep, "type", newtype);
	Setattr(ep, "name", name ? NewStringf("%s%d", name, ++i) : 0);
	ep = nextSibling(ep);
      }
      expanded = ParmList_replace_last(p, expanded);
      Setattr(n, attribute, expanded);
    }
  }
}

/* -----------------------------------------------------------------------------
 * expand_parms()
 *
 * Expand variadic parameters in parameter lists and add parameters to patchlist
 * and typelist for later template parameter substitutions.
 * ----------------------------------------------------------------------------- */

static void expand_parms(Node *n, const char *attribute, Parm *unexpanded_variadic_parm, ParmList *expanded_variadic_parms, List *patchlist, List *typelist, int is_pattern) {
  ParmList *p;
  expand_variadic_parms(n, attribute, unexpanded_variadic_parm, expanded_variadic_parms);
  p = Getattr(n, attribute);
  add_parms(p, patchlist, typelist, is_pattern);
}

/* -----------------------------------------------------------------------------
 * cparse_template_expand()
 *
 * Expands a template node into a specialized version.  This is done by
 * patching typenames and other aspects of the node according to a list of
 * template parameters
 * ----------------------------------------------------------------------------- */

static void cparse_template_expand(Node *templnode, Node *n, String *tname, String *rname, String *templateargs, List *patchlist, List *typelist, List *cpatchlist, Parm *unexpanded_variadic_parm, ParmList *expanded_variadic_parms) {
  static int expanded = 0;
  String *nodeType;
  if (!n)
    return;
  nodeType = nodeType(n);
  if (Getattr(n, "error"))
    return;

  if (Equal(nodeType, "template")) {
    /* Change the node type back to normal */
    if (!expanded) {
      expanded = 1;
      set_nodeType(n, Getattr(n, "templatetype"));
      cparse_template_expand(templnode, n, tname, rname, templateargs, patchlist, typelist, cpatchlist, unexpanded_variadic_parm, expanded_variadic_parms);
      expanded = 0;
      return;
    } else {
      /* Called when template appears inside another template */
      /* Member templates */

      set_nodeType(n, Getattr(n, "templatetype"));
      cparse_template_expand(templnode, n, tname, rname, templateargs, patchlist, typelist, cpatchlist, unexpanded_variadic_parm, expanded_variadic_parms);
      set_nodeType(n, "template");
      return;
    }
  } else if (Equal(nodeType, "cdecl")) {
    /* A simple C declaration */
    SwigType *t, *v, *d;
    String *code;
    t = Getattr(n, "type");
    v = Getattr(n, "value");
    d = Getattr(n, "decl");

    code = Getattr(n, "code");

    Append(typelist, t);
    Append(typelist, d);
    Append(patchlist, v);
    Append(cpatchlist, code);

    if (Getattr(n, "conversion_operator")) {
      /* conversion operator "name" and "sym:name" attributes are unusual as they contain c++ types, so treat as code for patching */
      Append(cpatchlist, Getattr(n, "name"));
      if (Getattr(n, "sym:name")) {
	Append(cpatchlist, Getattr(n, "sym:name"));
      }
    }
    if (Strstr(Getattr(n, "storage"), "friend")) {
      String *symname = Getattr(n, "sym:name");
      if (symname) {
	String *stripped_name = SwigType_templateprefix(symname);
	Setattr(n, "sym:name", stripped_name);
	Delete(stripped_name);
      }
      Append(typelist, Getattr(n, "name"));
    }

    expand_parms(n, "parms", unexpanded_variadic_parm, expanded_variadic_parms, cpatchlist, typelist, 0);
    expand_parms(n, "throws", unexpanded_variadic_parm, expanded_variadic_parms, cpatchlist, typelist, 0);

  } else if (Equal(nodeType, "class")) {
    /* Patch base classes */
    {
      int b = 0;
      for (b = 0; b < 3; ++b) {
	List *bases = Getattr(n, baselists[b]);
	if (bases) {
	  int i;
	  int ilen = Len(bases);
	  for (i = 0; i < ilen; i++) {
	    String *name = Copy(Getitem(bases, i));
	    if (SwigType_isvariadic(name)) {
	      Parm *parm = NewParmWithoutFileLineInfo(name, 0);
	      Node *temp_parm_node = NewHash();
	      Setattr(temp_parm_node, "variadicbaseparms", parm);
	      assert(i == ilen - 1);
	      Delitem(bases, i);
	      expand_variadic_parms(temp_parm_node, "variadicbaseparms", unexpanded_variadic_parm, expanded_variadic_parms);
	      {
		Parm *vp = Getattr(temp_parm_node, "variadicbaseparms");
		while (vp) {
		  String *name = Copy(Getattr(vp, "type"));
		  Append(bases, name);
		  Append(typelist, name);
		  vp = nextSibling(vp);
		}
	      }
	      Delete(temp_parm_node);
	    } else {
	      Setitem(bases, i, name);
	      Append(typelist, name);
	    }
	  }
	}
      }
    }
    /* Patch children */
    {
      Node *cn = firstChild(n);
      while (cn) {
	cparse_template_expand(templnode, cn, tname, rname, templateargs, patchlist, typelist, cpatchlist, unexpanded_variadic_parm, expanded_variadic_parms);
	cn = nextSibling(cn);
      }
    }
  } else if (Equal(nodeType, "classforward")) {
    /* Nothing to expand */
  } else if (Equal(nodeType, "constructor")) {
    if (!(Getattr(n, "templatetype"))) {
      String *symname = Getattr(n, "sym:name");
      String *name;
      if (symname) {
	String *stripped_name = SwigType_templateprefix(symname);
	if (Strstr(tname, stripped_name)) {
	  Replaceid(symname, stripped_name, tname);
	}
	Delete(stripped_name);
      }
      name = Getattr(n, "sym:name");
      if (name) {
	if (strchr(Char(name), '<')) {
	  Clear(name);
	  Append(name, rname);
	} else {
	  String *tmp = Copy(name);
	  Replace(tmp, tname, rname, DOH_REPLACE_ANY);
	  Clear(name);
	  Append(name, tmp);
	  Delete(tmp);
	}
      }
    }
    Append(cpatchlist, Getattr(n, "code"));
    Append(typelist, Getattr(n, "decl"));
    expand_parms(n, "parms", unexpanded_variadic_parm, expanded_variadic_parms, cpatchlist, typelist, 0);
    expand_parms(n, "throws", unexpanded_variadic_parm, expanded_variadic_parms, cpatchlist, typelist, 0);
  } else if (Equal(nodeType, "destructor")) {
    /* We only need to patch the dtor of the template itself, not the destructors of any nested classes, so check that the parent of this node is the root
     * template node, with the special exception for %extend which adds its methods under an intermediate node. */
    Node* parent = parentNode(n);
    if (parent == templnode || (parentNode(parent) == templnode && Equal(nodeType(parent), "extend"))) {
      String *symname = Getattr(n, "sym:name");
      if (symname)
	Replace(symname, tname, rname, DOH_REPLACE_ANY);
      Append(cpatchlist, Getattr(n, "code"));
    }
  } else if (Equal(nodeType, "using")) {
    String *name = Getattr(n, "name");
    String *uname = Getattr(n, "uname");
    if (uname && strchr(Char(uname), '<')) {
      Append(patchlist, uname);
    }
    if (!(Getattr(n, "templatetype"))) {
      /* Copied from handling "constructor" .. not sure if all this is needed */
      String *symname;
      String *stripped_name = SwigType_templateprefix(name);
      if (Strstr(tname, stripped_name)) {
	Replaceid(name, stripped_name, tname);
      }
      Delete(stripped_name);
      symname = Getattr(n, "sym:name");
      if (symname) {
	stripped_name = SwigType_templateprefix(symname);
	if (Strstr(tname, stripped_name)) {
	  Replaceid(symname, stripped_name, tname);
	}
	Delete(stripped_name);
      }
      if (strchr(Char(name), '<')) {
	Append(patchlist, Getattr(n, "name"));
      }
      name = Getattr(n, "sym:name");
      if (name) {
	if (strchr(Char(name), '<')) {
	  Clear(name);
	  Append(name, rname);
	} else {
	  String *tmp = Copy(name);
	  Replace(tmp, tname, rname, DOH_REPLACE_ANY);
	  Clear(name);
	  Append(name, tmp);
	  Delete(tmp);
	}
      }
    }

    if (Getattr(n, "namespace")) {
      /* Namespace link.   This is nasty.  Is other namespace defined? */

    }
  } else {
    /* Look for obvious parameters */
    Node *cn;
    Append(cpatchlist, Getattr(n, "code"));
    Append(typelist, Getattr(n, "type"));
    Append(typelist, Getattr(n, "decl"));
    expand_parms(n, "parms", unexpanded_variadic_parm, expanded_variadic_parms, cpatchlist, typelist, 0);
    expand_parms(n, "kwargs", unexpanded_variadic_parm, expanded_variadic_parms, cpatchlist, typelist, 0);
    expand_parms(n, "pattern", unexpanded_variadic_parm, expanded_variadic_parms, cpatchlist, typelist, 1);
    expand_parms(n, "throws", unexpanded_variadic_parm, expanded_variadic_parms, cpatchlist, typelist, 0);
    cn = firstChild(n);
    while (cn) {
      cparse_template_expand(templnode, cn, tname, rname, templateargs, patchlist, typelist, cpatchlist, unexpanded_variadic_parm, expanded_variadic_parms);
      cn = nextSibling(cn);
    }
  }
}

/* -----------------------------------------------------------------------------
 * cparse_fix_function_decl()
 *
 * Move the prefix of the "type" attribute (excluding any trailing qualifier)
 * to the end of the "decl" attribute.
 * Examples:
 *   decl="f().", type="p.q(const).char"  => decl="f().p.", type="q(const).char"
 *   decl="f().p.", type="p.SomeClass"    => decl="f().p.p.", type="SomeClass"
 *   decl="f().", type="r.q(const).p.int" => decl="f().r.q(const).p.", type="int"
 * ----------------------------------------------------------------------------- */

static void cparse_fix_function_decl(String *name, SwigType *decl, SwigType *type) {
  String *prefix;
  int prefixLen;
  SwigType *last;

  /* The type's prefix is what potentially has to be moved to the end of 'decl' */
  prefix = SwigType_prefix(type);

  /* First some parts (qualifier and array) have to be removed from prefix
     in order to remain in the 'type' attribute. */
  last = SwigType_last(prefix);
  while (last) {
    if (SwigType_isqualifier(last) || SwigType_isarray(last)) {
      /* Keep this part in the 'type' */
      Delslice(prefix, Len(prefix) - Len(last), DOH_END);
      Delete(last);
      last = SwigType_last(prefix);
    } else {
      /* Done with processing prefix */
      Delete(last);
      last = 0;
    }
  }

  /* Transfer prefix from the 'type' to the 'decl' attribute */
  prefixLen = Len(prefix);
  if (prefixLen > 0) {
    Append(decl, prefix);
    Delslice(type, 0, prefixLen);
    if (template_debug) {
      Printf(stdout, "    change function '%s' to type='%s', decl='%s'\n", name, type, decl);
    }
  }

  Delete(prefix);
}

/* -----------------------------------------------------------------------------
 * cparse_postprocess_expanded_template()
 *
 * This function postprocesses the given node after template expansion.
 * Currently the only task to perform is fixing function decl and type attributes.
 * ----------------------------------------------------------------------------- */

static void cparse_postprocess_expanded_template(Node *n) {
  String *nodeType;
  if (!n)
    return;
  nodeType = nodeType(n);
  if (Getattr(n, "error"))
    return;

  if (Equal(nodeType, "cdecl")) {
    /* A simple C declaration */
    SwigType *d = Getattr(n, "decl");
    if (d && SwigType_isfunction(d)) {
      /* A function node */
      SwigType *t = Getattr(n, "type");
      if (t) {
	String *name = Getattr(n, "name");
	cparse_fix_function_decl(name, d, t);
      }
    }
  } else {
    /* Look for any children */
    Node *cn = firstChild(n);
    while (cn) {
      cparse_postprocess_expanded_template(cn);
      cn = nextSibling(cn);
    }
  }
}

/* -----------------------------------------------------------------------------
 * partial_arg()
 *
 * Return a parameter with type that matches the specialized template argument.
 * If the input has no partial type, the name is not set in the returned parameter.
 *
 * type: an instantiated template parameter type, for example: Vect<(int)>
 * partialtype: type from specialized template where parameter name has been
 * replaced by a $ variable, for example: Vect<($1)>
 *
 * Returns a parameter of type 'int' and name $1 for the two example parameters above.
 * ----------------------------------------------------------------------------- */

static Parm *partial_arg(const SwigType *type, const SwigType *partialtype) {
  SwigType *parmtype;
  String *parmname = 0;
  const char *cp = Char(partialtype);
  const char *c = strchr(cp, '$');

  if (c) {
    int suffix_length;
    int prefix_length = (int)(c - cp);
    int type_length = Len(type);
    const char *suffix = c;
    String *prefix = NewStringWithSize(cp, prefix_length);
    while (++suffix) {
      if (!isdigit((int)*suffix))
	break;
    }
    parmname = NewStringWithSize(c, (int)(suffix - c)); /* $1, $2 etc */
    suffix_length = (int)strlen(suffix);
    assert(Strstr(type, prefix) == Char(type)); /* check that the start of both types match */
    assert(strcmp(Char(type) + type_length - suffix_length, suffix) == 0); /* check that the end of both types match */
    parmtype = NewStringWithSize(Char(type) + prefix_length, type_length - suffix_length - prefix_length);
    Delete(prefix);
  } else {
    parmtype = Copy(type);
  }
  return NewParmWithoutFileLineInfo(parmtype, parmname);
}

/* -----------------------------------------------------------------------------
 * Swig_cparse_template_expand()
 * ----------------------------------------------------------------------------- */

int Swig_cparse_template_expand(Node *n, String *rname, ParmList *tparms, Symtab *tscope) {
  List *patchlist, *cpatchlist, *typelist;
  String *templateargs;
  String *tname;
  String *name_with_templateargs = 0;
  String *tbase;
  Parm *unexpanded_variadic_parm = 0;
  ParmList *expanded_variadic_parms = 0;
  ParmList *templateparms = Getattr(n, "templateparms");
  ParmList *templateparmsraw = 0;
  patchlist = NewList();  /* List of String * ("name" and "value" attributes) */
  cpatchlist = NewList(); /* List of String * (code) */
  typelist = NewList();   /* List of SwigType * types */

  templateargs = NewStringEmpty();
  SwigType_add_template(templateargs, tparms);

  tname = Copy(Getattr(n, "name"));
  tbase = Swig_scopename_last(tname);

  if (Getattr(n, "partialargs")) {
    /* Partial specialization */
    Parm *p, *tp;
    ParmList *ptargs = SwigType_function_parms(Getattr(n, "partialargs"), n);
    p = ptargs;
    tp = tparms;
    /* Adjust templateparms so that the type is expanded, eg typename => int */
    while (p && tp) {
      SwigType *ptype;
      SwigType *tptype;
      SwigType *partial_type;
      ptype = Getattr(p, "type");
      tptype = Getattr(tp, "type");
      if (ptype && tptype) {
	SwigType *ty = Swig_symbol_typedef_reduce(tptype, tscope);
	Parm *partial_parm = partial_arg(ty, ptype);
	String *partial_name = Getattr(partial_parm, "name");
	partial_type = Copy(Getattr(partial_parm, "type"));
	/*      Printf(stdout,"partial '%s' '%s'  ---> '%s'\n", tptype, ptype, partial_type); */
	if (partial_name && strchr(Char(partial_name), '$') == Char(partial_name)) {
	  int index = atoi(Char(partial_name) + 1) - 1;
	  Parm *parm;
	  assert(index >= 0);
	  parm = ParmList_nth_parm(templateparms, index);
	  assert(parm);
	  if (parm) {
	    Setattr(parm, "type", partial_type);
	  }
	}
	Delete(partial_parm);
	Delete(partial_type);
	Delete(ty);
      }
      p = nextSibling(p);
      tp = nextSibling(tp);
    }
    Delete(ptargs);
  } else {
    Setattr(n, "templateparmsraw", Getattr(n, "templateparms"));
    templateparms = CopyParmList(tparms);
    Setattr(n, "templateparms", templateparms);
  }

  /* TODO: variadic parms for partially specialized templates */
  templateparmsraw = Getattr(n, "templateparmsraw");
  unexpanded_variadic_parm = ParmList_variadic_parm(templateparmsraw);
  if (unexpanded_variadic_parm)
    expanded_variadic_parms = ParmList_nth_parm(templateparms, ParmList_len(templateparmsraw) - 1);

  /*  Printf(stdout,"targs = '%s'\n", templateargs);
     Printf(stdout,"rname = '%s'\n", rname);
     Printf(stdout,"tname = '%s'\n", tname);  */
  cparse_template_expand(n, n, tname, rname, templateargs, patchlist, typelist, cpatchlist, unexpanded_variadic_parm, expanded_variadic_parms);

  /* Set the name */
  {
    String *name = Getattr(n, "name");
    if (name) {
      String *nodeType = nodeType(n);
      name_with_templateargs = NewStringf("%s%s", name, templateargs);
      if (!(Equal(nodeType, "constructor") || Equal(nodeType, "destructor"))) {
	Setattr(n, "name", name_with_templateargs);
      }
    }
  }

  /* Patch all of the types */
  {
    Parm *tp = Getattr(n, "templateparms");
    /*    Printf(stdout,"%s\n", ParmList_str_defaultargs(tp)); */

    if (tp) {
      Symtab *tsdecl = Getattr(n, "sym:symtab");
      String *tsname = Getattr(n, "sym:name");
      while (tp) {
	String *name, *value, *valuestr, *tmp, *tmpr;
	int sz, i;
	String *dvalue = 0;
	String *qvalue = 0;

	name = Getattr(tp, "name");
	value = Getattr(tp, "value");

	if (name) {
	  if (!value)
	    value = Getattr(tp, "type");
	  qvalue = Swig_symbol_typedef_reduce(value, tsdecl);
	  dvalue = Swig_symbol_type_qualify(qvalue, tsdecl);
	  if (SwigType_istemplate(dvalue)) {
	    String *ty = Swig_symbol_template_deftype(dvalue, tscope);
	    Delete(dvalue);
	    dvalue = ty;
	  }

	  assert(dvalue);
	  valuestr = SwigType_str(dvalue, 0);
	  sz = Len(patchlist);
	  for (i = 0; i < sz; i++) {
	    /* Patch String or SwigType with SwigType, eg T => int in Foo<(T)>, or TT => Hello<(int)> in X<(TT)>::meth */
	    String *s = Getitem(patchlist, i);
	    Replace(s, name, dvalue, DOH_REPLACE_ID);
	  }

	  sz = Len(typelist);
	  for (i = 0; i < sz; i++) {
	    SwigType *s = Getitem(typelist, i);
	    Node *tynode;
	    String *tyname;

	    SwigType_variadic_replace(s, unexpanded_variadic_parm, expanded_variadic_parms);

	    /*
	      The approach of 'trivially' replacing template arguments is kind of fragile.
	      In particular if types with similar name in different namespaces appear.
	      We will not replace template args if a type/class exists with the same
	      name which is not a template.
	    */
	    tynode = Swig_symbol_clookup(s, 0);
	    tyname  = tynode ? Getattr(tynode, "sym:name") : 0;
	    /*
	    Printf(stdout, "  replacing %s with %s to %s or %s to %s\n", s, name, dvalue, tbase, name_with_templateargs);
	    Printf(stdout, "    %d %s to %s\n", tp == unexpanded_variadic_parm, name, ParmList_str_defaultargs(expanded_variadic_parms));
	    */
	    if (!tyname || !tsname || !Equal(tyname, tsname) || Getattr(tynode, "templatetype")) {
	      SwigType_typename_replace(s, name, dvalue);
	      SwigType_typename_replace(s, tbase, name_with_templateargs);
	    }
	  }

	  tmp = NewStringf("#%s", name);
	  tmpr = NewStringf("\"%s\"", valuestr);

	  sz = Len(cpatchlist);
	  for (i = 0; i < sz; i++) {
	    /* Patch String with C++ String type, eg T => int in Foo< T >, or TT => Hello< int > in X< TT >::meth */
	    String *s = Getitem(cpatchlist, i);
	    /* Stringising that ought to be done in the preprocessor really, eg #T => "int" */
	    Replace(s, tmp, tmpr, DOH_REPLACE_ID);
	    Replace(s, name, valuestr, DOH_REPLACE_ID);
	  }
	  Delete(tmp);
	  Delete(tmpr);
	  Delete(valuestr);
	  Delete(dvalue);
	  Delete(qvalue);
	}
	tp = nextSibling(tp);
      }
    } else {
      /* No template parameters at all.  This could be a specialization */
      int i, sz;
      sz = Len(typelist);
      for (i = 0; i < sz; i++) {
	String *s = Getitem(typelist, i);
	SwigType_variadic_replace(s, unexpanded_variadic_parm, expanded_variadic_parms);
	SwigType_typename_replace(s, tbase, name_with_templateargs);
      }
    }
  }
  cparse_postprocess_expanded_template(n);

  /* Patch bases */
  {
    List *bases = Getattr(n, "baselist");
    if (bases) {
      Iterator b;
      for (b = First(bases); b.item; b = Next(b)) {
	String *qn = Swig_symbol_type_qualify(b.item, tscope);
	Clear(b.item);
	Append(b.item, qn);
	Delete(qn);
      }
    }
  }
  Delete(name_with_templateargs);
  Delete(patchlist);
  Delete(cpatchlist);
  Delete(typelist);
  Delete(tbase);
  Delete(tname);
  Delete(templateargs);

  return 0;
}

typedef enum { ExactNoMatch = -2, PartiallySpecializedNoMatch = -1, PartiallySpecializedMatch = 1, ExactMatch = 2 } EMatch;

/* -----------------------------------------------------------------------------
 * is_exact_partial_type()
 *
 * Return 1 if parm matches $1, $2, $3 etc exactly without any other prefixes or
 * suffixes. Return 0 otherwise.
 * ----------------------------------------------------------------------------- */

static int is_exact_partial_type(const SwigType *type) {
  const char *c = Char(type);
  int is_exact = 0;
  if (*c == '$' && isdigit((int)*(c + 1))) {
    const char *suffix = c + 1;
    while (++suffix) {
      if (!isdigit((int)*suffix))
	break;
    }
    is_exact = (*suffix == 0);
  }
  return is_exact;
}

/* -----------------------------------------------------------------------------
 * does_parm_match()
 *
 * Template argument deduction - check if a template type matches a partially specialized
 * template parameter type. Typedef reduce 'partial_parm_type' to see if it matches 'type'.
 *
 * type - template parameter type to match against
 * partial_parm_type - specialized template type, for example, r.$1 (partially specialized) or r.int (fully specialized)
 * tscope - template scope
 * specialization_priority - (output) contains a value indicating how good the match is
 *   (higher is better) only set if return is set to PartiallySpecializedMatch or ExactMatch.
 * ----------------------------------------------------------------------------- */

static EMatch does_parm_match(SwigType *type, SwigType *partial_parm_type, Symtab *tscope, int *specialization_priority) {
  static const int EXACT_MATCH_PRIORITY = 99999; /* a number bigger than the length of any conceivable type */
  static const int TEMPLATE_MATCH_PRIORITY = 1000; /* a priority added for each nested template, assumes max length of any prefix, such as r.q(const). , is less than this number */
  SwigType *ty = Swig_symbol_typedef_reduce(type, tscope);
  SwigType *pp_prefix = SwigType_prefix(partial_parm_type);
  int pp_len = Len(pp_prefix);
  EMatch match = Strchr(partial_parm_type, '$') == 0 ? ExactNoMatch : PartiallySpecializedNoMatch;
  *specialization_priority = -1;

  if (Equal(ty, partial_parm_type)) {
    match = ExactMatch;
    *specialization_priority = EXACT_MATCH_PRIORITY; /* exact matches always take precedence */
  } else if (match == PartiallySpecializedNoMatch) {
    if ((pp_len > 0 && Strncmp(ty, pp_prefix, pp_len) == 0)) {
      /*
	 Type starts with pp_prefix, so it is a partial specialization type match, for example,
	 all of the following could match the type in the %template:
	   template <typename T> struct XX {};
	   template <typename T> struct XX<T &> {};         // r.$1
	   template <typename T> struct XX<T const&> {};    // r.q(const).$1
	   template <typename T> struct XX<T *const&> {};   // r.q(const).p.$1
	   %template(XXX) XX<int *const&>;                  // r.q(const).p.int
	 where type="r.q(const).p.int" will match either of pp_prefix="r.", pp_prefix="r.q(const)." pp_prefix="r.q(const).p."
      */
      match = PartiallySpecializedMatch;
      *specialization_priority = pp_len;
    } else if (pp_len == 0 && is_exact_partial_type(partial_parm_type)) {
      /*
	 Type without a prefix match, as in $1 for int
	   template <typename T, typename U> struct XX {};
	   template <typename T, typename U> struct XX<T, U &> {};  // $1,r.$2
	   %template(XXX) XX<int, double&>;                         // int,r.double
       */
      match = PartiallySpecializedMatch;
      *specialization_priority = pp_len;
    } else {
      /*
	Check for template types that are templates such as
	  template<typename V> struct Vect {};
	  template<typename T> class XX {};
	  template<class TT> class XX<Vect<TT>> {};
	  %template(XXVectInt) XX<Vect<int>>;
	matches type="Vect<(int)>" and partial_parm_type="Vect<($1)>"
       */
      if (SwigType_istemplate(partial_parm_type) && SwigType_istemplate(ty)) {

	SwigType *qt = Swig_symbol_typedef_reduce(ty, tscope);
	String *tsuffix = SwigType_templatesuffix(qt);

	SwigType *pp_qt = Swig_symbol_typedef_reduce(partial_parm_type, tscope);
	String *pp_tsuffix = SwigType_templatesuffix(pp_qt);

	if (Equal(tsuffix, pp_tsuffix) && Len(tsuffix) == 0) {
	  String *tprefix = SwigType_templateprefix(qt);
	  String *qprefix = SwigType_typedef_qualified(tprefix);

	  String *pp_tprefix = SwigType_templateprefix(pp_qt);
	  String *pp_qprefix = SwigType_typedef_qualified(pp_tprefix);

	  if (Equal(qprefix, pp_qprefix)) {
	    String *templateargs = SwigType_templateargs(qt);
	    List *parms = SwigType_parmlist(templateargs);
	    Iterator pi = First(parms);
	    Parm *p = pi.item;

	    String *pp_templateargs = SwigType_templateargs(pp_qt);
	    List *pp_parms = SwigType_parmlist(pp_templateargs);
	    Iterator pp_pi = First(pp_parms);
	    Parm *pp = pp_pi.item;

	    if (p && pp) {
	      /* Implementation is limited to matching single parameter templates only for now */
	      int priority;
	      match = does_parm_match(p, pp, tscope, &priority);
	      if (match <= PartiallySpecializedNoMatch) {
		*specialization_priority = priority;
	      } else {
		*specialization_priority = priority + TEMPLATE_MATCH_PRIORITY;
	      }
	    }

	    Delete(pp_parms);
	    Delete(pp_templateargs);
	    Delete(parms);
	    Delete(templateargs);
	  }

	  Delete(pp_qprefix);
	  Delete(pp_tprefix);
	  Delete(qprefix);
	  Delete(tprefix);
	}

	Delete(pp_tsuffix);
	Delete(pp_qt);
	Delete(tsuffix);
	Delete(qt);
      }
    }
  }
  /*
  Printf(stdout, "      does_parm_match %2d %5d [%s] [%s]\n", match, *specialization_priority, type, partial_parm_type);
  */
  Delete(ty);
  return match;
}

/* -----------------------------------------------------------------------------
 * template_locate()
 *
 * Search for a template that matches name with given parameters.
 * ----------------------------------------------------------------------------- */

static Node *template_locate(String *name, Parm *instantiated_parms, String *symname, Symtab *tscope) {
  Node *n = 0;
  String *tname = 0;
  Node *templ;
  Symtab *primary_scope = 0;
  List *possiblepartials = 0;
  Parm *p;
  Parm *parms = 0;
  Parm *targs;
  ParmList *expandedparms;
  int *priorities_matrix = 0;
  int max_possible_partials = 0;
  int posslen = 0;

  if (template_debug) {
    tname = Copy(name);
    SwigType_add_template(tname, instantiated_parms);
    Printf(stdout, "\n");
    if (symname)
      Swig_diagnostic(cparse_file, cparse_line, "Template debug: Searching for match to: '%s' for instantiation of template named '%s'\n", tname, symname);
    else
      Swig_diagnostic(cparse_file, cparse_line, "Template debug: Searching for match to: '%s' for instantiation of empty template\n", tname);
    Delete(tname);
    tname = 0;
  }

  /* Search for primary (unspecialized) template */
  templ = Swig_symbol_clookup(name, 0);

  if (templ) {
    /* TODO: check that this is not a specialization (might be a user error specializing a template before a primary template), but note https://stackoverflow.com/questions/9757642/wrapping-specialised-c-template-class-with-swig */
    if (template_debug) {
      Printf(stdout, "    found primary template <%s> '%s'\n", ParmList_str_defaultargs(Getattr(templ, "templateparms")), Getattr(templ, "name"));
    }

    tname = Copy(name);
    parms = CopyParmList(instantiated_parms);

    /* All template specializations must be in the primary template's scope, store the symbol table for this scope for specialization lookups */
    primary_scope = Getattr(templ, "sym:symtab");

    /* Add default values from primary template */
    targs = Getattr(templ, "templateparms");
    expandedparms = Swig_symbol_template_defargs(parms, targs, tscope, primary_scope);

    /* Qualify template parameters */
    p = expandedparms;
    while (p) {
      SwigType *ty = Getattr(p, "type");
      if (ty) {
	SwigType *nt = Swig_symbol_type_qualify(ty, tscope);
	Setattr(p, "type", nt);
	Delete(nt);
      }
      p = nextSibling(p);
    }
    SwigType_add_template(tname, expandedparms);

    /* Search for an explicit (exact) specialization. Example: template<> class name<int> { ... } */
    {
      if (template_debug) {
	Printf(stdout, "    searching for : '%s' (explicit specialization)\n", tname);
      }
      n = Swig_symbol_clookup_local(tname, primary_scope);
      if (!n) {
	SwigType *rname = Swig_symbol_typedef_reduce(tname, tscope);
	if (!Equal(rname, tname)) {
	  if (template_debug) {
	    Printf(stdout, "    searching for : '%s' (explicit specialization with typedef reduction)\n", rname);
	  }
	  n = Swig_symbol_clookup_local(rname, primary_scope);
	}
	Delete(rname);
      }
      if (n) {
	Node *tn;
	String *nodeType = nodeType(n);
	if (Equal(nodeType, "template")) {
	  if (template_debug) {
	    Printf(stdout, "    explicit specialization found: '%s'\n", Getattr(n, "name"));
	  }
	  goto success;
	}
	tn = Getattr(n, "template");
	if (tn) {
	  /* Previously wrapped by a template instantiation */
	  Node *previous_named_instantiation = GetFlag(n, "hidden") ? Getattr(n, "csym:nextSibling") : n; /* "hidden" is set when "sym:name" is a __dummy_ name */
	  if (!symname) {
	    /* Quietly ignore empty template instantiations if there is a previous (empty or non-empty) template instantiation */
	    if (template_debug) {
	      if (previous_named_instantiation)
		Printf(stdout, "    previous instantiation with name '%s' found: '%s' - duplicate empty template instantiation ignored\n", Getattr(previous_named_instantiation, "sym:name"), Getattr(n, "name"));
	      else
		Printf(stdout, "    previous empty template instantiation found: '%s' - duplicate empty template instantiation ignored\n", Getattr(n, "name"));
	    }
	    return 0;
	  }
	  /* Accept a second instantiation only if previous template instantiation is empty */
	  if (previous_named_instantiation) {
	    String *previous_name = Getattr(previous_named_instantiation, "name");
	    String *previous_symname = Getattr(previous_named_instantiation, "sym:name");
	    String *unprocessed_tname = Copy(name);
	    SwigType_add_template(unprocessed_tname, instantiated_parms);

	    if (template_debug)
	      Printf(stdout, "    previous instantiation with name '%s' found: '%s' - duplicate instantiation ignored\n", previous_symname, Getattr(n, "name"));
	    SWIG_WARN_NODE_BEGIN(n);
	    Swig_warning(WARN_TYPE_REDEFINED, cparse_file, cparse_line, "Duplicate template instantiation of '%s' with name '%s' ignored,\n", SwigType_namestr(unprocessed_tname), symname);
	    Swig_warning(WARN_TYPE_REDEFINED, Getfile(n), Getline(n), "previous instantiation of '%s' with name '%s'.\n", SwigType_namestr(previous_name), previous_symname);
	    SWIG_WARN_NODE_END(n);

	    Delete(unprocessed_tname);
	    return 0;
	  }
	  if (template_debug)
	    Printf(stdout, "    previous empty template instantiation found: '%s' - using as duplicate instantiation overrides empty template instantiation\n", Getattr(n, "name"));
	  n = tn;
	  goto success;
	}
	Swig_error(cparse_file, cparse_line, "'%s' is not defined as a template. (%s)\n", name, nodeType(n));
	Delete(tname);
	Delete(parms);
	return 0;	  /* Found a match, but it's not a template of any kind. */
      }
    }

    /* Search for partial specializations.
     * Example: template<typename T> class name<T *> { ... } 

     * There are 3 types of template arguments:
     * (1) Template type arguments
     * (2) Template non type arguments
     * (3) Template template arguments
     * only (1) is really supported for partial specializations
     */

    /* Rank each template parameter against the desired template parameters then build a matrix of best matches */
    possiblepartials = NewList();
    {
      List *partials;

      partials = Getattr(templ, "partials"); /* note that these partial specializations do not include explicit specializations */
      if (partials) {
	Iterator pi;
	int parms_len = ParmList_len(parms); /* max parameters including defaulted parameters from primary template (ie max parameters) */
	int *priorities_row;
	max_possible_partials = Len(partials);
	priorities_matrix = (int *)Malloc(sizeof(int) * max_possible_partials * parms_len); /* slightly wasteful allocation for max possible matches */
	priorities_row = priorities_matrix;
	for (pi = First(partials); pi.item; pi = Next(pi)) {
	  Parm *p = parms;
	  int i = 1;
	  Parm *partialparms = Getattr(pi.item, "partialparms");
	  Parm *pp = partialparms;
	  String *templcsymname = Getattr(pi.item, "templcsymname");
	  if (template_debug) {
	    Printf(stdout, "    checking match: '%s' (partial specialization)\n", templcsymname);
	  }
	  if (ParmList_len(partialparms) == parms_len) {
	    int all_parameters_match = 1;
	    while (p && pp) {
	      SwigType *t;
	      t = Getattr(p, "type");
	      if (!t)
		t = Getattr(p, "value");
	      if (t) {
		EMatch match = does_parm_match(t, Getattr(pp, "type"), tscope, priorities_row + i - 1);
		if (match < (int)PartiallySpecializedMatch) {
		  all_parameters_match = 0;
		  break;
		}
	      }
	      i++;
	      p = nextSibling(p);
	      pp = nextSibling(pp);
	    }
	    if (all_parameters_match) {
	      Append(possiblepartials, pi.item);
	      priorities_row += parms_len;
	    }
	  }
	}
      }
    }

    posslen = Len(possiblepartials);
    if (template_debug) {
      int i;
      if (posslen == 0)
	Printf(stdout, "    matched partials: NONE\n");
      else if (posslen == 1)
	Printf(stdout, "    chosen partial: '%s'\n", Getattr(Getitem(possiblepartials, 0), "templcsymname"));
      else {
	Printf(stdout, "    possibly matched partials:\n");
	for (i = 0; i < posslen; i++) {
	  Printf(stdout, "      '%s'\n", Getattr(Getitem(possiblepartials, i), "templcsymname"));
	}
      }
    }

    if (posslen > 1) {
      /* Now go through all the possibly matched partial specialization templates and look for a non-ambiguous match.
       * Exact matches rank the highest and deduced parameters are ranked by how specialized they are, eg looking for
       * a match to const int *, the following rank (highest to lowest):
       *   const int * (exact match)
       *   const T *
       *   T *
       *   T
       *
       *   An ambiguous example when attempting to match as either specialization could match: %template() X<int *, double *>;
       *   template<typename T1, typename T2> class X {};  // primary template
       *   template<typename T1> class X<T1, double *> {}; // specialization (1)
       *   template<typename T2> class X<int *, T2> {};    // specialization (2)
       *
       */
      if (template_debug) {
	int row, col;
	int parms_len = ParmList_len(parms);
	Printf(stdout, "      parameter priorities matrix (%d parms):\n", parms_len);
	for (row = 0; row < posslen; row++) {
	  int *priorities_row = priorities_matrix + row*parms_len;
	  Printf(stdout, "        ");
	  for (col = 0; col < parms_len; col++) {
	    Printf(stdout, "%5d ", priorities_row[col]);
	  }
	  Printf(stdout, "\n");
	}
      }
      {
	int row, col;
	int parms_len = ParmList_len(parms);
	/* Printf(stdout, "      parameter priorities inverse matrix (%d parms):\n", parms_len); */
	for (col = 0; col < parms_len; col++) {
	  int *priorities_col = priorities_matrix + col;
	  int maxpriority = -1;
	  /* 
	     Printf(stdout, "max_possible_partials: %d col:%d\n", max_possible_partials, col);
	     Printf(stdout, "        ");
	     */
	  /* determine the highest rank for this nth parameter */
	  for (row = 0; row < posslen; row++) {
	    int *element_ptr = priorities_col + row*parms_len;
	    int priority = *element_ptr;
	    if (priority > maxpriority)
	      maxpriority = priority;
	    /* Printf(stdout, "%5d ", priority); */
	  }
	  /* Printf(stdout, "\n"); */
	  /* flag all the parameters which equal the highest rank */
	  for (row = 0; row < posslen; row++) {
	    int *element_ptr = priorities_col + row*parms_len;
	    int priority = *element_ptr;
	    *element_ptr = (priority >= maxpriority) ? 1 : 0;
	  }
	}
      }
      {
	int row, col;
	int parms_len = ParmList_len(parms);
	Iterator pi = First(possiblepartials);
	Node *chosenpartials = NewList();
	if (template_debug)
	  Printf(stdout, "      priority flags matrix:\n");
	for (row = 0; row < posslen; row++) {
	  int *priorities_row = priorities_matrix + row*parms_len;
	  int highest_count = 0; /* count of highest priority parameters */
	  for (col = 0; col < parms_len; col++) {
	    highest_count += priorities_row[col];
	  }
	  if (template_debug) {
	    Printf(stdout, "        ");
	    for (col = 0; col < parms_len; col++) {
	      Printf(stdout, "%5d ", priorities_row[col]);
	    }
	    Printf(stdout, "\n");
	  }
	  if (highest_count == parms_len) {
	    Append(chosenpartials, pi.item);
	  }
	  pi = Next(pi);
	}
	if (Len(chosenpartials) > 0) {
	  /* one or more best match found */
	  Delete(possiblepartials);
	  possiblepartials = chosenpartials;
	  posslen = Len(possiblepartials);
	} else {
	  /* no best match found */
	  Delete(chosenpartials);
	}
      }
    }

    if (posslen > 0) {
      String *s = Getattr(Getitem(possiblepartials, 0), "templcsymname");
      n = Swig_symbol_clookup_local(s, primary_scope);
      if (posslen > 1) {
	int i;
	if (n) {
	  Swig_warning(WARN_PARSE_TEMPLATE_AMBIG, cparse_file, cparse_line, "Instantiation of template '%s' is ambiguous,\n", SwigType_namestr(tname));
	  Swig_warning(WARN_PARSE_TEMPLATE_AMBIG, Getfile(n), Getline(n), "  instantiation '%s' used,\n", SwigType_namestr(Getattr(n, "name")));
	}
	for (i = 1; i < posslen; i++) {
	  String *templcsymname = Getattr(Getitem(possiblepartials, i), "templcsymname");
	  Node *ignored_node = Swig_symbol_clookup_local(templcsymname, primary_scope);
	  assert(ignored_node);
	  Swig_warning(WARN_PARSE_TEMPLATE_AMBIG, Getfile(ignored_node), Getline(ignored_node), "  instantiation '%s' ignored.\n", SwigType_namestr(Getattr(ignored_node, "name")));
	}
      }
    }

    if (!n) {
      if (template_debug) {
	Printf(stdout, "    chosen primary template: '%s'\n", Getattr(templ, "name"));
      }
      n = templ;
    }
  } else {
    if (template_debug) {
      Printf(stdout, "    primary template not found\n");
    }
    /* Give up if primary (unspecialized) template not found as specializations will only exist if there is a primary template */
    n = 0;
  }

  if (!n) {
    Swig_error(cparse_file, cparse_line, "Template '%s' undefined.\n", name);
  } else if (n) {
    String *nodeType = nodeType(n);
    if (!Equal(nodeType, "template")) {
      Swig_error(cparse_file, cparse_line, "'%s' is not defined as a template. (%s)\n", name, nodeType);
      n = 0;
    }
  }
success:
  Delete(tname);
  Delete(possiblepartials);
  if ((template_debug) && (n)) {
    /*
    Printf(stdout, "Node: %p\n", n);
    Swig_print_node(n);
    */
    Printf(stdout, "    chosen template:'%s'\n", Getattr(n, "name"));
  }
  Delete(parms);
  Free(priorities_matrix);
  return n;
}


/* -----------------------------------------------------------------------------
 * Swig_cparse_template_locate()
 *
 * Search for a template that matches name with given parameters and mark it for instantiation.
 * For class templates marks the specialized template should there be one.
 * For function templates marks all the unspecialized templates even if specialized
 * templates exists.
 * ----------------------------------------------------------------------------- */

Node *Swig_cparse_template_locate(String *name, Parm *instantiated_parms, String *symname, Symtab *tscope) {
  Node *match = 0;
  Node *n = template_locate(name, instantiated_parms, symname, tscope); /* this function does what we want for class templates */

  if (n) {
    String *nodeType = nodeType(n);
    assert(Equal(nodeType, "template"));
    String *templatetype = Getattr(n, "templatetype");

    if (Equal(templatetype, "class") || Equal(templatetype, "classforward")) {
      Node *primary = Getattr(n, "primarytemplate");
      Parm *tparmsfound = Getattr(primary ? primary : n, "templateparms");
      int specialized = !tparmsfound; /* fully specialized (an explicit specialization) */
      int variadic = ParmList_variadic_parm(tparmsfound) != 0;
      match = n;
      if (!specialized) {
	if (!variadic && (ParmList_len(instantiated_parms) > ParmList_len(tparmsfound))) {
	  Swig_error(cparse_file, cparse_line, "Too many template parameters. Maximum of %d.\n", ParmList_len(tparmsfound));
	  match = 0;
	} else if (ParmList_len(instantiated_parms) < ParmList_numrequired(tparmsfound) - (variadic ? 1 : 0)) { /* Variadic parameter is optional */
	  Swig_error(cparse_file, cparse_line, "Not enough template parameters specified. Minimum of %d required.\n", (ParmList_numrequired(tparmsfound) - (variadic ? 1 : 0)) );
	  match = 0;
	}
      }
      if (match)
	SetFlag(n, "instantiate");
    } else {
      Node *firstn = 0;
      /* If not a class template we must have a function template.
         The template found is not necessarily the one we want when dealing with templated
         functions. We don't want any specialized function templates as they won't have
         the default parameters. Let's look for the unspecialized template. Also make sure
         the number of template parameters is correct as it is possible to overload a
         function template with different numbers of template parameters. */

      if (template_debug) {
	Printf(stdout, "    Not a class template, seeking all appropriate primary function templates\n");
      }

      firstn = Swig_symbol_clookup_local(name, 0);
      n = firstn;
      /* First look for all overloaded functions (non-variadic) template matches.
       * Looking for all template parameter matches only (not function parameter matches) 
       * as %template instantiation uses template parameters without any function parameters. */
      while (n) {
	if (Strcmp(nodeType(n), "template") == 0) {
	  Parm *tparmsfound = Getattr(n, "templateparms");
	  if (!ParmList_variadic_parm(tparmsfound)) {
	    if (ParmList_len(instantiated_parms) == ParmList_len(tparmsfound)) {
	      /* successful match */
	      if (template_debug) {
		Printf(stdout, "    found: template <%s> '%s' (%s)\n", ParmList_str_defaultargs(Getattr(n, "templateparms")), name, ParmList_str_defaultargs(Getattr(n, "parms")));
	      }
	      SetFlag(n, "instantiate");
	      if (!match)
		match = n; /* first match */
	    }
	  }
	}
	/* repeat to find all matches with correct number of templated parameters */
	n = Getattr(n, "sym:nextSibling");
      }

      /* Only consider variadic templates if there are no non-variadic template matches */
      if (!match) {
	n = firstn;
	while (n) {
	  if (Strcmp(nodeType(n), "template") == 0) {
	    Parm *tparmsfound = Getattr(n, "templateparms");
	    if (ParmList_variadic_parm(tparmsfound)) {
	      if (ParmList_len(instantiated_parms) >= ParmList_len(tparmsfound) - 1) {
		/* successful variadic match */
		if (template_debug) {
		  Printf(stdout, "    found: template <%s> '%s' (%s)\n", ParmList_str_defaultargs(Getattr(n, "templateparms")), name, ParmList_str_defaultargs(Getattr(n, "parms")));
		}
		SetFlag(n, "instantiate");
		if (!match)
		  match = n; /* first match */
	      }
	    }
	  }
	  /* repeat to find all matches with correct number of templated parameters */
	  n = Getattr(n, "sym:nextSibling");
	}
      }

      if (!match) {
	Swig_error(cparse_file, cparse_line, "No matching function template '%s' found.\n", name);
      }
    }
  }

  return match;
}

/* -----------------------------------------------------------------------------
 * merge_parameters()
 *
 * expanded_templateparms are the template parameters passed to %template.
 * This function adds missing parameter name and type attributes from the chosen
 * template (templateparms).
 *
 * Grab the parameter names from templateparms.
 * Non-type template parameters have no type information in expanded_templateparms.
 * Grab them from templateparms.
 *
 * Return 1 if there are variadic template parameters, 0 otherwise.
 * ----------------------------------------------------------------------------- */

static int merge_parameters(ParmList *expanded_templateparms, ParmList *templateparms) {
  Parm *p = expanded_templateparms;
  Parm *tp = templateparms;
  while (p && tp) {
    Setattr(p, "name", Getattr(tp, "name"));
    if (!Getattr(p, "type"))
      Setattr(p, "type", Getattr(tp, "type"));
    p = nextSibling(p);
    tp = nextSibling(tp);
  }
  return ParmList_variadic_parm(templateparms) ? 1 : 0;
}

/* -----------------------------------------------------------------------------
 * use_mark_defaults()
 *
 * Mark and use all the template parameters that are expanded from a default value
 * ----------------------------------------------------------------------------- */

static void use_mark_defaults(ParmList *defaults) {
  Parm *tp = defaults;
  while (tp) {
    Setattr(tp, "default", "1");
    Setattr(tp, "type", Getattr(tp, "value"));
    tp = nextSibling(tp);
  }
}

/* -----------------------------------------------------------------------------
 * use_mark_specialized_defaults()
 *
 * Modify extra defaulted parameters ready for adding to specialized template parameters list
 * ----------------------------------------------------------------------------- */

static void use_mark_specialized_defaults(ParmList *defaults) {
  Parm *tp = defaults;
  while (tp) {
    Setattr(tp, "default", "1");
    Setattr(tp, "type", Getattr(tp, "value"));
    Delattr(tp, "name");
    tp = nextSibling(tp);
  }
}

/* -----------------------------------------------------------------------------
 * expand_defaults()
 *
 * Replace parameter types in default argument values, example:
 * input:  int K,int T,class C=Less<(K)>
 * output: int K,int T,class C=Less<(int)>
 * ----------------------------------------------------------------------------- */

static void expand_defaults(ParmList *expanded_templateparms) {
  Parm *tp = expanded_templateparms;
  while (tp) {
    Parm *p = expanded_templateparms;
    String *tv = Getattr(tp, "value");
    if (!tv)
      tv = Getattr(tp, "type");
    while(p) {
      String *name = Getattr(p, "name");
      String *value = Getattr(p, "value");
      if (!value)
	value = Getattr(p, "type");
      if (name)
	Replaceid(tv, name, value);
      p = nextSibling(p);
    }
    tp = nextSibling(tp);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_cparse_template_parms_expand()
 *
 * instantiated_parms: template parameters passed to %template
 * primary: primary template node
 *
 * Expand the instantiated_parms and return a parameter list with default
 * arguments filled in where necessary.
 * ----------------------------------------------------------------------------- */

ParmList *Swig_cparse_template_parms_expand(ParmList *instantiated_parms, Node *primary, Node *templ) {
  ParmList *expanded_templateparms = CopyParmList(instantiated_parms);
  String *templatetype = Getattr(primary, "templatetype");

  if (Equal(templatetype, "class") || Equal(templatetype, "classforward")) {
    /* Class template */
    ParmList *templateparms = Getattr(primary, "templateparms");
    int variadic = merge_parameters(expanded_templateparms, templateparms);
    /* Add default arguments from primary template */
    if (!variadic) {
      ParmList *defaults_start = ParmList_nth_parm(templateparms, ParmList_len(instantiated_parms));
      if (defaults_start) {
	ParmList *defaults = CopyParmList(defaults_start);
	use_mark_defaults(defaults);
	expanded_templateparms = ParmList_join(expanded_templateparms, defaults);
	expand_defaults(expanded_templateparms);
      }
    }
  } else {
    /* Function template */
    /* TODO: Default template parameters support was only added in C++11 */
    ParmList *templateparms = Getattr(templ, "templateparms");
    merge_parameters(expanded_templateparms, templateparms);
  }

  return expanded_templateparms;
}

/* -----------------------------------------------------------------------------
 * Swig_cparse_template_partialargs_expand()
 *
 * partially_specialized_parms: partially specialized template parameters
 * primary: primary template node
 * templateparms: primary template parameters (providing the defaults)
 *
 * Expand the partially_specialized_parms and return a parameter list with default
 * arguments filled in where necessary.
 * ----------------------------------------------------------------------------- */

ParmList *Swig_cparse_template_partialargs_expand(ParmList *partially_specialized_parms, Node *primary, ParmList *templateparms) {
  ParmList *expanded_templateparms = CopyParmList(partially_specialized_parms);
  String *templatetype = Getattr(primary, "templatetype");

  if (Equal(templatetype, "class") || Equal(templatetype, "classforward")) {
    /* Class template */
    int variadic = ParmList_variadic_parm(templateparms) ? 1 : 0;
    /* Add default arguments from primary template */
    if (!variadic) {
      ParmList *defaults_start = ParmList_nth_parm(templateparms, ParmList_len(partially_specialized_parms));
      if (defaults_start) {
	ParmList *defaults = CopyParmList(defaults_start);
	use_mark_specialized_defaults(defaults);
	expanded_templateparms = ParmList_join(expanded_templateparms, defaults);
	expand_defaults(expanded_templateparms);
      }
    }
  } else {
    /* Function template */
    /* TODO: Default template parameters support was only added in C++11 */
  }

  return expanded_templateparms;
}
