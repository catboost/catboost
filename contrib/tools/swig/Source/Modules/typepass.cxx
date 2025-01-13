/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * typepass.cxx
 *
 * This module builds all of the internal type information by collecting
 * typedef declarations as well as registering classes, structures, and unions.
 * This information is needed to correctly handle shadow classes and other
 * advanced features.   This phase of compilation is also used to perform
 * type-expansion.  All types are fully qualified with namespace prefixes
 * and other information needed for compilation.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"

struct normal_node {
  Symtab *symtab;
  Hash *typescope;
  List *normallist;
  normal_node *next;
};

static normal_node *patch_list = 0;

/* Singleton class - all non-static methods in this class are private */
class TypePass:private Dispatcher {
  Node *inclass;
  Node *module;
  int importmode;
  String *nsname;
  String *nssymname;
  Hash *classhash;
  List *normalize;

  TypePass() :
    inclass(0),
    module(0),
    importmode(0),
    nsname(0),
    nssymname(0),
    classhash(0),
    normalize(0) {
  }

  /* Normalize a type. Replaces type with fully qualified version */
  void normalize_type(SwigType *ty) {
    SwigType *qty;
    if (CPlusPlus) {
      Replaceall(ty, "struct ", "");
      Replaceall(ty, "union ", "");
      Replaceall(ty, "class ", "");
    }

    qty = SwigType_typedef_qualified(ty);
    /*    Printf(stdout,"%s --> %s\n", ty, qty); */
    Clear(ty);
    Append(ty, qty);
    Delete(qty);
  }

  /* Normalize a parameter list */

  void normalize_parms(ParmList *p) {
    while (p) {
      SwigType *ty = Getattr(p, "type");
      normalize_type(ty);
      /* This is a check for a function type */
      {
	SwigType *qty = SwigType_typedef_resolve_all(ty);
	if (SwigType_isfunction(qty)) {
	  SwigType_add_pointer(ty);
	}
	Delete(qty);
      }

      String *value = Getattr(p, "value");
      if (value) {
	Node *n = Swig_symbol_clookup(value, 0);
	if (n) {
	  String *q = Swig_symbol_qualified(n);
	  if (q && Len(q)) {
	    String *vb = Swig_scopename_last(value);
	    Clear(value);
	    Printf(value, "%s::%s", SwigType_namestr(q), vb);
	    Delete(q);
	  }
	}
      }
      if (value && SwigType_istemplate(value)) {
	String *nv = SwigType_namestr(value);
	Setattr(p, "value", nv);
      }
      p = nextSibling(p);
    }
  }

  void normalize_later(ParmList *p) {
    while (p) {
      SwigType *ty = Getattr(p, "type");
      Append(normalize, ty);
      p = nextSibling(p);
    }
  }

  /* Walk through entries in normalize list and patch them up */
  void normalize_list() {
    Hash *currentsym = Swig_symbol_current();

    normal_node *nn = patch_list;
    normal_node *np;
    while (nn) {
      Swig_symbol_setscope(nn->symtab);
      SwigType_set_scope(nn->typescope);
      Iterator t;
      for (t = First(nn->normallist); t.item; t = Next(t)) {
	normalize_type(t.item);
      }
      Delete(nn->normallist);
      np = nn->next;
      delete(nn);
      nn = np;
    }
    Swig_symbol_setscope(currentsym);
  }

  /* generate C++ inheritance type-relationships */
  void cplus_inherit_types_impl(Node *first, Node *cls, String *clsname, const char *bases, const char *baselist, int ispublic, String *cast = 0) {

    if (first == cls)
      return;			/* The Marcelo check */
    if (!cls)
      cls = first;
    List *alist = 0;
    List *ilist = Getattr(cls, bases);
    if (!ilist) {
      List *nlist = Getattr(cls, baselist);
      if (nlist) {
	int len = Len(nlist);
	int i;
	for (i = 0; i < len; i++) {
	  Node *bcls = 0;
	  int clsforward = 0;
	  String *bname = Getitem(nlist, i);
	  String *sname = bname;
	  String *tname = 0;

	  /* Try to locate the base class.   We look in the symbol table and we chase 
	     typedef declarations to get to the base class if necessary */
	  Symtab *st = Getattr(cls, "sym:symtab");

	  if (SwigType_istemplate(bname)) {
	    tname = SwigType_typedef_resolve_all(bname);
	    sname = tname;
	  }
	  while (1) {
	    String *qsname = SwigType_typedef_qualified(sname);
	    bcls = Swig_symbol_clookup(qsname, st);
	    Delete(qsname);
	    if (bcls) {
	      if (Strcmp(nodeType(bcls), "class") != 0) {
		/* Not a class.   The symbol could be a typedef. */
		if (checkAttribute(bcls, "storage", "typedef")) {
		  SwigType *decl = Getattr(bcls, "decl");
		  if (!decl || !(Len(decl))) {
		    sname = Getattr(bcls, "type");
		    st = Getattr(bcls, "sym:symtab");
		    if (SwigType_istemplate(sname)) {
		      if (tname)
			Delete(tname);
		      tname = SwigType_typedef_resolve_all(sname);
		      sname = tname;
		    }
		    continue;
		  }
		  // A case when both outer and nested classes inherit from the same parent. Constructor may be found instead of the class itself.
		} else if (GetFlag(cls, "nested") && checkAttribute(bcls, "nodeType", "constructor")) {
		  bcls = Getattr(bcls, "parentNode");
		  if (Getattr(bcls, "typepass:visit")) {
		    if (!Getattr(bcls, "feature:onlychildren")) {
		      if (!ilist)
			ilist = alist = NewList();
		      Append(ilist, bcls);
		    } else {
		      if (!GetFlag(bcls, "feature:ignore")) {
			Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(bname), Getline(bname), "Base class '%s' has no name as it is an empty template instantiated with '%%template()'. Ignored.\n", SwigType_namestr(bname));
			Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(bcls), Getline(bcls), "The %%template directive must be written before '%s' is used as a base class and be declared with a name.\n", SwigType_namestr(bname));
		      }
		    }
		  }
		  break;
		}
		if (Strcmp(nodeType(bcls), "classforward") != 0) {
		  Swig_error(Getfile(bname), Getline(bname), "'%s' is not a valid base class.\n", SwigType_namestr(bname));
		  Swig_error(Getfile(bcls), Getline(bcls), "See definition of '%s'.\n", SwigType_namestr(bname));
		} else {
		  Swig_warning(WARN_TYPE_INCOMPLETE, Getfile(bname), Getline(bname), "Base class '%s' is incomplete.\n", SwigType_namestr(bname));
		  Swig_warning(WARN_TYPE_INCOMPLETE, Getfile(bcls), Getline(bcls), "Only forward declaration '%s' was found.\n", SwigType_namestr(bname));
		  clsforward = 1;
		}
		bcls = 0;
	      } else {
		if (Getattr(bcls, "typepass:visit")) {
		  if (!Getattr(bcls, "feature:onlychildren")) {
		    if (!ilist)
		      ilist = alist = NewList();
		    Append(ilist, bcls);
		  } else {
		    if (!GetFlag(bcls, "feature:ignore")) {
		      Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(bname), Getline(bname), "Base class '%s' has no name as it is an empty template instantiated with '%%template()'. Ignored.\n", SwigType_namestr(bname));
		      Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(bcls), Getline(bcls), "The %%template directive must be written before '%s' is used as a base class and be declared with a name.\n", SwigType_namestr(bname));
		    }
		  }
		} else {
		  Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(bname), Getline(bname), "Base class '%s' undefined.\n", SwigType_namestr(bname));
		  Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(bcls), Getline(bcls), "'%s' must be defined before it is used as a base class.\n", SwigType_namestr(bname));
		}
	      }
	    }
	    break;
	  }

	  if (tname)
	    Delete(tname);
	  if (!bcls) {
	    if (!clsforward && !GetFlag(cls, "feature:ignore")) {
	      if (ispublic && !Getmeta(bname, "already_warned")) {
		Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(bname), Getline(bname), "Nothing known about base class '%s'. Ignored.\n", SwigType_namestr(bname));
		if (Strchr(bname, '<')) {
		  Swig_warning(WARN_TYPE_UNDEFINED_CLASS, Getfile(bname), Getline(bname), "Maybe you forgot to instantiate '%s' using %%template.\n", SwigType_namestr(bname));
		}
		Setmeta(bname, "already_warned", "1");
	      }
	    }
	    SwigType_inherit(clsname, bname, cast, 0);
	  }
	}
      }
      if (ilist) {
	Setattr(cls, bases, ilist);
      }
    }
    if (alist)
      Delete(alist);

    if (!ilist)
      return;
    int len = Len(ilist);
    int i;
    for (i = 0; i < len; i++) {
      Node *bclass = Getitem(ilist, i);
      SwigType *bname = Getattr(bclass, "name");
      Hash *scopes = Getattr(bclass, "typescope");
      SwigType_inherit(clsname, bname, cast, 0);
      if (ispublic && !GetFlag(bclass, "feature:ignore")) {
	String *smart = Getattr(first, "smart");
	if (smart) {
	  /* Record a (fake) inheritance relationship between smart pointer
	     and smart pointer to base class, so that smart pointer upcasts
	     are automatically generated. */
	  SwigType *bsmart = Getattr(bclass, "smart");
	  if (bsmart) {
	    String *smartnamestr = SwigType_namestr(smart);
	    String *bsmartnamestr = SwigType_namestr(bsmart);

	    /* construct casting code */
	    String *convcode = NewStringf("\n    *newmemory = SWIG_CAST_NEW_MEMORY;\n    return (void *) new %s(*(%s *)$from);\n", bsmartnamestr, smartnamestr);

	    /* setup inheritance relationship between smart pointer templates */
	    SwigType_inherit(smart, bsmart, 0, convcode);

	    Delete(bsmartnamestr);
	    Delete(smartnamestr);
	    Delete(convcode);
	  } else {
	    Swig_warning(WARN_LANG_SMARTPTR_MISSING, Getfile(first), Getline(first), "Base class '%s' of '%s' is not similarly marked as a smart pointer.\n", SwigType_namestr(Getattr(bclass, "name")), SwigType_namestr(Getattr(first, "name")));
	  }
	} else {
	  if (GetFlag(bclass, "smart"))
	    if (!GetFlag(first, "feature:ignore"))
	      Swig_warning(WARN_LANG_SMARTPTR_MISSING, Getfile(first), Getline(first), "Derived class '%s' of '%s' is not similarly marked as a smart pointer.\n", SwigType_namestr(Getattr(first, "name")), SwigType_namestr(Getattr(bclass, "name")));
	}
      }
      if (!importmode) {
	String *btype = Copy(bname);
	SwigType_add_pointer(btype);
	SwigType_remember(btype);
	Delete(btype);
      }
      if (scopes) {
	SwigType_inherit_scope(scopes);
      }
      /* Set up inheritance in the symbol table */
      Symtab *st = Getattr(cls, "symtab");
      Symtab *bst = Getattr(bclass, "symtab");
      if (st == bst) {
	Swig_warning(WARN_PARSE_REC_INHERITANCE, Getfile(cls), Getline(cls), "Recursive scope inheritance of '%s'.\n", SwigType_namestr(Getattr(cls, "name")));
	continue;
      }
      Symtab *s = Swig_symbol_current();
      Swig_symbol_setscope(st);
      Swig_symbol_inherit(bst);
      Swig_symbol_setscope(s);

      /* Recursively hit base classes */
      String *namestr = SwigType_namestr(Getattr(bclass, "name"));
      String *newcast = NewStringf("(%s *)%s", namestr, cast);
      Delete(namestr);
      cplus_inherit_types_impl(first, bclass, clsname, bases, baselist, ispublic, newcast);
      Delete(newcast);
    }
  }

  void append_list(List *lb, List *la) {
    if (la && lb) {
      for (Iterator bi = First(la); bi.item; bi = Next(bi)) {
	Append(lb, bi.item);
      }
    }
  }

  void cplus_inherit_types(Node *first, Node *cls, String *clsname, String *cast = 0) {
    cplus_inherit_types_impl(first, cls, clsname, "bases", "baselist", 1, cast);
    cplus_inherit_types_impl(first, cls, clsname, "protectedbases", "protectedbaselist", 0, cast);
    cplus_inherit_types_impl(first, cls, clsname, "privatebases", "privatebaselist", 0, cast);

    if (!cls)
      cls = first;

    List *allbases = NewList();
    append_list(allbases, Getattr(cls, "bases"));
    append_list(allbases, Getattr(cls, "protectedbases"));
    append_list(allbases, Getattr(cls, "privatebases"));
    if (Len(allbases)) {
      Setattr(cls, "allbases", allbases);
    }
    Delete(allbases);
  }

  /* ------------------------------------------------------------
   * nspace_setting()
   *
   * Configures "sym:nspace" on the node and returns an overridden
   * nspace value when %nspacemove is used.
   * outer should point to parent class.
   * ------------------------------------------------------------ */

  String *nspace_setting(Node *n, Node *outer) {
    String *nssymname_new = nssymname;
    String *feature_nspace = GetFlagAttr(n, "feature:nspace");
    String *nspace = Copy(feature_nspace);

    // Check validity - single colons not allowed
    const char *c = Char(feature_nspace);
    int valid = 1;
    while(c && *c) {
      if (*(c++) == ':' && *(c++) != ':') {
	valid = 0;
	break;
      }
    }

    // Remove whitespace
    Replaceall(nspace, " ", "");
    Replaceall(nspace, "\t", "");

    valid = valid && (feature_nspace ? Equal(feature_nspace, "1") || Swig_scopename_isvalid(nspace) : 1);
    Replaceall(nspace, "::", ".");
    if (valid) {
      if (outer) {
	// Nested class or enum in a class
	String *outer_nspace = Getattr(outer, "sym:nspace");
	String *nspace_attribute = Getattr(n, "feature:nspace");
	bool warn = false;
	if (outer_nspace) {
	  if (Equal(nspace_attribute, "0")) {
	    warn = true;
	  } else if (nspace && !(Equal(nspace, "1") || Equal(nspace, outer_nspace))) {
	    warn = true;
	  }
	} else if (nspace) {
	  warn = true;
	}
	if (warn) {
	  String *outer_nspace_feature = Copy(outer_nspace);
	  Replaceall(outer_nspace_feature, ".", "::");
	  Swig_warning(WARN_TYPE_NSPACE_SETTING, Getfile(n), Getline(n), "Ignoring nspace setting (%s) for '%s',\n", nspace_attribute, Swig_name_decl(n));
	  Swig_warning(WARN_TYPE_NSPACE_SETTING, Getfile(outer), Getline(outer), "as it conflicts with the nspace setting (%s) for outer class '%s'.\n", outer_nspace_feature, Swig_name_decl(outer));
	}
	Setattr(n, "sym:nspace", outer_nspace);
      } else {
	if (nspace) {
	  if (Equal(nspace, "1")) {
	    if (nssymname)
	      Setattr(n, "sym:nspace", nssymname);
	  } else {
	    Setattr(n, "sym:nspace", nspace);
	    nssymname_new = nspace;
	  }
	}
      }
    } else {
      Swig_error(Getfile(n), Getline(n), "'%s' is not a valid identifier for nspace.\n", feature_nspace);
    }
    Delete(nspace);
    return nssymname_new;
  }

  /* ------------------------------------------------------------
   * top()
   * ------------------------------------------------------------ */

  virtual int top(Node *n) {
    importmode = 0;
    module = Getattr(n, "module");
    inclass = 0;
    normalize = 0;
    nsname = 0;
    nssymname = 0;
    classhash = Getattr(n, "classes");
    emit_children(n);
    normalize_list();
    SwigType_set_scope(0);
    return SWIG_OK;
  }


  /* ------------------------------------------------------------
   * moduleDirective()
   * ------------------------------------------------------------ */

  virtual int moduleDirective(Node *n) {
    if (!module) {
      module = n;
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * importDirective()
   * ------------------------------------------------------------ */

  virtual int importDirective(Node *n) {
    String *oldmodule = module;
    int oldimport = importmode;
    importmode = 1;
    module = 0;
    emit_children(n);
    importmode = oldimport;
    module = oldmodule;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * includeDirective()
   * externDirective()
   * extendDirective()
   * ------------------------------------------------------------ */

  virtual int includeDirective(Node *n) {
    return emit_children(n);
  }
  virtual int externDeclaration(Node *n) {
    return emit_children(n);
  }
  virtual int extendDirective(Node *n) {
    return emit_children(n);
  }

  /* ------------------------------------------------------------
   * classDeclaration()
   * ------------------------------------------------------------ */

  virtual int classDeclaration(Node *n) {
    String *name = Getattr(n, "name");
    String *tdname = Getattr(n, "tdname");
    String *unnamed = Getattr(n, "unnamed");
    String *storage = Getattr(n, "storage");
    String *kind = Getattr(n, "kind");
    save_value<Node*> oldinclass(inclass);
    List *olist = normalize;
    Symtab *symtab;
    String *nname = 0;
    String *fname = 0;
    String *scopename = 0;
    String *template_default_expanded = 0;

    normalize = NewList();

    if (name) {
      if (SwigType_istemplate(name)) {
	// We need to fully resolve the name and expand default template parameters to make templates work correctly */
	Node *cn;
	SwigType *resolved_name = SwigType_typedef_resolve_all(name);
	SwigType *deftype_name = Swig_symbol_template_deftype(resolved_name, 0);
	fname = Copy(resolved_name);
	if (!Equal(resolved_name, deftype_name))
	  template_default_expanded = Copy(deftype_name);
	if (!Equal(fname, name) && (cn = Swig_symbol_clookup_local(fname, 0))) {
	  if ((n == cn)
	      || (Strcmp(nodeType(cn), "template") == 0)
	      || (Getattr(cn, "feature:onlychildren") != 0)
	      || (Getattr(n, "feature:onlychildren") != 0)) {
	    Swig_symbol_cadd(fname, n);
	    if (template_default_expanded)
	      Swig_symbol_cadd(template_default_expanded, n);
	    SwigType_typedef_class(fname);
	    scopename = Copy(fname);
	  } else {
	    // Arguably the parser should instead ignore these duplicate template instantiations, in particular for ensuring the first parsed instantiation is used
	    SetFlag(n, "feature:ignore");
	    Swig_warning(WARN_TYPE_REDEFINED, Getfile(n), Getline(n), "Duplicate template instantiation of '%s' with name '%s' ignored,\n", SwigType_namestr(name), Getattr(n, "sym:name"));
	    Swig_warning(WARN_TYPE_REDEFINED, Getfile(cn), Getline(cn), "previous instantiation of '%s' with name '%s'.\n", SwigType_namestr(Getattr(cn, "name")), Getattr(cn, "sym:name"));
	    scopename = 0;
	  }
	} else {
	  Swig_symbol_cadd(fname, n);
	  SwigType_typedef_class(fname);
	  scopename = Copy(fname);
	}
	Delete(deftype_name);
	Delete(resolved_name);
      } else {
	if ((CPlusPlus) || (unnamed)) {
	  SwigType_typedef_class(name);
	} else {
	  SwigType_typedef_class(NewStringf("%s %s", kind, name));
	}
	scopename = Copy(name);
      }
    } else {
      scopename = 0;
    }

    Setattr(n, "typepass:visit", "1");

    /* Need to set up a typedef if unnamed */
    if (unnamed && tdname && (Cmp(storage, "typedef") == 0)) {
      SwigType_typedef(unnamed, tdname);
    }
    // name of the outer class should already be patched to contain its outer classes names, but not to contain namespaces
    // namespace name (if present) is added after processing child nodes
    if (Getattr(n, "nested:outer") && name) {
      String *outerName = Getattr(Getattr(n, "nested:outer"), "name");
      name = NewStringf("%s::%s", outerName, name);
      Setattr(n, "name", name);
      if (tdname) {
	tdname = NewStringf("%s::%s", outerName, tdname);
	Setattr(n, "tdname", tdname);
      }
    }

    if (nsname && name) {
      nname = NewStringf("%s::%s", nsname, name);
      String *tdname = Getattr(n, "tdname");
      if (tdname) {
	tdname = NewStringf("%s::%s", nsname, tdname);
	Setattr(n, "tdname", tdname);
      }
    }

    String *oldnssymname = nssymname;
    nssymname = nspace_setting(n, Getattr(n, "nested:outer"));

    SwigType_new_scope(scopename);
    SwigType_attach_symtab(Getattr(n, "symtab"));

    if (!GetFlag(n, "feature:ignore")) {
      SwigType *smart = Swig_cparse_smartptr(n);
      if (smart) {
	// Resolve the type in 'feature:smartptr' in the scope of the class it is attached to
	normalize_type(smart);
	Setattr(n, "smart", smart);
	Delete(smart);
      }
    }

    /* Inherit type definitions into the class */
    if (name && !(GetFlag(n, "nested") && !checkAttribute(n, "access", "public") && 
      (GetFlag(n, "feature:flatnested") || Language::instance()->nestedClassesSupport() == Language::NCS_None))) {
      cplus_inherit_types(n, 0, nname ? nname : (fname ? fname : name));
    }

    inclass = n;
    symtab = Swig_symbol_setscope(Getattr(n, "symtab"));
    emit_children(n);
    Swig_symbol_setscope(symtab);

    Hash *ts = SwigType_pop_scope();
    Setattr(n, "typescope", ts);
    Delete(ts);
    Setattr(n, "module", module);

    // When a fully qualified templated type with default parameters is used in the parsed code, 
    // the following additional symbols and scopes are needed for successful lookups
    if (template_default_expanded) {
      Swig_symbol_alias(template_default_expanded, Getattr(n, "symtab"));
      SwigType_scope_alias(template_default_expanded, Getattr(n, "typescope"));
    }

    nssymname = oldnssymname;

    /* Normalize deferred types */
    {
      normal_node *nn = new normal_node();
      nn->normallist = normalize;
      nn->symtab = Getattr(n, "symtab");
      nn->next = patch_list;
      nn->typescope = Getattr(n, "typescope");
      patch_list = nn;
    }

    normalize = olist;

    /* If in a namespace, patch the class name */
    if (nname) {
      Setattr(n, "name", nname);
      Delete(nname);
    }
    Delete(fname);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * templateDeclaration()
   * ------------------------------------------------------------ */

  virtual int templateDeclaration(Node *n) {
    String *name = Getattr(n, "name");
    String *ttype = Getattr(n, "templatetype");
    if (Strcmp(ttype, "class") == 0) {
      String *rname = SwigType_typedef_resolve_all(name);
      SwigType_typedef_class(rname);
      Delete(rname);
    } else if (Strcmp(ttype, "classforward") == 0) {
      String *rname = SwigType_typedef_resolve_all(name);
      SwigType_typedef_class(rname);
      Delete(rname);
      /*      SwigType_typedef_class(name); */
    } else if (Strcmp(ttype, "cdecl") == 0) {
      String *rname = SwigType_typedef_resolve_all(name);
      SwigType_typedef_class(rname);
      Delete(rname);
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * lambdaDeclaration()
   * ------------------------------------------------------------ */

  virtual int lambdaDeclaration(Node *) {
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * classforwardDeclaration()
   * ------------------------------------------------------------ */

  virtual int classforwardDeclaration(Node *n) {

    /* Can't do inside a C struct because it breaks C nested structure wrapping */
    if ((!inclass) || (CPlusPlus)) {
      String *name = Getattr(n, "name");
      SwigType_typedef_class(name);
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * namespaceDeclaration()
   * ------------------------------------------------------------ */

  virtual int namespaceDeclaration(Node *n) {
    Symtab *symtab;
    String *name = Getattr(n, "name");
    String *alias = Getattr(n, "alias");
    List *olist = normalize;
    normalize = NewList();
    if (alias) {
      Typetab *ts = Getattr(n, "typescope");
      if (!ts) {
	/* Create an empty scope for the alias */
	Node *ns = Getattr(n, "namespace");
	SwigType_scope_alias(name, Getattr(ns, "typescope"));
	ts = Getattr(ns, "typescope");
	Setattr(n, "typescope", ts);
      }
      /* Namespace alias */
      return SWIG_OK;
    } else {
      if (name) {
	Node *nn = Swig_symbol_clookup(name, n);
	Hash *ts = 0;
	if (nn)
	  ts = Getattr(nn, "typescope");
	if (!ts) {
	  SwigType_new_scope(name);
	  SwigType_attach_symtab(Getattr(n, "symtab"));
	} else {
	  SwigType_set_scope(ts);
	}
      }
      String *oldnsname = nsname;
      String *oldnssymname = nssymname;
      nsname = Swig_symbol_qualified(Getattr(n, "symtab"));
      nssymname = Swig_symbol_qualified_language_scopename(Getattr(n, "symtab"));
      symtab = Swig_symbol_setscope(Getattr(n, "symtab"));
      emit_children(n);
      Swig_symbol_setscope(symtab);

      if (name) {
	Hash *ts = SwigType_pop_scope();
	Setattr(n, "typescope", ts);
	Delete(ts);
      }

      /* Normalize deferred types */
      {
	normal_node *nn = new normal_node();
	nn->normallist = normalize;
	nn->symtab = Getattr(n, "symtab");
	nn->next = patch_list;
	nn->typescope = Getattr(n, "typescope");
	patch_list = nn;
      }
      normalize = olist;

      Delete(nssymname);
      nssymname = oldnssymname;
      Delete(nsname);
      nsname = oldnsname;
      return SWIG_OK;
    }
  }

  /* ------------------------------------------------------------
   * cDeclaration()
   * ------------------------------------------------------------ */

  virtual int cDeclaration(Node *n) {
    if (NoExcept) {
      Delattr(n, "throws");
    }

    /* Normalize types. */
    SwigType *ty = Getattr(n, "type");
    if (!ty) {
      return SWIG_OK;
    }
    normalize_type(ty);
    SwigType *decl = Getattr(n, "decl");
    if (decl) {
      normalize_type(decl);
    }
    normalize_parms(Getattr(n, "parms"));
    normalize_parms(Getattr(n, "throws"));
    if (GetFlag(n, "conversion_operator")) {
      /* The call to the operator in the generated wrapper must be fully qualified in order to compile */
      SwigType *name = Getattr(n, "name");
      SwigType *qualifiedname = Swig_symbol_string_qualify(name, 0);
      Clear(name);
      Append(name, qualifiedname);
      Delete(qualifiedname);
    }

    if (checkAttribute(n, "storage", "typedef")) {
      String *name = Getattr(n, "name");
      ty = Getattr(n, "type");
      decl = Getattr(n, "decl");
      SwigType *t = Copy(ty);
      {
	/* If the typename is qualified, make sure the scopename is fully qualified when making a typedef */
	if (Swig_scopename_check(t) && strncmp(Char(t), "::", 2)) {
	  String *base, *prefix, *qprefix;
	  base = Swig_scopename_last(t);
	  prefix = Swig_scopename_prefix(t);
	  qprefix = SwigType_typedef_qualified(prefix);
	  Delete(t);
	  t = NewStringf("%s::%s", qprefix, base);
	  Delete(base);
	  Delete(prefix);
	  Delete(qprefix);
	}
      }
      SwigType_push(t, decl);
      if (CPlusPlus) {
	Replaceall(t, "struct ", "");
	Replaceall(t, "union ", "");
	Replaceall(t, "class ", "");
      }
      SwigType_typedef(t, name);
    }
    /* If namespaces are active.  We need to patch the name with a namespace prefix */
    if (nsname && !inclass) {
      String *name = Getattr(n, "name");
      if (name) {
	String *nname = NewStringf("%s::%s", nsname, name);
	Setattr(n, "name", nname);
	Delete(nname);
      }
    }
    clean_overloaded(n);
    return SWIG_OK;
  }


  /* ------------------------------------------------------------
   * constructorDeclaration()
   * ------------------------------------------------------------ */

  virtual int constructorDeclaration(Node *n) {
    if (NoExcept) {
      Delattr(n, "throws");
    }

    normalize_parms(Getattr(n, "parms"));
    normalize_parms(Getattr(n, "throws"));

    clean_overloaded(n);
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * destructorDeclaration()
   * ------------------------------------------------------------ */

  virtual int destructorDeclaration(Node *) {
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constantDirective()
   * ------------------------------------------------------------ */

  virtual int constantDirective(Node *n) {
    SwigType *ty = Getattr(n, "type");
    if (ty) {
      Setattr(n, "type", SwigType_typedef_qualified(ty));
    }
    return SWIG_OK;
  }


  /* ------------------------------------------------------------
   * enumDeclaration()
   * ------------------------------------------------------------ */

  virtual int enumDeclaration(Node *n) {
    String *name = Getattr(n, "name");

    if (name) {
      String *scope = 0;

      // Add a typedef to the type table so that we can use 'enum Name' as well as just 'Name'
      if (nsname || inclass) {

	// But first correct the name and tdname to contain the fully qualified scopename
	if (nsname && inclass) {
	  scope = NewStringf("%s::%s", nsname, Getattr(inclass, "name"));
	} else if (nsname) {
	  scope = NewStringf("%s", nsname);
	} else if (inclass) {
	  scope = NewStringf("%s", Getattr(inclass, "name"));
	}

	String *nname = NewStringf("%s::%s", scope, name);
	Setattr(n, "name", nname);

	String *tdname = Getattr(n, "tdname");
	if (tdname) {
	  tdname = NewStringf("%s::%s", scope, tdname);
	  Setattr(n, "tdname", tdname);
	}

	SwigType *t = NewStringf("enum %s", nname);
	SwigType_typedef(t, name);
      } else {
	SwigType *t = NewStringf("enum %s", name);
	SwigType_typedef(t, name);
      }
      Delete(scope);
    }

    String *tdname = Getattr(n, "tdname");
    String *unnamed = Getattr(n, "unnamed");
    String *storage = Getattr(n, "storage");

    // Construct enumtype - for declaring an enum of this type with SwigType_ltype() etc
    String *enumtype = 0;
    if (unnamed && tdname && (Cmp(storage, "typedef") == 0)) {
      enumtype = Copy(Getattr(n, "tdname"));
    } else if (name) {
      enumtype = NewStringf("%s%s", CPlusPlus ? "" : "enum ", Getattr(n, "name"));
    } else {
      // anonymous enums
      enumtype = Copy(Getattr(n, "type"));
    }
    Setattr(n, "enumtype", enumtype);

    String *oldnssymname = nssymname;
    Node *parent = parentNode(n);
    nssymname = nspace_setting(n, parent && Equal(nodeType(parent), "class") ? parent : NULL);

    // This block of code is for dealing with %ignore on an enum item where the target language
    // attempts to use the C enum value in the target language itself and expects the previous enum value
    // to be one more than the previous value... the previous enum item might not exist if it is ignored!
    // - It sets the first non-ignored enum item with the "firstenumitem" attribute.
    // - It adds an enumvalue attribute if the previous enum item is ignored
    {
      Node *c;
      int count = 0;
      String *previous = 0;
      bool previous_ignored = false;
      bool firstenumitem = false;
      for (c = firstChild(n); c; c = nextSibling(c)) {
	assert(strcmp(Char(nodeType(c)), "enumitem") == 0);

	bool reset;
	String *enumvalue = Getattr(c, "enumvalue");
	if (GetFlag(c, "feature:ignore") || !Getattr(c, "sym:name")) {
	  reset = enumvalue ? true : false;
	  previous_ignored = true;
	} else {
	  if (!enumvalue && previous_ignored) {
	    if (previous)
	      Setattr(c, "enumvalue", NewStringf("(%s) + %d", previous, count+1));
	    else
	      Setattr(c, "enumvalue", NewStringf("%d", count));
	    SetFlag(c, "virtenumvalue"); // identify enumvalue as virtual, ie not from the parsed source
	  }
	  if (!firstenumitem) {
	    SetFlag(c, "firstenumitem");
	    firstenumitem = true;
	  }
	  reset = true;
	  previous_ignored = false;
	}
	if (reset) {
	  previous = enumvalue ? enumvalue : Getattr(c, "name");
	  count = 0;
	} else {
	  count++;
	}
      }
    }

    emit_children(n);

    nssymname = oldnssymname;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * enumvalueDeclaration()
   * ------------------------------------------------------------ */

  virtual int enumvalueDeclaration(Node *n) {
    String *name = Getattr(n, "name");
    String *value = Getattr(n, "value");
    String *scopedenum = Getattr(parentNode(n), "scopedenum");
    if (!value)
      value = name;
    if (Strcmp(value, name) == 0) {
      String *new_value;
      if ((nsname || inclass || scopedenum) && cparse_cplusplus) {
	new_value = NewStringf("%s::%s", SwigType_namestr(Swig_symbol_qualified(n)), value);
      } else {
	new_value = NewString(value);
      }
      if ((nsname || inclass || scopedenum) && !cparse_cplusplus) {
	String *cppvalue = NewStringf("%s::%s", SwigType_namestr(Swig_symbol_qualified(n)), value);
	Setattr(n, "cppvalue", cppvalue); /* for target languages that always generate C++ code even when wrapping C code */
      }
      Setattr(n, "value", new_value);
      Delete(new_value);
    }
    Node *next = nextSibling(n);

    // Make up an enumvalue if one was not specified in the parsed code (not designed to be used on enum items and %ignore - enumvalue will be set instead)
    if (!GetFlag(n, "feature:ignore")) {
      if (Getattr(n, "_last") && !Getattr(n, "enumvalue")) {	// Only the first enum item has _last set (Note: first non-ignored enum item has firstenumitem set)
	Setattr(n, "enumvalueex", "0");
      }
      if (next && !Getattr(next, "enumvalue")) {
	Setattr(next, "enumvalueex", NewStringf("%s + 1", Getattr(n, "sym:name")));
      }
    }

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * enumforwardDeclaration()
   * ------------------------------------------------------------ */

  virtual int enumforwardDeclaration(Node *n) {

    // Use enumDeclaration() to do all the hard work.
    // Note that no children can be emitted in a forward declaration as there aren't any.
    int result = enumDeclaration(n);
    if (result == SWIG_OK) {
      // Detect when the real enum matching the forward enum declaration has not been parsed/declared
      SwigType *ty = SwigType_typedef_resolve_all(Getattr(n, "type"));
      Replaceall(ty, "enum ", "");
      Node *nn = Swig_symbol_clookup(ty, 0);

      String *nodetype = nn ? nodeType(nn) : 0;
      if (nodetype) {
	if (Equal(nodetype, "enumforward")) {
	  SetFlag(nn, "enumMissing");
	} // if a real enum was declared this would be an "enum" node type
      }
      Delete(ty);
    }
    return result;
  }

#ifdef DEBUG_OVERLOADED
  static void show_overloaded(Node *n) {
    Node *c = Getattr(n, "sym:overloaded");
    Node *checkoverloaded = c;
    Printf(stdout, "-------------------- overloaded start %s sym:overloaded():%p -------------------------------\n", Getattr(n, "name"), c);
    while (c) {
      if (Getattr(c, "error")) {
        c = Getattr(c, "sym:nextSibling");
        continue;
      }
      if (Getattr(c, "sym:overloaded") != checkoverloaded) {
        Printf(stdout, "sym:overloaded error c:%p checkoverloaded:%p\n", c, checkoverloaded);
        Swig_print_node(c);
        Exit(EXIT_FAILURE);
      }

      String *decl = Strcmp(nodeType(c), "using") == 0 ? NewString("------") : Getattr(c, "decl");
      Printf(stdout, "  show_overloaded %s::%s(%s)          [%s] nodeType:%s\n", parentNode(c) ? Getattr(parentNode(c), "name") : "NOPARENT", Getattr(c, "name"), decl, Getattr(c, "sym:overname"), nodeType(c));
      if (!Getattr(c, "sym:overloaded")) {
        Printf(stdout, "sym:overloaded error.....%p\n", c);
        Swig_print_node(c);
        Exit(EXIT_FAILURE);
      }
      c = Getattr(c, "sym:nextSibling");
    }
    Printf(stdout, "-------------------- overloaded end   %s -------------------------------\n", Getattr(n, "name"));
  }
#endif

  /* ------------------------------------------------------------
   * usingDeclaration()
   * ------------------------------------------------------------ */

  virtual int usingDeclaration(Node *n) {
    if (Getattr(n, "namespace")) {
      /* using namespace id */

      /* For a namespace import.   We set up inheritance in the type system */
      Node *ns = Getattr(n, "node");
      if (ns) {
	Typetab *ts = Getattr(ns, "typescope");
	if (ts) {
	  SwigType_using_scope(ts);
	}
      }
      return SWIG_OK;
    } else {
      Node *ns;
      /* using id */
      Symtab *stab = Getattr(n, "sym:symtab");
      if (stab) {
	String *uname = Getattr(n, "uname");
	ns = Swig_symbol_clookup(uname, stab);
	if (!ns && SwigType_istemplate(uname)) {
	  String *tmp = Swig_symbol_template_deftype(uname, 0);
	  if (!Equal(tmp, uname)) {
	    ns = Swig_symbol_clookup(tmp, stab);
	  }
	  Delete(tmp);
	}
      } else {
	ns = 0;
      }
      // Note that Allocate::usingDeclaration will warn when using member is not found (when ns is zero)
      if (ns) {
	/* Only a single symbol is being used.  There are only a few symbols that
	   we actually care about.  These are typedef, class declarations, and enum */
	String *ntype = nodeType(ns);
	if (Equal(ntype, "cdecl") || Equal(ntype, "constructor")) {
	  if (checkAttribute(ns, "storage", "typedef")) {
	    /* A typedef declaration */
	    String *uname = Getattr(n, "uname");
	    SwigType_typedef_using(uname);
	  }
	} else if ((Strcmp(ntype, "class") == 0) || ((Strcmp(ntype, "classforward") == 0))) {
	  /* We install the using class name as kind of a typedef back to the original class */
	  String *uname = Getattr(n, "uname");
	  /* Import into current type scope */
	  SwigType_typedef_using(uname);
	} else if (Strcmp(ntype, "enum") == 0) {
	  SwigType_typedef_using(Getattr(n, "uname"));
	} else if (Strcmp(ntype, "template") == 0) {
	  SwigType_typedef_using(Getattr(n, "uname"));
	}
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * typemapDirective()
   * ------------------------------------------------------------ */

  virtual int typemapDirective(Node *n) {
    if (inclass || nsname) {
      Node *items = firstChild(n);
      while (items) {
	Parm *pattern = Getattr(items, "pattern");
	Parm *parms = Getattr(items, "parms");
	normalize_later(pattern);
	normalize_later(parms);
	items = nextSibling(items);
      }
    }
    return SWIG_OK;
  }


  /* ------------------------------------------------------------
   * typemapcopyDirective()
   * ------------------------------------------------------------ */

  virtual int typemapcopyDirective(Node *n) {
    if (inclass || nsname) {
      Node *items = firstChild(n);
      ParmList *pattern = Getattr(n, "pattern");
      normalize_later(pattern);
      while (items) {
	ParmList *npattern = Getattr(items, "pattern");
	normalize_later(npattern);
	items = nextSibling(items);
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * applyDirective()
   * ------------------------------------------------------------ */

  virtual int applyDirective(Node *n) {
    if (inclass || nsname) {
      ParmList *pattern = Getattr(n, "pattern");
      normalize_later(pattern);
      Node *items = firstChild(n);
      while (items) {
	Parm *apattern = Getattr(items, "pattern");
	normalize_later(apattern);
	items = nextSibling(items);
      }
    }
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * clearDirective()
   * ------------------------------------------------------------ */

  virtual int clearDirective(Node *n) {
    if (inclass || nsname) {
      Node *p;
      for (p = firstChild(n); p; p = nextSibling(p)) {
	ParmList *pattern = Getattr(p, "pattern");
	normalize_later(pattern);
      }
    }
    return SWIG_OK;
  }

public:
  static void pass(Node *n) {
    TypePass t;
    t.top(n);
  }
};

void Swig_process_types(Node *n) {
  if (!n)
    return;
  TypePass::pass(n);
}

