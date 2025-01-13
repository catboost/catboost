/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * allocate.cxx
 *
 * This module also has two main purposes modifying the parse tree.
 *
 * First, it is responsible for adding in using declarations from base class
 * members into the parse tree.
 *
 * Second, after each class declaration, it analyses if the class/struct supports
 * default constructors and destructors in C++.   There are several rules that
 * define this behavior including pure abstract methods, private sections,
 * and non-default constructors in base classes.  See the ARM or
 * Doc/Manual/SWIGPlus.html for details.
 *
 * Once the analysis is complete, the non-explicit/implied default constructors
 * and destructors are added to the parse tree. Implied copy constructors are
 * added too if requested via the copyctor feature. Detection of implied
 * assignment operators is also handled as assigment is required in the generated
 * code for variable setters.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"

static int virtual_elimination_mode = 0;	/* set to 0 on default */

/* Set virtual_elimination_mode */
void Wrapper_virtual_elimination_mode_set(int flag) {
  virtual_elimination_mode = flag;
}

/* Helper function to assist with abstract class checking.  
   This is a major hack. Sorry.  */

extern "C" {
  static String *search_decl = 0;	/* Declarator being searched */
  static Node *check_implemented(Node *n) {
    String *decl;
    if (!n)
       return 0;
    while (n) {
      if (Strcmp(nodeType(n), "cdecl") == 0) {
	decl = Getattr(n, "decl");
	if (SwigType_isfunction(decl)) {
	  SwigType *decl1 = SwigType_typedef_resolve_all(decl);
	  SwigType *decl2 = SwigType_pop_function(decl1);
	  if (Strcmp(decl2, search_decl) == 0) {
	    if (!GetFlag(n, "abstract")) {
	      Delete(decl1);
	      Delete(decl2);
	      return n;
	    }
	  }
	  Delete(decl1);
	  Delete(decl2);
	}
      }
      n = Getattr(n, "csym:nextSibling");
    }
    return 0;
  }
}

class Allocate:public Dispatcher {
  Node *inclass;
  int extendmode;

  /* Checks if a function, n, is the same as any in the base class, ie if the method is polymorphic.
   * Also checks for methods which will be hidden (ie a base has an identical non-virtual method).
   * Both methods must have public access for a match to occur. */
  int function_is_defined_in_bases(Node *n, Node *bases) {

    if (!bases)
      return 0;

    String *this_decl = Getattr(n, "decl");
    if (!this_decl)
       return 0;

    String *name = Getattr(n, "name");
    String *this_type = Getattr(n, "type");
    String *resolved_decl = SwigType_typedef_resolve_all(this_decl);

    // Search all base classes for methods with same signature
    for (int i = 0; i < Len(bases); i++) {
      Node *b = Getitem(bases, i);
      Node *base = firstChild(b);
      while (base) {
	if (Strcmp(nodeType(base), "extend") == 0) {
	  // Loop through all the %extend methods
	  Node *extend = firstChild(base);
	  while (extend) {
	    if (function_is_defined_in_bases_seek(n, b, extend, this_decl, name, this_type, resolved_decl)) {
	      Delete(resolved_decl);
	      return 1;
	    }
	    extend = nextSibling(extend);
	  }
	} else if (Strcmp(nodeType(base), "using") == 0) {
	  // Loop through all the using declaration methods
	  Node *usingdecl = firstChild(base);
	  while (usingdecl) {
	    if (function_is_defined_in_bases_seek(n, b, usingdecl, this_decl, name, this_type, resolved_decl)) {
	      Delete(resolved_decl);
	      return 1;
	    }
	    usingdecl = nextSibling(usingdecl);
	  }
	} else {
	  // normal methods
	  if (function_is_defined_in_bases_seek(n, b, base, this_decl, name, this_type, resolved_decl)) {
	    Delete(resolved_decl);
	    return 1;
	  }
	}
	base = nextSibling(base);
      }
    }
    Delete(resolved_decl);
    resolved_decl = 0;
    for (int j = 0; j < Len(bases); j++) {
      Node *b = Getitem(bases, j);
      if (function_is_defined_in_bases(n, Getattr(b, "allbases")))
	return 1;
    }
    return 0;
  }

  /* Helper function for function_is_defined_in_bases */
  int function_is_defined_in_bases_seek(Node *n, Node *b, Node *base, String *this_decl, String *name, String *this_type, String *resolved_decl) {

    String *base_decl = Getattr(base, "decl");
    SwigType *base_type = Getattr(base, "type");
    if (base_decl && base_type) {
      if (checkAttribute(base, "name", name) && !GetFlag(b, "feature:ignore") /* whole class is ignored */ ) {
	if (SwigType_isfunction(resolved_decl) && SwigType_isfunction(base_decl)) {
	  // We have found a method that has the same name as one in a base class
	  bool covariant_returntype = false;
	  bool returntype_match = Strcmp(base_type, this_type) == 0 ? true : false;
	  bool decl_match = Strcmp(base_decl, this_decl) == 0 ? true : false;
	  if (returntype_match && decl_match) {
	    // Exact match - we have found a method with identical signature
	    // No typedef resolution was done, but skipping it speeds things up slightly
	  } else {
	    // Either we have:
	    //  1) matching methods but are one of them uses a different typedef (return type or parameter) to the one in base class' method
	    //  2) matching polymorphic methods with covariant return type
	    //  3) a non-matching method (ie an overloaded method of some sort)
	    //  4) a matching method which is not polymorphic, ie it hides the base class' method

	    // Check if fully resolved return types match (including
	    // covariant return types)
	    if (!returntype_match) {
	      String *this_returntype = function_return_type(n);
	      String *base_returntype = function_return_type(base);
	      returntype_match = Strcmp(this_returntype, base_returntype) == 0 ? true : false;
	      if (!returntype_match) {
		covariant_returntype = SwigType_issubtype(this_returntype, base_returntype) ? true : false;
		returntype_match = covariant_returntype;
	      }
	      Delete(this_returntype);
	      Delete(base_returntype);
	    }
	    // The return types must match at this point, for the whole method to match
	    if (returntype_match && !decl_match) {
	      // Now need to check the parameter list
	      // First do an inexpensive parameter count
	      ParmList *this_parms = Getattr(n, "parms");
	      ParmList *base_parms = Getattr(base, "parms");
	      if (ParmList_len(this_parms) == ParmList_len(base_parms)) {
		// Number of parameters are the same, now check that all the parameters match
		SwigType *base_fn = NewString("");
		SwigType *this_fn = NewString("");
		SwigType_add_function(base_fn, base_parms);
		SwigType_add_function(this_fn, this_parms);
		base_fn = SwigType_typedef_resolve_all(base_fn);
		this_fn = SwigType_typedef_resolve_all(this_fn);
		if (Strcmp(base_fn, this_fn) == 0) {
		  // Finally check that the qualifiers match
		  int base_qualifier = SwigType_isqualifier(resolved_decl);
		  int this_qualifier = SwigType_isqualifier(base_decl);
		  if (base_qualifier == this_qualifier) {
		    decl_match = true;
		  }
		}
		Delete(base_fn);
		Delete(this_fn);
	      }
	    }
	  }
	  //Printf(stderr,"look %s %s %d %d\n",base_decl, this_decl, returntype_match, decl_match);

	  if (decl_match && returntype_match) {
	    // Found an identical method in the base class
	    bool this_wrapping_protected_members = is_member_director(n) ? true : false;	// This should really check for dirprot rather than just being a director method
	    bool base_wrapping_protected_members = is_member_director(base) ? true : false;	// This should really check for dirprot rather than just being a director method
	    bool both_have_public_access = is_public(n) && is_public(base);
	    bool both_have_protected_access = (is_protected(n) && this_wrapping_protected_members) && (is_protected(base) && base_wrapping_protected_members);
	    bool both_have_private_access = is_private(n) && is_private(base);
	    if (checkAttribute(base, "storage", "virtual")) {
	      // Found a polymorphic method.
	      // Mark the polymorphic method, in case the virtual keyword was not used.
	      Setattr(n, "storage", "virtual");
	      if (!GetFlag(b, "feature:interface")) { // interface implementation neither hides nor overrides
		if (both_have_public_access || both_have_protected_access) {
		  if (!is_non_public_base(inclass, b))
		    Setattr(n, "override", base);	// Note C# definition of override, ie access must be the same
		}
		else if (!both_have_private_access) {
		  // Different access
		  if (this_wrapping_protected_members || base_wrapping_protected_members)
		    if (!is_non_public_base(inclass, b))
		      Setattr(n, "hides", base);	// Note C# definition of hiding, ie hidden if access is different
		}
	      }
	      // Try and find the most base's covariant return type
	      SwigType *most_base_covariant_type = Getattr(base, "covariant");
	      if (!most_base_covariant_type && covariant_returntype)
		most_base_covariant_type = function_return_type(base, false);

	      if (!most_base_covariant_type) {
		// Eliminate the derived virtual method.
		if (virtual_elimination_mode && !is_member_director(n))
		  if (both_have_public_access)
		    if (!is_non_public_base(inclass, b))
		      if (!Swig_symbol_isoverloaded(n)) {
			// Don't eliminate if an overloaded method as this hides the method
			// in the scripting languages: the dispatch function will hide the base method if ignored.
			SetFlag(n, "feature:ignore");
			SetFlag(n, "fvirtual:ignore");
		      }
	      } else {
		// Some languages need to know about covariant return types
		Setattr(n, "covariant", most_base_covariant_type);
	      }

	    } else {
	      // Found an identical method in the base class, but it is not polymorphic.
	      if (both_have_public_access || both_have_protected_access)
		if (!is_non_public_base(inclass, b))
		  Setattr(n, "hides", base);
	    }
	    if (both_have_public_access || both_have_protected_access)
	      return 1;
	  }
	}
      }
    }
    return 0;
  }

  /* Determines whether the base class, b, is in the list of private
   * or protected base classes for class n. */
  bool is_non_public_base(Node *n, Node *b) {
    bool non_public_base = false;
    Node *bases = Getattr(n, "privatebases");
    if (bases) {
      for (int i = 0; i < Len(bases); i++) {
	Node *base = Getitem(bases, i);
	if (base == b)
	  non_public_base = true;
      }
    }
    bases = Getattr(n, "protectedbases");
    if (bases) {
      for (int i = 0; i < Len(bases); i++) {
	Node *base = Getitem(bases, i);
	if (base == b)
	  non_public_base = true;
      }
    }
    return non_public_base;
  }

  /* Returns the return type for a function. The node n should be a function.
     If resolve is true the fully returned type is fully resolved.
     Caller is responsible for deleting returned string. */
  String *function_return_type(Node *n, bool resolve = true) {
    String *decl = Getattr(n, "decl");
    SwigType *type = Getattr(n, "type");
    String *ty = NewString(type);
    SwigType_push(ty, decl);
    if (SwigType_isqualifier(ty))
      Delete(SwigType_pop(ty));
    Delete(SwigType_pop_function(ty));
    if (resolve) {
      String *unresolved = ty;
      ty = SwigType_typedef_resolve_all(unresolved);
      Delete(unresolved);
    }
    return ty;
  }

  /* Checks if a class member is the same as inherited from the class bases */
  int class_member_is_defined_in_bases(Node *member, Node *classnode) {
    Node *bases;		/* bases is the closest ancestors of classnode */
    int defined = 0;

    bases = Getattr(classnode, "allbases");
    if (!bases)
      return 0;

    {
      int old_mode = virtual_elimination_mode;
      if (is_member_director(classnode, member))
	virtual_elimination_mode = 0;

      if (function_is_defined_in_bases(member, bases)) {
	defined = 1;
      }

      virtual_elimination_mode = old_mode;
    }

    if (defined)
      return 1;
    else
      return 0;
  }

  /* Checks to see if a class is abstract through inheritance,
     and saves the first node that seems to be abstract.
   */
  int is_abstract_inherit(Node *n, Node *base = 0, int first = 0) {
    if (!first && (base == n))
      return 0;
    if (!base) {
      /* Root node */
      Symtab *stab = Getattr(n, "symtab");	/* Get symbol table for node */
      Symtab *oldtab = Swig_symbol_setscope(stab);
      int ret = is_abstract_inherit(n, n, 1);
      Swig_symbol_setscope(oldtab);
      return ret;
    }
    List *abstracts = Getattr(base, "abstracts");
    if (abstracts) {
      int dabstract = 0;
      int len = Len(abstracts);
      for (int i = 0; i < len; i++) {
	Node *nn = Getitem(abstracts, i);
	String *name = Getattr(nn, "name");
	if (!name)
	  continue;
	if (Strchr(name, '~'))
	  continue;		/* Don't care about destructors */
	String *base_decl = Getattr(nn, "decl");
	if (base_decl)
	  base_decl = SwigType_typedef_resolve_all(base_decl);
	if (SwigType_isfunction(base_decl))
	  search_decl = SwigType_pop_function(base_decl);
	Node *dn = Swig_symbol_clookup_local_check(name, 0, check_implemented);
	Delete(search_decl);
	Delete(base_decl);

	if (!dn) {
	  List *nabstracts = Getattr(n, "abstracts");
	  if (!nabstracts) {
	    nabstracts = NewList();
	    Setattr(n, "abstracts", nabstracts);
	    Delete(nabstracts);
	  }
	  Append(nabstracts, nn);
	  if (!Getattr(n, "abstracts:firstnode")) {
	    Setattr(n, "abstracts:firstnode", nn);
	  }
	  dabstract = base != n;
	}
      }
      if (dabstract)
	return 1;
    }
    List *bases = Getattr(base, "allbases");
    if (!bases)
      return 0;
    for (int i = 0; i < Len(bases); i++) {
      if (is_abstract_inherit(n, Getitem(bases, i))) {
	return 1;
      }
    }
    return 0;
  }


  /* Grab methods used by smart pointers */

  List *smart_pointer_methods(Node *cls, List *methods, int isconst, String *classname = 0) {
    if (!methods) {
      methods = NewList();
    }

    Node *c = firstChild(cls);

    while (c) {
      if (Getattr(c, "error") || GetFlag(c, "feature:ignore")) {
	c = nextSibling(c);
	continue;
      }
      if (!isconst && (Strcmp(nodeType(c), "extend") == 0)) {
	methods = smart_pointer_methods(c, methods, isconst, Getattr(cls, "name"));
      } else if (Strcmp(nodeType(c), "cdecl") == 0) {
	if (!GetFlag(c, "feature:ignore")) {
	  String *storage = Getattr(c, "storage");
	  if (!((Cmp(storage, "typedef") == 0))
	      && !Strstr(storage, "friend")) {
	    String *name = Getattr(c, "name");
	    String *symname = Getattr(c, "sym:name");
	    Node *e = Swig_symbol_clookup_local(name, 0);
	    if (e && is_public(e) && !GetFlag(e, "feature:ignore") && (Cmp(symname, Getattr(e, "sym:name")) == 0)) {
	      Swig_warning(WARN_LANG_DEREF_SHADOW, Getfile(e), Getline(e), "Declaration of '%s' shadows declaration accessible via operator->(),\n", name);
	      Swig_warning(WARN_LANG_DEREF_SHADOW, Getfile(c), Getline(c), "previous declaration of '%s'.\n", name);
	    } else {
	      /* Make sure node with same name doesn't already exist */
	      int k;
	      int match = 0;
	      for (k = 0; k < Len(methods); k++) {
		e = Getitem(methods, k);
		if (Cmp(symname, Getattr(e, "sym:name")) == 0) {
		  match = 1;
		  break;
		}
		if (!Getattr(e, "sym:name") && (Cmp(name, Getattr(e, "name")) == 0)) {
		  match = 1;
		  break;
		}
	      }
	      if (!match) {
		Node *cc = c;
		while (cc) {
		  Node *cp = cc;
		  if (classname) {
		    Setattr(cp, "extendsmartclassname", classname);
		  }
		  Setattr(cp, "allocate:smartpointeraccess", "1");
		  /* If constant, we have to be careful */
		  if (isconst) {
		    SwigType *decl = Getattr(cp, "decl");
		    if (decl) {
		      if (SwigType_isfunction(decl)) {	/* If method, we only add if it's a const method */
			if (SwigType_isconst(decl)) {
			  Append(methods, cp);
			}
		      } else {
			Append(methods, cp);
		      }
		    } else {
		      Append(methods, cp);
		    }
		  } else {
		    Append(methods, cp);
		  }
		  cc = Getattr(cc, "sym:nextSibling");
		}
	      }
	    }
	  }
	}
      }

      c = nextSibling(c);
    }
    /* Look for methods in base classes */
    {
      Node *bases = Getattr(cls, "bases");
      int k;
      for (k = 0; k < Len(bases); k++) {
	smart_pointer_methods(Getitem(bases, k), methods, isconst);
      }
    }
    /* Remove protected/private members */
    {
      for (int i = 0; i < Len(methods);) {
	Node *n = Getitem(methods, i);
	if (!is_public(n)) {
	  Delitem(methods, i);
	  continue;
	}
	i++;
      }
    }
    return methods;
  }

  void mark_exception_classes(ParmList *p) {
    while (p) {
      SwigType *ty = Getattr(p, "type");
      SwigType *t = SwigType_typedef_resolve_all(ty);
      if (SwigType_isreference(t) || SwigType_ispointer(t) || SwigType_isarray(t)) {
	Delete(SwigType_pop(t));
      }
      Node *c = Swig_symbol_clookup(t, 0);
      if (c) {
	if (!GetFlag(c, "feature:exceptionclass")) {
	  SetFlag(c, "feature:exceptionclass");
	}
      }
      p = nextSibling(p);
      Delete(t);
    }
  }


  void process_exceptions(Node *n) {
    ParmList *catchlist = 0;
    /* 
       the "catchlist" attribute is used to emit the block

       try {$action;} 
       catch <list of catches>;

       in emit.cxx

       and is either constructed from the "feature:catches" feature
       or copied from the node "throws" list.
     */
    String *scatchlist = Getattr(n, "feature:catches");
    if (scatchlist) {
      catchlist = Swig_cparse_parms(scatchlist, n);
      if (catchlist) {
	Setattr(n, "catchlist", catchlist);
	mark_exception_classes(catchlist);
	Delete(catchlist);
      }
    }
    ParmList *throws = Getattr(n, "throws");
    if (throws) {
      /* if there is no explicit catchlist, we catch everything in the throws list */
      if (!catchlist) {
	Setattr(n, "catchlist", throws);
      }
      mark_exception_classes(throws);
    }
  }

/* -----------------------------------------------------------------------------
 * clone_member_for_using_declaration()
 *
 * Create a new member (constructor or method) by copying it from member c, ready
 * for adding as a child to the using declaration node n.
 * ----------------------------------------------------------------------------- */

  Node *clone_member_for_using_declaration(Node *c, Node *n) {
    Node *parent = parentNode(n);
    String *decl = Getattr(c, "decl");
    String *symname = Getattr(n, "sym:name");
    int match = 0;

    Node *over = Getattr(n, "sym:overloaded");
    while (over) {
      String *odecl = Getattr(over, "decl");
      if (Cmp(decl, odecl) == 0) {
	match = 1;
	break;
      }
      over = Getattr(over, "sym:nextSibling");
    }

    if (match) {
      /* Don't generate a method if the method is overridden in this class,
       * for example don't generate another m(bool) should there be a Base::m(bool) :
       * struct Derived : Base {
       *   void m(bool);
       *   using Base::m;
       * };
       */
      return 0;
    }

    Node *nn = copyNode(c);
    Setfile(nn, Getfile(n));
    Setline(nn, Getline(n));
    if (!Getattr(nn, "sym:name"))
      Setattr(nn, "sym:name", symname);
    Symtab *st = Getattr(n, "sym:symtab");
    assert(st);
    Setattr(nn, "sym:symtab", st);
    // The real parent is the "using" declaration node, but subsequent code generally handles
    // and expects a class member to point to the parent class node
    Setattr(nn, "parentNode", parent);

    if (Equal(nodeType(c), "constructor")) {
      Setattr(nn, "name", Getattr(n, "name"));
      Setattr(nn, "sym:name", Getattr(n, "sym:name"));
      // Note that the added constructor's access is the same as that of
      // the base class' constructor not of the using declaration.
      // It has already been set correctly and should not be changed.
    } else {
      // Access might be different from the method in the base class
      Delattr(nn, "access");
      Setattr(nn, "access", Getattr(n, "access"));
    }

    if (!GetFlag(nn, "feature:ignore")) {
      ParmList *parms = CopyParmList(Getattr(c, "parms"));
      int is_pointer = SwigType_ispointer_return(Getattr(nn, "decl"));
      int is_void = checkAttribute(nn, "type", "void") && !is_pointer;
      Setattr(nn, "parms", parms);
      Delete(parms);
      if (Getattr(n, "feature:extend")) {
	String *ucode = is_void ? NewStringf("{ self->%s(", Getattr(n, "uname")) : NewStringf("{ return self->%s(", Getattr(n, "uname"));

	for (ParmList *p = parms; p;) {
	  Append(ucode, Getattr(p, "name"));
	  p = nextSibling(p);
	  if (p)
	    Append(ucode, ",");
	}
	Append(ucode, "); }");
	Setattr(nn, "code", ucode);
	Delete(ucode);
      }
      ParmList *throw_parm_list = Getattr(c, "throws");
      if (throw_parm_list)
	Setattr(nn, "throws", CopyParmList(throw_parm_list));
    } else {
      Delete(nn);
      nn = 0;
    }
    return nn;
  }

/* -----------------------------------------------------------------------------
 * add_member_for_using_declaration()
 *
 * Add a new member (constructor or method) by copying it from member c.
 * Add it into the linked list of members under the using declaration n (ccount,
 * unodes and last_unodes are used for this).
 * ----------------------------------------------------------------------------- */

  void add_member_for_using_declaration(Node *c, Node *n, int &ccount, Node *&unodes, Node *&last_unodes) {
    if (GetFlag(c, "fvirtual:ignore")) {
      // This node was ignored by fvirtual. Thus, it has feature:ignore set. 
      // However, we may have new sibling overrides that will make us want to keep it.
      // Hence, temporarily unset the feature:ignore flag.
      UnsetFlag(c, "feature:ignore");
    }

    if (!(Swig_storage_isstatic(c)
	  || checkAttribute(c, "storage", "typedef")
	  || Strstr(Getattr(c, "storage"), "friend")
	  || (Getattr(c, "feature:extend") && !Getattr(c, "code"))
	  || GetFlag(c, "feature:ignore"))) {

      String *symname = Getattr(n, "sym:name");
      String *csymname = Getattr(c, "sym:name");
      Node *parent = parentNode(n);
      bool using_inherited_constructor_symname_okay = Equal(nodeType(c), "constructor") && Equal(symname, Getattr(parent, "sym:name"));
      if (!csymname || Equal(csymname, symname) || using_inherited_constructor_symname_okay) {
	Node *nn = clone_member_for_using_declaration(c, n);
	if (nn) {
	  ccount++;
	  if (!last_unodes) {
	    last_unodes = nn;
	    unodes = nn;
	  } else {
	    Setattr(nn, "previousSibling", last_unodes);
	    Setattr(last_unodes, "nextSibling", nn);
	    Setattr(nn, "sym:previousSibling", last_unodes);
	    Setattr(last_unodes, "sym:nextSibling", nn);
	    Setattr(nn, "sym:overloaded", unodes);
	    Setattr(unodes, "sym:overloaded", unodes);
	    last_unodes = nn;
	  }
	}
      } else {
	Swig_warning(WARN_LANG_USING_NAME_DIFFERENT, Getfile(n), Getline(n), "Using declaration %s, with name '%s', is not actually using\n", SwigType_namestr(Getattr(n, "uname")), symname);
	Swig_warning(WARN_LANG_USING_NAME_DIFFERENT, Getfile(c), Getline(c), "the method from %s, with name '%s', as the names are different.\n", Swig_name_decl(c), csymname);
      }
    }
    if (GetFlag(c, "fvirtual:ignore")) {
      SetFlag(c, "feature:ignore");
    }
  }

  bool is_assignable_type(const SwigType *type) {
    bool assignable = true;
    if (SwigType_type(type) == T_USER) {
      Node *cn = Swig_symbol_clookup(type, 0);
      if (cn) {
	if ((Strcmp(nodeType(cn), "class") == 0)) {
	  if (Getattr(cn, "allocate:noassign")) {
	    assignable = false;
	  }
	}
      }
    } else if (SwigType_isarray(type)) {
      SwigType *array_type = SwigType_array_type(type);
      assignable = is_assignable_type(array_type);
    } else if (SwigType_isreference(type) || SwigType_isrvalue_reference(type)) {
      SwigType *base_type = Copy(type);
      SwigType_del_element(base_type);
      assignable = is_assignable_type(base_type);
      Delete(base_type);
    }
    return assignable;
  }

  bool is_assignable(Node *n, bool &is_reference, bool &is_const) {
    SwigType *ty = Copy(Getattr(n, "type"));
    SwigType_push(ty, Getattr(n, "decl"));
    SwigType *ftd = SwigType_typedef_resolve_all(ty);
    SwigType *td = SwigType_strip_qualifiers(ftd);

    bool assignable = is_assignable_type(td);
    is_reference = SwigType_isreference(td) || SwigType_isrvalue_reference(td);
    is_const = !SwigType_ismutable(ftd);
    if (GetFlag(n, "hasconsttype"))
      is_const = true;

    Delete(ty);
    Delete(ftd);
    Delete(td);
    return assignable;
  }

public:
Allocate():
  inclass(NULL), extendmode(0) {
  }

  virtual int top(Node *n) {
    cplus_mode = PUBLIC;
    inclass = 0;
    extendmode = 0;
    emit_children(n);
    return SWIG_OK;
  }

  virtual int importDirective(Node *n) {
    return emit_children(n);
  }
  virtual int includeDirective(Node *n) {
    return emit_children(n);
  }
  virtual int externDeclaration(Node *n) {
    return emit_children(n);
  }
  virtual int namespaceDeclaration(Node *n) {
    return emit_children(n);
  }
  virtual int extendDirective(Node *n) {
    extendmode = 1;
    emit_children(n);
    extendmode = 0;
    return SWIG_OK;
  }

  virtual int classDeclaration(Node *n) {
    Symtab *symtab = Swig_symbol_current();
    Swig_symbol_setscope(Getattr(n, "symtab"));
    save_value<Node*> oldInclass(inclass);
    save_value<AccessMode> oldAcessMode(cplus_mode);
    save_value<int> oldExtendMode(extendmode);
    if (Getattr(n, "template"))
      extendmode = 0;
    if (!CPlusPlus) {
      /* Always have default constructors/destructors in C */
      Setattr(n, "allocate:default_constructor", "1");
      Setattr(n, "allocate:default_destructor", "1");
    }

    if (Getattr(n, "allocate:visit")) {
      Swig_symbol_setscope(symtab);
      return SWIG_OK;
    }
    Setattr(n, "allocate:visit", "1");

    /* Always visit base classes first */
    {
      List *bases = Getattr(n, "bases");
      if (bases) {
	for (int i = 0; i < Len(bases); i++) {
	  Node *b = Getitem(bases, i);
	  classDeclaration(b);
	}
      }
    }
    inclass = n;
    String *kind = Getattr(n, "kind");
    if (Strcmp(kind, "class") == 0) {
      cplus_mode = PRIVATE;
    } else {
      cplus_mode = PUBLIC;
    }

    emit_children(n);

    /* Check if the class is abstract via inheritance.   This might occur if a class didn't have
       any pure virtual methods of its own, but it didn't implement all of the pure methods in
       a base class */
    if (!Getattr(n, "abstracts") && is_abstract_inherit(n)) {
      if (((Getattr(n, "allocate:public_constructor") || (!GetFlag(n, "feature:nodefault") && !Getattr(n, "allocate:has_constructor"))))) {
	if (!GetFlag(n, "feature:notabstract")) {
	  Node *na = Getattr(n, "abstracts:firstnode");
	  if (na) {
	    Swig_warning(WARN_TYPE_ABSTRACT, Getfile(n), Getline(n),
			 "Class '%s' might be abstract, " "no constructors generated,\n", SwigType_namestr(Getattr(n, "name")));
	    Swig_warning(WARN_TYPE_ABSTRACT, Getfile(na), Getline(na), "Method %s might not be implemented.\n", Swig_name_decl(na));
	    if (!Getattr(n, "abstracts")) {
	      List *abstracts = NewList();
	      Append(abstracts, na);
	      Setattr(n, "abstracts", abstracts);
	      Delete(abstracts);
	    }
	  }
	}
      }
    }

    if (!Getattr(n, "allocate:has_constructor")) {
      /* No constructor is defined.  We need to check a few things */
      /* If class is abstract.  No default constructor. Sorry */
      if (Getattr(n, "abstracts")) {
	Delattr(n, "allocate:default_constructor");
      }
      if (!Getattr(n, "allocate:default_constructor")) {
	// No default constructor if either the default constructor or copy constructor is declared as deleted
	if (!GetFlag(n, "allocate:deleted_default_constructor") && !GetFlag(n, "allocate:deleted_copy_constructor")) {
	  /* Check base classes */
	  List *bases = Getattr(n, "allbases");
	  int allows_default = 1;

	  for (int i = 0; i < Len(bases); i++) {
	    Node *n = Getitem(bases, i);
	    /* If base class does not allow default constructor, we don't allow it either */
	    if (!Getattr(n, "allocate:default_constructor") && (!Getattr(n, "allocate:default_base_constructor"))) {
	      allows_default = 0;
	    }
	    /* not constructible if base destructor is deleted */
	    if (Getattr(n, "allocate:deleted_default_destructor")) {
	      allows_default = 0;
	    }
	  }
	  if (allows_default) {
	    Setattr(n, "allocate:default_constructor", "1");
	  }
	}
      }
    }

    if (!Getattr(n, "allocate:has_copy_constructor")) {
      if (Getattr(n, "abstracts")) {
	Delattr(n, "allocate:copy_constructor");
	Delattr(n, "allocate:copy_constructor_non_const");
      }
      if (!Getattr(n, "allocate:copy_constructor")) {
	// No copy constructor if the copy constructor is declared as deleted
	if (!GetFlag(n, "allocate:deleted_copy_constructor")) {
	  /* Check base classes */
	  List *bases = Getattr(n, "allbases");
	  int allows_copy = 1;
	  int must_be_copy_non_const = 0;

	  for (int i = 0; i < Len(bases); i++) {
	    Node *n = Getitem(bases, i);
	    /* If base class does not allow copy constructor, we don't allow it either */
	    if (!Getattr(n, "allocate:copy_constructor") && (!Getattr(n, "allocate:copy_base_constructor"))) {
	      allows_copy = 0;
	    }
	    /* not constructible if base destructor is deleted */
	    if (Getattr(n, "allocate:deleted_default_destructor")) {
	      allows_copy = 0;
	    }
	    if (Getattr(n, "allocate:copy_constructor_non_const") || (Getattr(n, "allocate:copy_base_constructor_non_const"))) {
	      must_be_copy_non_const = 1;
	    }
	  }
	  if (allows_copy) {
	    Setattr(n, "allocate:copy_constructor", "1");
	  }
	  if (must_be_copy_non_const) {
	    Setattr(n, "allocate:copy_constructor_non_const", "1");
	  }
	}
      }
    }

    if (!Getattr(n, "allocate:has_destructor")) {
      /* No destructor was defined */
      /* No destructor if the destructor is declared as deleted */
      if (!GetFlag(n, "allocate:deleted_default_destructor")) {
	/* Check base classes */
	List *bases = Getattr(n, "allbases");
	int allows_destruct = 1;

	for (int i = 0; i < Len(bases); i++) {
	  Node *n = Getitem(bases, i);
	  /* If base class does not allow default destructor, we don't allow it either */
	  if (!Getattr(n, "allocate:default_destructor") && (!Getattr(n, "allocate:default_base_destructor"))) {
	    allows_destruct = 0;
	  }
	}
	if (allows_destruct) {
	  Setattr(n, "allocate:default_destructor", "1");
	}
      }
    }

    if (!Getattr(n, "allocate:has_assign")) {
      /* No assignment operator was defined */
      List *bases = Getattr(n, "allbases");
      int allows_assign = 1;

      for (int i = 0; i < Len(bases); i++) {
	Node *n = Getitem(bases, i);
	/* If base class does not allow assignment, we don't allow it either */
	if (Getattr(n, "allocate:noassign")) {
	  allows_assign = 0;
	}
      }
      /* If any member variables are non assignable, this class is also non assignable by default */
      if (GetFlag(n, "allocate:has_nonassignable")) {
	allows_assign = 0;
      }
      if (!allows_assign) {
	Setattr(n, "allocate:noassign", "1");
      }
    }

    if (!Getattr(n, "allocate:has_new")) {
      /* No new operator was defined */
      List *bases = Getattr(n, "allbases");
      int allows_new = 1;

      for (int i = 0; i < Len(bases); i++) {
	Node *n = Getitem(bases, i);
	/* If base class does not allow new operator, we don't allow it either */
	if (Getattr(n, "allocate:has_new")) {
	  allows_new = !Getattr(n, "allocate:nonew");
	}
      }
      if (!allows_new) {
	Setattr(n, "allocate:nonew", "1");
      }
    }

    /* Check if base classes allow smart pointers, but might be hidden */
    if (!Getattr(n, "allocate:smartpointer")) {
      Node *sp = Swig_symbol_clookup("operator ->", 0);
      if (sp) {
	/* Look for parent */
	Node *p = parentNode(sp);
	if (Strcmp(nodeType(p), "extend") == 0) {
	  p = parentNode(p);
	}
	if (Strcmp(nodeType(p), "class") == 0) {
	  if (GetFlag(p, "feature:ignore")) {
	    Setattr(n, "allocate:smartpointer", Getattr(p, "allocate:smartpointer"));
	  }
	}
      }
    }

    Swig_interface_propagate_methods(n);

    /* Only care about default behavior.  Remove temporary values */
    Setattr(n, "allocate:visit", "1");
    Swig_symbol_setscope(symtab);

    /* Now we can add the additional implied constructors and destructors to the parse tree */
    if (!ImportMode && !GetFlag(n, "feature:ignore")) {
      int dir = 0;
      if (Swig_directors_enabled()) {
	int ndir = GetFlag(n, "feature:director");
	int nndir = GetFlag(n, "feature:nodirector");
	/* 'nodirector' has precedence over 'director' */
	dir = (ndir || nndir) ? (ndir && !nndir) : 0;
      }
      int abstract = !dir && abstractClassTest(n);
      int odefault = !GetFlag(n, "feature:nodefault");

      /* default constructor */
      if (!abstract && !GetFlag(n, "feature:nodefaultctor") && odefault) {
	if (!Getattr(n, "allocate:has_constructor") && Getattr(n, "allocate:default_constructor")) {
	  addDefaultConstructor(n);
	}
      }
      /* copy constructor */
      if (CPlusPlus && !abstract && GetFlag(n, "feature:copyctor")) {
	if (!Getattr(n, "allocate:has_copy_constructor") && Getattr(n, "allocate:copy_constructor")) {
	  addCopyConstructor(n);
	}
      }
      /* default destructor */
      if (!GetFlag(n, "feature:nodefaultdtor") && odefault) {
	if (!Getattr(n, "allocate:has_destructor") && Getattr(n, "allocate:default_destructor")) {
	  addDestructor(n);
	}
      }
    }

    return SWIG_OK;
  }

  virtual int accessDeclaration(Node *n) {
    cplus_mode = accessModeFromString(Getattr(n, "kind"));
    return SWIG_OK;
  }

  virtual int usingDeclaration(Node *n) {

    if (GetFlag(n, "feature:ignore"))
      return SWIG_OK;

    if (!Getattr(n, "namespace")) {
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
      if (!ns) {
	if (is_public(n)) {
	  Swig_warning(WARN_PARSE_USING_UNDEF, Getfile(n), Getline(n), "Nothing known about '%s'.\n", SwigType_namestr(Getattr(n, "uname")));
	}
      } else if (Equal(nodeType(ns), "constructor") && !GetFlag(n, "usingctor")) {
	Swig_warning(WARN_PARSE_USING_CONSTRUCTOR, Getfile(n), Getline(n), "Using declaration '%s' for inheriting constructors uses base '%s' which is not an immediate base of '%s'.\n", SwigType_namestr(Getattr(n, "uname")), SwigType_namestr(Getattr(ns, "name")), SwigType_namestr(Getattr(parentNode(n), "name")));
      } else {
	if (inclass && Getattr(n, "sym:name")) {
	  {
	    String *ntype = nodeType(ns);
	    if (Equal(ntype, "cdecl") || Equal(ntype, "constructor") || Equal(ntype, "template") || Equal(ntype, "using")) {
	      /* Add a new class member to the parse tree (copy it from the base class member pointed to by the using declaration in node n) */
	      Node *c = ns;
	      Node *unodes = 0, *last_unodes = 0;
	      int ccount = 0;

	      while (c) {
		String *cnodetype = nodeType(c);
		if (Equal(cnodetype, "cdecl")) {
		  add_member_for_using_declaration(c, n, ccount, unodes, last_unodes);
		} else if (Equal(cnodetype, "constructor")) {
		  add_member_for_using_declaration(c, n, ccount, unodes, last_unodes);
		} else if (Equal(cnodetype, "template")) {
		  // A templated member (in a non-template class or in a template class that where the member has a separate template declaration)
		  // Find the template instantiations in the using declaration (base class)
		  for (Node *member = ns; member; member = nextSibling(member)) {
		    /* Constructors have already been handled, only add member functions
		     * This adds an implicit template instantiation and is a bit unusual as SWIG requires explicit %template for other template instantiations.
		     * However, of note, is that there is no valid C++ syntax for a template instantiation to introduce a name via a using declaration...
		     *
		     *   struct Base { template <typename T> void template_method(T, T) {} };
		     *   struct Derived : Base { using Base::template_method; };
		     *   %template()   Base::template_method<int>;              // SWIG template instantiation
		     *   template void Base::template_method<int>(int, int);    // C++ template instantiation
		     *   template void Derived::template_method<int>(int, int); // Not valid C++
		    */
		    if (Getattr(member, "template") == ns && checkAttribute(ns, "templatetype", "cdecl")) {
		      if (!GetFlag(member, "feature:ignore") && !Getattr(member, "error")) {
			add_member_for_using_declaration(member, n, ccount, unodes, last_unodes);
		      }
		    }
		  }
		} else if (Equal(cnodetype, "using")) {
		  for (Node *member = firstChild(c); member; member = nextSibling(member)) {
		    add_member_for_using_declaration(member, n, ccount, unodes, last_unodes);
		  }
		}
		c = Getattr(c, "csym:nextSibling");
	      }
	      if (unodes) {
		set_firstChild(n, unodes);
		if (ccount > 1) {
		  if (!Getattr(n, "sym:overloaded")) {
		    Setattr(n, "sym:overloaded", n);
		    Setattr(n, "sym:overname", "_SWIG_0");
		  }
		}
	      }

	      /* Hack the parse tree symbol table for overloaded methods. Replace the "using" node with the
	       * list of overloaded methods we have just added in as child nodes to the "using" node.
	       * The node will still exist, it is just the symbol table linked list of overloaded methods
	       * which is hacked. */
	      if (Getattr(n, "sym:overloaded")) {
		int cnt = 0;
		Node *ps = Getattr(n, "sym:previousSibling");
		Node *ns = Getattr(n, "sym:nextSibling");
		Node *fc = firstChild(n);
		Node *firstoverloaded = Getattr(n, "sym:overloaded");
#ifdef DEBUG_OVERLOADED
		show_overloaded(firstoverloaded);
#endif

		if (firstoverloaded == n) {
		  // This 'using' node we are cutting out was the first node in the overloaded list. 
		  // Change the first node in the list
		  Delattr(firstoverloaded, "sym:overloaded");
		  firstoverloaded = fc ? fc : ns;

		  // Correct all the sibling overloaded methods (before adding in new methods)
		  Node *nnn = ns;
		  while (nnn) {
		    Setattr(nnn, "sym:overloaded", firstoverloaded);
		    nnn = Getattr(nnn, "sym:nextSibling");
		  }
		}

		if (!fc) {
		  // Remove from overloaded list ('using' node does not actually end up adding in any methods)
		  if (ps) {
		    Setattr(ps, "sym:nextSibling", ns);
		  }
		  if (ns) {
		    Setattr(ns, "sym:previousSibling", ps);
		  }
		} else {
		  // The 'using' node results in methods being added in - slot in these methods here
		  Node *pp = fc;
		  while (pp) {
		    Node *ppn = Getattr(pp, "sym:nextSibling");
		    Setattr(pp, "sym:overloaded", firstoverloaded);
		    Setattr(pp, "sym:overname", NewStringf("%s_%d", Getattr(n, "sym:overname"), cnt++));
		    if (ppn)
		      pp = ppn;
		    else
		      break;
		  }
		  if (ps) {
		    Setattr(ps, "sym:nextSibling", fc);
		    Setattr(fc, "sym:previousSibling", ps);
		  }
		  if (ns) {
		    Setattr(ns, "sym:previousSibling", pp);
		    Setattr(pp, "sym:nextSibling", ns);
		  }
		}
		Delattr(n, "sym:previousSibling");
		Delattr(n, "sym:nextSibling");
		Delattr(n, "sym:overloaded");
		Delattr(n, "sym:overname");
		clean_overloaded(firstoverloaded);
#ifdef DEBUG_OVERLOADED
		show_overloaded(firstoverloaded);
#endif
	      }
	    }
	  }
	}
      }

      Node *c = 0;
      for (c = firstChild(n); c; c = nextSibling(c)) {
	if (Equal(nodeType(c), "cdecl")) {
	  process_exceptions(c);

	  if (inclass)
	    class_member_is_defined_in_bases(c, inclass);
	} else if (Equal(nodeType(c), "constructor")) {
	  constructorDeclaration(c);
	}
      }
    }

    return SWIG_OK;
  }

  virtual int cDeclaration(Node *n) {

    process_exceptions(n);

    if (inclass) {
      /* check whether the member node n is defined in class node in class's bases */
      class_member_is_defined_in_bases(n, inclass);

      /* Check to see if this is a static member or not.  If so, we add an attribute
         cplus:staticbase that saves the current class */

      int is_static = Swig_storage_isstatic(n);
      if (is_static) {
	Setattr(n, "cplus:staticbase", inclass);
      }

      if (Cmp(Getattr(n, "kind"), "variable") == 0) {
        /* Check member variable to determine whether assignment is valid */
	bool is_reference;
	bool is_const;
	bool assignable = is_assignable(n, is_reference, is_const);
	if (!assignable || is_const) {
	  SetFlag(n, "feature:immutable");
	}
	if (!is_static) {
	  if (!assignable || is_reference || is_const)
	    SetFlag(inclass, "allocate:has_nonassignable"); // The class has a variable that cannot be assigned to
	}
      }

      String *name = Getattr(n, "name");
      if (cplus_mode != PUBLIC) {
	if (Strcmp(name, "operator =") == 0) {
	  /* Look for a private assignment operator */
	  if (!GetFlag(n, "deleted"))
	    Setattr(inclass, "allocate:has_assign", "1");
	  Setattr(inclass, "allocate:noassign", "1");
	} else if (Strcmp(name, "operator new") == 0) {
	  /* Look for a private new operator */
	  if (!GetFlag(n, "deleted"))
	    Setattr(inclass, "allocate:has_new", "1");
	  Setattr(inclass, "allocate:nonew", "1");
	}
      } else {
	if (Strcmp(name, "operator =") == 0) {
	  if (!GetFlag(n, "deleted"))
	    Setattr(inclass, "allocate:has_assign", "1");
	  else
	    Setattr(inclass, "allocate:noassign", "1");
	} else if (Strcmp(name, "operator new") == 0) {
	  if (!GetFlag(n, "deleted"))
	    Setattr(inclass, "allocate:has_new", "1");
	  else
	    Setattr(inclass, "allocate:nonew", "1");
	}
	/* Look for smart pointer operator */
	if ((Strcmp(name, "operator ->") == 0) && (!GetFlag(n, "feature:ignore"))) {
	  /* Look for version with no parameters */
	  Node *sn = n;
	  while (sn) {
	    if (!Getattr(sn, "parms")) {
	      SwigType *type = SwigType_typedef_resolve_all(Getattr(sn, "type"));
	      SwigType_push(type, Getattr(sn, "decl"));
	      Delete(SwigType_pop_function(type));
	      SwigType *base = SwigType_base(type);
	      Node *sc = Swig_symbol_clookup(base, 0);
	      if ((sc) && (Strcmp(nodeType(sc), "class") == 0)) {
		if (SwigType_check_decl(type, "p.")) {
		  /* Need to check if type is a const pointer */
		  int isconst = 0;
		  Delete(SwigType_pop(type));
		  if (SwigType_isconst(type)) {
		    isconst = !Getattr(inclass, "allocate:smartpointermutable");
		    Setattr(inclass, "allocate:smartpointerconst", "1");
		  }
		  else {
		    Setattr(inclass, "allocate:smartpointermutable", "1");
		  }
		  List *methods = smart_pointer_methods(sc, 0, isconst);
		  Setattr(inclass, "allocate:smartpointer", methods);
		  Setattr(inclass, "allocate:smartpointerpointeeclassname", Getattr(sc, "name"));
		} else {
		  /* Hmmm.  The return value is not a pointer.  If the type is a value
		     or reference.  We're going to chase it to see if another operator->()
		     can be found */
		  if ((SwigType_check_decl(type, "")) || (SwigType_check_decl(type, "r."))) {
		    Node *nn = Swig_symbol_clookup("operator ->", Getattr(sc, "symtab"));
		    if (nn) {
		      Delete(base);
		      Delete(type);
		      sn = nn;
		      continue;
		    }
		  }
		}
	      }
	      Delete(base);
	      Delete(type);
	      break;
	    }
	  }
	}
      }
    } else {
      if (Cmp(Getattr(n, "kind"), "variable") == 0) {
	bool is_reference;
	bool is_const;
	bool assignable = is_assignable(n, is_reference, is_const);
	if (!assignable || is_const) {
	  SetFlag(n, "feature:immutable");
	}
      }
    }
    return SWIG_OK;
  }

  virtual int templateDeclaration(Node *n) {
    String *ttype = Getattr(n, "templatetype");
    if (Equal(ttype, "constructor")) {
      // Templated constructors need to be taken account of even if not instantiated with %template
      constructorDeclaration(n);
    }
    return SWIG_OK;
  }

  virtual int constructorDeclaration(Node *n) {
    if (!inclass)
      return SWIG_OK;

    Parm *parms = Getattr(n, "parms");
    bool deleted_constructor = (GetFlag(n, "deleted"));
    bool default_constructor = !ParmList_numrequired(parms);
    AccessMode access_mode = accessModeFromString(Getattr(n, "access"));
    process_exceptions(n);

    if (!deleted_constructor) {
      if (!extendmode) {
	if (default_constructor) {
	  /* Class does define a default constructor */
	  /* However, we had better see where it is defined */
	  if (access_mode == PUBLIC) {
	    Setattr(inclass, "allocate:default_constructor", "1");
	  } else if (access_mode == PROTECTED) {
	    Setattr(inclass, "allocate:default_base_constructor", "1");
	  }
	}
	/* Class defines some kind of constructor. May or may not be public */
	Setattr(inclass, "allocate:has_constructor", "1");
	if (access_mode == PUBLIC) {
	  Setattr(inclass, "allocate:public_constructor", "1");
	}
      } else {
	Setattr(inclass, "allocate:has_constructor", "1");
	Setattr(inclass, "allocate:public_constructor", "1");
      }
    } else {
      if (default_constructor && !extendmode)
	SetFlag(inclass, "allocate:deleted_default_constructor");
    }

    /* See if this is a copy constructor */
    if (parms && (ParmList_numrequired(parms) == 1)) {
      /* Look for a few cases. X(const X &), X(X &), X(X *) */
      int copy_constructor = 0;
      int copy_constructor_non_const = 0;
      SwigType *type = Getattr(inclass, "name");
      String *tn = NewStringf("r.q(const).%s", type);
      String *cc = SwigType_typedef_resolve_all(tn);
      SwigType *rt = SwigType_typedef_resolve_all(Getattr(parms, "type"));
      if (SwigType_istemplate(type)) {
	String *tmp = Swig_symbol_template_deftype(cc, 0);
	Delete(cc);
	cc = tmp;
	tmp = Swig_symbol_template_deftype(rt, 0);
	Delete(rt);
	rt = tmp;
      }
      if (Strcmp(cc, rt) == 0) {
	copy_constructor = 1;
      } else {
	Delete(cc);
	cc = NewStringf("r.%s", Getattr(inclass, "name"));
	if (Strcmp(cc, Getattr(parms, "type")) == 0) {
	  copy_constructor = 1;
	  copy_constructor_non_const = 1;
	} else {
	  Delete(cc);
	  cc = NewStringf("p.%s", Getattr(inclass, "name"));
	  String *ty = SwigType_strip_qualifiers(Getattr(parms, "type"));
	  if (Strcmp(cc, ty) == 0) {
	    copy_constructor = 1;
	  }
	  Delete(ty);
	}
      }
      Delete(cc);
      Delete(rt);
      Delete(tn);

      if (copy_constructor) {
	if (!deleted_constructor) {
	  Setattr(n, "copy_constructor", "1");
	  Setattr(inclass, "allocate:has_copy_constructor", "1");
	  if (access_mode == PUBLIC) {
	    Setattr(inclass, "allocate:copy_constructor", "1");
	  } else if (access_mode == PROTECTED) {
	    Setattr(inclass, "allocate:copy_base_constructor", "1");
	  }
	  if (copy_constructor_non_const) {
	    Setattr(n, "copy_constructor_non_const", "1");
	    Setattr(inclass, "allocate:has_copy_constructor_non_const", "1");
	    if (access_mode == PUBLIC) {
	      Setattr(inclass, "allocate:copy_constructor_non_const", "1");
	    } else if (access_mode == PROTECTED) {
	      Setattr(inclass, "allocate:copy_base_constructor_non_const", "1");
	    }
	  }
	} else {
	  if (!extendmode)
	    SetFlag(inclass, "allocate:deleted_copy_constructor");
	}
      }
    }
    return SWIG_OK;
  }

  virtual int destructorDeclaration(Node *n) {
    (void) n;
    if (!inclass)
      return SWIG_OK;

    if (!GetFlag(n, "deleted")) {
      if (!extendmode) {
	Setattr(inclass, "allocate:has_destructor", "1");
	if (cplus_mode == PUBLIC) {
	  Setattr(inclass, "allocate:default_destructor", "1");
	} else if (cplus_mode == PROTECTED) {
	  Setattr(inclass, "allocate:default_base_destructor", "1");
	} else if (cplus_mode == PRIVATE) {
	  Setattr(inclass, "allocate:private_destructor", "1");
	}
      } else {
	Setattr(inclass, "allocate:has_destructor", "1");
	Setattr(inclass, "allocate:default_destructor", "1");
      }
    } else {
      if (!extendmode)
	SetFlag(inclass, "allocate:deleted_default_destructor");
    }

    return SWIG_OK;
  }

static void addCopyConstructor(Node *n) {
  Node *cn = NewHash();
  set_nodeType(cn, "constructor");
  Setattr(cn, "access", "public");
  Setfile(cn, Getfile(n));
  Setline(cn, Getline(n));

  int copy_constructor_non_const = GetFlag(n, "allocate:copy_constructor_non_const");
  String *cname = Getattr(n, "name");
  SwigType *type = Copy(cname);
  String *lastname = Swig_scopename_last(cname);
  String *name = SwigType_templateprefix(lastname);
  String *cc = NewStringf(copy_constructor_non_const ? "r.%s" : "r.q(const).%s", type);
  String *decl = NewStringf("f(%s).", cc);
  String *oldname = Getattr(n, "sym:name");

  if (Getattr(n, "allocate:has_constructor")) {
    // to work properly with '%rename Class', we must look
    // for any other constructor in the class, which has not been
    // renamed, and use its name as oldname.
    Node *c;
    for (c = firstChild(n); c; c = nextSibling(c)) {
      if (Equal(nodeType(c), "constructor")) {
	String *csname = Getattr(c, "sym:name");
	String *clast = Swig_scopename_last(Getattr(c, "name"));
	if (Equal(csname, clast)) {
	  oldname = csname;
	  Delete(clast);
	  break;
	}
	Delete(clast);
      }
    }
  }

  String *symname = Swig_name_make(cn, cname, name, decl, oldname);
  if (Strcmp(symname, "$ignore") != 0) {
    Parm *p = NewParm(cc, "other", n);

    Setattr(cn, "name", name);
    Setattr(cn, "sym:name", symname);
    SetFlag(cn, "feature:new");
    Setattr(cn, "decl", decl);
    Setattr(cn, "ismember", "1");
    Setattr(cn, "parentNode", n);
    Setattr(cn, "parms", p);
    Setattr(cn, "copy_constructor", "1");

    Symtab *oldscope = Swig_symbol_setscope(Getattr(n, "symtab"));
    Node *on = Swig_symbol_add(symname, cn);
    Swig_features_get(Swig_cparse_features(), Swig_symbol_qualifiedscopename(0), name, decl, cn);
    Swig_symbol_setscope(oldscope);

    if (on == cn) {
      Node *access = NewHash();
      set_nodeType(access, "access");
      Setattr(access, "kind", "public");
      appendChild(n, access);
      appendChild(n, cn);
      Setattr(n, "has_copy_constructor", "1");
      Setattr(n, "copy_constructor_decl", decl);
      Setattr(n, "allocate:copy_constructor", "1");
      Delete(access);
    }
  }
  Delete(cn);
  Delete(lastname);
  Delete(name);
  Delete(decl);
  Delete(symname);
}

static void addDefaultConstructor(Node *n) {
  Node *cn = NewHash();
  set_nodeType(cn, "constructor");
  Setattr(cn, "access", "public");
  Setfile(cn, Getfile(n));
  Setline(cn, Getline(n));

  String *cname = Getattr(n, "name");
  String *lastname = Swig_scopename_last(cname);
  String *name = SwigType_templateprefix(lastname);
  String *decl = NewString("f().");
  String *oldname = Getattr(n, "sym:name");
  String *symname = Swig_name_make(cn, cname, name, decl, oldname);

  if (Strcmp(symname, "$ignore") != 0) {
    Setattr(cn, "name", name);
    Setattr(cn, "sym:name", symname);
    SetFlag(cn, "feature:new");
    Setattr(cn, "decl", decl);
    Setattr(cn, "ismember", "1");
    Setattr(cn, "parentNode", n);
    Setattr(cn, "default_constructor", "1");

    Symtab *oldscope = Swig_symbol_setscope(Getattr(n, "symtab"));
    Node *on = Swig_symbol_add(symname, cn);
    Swig_features_get(Swig_cparse_features(), Swig_symbol_qualifiedscopename(0), name, decl, cn);
    Swig_symbol_setscope(oldscope);

    if (on == cn) {
      Node *access = NewHash();
      set_nodeType(access, "access");
      Setattr(access, "kind", "public");
      appendChild(n, access);
      appendChild(n, cn);
      Setattr(n, "has_default_constructor", "1");
      Setattr(n, "allocate:default_constructor", "1");
      Delete(access);
    }
  }
  Delete(cn);
  Delete(lastname);
  Delete(name);
  Delete(decl);
  Delete(symname);
}

static void addDestructor(Node *n) {
  Node *cn = NewHash();
  set_nodeType(cn, "destructor");
  Setattr(cn, "access", "public");
  Setfile(cn, Getfile(n));
  Setline(cn, Getline(n));

  String *cname = Getattr(n, "name");
  String *lastname = Swig_scopename_last(cname);
  String *name = SwigType_templateprefix(lastname);
  Insert(name, 0, "~");
  String *decl = NewString("f().");
  String *symname = Swig_name_make(cn, cname, name, decl, 0);
  if (Strcmp(symname, "$ignore") != 0) {
    String *possible_nonstandard_symname = NewStringf("~%s", Getattr(n, "sym:name"));

    Setattr(cn, "name", name);
    Setattr(cn, "sym:name", symname);
    Setattr(cn, "decl", "f().");
    Setattr(cn, "ismember", "1");
    Setattr(cn, "parentNode", n);

    Symtab *oldscope = Swig_symbol_setscope(Getattr(n, "symtab"));
    Node *nonstandard_destructor = Equal(possible_nonstandard_symname, symname) ? 0 : Swig_symbol_clookup(possible_nonstandard_symname, 0);
    Node *on = Swig_symbol_add(symname, cn);
    Swig_features_get(Swig_cparse_features(), Swig_symbol_qualifiedscopename(0), name, decl, cn);
    Swig_symbol_setscope(oldscope);

    if (on == cn) {
      // SWIG accepts a non-standard named destructor in %extend that uses a typedef for the destructor name
      // For example: typedef struct X {} XX; %extend X { ~XX() {...} }
      // Don't add another destructor if a nonstandard one has been declared
      if (!nonstandard_destructor) {
	Node *access = NewHash();
	set_nodeType(access, "access");
	Setattr(access, "kind", "public");
	appendChild(n, access);
	appendChild(n, cn);
	Setattr(n, "has_destructor", "1");
	Setattr(n, "allocate:has_destructor", "1");
	Delete(access);
      }
    }
    Delete(possible_nonstandard_symname);
  }
  Delete(cn);
  Delete(lastname);
  Delete(name);
  Delete(decl);
  Delete(symname);
}

};

void Swig_default_allocators(Node *n) {
  if (!n)
    return;
  Allocate *a = new Allocate;
  a->top(n);
  delete a;
}
