/* ----------------------------------------------------------------------------- 
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * interface.cxx
 *
 * This module contains support for the interface feature.
 * This feature is used in language modules where the target language does not
 * naturally support C++ style multiple inheritance, but does support inheritance 
 * from multiple interfaces.
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"
#include "cparse.h"

static bool interface_feature_enabled = false;

/* -----------------------------------------------------------------------------
 * collect_interface_methods()
 *
 * Create a list of all the methods from the base classes of class n that are
 * marked as an interface. The resulting list is thus the list of methods that
 * need to be implemented in order for n to be non-abstract.
 * ----------------------------------------------------------------------------- */

static List *collect_interface_methods(Node *n) {
  List *methods = NewList();
  if (List *bases = Getattr(n, "interface:bases")) {
    for (Iterator base = First(bases); base.item; base = Next(base)) {
      Node *cls = base.item;
      if (cls == n)
	continue;
      for (Node *child = firstChild(cls); child; child = nextSibling(child)) {
	if (Cmp(nodeType(child), "cdecl") == 0) {
	  if (GetFlag(child, "feature:ignore") || Getattr(child, "interface:owner"))
	    continue; // skip methods propagated to bases
	  if (!checkAttribute(child, "kind", "function"))
	    continue;
	  if (checkAttribute(child, "storage", "static"))
	    continue; // accept virtual methods, non-virtual methods too... mmm??. Warn that the interface class has something that is not a virtual method?
	  Node *nn = copyNode(child);
	  Setattr(nn, "interface:owner", cls);
	  ParmList *parms = CopyParmList(Getattr(child, "parms"));
	  Setattr(nn, "parms", parms);
	  Delete(parms);
	  ParmList *throw_parm_list = Getattr(child, "throws");
	  if (throw_parm_list)
	    Setattr(nn, "throws", CopyParmList(throw_parm_list));
	  Append(methods, nn);
	}
      }
    }
  }
  return methods;
}

/* -----------------------------------------------------------------------------
 * collect_interface_bases
 * ----------------------------------------------------------------------------- */

static void collect_interface_bases(List *bases, Node *n) {
  if (GetFlag(n, "feature:interface")) {
    String *name = Getattr(n, "interface:name");
    if (!Getattr(bases, name))
      Append(bases, n);
  }

  if (List *baselist = Getattr(n, "bases")) {
    for (Iterator base = First(baselist); base.item; base = Next(base)) {
      if (!GetFlag(base.item, "feature:ignore")) {
	if (GetFlag(base.item, "feature:interface"))
	  collect_interface_bases(bases, base.item);
      }
    }
  }
}

/* -----------------------------------------------------------------------------
 * collect_interface_base_classes()
 *
 * Create a hash containing all the classes up the inheritance hierarchy
 * marked with feature:interface (including this class n).
 * Stops going up the inheritance chain as soon as a class is found without
 * feature:interface.
 * The idea is to find all the base interfaces that a class must implement.
 * ----------------------------------------------------------------------------- */

static void collect_interface_base_classes(Node *n) {
  if (GetFlag(n, "feature:interface")) {
    // check all bases are also interfaces
    if (List *baselist = Getattr(n, "bases")) {
      for (Iterator base = First(baselist); base.item; base = Next(base)) {
	if (!GetFlag(base.item, "feature:ignore")) {
	  if (!GetFlag(base.item, "feature:interface")) {
	    Swig_error(Getfile(n), Getline(n), "Base class '%s' of '%s' is not similarly marked as an interface.\n", SwigType_namestr(Getattr(base.item, "name")), SwigType_namestr(Getattr(n, "name")));
	    Exit(EXIT_FAILURE);
	  }
	}
      }
    }
  }

  List *interface_bases = NewList();
  collect_interface_bases(interface_bases, n);
  if (Len(interface_bases) == 0)
    Delete(interface_bases);
  else
    Setattr(n, "interface:bases", interface_bases);
}

/* -----------------------------------------------------------------------------
 * process_interface_name()
 * ----------------------------------------------------------------------------- */

static void process_interface_name(Node *n) {
  if (GetFlag(n, "feature:interface")) {
    String *interface_name = Getattr(n, "feature:interface:name");
    if (!Len(interface_name)) {
      Swig_error(Getfile(n), Getline(n), "The interface feature for '%s' is missing the name attribute.\n", SwigType_namestr(Getattr(n, "name")));
      Exit(EXIT_FAILURE);
    }
    if (Strchr(interface_name, '%')) {
      String *name = NewStringf(interface_name, Getattr(n, "sym:name"));
      Setattr(n, "interface:name", name);
    } else {
      Setattr(n, "interface:name", interface_name);
    }
  }
}

/* -----------------------------------------------------------------------------
 * Swig_interface_propagate_methods()
 *
 * Find all the base classes marked as an interface (with feature:interface) for
 * class node n. For each of these, add all of its methods as methods of n so that
 * n is not abstract. If class n is also marked as an interface, it will remain
 * abstract and not have any methods added.
 * ----------------------------------------------------------------------------- */

void Swig_interface_propagate_methods(Node *n) {
  if (interface_feature_enabled) {
    process_interface_name(n);
    collect_interface_base_classes(n);
    List *methods = collect_interface_methods(n);
    bool is_interface = GetFlag(n, "feature:interface") ? true : false;
    for (Iterator mi = First(methods); mi.item; mi = Next(mi)) {
      if (!is_interface && GetFlag(mi.item, "abstract"))
	continue;
      String *this_decl = Getattr(mi.item, "decl");
      String *this_decl_resolved = SwigType_typedef_resolve_all(this_decl);
      bool identically_overloaded_method = false; // true when a base class' method is implemented in n
      if (SwigType_isfunction(this_decl_resolved)) {
	String *name = Getattr(mi.item, "name");
	for (Node *child = firstChild(n); child; child = nextSibling(child)) {
	  if (Getattr(child, "interface:owner"))
	    break; // at the end of the list are newly appended methods
	  if (Cmp(nodeType(child), "cdecl") == 0) {
	    if (checkAttribute(child, "name", name)) {
	      String *decl = SwigType_typedef_resolve_all(Getattr(child, "decl"));
	      identically_overloaded_method = Strcmp(decl, this_decl_resolved) == 0;
	      Delete(decl);
	      if (identically_overloaded_method)
		break;
	    }
	  }
	}
      }
      Delete(this_decl_resolved);
      if (!identically_overloaded_method) {
	// Add method copied from base class to this derived class
	Node *cn = mi.item;
	Delattr(cn, "sym:overname");
	String *prefix = Getattr(n, "name");
	String *name = Getattr(cn, "name");
	String *decl = Getattr(cn, "decl");
	String *oldname = Getattr(cn, "sym:name");

	String *symname = Swig_name_make(cn, prefix, name, decl, oldname);
	if (Strcmp(symname, "$ignore") != 0) {
	  Symtab *oldscope = Swig_symbol_setscope(Getattr(n, "symtab"));
	  Node *on = Swig_symbol_add(symname, cn);
	  (void)on;
	  assert(on == cn);

	  // Features from the copied base class method are already present, now add in features specific to the added method in the derived class
	  Swig_features_get(Swig_cparse_features(), Swig_symbol_qualifiedscopename(0), name, decl, cn);
	  Swig_symbol_setscope(oldscope);
	  appendChild(n, cn);
	}
      } else {
	Delete(mi.item);
      }
    }
    Delete(methods);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_interface_feature_enable()
 *
 * Turn on interface feature support
 * ----------------------------------------------------------------------------- */

void Swig_interface_feature_enable() {
  interface_feature_enabled = true;
}
