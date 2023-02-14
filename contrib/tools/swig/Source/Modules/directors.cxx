/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3 
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at http://www.swig.org/legal.html.
 *
 * directors.cxx
 *
 * Director support functions.
 * Not all of these may be necessary, and some may duplicate existing functionality
 * in SWIG.  --MR
 * ----------------------------------------------------------------------------- */

#include "swigmod.h"

/* -----------------------------------------------------------------------------
 * Swig_csuperclass_call()
 *
 * Generates a fully qualified method call, including the full parameter list.
 * e.g. "base::method(i, j)"
 * ----------------------------------------------------------------------------- */

String *Swig_csuperclass_call(String *base, String *method, ParmList *l) {
  String *call = NewString("");
  int arg_idx = 0;
  Parm *p;
  if (base) {
    Printf(call, "%s::", base);
  }
  Printf(call, "%s(", method);
  for (p = l; p; p = nextSibling(p)) {
    String *pname = Getattr(p, "name");
    if (!pname && Cmp(Getattr(p, "type"), "void")) {
      pname = NewString("");
      Printf(pname, "arg%d", arg_idx++);
    }
    if (p != l)
      Printf(call, ", ");
    Printv(call, pname, NIL);
  }
  Printf(call, ")");
  return call;
}

/* -----------------------------------------------------------------------------
 * Swig_class_declaration()
 *
 * Generate the start of a class/struct declaration.
 * e.g. "class myclass"
 * ----------------------------------------------------------------------------- */

String *Swig_class_declaration(Node *n, String *name) {
  if (!name) {
    name = Getattr(n, "sym:name");
  }
  String *result = NewString("");
  String *kind = Getattr(n, "kind");
  Printf(result, "%s %s", kind, name);
  return result;
}

/* -----------------------------------------------------------------------------
 * Swig_class_name()
 * ----------------------------------------------------------------------------- */

String *Swig_class_name(Node *n) {
  String *name;
  name = Copy(Getattr(n, "sym:name"));
  return name;
}

/* -----------------------------------------------------------------------------
 * Swig_director_declaration()
 *
 * Generate the full director class declaration, complete with base classes.
 * e.g. "class SwigDirector_myclass : public myclass, public Swig::Director {"
 * ----------------------------------------------------------------------------- */

String *Swig_director_declaration(Node *n) {
  String *classname = Swig_class_name(n);
  String *directorname = Language::instance()->directorClassName(n);
  String *base = Getattr(n, "classtype");
  String *declaration = Swig_class_declaration(n, directorname);

  Printf(declaration, " : public %s, public Swig::Director {\n", base);
  Delete(classname);
  Delete(directorname);
  return declaration;
}


/* -----------------------------------------------------------------------------
 * Swig_method_call()
 * ----------------------------------------------------------------------------- */

String *Swig_method_call(const_String_or_char_ptr name, ParmList *parms) {
  String *func;
  int i = 0;
  int comma = 0;
  Parm *p = parms;
  SwigType *pt;
  String *nname;

  func = NewString("");
  nname = SwigType_namestr(name);
  Printf(func, "%s(", nname);
  while (p) {
    String *pname;
    pt = Getattr(p, "type");
    if ((SwigType_type(pt) != T_VOID)) {
      if (comma)
	Printf(func, ",");
      pname = Getattr(p, "name");
      Printf(func, "%s", pname);
      comma = 1;
      i++;
    }
    p = nextSibling(p);
  }
  Printf(func, ")");
  return func;
}

/* -----------------------------------------------------------------------------
 * Swig_method_decl()
 *
 * Return a stringified version of a C/C++ declaration.
 * ----------------------------------------------------------------------------- */

String *Swig_method_decl(SwigType *return_base_type, SwigType *decl, const_String_or_char_ptr id, List *args, int default_args) {
  String *result = NewString("");
  bool conversion_operator = Strstr(id, "operator ") != 0 && !return_base_type;

  Parm *parm = args;
  int arg_idx = 0;
  while (parm) {
    String *type = Getattr(parm, "type");
    String *name = Getattr(parm, "name");
    if (!name && Cmp(type, "void")) {
      name = NewString("");
      Printf(name, "arg%d", arg_idx++);
      Setattr(parm, "name", name);
    }
    parm = nextSibling(parm);
  }

  String *rettype = Copy(decl);
  String *quals = SwigType_pop_function_qualifiers(rettype);
  String *qualifiers = 0;
  if (quals)
    qualifiers = SwigType_str(quals, 0);

  String *popped_decl = SwigType_pop_function(rettype);
  if (return_base_type)
    Append(rettype, return_base_type);

  if (!conversion_operator) {
    SwigType *rettype_stripped = SwigType_strip_qualifiers(rettype);
    String *rtype = SwigType_str(rettype, 0);
    Append(result, rtype);
    if (SwigType_issimple(rettype_stripped) && return_base_type)
      Append(result, " ");
    Delete(rtype);
    Delete(rettype_stripped);
  }

  if (id)
    Append(result, id);

  String *args_string = default_args ? ParmList_str_defaultargs(args) : ParmList_str(args);
  Printv(result, "(", args_string, ")", NIL);

  if (qualifiers)
    Printv(result, " ", qualifiers, NIL);

  // Reformat result to how it has been historically
  Replaceall(result, ",", ", ");
  Replaceall(result, "=", " = ");

  Delete(args_string);
  Delete(popped_decl);
  Delete(qualifiers);
  Delete(quals);
  Delete(rettype);
  return result;
}

/* -----------------------------------------------------------------------------
 * Swig_director_emit_dynamic_cast()
 *
 * In order to call protected virtual director methods from the target language, we need
 * to add an extra dynamic_cast to call the public C++ wrapper in the director class. 
 * Also for non-static protected members when the allprotected option is on.
 * ----------------------------------------------------------------------------- */

void Swig_director_emit_dynamic_cast(Node *n, Wrapper *f) {
  // TODO: why is the storage element removed in staticmemberfunctionHandler ??
  if ((!is_public(n) && (is_member_director(n) || GetFlag(n, "explicitcall"))) || 
      (is_non_virtual_protected_access(n) && !(Swig_storage_isstatic_custom(n, "staticmemberfunctionHandler:storage") || 
                                               Swig_storage_isstatic(n))
                                          && !Equal(nodeType(n), "constructor"))) {
    Node *parent = Getattr(n, "parentNode");
    String *dirname;
    String *dirdecl;
    dirname = Language::instance()->directorClassName(parent);
    dirdecl = NewStringf("%s *darg = 0", dirname);
    Wrapper_add_local(f, "darg", dirdecl);
    Printf(f->code, "darg = dynamic_cast<%s *>(arg1);\n", dirname);
    Delete(dirname);
    Delete(dirdecl);
  }
}

/* -----------------------------------------------------------------------------
 * Swig_director_parms_fixup()
 *
 * For each parameter in the C++ member function, copy the parameter name
 * to its "lname"; this ensures that Swig_typemap_attach_parms() will do
 * the right thing when it sees strings like "$1" in "directorin" typemaps.
 * ----------------------------------------------------------------------------- */

void Swig_director_parms_fixup(ParmList *parms) {
  Parm *p;
  int i;
  for (i = 0, p = parms; p; p = nextSibling(p), ++i) {
    String *arg = Getattr(p, "name");
    String *lname = 0;

    if (!arg && !Equal(Getattr(p, "type"), "void")) {
      lname = NewStringf("arg%d", i);
      Setattr(p, "name", lname);
    } else
      lname = Copy(arg);

    Setattr(p, "lname", lname);
    Delete(lname);
  }
}

