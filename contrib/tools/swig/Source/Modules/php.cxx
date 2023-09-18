/* -----------------------------------------------------------------------------
 * This file is part of SWIG, which is licensed as a whole under version 3
 * (or any later version) of the GNU General Public License. Some additional
 * terms also apply to certain portions of SWIG. The full details of the SWIG
 * license and copyrights can be found in the LICENSE and COPYRIGHT files
 * included with the SWIG source code as distributed by the SWIG developers
 * and at https://www.swig.org/legal.html.
 *
 * php.cxx
 *
 * PHP language module for SWIG.
 * -----------------------------------------------------------------------------
 */

#include "swigmod.h"
#include <algorithm>
#include <ctype.h>
#include <errno.h>
#include <limits.h>

static const char *usage = "\
PHP Options (available with -php7)\n\
     -prefix <prefix> - Prepend <prefix> to all class names in PHP wrappers\n\
\n";

// How to wrap non-class functions, variables and constants:
// FIXME: Make this specifiable and also allow a real namespace.

// Wrap as global PHP names.
static bool wrap_nonclass_global = true;

// Wrap in a class to fake a namespace (for compatibility with SWIG's behaviour
// before PHP added namespaces.
static bool wrap_nonclass_fake_class = true;

static String *module = 0;
static String *cap_module = 0;
static String *prefix = 0;

static File *f_begin = 0;
static File *f_runtime = 0;
static File *f_runtime_h = 0;
static File *f_h = 0;
static File *f_directors = 0;
static File *f_directors_h = 0;

static String *s_header;
static String *s_wrappers;
static String *s_init;
static String *r_init;		// RINIT user code
static String *s_shutdown;	// MSHUTDOWN user code
static String *r_shutdown;	// RSHUTDOWN user code
static String *s_vdecl;
static String *s_cinit;		// consttab initialization code.
static String *s_oinit;
static String *s_arginfo;
static String *s_entry;
static String *cs_entry;
static String *all_cs_entry;
static String *fake_cs_entry;
static String *s_creation;
static String *pragma_incl;
static String *pragma_code;
static String *pragma_phpinfo;
static String *pragma_version;

static String *class_name = NULL;
static String *base_class = NULL;
static String *destructor_action = NULL;
static String *magic_set = NULL;
static String *magic_get = NULL;
static String *magic_isset = NULL;

// Class used as pseudo-namespace for compatibility.
static String *fake_class_name() {
  static String *result = NULL;
  if (!result) {
    result = Len(prefix) ? prefix : module;
    if (!fake_cs_entry) {
      fake_cs_entry = NewStringf("static const zend_function_entry class_%s_functions[] = {\n", result);
    }

    Printf(s_creation, "static zend_class_entry *SWIG_Php_ce_%s;\n\n",result);

    Printf(s_oinit, "  INIT_CLASS_ENTRY(internal_ce, \"%s\", class_%s_functions);\n", result, result);
    Printf(s_oinit, "  SWIG_Php_ce_%s = zend_register_internal_class(&internal_ce);\n", result);
    Printf(s_oinit, "\n");
  }
  return result;
}

static String *swig_wrapped_interface_ce() {
  static String *result = NULL;
  if (!result) {
    result = NewStringf("SWIG_Php_swig_wrapped_interface_ce");
    Printf(s_oinit, "  INIT_CLASS_ENTRY(%s, \"SWIG\\\\wrapped\", NULL);\n", result);
  }
  return result;
}

/* To reduce code size (generated and compiled) we only want to emit each
 * different arginfo once, so we need to track which have been used.
 */
static Hash *arginfo_used;

/* Track non-class pointer types we need to to wrap */
static Hash *raw_pointer_types = 0;

static int shadow = 1;

// These static variables are used to pass some state from Handlers into functionWrapper
static enum {
  standard = 0,
  memberfn,
  staticmemberfn,
  membervar,
  staticmembervar,
  constructor,
  destructor,
  directorconstructor,
  directordisown
} wrapperType = standard;

extern "C" {
  static void (*r_prevtracefunc) (const SwigType *t, String *mangled, String *clientdata) = 0;
}

static void SwigPHP_emit_pointer_type_registrations() {
  if (!raw_pointer_types)
    return;

  Iterator ki = First(raw_pointer_types);
  if (!ki.key)
    return;

  Printf(s_wrappers, "/* class object handlers for pointer wrappers */\n");
  Printf(s_wrappers, "static zend_object_handlers swig_ptr_object_handlers;\n\n");

  Printf(s_wrappers, "/* Object Creation Method for pointer wrapping class */\n");
  Printf(s_wrappers, "static zend_object *swig_ptr_object_new(zend_class_entry *ce) {\n");
  Printf(s_wrappers, "  swig_object_wrapper *obj = (swig_object_wrapper*)zend_object_alloc(sizeof(swig_object_wrapper), ce);\n");
  Printf(s_wrappers, "  zend_object_std_init(&obj->std, ce);\n");
  Printf(s_wrappers, "  object_properties_init(&obj->std, ce);\n");
  Printf(s_wrappers, "  obj->std.handlers = &swig_ptr_object_handlers;\n");
  Printf(s_wrappers, "  obj->newobject = 0;\n");
  Printf(s_wrappers, "  return &obj->std;\n");
  Printf(s_wrappers, "}\n\n");

  Printf(s_wrappers, "/* Implement __toString equivalent, since that worked for the old-style resource wrapped pointers. */\n");
  Append(s_wrappers, "#if PHP_MAJOR_VERSION < 8\n");
  Printf(s_wrappers, "static int swig_ptr_cast_object(zval *z, zval *retval, int type) {\n");
  Append(s_wrappers, "#elif PHP_MAJOR_VERSION > 8 || PHP_MINOR_VERSION >= 2\n");
  Printf(s_wrappers, "static ZEND_RESULT_CODE swig_ptr_cast_object(zend_object *zobj, zval *retval, int type) {\n");
  Append(s_wrappers, "#else\n");
  Printf(s_wrappers, "static int swig_ptr_cast_object(zend_object *zobj, zval *retval, int type) {\n");
  Append(s_wrappers, "#endif\n");
  Printf(s_wrappers, "  if (type == IS_STRING) {\n");
  Append(s_wrappers, "#if PHP_MAJOR_VERSION < 8\n");
  Printf(s_wrappers, "    swig_object_wrapper *obj = SWIG_Z_FETCH_OBJ_P(z);\n");
  Append(s_wrappers, "#else\n");
  Printf(s_wrappers, "    swig_object_wrapper *obj = swig_php_fetch_object(zobj);\n");
  Append(s_wrappers, "#endif\n");
  Printv(s_wrappers, "    ZVAL_NEW_STR(retval, zend_strpprintf(0, \"SWIGPointer(%p,owned=%d)\", obj->ptr, obj->newobject));\n", NIL);
  Printf(s_wrappers, "    return SUCCESS;\n");
  Printf(s_wrappers, "  }\n");
  Printf(s_wrappers, "  return FAILURE;\n");
  Printf(s_wrappers, "}\n\n");

  Printf(s_oinit, "\n  /* Register classes to represent non-class pointer types */\n");
  Printf(s_oinit, "  swig_ptr_object_handlers = *zend_get_std_object_handlers();\n");
  Printf(s_oinit, "  swig_ptr_object_handlers.offset = XtOffsetOf(swig_object_wrapper, std);\n");
  Printf(s_oinit, "  swig_ptr_object_handlers.cast_object = swig_ptr_cast_object;\n");

  while (ki.key) {
    String *type = ki.key;

    String *swig_wrapped = swig_wrapped_interface_ce();
    Printf(s_creation, "/* class entry for pointer to %s */\n", type);
    Printf(s_creation, "static zend_class_entry *SWIG_Php_ce_%s;\n\n", type);

    Printf(s_oinit, "  INIT_CLASS_ENTRY(internal_ce, \"%s\\\\%s\", NULL);\n", "SWIG", type);
    Printf(s_oinit, "  SWIG_Php_ce_%s = zend_register_internal_class(&internal_ce);\n", type);
    Printf(s_oinit, "  SWIG_Php_ce_%s->create_object = swig_ptr_object_new;\n", type);
    Printv(s_oinit, "  zend_do_implement_interface(SWIG_Php_ce_", type, ", &", swig_wrapped, ");\n", NIL);
    Printf(s_oinit, "  SWIG_TypeClientData(SWIGTYPE%s,SWIG_Php_ce_%s);\n", type, type);
    Printf(s_oinit, "\n");

    ki = Next(ki);
  }
}

static Hash *create_php_type_flags() {
  Hash *h = NewHash();
  Setattr(h, "array", "MAY_BE_ARRAY");
  Setattr(h, "bool", "MAY_BE_BOOL");
  Setattr(h, "callable", "MAY_BE_CALLABLE");
  Setattr(h, "float", "MAY_BE_DOUBLE");
  Setattr(h, "int", "MAY_BE_LONG");
  Setattr(h, "iterable", "MAY_BE_ITERABLE");
  Setattr(h, "mixed", "MAY_BE_MIXED");
  Setattr(h, "null", "MAY_BE_NULL");
  Setattr(h, "object", "MAY_BE_OBJECT");
  Setattr(h, "resource", "MAY_BE_RESOURCE");
  Setattr(h, "string", "MAY_BE_STRING");
  Setattr(h, "void", "MAY_BE_VOID");
  return h;
}

static Hash *php_type_flags = create_php_type_flags();

// php_class + ":" + php_method -> PHPTypes*
// ":" + php_function -> PHPTypes*
static Hash *all_phptypes = NewHash();

// php_class_name -> php_parent_class_name
static Hash *php_parent_class = NewHash();

// Track if a method is directed in a descendent class.
// php_class + ":" + php_method -> boolean (using SetFlag()/GetFlag()).
static Hash *has_directed_descendent = NewHash();

// Track required return type for parent class methods.
// php_class + ":" + php_method -> List of php types.
static Hash *parent_class_method_return_type = NewHash();

// Class encapsulating the machinery to add PHP type declarations.
class PHPTypes {
  // List with an entry for each parameter and one for the return type.
  //
  // We assemble the types in here before emitting them so for an overloaded
  // function we combine the type declarations from each overloaded form.
  List *merged_types;

  // List with an entry for each parameter which is passed "byref" in any
  // overloaded form.  We use this to pass such parameters by reference in
  // the dispatch function.  If NULL, no parameters are passed by reference.
  List *byref;

  // The id string used in the name of the arginfo for this object.
  String *arginfo_id;

  // The feature:php:type value: 0, 1 or -1 for "compatibility".
  int php_type_flag;

  // Does the node for this have directorNode set?
  bool has_director_node;

  // Used to clamp the required number of parameters in the arginfo to be
  // compatible with any parent class version of the method.
  int num_required;

  int get_byref(int key) const {
    return byref && key < Len(byref) && Getitem(byref, key) != None;
  }

  int size() const {
    return std::max(Len(merged_types), Len(byref));
  }

  String *get_phptype(int key, String *classtypes, List *more_return_types = NULL) {
    Clear(classtypes);
    // We want to minimise the list of class types by not redundantly listing
    // a class for which a super-class is also listed.  This canonicalisation
    // allows for more sharing of arginfo (which reduces module size), makes
    // for a cleaner list if it's shown to the user, and also will speed up
    // module load a bit.
    Hash *classes = NewHash();
    DOH *types = Getitem(merged_types, key);
    String *result = NewStringEmpty();
    if (more_return_types) {
      if (types != None) {
	merge_type_lists(types, more_return_types);
      }
    }
    if (types != None) {
      SortList(types, NULL);
      String *prev = NULL;
      for (Iterator i = First(types); i.item; i = Next(i)) {
	if (prev && Equal(prev, i.item)) {
	  // Skip duplicates when merging.
	  continue;
	}
	String *c = Getattr(php_type_flags, i.item);
	if (c) {
	  if (Len(result) > 0) Append(result, "|");
	  Append(result, c);
	} else {
	  SetFlag(classes, i.item);
	}
	prev = i.item;
      }
    }

    // Remove entries for which a super-class is also listed.
    Iterator i = First(classes);
    while (i.key) {
      String *this_class = i.key;
      // We must advance the iterator early so we don't delete the element it
      // points to.
      i = Next(i);
      String *parent = this_class;
      while ((parent = Getattr(php_parent_class, parent)) != NULL) {
	if (GetFlag(classes, parent)) {
	  Delattr(classes, this_class);
	  break;
	}
      }
    }

    List *sorted_classes = SortedKeys(classes, Strcmp);
    for (i = First(sorted_classes); i.item; i = Next(i)) {
      if (Len(classtypes) > 0) Append(classtypes, "|");
      Append(classtypes, prefix);
      Append(classtypes, i.item);
    }
    Delete(sorted_classes);

    // Make the mask 0 if there are only class names specified.
    if (Len(result) == 0) {
      Append(result, "0");
    }
    return result;
  }

public:
  PHPTypes(Node *n)
    : merged_types(NewList()),
      byref(NULL),
      num_required(INT_MAX) {
    String *php_type_feature = Getattr(n, "feature:php:type");
    php_type_flag = 0;
    if (php_type_feature != NULL) {
      if (Equal(php_type_feature, "1")) {
	php_type_flag = 1;
      } else if (!Equal(php_type_feature, "0")) {
	php_type_flag = -1;
      }
    }
    arginfo_id = Copy(Getattr(n, "sym:name"));
    has_director_node = (Getattr(n, "directorNode") != NULL);
  }

  ~PHPTypes() {
    Delete(merged_types);
    Delete(byref);
  }

  void adjust(int num_required_, bool php_constructor) {
    num_required = std::min(num_required, num_required_);
    if (php_constructor) {
      // Don't add a return type declaration for a PHP __construct method
      // (because there it has no return type as far as PHP is concerned).
      php_type_flag = 0;
    }
  }

  String *get_arginfo_id() const {
    return arginfo_id;
  }

  // key is 0 for return type, or >= 1 for parameters numbered from 1
  List *process_phptype(Node *n, int key, const String_or_char *attribute_name);

  // Merge entries from o_merge_list into merge_list, skipping any entries
  // already present.
  //
  // Both merge_list and o_merge_list should be in sorted order.
  static void merge_type_lists(List *merge_list, List *o_merge_list);

  void merge_from(const PHPTypes* o);

  void set_byref(int key) {
    if (!byref) {
      byref = NewList();
    }
    while (Len(byref) <= key) {
      Append(byref, None);
    }
    // If any overload takes a particular parameter by reference then the
    // dispatch function also needs to take that parameter by reference so
    // we can just set unconditionally here.
    Setitem(byref, key, ""); // Just needs to be something != None.
  }

  void emit_arginfo(DOH *item, String *key) {
    Setmark(item, 1);
    char *colon_ptr = Strchr(key, ':');
    assert(colon_ptr);
    int colon = (int)(colon_ptr - Char(key));
    if (colon > 0 && Strcmp(colon_ptr + 1, "__construct") != 0) {
      // See if there's a parent class which implements this method, and if so
      // emit its arginfo and then merge its PHPTypes into ours as we need to
      // be compatible with it (whether it is virtual or not).
      String *this_class = NewStringWithSize(Char(key), colon);
      String *parent = this_class;
      while ((parent = Getattr(php_parent_class, parent)) != NULL) {
	String *k = NewStringf("%s%s", parent, colon_ptr);
	DOH *item = Getattr(all_phptypes, k);
	if (item) {
	  PHPTypes *p = (PHPTypes*)Data(item);
	  if (!Getmark(item)) {
	    p->emit_arginfo(item, k);
	  }
	  merge_from(p);
	  Delete(k);
	  break;
	}
	Delete(k);
      }
      Delete(this_class);
    }

    // We want to only emit each different arginfo once, as that reduces the
    // size of both the generated source code and the compiled extension
    // module.  The parameters at this level are just named arg1, arg2, etc
    // so the arginfo will be the same for any function with the same number
    // of parameters and (if present) PHP type declarations for parameters and
    // return type.
    //
    // We generate the arginfo we want (taking care to normalise, e.g. the
    // lists of types are unique and in sorted order), then use the
    // arginfo_used Hash to see if we've already generated it.
    String *out_phptype = NULL;
    String *out_phpclasses = NewStringEmpty();

    // We provide a simple way to generate PHP return type declarations
    // except for directed methods.  The point of directors is to allow
    // subclassing in the target language, and if the wrapped method has
    // a return type declaration then an overriding method in user code
    // needs to have a compatible declaration.
    //
    // The upshot of this is that enabling return type declarations for
    // existing bindings would break compatibility with user code written
    // for an older version.  For parameters however the situation is
    // different because if the parent class declares types for parameters
    // a subclass overriding the function will be compatible whether it
    // declares them or not.
    //
    // directorNode being present seems to indicate if this method or one
    // it inherits from is directed, which is what we care about here.
    // Using (!is_member_director(n)) would get it wrong for testcase
    // director_frob.
    if (php_type_flag && (php_type_flag > 0 || !has_director_node)) {
      if (!GetFlag(has_directed_descendent, key)) {
	out_phptype = get_phptype(0, out_phpclasses, Getattr(parent_class_method_return_type, key));
      }
    }

    // ### in arginfo_code will be replaced with the id once that is known.
    String *arginfo_code = NewStringEmpty();
    if (out_phptype) {
      if (Len(out_phpclasses)) {
	Replace(out_phpclasses, "\\", "\\\\", DOH_REPLACE_ANY);
	Printf(arginfo_code, "ZEND_BEGIN_ARG_WITH_RETURN_OBJ_TYPE_MASK_EX(swig_arginfo_###, 0, %d, %s, %s)\n", num_required, out_phpclasses, out_phptype);
      } else {
	Printf(arginfo_code, "ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(swig_arginfo_###, 0, %d, %s)\n", num_required, out_phptype);
      }
    } else {
      Printf(arginfo_code, "ZEND_BEGIN_ARG_INFO_EX(swig_arginfo_###, 0, 0, %d)\n", num_required);
    }

    int phptypes_size = size();
    for (int param_count = 1; param_count < phptypes_size; ++param_count) {
      String *phpclasses = NewStringEmpty();
      String *phptype = get_phptype(param_count, phpclasses);

      int byref = get_byref(param_count);

      // FIXME: Should we be doing byref for return value as well?

      if (phptype) {
	if (Len(phpclasses)) {
	  // We need to double any backslashes (which are PHP namespace
	  // separators) in the PHP class names as they get turned into
	  // C strings by the ZEND_ARG_OBJ_TYPE_MASK macro.
	  Replace(phpclasses, "\\", "\\\\", DOH_REPLACE_ANY);
	  Printf(arginfo_code, " ZEND_ARG_OBJ_TYPE_MASK(%d,arg%d,%s,%s,NULL)\n", byref, param_count, phpclasses, phptype);
	} else {
	  Printf(arginfo_code, " ZEND_ARG_TYPE_MASK(%d,arg%d,%s,NULL)\n", byref, param_count, phptype);
	}
      } else {
	Printf(arginfo_code, " ZEND_ARG_INFO(%d,arg%d)\n", byref, param_count);
      }
    }
    Printf(arginfo_code, "ZEND_END_ARG_INFO()\n");

    String *arginfo_id_same = Getattr(arginfo_used, arginfo_code);
    if (arginfo_id_same) {
      Printf(s_arginfo, "#define swig_arginfo_%s swig_arginfo_%s\n", arginfo_id, arginfo_id_same);
    } else {
      // Not had this arginfo before.
      Setattr(arginfo_used, arginfo_code, arginfo_id);
      arginfo_code = Copy(arginfo_code);
      Replace(arginfo_code, "###", arginfo_id, DOH_REPLACE_FIRST);
      Append(s_arginfo, arginfo_code);
    }
    Delete(arginfo_code);
    arginfo_code = NULL;
  }
};

static PHPTypes *phptypes = NULL;

class PHP : public Language {
public:
  PHP() {
    director_language = 1;
  }

  /* ------------------------------------------------------------
   * main()
   * ------------------------------------------------------------ */

  virtual void main(int argc, char *argv[]) {
    SWIG_library_directory("php");

    for (int i = 1; i < argc; i++) {
      if (strcmp(argv[i], "-prefix") == 0) {
	if (argv[i + 1]) {
	  prefix = NewString(argv[i + 1]);
	  Swig_mark_arg(i);
	  Swig_mark_arg(i + 1);
	  i++;
	} else {
	  Swig_arg_error();
	}
      } else if ((strcmp(argv[i], "-noshadow") == 0)) {
	shadow = 0;
	Swig_mark_arg(i);
      } else if (strcmp(argv[i], "-help") == 0) {
	fputs(usage, stdout);
      }
    }

    Preprocessor_define("SWIGPHP 1", 0);
    Preprocessor_define("SWIGPHP7 1", 0);
    SWIG_typemap_lang("php");
    SWIG_config_file("php.swg");
    allow_overloading();
  }

  /* ------------------------------------------------------------
   * top()
   * ------------------------------------------------------------ */

  virtual int top(Node *n) {

    String *filen;

    /* Check if directors are enabled for this module. */
    Node *mod = Getattr(n, "module");
    if (mod) {
      Node *options = Getattr(mod, "options");
      if (options && Getattr(options, "directors")) {
	allow_directors();
      }
    }

    /* Set comparison with null for ConstructorToFunction */
    setSubclassInstanceCheck(NewString("Z_TYPE_P($arg) != IS_NULL"));

    /* Initialize all of the output files */
    String *outfile = Getattr(n, "outfile");
    String *outfile_h = Getattr(n, "outfile_h");

    /* main output file */
    f_begin = NewFile(outfile, "w", SWIG_output_files());
    if (!f_begin) {
      FileErrorDisplay(outfile);
      Exit(EXIT_FAILURE);
    }
    f_runtime = NewStringEmpty();

    /* sections of the output file */
    s_init = NewStringEmpty();
    r_init = NewStringEmpty();
    s_shutdown = NewStringEmpty();
    r_shutdown = NewStringEmpty();
    s_header = NewString("/* header section */\n");
    s_wrappers = NewString("/* wrapper section */\n");
    s_creation = NewStringEmpty();
    /* subsections of the init section */
    s_vdecl = NewString("/* vdecl subsection */\n");
    s_cinit = NewString("  /* cinit subsection */\n");
    s_oinit = NewString("  /* oinit subsection */\n");
    pragma_phpinfo = NewStringEmpty();
    f_directors_h = NewStringEmpty();
    f_directors = NewStringEmpty();

    if (directorsEnabled()) {
      f_runtime_h = NewFile(outfile_h, "w", SWIG_output_files());
      if (!f_runtime_h) {
	FileErrorDisplay(outfile_h);
	Exit(EXIT_FAILURE);
      }
    }

    /* Register file targets with the SWIG file handler */
    Swig_register_filebyname("begin", f_begin);
    Swig_register_filebyname("runtime", f_runtime);
    Swig_register_filebyname("init", s_init);
    Swig_register_filebyname("rinit", r_init);
    Swig_register_filebyname("shutdown", s_shutdown);
    Swig_register_filebyname("rshutdown", r_shutdown);
    Swig_register_filebyname("header", s_header);
    Swig_register_filebyname("wrapper", s_wrappers);
    Swig_register_filebyname("director", f_directors);
    Swig_register_filebyname("director_h", f_directors_h);

    Swig_banner(f_begin);

    Swig_obligatory_macros(f_runtime, "PHP");

    if (directorsEnabled()) {
      Printf(f_runtime, "#define SWIG_DIRECTORS\n");
    }

    // We need to include php.h before string.h gets included, at least with
    // PHP 8.2.  Otherwise string.h is included without _GNU_SOURCE being
    // included and memrchr() doesn't get declared, and then inline code in
    // the PHP headers defines _GNU_SOURCE, includes string.h (which is a
    // no op thanks to the include gaurds), then tries to use memrchr() and
    // fails.
    //
    // We also need to suppress -Wdeclaration-after-statement if enabled
    // since with PHP 8.2 zend_operators.h contains inline code which triggers
    // this warning and our testsuite uses with option and -Werror.  I don't
    // see a good way to only do this within our testsuite, but disabling
    // it globally like this shouldn't be problematic.
    Append(f_runtime,
	   "\n"
	   "#if defined __GNUC__ && !defined __cplusplus\n"
	   "# if __GNUC__ >= 4\n"
	   "#  pragma GCC diagnostic push\n"
	   "#  pragma GCC diagnostic ignored \"-Wdeclaration-after-statement\"\n"
	   "# endif\n"
	   "#endif\n"
	   "#include \"php.h\"\n"
	   "#if defined __GNUC__ && !defined __cplusplus\n"
	   "# if __GNUC__ >= 4\n"
	   "#  pragma GCC diagnostic pop\n"
	   "# endif\n"
	   "#endif\n\n");

    /* Set the module name */
    module = Copy(Getattr(n, "name"));
    cap_module = NewStringf("%(upper)s", module);
    if (!prefix)
      prefix = NewStringEmpty();

    if (directorsEnabled()) {
      Swig_banner(f_directors_h);
      Printf(f_directors_h, "\n");
      Printf(f_directors_h, "#ifndef SWIG_%s_WRAP_H_\n", cap_module);
      Printf(f_directors_h, "#define SWIG_%s_WRAP_H_\n\n", cap_module);

      String *filename = Swig_file_filename(outfile_h);
      Printf(f_directors, "\n#include \"%s\"\n\n", filename);
      Delete(filename);
    }

    /* sub-sections of the php file */
    pragma_code = NewStringEmpty();
    pragma_incl = NewStringEmpty();
    pragma_version = NULL;

    /* Initialize the rest of the module */

    /* start the header section */
    Printf(s_header, "#define SWIG_name  \"%s\"\n", module);
    Printf(s_header, "#ifdef __cplusplus\n");
    Printf(s_header, "extern \"C\" {\n");
    Printf(s_header, "#endif\n");
    Printf(s_header, "#include \"php_ini.h\"\n");
    Printf(s_header, "#include \"ext/standard/info.h\"\n");
    Printf(s_header, "#include \"php_%s.h\"\n", module);
    Printf(s_header, "#ifdef __cplusplus\n");
    Printf(s_header, "}\n");
    Printf(s_header, "#endif\n\n");

    if (directorsEnabled()) {
      // Insert director runtime
      Swig_insert_file("director_common.swg", s_header);
      Swig_insert_file("director.swg", s_header);
    }

    /* Create the .h file too */
    filen = NewStringEmpty();
    Printv(filen, SWIG_output_directory(), "php_", module, ".h", NIL);
    f_h = NewFile(filen, "w", SWIG_output_files());
    if (!f_h) {
      FileErrorDisplay(filen);
      Exit(EXIT_FAILURE);
    }

    Swig_banner(f_h);

    Printf(f_h, "\n");
    Printf(f_h, "#ifndef PHP_%s_H\n", cap_module);
    Printf(f_h, "#define PHP_%s_H\n\n", cap_module);
    Printf(f_h, "extern zend_module_entry %s_module_entry;\n", module);
    Printf(f_h, "#define phpext_%s_ptr &%s_module_entry\n\n", module, module);
    Printf(f_h, "#ifdef PHP_WIN32\n");
    Printf(f_h, "# define PHP_%s_API __declspec(dllexport)\n", cap_module);
    Printf(f_h, "#else\n");
    Printf(f_h, "# define PHP_%s_API\n", cap_module);
    Printf(f_h, "#endif\n\n");

    /* start the arginfo section */
    s_arginfo = NewString("/* arginfo subsection */\n");
    arginfo_used = NewHash();

    /* start the function entry section */
    s_entry = NewString("/* entry subsection */\n");

    /* holds all the per-class function entry sections */
    all_cs_entry = NewString("/* class entry subsection */\n");
    cs_entry = NULL;
    fake_cs_entry = NULL;

    Printf(s_entry, "/* Every non-class user visible function must have an entry here */\n");
    Printf(s_entry, "static const zend_function_entry module_%s_functions[] = {\n", module);

    /* Emit all of the code */
    Language::top(n);

    /* Emit all the arginfo.  We sort the keys so the output order doesn't depend on
     * hashkey order.
     */
    {
      List *sorted_keys = SortedKeys(all_phptypes, Strcmp);
      for (Iterator k = First(sorted_keys); k.item; k = Next(k)) {
	DOH *val = Getattr(all_phptypes, k.item);
	if (!Getmark(val)) {
	  PHPTypes *p = (PHPTypes*)Data(val);
	  p->emit_arginfo(val, k.item);
	}
      }
      Delete(sorted_keys);
    }

    SwigPHP_emit_pointer_type_registrations();
    Dump(s_creation, s_header);
    Delete(s_creation);
    s_creation = NULL;

    /* start the init section */
    {
      String *s_init_old = s_init;
      s_init = NewString("/* init section */\n");
      Printv(s_init, "zend_module_entry ", module, "_module_entry = {\n", NIL);
      Printf(s_init, "    STANDARD_MODULE_HEADER,\n");
      Printf(s_init, "    \"%s\",\n", module);
      Printf(s_init, "    module_%s_functions,\n", module);
      Printf(s_init, "    PHP_MINIT(%s),\n", module);
      if (Len(s_shutdown) > 0) {
	Printf(s_init, "    PHP_MSHUTDOWN(%s),\n", module);
      } else {
	Printf(s_init, "    NULL, /* No MSHUTDOWN code */\n");
      }
      if (Len(r_init) > 0) {
	Printf(s_init, "    PHP_RINIT(%s),\n", module);
      } else {
	Printf(s_init, "    NULL, /* No RINIT code */\n");
      }
      if (Len(r_shutdown) > 0) {
	Printf(s_init, "    PHP_RSHUTDOWN(%s),\n", module);
      } else {
	Printf(s_init, "    NULL, /* No RSHUTDOWN code */\n");
      }
      if (Len(pragma_phpinfo) > 0) {
	Printf(s_init, "    PHP_MINFO(%s),\n", module);
      } else {
	Printf(s_init, "    NULL, /* No MINFO code */\n");
      }
      if (Len(pragma_version) > 0) {
	Printf(s_init, "    \"%s\",\n", pragma_version);
      } else {
	Printf(s_init, "    NO_VERSION_YET,\n");
      }
      Printf(s_init, "    STANDARD_MODULE_PROPERTIES\n");
      Printf(s_init, "};\n\n");

      Printf(s_init, "#ifdef __cplusplus\n");
      Printf(s_init, "extern \"C\" {\n");
      Printf(s_init, "#endif\n");
      // We want to write "SWIGEXPORT ZEND_GET_MODULE(%s)" but ZEND_GET_MODULE
      // in PHP7 has "extern "C" { ... }" around it so we can't do that.
      Printf(s_init, "SWIGEXPORT zend_module_entry *get_module(void) { return &%s_module_entry; }\n", module);
      Printf(s_init, "#ifdef __cplusplus\n");
      Printf(s_init, "}\n");
      Printf(s_init, "#endif\n\n");

      Printf(s_init, "#define SWIG_php_minit PHP_MINIT_FUNCTION(%s)\n\n", module);

      Printv(s_init, s_init_old, NIL);
      Delete(s_init_old);
    }

    /* We have to register the constants before they are (possibly) used
     * by the pointer typemaps. This all needs re-arranging really as
     * things are being called in the wrong order
     */

    Printf(s_oinit, "  /* end oinit subsection */\n");
    Printf(s_init, "%s\n", s_oinit);

    /* Constants generated during top call */
    Printf(s_cinit, "  /* end cinit subsection */\n");
    Printf(s_init, "%s\n", s_cinit);
    Clear(s_cinit);
    Delete(s_cinit);

    Printf(s_init, "  return SUCCESS;\n");
    Printf(s_init, "}\n\n");

    // Now do REQUEST init which holds any user specified %rinit, and also vinit
    if (Len(r_init) > 0) {
      Printf(f_h, "PHP_RINIT_FUNCTION(%s);\n", module);

      Printf(s_init, "PHP_RINIT_FUNCTION(%s)\n{\n", module);
      Printv(s_init,
	     "/* rinit section */\n",
	     r_init, "\n",
	     NIL);

      Printf(s_init, "  return SUCCESS;\n");
      Printf(s_init, "}\n\n");
    }

    Printf(f_h, "PHP_MINIT_FUNCTION(%s);\n", module);

    if (Len(s_shutdown) > 0) {
      Printf(f_h, "PHP_MSHUTDOWN_FUNCTION(%s);\n", module);

      Printv(s_init, "PHP_MSHUTDOWN_FUNCTION(", module, ")\n"
		     "/* shutdown section */\n"
		     "{\n",
		     s_shutdown,
		     "  return SUCCESS;\n"
		     "}\n\n", NIL);
    }

    if (Len(r_shutdown) > 0) {
      Printf(f_h, "PHP_RSHUTDOWN_FUNCTION(%s);\n", module);

      Printf(s_init, "PHP_RSHUTDOWN_FUNCTION(%s)\n{\n", module);
      Printf(s_init, "/* rshutdown section */\n");
      Printf(s_init, "%s\n", r_shutdown);
      Printf(s_init, "    return SUCCESS;\n");
      Printf(s_init, "}\n\n");
    }

    if (Len(pragma_phpinfo) > 0) {
      Printf(f_h, "PHP_MINFO_FUNCTION(%s);\n", module);

      Printf(s_init, "PHP_MINFO_FUNCTION(%s)\n{\n", module);
      Printf(s_init, "%s", pragma_phpinfo);
      Printf(s_init, "}\n");
    }

    Printf(s_init, "/* end init section */\n");

    Printf(f_h, "\n#endif /* PHP_%s_H */\n", cap_module);

    Delete(f_h);

    String *type_table = NewStringEmpty();
    SwigType_emit_type_table(f_runtime, type_table);
    Printf(s_header, "%s", type_table);
    Delete(type_table);

    /* Oh dear, more things being called in the wrong order. This whole
     * function really needs totally redoing.
     */

    if (directorsEnabled()) {
      Dump(f_directors_h, f_runtime_h);
      Printf(f_runtime_h, "\n");
      Printf(f_runtime_h, "#endif\n");
      Delete(f_runtime_h);
    }

    Printf(s_header, "/* end header section */\n");
    Printf(s_wrappers, "/* end wrapper section */\n");
    Printf(s_vdecl, "/* end vdecl subsection */\n");

    Dump(f_runtime, f_begin);
    Printv(f_begin, s_header, NIL);
    if (directorsEnabled()) {
      Dump(f_directors, f_begin);
    }
    Printv(f_begin, s_vdecl, s_wrappers, NIL);
    Printv(f_begin, s_arginfo, "\n\n", all_cs_entry, "\n\n", s_entry,
	" ZEND_FE_END\n};\n\n", NIL);
    if (fake_cs_entry) {
      Printv(f_begin, fake_cs_entry, " ZEND_FE_END\n};\n\n", NIL);
      Delete(fake_cs_entry);
      fake_cs_entry = NULL;
    }
    Printv(f_begin, s_init, NIL);
    Delete(s_header);
    Delete(s_wrappers);
    Delete(s_init);
    Delete(s_vdecl);
    Delete(all_cs_entry);
    Delete(s_entry);
    Delete(s_arginfo);
    Delete(f_runtime);
    Delete(f_begin);
    Delete(arginfo_used);

    if (Len(pragma_incl) > 0 || Len(pragma_code) > 0) {
      /* PHP module file */
      String *php_filename = NewStringEmpty();
      Printv(php_filename, SWIG_output_directory(), module, ".php", NIL);

      File *f_phpcode = NewFile(php_filename, "w", SWIG_output_files());
      if (!f_phpcode) {
	FileErrorDisplay(php_filename);
	Exit(EXIT_FAILURE);
      }

      Printf(f_phpcode, "<?php\n\n");

      if (Len(pragma_incl) > 0) {
	Printv(f_phpcode, pragma_incl, "\n", NIL);
      }

      if (Len(pragma_code) > 0) {
	Printv(f_phpcode, pragma_code, "\n", NIL);
      }

      Delete(f_phpcode);
      Delete(php_filename);
    }

    return SWIG_OK;
  }

  /* Just need to append function names to function table to register with PHP. */
  void create_command(String *cname, String *fname, Node *n, bool dispatch, String *modes = NULL) {
    // This is for the single main zend_function_entry record
    ParmList *l = Getattr(n, "parms");
    if (cname && !Equal(Getattr(n, "storage"), "friend")) {
      Printf(f_h, "static PHP_METHOD(%s%s,%s);\n", prefix, cname, fname);
      if (wrapperType != staticmemberfn &&
	  wrapperType != staticmembervar &&
	  !Equal(fname, "__construct")) {
	// Skip the first entry in the parameter list which is the this pointer.
	if (l) l = Getattr(l, "tmap:in:next");
	// FIXME: does this throw the phptype key value off?
      }
    } else {
      if (dispatch) {
	Printf(f_h, "static ZEND_NAMED_FUNCTION(%s);\n", fname);
      } else {
	Printf(f_h, "static PHP_FUNCTION(%s);\n", fname);
      }
    }

    phptypes->adjust(emit_num_required(l), Equal(fname, "__construct") ? true : false);

    String *arginfo_id = phptypes->get_arginfo_id();
    String *s = cs_entry;
    if (!s) s = s_entry;
    if (cname && !Equal(Getattr(n, "storage"), "friend")) {
      Printf(all_cs_entry, " PHP_ME(%s%s,%s,swig_arginfo_%s,%s)\n", prefix, cname, fname, arginfo_id, modes);
    } else {
      if (dispatch) {
	if (wrap_nonclass_global) {
	  Printf(s, " ZEND_NAMED_FE(%(lower)s,%s,swig_arginfo_%s)\n", Getattr(n, "sym:name"), fname, arginfo_id);
	}

	if (wrap_nonclass_fake_class) {
	  (void)fake_class_name();
	  Printf(fake_cs_entry, " ZEND_NAMED_ME(%(lower)s,%s,swig_arginfo_%s,ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)\n", Getattr(n, "sym:name"), fname, arginfo_id);
	}
      } else {
	if (wrap_nonclass_global) {
	  Printf(s, " PHP_FE(%s,swig_arginfo_%s)\n", fname, arginfo_id);
	}

	if (wrap_nonclass_fake_class) {
	  String *fake_class = fake_class_name();
	  Printf(fake_cs_entry, " PHP_ME(%s,%s,swig_arginfo_%s,ZEND_ACC_PUBLIC|ZEND_ACC_STATIC)\n", fake_class, fname, arginfo_id);
	}
      }
    }
  }

  /* ------------------------------------------------------------
   * dispatchFunction()
   * ------------------------------------------------------------ */
  void dispatchFunction(Node *n, int constructor) {
    /* Last node in overloaded chain */

    int maxargs;
    String *tmp = NewStringEmpty();
    String *dispatch = Swig_overload_dispatch(n, "%s(INTERNAL_FUNCTION_PARAM_PASSTHRU); return;", &maxargs);

    /* Generate a dispatch wrapper for all overloaded functions */

    Wrapper *f = NewWrapper();
    String *symname = Getattr(n, "sym:name");
    String *wname = NULL;
    String *modes = NULL;
    bool constructorRenameOverload = false;

    if (constructor) {
      if (!Equal(class_name, Getattr(n, "constructorHandler:sym:name"))) {
	// Renamed constructor - turn into static factory method
	constructorRenameOverload = true;
	wname = Copy(Getattr(n, "constructorHandler:sym:name"));
      } else {
	wname = NewString("__construct");
      }
    } else if (class_name) {
      wname = Getattr(n, "wrapper:method:name");
    } else {
      wname = Swig_name_wrapper(symname);
    }

    if (constructor) {
      modes = NewString("ZEND_ACC_PUBLIC | ZEND_ACC_CTOR");
      if (constructorRenameOverload) {
	Append(modes, " | ZEND_ACC_STATIC");
      }
    } else if (wrapperType == staticmemberfn || Cmp(Getattr(n, "storage"), "static") == 0) {
      modes = NewString("ZEND_ACC_PUBLIC | ZEND_ACC_STATIC");
    } else {
      modes = NewString("ZEND_ACC_PUBLIC");
    }

    create_command(class_name, wname, n, true, modes);

    if (class_name && !Equal(Getattr(n, "storage"), "friend")) {
      Printv(f->def, "static PHP_METHOD(", prefix, class_name, ",", wname, ") {\n", NIL);
    } else {
      Printv(f->def, "static ZEND_NAMED_FUNCTION(", wname, ") {\n", NIL);
    }

    Wrapper_add_local(f, "argc", "int argc");

    Printf(tmp, "zval argv[%d]", maxargs);
    Wrapper_add_local(f, "argv", tmp);

    Printf(f->code, "argc = ZEND_NUM_ARGS();\n");

    Printf(f->code, "zend_get_parameters_array_ex(argc, argv);\n");

    Replaceall(dispatch, "$args", "self,args");

    Printv(f->code, dispatch, "\n", NIL);

    Printf(f->code, "zend_throw_exception(zend_ce_type_error, \"No matching function for overloaded '%s'\", 0);\n", symname);
    Printv(f->code, "fail:\n", NIL);
    Printv(f->code, "return;\n", NIL);
    Printv(f->code, "}\n", NIL);
    Wrapper_print(f, s_wrappers);

    DelWrapper(f);
    Delete(dispatch);
    Delete(tmp);
  }

  /* ------------------------------------------------------------
   * functionWrapper()
   * ------------------------------------------------------------ */

  /* Helper function to check if class is wrapped */
  bool is_class_wrapped(String *className) {
    if (!className)
      return false;
    Node *n = symbolLookup(className);
    return n && Getattr(n, "classtype") != NULL;
  }

  void generate_magic_property_methods(Node *class_node) {
    String *swig_base = base_class;
    if (Equal(swig_base, "Exception") || !is_class_wrapped(swig_base)) {
      swig_base = NULL;
    }

    static bool generated_magic_arginfo = false;
    if (!generated_magic_arginfo) {
      // Create arginfo entries for __get, __set and __isset.
      Append(s_arginfo,
	     "ZEND_BEGIN_ARG_INFO_EX(swig_magic_arginfo_get, 0, 0, 1)\n"
	     " ZEND_ARG_TYPE_MASK(0,arg1,MAY_BE_STRING,NULL)\n"
	     "ZEND_END_ARG_INFO()\n");
      Append(s_arginfo,
	     "ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(swig_magic_arginfo_set, 0, 1, MAY_BE_VOID)\n"
	     " ZEND_ARG_TYPE_MASK(0,arg1,MAY_BE_STRING,NULL)\n"
	     " ZEND_ARG_INFO(0,arg2)\n"
	     "ZEND_END_ARG_INFO()\n");
      Append(s_arginfo,
	     "ZEND_BEGIN_ARG_WITH_RETURN_TYPE_MASK_EX(swig_magic_arginfo_isset, 0, 1, MAY_BE_BOOL)\n"
	     " ZEND_ARG_TYPE_MASK(0,arg1,MAY_BE_STRING,NULL)\n"
	     "ZEND_END_ARG_INFO()\n");
      generated_magic_arginfo = true;
    }

    Wrapper *f = NewWrapper();

    Printf(f_h, "PHP_METHOD(%s%s,__set);\n", prefix, class_name);
    Printf(all_cs_entry, " PHP_ME(%s%s,__set,swig_magic_arginfo_set,ZEND_ACC_PUBLIC)\n", prefix, class_name);
    Printf(f->code, "PHP_METHOD(%s%s,__set) {\n", prefix, class_name);

    Printf(f->code, "  swig_object_wrapper *arg = SWIG_Z_FETCH_OBJ_P(ZEND_THIS);\n");
    Printf(f->code, "  zval args[2];\n zval tempZval;\n  zend_string *arg2 = 0;\n\n");
    Printf(f->code, "  if(ZEND_NUM_ARGS() != 2 || zend_get_parameters_array_ex(2, args) != SUCCESS) {\n");
    Printf(f->code, "\tWRONG_PARAM_COUNT;\n}\n\n");
    Printf(f->code, "  if (!arg) {\n");
    Printf(f->code, "    zend_throw_exception(zend_ce_type_error, \"this pointer is NULL\", 0);\n");
    Printf(f->code, "    return;\n");
    Printf(f->code, "  }\n");
    Printf(f->code, "  arg2 = Z_STR(args[0]);\n\n");

    Printf(f->code, "if (!arg2) {\n  RETVAL_NULL();\n}\n");
    if (magic_set) {
      Append(f->code, magic_set);
    }
    Printf(f->code, "\nelse if (strcmp(ZSTR_VAL(arg2),\"thisown\") == 0) {\n");
    Printf(f->code, "arg->newobject = zval_get_long(&args[1]);\n");
    if (Swig_directorclass(class_node)) {
      Printv(f->code, "if (arg->newobject == 0) {\n",
		      "  Swig::Director *director = SWIG_DIRECTOR_CAST((", Getattr(class_node, "classtype"), "*)(arg->ptr));\n",
		      "  if (director) director->swig_disown();\n",
		      "}\n", NIL);
    }
    if (swig_base) {
      Printf(f->code, "} else {\nPHP_MN(%s%s___set)(INTERNAL_FUNCTION_PARAM_PASSTHRU);\n", prefix, swig_base);
    } else if (Getattr(class_node, "feature:php:allowdynamicproperties")) {
      Printf(f->code, "} else {\nadd_property_zval_ex(ZEND_THIS, ZSTR_VAL(arg2), ZSTR_LEN(arg2), &args[1]);\n");
    }
    Printf(f->code, "}\n");

    Printf(f->code, "fail:\n");
    Printf(f->code, "return;\n");
    Printf(f->code, "}\n\n\n");


    Printf(f_h, "PHP_METHOD(%s%s,__get);\n", prefix, class_name);
    Printf(all_cs_entry, " PHP_ME(%s%s,__get,swig_magic_arginfo_get,ZEND_ACC_PUBLIC)\n", prefix, class_name);
    Printf(f->code, "PHP_METHOD(%s%s,__get) {\n",prefix, class_name);

    Printf(f->code, "  swig_object_wrapper *arg = SWIG_Z_FETCH_OBJ_P(ZEND_THIS);\n");
    Printf(f->code, "  zval args[1];\n zval tempZval;\n  zend_string *arg2 = 0;\n\n");
    Printf(f->code, "  if(ZEND_NUM_ARGS() != 1 || zend_get_parameters_array_ex(1, args) != SUCCESS) {\n");
    Printf(f->code, "\tWRONG_PARAM_COUNT;\n}\n\n");
    Printf(f->code, "  if (!arg) {\n");
    Printf(f->code, "    zend_throw_exception(zend_ce_type_error, \"this pointer is NULL\", 0);\n");
    Printf(f->code, "    return;\n");
    Printf(f->code, "  }\n");
    Printf(f->code, "  arg2 = Z_STR(args[0]);\n\n");

    Printf(f->code, "if (!arg2) {\n  RETVAL_NULL();\n}\n");
    if (magic_get) {
      Append(f->code, magic_get);
    }
    Printf(f->code, "\nelse if (strcmp(ZSTR_VAL(arg2),\"thisown\") == 0) {\n");
    Printf(f->code, "if(arg->newobject) {\nRETVAL_LONG(1);\n}\nelse {\nRETVAL_LONG(0);\n}\n}\n\n");
    Printf(f->code, "else {\n");
    if (swig_base) {
      Printf(f->code, "PHP_MN(%s%s___get)(INTERNAL_FUNCTION_PARAM_PASSTHRU);\n}\n", prefix, swig_base);
    } else {
      // __get is only called if the property isn't set on the zend_object.
      Printf(f->code, "RETVAL_NULL();\n}\n");
    }

    Printf(f->code, "fail:\n");
    Printf(f->code, "return;\n");
    Printf(f->code, "}\n\n\n");


    Printf(f_h, "PHP_METHOD(%s%s,__isset);\n", prefix, class_name);
    Printf(all_cs_entry, " PHP_ME(%s%s,__isset,swig_magic_arginfo_isset,ZEND_ACC_PUBLIC)\n", prefix, class_name);
    Printf(f->code, "PHP_METHOD(%s%s,__isset) {\n",prefix, class_name);

    Printf(f->code, "  swig_object_wrapper *arg = SWIG_Z_FETCH_OBJ_P(ZEND_THIS);\n");
    Printf(f->code, "  zval args[1];\n  zend_string *arg2 = 0;\n\n");
    Printf(f->code, "  if(ZEND_NUM_ARGS() != 1 || zend_get_parameters_array_ex(1, args) != SUCCESS) {\n");
    Printf(f->code, "\tWRONG_PARAM_COUNT;\n}\n\n");
    Printf(f->code, "  if(!arg) {\n");
    Printf(f->code, "    zend_throw_exception(zend_ce_type_error, \"this pointer is NULL\", 0);\n");
    Printf(f->code, "    return;\n");
    Printf(f->code, "  }\n");
    Printf(f->code, "  arg2 = Z_STR(args[0]);\n\n");

    Printf(f->code, "if (!arg2) {\n  RETVAL_FALSE;\n}\n");
    Printf(f->code, "\nelse if (strcmp(ZSTR_VAL(arg2),\"thisown\") == 0) {\n");
    Printf(f->code, "RETVAL_TRUE;\n}\n\n");
    if (magic_isset) {
      Append(f->code, magic_isset);
    }
    Printf(f->code, "else {\n");
    if (swig_base) {
      Printf(f->code, "PHP_MN(%s%s___isset)(INTERNAL_FUNCTION_PARAM_PASSTHRU);\n}\n", prefix, swig_base);
    } else {
      // __isset is only called if the property isn't set on the zend_object.
      Printf(f->code, "RETVAL_FALSE;\n}\n");
    }

    Printf(f->code, "fail:\n");
    Printf(f->code, "return;\n");
    Printf(f->code, "}\n\n\n");

    Wrapper_print(f, s_wrappers);
    DelWrapper(f);
    f = NULL;

    Delete(magic_set);
    Delete(magic_get);
    Delete(magic_isset);
    magic_set = NULL;
    magic_get = NULL;
    magic_isset = NULL;
  }

  String *getAccessMode(String *access) {
    if (Cmp(access, "protected") == 0) {
      return NewString("ZEND_ACC_PROTECTED");
    } else if (Cmp(access, "private") == 0) {
      return NewString("ZEND_ACC_PRIVATE");
    }
    return NewString("ZEND_ACC_PUBLIC");
  }

  bool is_setter_method(Node *n) {
    const char *p = GetChar(n, "sym:name");
    if (strlen(p) > 4) {
      p += strlen(p) - 4;
      if (strcmp(p, "_set") == 0) {
	return true;
      }
    }
    return false;
  }

  bool is_getter_method(Node *n) {
    const char *p = GetChar(n, "sym:name");
    if (strlen(p) > 4) {
      p += strlen(p) - 4;
      if (strcmp(p, "_get") == 0) {
	return true;
      }
    }
    return false;
  }

  virtual int functionWrapper(Node *n) {
    if (wrapperType == directordisown) {
      // Handled via __set magic method - no explicit wrapper method wanted.
      return SWIG_OK;
    }
    String *name = GetChar(n, "name");
    String *iname = GetChar(n, "sym:name");
    SwigType *d = Getattr(n, "type");
    ParmList *l = Getattr(n, "parms");
    String *nodeType = Getattr(n, "nodeType");
    int newobject = GetFlag(n, "feature:new");
    int constructor = (Cmp(nodeType, "constructor") == 0);

    Parm *p;
    int i;
    int numopt;
    String *tm;
    Wrapper *f;

    String *wname = NewStringEmpty();
    String *overloadwname = NULL;
    int overloaded = 0;
    String *modes = NULL;
    bool static_setter = false;
    bool static_getter = false;

    modes = getAccessMode(Getattr(n, "access"));

    if (constructor) {
      Append(modes, " | ZEND_ACC_CTOR");
    }
    if (wrapperType == staticmemberfn || Cmp(Getattr(n, "storage"), "static") == 0) {
      Append(modes, " | ZEND_ACC_STATIC");
    }
    if (GetFlag(n, "abstract") && Swig_directorclass(Swig_methodclass(n)) && !is_member_director(n))
      Append(modes, " | ZEND_ACC_ABSTRACT");

    if (Getattr(n, "sym:overloaded")) {
      overloaded = 1;
      overloadwname = NewString(Swig_name_wrapper(iname));
      Printf(overloadwname, "%s", Getattr(n, "sym:overname"));
    } else {
      if (!addSymbol(iname, n))
	return SWIG_ERROR;
    }

    if (constructor) {
      if (!Equal(class_name, Getattr(n, "constructorHandler:sym:name"))) {
	// Renamed constructor - turn into static factory method
	wname = Copy(Getattr(n, "constructorHandler:sym:name"));
      } else {
	wname = NewString("__construct");
      }
    } else if (wrapperType == membervar) {
      wname = Copy(Getattr(n, "membervariableHandler:sym:name"));
      if (is_setter_method(n)) {
	Append(wname, "_set");
      } else if (is_getter_method(n)) {
	Append(wname, "_get");
      }
    } else if (wrapperType == memberfn) {
      wname = Getattr(n, "memberfunctionHandler:sym:name");
    } else if (wrapperType == staticmembervar) {
      // Shape::nshapes -> nshapes
      wname = Getattr(n, "staticmembervariableHandler:sym:name");

      /* We get called twice for getter and setter methods. But to maintain
	 compatibility, Shape::nshapes() is being used for both setter and
	 getter methods. So using static_setter and static_getter variables
	 to generate half of the code each time.
       */
      static_setter = is_setter_method(n);

      if (is_getter_method(n)) {
	// This is to overcome types that can't be set and hence no setter.
	if (!Equal(Getattr(n, "feature:immutable"), "1"))
	  static_getter = true;
      }
    } else if (wrapperType == staticmemberfn) {
      wname = Getattr(n, "staticmemberfunctionHandler:sym:name");
    } else {
      if (class_name) {
	if (Cmp(Getattr(n, "storage"), "friend") == 0 && Cmp(Getattr(n, "view"), "globalfunctionHandler") == 0) {
	  wname = iname;
	} else {
	  wname = Getattr(n, "destructorHandler:sym:name");
	}
      } else {
	wname = iname;
      }
    }

    if (wrapperType == destructor) {
      // We don't explicitly wrap the destructor for PHP - Zend manages the
      // reference counting, and the user can just do `$obj = null;' or similar
      // to remove a reference to an object.
      Setattr(n, "wrap:name", wname);
      (void)emit_action(n);
      return SWIG_OK;
    }

    if (!static_getter) {
      // Create or find existing PHPTypes.
      phptypes = NULL;

      String *key;
      if (class_name && !Equal(Getattr(n, "storage"), "friend")) {
	key = NewStringf("%s:%s", class_name, wname);
      } else {
	key = NewStringf(":%s", wname);
      }

      PHPTypes *p = (PHPTypes*)GetVoid(all_phptypes, key);
      if (p) {
	// We already have an entry so use it.
	phptypes = p;
	Delete(key);
      } else {
	phptypes = new PHPTypes(n);
	SetVoid(all_phptypes, key, phptypes);
      }
    }

    f = NewWrapper();

    if (static_getter) {
      Printf(f->def, "{\n");
    }

    String *outarg = NewStringEmpty();
    String *cleanup = NewStringEmpty();

    if (!overloaded) {
      if (!static_getter) {
	if (class_name && !Equal(Getattr(n, "storage"), "friend")) {
	  Printv(f->def, "static PHP_METHOD(", prefix, class_name, ",", wname, ") {\n", NIL);
	} else {
	  if (wrap_nonclass_global) {
	    Printv(f->def, "static PHP_METHOD(", fake_class_name(), ",", wname, ") {\n",
			   "  PHP_FN(", wname, ")(INTERNAL_FUNCTION_PARAM_PASSTHRU);\n",
			   "}\n\n", NIL);
	  }

	  if (wrap_nonclass_fake_class) {
	    Printv(f->def, "static PHP_FUNCTION(", wname, ") {\n", NIL);
	  }
	}
      }
    } else {
      Printv(f->def, "static ZEND_NAMED_FUNCTION(", overloadwname, ") {\n", NIL);
    }

    emit_parameter_variables(l, f);
    /* Attach standard typemaps */

    emit_attach_parmmaps(l, f);

    if (wrapperType == memberfn || wrapperType == membervar) {
      // Assign "this" to arg1 and remove first entry from ParmList l.
      Printf(f->code, "arg1 = (%s)SWIG_Z_FETCH_OBJ_P(ZEND_THIS)->ptr;\n", SwigType_lstr(Getattr(l, "type"), ""));
      l = nextSibling(l);
    }

    // wrap:parms is used by overload resolution.
    Setattr(n, "wrap:parms", l);

    int num_arguments = emit_num_arguments(l);
    int num_required = emit_num_required(l);
    numopt = num_arguments - num_required;

    if (num_arguments > 0) {
      String *args = NewStringEmpty();
      Printf(args, "zval args[%d]", num_arguments);
      Wrapper_add_local(f, "args", args);
      Delete(args);
      args = NULL;
    }
    if (wrapperType == directorconstructor) {
      Wrapper_add_local(f, "arg0", "zval *arg0 = ZEND_THIS");
    }

    // This generated code may be called:
    // 1) as an object method, or
    // 2) as a class-method/function (without a "this_ptr")
    // Option (1) has "this_ptr" for "this", option (2) needs it as
    // first parameter

    // NOTE: possible we ignore this_ptr as a param for native constructor

    if (numopt > 0) {		// membervariable wrappers do not have optional args
      Wrapper_add_local(f, "arg_count", "int arg_count");
      Printf(f->code, "arg_count = ZEND_NUM_ARGS();\n");
      Printf(f->code, "if(arg_count<%d || arg_count>%d ||\n", num_required, num_arguments);
      Printf(f->code, "   zend_get_parameters_array_ex(arg_count,args)!=SUCCESS)\n");
      Printf(f->code, "\tWRONG_PARAM_COUNT;\n\n");
    } else if (static_setter || static_getter) {
      if (num_arguments == 0) {
	Printf(f->code, "if(ZEND_NUM_ARGS() == 0) {\n");
      } else {
	Printf(f->code, "if(ZEND_NUM_ARGS() == %d && zend_get_parameters_array_ex(%d, args) == SUCCESS) {\n", num_arguments, num_arguments);
      }
    } else {
      if (num_arguments == 0) {
	Printf(f->code, "if(ZEND_NUM_ARGS() != 0) {\n");
      } else {
	Printf(f->code, "if(ZEND_NUM_ARGS() != %d || zend_get_parameters_array_ex(%d, args) != SUCCESS) {\n", num_arguments, num_arguments);
      }
      Printf(f->code, "WRONG_PARAM_COUNT;\n}\n\n");
    }

    /* Now convert from PHP to C variables */
    // At this point, argcount if used is the number of deliberately passed args
    // not including this_ptr even if it is used.
    // It means error messages may be out by argbase with error
    // reports.  We can either take argbase into account when raising
    // errors, or find a better way of dealing with _thisptr.
    // I would like, if objects are wrapped, to assume _thisptr is always
    // _this and not the first argument.
    // This may mean looking at Language::memberfunctionHandler

    for (i = 0, p = l; i < num_arguments; i++) {
      /* Skip ignored arguments */
      while (checkAttribute(p, "tmap:in:numinputs", "0")) {
	p = Getattr(p, "tmap:in:next");
      }

      /* Check if optional */
      if (i >= num_required) {
	Printf(f->code, "\tif(arg_count > %d) {\n", i);
      }

      tm = Getattr(p, "tmap:in");
      if (!tm) {
	SwigType *pt = Getattr(p, "type");
	Swig_warning(WARN_TYPEMAP_IN_UNDEF, input_file, line_number, "Unable to use type %s as a function argument.\n", SwigType_str(pt, 0));
	p = nextSibling(p);
	continue;
      }

      phptypes->process_phptype(p, i + 1, "tmap:in:phptype");
      if (GetFlag(p, "tmap:in:byref")) phptypes->set_byref(i + 1);

      String *source = NewStringf("args[%d]", i);
      Replaceall(tm, "$input", source);
      Setattr(p, "emit:input", source);
      Printf(f->code, "%s\n", tm);
      if (i == 0 && Getattr(p, "self")) {
	Printf(f->code, "\tif(!arg1) {\n");
	Printf(f->code, "\t  zend_throw_exception(zend_ce_type_error, \"this pointer is NULL\", 0);\n");
	Printf(f->code, "\t  return;\n");
	Printf(f->code, "\t}\n");
      }

      if (i >= num_required) {
	Printf(f->code, "}\n");
      }

      p = Getattr(p, "tmap:in:next");

      Delete(source);
    }

    if (is_member_director(n)) {
      Wrapper_add_local(f, "director", "Swig::Director *director = 0");
      Append(f->code, "director = SWIG_DIRECTOR_CAST(arg1);\n");
      Wrapper_add_local(f, "upcall", "bool upcall = false");
      Printf(f->code, "upcall = (director && (director->swig_get_self()==Z_OBJ_P(ZEND_THIS)));\n");
    }

    Swig_director_emit_dynamic_cast(n, f);

    /* Insert constraint checking code */
    for (p = l; p;) {
      if ((tm = Getattr(p, "tmap:check"))) {
	Printv(f->code, tm, "\n", NIL);
	p = Getattr(p, "tmap:check:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert cleanup code */
    for (i = 0, p = l; p; i++) {
      if ((tm = Getattr(p, "tmap:freearg"))) {
	Printv(cleanup, tm, "\n", NIL);
	p = Getattr(p, "tmap:freearg:next");
      } else {
	p = nextSibling(p);
      }
    }

    /* Insert argument output code */
    for (i = 0, p = l; p; i++) {
      if ((tm = Getattr(p, "tmap:argout")) && Len(tm)) {
	Replaceall(tm, "$result", "return_value");
	Replaceall(tm, "$arg", Getattr(p, "emit:input"));
	Replaceall(tm, "$input", Getattr(p, "emit:input"));
	Printv(outarg, tm, "\n", NIL);
	p = Getattr(p, "tmap:argout:next");
      } else {
	p = nextSibling(p);
      }
    }

    if (!overloaded) {
      Setattr(n, "wrap:name", wname);
    } else {
      Setattr(n, "wrap:name", overloadwname);
    }
    Setattr(n, "wrapper:method:name", wname);

    bool php_constructor = (constructor && Cmp(class_name, Getattr(n, "constructorHandler:sym:name")) == 0);

    /* emit function call */
    String *actioncode = emit_action(n);

    if ((tm = Swig_typemap_lookup_out("out", n, Swig_cresult_name(), f, actioncode))) {
      Replaceall(tm, "$input", Swig_cresult_name());
      Replaceall(tm, "$result", php_constructor ? "ZEND_THIS" : "return_value");
      Replaceall(tm, "$owner", newobject ? "1" : "0");
      Printf(f->code, "%s\n", tm);
    } else {
      Swig_warning(WARN_TYPEMAP_OUT_UNDEF, input_file, line_number, "Unable to use return type %s in function %s.\n", SwigType_str(d, 0), name);
    }
    emit_return_variable(n, d, f);

    List *return_types = phptypes->process_phptype(n, 0, "tmap:out:phptype");

    if (class_name && !Equal(Getattr(n, "storage"), "friend")) {
      if (is_member_director(n)) {
	String *parent = class_name;
	while ((parent = Getattr(php_parent_class, parent)) != NULL) {
	  // Mark this method name as having no return type declaration for all
	  // classes we're derived from.
	  SetFlag(has_directed_descendent, NewStringf("%s:%s", parent, wname));
	}
      } else if (return_types) {
	String *parent = class_name;
	while ((parent = Getattr(php_parent_class, parent)) != NULL) {
	  String *key = NewStringf("%s:%s", parent, wname);
	  // The parent class method needs to have a superset of the possible
	  // return types of methods with the same name in subclasses.
	  List *v = Getattr(parent_class_method_return_type, key);
	  if (!v) {
	    // New entry.
	    Setattr(parent_class_method_return_type, key, Copy(return_types));
	  } else {
	    // Update existing entry.
	    PHPTypes::merge_type_lists(v, return_types);
	  }
	}
      }
    }

    if (outarg) {
      Printv(f->code, outarg, NIL);
    }

    if (static_setter && cleanup) {
      Printv(f->code, cleanup, NIL);
    }

    /* Look to see if there is any newfree cleanup code */
    if (GetFlag(n, "feature:new")) {
      if ((tm = Swig_typemap_lookup("newfree", n, Swig_cresult_name(), 0))) {
	Printf(f->code, "%s\n", tm);
	Delete(tm);
      }
    }

    /* See if there is any return cleanup code */
    if ((tm = Swig_typemap_lookup("ret", n, Swig_cresult_name(), 0))) {
      Printf(f->code, "%s\n", tm);
      Delete(tm);
    }

    if (static_getter) {
      Printf(f->code, "}\n");
    }

    if (static_setter || static_getter) {
      Printf(f->code, "}\n");
    }

    if (!static_setter) {
      Printf(f->code, "fail:\n");
      Printv(f->code, cleanup, NIL);
      Printf(f->code, "return;\n");
      Printf(f->code, "}\n");
    }

    Replaceall(f->code, "$cleanup", cleanup);
    Replaceall(f->code, "$symname", iname);

    Wrapper_print(f, s_wrappers);
    DelWrapper(f);
    f = NULL;

    if (overloaded) {
      if (!Getattr(n, "sym:nextSibling")) {
	dispatchFunction(n, constructor);
      }
    } else {
      if (!static_setter) {
	create_command(class_name, wname, n, false, modes);
      }
    }

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * globalvariableHandler()
   * ------------------------------------------------------------ */

  /* PHP doesn't support intercepting reads and writes to global variables
   * (nor static property reads and writes so we can't wrap them as static
   * properties on a dummy class) so just let SWIG do its default thing and
   * wrap them as name_get() and name_set().
   */
  //virtual int globalvariableHandler(Node *n) {
  //}

  /* ------------------------------------------------------------
   * constantWrapper()
   * ------------------------------------------------------------ */

  virtual int constantWrapper(Node *n) {
    String *name = GetChar(n, "name");
    String *iname = GetChar(n, "sym:name");
    SwigType *type = Getattr(n, "type");
    String *rawval = Getattr(n, "rawval");
    String *value = rawval ? rawval : Getattr(n, "value");
    String *tm;

    if (!addSymbol(iname, n))
      return SWIG_ERROR;

    SwigType_remember(type);

    String *wrapping_member_constant = Getattr(n, "memberconstantHandler:sym:name");
    if (!wrapping_member_constant) {
      {
	tm = Swig_typemap_lookup("consttab", n, name, 0);
	Replaceall(tm, "$value", value);
	if (Getattr(n, "tmap:consttab:rinit")) {
	  Printf(r_init, "%s\n", tm);
	} else {
	  Printf(s_cinit, "%s\n", tm);
	}
      }

      {
	tm = Swig_typemap_lookup("classconsttab", n, name, 0);

	Replaceall(tm, "$class", fake_class_name());
	Replaceall(tm, "$const_name", iname);
	Replaceall(tm, "$value", value);
	if (Getattr(n, "tmap:classconsttab:rinit")) {
	  Printf(r_init, "%s\n", tm);
	} else {
	  Printf(s_cinit, "%s\n", tm);
	}
      }
    } else {
      tm = Swig_typemap_lookup("classconsttab", n, name, 0);
      Replaceall(tm, "$class", class_name);
      Replaceall(tm, "$const_name", wrapping_member_constant);
      Replaceall(tm, "$value", value);
      if (Getattr(n, "tmap:classconsttab:rinit")) {
	Printf(r_init, "%s\n", tm);
      } else {
	Printf(s_cinit, "%s\n", tm);
      }
    }

    wrapperType = standard;
    return SWIG_OK;
  }

  /*
   * PHP::pragma()
   *
   * Pragma directive.
   *
   * %pragma(php) code="String"         # Includes a string in the .php file
   * %pragma(php) include="file.php"    # Includes a file in the .php file
   */

  virtual int pragmaDirective(Node *n) {
    if (!ImportMode) {
      String *lang = Getattr(n, "lang");
      String *type = Getattr(n, "name");
      String *value = Getattr(n, "value");

      if (Strcmp(lang, "php") == 0) {
	if (Strcmp(type, "code") == 0) {
	  if (value) {
	    Printf(pragma_code, "%s\n", value);
	  }
	} else if (Strcmp(type, "include") == 0) {
	  if (value) {
	    Printf(pragma_incl, "include '%s';\n", value);
	  }
	} else if (Strcmp(type, "phpinfo") == 0) {
	  if (value) {
	    Printf(pragma_phpinfo, "%s\n", value);
	  }
	} else if (Strcmp(type, "version") == 0) {
	  if (value) {
	    pragma_version = value;
	  }
	} else {
	  Swig_warning(WARN_PHP_UNKNOWN_PRAGMA, input_file, line_number, "Unrecognized pragma <%s>.\n", type);
	}
      }
    }
    return Language::pragmaDirective(n);
  }

  /* ------------------------------------------------------------
   * classDeclaration()
   * ------------------------------------------------------------ */

  virtual int classDeclaration(Node *n) {
    return Language::classDeclaration(n);
  }

  /* ------------------------------------------------------------
   * classHandler()
   * ------------------------------------------------------------ */

  virtual int classHandler(Node *n) {
    String *symname = Getattr(n, "sym:name");

    class_name = symname;
    base_class = NULL;
    destructor_action = NULL;

    Printf(all_cs_entry, "static const zend_function_entry class_%s_functions[] = {\n", class_name);

    // namespace code to introduce namespaces into wrapper classes.
    //if (nameSpace != NULL)
      //Printf(s_oinit, "INIT_CLASS_ENTRY(internal_ce, \"%s\\\\%s\", class_%s_functions);\n", nameSpace, class_name, class_name);
    //else
    Printf(s_oinit, "  INIT_CLASS_ENTRY(internal_ce, \"%s%s\", class_%s_functions);\n", prefix, class_name, class_name);

    if (shadow) {
      char *rename = GetChar(n, "sym:name");

      if (!addSymbol(rename, n))
	return SWIG_ERROR;

      /* Deal with inheritance */
      List *baselist = Getattr(n, "bases");
      if (baselist) {
	Iterator base = First(baselist);
	while (base.item) {
	  if (!GetFlag(base.item, "feature:ignore")) {
	    if (!base_class) {
	      base_class = Getattr(base.item, "sym:name");
	    } else {
	      /* Warn about multiple inheritance for additional base class(es) */
	      String *proxyclassname = SwigType_str(Getattr(n, "classtypeobj"), 0);
	      String *baseclassname = SwigType_str(Getattr(base.item, "name"), 0);
	      Swig_warning(WARN_PHP_MULTIPLE_INHERITANCE, input_file, line_number,
			   "Warning for %s, base %s ignored. Multiple inheritance is not supported in PHP.\n", proxyclassname, baseclassname);
	    }
	  }
	  base = Next(base);
	}
      }
    }

    if (GetFlag(n, "feature:exceptionclass") && Getattr(n, "feature:except")) {
      /* PHP requires thrown objects to be instances of or derived from
       * Exception, so that really needs to take priority over any
       * explicit base class.
       */
      if (base_class) {
	String *proxyclassname = SwigType_str(Getattr(n, "classtypeobj"), 0);
	Swig_warning(WARN_PHP_MULTIPLE_INHERITANCE, input_file, line_number,
		     "Warning for %s, base %s ignored. Multiple inheritance is not supported in PHP.\n", proxyclassname, base_class);
      }
      base_class = NewString("Exception");
    }

    if (!base_class) {
      Printf(s_oinit, "  SWIG_Php_ce_%s = zend_register_internal_class(&internal_ce);\n", class_name);
    } else if (Equal(base_class, "Exception")) {
      Printf(s_oinit, "  SWIG_Php_ce_%s = zend_register_internal_class_ex(&internal_ce, zend_ce_exception);\n", class_name);
    } else if (is_class_wrapped(base_class)) {
      Printf(s_oinit, "  SWIG_Php_ce_%s = zend_register_internal_class_ex(&internal_ce, SWIG_Php_ce_%s);\n", class_name, base_class);
      Setattr(php_parent_class, class_name, base_class);
    } else {
      Printf(s_oinit, "  {\n");
      Printf(s_oinit, "    swig_type_info *type_info = SWIG_MangledTypeQueryModule(swig_module.next, &swig_module, \"_p_%s\");\n", base_class);
      Printf(s_oinit, "    SWIG_Php_ce_%s = zend_register_internal_class_ex(&internal_ce, (zend_class_entry*)(type_info ? type_info->clientdata : NULL));\n", class_name);
      Printf(s_oinit, "  }\n");
    }

    if (Getattr(n, "abstracts") && !GetFlag(n, "feature:notabstract")) {
      Printf(s_oinit, "  SWIG_Php_ce_%s->ce_flags |= ZEND_ACC_EXPLICIT_ABSTRACT_CLASS;\n", class_name);
    }
    if (Getattr(n, "feature:php:allowdynamicproperties")) {
      Append(s_oinit, "#ifdef ZEND_ACC_ALLOW_DYNAMIC_PROPERTIES\n");
      Printf(s_oinit, "  SWIG_Php_ce_%s->ce_flags |= ZEND_ACC_ALLOW_DYNAMIC_PROPERTIES;\n", class_name);
      Append(s_oinit, "#endif\n");
    } else {
      Append(s_oinit, "#ifdef ZEND_ACC_NO_DYNAMIC_PROPERTIES\n");
      Printf(s_oinit, "  SWIG_Php_ce_%s->ce_flags |= ZEND_ACC_NO_DYNAMIC_PROPERTIES;\n", class_name);
      Append(s_oinit, "#endif\n");
    }
    String *swig_wrapped = swig_wrapped_interface_ce();
    Printv(s_oinit, "  zend_do_implement_interface(SWIG_Php_ce_", class_name, ", &", swig_wrapped, ");\n", NIL);

    {
      Node *node = NewHash();
      Setattr(node, "type", Getattr(n, "name"));
      Setfile(node, Getfile(n));
      Setline(node, Getline(n));
      String *interfaces = Swig_typemap_lookup("phpinterfaces", node, "", 0);
      Replaceall(interfaces, " ", "");
      if (interfaces && Len(interfaces) > 0) {
	// It seems we need to wait until RINIT time to look up class entries
	// for interfaces by name.  The downside is that this then happens for
	// every request.
	//
	// Most pre-defined interfaces are accessible via zend_class_entry*
	// variables declared in the PHP C API - these we can use at MINIT
	// time, so we special case them.  This will also be a little faster
	// than looking up by name.
	Printv(s_header,
	       "#ifdef __cplusplus\n",
	       "extern \"C\" {\n",
	       "#endif\n",
	       NIL);

	String *r_init_prefix = NewStringEmpty();

	List *interface_list = Split(interfaces, ',', -1);
	int num_interfaces = Len(interface_list);
	for (int i = 0; i < num_interfaces; ++i) {
	  String *interface = Getitem(interface_list, i);
	  // We generate conditional code in both minit and rinit - then we or the user
	  // just need to define SWIG_PHP_INTERFACE_xxx_CE (and optionally
	  // SWIG_PHP_INTERFACE_xxx_HEADER) to handle interface `xxx` at minit-time.
	  Printv(s_header,
		 "#ifdef SWIG_PHP_INTERFACE_", interface, "_HEADER\n",
		 "# include SWIG_PHP_INTERFACE_", interface, "_HEADER\n",
		 "#endif\n",
		 NIL);
	  Printv(s_oinit,
		 "#ifdef SWIG_PHP_INTERFACE_", interface, "_CE\n",
		 "  zend_do_implement_interface(SWIG_Php_ce_", class_name, ", SWIG_PHP_INTERFACE_", interface, "_CE);\n",
		 "#endif\n",
		 NIL);
	  Printv(r_init_prefix,
		 "#ifndef SWIG_PHP_INTERFACE_", interface, "_CE\n",
		 "  {\n",
		 "    zend_class_entry *swig_interface_ce = zend_lookup_class(zend_string_init(\"", interface, "\", sizeof(\"", interface, "\") - 1, 0));\n",
		 "    if (swig_interface_ce)\n",
		 "      zend_do_implement_interface(SWIG_Php_ce_", class_name, ", swig_interface_ce);\n",
                 "    else\n",
                 "      zend_throw_exception(zend_ce_error, \"Interface \\\"", interface, "\\\" not found\", 0);\n",
		 "  }\n",
		 "#endif\n",
		 NIL);
	}

	// Handle interfaces at the start of rinit so that they're added
	// before any potential constant objects, etc which might be created
	// later in rinit.
	Insert(r_init, 0, r_init_prefix);
	Delete(r_init_prefix);

	Printv(s_header,
	       "#ifdef __cplusplus\n",
	       "}\n",
	       "#endif\n",
	       NIL);
      }
      Delete(interfaces);
    }

    Language::classHandler(n);

    static bool emitted_base_object_handlers = false;
    if (!emitted_base_object_handlers) {
      Printf(s_creation, "static zend_object_handlers Swig_Php_base_object_handlers;\n\n");

      // Set up a base zend_object_handlers structure which we can use as-is
      // for classes without a destructor, and copy as the basis for other
      // classes.
      Printf(s_oinit, "  Swig_Php_base_object_handlers = *zend_get_std_object_handlers();\n");
      Printf(s_oinit, "  Swig_Php_base_object_handlers.offset = XtOffsetOf(swig_object_wrapper, std);\n");
      Printf(s_oinit, "  Swig_Php_base_object_handlers.clone_obj = NULL;\n");
      emitted_base_object_handlers = true;
    }

    Printf(s_creation, "static zend_class_entry *SWIG_Php_ce_%s;\n\n", class_name);

    if (Getattr(n, "has_destructor")) {
      if (destructor_action ? Equal(destructor_action, "free((char *) arg1);") : !CPlusPlus) {
	// We can use a single function if the destructor action calls free()
	// (either explicitly or as the default in C-mode) since free() doesn't
	// care about the object's type.  We currently only check for the exact
	// code that Swig_cdestructor_call() emits.
	static bool emitted_common_cdestructor = false;
	if (!emitted_common_cdestructor) {
	  Printf(s_creation, "static zend_object_handlers Swig_Php_common_c_object_handlers;\n\n");
	  Printf(s_creation, "static void SWIG_Php_common_c_free_obj(zend_object *object) {free(SWIG_Php_free_obj(object));}\n\n");
	  Printf(s_creation, "static zend_object *SWIG_Php_common_c_create_object(zend_class_entry *ce) {return SWIG_Php_do_create_object(ce, &Swig_Php_common_c_object_handlers);}\n");

	  Printf(s_oinit, "  Swig_Php_common_c_object_handlers = Swig_Php_base_object_handlers;\n");
	  Printf(s_oinit, "  Swig_Php_common_c_object_handlers.free_obj = SWIG_Php_common_c_free_obj;\n");

	  emitted_common_cdestructor = true;
	}

	Printf(s_oinit, "  SWIG_Php_ce_%s->create_object = SWIG_Php_common_c_create_object;\n", class_name);
      } else {
	Printf(s_creation, "static zend_object_handlers %s_object_handlers;\n", class_name);
	Printf(s_creation, "static zend_object *SWIG_Php_create_object_%s(zend_class_entry *ce) {return SWIG_Php_do_create_object(ce, &%s_object_handlers);}\n", class_name, class_name);

	Printf(s_creation, "static void SWIG_Php_free_obj_%s(zend_object *object) {",class_name);
	String *type = Getattr(n, "classtype");
	// Special case handling the delete call generated by
	// Swig_cppdestructor_call() and generate simpler code.
	if (destructor_action && !Equal(destructor_action, "delete arg1;")) {
	  Printv(s_creation, "\n"
		 "  ", type, " *arg1 = (" , type, " *)SWIG_Php_free_obj(object);\n"
		 "  if (arg1) {\n"
		 "    ", destructor_action, "\n"
		 "  }\n", NIL);
	} else {
	  Printf(s_creation, "delete (%s *)SWIG_Php_free_obj(object);", type);
	}
	Printf(s_creation, "}\n\n");

	Printf(s_oinit, "  SWIG_Php_ce_%s->create_object = SWIG_Php_create_object_%s;\n", class_name, class_name);
	Printf(s_oinit, "  %s_object_handlers = Swig_Php_base_object_handlers;\n", class_name);
	Printf(s_oinit, "  %s_object_handlers.free_obj = SWIG_Php_free_obj_%s;\n", class_name, class_name);
      }
    } else {
      static bool emitted_destructorless_create_object = false;
      if (!emitted_destructorless_create_object) {
	emitted_destructorless_create_object = true;
	Printf(s_creation, "static zend_object *SWIG_Php_create_object(zend_class_entry *ce) {return SWIG_Php_do_create_object(ce, &Swig_Php_base_object_handlers);}\n", class_name);
      }

      Printf(s_oinit, "  SWIG_Php_ce_%s->create_object = SWIG_Php_create_object;\n", class_name);
    }

    // If not defined we aren't wrapping any functions which use this type as a
    // parameter or return value, in which case we don't need the clientdata
    // set.
    Printf(s_oinit, "#ifdef SWIGTYPE_p%s\n", SwigType_manglestr(Getattr(n, "classtypeobj")));
    Printf(s_oinit, "  SWIG_TypeClientData(SWIGTYPE_p%s,SWIG_Php_ce_%s);\n", SwigType_manglestr(Getattr(n, "classtypeobj")), class_name);
    Printf(s_oinit, "#endif\n");
    Printf(s_oinit, "\n");

    generate_magic_property_methods(n);
    Printf(all_cs_entry, " ZEND_FE_END\n};\n\n");

    class_name = NULL;
    base_class = NULL;
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * memberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int memberfunctionHandler(Node *n) {
    wrapperType = memberfn;
    Language::memberfunctionHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * membervariableHandler()
   * ------------------------------------------------------------ */

  virtual int membervariableHandler(Node *n) {
    if (magic_set == NULL) {
      magic_set = NewStringEmpty();
      magic_get = NewStringEmpty();
      magic_isset = NewStringEmpty();
    }

    String *v_name = GetChar(n, "name");

    Printf(magic_set, "\nelse if (strcmp(ZSTR_VAL(arg2),\"%s\") == 0) {\n", v_name);
    Printf(magic_set, "ZVAL_STRING(&tempZval, \"%s_set\");\n", v_name);
    Printf(magic_set, "call_user_function(EG(function_table),ZEND_THIS,&tempZval,return_value,1,&args[1]);\n}\n");

    Printf(magic_get, "\nelse if (strcmp(ZSTR_VAL(arg2),\"%s\") == 0) {\n", v_name);
    Printf(magic_get, "ZVAL_STRING(&tempZval, \"%s_get\");\n", v_name);
    Printf(magic_get, "call_user_function(EG(function_table),ZEND_THIS,&tempZval,return_value,0,NULL);\n}\n");

    Printf(magic_isset, "\nelse if (strcmp(ZSTR_VAL(arg2),\"%s\") == 0) {\n", v_name);
    Printf(magic_isset, "RETVAL_TRUE;\n}\n");

    wrapperType = membervar;
    Language::membervariableHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * staticmembervariableHandler()
   * ------------------------------------------------------------ */

  virtual int staticmembervariableHandler(Node *n) {
    wrapperType = staticmembervar;
    Language::staticmembervariableHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * staticmemberfunctionHandler()
   * ------------------------------------------------------------ */

  virtual int staticmemberfunctionHandler(Node *n) {
    wrapperType = staticmemberfn;
    Language::staticmemberfunctionHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  int abstractConstructorHandler(Node *) {
    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * constructorHandler()
   * ------------------------------------------------------------ */

  virtual int constructorHandler(Node *n) {
    if (Swig_directorclass(n)) {
      String *ctype = GetChar(Swig_methodclass(n), "classtype");
      String *sname = GetChar(Swig_methodclass(n), "sym:name");
      String *args = NewStringEmpty();
      ParmList *p = Getattr(n, "parms");
      int i;

      for (i = 0; p; p = nextSibling(p), i++) {
	if (i) {
	  Printf(args, ", ");
	}
	if (Strcmp(GetChar(p, "type"), SwigType_str(GetChar(p, "type"), 0))) {
	  SwigType *t = Getattr(p, "type");
	  Printf(args, "%s", SwigType_rcaststr(t, 0));
	  if (SwigType_isreference(t)) {
	    Append(args, "*");
	  }
	}
	Printf(args, "arg%d", i+1);
      }

      /* director ctor code is specific for each class */
      Delete(director_ctor_code);
      director_ctor_code = NewStringEmpty();
      director_prot_ctor_code = NewStringEmpty();
      Printf(director_ctor_code, "if (Z_OBJCE_P(arg0) == SWIG_Php_ce_%s) { /* not subclassed */\n", class_name);
      Printf(director_prot_ctor_code, "if (Z_OBJCE_P(arg0) == SWIG_Php_ce_%s) { /* not subclassed */\n", class_name);
      Printf(director_ctor_code, "  %s = new %s(%s);\n", Swig_cresult_name(), ctype, args);
      Printf(director_prot_ctor_code,
	     "  zend_throw_exception(zend_ce_type_error, \"accessing abstract class or protected constructor\", 0);\n"
	     "  return;\n");
      if (i) {
	Insert(args, 0, ", ");
      }
      Printf(director_ctor_code, "} else {\n  %s = (%s *)new SwigDirector_%s(arg0%s);\n}\n", Swig_cresult_name(), ctype, sname, args);
      Printf(director_prot_ctor_code, "} else {\n  %s = (%s *)new SwigDirector_%s(arg0%s);\n}\n", Swig_cresult_name(), ctype, sname, args);
      Delete(args);

      wrapperType = directorconstructor;
    } else {
      wrapperType = constructor;
    }
    Language::constructorHandler(n);
    wrapperType = standard;

    return SWIG_OK;
  }

  /* ------------------------------------------------------------
   * destructorHandler()
   * ------------------------------------------------------------ */
  virtual int destructorHandler(Node *n) {
    wrapperType = destructor;
    Language::destructorHandler(n);
    destructor_action = Getattr(n, "wrap:action");
    wrapperType = standard;
    return SWIG_OK;
  }

  int classDirectorInit(Node *n) {
    String *declaration = Swig_director_declaration(n);
    Printf(f_directors_h, "%s\n", declaration);
    Printf(f_directors_h, "public:\n");
    Delete(declaration);
    return Language::classDirectorInit(n);
  }

  int classDirectorEnd(Node *n) {
    Printf(f_directors_h, "};\n");
    return Language::classDirectorEnd(n);
  }

  int classDirectorConstructor(Node *n) {
    Node *parent = Getattr(n, "parentNode");
    String *decl = Getattr(n, "decl");
    String *supername = Swig_class_name(parent);
    String *classname = NewStringEmpty();
    Printf(classname, "SwigDirector_%s", supername);

    /* insert self parameter */
    Parm *p;
    ParmList *superparms = Getattr(n, "parms");
    ParmList *parms = CopyParmList(superparms);
    String *type = NewString("zval");
    SwigType_add_pointer(type);
    p = NewParm(type, NewString("self"), n);
    set_nextSibling(p, parms);
    parms = p;

    if (!Getattr(n, "defaultargs")) {
      // There should always be a "self" parameter first.
      assert(ParmList_len(parms) > 0);

      /* constructor */
      {
	Wrapper *w = NewWrapper();
	String *call;
	String *basetype = Getattr(parent, "classtype");

	String *target = Swig_method_decl(0, decl, classname, parms, 0);
	call = Swig_csuperclass_call(0, basetype, superparms);
	Printf(w->def, "%s::%s: %s, Swig::Director(self) {", classname, target, call);
	Append(w->def, "}");
	Delete(target);
	Wrapper_print(w, f_directors);
	Delete(call);
	DelWrapper(w);
      }

      /* constructor header */
      {
	String *target = Swig_method_decl(0, decl, classname, parms, 1);
	Printf(f_directors_h, "    %s;\n", target);
	Delete(target);
      }
    }
    return Language::classDirectorConstructor(n);
  }

  int classDirectorMethod(Node *n, Node *parent, String *super) {
    int is_void = 0;
    int is_pointer = 0;
    String *decl = Getattr(n, "decl");
    String *returntype = Getattr(n, "type");
    String *name = Getattr(n, "name");
    String *classname = Getattr(parent, "sym:name");
    String *c_classname = Getattr(parent, "name");
    String *symname = Getattr(n, "sym:name");
    String *declaration = NewStringEmpty();
    ParmList *l = Getattr(n, "parms");
    Wrapper *w = NewWrapper();
    String *tm;
    String *wrap_args = NewStringEmpty();
    String *value = Getattr(n, "value");
    String *storage = Getattr(n, "storage");
    bool pure_virtual = false;
    int status = SWIG_OK;
    int idx;
    bool ignored_method = GetFlag(n, "feature:ignore") ? true : false;

    if (Cmp(storage, "virtual") == 0) {
      if (Cmp(value, "0") == 0) {
	pure_virtual = true;
      }
    }

    /* determine if the method returns a pointer */
    is_pointer = SwigType_ispointer_return(decl);
    is_void = (Cmp(returntype, "void") == 0 && !is_pointer);

    /* virtual method definition */
    String *target;
    String *pclassname = NewStringf("SwigDirector_%s", classname);
    String *qualified_name = NewStringf("%s::%s", pclassname, name);
    SwigType *rtype = Getattr(n, "conversion_operator") ? 0 : Getattr(n, "classDirectorMethods:type");
    target = Swig_method_decl(rtype, decl, qualified_name, l, 0);
    Printf(w->def, "%s", target);
    Delete(qualified_name);
    Delete(target);
    /* header declaration */
    target = Swig_method_decl(rtype, decl, name, l, 1);
    Printf(declaration, "    virtual %s", target);
    Delete(target);

    // Get any exception classes in the throws typemap
    if (Getattr(n, "noexcept")) {
      Append(w->def, " noexcept");
      Append(declaration, " noexcept");
    }
    ParmList *throw_parm_list = 0;

    if ((throw_parm_list = Getattr(n, "throws")) || Getattr(n, "throw")) {
      Parm *p;
      int gencomma = 0;

      Append(w->def, " throw(");
      Append(declaration, " throw(");

      if (throw_parm_list)
	Swig_typemap_attach_parms("throws", throw_parm_list, 0);
      for (p = throw_parm_list; p; p = nextSibling(p)) {
	if (Getattr(p, "tmap:throws")) {
	  if (gencomma++) {
	    Append(w->def, ", ");
	    Append(declaration, ", ");
	  }
	  String *str = SwigType_str(Getattr(p, "type"), 0);
	  Append(w->def, str);
	  Append(declaration, str);
	  Delete(str);
	}
      }

      Append(w->def, ")");
      Append(declaration, ")");
    }

    Append(w->def, " {");
    Append(declaration, ";\n");

    /* declare method return value
     * if the return value is a reference or const reference, a specialized typemap must
     * handle it, including declaration of c_result ($result).
     */
    if (!is_void && (!ignored_method || pure_virtual)) {
      if (!SwigType_isclass(returntype)) {
	if (!(SwigType_ispointer(returntype) || SwigType_isreference(returntype))) {
	  String *construct_result = NewStringf("= SwigValueInit< %s >()", SwigType_lstr(returntype, 0));
	  Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), construct_result, NIL);
	  Delete(construct_result);
	} else {
	  Wrapper_add_localv(w, "c_result", SwigType_lstr(returntype, "c_result"), "= 0", NIL);
	}
      } else {
	String *cres = SwigType_lstr(returntype, "c_result");
	Printf(w->code, "%s;\n", cres);
	Delete(cres);
      }
    }

    if (ignored_method) {
      if (!pure_virtual) {
	if (!is_void)
	  Printf(w->code, "return ");
	String *super_call = Swig_method_call(super, l);
	Printf(w->code, "%s;\n", super_call);
	Delete(super_call);
      } else {
	Printf(w->code, "Swig::DirectorPureVirtualException::raise(\"Attempted to invoke pure virtual method %s::%s\");\n", SwigType_namestr(c_classname),
	       SwigType_namestr(name));
      }
    } else {
      /* attach typemaps to arguments (C/C++ -> PHP) */
      Swig_director_parms_fixup(l);

      /* remove the wrapper 'w' since it was producing spurious temps */
      Swig_typemap_attach_parms("in", l, 0);
      Swig_typemap_attach_parms("directorin", l, w);
      Swig_typemap_attach_parms("directorargout", l, w);

      Parm *p;

      /* build argument list and type conversion string */
      idx = 0;
      p = l;
      while (p) {
	if (checkAttribute(p, "tmap:in:numinputs", "0")) {
	  p = Getattr(p, "tmap:in:next");
	  continue;
	}

	String *pname = Getattr(p, "name");
	String *ptype = Getattr(p, "type");

	if ((tm = Getattr(p, "tmap:directorin")) != 0) {
	  String *parse = Getattr(p, "tmap:directorin:parse");
	  if (!parse) {
	    String *input = NewStringf("&args[%d]", idx++);
	    Setattr(p, "emit:directorinput", input);
	    Replaceall(tm, "$input", input);
	    Delete(input);
	    Replaceall(tm, "$owner", "0");
	    Printv(wrap_args, tm, "\n", NIL);
	  } else {
	    Setattr(p, "emit:directorinput", pname);
	    Replaceall(tm, "$input", pname);
	    Replaceall(tm, "$owner", "0");
	    if (Len(tm) == 0)
	      Append(tm, pname);
	  }
	  p = Getattr(p, "tmap:directorin:next");
	  continue;
	} else if (Cmp(ptype, "void")) {
	  Swig_warning(WARN_TYPEMAP_DIRECTORIN_UNDEF, input_file, line_number,
	      "Unable to use type %s as a function argument in director method %s::%s (skipping method).\n", SwigType_str(ptype, 0),
	      SwigType_namestr(c_classname), SwigType_namestr(name));
	  status = SWIG_NOWRAP;
	  break;
	}
	p = nextSibling(p);
      }

      if (!idx) {
	Printf(w->code, "zval *args = NULL;\n");
      } else {
	Printf(w->code, "zval args[%d];\n", idx);
      }
      // typemap_directorout testcase requires that 0 can be assigned to the
      // variable named after the result of Swig_cresult_name(), so that can't
      // be a zval - make it a pointer to one instead.
      Printf(w->code, "zval swig_zval_result;\n");
      Printf(w->code, "zval * SWIGUNUSED %s = &swig_zval_result;\n", Swig_cresult_name());

      /* wrap complex arguments to zvals */
      Append(w->code, wrap_args);

      const char *funcname = GetChar(n, "sym:name");
      Append(w->code, "{\n");
      Append(w->code, "#if PHP_MAJOR_VERSION < 8\n");
      Printf(w->code, "zval swig_funcname;\n");
      Printf(w->code, "ZVAL_STRINGL(&swig_funcname, \"%s\", %d);\n", funcname, strlen(funcname));
      Printf(w->code, "call_user_function(EG(function_table), &swig_self, &swig_funcname, &swig_zval_result, %d, args);\n", idx);
      Append(w->code, "#else\n");
      Printf(w->code, "zend_string *swig_funcname = zend_string_init(\"%s\", %d, 0);\n", funcname, strlen(funcname));
      Append(w->code, "zend_function *swig_zend_func = zend_std_get_method(&Z_OBJ(swig_self), swig_funcname, NULL);\n");
      Append(w->code, "zend_string_release(swig_funcname);\n");
      Printf(w->code, "if (swig_zend_func) zend_call_known_instance_method(swig_zend_func, Z_OBJ(swig_self), &swig_zval_result, %d, args);\n", idx);
      Append(w->code, "#endif\n");

      /* exception handling */
      tm = Swig_typemap_lookup("director:except", n, Swig_cresult_name(), 0);
      if (!tm) {
	tm = Getattr(n, "feature:director:except");
	if (tm)
	  tm = Copy(tm);
      }
      if (!tm || Len(tm) == 0 || Equal(tm, "1")) {
	// Skip marshalling the return value as there isn't one.
	tm = NewString("if ($error) SWIG_fail;");
      }

      Replaceall(tm, "$error", "EG(exception)");
      Printv(w->code, Str(tm), "\n}\n{\n", NIL);
      Delete(tm);

      /* marshal return value from PHP to C/C++ type */

      String *cleanup = NewStringEmpty();
      String *outarg = NewStringEmpty();

      idx = 0;

      /* marshal return value */
      if (!is_void) {
	tm = Swig_typemap_lookup("directorout", n, Swig_cresult_name(), w);
	if (tm != 0) {
	  Replaceall(tm, "$input", Swig_cresult_name());
	  char temp[24];
	  sprintf(temp, "%d", idx);
	  Replaceall(tm, "$argnum", temp);

	  /* TODO check this */
	  if (Getattr(n, "wrap:disown")) {
	    Replaceall(tm, "$disown", "SWIG_POINTER_DISOWN");
	  } else {
	    Replaceall(tm, "$disown", "0");
	  }
	  Replaceall(tm, "$result", "c_result");
	  Printv(w->code, tm, "\n", NIL);
	  Delete(tm);
	} else {
	  Swig_warning(WARN_TYPEMAP_DIRECTOROUT_UNDEF, input_file, line_number,
		       "Unable to use return type %s in director method %s::%s (skipping method).\n", SwigType_str(returntype, 0),
		       SwigType_namestr(c_classname), SwigType_namestr(name));
	  status = SWIG_ERROR;
	}
      }

      /* marshal outputs */
      for (p = l; p;) {
	if ((tm = Getattr(p, "tmap:directorargout")) != 0) {
	  Replaceall(tm, "$result", Swig_cresult_name());
	  Replaceall(tm, "$input", Getattr(p, "emit:directorinput"));
	  Printv(w->code, tm, "\n", NIL);
	  p = Getattr(p, "tmap:directorargout:next");
	} else {
	  p = nextSibling(p);
	}
      }

      Append(w->code, "}\n");

      Delete(cleanup);
      Delete(outarg);
    }

    Append(w->code, "fail: ;\n");
    if (!is_void) {
      if (!(ignored_method && !pure_virtual)) {
	String *rettype = SwigType_str(returntype, 0);
	if (!SwigType_isreference(returntype)) {
	  Printf(w->code, "return (%s) c_result;\n", rettype);
	} else {
	  Printf(w->code, "return (%s) *c_result;\n", rettype);
	}
	Delete(rettype);
      }
    }
    Append(w->code, "}\n");

    // We expose protected methods via an extra public inline method which makes a straight call to the wrapped class' method
    String *inline_extra_method = NewStringEmpty();
    if (dirprot_mode() && !is_public(n) && !pure_virtual) {
      Printv(inline_extra_method, declaration, NIL);
      String *extra_method_name = NewStringf("%sSwigPublic", name);
      Replaceall(inline_extra_method, name, extra_method_name);
      Replaceall(inline_extra_method, ";\n", " {\n      ");
      if (!is_void)
	Printf(inline_extra_method, "return ");
      String *methodcall = Swig_method_call(super, l);
      Printv(inline_extra_method, methodcall, ";\n    }\n", NIL);
      Delete(methodcall);
      Delete(extra_method_name);
    }

    /* emit the director method */
    if (status == SWIG_OK) {
      if (!Getattr(n, "defaultargs")) {
	Replaceall(w->code, "$symname", symname);
	Wrapper_print(w, f_directors);
	Printv(f_directors_h, declaration, NIL);
	Printv(f_directors_h, inline_extra_method, NIL);
      }
    }

    /* clean up */
    Delete(wrap_args);
    Delete(pclassname);
    DelWrapper(w);
    return status;
  }

  int classDirectorDisown(Node *n) {
    wrapperType = directordisown;
    int result = Language::classDirectorDisown(n);
    wrapperType = standard;
    return result;
  }
};				/* class PHP */

static PHP *maininstance = 0;

List *PHPTypes::process_phptype(Node *n, int key, const String_or_char *attribute_name) {

  while (Len(merged_types) <= key) {
    Append(merged_types, NewList());
  }

  String *phptype = Getattr(n, attribute_name);
  if (!phptype || Len(phptype) == 0) {
    // There's no type declaration, so any merged version has no type declaration.
    //
    // Use a DOH None object as a marker to indicate there's no type
    // declaration for this parameter/return value (you can't store NULL as a
    // value in a DOH List).
    Setitem(merged_types, key, None);
    return NULL;
  }

  DOH *merge_list = Getitem(merged_types, key);
  if (merge_list == None) return NULL;

  List *types = Split(phptype, '|', -1);
  String *first_type = Getitem(types, 0);
  if (Char(first_type)[0] == '?') {
    if (Len(types) > 1) {
      Printf(stderr, "warning: Invalid phptype: '%s' (can't use ? and | together)\n", phptype);
    }
    // Treat `?foo` just like `foo|null`.
    Append(types, "null");
    Setitem(types, 0, NewString(Char(first_type) + 1));
  }

  SortList(types, NULL);
  String *prev = NULL;
  for (Iterator i = First(types); i.item; i = Next(i)) {
    if (prev && Equal(prev, i.item)) {
      Printf(stderr, "warning: Invalid phptype: '%s' (duplicate entry for '%s')\n", phptype, i.item);
      continue;
    }

    if (key > 0 && Equal(i.item, "void")) {
      // Reject void for parameter type.
      Printf(stderr, "warning: Invalid phptype: '%s' ('%s' can't be used as a parameter phptype)\n", phptype, i.item);
      continue;
    }

    if (Equal(i.item, "SWIGTYPE")) {
      String *type = Getattr(n, "type");
      Node *class_node = maininstance->classLookup(type);
      if (class_node) {
	// FIXME: Prefix classname with a backslash to prevent collisions
	// with built-in types?  Or are non of those valid anyway and so will
	// have been renamed at this point?
	Append(merge_list, Getattr(class_node, "sym:name"));
      } else {
	// SWIG wraps a pointer to a non-object type as an object in a PHP
	// class named based on the SWIG-mangled C/C++ type.
	//
	// FIXME: We should check this is actually a known pointer to
	// non-object type so we complain about `phptype="SWIGTYPE"` being
	// used for PHP types like `int` or `string` (currently this only
	// fails at runtime and the error isn't very helpful).  We could
	// check the condition
	//
	//   raw_pointer_types && Getattr(raw_pointer_types, SwigType_manglestr(type))
	//
	// except that raw_pointer_types may not have been fully filled in when
	// we are called.
	Append(merge_list, NewStringf("SWIG\\%s", SwigType_manglestr(type)));
      }
    } else {
      Append(merge_list, i.item);
    }
    prev = i.item;
  }
  SortList(merge_list, NULL);
  return merge_list;
}

void PHPTypes::merge_type_lists(List *merge_list, List *o_merge_list) {
  int i = 0, j = 0;
  while (j < Len(o_merge_list)) {
    String *candidate = Getitem(o_merge_list, j);
    while (i < Len(merge_list)) {
      int cmp = Cmp(Getitem(merge_list, i), candidate);
      if (cmp == 0)
	goto handled;
      if (cmp > 0)
	break;
      ++i;
    }
    Insert(merge_list, i, candidate);
    ++i;
handled:
    ++j;
  }
}

void PHPTypes::merge_from(const PHPTypes* o) {
  num_required = std::min(num_required, o->num_required);

  if (o->byref) {
    if (byref == NULL) {
      byref = Copy(o->byref);
    } else {
      int len = std::min(Len(byref), Len(o->byref));
      // Start at 1 because we only want to merge parameter types, and key 0 is
      // the return type.
      for (int key = 1; key < len; ++key) {
	if (Getitem(byref, key) == None &&
	    Getitem(o->byref, key) != None) {
	  Setitem(byref, key, "");
	}
      }
      for (int key = len; key < Len(o->byref); ++key) {
	Append(byref, Getitem(o->byref, key));
      }
    }
  }

  int len = std::min(Len(merged_types), Len(o->merged_types));
  for (int key = 0; key < len; ++key) {
    DOH *merge_list = Getitem(merged_types, key);
    // None trumps anything else in the merge.
    if (merge_list == None) continue;
    DOH *o_merge_list = Getitem(o->merged_types, key);
    if (o_merge_list == None) {
      Setitem(merged_types, key, None);
      continue;
    }
    merge_type_lists(merge_list, o_merge_list);
  }
  // Copy over any additional entries.
  for (int key = len; key < Len(o->merged_types); ++key) {
    Append(merged_types, Copy(Getitem(o->merged_types, key)));
  }
}

// Collect non-class pointer types from the type table so we can set up PHP
// classes for them later.
//
// NOTE: it's a function NOT A PHP::METHOD
extern "C" {
static void typetrace(const SwigType *ty, String *mangled, String *clientdata) {
  if (maininstance->classLookup(ty) == NULL) {
    // a non-class pointer
    if (!raw_pointer_types) {
      raw_pointer_types = NewHash();
    }
    Setattr(raw_pointer_types, mangled, mangled);
  }
  if (r_prevtracefunc)
    (*r_prevtracefunc) (ty, mangled, clientdata);
}
}

/* -----------------------------------------------------------------------------
 * new_swig_php()    - Instantiate module
 * ----------------------------------------------------------------------------- */

static Language *new_swig_php() {
  maininstance = new PHP;
  if (!r_prevtracefunc) {
    r_prevtracefunc = SwigType_remember_trace(typetrace);
  } else {
    Printf(stderr, "php Typetrace vector already saved!\n");
    assert(0);
  }
  return maininstance;
}

extern "C" Language *swig_php(void) {
  return new_swig_php();
}
