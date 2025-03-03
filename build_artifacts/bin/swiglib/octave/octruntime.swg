#ifdef SWIG_OCTAVE_EXTERNAL_OCTHEADERS
%insert(runtime) %{
#include "octheaders.hpp"
%}
#else
%insert(runtime) "octheaders.hpp";
#endif

%insert(runtime) "swigrun.swg";
%insert(runtime) "swigerrors.swg";
%insert(runtime) "octrun.swg";

%insert(initbeforefunc) "swiginit.swg"

%insert(initbeforefunc) %{

static bool SWIG_init_user(octave_swig_type* module_ns);

SWIGINTERN bool SWIG_Octave_LoadModule(std::string name) {
  bool retn = false;
  {
#if SWIG_OCTAVE_PREREQ(4,2,0)
    octave::unwind_protect frame;
    frame.protect_var(discard_error_messages);          discard_error_messages = true;
    frame.protect_var(discard_warning_messages);        discard_warning_messages = true;
#elif SWIG_OCTAVE_PREREQ(3,3,50)
    unwind_protect frame;
    frame.protect_var(error_state);                     error_state = 0;
    frame.protect_var(warning_state);                   warning_state = 0;
    frame.protect_var(discard_error_messages);          discard_error_messages = true;
    frame.protect_var(discard_warning_messages);        discard_warning_messages = true;
#else
    unwind_protect::begin_frame("SWIG_Octave_LoadModule");
    unwind_protect_int(error_state);                    error_state = 0;
    unwind_protect_int(warning_state);                  warning_state = 0;
    unwind_protect_bool(discard_error_messages);        discard_error_messages = true;
    unwind_protect_bool(discard_warning_messages);      discard_warning_messages = true;
#endif
#if SWIG_OCTAVE_PREREQ(4,2,0)
    try {
#if SWIG_OCTAVE_PREREQ(4,4,0)
      octave::feval(name, octave_value_list(), 0);
#else
      feval(name, octave_value_list(), 0);
#endif
      retn = true;
    } catch (octave::execution_exception&) { }
#else
    feval(name, octave_value_list(), 0);
    retn = (error_state == 0);
#endif
#if !SWIG_OCTAVE_PREREQ(3,3,50)
    unwind_protect::run_frame("SWIG_Octave_LoadModule");
#endif
  }
  if (!retn) {
    error(SWIG_name_d ": could not load module `%s'", name.c_str());
  }
  return retn;
}

SWIGINTERN bool SWIG_Octave_InstallFunction(octave_function *octloadfcn, std::string name) {
  bool retn = false;
  {
#if SWIG_OCTAVE_PREREQ(4,2,0)
    octave::unwind_protect frame;
    frame.protect_var(discard_error_messages);          discard_error_messages = true;
    frame.protect_var(discard_warning_messages);        discard_warning_messages = true;
#elif SWIG_OCTAVE_PREREQ(3,3,50)
    unwind_protect frame;
    frame.protect_var(error_state);                     error_state = 0;
    frame.protect_var(warning_state);                   warning_state = 0;
    frame.protect_var(discard_error_messages);          discard_error_messages = true;
    frame.protect_var(discard_warning_messages);        discard_warning_messages = true;
#else
    unwind_protect::begin_frame("SWIG_Octave_InstallFunction");
    unwind_protect_int(error_state);                    error_state = 0;
    unwind_protect_int(warning_state);                  warning_state = 0;
    unwind_protect_bool(discard_error_messages);        discard_error_messages = true;
    unwind_protect_bool(discard_warning_messages);      discard_warning_messages = true;
#endif
    octave_value_list args;
    args.append(name);
    args.append(octloadfcn->fcn_file_name());
#if SWIG_OCTAVE_PREREQ(4,2,0)
    try {
#if SWIG_OCTAVE_PREREQ(4,4,0)
      octave::feval("autoload", args, 0);
#else
      feval("autoload", args, 0);
#endif
      retn = true;
    } catch (octave::execution_exception&) { }
#else
    feval("autoload", args, 0);
    retn = (error_state == 0);
#endif
#if !SWIG_OCTAVE_PREREQ(3,3,50)
    unwind_protect::run_frame("SWIG_Octave_InstallFunction");
#endif
  }
  if (!retn) {
    error(SWIG_name_d ": could not load function `%s'", name.c_str());
  }
  return retn;
}

static const char *const subclass_usage = "-*- texinfo -*- \n\
@deftypefn {Loadable Function} {} subclass()\n\
@deftypefnx{Loadable Function} {} subclass(@var{swigclass}, @var{name}, @var{fcn}, @dots{})\n\
Subclass a C++ class from within Octave, and provide implementations of its virtual methods.\n\
\n\
See the SWIG manual for usage examples.\n\
@end deftypefn";

DEFUN_DLD( subclass, args, nargout, subclass_usage ) {
  octave_swig_type *top = new octave_swig_type;
  for (int j = 0; j < args.length(); ++j) {
    if (args(j).type_id() == octave_swig_ref::static_type_id()) {
      octave_swig_ref *osr = static_cast < octave_swig_ref *>(args(j).internal_rep());
      octave_swig_type *ost = osr->get_ptr();
      if (!ost->is_owned()) {
        error("subclass: cannot subclass object not constructed on octave side");
        return octave_value_list();
      }
      top->merge(*ost);
    } else if (args(j).is_function_handle()) {
      top->assign(args(j).fcn_handle_value()->fcn_name(), args(j));
    } else if (args(j).is_string()) {
      if (j + 1 >= args.length()) {
        error("subclass: member assignments must be of string,value form");
        return octave_value_list();
      }
      top->assign(args(j).string_value(), args(j + 1));
      ++j;
    } else {
      error("subclass: invalid arguments to subclass()");
      return octave_value_list();
    }
  }
  return octave_value(Swig::swig_value_ref(top));
}

static const char *const swig_type_usage = "-*- texinfo -*- \n\
@deftypefn {Loadable Function} {} swig_type(@var{swigref})\n\
Return the underlying C/C++ type name of a SWIG-wrapped object.\n\
@end deftypefn";

DEFUN_DLD( swig_type, args, nargout, swig_type_usage ) {
  if (args.length() != 1) {
    error("swig_type: must be called with only a single object");
    return octave_value_list();
  }
  octave_swig_type *ost = Swig::swig_value_deref(args(0));
  if (!ost) {
    error("swig_type: object is not a swig_ref");
    return octave_value_list();
  }
  return octave_value(ost->swig_type_name());
}

static const char *const swig_typequery_usage = "-*- texinfo -*- \n\
@deftypefn {Loadable Function} {} swig_typequery(@var{string})\n\
Return @var{string} if it is a recognised SWIG-wrapped C/C++ type name;\n\
otherwise return `<unknown>'.\n\
@end deftypefn";

DEFUN_DLD( swig_typequery, args, nargout, swig_typequery_usage ) {
  if (args.length() != 1 || !args(0).is_string()) {
    error("swig_typequery: must be called with single string argument");
    return octave_value_list();
  }
  swig_module_info *module = SWIG_GetModule(0);
  swig_type_info *type = SWIG_TypeQueryModule(module, module, args(0).string_value().c_str());
  if (!type)
    return octave_value("<unknown>");
  return octave_value(type->name);
}

static const char *const swig_this_usage = "-*- texinfo -*- \n\
@deftypefn {Loadable Function} {} swig_this(@var{swigref})\n\
Return the underlying C/C++ pointer of a SWIG-wrapped object.\n\
@end deftypefn";

DEFUN_DLD( swig_this, args, nargout, swig_this_usage ) {
  if (args.length() != 1) {
    error("swig_this: must be called with only a single object");
    return octave_value_list();
  }
  if (args(0).is_matrix_type() && args(0).rows() == 0 && args(0).columns() == 0)
    return octave_value(octave_uint64(0));
  octave_swig_type *ost = Swig::swig_value_deref(args(0));
  if (!ost) {
    error("swig_this: object is not a swig_ref");
    return octave_value_list();
  }
  return octave_value(octave_uint64((unsigned long long) ost->swig_this()));
}

static const char *const swig_octave_prereq_usage = "-*- texinfo -*- \n\
@deftypefn {Loadable Function} {} swig_octave_prereq(@var{major}, @var{minor}, @var{patch})\n\
Return true if the version of Octave is at least @var{major}.@var{minor}.@var{patch}.\n\
@end deftypefn";

DEFUN_DLD( swig_octave_prereq, args, nargout, swig_octave_prereq_usage ) {
  if (args.length() != 3) {
    error("swig_octave_prereq: must be called with 3 arguments");
    return octave_value_list();
  }
  const int major = args(0).int_value();
  const int minor = args(1).int_value();
  const int patch = args(2).int_value();
  const bool prereq = SWIG_OCTAVE_PREREQ(major, minor, patch);
  return octave_value(prereq);
}

static const char *const swig_exit_usage = "-*- texinfo -*- \n\
@deftypefn {Loadable Function} {} swig_exit([@var{exit_status}])\n\
Exit Octave without performing any memory cleanup.\n\
@end deftypefn";

DEFUN_DLD( swig_exit, args, nargout, swig_exit_usage ) {
  if (args.length() > 1) {
    error("swig_exit: must be called with at most one arguments");
    return octave_value_list();
  }
  int exit_status = 0;
  if (args.length() == 1) {
    exit_status = args(0).int_value();
  }
  ::_Exit(exit_status);
  return octave_value();
}

static const char *const SWIG_name_usage = "-*- texinfo -*- \n\
@deftypefn {Loadable Module} {} " SWIG_name_d "\n\
Loads the SWIG-generated module `" SWIG_name_d "'.\n\
@end deftypefn";

DEFUN_DLD( SWIG_name, args, nargout, SWIG_name_usage ) {

  static octave_swig_type* module_ns = 0;

  // workaround to prevent octave seg-faulting on exit: set Octave exit function
  // octave_exit to _Exit, which exits immediately without trying to cleanup memory.
  // definitely affected version 3.2.*, not sure about 3.3.*, seems to be fixed in
  // version 3.4.*, reappeared in 4.2.*, hack not possible in 4.4.* or later due to
  // removal of octave_exit, so turn on for all versions between 3.2.*. and 4.4.*.
  // can be turned off with macro definition.
#ifndef SWIG_OCTAVE_NO_SEGFAULT_HACK
#if !SWIG_OCTAVE_PREREQ(4,4,0)
#if SWIG_OCTAVE_PREREQ(3,2,0)
  octave_exit = ::_Exit;
#endif
#endif
#endif

  // check for no input and output args
  if (args.length() != 0 || nargout != 0) {
    print_usage();
    return octave_value_list();
  }

  // create module on first function call
  if (!module_ns) {

    // workaround bug in octave where installing global variable of custom type and then
    // exiting without explicitly clearing the variable causes octave to segfault.
#if SWIG_OCTAVE_PREREQ(3,2,0)
    octave_value_list eval_args;
    eval_args.append("base");
    eval_args.append("function __swig_atexit__; "
                     "  if mislocked() "
                     "    clear -all; "
                     "  else "
                     "    mlock(); "
                     "  endif; "
                     "endfunction; "
                     "__swig_atexit__; "
                     "atexit(\"__swig_atexit__\", false); "
                     "atexit(\"__swig_atexit__\")");
#if SWIG_OCTAVE_PREREQ(4,4,0)
    octave::feval("evalin", eval_args, 0);
#else
    feval("evalin", eval_args, 0);
#endif
#endif

#if SWIG_OCTAVE_PREREQ(4,4,0)
    {
      octave::type_info& typeinfo = octave::interpreter::the_interpreter()->get_type_info();
      string_vector types = typeinfo.installed_type_names();
      bool register_octave_swig_ref = true;
      bool register_octave_swig_packed = true;
      for (int i = 0; i < types.numel(); ++i) {
        if (types(i) == octave_swig_ref::static_type_name()) {
          register_octave_swig_ref = false;
        }
        if (types(i) == octave_swig_packed::static_type_name()) {
          register_octave_swig_packed = false;
        }
      }
      if (register_octave_swig_ref) {
        octave_swig_ref::register_type();
      }
      if (register_octave_swig_packed) {
        octave_swig_packed::register_type();
      }
    }
#else
    octave_swig_ref::register_type();
    octave_swig_packed::register_type();
#endif
    SWIG_InitializeModule(0);
    SWIG_PropagateClientData();

#if SWIG_OCTAVE_PREREQ(4,4,0)
    octave::call_stack& stack = octave::interpreter::the_interpreter()->get_call_stack();
    octave_function *me = stack.current();
#else
    octave_function *me = octave_call_stack::current();
#endif

    if (!SWIG_Octave_InstallFunction(me, "subclass")) {
      return octave_value_list();
    }
    if (!SWIG_Octave_InstallFunction(me, "swig_type")) {
      return octave_value_list();
    }
    if (!SWIG_Octave_InstallFunction(me, "swig_typequery")) {
      return octave_value_list();
    }
    if (!SWIG_Octave_InstallFunction(me, "swig_this")) {
      return octave_value_list();
    }
    if (!SWIG_Octave_InstallFunction(me, "swig_octave_prereq")) {
      return octave_value_list();
    }
    if (!SWIG_Octave_InstallFunction(me, "swig_exit")) {
      return octave_value_list();
    }

    octave_swig_type* cvar_ns=0;
    if (std::string(SWIG_global_name) != ".") {
      cvar_ns=new octave_swig_type;
      for (int j=0;swig_globals[j].name;++j)
        if (swig_globals[j].get_method)
          cvar_ns->assign(swig_globals[j].name,&swig_globals[j]);
    }

    module_ns=new octave_swig_type(0, 0, 0, true);
    if (std::string(SWIG_global_name) != ".") {
      module_ns->assign(SWIG_global_name,Swig::swig_value_ref(cvar_ns));
    }
    else {
      for (int j=0;swig_globals[j].name;++j)
        if (swig_globals[j].get_method)
          module_ns->assign(swig_globals[j].name,&swig_globals[j]);
    }
    for (int j=0;swig_globals[j].name;++j)
      if (swig_globals[j].method)
        module_ns->assign(swig_globals[j].name,&swig_globals[j]);

    // * need better solution here; swig_type -> octave_class mapping is
    // * really n-to-1, in some cases such as template partial spec, etc.
    // * see failing tests.
    for (int j=0;swig_types[j];++j)
      if (swig_types[j]->clientdata) {
        swig_octave_class* c=(swig_octave_class*)swig_types[j]->clientdata;
        module_ns->assign(c->name,
                        Swig::swig_value_ref
                        (new octave_swig_type(0,swig_types[j])));
      }

    if (!SWIG_init_user(module_ns)) {
      delete module_ns;
      module_ns=0;
      return octave_value_list();
    }

    SWIG_InstallOps(octave_swig_ref::static_type_id());

    octave_swig_type::swig_member_const_iterator mb;
    for (mb = module_ns->swig_members_begin(); mb != module_ns->swig_members_end(); ++mb) {
      if (mb->second.first && mb->second.first->method) {
        if (!SWIG_Octave_InstallFunction(me, mb->first)) {
          return octave_value_list();
        }
      }
    }

#if SWIG_OCTAVE_PREREQ(4,4,0)
    octave::interpreter::the_interpreter()->mlock();
#elif SWIG_OCTAVE_PREREQ(3,2,0)
    mlock();
#else
    mlock(me->name());
#endif

  }

  octave_swig_type::swig_member_const_iterator mb;
  for (mb = module_ns->swig_members_begin(); mb != module_ns->swig_members_end(); ++mb) {
    if (mb->second.second.is_defined()) {
      SWIG_Octave_SetGlobalValue(mb->first, mb->second.second);
      SWIG_Octave_LinkGlobalValue(mb->first);
    }
  }

  SWIG_Octave_SetGlobalValue(SWIG_name_d, module_ns->as_value());
  SWIG_Octave_LinkGlobalValue(SWIG_name_d);

  return octave_value_list();

}

%}
