//
//  Copyright 2019 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// -----------------------------------------------------------------------------
// File: flag.h
// -----------------------------------------------------------------------------
//
// This header file defines the `y_absl::Flag<T>` type for holding command-line
// flag data, and abstractions to create, get and set such flag data.
//
// It is important to note that this type is **unspecified** (an implementation
// detail) and you do not construct or manipulate actual `y_absl::Flag<T>`
// instances. Instead, you define and declare flags using the
// `Y_ABSL_FLAG()` and `Y_ABSL_DECLARE_FLAG()` macros, and get and set flag values
// using the `y_absl::GetFlag()` and `y_absl::SetFlag()` functions.

#ifndef Y_ABSL_FLAGS_FLAG_H_
#define Y_ABSL_FLAGS_FLAG_H_

#include <cstdint>
#include <util/generic/string.h>
#include <type_traits>

#include "y_absl/base/attributes.h"
#include "y_absl/base/config.h"
#include "y_absl/base/optimization.h"
#include "y_absl/flags/commandlineflag.h"
#include "y_absl/flags/config.h"
#include "y_absl/flags/internal/flag.h"
#include "y_absl/flags/internal/registry.h"
#include "y_absl/strings/string_view.h"

namespace y_absl {
Y_ABSL_NAMESPACE_BEGIN

// Flag
//
// An `y_absl::Flag` holds a command-line flag value, providing a runtime
// parameter to a binary. Such flags should be defined in the global namespace
// and (preferably) in the module containing the binary's `main()` function.
//
// You should not construct and cannot use the `y_absl::Flag` type directly;
// instead, you should declare flags using the `Y_ABSL_DECLARE_FLAG()` macro
// within a header file, and define your flag using `Y_ABSL_FLAG()` within your
// header's associated `.cc` file. Such flags will be named `FLAGS_name`.
//
// Example:
//
//    .h file
//
//      // Declares usage of a flag named "FLAGS_count"
//      Y_ABSL_DECLARE_FLAG(int, count);
//
//    .cc file
//
//      // Defines a flag named "FLAGS_count" with a default `int` value of 0.
//      Y_ABSL_FLAG(int, count, 0, "Count of items to process");
//
// No public methods of `y_absl::Flag<T>` are part of the Abseil Flags API.
//
// For type support of Abseil Flags, see the marshalling.h header file, which
// discusses supported standard types, optional flags, and additional Abseil
// type support.

template <typename T>
using Flag = flags_internal::Flag<T>;

// GetFlag()
//
// Returns the value (of type `T`) of an `y_absl::Flag<T>` instance, by value. Do
// not construct an `y_absl::Flag<T>` directly and call `y_absl::GetFlag()`;
// instead, refer to flag's constructed variable name (e.g. `FLAGS_name`).
// Because this function returns by value and not by reference, it is
// thread-safe, but note that the operation may be expensive; as a result, avoid
// `y_absl::GetFlag()` within any tight loops.
//
// Example:
//
//   // FLAGS_count is a Flag of type `int`
//   int my_count = y_absl::GetFlag(FLAGS_count);
//
//   // FLAGS_firstname is a Flag of type `TString`
//   TString first_name = y_absl::GetFlag(FLAGS_firstname);
template <typename T>
Y_ABSL_MUST_USE_RESULT T GetFlag(const y_absl::Flag<T>& flag) {
  return flags_internal::FlagImplPeer::InvokeGet<T>(flag);
}

// SetFlag()
//
// Sets the value of an `y_absl::Flag` to the value `v`. Do not construct an
// `y_absl::Flag<T>` directly and call `y_absl::SetFlag()`; instead, use the
// flag's variable name (e.g. `FLAGS_name`). This function is
// thread-safe, but is potentially expensive. Avoid setting flags in general,
// but especially within performance-critical code.
template <typename T>
void SetFlag(y_absl::Flag<T>* flag, const T& v) {
  flags_internal::FlagImplPeer::InvokeSet(*flag, v);
}

// Overload of `SetFlag()` to allow callers to pass in a value that is
// convertible to `T`. E.g., use this overload to pass a "const char*" when `T`
// is `TString`.
template <typename T, typename V>
void SetFlag(y_absl::Flag<T>* flag, const V& v) {
  T value(v);
  flags_internal::FlagImplPeer::InvokeSet(*flag, value);
}

// GetFlagReflectionHandle()
//
// Returns the reflection handle corresponding to specified Abseil Flag
// instance. Use this handle to access flag's reflection information, like name,
// location, default value etc.
//
// Example:
//
//   TString = y_absl::GetFlagReflectionHandle(FLAGS_count).DefaultValue();

template <typename T>
const CommandLineFlag& GetFlagReflectionHandle(const y_absl::Flag<T>& f) {
  return flags_internal::FlagImplPeer::InvokeReflect(f);
}

Y_ABSL_NAMESPACE_END
}  // namespace y_absl


// Y_ABSL_FLAG()
//
// This macro defines an `y_absl::Flag<T>` instance of a specified type `T`:
//
//   Y_ABSL_FLAG(T, name, default_value, help);
//
// where:
//
//   * `T` is a supported flag type (see the list of types in `marshalling.h`),
//   * `name` designates the name of the flag (as a global variable
//     `FLAGS_name`),
//   * `default_value` is an expression holding the default value for this flag
//     (which must be implicitly convertible to `T`),
//   * `help` is the help text, which can also be an expression.
//
// This macro expands to a flag named 'FLAGS_name' of type 'T':
//
//   y_absl::Flag<T> FLAGS_name = ...;
//
// Note that all such instances are created as global variables.
//
// For `Y_ABSL_FLAG()` values that you wish to expose to other translation units,
// it is recommended to define those flags within the `.cc` file associated with
// the header where the flag is declared.
//
// Note: do not construct objects of type `y_absl::Flag<T>` directly. Only use the
// `Y_ABSL_FLAG()` macro for such construction.
#define Y_ABSL_FLAG(Type, name, default_value, help) \
  Y_ABSL_FLAG_IMPL(Type, name, default_value, help)

// Y_ABSL_FLAG().OnUpdate()
//
// Defines a flag of type `T` with a callback attached:
//
//   Y_ABSL_FLAG(T, name, default_value, help).OnUpdate(callback);
//
// `callback` should be convertible to `void (*)()`.
//
// After any setting of the flag value, the callback will be called at least
// once. A rapid sequence of changes may be merged together into the same
// callback. No concurrent calls to the callback will be made for the same
// flag. Callbacks are allowed to read the current value of the flag but must
// not mutate that flag.
//
// The update mechanism guarantees "eventual consistency"; if the callback
// derives an auxiliary data structure from the flag value, it is guaranteed
// that eventually the flag value and the derived data structure will be
// consistent.
//
// Note: Y_ABSL_FLAG.OnUpdate() does not have a public definition. Hence, this
// comment serves as its API documentation.

// -----------------------------------------------------------------------------
// Implementation details below this section
// -----------------------------------------------------------------------------

// Y_ABSL_FLAG_IMPL macro definition conditional on Y_ABSL_FLAGS_STRIP_NAMES
#define Y_ABSL_FLAG_IMPL_FLAG_PTR(flag) flag
#define Y_ABSL_FLAG_IMPL_HELP_ARG(name)                      \
  y_absl::flags_internal::HelpArg<AbslFlagHelpGenFor##name>( \
      FLAGS_help_storage_##name)
#define Y_ABSL_FLAG_IMPL_DEFAULT_ARG(Type, name) \
  y_absl::flags_internal::DefaultArg<Type, AbslFlagDefaultGenFor##name>(0)

#if Y_ABSL_FLAGS_STRIP_NAMES
#define Y_ABSL_FLAG_IMPL_FLAGNAME(txt) ""
#define Y_ABSL_FLAG_IMPL_FILENAME() ""
#define Y_ABSL_FLAG_IMPL_REGISTRAR(T, flag)                                      \
  y_absl::flags_internal::FlagRegistrar<T, false>(Y_ABSL_FLAG_IMPL_FLAG_PTR(flag), \
                                                nullptr)
#else
#define Y_ABSL_FLAG_IMPL_FLAGNAME(txt) txt
#define Y_ABSL_FLAG_IMPL_FILENAME() __FILE__
#define Y_ABSL_FLAG_IMPL_REGISTRAR(T, flag)                                     \
  y_absl::flags_internal::FlagRegistrar<T, true>(Y_ABSL_FLAG_IMPL_FLAG_PTR(flag), \
                                               __FILE__)
#endif

// Y_ABSL_FLAG_IMPL macro definition conditional on Y_ABSL_FLAGS_STRIP_HELP

#if Y_ABSL_FLAGS_STRIP_HELP
#define Y_ABSL_FLAG_IMPL_FLAGHELP(txt) y_absl::flags_internal::kStrippedFlagHelp
#else
#define Y_ABSL_FLAG_IMPL_FLAGHELP(txt) txt
#endif

// AbslFlagHelpGenFor##name is used to encapsulate both immediate (method Const)
// and lazy (method NonConst) evaluation of help message expression. We choose
// between the two via the call to HelpArg in y_absl::Flag instantiation below.
// If help message expression is constexpr evaluable compiler will optimize
// away this whole struct.
// TODO(rogeeff): place these generated structs into local namespace and apply
// Y_ABSL_INTERNAL_UNIQUE_SHORT_NAME.
// TODO(rogeeff): Apply __attribute__((nodebug)) to FLAGS_help_storage_##name
#define Y_ABSL_FLAG_IMPL_DECLARE_HELP_WRAPPER(name, txt)                       \
  struct AbslFlagHelpGenFor##name {                                          \
    /* The expression is run in the caller as part of the   */               \
    /* default value argument. That keeps temporaries alive */               \
    /* long enough for NonConst to work correctly.          */               \
    static constexpr y_absl::string_view Value(                                \
        y_absl::string_view absl_flag_help = Y_ABSL_FLAG_IMPL_FLAGHELP(txt)) {   \
      return absl_flag_help;                                                 \
    }                                                                        \
    static TString NonConst() { return TString(Value()); }           \
  };                                                                         \
  constexpr auto FLAGS_help_storage_##name Y_ABSL_INTERNAL_UNIQUE_SMALL_NAME() \
      Y_ABSL_ATTRIBUTE_SECTION_VARIABLE(flags_help_cold) =                     \
          y_absl::flags_internal::HelpStringAsArray<AbslFlagHelpGenFor##name>( \
              0);

#define Y_ABSL_FLAG_IMPL_DECLARE_DEF_VAL_WRAPPER(name, Type, default_value)     \
  struct AbslFlagDefaultGenFor##name {                                        \
    Type value = y_absl::flags_internal::InitDefaultValue<Type>(default_value); \
    static void Gen(void* absl_flag_default_loc) {                            \
      new (absl_flag_default_loc) Type(AbslFlagDefaultGenFor##name{}.value);  \
    }                                                                         \
  };

// Y_ABSL_FLAG_IMPL
//
// Note: Name of registrar object is not arbitrary. It is used to "grab"
// global name for FLAGS_no<flag_name> symbol, thus preventing the possibility
// of defining two flags with names foo and nofoo.
#define Y_ABSL_FLAG_IMPL(Type, name, default_value, help)                       \
  extern ::y_absl::Flag<Type> FLAGS_##name;                                     \
  namespace y_absl /* block flags in namespaces */ {}                           \
  Y_ABSL_FLAG_IMPL_DECLARE_DEF_VAL_WRAPPER(name, Type, default_value)           \
  Y_ABSL_FLAG_IMPL_DECLARE_HELP_WRAPPER(name, help)                             \
  Y_ABSL_CONST_INIT y_absl::Flag<Type> FLAGS_##name{                              \
      Y_ABSL_FLAG_IMPL_FLAGNAME(#name), Y_ABSL_FLAG_IMPL_FILENAME(),              \
      Y_ABSL_FLAG_IMPL_HELP_ARG(name), Y_ABSL_FLAG_IMPL_DEFAULT_ARG(Type, name)}; \
  extern y_absl::flags_internal::FlagRegistrarEmpty FLAGS_no##name;             \
  y_absl::flags_internal::FlagRegistrarEmpty FLAGS_no##name =                   \
      Y_ABSL_FLAG_IMPL_REGISTRAR(Type, FLAGS_##name)

// Y_ABSL_RETIRED_FLAG
//
// Designates the flag (which is usually pre-existing) as "retired." A retired
// flag is a flag that is now unused by the program, but may still be passed on
// the command line, usually by production scripts. A retired flag is ignored
// and code can't access it at runtime.
//
// This macro registers a retired flag with given name and type, with a name
// identical to the name of the original flag you are retiring. The retired
// flag's type can change over time, so that you can retire code to support a
// custom flag type.
//
// This macro has the same signature as `Y_ABSL_FLAG`. To retire a flag, simply
// replace an `Y_ABSL_FLAG` definition with `Y_ABSL_RETIRED_FLAG`, leaving the
// arguments unchanged (unless of course you actually want to retire the flag
// type at this time as well).
//
// `default_value` is only used as a double check on the type. `explanation` is
// unused.
// TODO(rogeeff): replace RETIRED_FLAGS with FLAGS once forward declarations of
// retired flags are cleaned up.
#define Y_ABSL_RETIRED_FLAG(type, name, default_value, explanation)      \
  static y_absl::flags_internal::RetiredFlag<type> RETIRED_FLAGS_##name; \
  Y_ABSL_ATTRIBUTE_UNUSED static const auto RETIRED_FLAGS_REG_##name =   \
      (RETIRED_FLAGS_##name.Retire(#name),                             \
       ::y_absl::flags_internal::FlagRegistrarEmpty{})

#endif  // Y_ABSL_FLAGS_FLAG_H_
