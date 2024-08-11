/* ATTRIBUTE_* macros for using attributes in GCC and similar compilers

   Copyright 2020 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published
   by the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* Written by Paul Eggert.  */

/* Provide public ATTRIBUTE_* names for the private _GL_ATTRIBUTE_*
   macros used within Gnulib.  */

/* These attributes can be placed in two ways:
     - At the start of a declaration (i.e. even before storage-class
       specifiers!); then they apply to all entities that are declared
       by the declaration.
     - Immediately after the name of an entity being declared by the
       declaration; then they apply to that entity only.  */

#ifndef _GL_ATTRIBUTE_H
#define _GL_ATTRIBUTE_H


/* This file defines two types of attributes:
   * C2X standard attributes.  These have macro names that do not begin with
     'ATTRIBUTE_'.
   * Selected GCC attributes; see:
     https://gcc.gnu.org/onlinedocs/gcc/Common-Function-Attributes.html
     https://gcc.gnu.org/onlinedocs/gcc/Common-Variable-Attributes.html
     https://gcc.gnu.org/onlinedocs/gcc/Common-Type-Attributes.html
     These names begin with 'ATTRIBUTE_' to avoid name clashes.  */


/* =============== Attributes for specific kinds of functions =============== */

/* Attributes for functions that should not be used.  */

/* Warn if the entity is used.  */
/* Applies to:
     - function, variable,
     - struct, union, struct/union member,
     - enumeration, enumeration item,
     - typedef,
   in C++ also: namespace, class, template specialization.  */
#define DEPRECATED _GL_ATTRIBUTE_DEPRECATED

/* If a function call is not optimized way, warn with MSG.  */
/* Applies to: functions.  */
#define ATTRIBUTE_WARNING(msg) _GL_ATTRIBUTE_WARNING (msg)

/* If a function call is not optimized way, report an error with MSG.  */
/* Applies to: functions.  */
#define ATTRIBUTE_ERROR(msg) _GL_ATTRIBUTE_ERROR (msg)


/* Attributes for memory-allocating functions.  */

/* The function returns a pointer to freshly allocated memory.  */
/* Applies to: functions.  */
#define ATTRIBUTE_MALLOC _GL_ATTRIBUTE_MALLOC

/* ATTRIBUTE_ALLOC_SIZE ((N)) - The Nth argument of the function
   is the size of the returned memory block.
   ATTRIBUTE_ALLOC_SIZE ((M, N)) - Multiply the Mth and Nth arguments
   to determine the size of the returned memory block.  */
/* Applies to: function, pointer to function, function types.  */
#define ATTRIBUTE_ALLOC_SIZE(args) _GL_ATTRIBUTE_ALLOC_SIZE (args)


/* Attributes for variadic functions.  */

/* The variadic function expects a trailing NULL argument.
   ATTRIBUTE_SENTINEL () - The last argument is NULL (requires C99).
   ATTRIBUTE_SENTINEL ((N)) - The (N+1)st argument from the end is NULL.  */
/* Applies to: functions.  */
#define ATTRIBUTE_SENTINEL(pos) _GL_ATTRIBUTE_SENTINEL (pos)


/* ================== Attributes for compiler diagnostics ================== */

/* Attributes that help the compiler diagnose programmer mistakes.
   Some of them may also help for some compiler optimizations.  */

/* ATTRIBUTE_FORMAT ((ARCHETYPE, STRING-INDEX, FIRST-TO-CHECK)) -
   The STRING-INDEXth function argument is a format string of style
   ARCHETYPE, which is one of:
     printf, gnu_printf
     scanf, gnu_scanf,
     strftime, gnu_strftime,
     strfmon,
   or the same thing prefixed and suffixed with '__'.
   If FIRST-TO-CHECK is not 0, arguments starting at FIRST-TO_CHECK
   are suitable for the format string.  */
/* Applies to: functions.  */
#define ATTRIBUTE_FORMAT(spec) _GL_ATTRIBUTE_FORMAT (spec)

/* ATTRIBUTE_NONNULL ((N1, N2,...)) - Arguments N1, N2,... must not be NULL.
   ATTRIBUTE_NONNULL () - All pointer arguments must not be null.  */
/* Applies to: functions.  */
#define ATTRIBUTE_NONNULL(args) _GL_ATTRIBUTE_NONNULL (args)

/* The function's return value is a non-NULL pointer.  */
/* Applies to: functions.  */
#define ATTRIBUTE_RETURNS_NONNULL _GL_ATTRIBUTE_RETURNS_NONNULL

/* Warn if the caller does not use the return value,
   unless the caller uses something like ignore_value.  */
/* Applies to: function, enumeration, class.  */
#define NODISCARD _GL_ATTRIBUTE_NODISCARD


/* Attributes that disable false alarms when the compiler diagnoses
   programmer "mistakes".  */

/* Do not warn if the entity is not used.  */
/* Applies to:
     - function, variable,
     - struct, union, struct/union member,
     - enumeration, enumeration item,
     - typedef,
   in C++ also: class.  */
#define MAYBE_UNUSED _GL_ATTRIBUTE_MAYBE_UNUSED

/* The contents of a character array is not meant to be NUL-terminated.  */
/* Applies to: struct/union members and variables that are arrays of element
   type '[[un]signed] char'.  */
#define ATTRIBUTE_NONSTRING _GL_ATTRIBUTE_NONSTRING

/* Do not warn if control flow falls through to the immediately
   following 'case' or 'default' label.  */
/* Applies to: Empty statement (;), inside a 'switch' statement.  */
#define FALLTHROUGH _GL_ATTRIBUTE_FALLTHROUGH


/* ================== Attributes for debugging information ================== */

/* Attributes regarding debugging information emitted by the compiler.  */

/* Omit the function from stack traces when debugging.  */
/* Applies to: function.  */
#define ATTRIBUTE_ARTIFICIAL _GL_ATTRIBUTE_ARTIFICIAL

/* Make the entity visible to debuggers etc., even with '-fwhole-program'.  */
/* Applies to: functions, variables.  */
#define ATTRIBUTE_EXTERNALLY_VISIBLE _GL_ATTRIBUTE_EXTERNALLY_VISIBLE


/* ========== Attributes that mainly direct compiler optimizations ========== */

/* The function does not throw exceptions.  */
/* Applies to: functions.  */
#define ATTRIBUTE_NOTHROW _GL_ATTRIBUTE_NOTHROW

/* Do not inline the function.  */
/* Applies to: functions.  */
#define ATTRIBUTE_NOINLINE _GL_ATTRIBUTE_NOINLINE

/* Always inline the function, and report an error if the compiler
   cannot inline.  */
/* Applies to: function.  */
#define ATTRIBUTE_ALWAYS_INLINE _GL_ATTRIBUTE_ALWAYS_INLINE

/* It is OK for a compiler to omit duplicate calls with the same arguments.
   This attribute is safe for a function that neither depends on
   nor affects observable state, and always returns exactly once -
   e.g., does not loop forever, and does not call longjmp.
   (This attribute is stricter than ATTRIBUTE_PURE.)  */
/* Applies to: functions.  */
#define ATTRIBUTE_CONST _GL_ATTRIBUTE_CONST

/* It is OK for a compiler to omit duplicate calls with the same
   arguments if observable state is not changed between calls.
   This attribute is safe for a function that does not affect
   observable state, and always returns exactly once.
   (This attribute is looser than ATTRIBUTE_CONST.)  */
/* Applies to: functions.  */
#define ATTRIBUTE_PURE _GL_ATTRIBUTE_PURE

/* The function is rarely executed.  */
/* Applies to: functions.  */
#define ATTRIBUTE_COLD _GL_ATTRIBUTE_COLD

/* If called from some other compilation unit, the function executes
   code from that unit only by return or by exception handling,
   letting the compiler optimize that unit more aggressively.  */
/* Applies to: functions.  */
#define ATTRIBUTE_LEAF _GL_ATTRIBUTE_LEAF

/* For struct members: The member has the smallest possible alignment.
   For struct, union, class: All members have the smallest possible alignment,
   minimizing the memory required.  */
/* Applies to: struct members, struct, union,
   in C++ also: class.  */
#define ATTRIBUTE_PACKED _GL_ATTRIBUTE_PACKED


/* ================ Attributes that make invalid code valid ================ */

/* Attributes that prevent fatal compiler optimizations for code that is not
   fully ISO C compliant.  */

/* Pointers to the type may point to the same storage as pointers to
   other types, thus disabling strict aliasing optimization.  */
/* Applies to: types.  */
#define ATTRIBUTE_MAY_ALIAS _GL_ATTRIBUTE_MAY_ALIAS


#endif /* _GL_ATTRIBUTE_H */
