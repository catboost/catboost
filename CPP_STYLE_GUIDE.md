# C++ Style Guide

## Background

This guide describes the conventions for formatting C/C++ code. Consistent code formatting (following coding conventions) is essential for collaboration on large projects, because it makes it significantly easier and faster to read someone else's code.

**The point is that you don't write a program for a computer ("But it works!") or for yourself ("Just ask and I'll tell you how it works"), but for all developers who are reading, modifying, and debugging it, and their needs should be respected.**

Please be aware that use of imperative, object-oriented C++ (version C++14) is assumed. This means:
- Don't use plain C. Use the new operator, not malloc. Assume that the new operator can throw an exception and doesn't return "0" for unsuccessful memory allocation.
- Use C++ as an imperative language, not as a functional language. You can use templates, but not for things like compile-time calculations. Use range-based for, not for_each.
- Don't use boost, and limit use of STL (Use optimized counterparts available in /util instead of some of the parts that are poorly implemented in STL, such as std::string -> TString, std::streams -> TInputStream, TOutputStream). We don't encourage reinventing the wheel (we've already invented everything and just check the /util).
- A restricted subset of the new C++14 syntax is in use that works with all supported compilers (see [*C++14 features* below](#c-14-features)) .

Currently, there is a small subset of code in the source tree that doesn't conform to the standards outlined below (partly because some of these code blocks were written about 15+ years ago). Nevertheless, this standard must be followed when writing new code.

### Style correction policy

- The general rule for fixing the style in old code is "one file – one style." In other words, if you are fixing an error and this involves changes to just a few lines, follow the original style in the file. If the fixed code is written in the new style, then you need to change all the code in the entire file for consistency, and in some cases even the entire code for the class (for example, if fields or methods are renamed).
- If the author of the code doesn't agree with the style of the fixes, they can rewrite it, but the author's changes must match the style described in this guide (including the previous point).
- Commits that fix style should be differentiated from commits that change functionality.
- Don't change the style in files that you aren't modifying, and don't change the line formatting if it follows the recommendations in this guide. Don't forget that every line has an author, and this is tracked in the VCS (e.g. svn, git, hg, etc.) log. When you change someone else's formatting, you become the author of that line without reason for it.

### Corrections to this guide

- Like any other document that describes the preferences of a group of people, this set of conventions is the result of lengthy debates and mutual compromise. So before fixing anything in it, please approve the changes with everyone involved. If you don’t know who to go to for approval, please refrain from making changes.
- When putting together this guide, the authors tried to streamline it as much as possible and only specify the issues that negatively affect code readability, since any additional rules do little more than upset the people who have to learn new habits. For this reason, when proposing a correction or addition, always use the following algorithm: a) justify why this issue is an eyesore for the majority of people, and therefore must be specified; b) if you have proven the first point, then suggest exactly how this issue should be specified; C) explain how the proposed specification minimizes the collective re-learning efforts of the entire team.

### Tools

To automatically format C++ files, use the `ya style` command. It is based on clang-format with the correct config (located at devtools/ya/handlers/style/config), which can be used separately if your editor uses clang-format directly.

## Names

A name should reflect the essence of the data, the type, or the action that it names. Only commonly-used abbreviations are allowed in names. Conventional single-letter names (i, j, k) are only allowed for counters and iterators. Structures are also classes, and everything related to classes also applies to structures (unless explicitly stated otherwise).

### Variables

#### Capitalization, prefixes, and underscores

- Local and global variables begin with a lowercase letter.

- Function names begin with an uppercase letter.

- Function pointers, like ordinary variables, begin with a lowercase letter: `auto localFunction = [&]() { ... }`

- Function arguments begin with a lowercase letter.

- Class members begin with an uppercase letter.

- Class methods begin with an uppercase letter.

- Class names and type definitions (typedefs) are preceded by the prefix T, followed by the name of the class beginning with an uppercase letter. The names of virtual interfaces start with 'I'.

- All global constants and defines are fully capitalized.

- Tokens in complex names of variables and functions are differentiated by capitalizing the first letter of the token (without inserting an underscore).  Tokens in fully capitalized names of constants are separated by underscores.

- Using the underscore as the first character of a name is prohibited.

- Hungarian notation is prohibited.

```cpp
class TClass {
public:
    int Size;
    int GetSize() {
    return Size;
    }
};
TClass object;
int GetValue();
void SetValue(int val);
```

*Exception:* The names of functions, classes, and so on that mimic or extend functions of standard libraries (libc, stl, etc.) should follow the library's naming convention. Examples are yvector, fget, autoarray, sprintf, equivalents of the main function. These classes and functions are usually located in /util.

#### Macros

If you do have to use a macro, you need to make sure that it is unique (for example, it should match the hierarchy of directories in the path to this file). If the macro is intended to be used as part of your library's API, then the macro must have the Y_ prefix (for example, see the macros in util/system/compiler.h).

#### Enumerations

Global enums should be named using the same rules as for classes, but with a capital letter E. The members of these enums should be named using all capital letters, just as for global constants, which is what they actually are. Names must have a prefix formed by the first letters of the enum.

```cpp
enum EFetchType {
    FT_SIMPLE,
    FT_RELFORM_DEBUG,
    FT_ATR_DEBUG,
    FT_SELECTED,
    FT_ATR_SELECTED,
    FT_LARGE_YA
};
```

For enum members of a class, follow the same rules as for other members of a class, since this is the same thing as constant members:
```cpp
class TIndicator {
public:
    enum EStatus {
        Created,
        Running,
        Suspended,
        Aborted,
        Failed,
        Finished
    };
    ...
};
```

Unnamed enums are allowed only in class members:
```cpp
class TFile {
    public:
    enum {
        Invalid = -1
    };
    ...
};
```

C++11 enums should be formed in the same way as enum class members, since they have a similar scope.
```cpp
enum class EStatus {
    Created,
    Running,
    Suspended,
    Aborted,
    Failed,
    Finished
};
```

Do not create your own constructions for converting enums to TString and back. Use GENERATE_ENUM_SERIALIZATION.

Instead of the last field with the number of fields in the enum, you can use `GENERATE_ENUM_SERIALIZATION_WITH_HEADER`.

#### Semantics and pragmatics of naming

Try to make sure that the program makes sense in English, meaning it resembles a cohesive and meaningful English text:

- The first token in the function name should reflect the action performed by the function.
- The names of container variables are plural.

For counter variables, do not use the names DocNum, NumDoc, DocsCount, DocsNum, and CountDoc because they are ungrammatical and ambiguous. For the number of elements (such as documents), you can use `NumDocs` or `DocCount`. For a function that explicitly counts this number for a long time, `CountDocs()` is acceptable.

## Formatting

### Tabs

Don't use tabs in a text editor. The reason is because this is the only way to ensure that your program is readable on any device.  Make sure that your text editor has an option to replace the tab character with spaces. For example, in the TextPad editor, select the "Convert new tabs to spaces" option.

### Indents

Our standard indent is 4 spaces.  The indent should be filled with spaces, even if you use the Tab button.

### Block style

For block operators, use the 1TBS style:

```cpp
if (something) { // K&R style
    One();
    Two();
} else {
    Three();
    Four();
}
for (int i = 0; i < N; ++i) { // K&R style
    // do something...
}
```

Multi-line conditions are an exception (if the condition doesn't fit on one line, split it into several), and they are written like this:
```cpp
if (a && b && c &&
    d && e)
{
    Op();
}
```

For functions and methods, you can use either of two styles:
```cpp
Func1(a, b, c)
{
}
```

or
```cpp
Func1(a, b, c) {
}
```

The style of the curly brackets must be consistent within the same file.

**Short blocks**

Single-line bodies of operators and inline functions must begin with a new line. Bodies of operators and functions declared in the same line make debugging difficult.
```cpp
if (something)
    A();
```

The subordinate operator must not be empty. Not allowed:
```cpp
for (int i = 0; i < 100; i++);
```
The reason is this text looks like a typo that wasn't caught at the compilation stage.

**Operators**

Don't use more than one operator per line.

**Blank lines**

We recommended leaving blank lines between separate logical blocks of code. This greatly improves readability.

### Spaces

#### Operator symbols

All operator symbols, with the exception of unary operators and the member access operator for structures, should have a space on both sides:
```cpp
a = b;
x += 3;
z = a / 6;
```

This includes the assignment operator. In other words, write:
```cpp
if (!x.a || ~(b->c - e::d) == 0)
    z = 0;

void F() throw () {
}

struct T {
    void F() const throw () {
    }
};
```

#### Brackets

Do not put a space after a function name, after the opening parenthesis, or before the closing bracket:
```cpp
Func(a, b, c);
```

Do put a space between the operator and the bracket:
```cpp
if ()
for ()
while ()
```

The spaces inside brackets should look like this:
```cpp
Func(a, b, c);
(a + b)
```


Inside a range-based for:
```cpp
for (auto& x : c) {
}
```

Asymmetric spaces are not allowed.

When instantiating templates, use triangular brackets without spaces.
```cpp
vector<vector<int>> matrix;
```

#### End of line

There shouldn't be any spaces at the end of a line. Use the options in your text editor to control this.

Settings in text editors

- TextPad: "Strip trailing spaces from lines when saving".

- Vim
```vim
augroup vimrc
  " Automatically delete trailing DOS-returns and whitespace on file open and
  " write.
  autocmd BufRead,BufWritePre,FileWritePre * silent! %s/[\r \t]\+$//
augroup END
```

- Emacs
```
(add-hook 'c-mode-common-hook
          (lambda () (add-to-list 'write-file-functions 'delete-trailing-whitespace)))
```

## Lambda functions

Single-line lambdas are allowed only in one case: to define the function where it is used. However, the lambda function itself should not violate the other rules of the style guide:
```cpp
Sort(a.begin(), a.end(), [](int x, int y) -> bool {return x < y;}); //OK
```

```cpp
Sort(a.begin(), a.end(), [](int x, int y) -> bool {int z = x - y; return z < 0;}); //not OK - you can't have 2 statements on the same line
```

In all other cases, they should be formatted as follows:
```cpp
auto f = [](int x, int y) -> bool { //K&R style, the same as for for/if/while
    return x < y;
};

// you can also use 'auto&& f' if the lambda function is "heavy"

Sort(a.begin(), a.end(), f);
```

## Variables and classes

### Variable declarations

The preferred format is "one declaration per line." It is allowed to declare multiple variables of the same type on the same line. It is not allowed to mix arrays, pointers, references, and simple types. Do not use line wrapping in the declaration.
```cpp
int    level;                  // preffered
int    size;

int    level, size;            // allowed

int    level,
       size;                   // prohibited: line wrapping

int level, array[16], *pValue; // prohibited: mixed types
```

### Class and structure declarations

- A structure can only contain open members. You don't need to specify `public` for it. If the structure contains anything other than members, a constructor, and a destructor, we recommend that you rename it to a class.

- The scope labels start from the same column where the class declaration begins. Specifying scopes is mandatory, including the first private scope.

- Members and methods can't be in the same section of scopes. They should be separated by re-specifying the scope. There should be a minimal number of scope labels, reduced to the fewest possible by changing the order of the parts of the class declaration.

- Within one scope:
   * Constructors must precede the destructor.
   * A destructor must precede redefined operators.
   * Redefined operators must precede the rest of the methods.

- A public scope with methods must precede `protected` and `private` scopes with methods.

- Class data members should be placed at the beginning or at the end of the class description. Class type descriptions can precede data descriptions.

```cpp
class TClass {
private:
    int Member; // comments about Member
public:
    TClass();
    TClass(const TClass& other);
    ~TClass();
};
```

The word `template` should start a separate line.

### Constructors

Constructors should be formatted as follows:
```cpp
TClass::TClass()
    : FieldA(1)
    , FieldB("value")
    , FieldC(true)
{
    // some more
    // code here
}
```

### Initializing class data

One of the following variations is allowed:
```cpp
struct T {
    int X = 0;
    double Y = 1.0;
};
```

or

```cpp
struct T {
    int X;
    double Y;

    T() //this implementation can also be in a .cpp file 
        : X(0)
        , Y(1.0)
    {
    }
};
```

The reason is that if you mix two types of initialization, it's much easier to forget to initialize some member of the class, since the initialization code is "spread out" (possibly across multiple source files).

## Namespaces

Namespaces should be formatted like classes, except for the name. Namespaces must begin with a capital letter N:
```cpp
namespace NStl {
    namespace NPrivate {
        //the namespace nesting level is restricted to two
    }

    class TVector {
    };
}
```

## C++14 features

### Constexpr

Only constexpr from C++11 is allowed, because constexpr for non-constant methods is not supported by MSVC.

### Variable templates

Variable templates can be used because MSVC 2015 Update 3 or newer is in use and supports that.

### Alternative function syntax
```cpp
auto f() -> decltype() {}
```
or
```cpp
auto f() {}
```

Only use it where it is truly necessary.

### nullptr/0/NULL

Always use nullptr.

### using/typedef

In new code, prefer 'using' (as a more general mechanism), except when this is not possible. There are cases when the combination of 'using' + templates with an unknown number of parameters + function type leads to compilation errors in MSVC:

```cpp
template <class R, class Args...>
struct T {
    using TSignature = R (Args...);
};
```

In this case, you should use typedef.

### override

In derived classes, use override without virtual.
```cpp
class A {
    virtual ui32 f(ui32 k) const {
    }
};

class B: public A {
    ui32 f(ui32 k) const override {
    }
};
```

## Comments

Comments are for explaining the code where they are located. Do not use comments to remove an unnecessary function or block, especially if this is the old version of a function you corrected. Simply delete any unnecessary parts of the code – you can always go to VCS (e.g. svn, git, hg, etc.) to retrieve the deleted section if you suddenly realize how useful it was. The main harm from commenting previous versions of the code, instead of removing them, is that VCS diff won't work correctly.

Comments should be written in English with correct spelling and grammar.

It is useful to explain the purpose of each class member in the class description. MSVC editor displays this line in the tooltip in "smart editing" mode.

Doxygen-style comments are encouraged.

To make it easier to search for your TODO comments in the code, use one of two formats:
```cpp
// Presumably a temporary comment with notes to yourself:
// TODO (username): fix me later
```

```cpp
// Comment with the ticket:
// TODO (ticket_number): fix me later
```

## Files and preprocessor

### File names

Capital letters are not allowed in file names. File extensions for C++: "cpp", "h".

### Preprocessor

Indents in the preprocessor are also 4 spaces. Keep the hash in the first position.
```cpp
#ifdef Z
#   include
#elif Z
#   define
#   if
#       define
#   endif
#else
#   include
#endif
```

With a preprocessing conditional in the middle of the file, we start in the first position.
```cpp
func A() {
    int x;
#ifdef TEST_func_A // ifndef + else = schisophrenia
    x = 0;
#else
    x = 1;
#endif
}
```

### include

The include files should not be interdependent, meaning an include file must be compileable by itself as a separate compilation unit. If the include file contains references to types that are not described in it:

- If this is a standard type, include the minimum standard include file, such as cstddef or cstdio.
- If this is the name of a class, structure, or enumeration, and it is used by a reference or pointer, write a forward declaration directly in the include file.
- In all other cases, include a file with the declaration of the corresponding class.

The using namespace declaration is not allowed inside include files.

Include files should be specified in the order of less general to more general (regardless of whether it's in cpp or another include), so that a more specific file is included before a more general file. This order allows you to once again check the independence of the other included header files. For example, for the library/json/some_program/some_class.cpp file, the order of inclusion is:

- The paired header file. Always in quotation marks.
```cpp
#include "some_class.h"
```

- Files from the local directory. Always in quotation marks.
```cpp
#include "other_class.h"
#include "other_class2.h"
```

- Next, groups of files from the project superdirectories in order of nesting.
```cpp
// library/json
#include <library/json/json_reader.h>

// library
#include <library/string_utils/base64/base64.h>
#include <library/threading/local_executor/local_executor.h>
```

- Everything else except util.
```cpp
#include <contrib/libs/rapidjson/include/rapidjson/reader.h>
```

- util
```cpp
#include <util/folder/dirut.h>
#include <util/system/yassert.h>
```

- C header files.
```cpp
#include <cmath>
#include <cstdio>
#include <ctime>
```

- System header files if the code can't be migrated.
```cpp
#include <Windows.h>
```

Thus, all non-local names (from other directories) are written in angle brackets.
Within each group, alphabetical sorting is preferable.

To include files just once:
```cpp
#pragma once
..
..
```

## Error handling

### Run-time errors

Errors should be handled using exceptions:
```cpp
#include <util/generic/yexception.h>

class TSomeException: public yexception {
....
};

...

if (errorHappened) {
    ythrow TSomeException(numericCode) << "error happened (" << usefulDescription << ")";
}
```

Error handling using return codes like
```cpp
if (Func1() == ERROR1) {
    return MY_ERROR1;
}

if (Func2() == ERROR2) {
    return MY_ERROR2;
}
```
is prohibited everywhere except in specially stipulated cases:

- In C code.
- Processing return codes of C functions (such as libc).
- In places that are particularly critical to performance (each such case is considered separately).

### Invariant verification

To test various kinds of compile-time invariants (for example, sizeof(int) == 4), use static_assert. To test run-time invariants, instead of assert(), use the Y_ASSERT() macro, since it is better integrated into Visual Studio.

## Cross-platform wrappers

Calling platform-dependent system functions is allowed only in /util. In order to use specific system primitives, use the cross-platform wrappers from /util. If the necessary wrapper does not exist, you can write one (preferably using OOP) and add it to util (don't forget the code review).

## Exceptions to the general rules

### contrib

The /contrib folder contains libraries and programs from 3rd parties. Obviously, they use their own style of writing code. If there is a need to add something to contrib that isn't there yet, create a ticket for discussion and decision making.

