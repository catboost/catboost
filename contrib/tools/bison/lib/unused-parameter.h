/* A C macro for declaring that specific function parameters are not used.
   Copyright (C) 2008-2020 Free Software Foundation, Inc.

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

/* _GL_UNUSED_PARAMETER is a marker that can be appended to function parameter
   declarations for parameters that are not used.  This helps to reduce
   warnings, such as from GCC -Wunused-parameter.  The syntax is as follows:
       type param _GL_UNUSED_PARAMETER
   or more generally
       param_decl _GL_UNUSED_PARAMETER
   For example:
       int param _GL_UNUSED_PARAMETER
       int *(*param)(void) _GL_UNUSED_PARAMETER
   Other possible, but obscure and discouraged syntaxes:
       int _GL_UNUSED_PARAMETER *(*param)(void)
       _GL_UNUSED_PARAMETER int *(*param)(void)
 */
#ifndef _GL_UNUSED_PARAMETER
# define _GL_UNUSED_PARAMETER _GL_ATTRIBUTE_MAYBE_UNUSED
#endif
