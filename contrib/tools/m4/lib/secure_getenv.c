/* Look up an environment variable, returning NULL in insecure situations.

   Copyright 2013-2016 Free Software Foundation, Inc.

   This program is free software: you can redistribute it and/or modify it
   under the terms of the GNU General Public License as published
   by the Free Software Foundation; either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <http://www.gnu.org/licenses/>.  */

#include <config.h>

#include <stdlib.h>

#if !HAVE___SECURE_GETENV
# if HAVE_ISSETUGID || (HAVE_GETUID && HAVE_GETEUID && HAVE_GETGID && HAVE_GETEGID)
#  include <unistd.h>
# endif
#endif

char *
secure_getenv (char const *name)
{
#if HAVE___SECURE_GETENV /* glibc */
  return __secure_getenv (name);
#elif HAVE_ISSETUGID /* OS X, FreeBSD, NetBSD, OpenBSD */
  if (issetugid ())
    return NULL;
  return getenv (name);
#elif HAVE_GETUID && HAVE_GETEUID && HAVE_GETGID && HAVE_GETEGID /* other Unix */
  if (geteuid () != getuid () || getegid () != getgid ())
    return NULL;
  return getenv (name);
#elif (defined _WIN32 || defined __WIN32__) && ! defined __CYGWIN__ /* native Windows */
  /* On native Windows, there is no such concept as setuid or setgid binaries.
     - Programs launched as system services have high privileges, but they don't
       inherit environment variables from a user.
     - Programs launched by a user with "Run as Administrator" have high
       privileges and use the environment variables, but the user has been asked
       whether he agrees.
     - Programs launched by a user without "Run as Administrator" cannot gain
       high privileges, therefore there is no risk. */
  return getenv (name);
#else
  return NULL;
#endif
}
