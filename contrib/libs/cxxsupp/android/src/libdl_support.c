/*
 * Copyright (C) 2014 The Android Open Source Project
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT
 * OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

// Contains wrapper to missing dladdr in API < 8

#if !__LP64__

#include <dlfcn.h>
#include <android/api-level.h>

#if __ANDROID_API__ < 8
typedef int Dl_info;
#endif // __ANDROID_API__ >= 8

typedef int (*dladdr_func_t)(const void *addr, Dl_info *info);

#if defined(__cplusplus)
extern "C"
#endif
int my_dladdr(const void *addr, Dl_info *info)
{
    static int initialized = 0;
    static dladdr_func_t p_dladdr = 0;
    if (!p_dladdr && !initialized) {
        void *libdl = dlopen("libdl.so", RTLD_NOW);
      // Other thread may enter here simultaneously but p_dladdr should be
      // set to the same addres for "dladdr"
        if (libdl) {
#if __ANDROID_API__ < 8
	   p_dladdr = 0;
#else
	   p_dladdr = (dladdr_func_t)dlsym(libdl, "dladdr");
#endif
        }
        initialized = 1;
    }
    return p_dladdr? p_dladdr(addr, info) : 0;
}


#ifdef TEST
#include <stdio.h>
int main()
{
    void *libdl = dlopen("libc.so", RTLD_NOW);
    void *h = dlsym(libdl, "printf");
    Dl_info info;
    int r = my_dladdr((char*)h+1, &info);
    if (r)
        printf("%p: %s, %x, %s, %p\n", h, info.dli_fname, info.dli_fbase, info.dli_sname, info.dli_saddr);
}

#endif // TEST
#endif // __LP64__
