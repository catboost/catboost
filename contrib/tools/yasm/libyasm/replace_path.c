/*
 * Copyright (c) 1988, 1993, 2019
 *      The Regents of the University of California.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 3. Neither the name of the University nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */
#include "util.h"

char * yasm_replace_path(const char* replace_map[], int size, const char* str, int pref_len) {
    int i;
    for (i=0; i<size; i++) {
        const char* pos = strchr(replace_map[i], '=');
        if (!pos) {
            continue;
        }
        int repl_size = pos - replace_map[i];
        if (pref_len < repl_size) {
            continue;
        }
        if (strncmp(replace_map[i], str, repl_size)) {
            continue;
        }
        int subs_size = strlen(replace_map[i]) - (repl_size + 1);
        int size = subs_size + pref_len - repl_size + 1;
        char* out = yasm_xmalloc(size);
        strncpy(out, pos + 1, subs_size);
        strncpy(out + subs_size, str + repl_size, pref_len - repl_size);
        out[size - 1] = '\0';
        return out;
    }
    return yasm__xstrndup(str, pref_len);
}
