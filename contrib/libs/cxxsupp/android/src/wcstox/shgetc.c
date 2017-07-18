#include "shgetc.h"

#include <stdio.h>

void shinit_wcstring(struct fake_file_t *f, const wchar_t* wcs) {
    f->rstart = wcs;
    f->rpos = wcs;
    f->rend = wcs + wcslen(wcs);
    f->extra_eof = 0;
}

int shgetc(struct fake_file_t *f) {
    if (f->rpos >= f->rend) {
        f->extra_eof ++;
        return EOF;
    }
    wchar_t wc = *f->rpos++;
    int ch = (wc < 128) ? (int)wc : '@';
    return ch;
}

void shunget(struct fake_file_t *f) {
    if (f->extra_eof) {
        f->extra_eof--;
    } else if (f->rpos > f->rstart) {
        f->rpos--;
    }
}

void shlim(struct fake_file_t *f, off_t lim) {
    int off = f->rpos - f->rstart;
    if (off > lim)
        f->rpos = f->rstart + lim;

}

off_t shcnt(struct fake_file_t *f) {
    return (off_t)(f->rpos - f->rstart);
}
