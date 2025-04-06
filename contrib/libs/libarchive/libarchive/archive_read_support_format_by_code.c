/*-
 * Copyright (c) 2003-2011 Tim Kientzle
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR(S) ``AS IS'' AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
 * OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
 * IN NO EVENT SHALL THE AUTHOR(S) BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
 * NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
 * THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "archive_platform.h"

#ifdef HAVE_ERRNO_H
#include <errno.h>
#endif

#include "archive.h"
#include "archive_private.h"

int
archive_read_support_format_by_code(struct archive *a, int format_code)
{
	archive_check_magic(a, ARCHIVE_READ_MAGIC,
	    ARCHIVE_STATE_NEW, "archive_read_support_format_by_code");

	switch (format_code & ARCHIVE_FORMAT_BASE_MASK) {
	case ARCHIVE_FORMAT_7ZIP:
		return archive_read_support_format_7zip(a);
	case ARCHIVE_FORMAT_AR:
		return archive_read_support_format_ar(a);
	case ARCHIVE_FORMAT_CAB:
		return archive_read_support_format_cab(a);
	case ARCHIVE_FORMAT_CPIO:
		return archive_read_support_format_cpio(a);
	case ARCHIVE_FORMAT_EMPTY:
		return archive_read_support_format_empty(a);
	case ARCHIVE_FORMAT_ISO9660:
		return archive_read_support_format_iso9660(a);
	case ARCHIVE_FORMAT_LHA:
		return archive_read_support_format_lha(a);
	case ARCHIVE_FORMAT_MTREE:
		return archive_read_support_format_mtree(a);
	case ARCHIVE_FORMAT_RAR:
		return archive_read_support_format_rar(a);
	case ARCHIVE_FORMAT_RAR_V5:
		return archive_read_support_format_rar5(a);
	case ARCHIVE_FORMAT_RAW:
		return archive_read_support_format_raw(a);
	case ARCHIVE_FORMAT_TAR:
		return archive_read_support_format_tar(a);
	case ARCHIVE_FORMAT_WARC:
		return archive_read_support_format_warc(a);
	case ARCHIVE_FORMAT_XAR:
		return archive_read_support_format_xar(a);
	case ARCHIVE_FORMAT_ZIP:
		return archive_read_support_format_zip(a);
	}
	archive_set_error(a, ARCHIVE_ERRNO_PROGRAMMER,
	    "Invalid format code specified");
	return (ARCHIVE_FATAL);
}
