# Copyright (c) 2011, SmartFile <btimby@smartfile.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the organization nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import stat
import sys
import math
import time
import logging
import warnings

import contextlib2

from libarchive import _libarchive
import six

logger = logging.getLogger(__name__)

# Suggested block size for libarchive. Libarchive may adjust it.
BLOCK_SIZE = 10240

MTIME_FORMAT = ''

# Default encoding scheme.
ENCODING = 'utf-8'

if six.PY2:
    def encode(value, encoding):
        if type(value) == str:
            value = value.decode(encoding, errors='ignore')
        return value.encode(encoding)
else:
    def encode(value, encoding):
        return value.encode(encoding)


# Functions to initialize read/write for various libarchive supported formats and filters.
FORMATS = {
    None: (_libarchive.archive_read_support_format_all, None),
    'tar': (_libarchive.archive_read_support_format_tar, _libarchive.archive_write_set_format_ustar),
    'pax': (_libarchive.archive_read_support_format_tar, _libarchive.archive_write_set_format_pax),
    'gnu': (_libarchive.archive_read_support_format_gnutar, _libarchive.archive_write_set_format_gnutar),
    'zip': (_libarchive.archive_read_support_format_zip, _libarchive.archive_write_set_format_zip),
    'rar': (_libarchive.archive_read_support_format_rar, None),
    '7zip': (_libarchive.archive_read_support_format_7zip, None),
    'ar': (_libarchive.archive_read_support_format_ar, None),
    'cab': (_libarchive.archive_read_support_format_cab, None),
    'cpio': (_libarchive.archive_read_support_format_cpio, _libarchive.archive_write_set_format_cpio_newc),
    'iso': (_libarchive.archive_read_support_format_iso9660, _libarchive.archive_write_set_format_iso9660),
    'lha': (_libarchive.archive_read_support_format_lha, None),
    'xar': (_libarchive.archive_read_support_format_xar, _libarchive.archive_write_set_format_xar),
}

FILTERS = {
    None: (_libarchive.archive_read_support_filter_all, _libarchive.archive_write_add_filter_none),
    'bzip2': (_libarchive.archive_read_support_filter_bzip2, _libarchive.archive_write_add_filter_bzip2),
    'gzip': (_libarchive.archive_read_support_filter_gzip, _libarchive.archive_write_add_filter_gzip),
    'zstd': (_libarchive.archive_read_support_filter_zstd, _libarchive.archive_write_add_filter_zstd),
}

# Map file extensions to formats and filters. To support quick detection.
FORMAT_EXTENSIONS = {
    '.tar': 'tar',
    '.zip': 'zip',
    '.rar': 'rar',
    '.7z': '7zip',
    '.ar': 'ar',
    '.cab': 'cab',
    '.rpm': 'cpio',
    '.cpio': 'cpio',
    '.iso': 'iso',
    '.lha': 'lha',
    '.xar': 'xar',
}
FILTER_EXTENSIONS = {
    '.bz2': 'bzip2',
    '.gz': 'gzip',
    '.zst': 'zstd',
}


class EOF(Exception):
    '''Raised by ArchiveInfo.from_archive() when unable to read the next
    archive header.'''
    pass


def get_error(archive):
    '''Retrieves the last error description for the given archive instance.'''
    return _libarchive.archive_error_string(archive)


def call_and_check(func, archive, *args):
    '''Executes a libarchive function and raises an exception when appropriate.'''
    ret = func(*args)
    if ret == _libarchive.ARCHIVE_OK:
        return
    elif ret == _libarchive.ARCHIVE_WARN:
        warnings.warn('Warning executing function: %s.' % get_error(archive), RuntimeWarning)
    elif ret == _libarchive.ARCHIVE_EOF:
        raise EOF()
    else:
        raise Exception('Fatal error executing function, message is: %s.' % get_error(archive))


def get_func(name, items, index):
    item = items.get(name, None)
    if item is None:
        return None
    return item[index]


def guess_format(filename):
    filename, ext = os.path.splitext(filename)
    filter = FILTER_EXTENSIONS.get(ext)
    if filter:
        filename, ext = os.path.splitext(filename)
    format = FORMAT_EXTENSIONS.get(ext)
    return format, filter


def is_archive_name(filename, formats=None):
    '''Quick check to see if the given file has an extension indiciating that it is
    an archive. The format parameter can be used to limit what archive format is acceptable.
    If omitted, all supported archive formats will be checked.

    This function will return the name of the most likely archive format, None if the file is
    unlikely to be an archive.'''
    if formats is None:
        formats = FORMAT_EXTENSIONS.values()
    format, filter = guess_format(filename)
    if format in formats:
        return format


def is_archive(f, formats=(None, ), filters=(None, )):
    '''Check to see if the given file is actually an archive. The format parameter
    can be used to specify which archive format is acceptable. If ommitted, all supported
    archive formats will be checked. It opens the file using libarchive. If no error is
    received, the file was successfully detected by the libarchive bidding process.

    This procedure is quite costly, so you should avoid calling it unless you are reasonably
    sure that the given file is an archive. In other words, you may wish to filter large
    numbers of file names using is_archive_name() before double-checking the positives with
    this function.

    This function will return True if the file can be opened as an archive using the given
    format(s)/filter(s).'''
    with contextlib2.ExitStack() as exit_stack:
        if isinstance(f, six.string_types):
            f = exit_stack.enter_context(open(f, 'rb'))
        a = _libarchive.archive_read_new()
        for format in formats:
            format = get_func(format, FORMATS, 0)
            if format is None:
                return False
            format(a)
        for filter in filters:
            filter = get_func(filter, FILTERS, 0)
            if filter is None:
                return False
            filter(a)
        try:
            try:
                call_and_check(_libarchive.archive_read_open_fd, a, a, f.fileno(), BLOCK_SIZE)
                return True
            except:
                return False
        finally:
            _libarchive.archive_read_close(a)
            _libarchive.archive_read_free(a)


def get_archive_filter_names(filename):
    with open(filename, 'rb') as afile:
        a = _libarchive.archive_read_new()
        try:
            format_func = get_func(None, FORMATS, 0)
            format_func(a)
            filter_func = get_func(None, FILTERS, 0)
            filter_func(a)
            if _libarchive.archive_read_open_fd(a, afile.fileno(), BLOCK_SIZE) == _libarchive.ARCHIVE_OK:
                try:
                    nfilter = _libarchive.archive_filter_count(a)
                    return [_libarchive.archive_filter_name(a, i).decode(ENCODING) for i in range(nfilter)]
                finally:
                    _libarchive.archive_read_close(a)
        finally:
            _libarchive.archive_read_free(a)
    return []


class EntryReadStream(object):
    '''A file-like object for reading an entry from the archive.'''
    def __init__(self, archive, size):
        self.archive = archive
        self.closed = False
        self.size = size
        self.bytes = 0

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return

    def __iter__(self):
        if self.closed:
            return
        while True:
            data = self.read(BLOCK_SIZE)
            if not data:
                break
            yield data

    def __len__(self):
        return self.size

    def tell(self):
        return self.bytes

    def read(self, bytes=-1):
        if self.closed:
            return
        if self.bytes == self.size:
            # EOF already reached.
            return
        if bytes < 0:
            bytes = self.size - self.bytes
        elif self.bytes + bytes > self.size:
            # Limit read to remaining bytes
            bytes = self.size - self.bytes
        # Read requested bytes
        data = _libarchive.archive_read_data_into_str(self.archive._a, bytes)
        self.bytes += len(data)
        return data

    def close(self):
        if self.closed:
            return
        # Call archive.close() with _defer True to let it know we have been
        # closed and it is now safe to actually close.
        self.archive.close(_defer=True)
        self.archive = None
        self.closed = True


class EntryWriteStream(object):
    '''A file-like object for writing an entry to an archive.

    If the size is known ahead of time and provided, then the file contents
    are not buffered but flushed directly to the archive. If size is omitted,
    then the file contents are buffered and flushed in the close() method.'''
    def __init__(self, archive, pathname, size=None):
        self.archive = archive
        self.entry = Entry(pathname=pathname, mtime=time.time(), mode=stat.S_IFREG)
        if size is None:
            self.buffer = six.StringIO()
        else:
            self.buffer = None
            self.entry.size = size
            self.entry.to_archive(self.archive)
        self.bytes = 0
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __del__(self):
        self.close()

    def __len__(self):
        return self.bytes

    def tell(self):
        return self.bytes

    def write(self, data):
        if self.closed:
            raise Exception('Cannot write to closed stream.')
        if self.buffer:
            self.buffer.write(data)
        else:
            _libarchive.archive_write_data_from_str(self.archive._a, data)
        self.bytes += len(data)

    def close(self):
        if self.closed:
            return
        if self.buffer:
            self.entry.size = self.buffer.tell()
            self.entry.to_archive(self.archive)
            _libarchive.archive_write_data_from_str(self.archive._a, self.buffer.getvalue())
        _libarchive.archive_write_finish_entry(self.archive._a)

        # Call archive.close() with _defer True to let it know we have been
        # closed and it is now safe to actually close.
        self.archive.close(_defer=True)
        self.archive = None
        self.closed = True


class Entry(object):
    '''An entry within an archive. Represents the header data and it's location within the archive.'''
    def __init__(self, pathname=None, size=None, mtime=None, mode=None, hpos=None, encoding=ENCODING):
        self.pathname = pathname
        self.size = size
        self.mtime = mtime
        self.mode = mode
        self.hpos = hpos
        self.encoding = encoding
        self.linkname = None
        self.id = None
        self.hardlink = None

    @property
    def header_position(self):
        return self.hpos

    @classmethod
    def from_archive(cls, archive, encoding=ENCODING):
        '''Instantiates an Entry class and sets all the properties from an archive header.'''
        e = _libarchive.archive_entry_new()
        try:
            call_and_check(_libarchive.archive_read_next_header2, archive._a, archive._a, e)
            mode = _libarchive.archive_entry_filetype(e)
            mode |= _libarchive.archive_entry_perm(e)
            mtime = _libarchive.archive_entry_mtime(e) + _libarchive.archive_entry_mtime_nsec(e) / 1000000000.0
            # use current time as mtime if stored mtime is equal to 0
            mtime = mtime or time.time()
            entry = cls(
                pathname=_libarchive.archive_entry_pathname(e).decode(encoding),
                size=_libarchive.archive_entry_size(e),
                mtime=mtime,
                mode=mode,
                hpos=archive.header_position,
            )
            # check hardlinkness first to processes hardlinks to the symlinks correctly
            hardlink = _libarchive.archive_entry_hardlink(e)
            if hardlink:
                entry.hardlink = hardlink
            elif entry.issym():
                entry.linkname = _libarchive.archive_entry_symlink(e)
        finally:
            _libarchive.archive_entry_free(e)
        return entry

    @classmethod
    def from_file(cls, f, entry=None, encoding=ENCODING, mtime=None):
        '''Instantiates an Entry class and sets all the properties from a file on the file system.
        f can be a file-like object or a path.'''
        if entry is None:
            entry = cls(encoding=encoding)
        if entry.pathname is None:
            if isinstance(f, six.string_types):
                st = os.lstat(f)
                entry.pathname = f
                entry.size = st.st_size
                entry.mtime = st.st_mtime if mtime is None else mtime
                entry.mode = st.st_mode
                entry.id = cls.get_entry_id(st)
                if entry.issym():
                    entry.linkname = os.readlink(f)
            elif hasattr(f, 'fileno'):
                st = os.fstat(f.fileno())
                entry.pathname = getattr(f, 'name', None)
                entry.size = st.st_size
                entry.mtime = st.st_mtime if mtime is None else mtime
                entry.mode = st.st_mode
                entry.id = cls.get_entry_id(st)
            else:
                entry.pathname = getattr(f, 'pathname', None)
                entry.size = getattr(f, 'size', 0)
                entry.mtime = getattr(f, 'mtime', time.time()) if mtime is None else mtime
                entry.mode = getattr(f, 'mode', stat.S_IFREG)
        return entry

    @staticmethod
    def get_entry_id(st):
        # windows doesn't have such information
        if st.st_ino and st.st_dev:
            return (st.st_dev, st.st_ino)
        return None

    def to_archive(self, archive):
        '''Creates an archive header and writes it to the given archive.'''
        e = _libarchive.archive_entry_new()
        try:
            _libarchive.archive_entry_set_pathname(e, encode(self.pathname, self.encoding))
            _libarchive.archive_entry_set_filetype(e, stat.S_IFMT(self.mode))
            _libarchive.archive_entry_set_perm(e, stat.S_IMODE(self.mode))

            nsec, sec = math.modf(self.mtime)
            nsec *= 1000000000
            _libarchive.archive_entry_set_mtime(e, int(sec), int(nsec))

            if self.ishardlink():
                _libarchive.archive_entry_set_size(e, 0)
                _libarchive.archive_entry_set_hardlink(e, encode(self.hardlink, self.encoding))
            elif self.issym():
                _libarchive.archive_entry_set_size(e, 0)
                _libarchive.archive_entry_set_symlink(e, encode(self.linkname, self.encoding))
            else:
                _libarchive.archive_entry_set_size(e, self.size)
            call_and_check(_libarchive.archive_write_header, archive._a, archive._a, e)
            #self.hpos = archive.header_position
        finally:
            _libarchive.archive_entry_free(e)

    def isdir(self):
        return stat.S_ISDIR(self.mode)

    def isfile(self):
        return stat.S_ISREG(self.mode)

    def issym(self):
        return stat.S_ISLNK(self.mode)

    def isfifo(self):
        return stat.S_ISFIFO(self.mode)

    def ischr(self):
        return stat.S_ISCHR(self.mode)

    def isblk(self):
        return stat.S_ISBLK(self.mode)

    def ishardlink(self):
        return bool(self.hardlink)


class Archive(object):
    '''A low-level archive reader which provides forward-only iteration. Consider
    this a light-weight pythonic libarchive wrapper.'''
    def __init__(self, f, mode='rb', format=None, filter=None, entry_class=Entry, encoding=ENCODING, blocksize=BLOCK_SIZE, filter_opts=None, format_opts=None, fsync=False, fixed_mtime=None):
        if six.PY2:
            assert mode in ('r', 'rb', 'w', 'wb', 'a', 'ab'), 'Mode should be "r[b]", "w[b]" or "a[b]".'
        else:
            assert mode in ('rb', 'wb', 'ab'), 'Mode should be "rb", "wb", or "ab".'
        self._stream = None
        self.encoding = encoding
        self.blocksize = blocksize
        self.file_handle = None
        self.fd = None
        self.filename = None
        self.fsync = fsync
        if isinstance(f, six.string_types):
            self.filename = f
            self.file_handle = open(f, mode)
            self.fd = self.file_handle.fileno()
            # Only close it if we opened it...
            self._defer_close = True
        elif hasattr(f, 'fileno'):
            self.filename = getattr(f, 'name', None)
            self.file_handle = f
            self.fd = self.file_handle.fileno()
            # Leave the fd alone, caller should manage it...
            self._defer_close = False
        elif isinstance(f, int):
            assert f >= 0, f
            self.fd = f
            # Leave the fd alone, caller should manage it...
            self._defer_close = False
        else:
            raise Exception('Provided file is not path or open file.')
        self.mode = mode
        # Guess the format/filter from file name (if not provided)
        if self.filename:
            if format is None:
                format = guess_format(self.filename)[0]
            if filter is None:
                filter = guess_format(self.filename)[1]
        self.format = format
        self.filter = filter
        # The class to use for entries.
        self.entry_class = entry_class
        self.fixed_mtime = fixed_mtime
        # Select filter/format functions.
        if self.mode.startswith('r'):
            self.format_func = get_func(self.format, FORMATS, 0)
            if self.format_func is None:
                raise Exception('Unsupported format %s' % format)
            self.filter_func = get_func(self.filter, FILTERS, 0)
            if self.filter_func is None:
                raise Exception('Unsupported filter %s' % filter)
        else:
            # TODO: how to support appending?
            if self.format is None:
                raise Exception('You must specify a format for writing.')
            self.format_func = get_func(self.format, FORMATS, 1)
            if self.format_func is None:
                raise Exception('Unsupported format %s' % format)
            self.filter_func = get_func(self.filter, FILTERS, 1)
            if self.filter_func is None:
                raise Exception('Unsupported filter %s' % filter)
        # Open the archive, apply filter/format functions.
        self.filter_opts = filter_opts
        self.format_opts = format_opts
        # Stores every added entry's id to handle hardlinks properly
        self.members = {}
        self.init()

    def __iter__(self):
        while True:
            try:
                yield self.entry_class.from_archive(self, encoding=self.encoding)
            except EOF:
                break

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def __del__(self):
        self.close()

    def init(self):
        def _apply_opts(f, opts):
            if opts:
                for opt_name, opt_val in opts.items():
                    call_and_check(f, self._a, self._a, None, encode(opt_name, self.encoding), encode(opt_val, self.encoding))

        if self.mode.startswith('r'):
            self._a = _libarchive.archive_read_new()
        else:
            self._a = _libarchive.archive_write_new()
        self.format_func(self._a)
        self.filter_func(self._a)
        if self.mode.startswith('r'):
            _apply_opts(_libarchive.archive_read_set_format_option, self.format_opts)
            _apply_opts(_libarchive.archive_read_set_filter_option, self.filter_opts)
            call_and_check(_libarchive.archive_read_open_fd, self._a, self._a, self.fd, self.blocksize)
        else:
            _apply_opts(_libarchive.archive_write_set_format_option, self.format_opts)
            _apply_opts(_libarchive.archive_write_set_filter_option, self.filter_opts)
            call_and_check(_libarchive.archive_write_open_fd, self._a, self._a, self.fd)
            # XXX Don't pad the last block to avoid badly formed archive with zstd filter
            call_and_check(_libarchive.archive_write_set_bytes_in_last_block, self._a, self._a, 1)

    def denit(self):
        '''Closes and deallocates the archive reader/writer.'''
        if getattr(self, '_a', None) is None:
            return
        try:
            if self.mode.startswith('r'):
                _libarchive.archive_read_close(self._a)
                _libarchive.archive_read_free(self._a)
            else:
                _libarchive.archive_write_close(self._a)
                _libarchive.archive_write_free(self._a)
        finally:
            # We only want one try at this...
            self._a = None

    def close(self, _defer=False):
        # _defer == True is how a stream can notify Archive that the stream is
        # now closed.  Calling it directly in not recommended.
        if _defer:
            # This call came from our open stream.
            self._stream = None
            if not self._defer_close:
                # We are not yet ready to close.
                return
        if self._stream is not None:
            # We have a stream open! don't close, but remember we were asked to.
            self._defer_close = True
            return
        self.denit()
        # If there is a file attached...
        if getattr(self, 'file_handle', None):
            # Make sure it is not already closed...
            if getattr(self.file_handle, 'closed', False):
                return
            # Flush it if not read-only...
            if not self.file_handle.mode.startswith('r'):
                self.file_handle.flush()
                if self.fsync:
                    os.fsync(self.fd)
            # and then close it, if we opened it...
            if getattr(self, 'close', None):
                self.file_handle.close()

    @property
    def header_position(self):
        '''The position within the file.'''
        return _libarchive.archive_read_header_position(self._a)

    def iterpaths(self):
        for entry in self:
            yield entry.pathname

    def read(self, size):
        '''Read current archive entry contents into string.'''
        return _libarchive.archive_read_data_into_str(self._a, size)

    def readpath(self, f):
        '''Write current archive entry contents to file. f can be a file-like object or
        a path.'''
        with contextlib2.ExitStack() as exit_stack:
            if isinstance(f, six.string_types):
                basedir = os.path.basename(f)
                if not os.path.exists(basedir):
                    os.makedirs(basedir)
                f = exit_stack.enter_context(open(f, 'wb'))
            return _libarchive.archive_read_data_into_fd(self._a, f.fileno())

    def readstream(self, size):
        '''Returns a file-like object for reading current archive entry contents.'''
        self._stream = EntryReadStream(self, size)
        return self._stream

    def write(self, member, data=None):
        '''Writes a string buffer to the archive as the given entry.'''
        if isinstance(member, six.string_types):
            if self.fixed_mtime is None:
                mtime = time.time()
            else:
                mtime = self.fixed_mtime
            # Use default mode
            member = self.entry_class(pathname=member, encoding=self.encoding, mtime=mtime, mode=stat.S_IFREG | 0o755)
        if data:
            member.size = len(data)
        member.to_archive(self)
        if data:
            _libarchive.archive_write_data_from_str(self._a, data)
        _libarchive.archive_write_finish_entry(self._a)

    def writepath(self, f, pathname=None):
        '''Writes a file to the archive. f can be a file-like object or a path. Uses
        write() to do the actual writing.'''
        member = self.entry_class.from_file(f, encoding=self.encoding, mtime=self.fixed_mtime)

        with contextlib2.ExitStack() as exit_stack:
            if isinstance(f, six.string_types):
                if os.path.isfile(f):
                    f = exit_stack.enter_context(open(f, 'rb'))
            if pathname:
                member.pathname = pathname

            # hardlinks and symlink has no data to be written
            if member.id in self.members:
                member.hardlink = self.members[member.id]
                self.write(member)
                return
            elif member.issym():
                self.write(member)
            elif hasattr(f, 'read') and hasattr(f, 'seek') and hasattr(f, 'tell'):
                self.write_from_file_object(member, f)
            elif hasattr(f, 'read'):
                # TODO: optimize this to write directly from f to archive.
                self.write(member, data=f.read())
            else:
                self.write(member)

        if member.id:
            self.members[member.id] = member.pathname

    def write_from_file_object(self, member, fileobj):
        if isinstance(member, six.string_types):
            member = self.entry_class(pathname=member, encoding=self.encoding, mtime=self.fixed_mtime)

        start = fileobj.tell()
        fileobj.seek(0, os.SEEK_END)
        size = fileobj.tell() - start
        fileobj.seek(start, os.SEEK_SET)

        if size:
            member.size = size
        member.to_archive(self)

        while size:
            data = fileobj.read(BLOCK_SIZE)
            if not data:
                break

            size -= len(data)
            if size < 0:
                msg = "File ({}) size has changed. Can't write more data than was declared in the tar header ({}). " \
                      "(probably file was changed during archiving)".format(member.pathname, member.size)
                logger.warning(msg)
                # write rest expected data (size is negative)
                _libarchive.archive_write_data_from_str(self._a, data[:size])
                break

            _libarchive.archive_write_data_from_str(self._a, data)

        _libarchive.archive_write_finish_entry(self._a)

    def writestream(self, pathname, size=None):
        '''Returns a file-like object for writing a new entry.'''
        self._stream = EntryWriteStream(self, pathname, size)
        return self._stream

    def printlist(self, s=sys.stdout):
        for entry in self:
            s.write(entry.size)
            s.write('\t')
            s.write(entry.mtime.strftime(MTIME_FORMAT))
            s.write('\t')
            s.write(entry.pathname)
        s.flush()


class SeekableArchive(Archive):
    '''A class that provides random-access to archive entries. It does this by using one
    or many Archive instances to seek to the correct location. The best performance will
    occur when reading archive entries in the order in which they appear in the archive.
    Reading out of order will cause the archive to be closed and opened each time a
    reverse seek is needed.'''
    def __init__(self, f, **kwargs):
        self._stream = None
        # Convert file to open file. We need this to reopen the archive.
        mode = kwargs.setdefault('mode', 'rb')
        if isinstance(f, six.string_types):
            f = open(f, mode)
        super(SeekableArchive, self).__init__(f, **kwargs)
        self.entries = []
        self.eof = False

    def __iter__(self):
        for entry in self.entries:
            yield entry
        if not self.eof:
            try:
                for entry in super(SeekableArchive, self).__iter__():
                    self.entries.append(entry)
                    yield entry
            except StopIteration:
                self.eof = True

    def reopen(self):
        '''Seeks the underlying fd to 0 position, then opens the archive. If the archive
        is already open, this will effectively re-open it (rewind to the beginning).'''
        self.denit()
        self.file_handle.seek(0)
        self.init()

    def getentry(self, pathname):
        '''Take a name or entry object and returns an entry object.'''
        for entry in self:
            if entry.pathname == pathname:
                return entry
        raise KeyError(pathname)

    def seek(self, entry):
        '''Seeks the archive to the requested entry. Will reopen if necessary.'''
        move = entry.header_position - self.header_position
        if move != 0:
            if move < 0:
                # can't move back, re-open archive:
                self.reopen()
            # move to proper position in stream
            for curr in super(SeekableArchive, self).__iter__():
                if curr.header_position == entry.header_position:
                    break

    def read(self, member):
        '''Return the requested archive entry contents as a string.'''
        entry = self.getentry(member)
        self.seek(entry)
        return super(SeekableArchive, self).read(entry.size)

    def readpath(self, member, f):
        entry = self.getentry(member)
        self.seek(entry)
        return super(SeekableArchive, self).readpath(f)

    def readstream(self, member):
        '''Returns a file-like object for reading requested archive entry contents.'''
        entry = self.getentry(member)
        self.seek(entry)
        self._stream = EntryReadStream(self, entry.size)
        return self._stream
