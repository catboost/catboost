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

import time
from libarchive import is_archive, Entry, SeekableArchive
from tarfile import DEFAULT_FORMAT, USTAR_FORMAT, GNU_FORMAT, PAX_FORMAT, ENCODING
from tarfile import REGTYPE, AREGTYPE, LNKTYPE, SYMTYPE, DIRTYPE, FIFOTYPE, CONTTYPE, CHRTYPE, BLKTYPE, GNUTYPE_SPARSE

FORMAT_CONVERSION = {
    USTAR_FORMAT:       'tar',
    GNU_FORMAT:         'gnu',
    PAX_FORMAT:         'pax',
}


def is_tarfile(filename):
    return is_archive(filename, formats=('tar', 'gnu', 'pax'))


def open(**kwargs):
    return TarFile(**kwargs)


class TarInfo(Entry):
    def __init__(self, name):
        super(TarInfo, self).__init__(pathname=name)

    fromtarfile = Entry.from_archive

    def get_name(self):
        return self.pathname

    def set_name(self, value):
        self.pathname = value

    name = property(get_name, set_name)

    @property
    def get_type(self):
        for attr, type in (
                ('isdir', DIRTYPE), ('isfile', REGTYPE), ('issym', SYMTYPE),
                ('isfifo', FIFOTYPE), ('ischr', CHRTYPE), ('isblk', BLKTYPE),
            ):
            if getattr(self, attr)():
                return type

    def _get_missing(self):
        raise NotImplemented()

    def _set_missing(self, value):
        raise NotImplemented()

    pax_headers = property(_get_missing, _set_missing)


class TarFile(SeekableArchive):
    def __init__(self, name=None, mode='r', fileobj=None, format=DEFAULT_FORMAT, tarinfo=TarInfo, encoding=ENCODING):
        if name:
            f = name
        elif fileobj:
            f = fileobj
        try:
            format = FORMAT_CONVERSON.get(format)
        except KeyError:
            raise Exception('Invalid tar format: %s' % format)
        super(TarFile, self).__init__(f, mode=mode, format=format, entry_class=tarinfo, encoding=encoding)

    getmember   = SeekableArchive.getentry
    list        = SeekableArchive.printlist
    extract     = SeekableArchive.readpath
    extractfile = SeekableArchive.readstream

    def getmembers(self):
        return list(self)

    def getnames(self):
        return list(self.iterpaths)

    def next(self):
        pass # TODO: how to do this?

    def extract(self, member, path=None):
        if path is None:
            path = os.getcwd()
        if isinstance(member, basestring):
            f = os.path.join(path, member)
        else:
            f = os.path.join(path, member.pathname)
        return self.readpath(member, f)

    def add(self, name, arcname, recursive=True, exclude=None, filter=None):
        pass # TODO: implement this.

    def addfile(tarinfo, fileobj):
        return self.writepath(fileobj, tarinfo)

    def gettarinfo(name=None, arcname=None, fileobj=None):
        if name:
            f = name
        elif fileobj:
            f = fileobj
        entry = self.entry_class.from_file(f)
        if arcname:
            entry.pathname = arcname
        return entry

    def _get_missing(self):
        raise NotImplemented()

    def _set_missing(self, value):
        raise NotImplemented()

    pax_headers = property(_get_missing, _set_missing)
