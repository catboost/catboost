import _import_wrapper as iw

from _common import resolve_to_ymake_path
from _common import stripext
from _common import tobuilddir
from _common import get, make_tuples

import os


def retarget(unit, path):
    return os.path.join(unit.path(), os.path.basename(path))


class OmniIDL(iw.CustomCommand):
    def __init__(self, path, unit):
        self._path = path
        self._prefix = tobuilddir(stripext(retarget(unit, self._path)))

        if 'market/contrib/omniorb' not in self._path:
            unit.onpeerdir(['market/contrib/omniorb'])

        unit.onaddincl(['GLOBAL', resolve_to_ymake_path(os.path.dirname(self._prefix))])
        unit.onaddincl(['GLOBAL', 'market/contrib/omniorb/include'])

        flags = unit.get('OMNIIDL_FLAGS')

        if not flags:
            flags = '-bcxx -Wba -C. -I /usr/share/idl/yandex'

        custom_flags = unit.get('OMNIIDL_FLAGS_' + os.path.basename(path).replace('.', '_').upper())

        if custom_flags:
            flags += ' ' + custom_flags

        self._flags = ['-I', os.path.dirname(self._path)] + flags.split()

        if '--gen-headers' in self._flags:
            self._flags.remove('--gen-headers')
            self._genh = True
        else:
            self._genh = False

        if '-WbF' in self._flags:
            self._genh = True

    def tools(self):
        return ['market/contrib/omniorb/src/tool/omniidl', 'market/contrib/omniorb/src/tool/omnicpp']

    def descr(self):
        return ('IL', self._path, 'light-green')

    def flags(self):
        return self._flags + [self._genh]

    def input(self):
        return make_tuples([self._path])

    def output(self):
        prefix = self._prefix

        if '-WbF' in self._flags:
            return make_tuples([prefix + 'DynSK.h', prefix + 'SK.h', prefix + '_defs.hh', prefix + '_operators.hh', prefix + '_poa.hh'])

        if self._genh:
            return make_tuples([prefix + 'DynSK.h', prefix + 'SK.h', prefix + '.hh'])

        return make_tuples([prefix + 'DynSK.cc', prefix + 'SK.cc', prefix + '.hh'])

    def run(self, extra_args, omniidl, omnicpp):
        out = get(self.output, 0)

        self.call([omniidl, '-Y', omnicpp] + self._flags + [self._path], cwd=os.path.dirname(out))

        if self._genh:
            self.call(['mv', self._prefix + 'DynSK.cc', self._prefix + 'DynSK.h'])
            self.call(['mv', self._prefix + 'SK.cc', self._prefix + 'SK.h'])


def init():
    iw.addrule('idl', OmniIDL)
