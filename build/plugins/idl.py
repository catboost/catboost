import _import_wrapper as iw

from _common import stripext
from _common import tobuilddir
from _common import get, make_tuples

import os


def retarget(unit, path):
    return os.path.join(unit.path(), os.path.basename(path))


def repl(s):
    return s.replace('$S/', '${ARCADIA_ROOT}/').replace('$B/', '${ARCADIA_BUILD_ROOT}/')


class OmniIDL(iw.CustomCommand):
    def __init__(self, path, unit):
        self._path = path
        self._prefix = tobuilddir(stripext(retarget(unit, self._path)))

        if 'market/contrib/omniorb' not in self._path:
            unit.onpeerdir(['market/contrib/omniorb'])

        unit.onaddincl(['GLOBAL', repl(os.path.dirname(self._prefix))])
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

    def run(self, omniidl, omnicpp):
        out = get(self.output, 0)

        self.call([omniidl, '-Y', omnicpp] + self._flags + [self._path], cwd=os.path.dirname(out))

        if self._genh:
            self.call(['mv', self._prefix + 'DynSK.cc', self._prefix + 'DynSK.h'])
            self.call(['mv', self._prefix + 'SK.cc', self._prefix + 'SK.h'])


class IDLParser(object):
    def __init__(self, path, unit):
        self._path = path
        self._includes = []
        self._induced = []
        self._c_induced = []

        if '/market/contrib/omniorb' not in self._path:
            self._c_induced.append('$U/' + stripext(retarget(unit, self._path))[3:] + '.hh')
            # HACK
            self._c_induced.append('$S/market/contrib/omniorb/include/omniORB4/CORBA.h')

        with open(path, 'r') as f:
            for line in f:
                line = line.strip()

                if line.startswith('#include'):
                    line = line[8:].strip()
                    ch = line[0]
                    line = line[1:]
                    include = line[:line.find(ch)]

                    if include:
                        include = '$U/' + include

                        if include == '$U/corbaidl.idl':
                            include = '$S/market/contrib/omniorb/idl/corbaidl.idl'
                        elif include == '$U/pollable.idl':
                            include = '$S/market/contrib/omniorb/idl/pollable.idl'
                        elif include == '$U/compression.idl':
                            include = '$S/market/contrib/omniorb/idl/compression.idl'

                        self._includes.append(include)

                        if include.startswith('$U'):
                            self._induced.append(stripext(include) + '.hh')
                        else:
                            f = os.path.basename(include)
                            p = os.path.dirname(include)

                            self._induced.append('$B' + p[2:] + '/omniORB4/' + stripext(f) + '.hh')

    def includes(self):
        return self._includes

    def induced_deps(self):
        def filter_good():
            yield '$S/market/contrib/omniorb/include/omniORB4/CORBA.h'

            if '/market/contrib/omniorb' not in self._path:
                yield '$U/corbaidl_defs.hh'

            for x in self._induced:
                if os.path.basename(x) not in ('corbaidl.hh', 'pollable.hh'):
                    yield x

        ret = {
            'h': list(filter_good()),
            'cpp': self._c_induced
        }

        if iw.engine_version() >= 0:
            return ret

        return ret['h']


def init():
    iw.addparser('idl', IDLParser)
    iw.addrule('idl', OmniIDL)
