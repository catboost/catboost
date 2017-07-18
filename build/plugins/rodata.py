import os

import _common as common
import _import_wrapper as iw

from yasm import Yasm


class ROData(Yasm):
    def __init__(self, path, unit):
        Yasm.__init__(self, path, unit)

        if unit.enabled('darwin') or (unit.enabled('windows') and unit.enabled('arch_type_32')):
            self._prefix = '_'
        else:
            self._prefix = ''

    def run(self, binary):
        in_file = self.resolve_path(common.get(self.input, 0))
        in_file_no_ext = common.stripext(in_file)
        file_name = os.path.basename(in_file_no_ext)
        tmp_file = self.resolve_path(common.get(self.output, 0) + '.asm')

        with open(tmp_file, 'w') as f:
            f.write('global ' + self._prefix + file_name + '\n')
            f.write('global ' + self._prefix + file_name + 'Size' + '\n')
            f.write('SECTION .rodata\n')
            f.write(self._prefix + file_name + ':\nincbin "' + in_file + '"\n')
            f.write(self._prefix + file_name + 'Size:\ndd ' + str(os.path.getsize(in_file)) + '\n')

        return self.do_run(binary, tmp_file)


class RODataCXX(iw.CustomCommand):
    def __init__(self, path, unit):
        self._path = path
        self._base = os.path.basename(common.stripext(self._path))

    def descr(self):
        return 'RD', self._path, 'light-green'

    def input(self):
        return common.make_tuples([self._path])

    def main_out(self):
        return common.tobuilddir(common.stripext(self._path)) + '.cpp'

    def output(self):
        return common.make_tuples([self.main_out()])

    def run(self, binary):
        with open(self.resolve_path(self.main_out()), 'w') as f:
            f.write('static_assert(sizeof(unsigned int) == 4, "ups, something gone wrong");\n\n')
            f.write('extern "C" {\n')
            f.write('    extern const unsigned char ' + self._base + '[] = {\n')

            cnt = 0

            with open(self.resolve_path(self._path), 'r') as input:
                for ch in input.read():
                    f.write('0x%02x, ' % ord(ch))

                    cnt += 1

                    if cnt % 50 == 1:
                        f.write('\n')

            f.write('    };\n')
            f.write('    extern const unsigned int ' + self._base + 'Size = sizeof(' + self._base + ');\n')
            f.write('}\n')


def ro_data(path, unit):
    if unit.enabled('ARCH_AARCH64') or unit.enabled('ARCH_ARM') or unit.enabled('ARCH_PPC64LE'):
        return RODataCXX(path, unit)

    return ROData(path, unit)


def init():
    iw.addrule('rodata', ro_data)
