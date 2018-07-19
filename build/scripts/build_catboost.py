import sys
import os
import shutil
import re
import subprocess

def get_value(val):
    dct = val.split('=', 1)
    if len(dct) > 1:
        return dct[1]
    return ''


class BuildCbBase(object):
    def run(self, cbmodel, cbname, cb_cpp_path):

        data_prefix = "CB_External_"
        data = data_prefix + cbname
        datasize = data + "Size"

        cbtype = "const NCatboostCalcer::TCatboostCalcer"
        cbload = "(ReadModel({0}, {1}, EModelType::CatboostBinary))".format(data, datasize)

        cb_cpp_tmp_path = cb_cpp_path + ".tmp"
        cb_cpp_tmp = open(cb_cpp_tmp_path, 'w')

        cb_cpp_tmp.write("#include <kernel/catboost/catboost_calcer.h>\n")

        ro_data_path = os.path.dirname(cb_cpp_path) + "/" + data_prefix + cbname + ".rodata"
        cb_cpp_tmp.write("namespace{\n")
        cb_cpp_tmp.write("    extern \"C\" {\n")
        cb_cpp_tmp.write("        extern const unsigned char {1}{0}[];\n".format(cbname, data_prefix))
        cb_cpp_tmp.write("        extern const ui32 {1}{0}Size;\n".format(cbname, data_prefix))
        cb_cpp_tmp.write("    }\n")
        cb_cpp_tmp.write("}\n")
        archiverCall = subprocess.Popen([self.archiver, "-q", "-p", "-o", ro_data_path, cbmodel], stdout=None, stderr=subprocess.PIPE)
        archiverCall.wait()
        cb_cpp_tmp.write("extern {0} {1};\n".format(cbtype, cbname))
        cb_cpp_tmp.write("{0} {1}{2};".format(cbtype, cbname, cbload))
        cb_cpp_tmp.close()
        shutil.move(cb_cpp_tmp_path, cb_cpp_path)

class BuildCb(BuildCbBase):
    def run(self, argv):
        if len(argv) < 5:
            print >>sys.stderr, "BuildCb.Run(<ARCADIA_ROOT> <archiver> <mninfo> <mnname> <cppOutput> [params...])"
            sys.exit(1)

        self.SrcRoot = argv[0]
        self.archiver = argv[1]
        cbmodel = argv[2]
        cbname = argv[3]
        cb_cpp_path = argv[4]

        super(BuildCb, self).run(cbmodel, cbname, cb_cpp_path)


def build_cb_f(argv):
    build_cb = BuildCb()
    build_cb.run(argv)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print >>sys.stderr, "Usage: build_cb.py <funcName> <args...>"
        sys.exit(1)

    if (sys.argv[2:]):
        globals()[sys.argv[1]](sys.argv[2:])
    else:
        globals()[sys.argv[1]]()
