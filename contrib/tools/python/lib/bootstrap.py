import os
import sys


_SOURCE_ROOT = None


def truncate_path(path):
    path = os.path.normpath(path)
    if _SOURCE_ROOT and path.startswith(_SOURCE_ROOT):
        return os.path.relpath(path, _SOURCE_ROOT)
    return os.path.basename(path)


def module_load(name, path = None):
    import imp

    module_name = name
    module_dot = name.find(".")
    if module_dot >= 0:
        module_name = module_name[:module_dot]

    fileobject, pathname, description = imp.find_module(module_name, path and [path])

    if os.path.isdir(pathname):
        pathname = os.path.join(pathname, "")

        if module_dot >= 0:
            return module_load(name[module_dot + 1:], pathname)
        return module_load("__init__", pathname)

    return pathname, fileobject, name == "__init__"


def list_modules():
    import pydoc

    modules = []

    def module_callback(path, name, description):
        modules.append(name)

    module_scanner = pydoc.ModuleScanner()
    module_scanner.run(module_callback)
    return modules


def make_path_directories(path):
    path_directory = os.path.dirname(path)
    if not os.path.exists(path_directory):
        os.makedirs(path_directory)
    return path


def module_blob_compile(module_file, module_path):
    import marshal
    module_code = compile(module_file.read(), truncate_path(module_path), "exec")
    return marshal.dumps(module_code)


def module_blob_write(module_blob, module_blob_name, module_blob_file):
    if EXTERN:
        if _SOURCE_ROOT and _SOURCE_ROOT in module_blob:
            raise Exception('Unable to add module %s because the blob contains the source root %s' % (module_blob_name, _SOURCE_ROOT))
        if os.getcwd() in module_blob:
            raise Exception('Unable to add module %s because the blob contains the build root %s' % (module_blob_name, os.getcwd()))

        module_blob_file.write(module_blob)
        return

    module_blob_file.write(STATIC + "unsigned char %s[] = {" % module_blob_name)
    for module_blob_byte in module_blob:
        module_blob_file.write("%d," % ord(module_blob_byte))
    module_blob_file.write("};\n\n")


def module_table_write(module_blobs, module_table_file):
    if EXTERN:
        off = 0
        module_table_file.write('extern unsigned char %s[];\n\n' % EXTERN)
        for module_name, module_blob_name, module_blob_size in module_blobs:
            if module_blob_size < 0:
                module_blob_size = -module_blob_size
            module_table_file.write("#define %s (%s + %d)\n" % (module_blob_name, EXTERN, off))
            off += module_blob_size

    module_table_file.write("""\n#include "Python.h"\n""")
    module_table_file.write("\nstatic struct _frozen _PyImport_FrozenModules[] = {\n")

    for module_name, module_blob_name, module_blob_size in module_blobs:
        module_table_file.write("""\t{"%s", %s, %d},\n""" % (module_name, module_blob_name, module_blob_size))

    module_table_file.write("\t{0, 0, 0}\n};\n")
    module_table_file.write("\nstruct _frozen *PyImport_FrozenModules = _PyImport_FrozenModules;\n")


def load_module_list(module_list_path):
    return [line.strip() for line in open(module_list_path)]


if len(sys.argv) == 4:
    module_blob_path = make_path_directories(sys.argv[1])
    module_table_path = make_path_directories(sys.argv[2])
    modules_enabled = load_module_list(sys.argv[3])
    mode2 = 'w'
    STATIC = ''
    EXTERN = os.path.basename(module_blob_path)[:-7]
else:
    module_blob_path = make_path_directories(sys.argv[1])
    module_table_path = make_path_directories(sys.argv[1])
    modules_enabled = load_module_list(sys.argv[2])
    mode2 = 'a'
    STATIC = 'static '
    EXTERN = False


#modules = list_modules()

if 'ARCADIA_ROOT_DISTBUILD' in os.environ:
    _SOURCE_ROOT = os.path.normpath(os.environ['ARCADIA_ROOT_DISTBUILD'])

module_blobs = []

with open(module_blob_path, "wb") as module_blob_file:
    for module_name in modules_enabled:
        #if module_name not in modules: continue

        try:
            module_path, module_file, module_is_package = module_load(module_name)
        except ImportError as e:
            print e
            continue

        if module_file and module_path != __file__:
            module_blob_name = "M_" + "__".join(module_name.split("."))

            module_blob = module_blob_compile(module_file, module_path)
            module_blob_write(module_blob, module_blob_name, module_blob_file)
            module_blob_size = len(module_blob)

            if module_is_package:
                module_blob_size = -module_blob_size

            module_blobs.append((module_name, module_blob_name, module_blob_size))


with open(module_table_path, mode2) as module_table_file:
    module_table_write(module_blobs, module_table_file)
