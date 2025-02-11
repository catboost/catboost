import sys
import subprocess
import tempfile
import os
import shutil


class Opts(object):
    def __init__(self, args):
        self.archiver = args[0]
        self.arch_type = args[1]
        self.llvm_ar_format = args[2]
        self.build_root = args[3]
        self.plugin = args[4]
        self.enable_openssl3 = args[5]
        self.output = args[6]
        auto_input = args[7:]

        self.need_modify = False
        self.extra_args = []

        if self.arch_type.endswith('_AR'):
            if self.arch_type == 'GNU_AR':
                self.create_flags = ['rcs']
                self.modify_flags = ['-M']
            elif self.arch_type == 'LLVM_AR':
                self.create_flags = ['rcs', '--format=%s' % self.llvm_ar_format]
                self.modify_flags = ['-M']
            self.need_modify = any(item.endswith('.a') for item in auto_input)
            if self.need_modify:
                self.objs = list(filter(lambda x: x.endswith('.o'), auto_input))
                self.libs = list(filter(lambda x: x.endswith('.a'), auto_input))
            else:
                self.objs = auto_input
                self.libs = []
            self.output_opts = [self.output]
        elif self.arch_type == 'LIBTOOL':
            self.create_flags = ['-static']
            self.objs = auto_input
            self.libs = []
            self.output_opts = ['-o', self.output]
        elif self.arch_type == 'LIB':
            self.create_flags = []
            self.extra_args = list(filter(lambda x: x.startswith('/'), auto_input))
            self.objs = list(filter(lambda x: not x.startswith('/'), auto_input))
            self.libs = []
            self.output_opts = ['/OUT:' + self.output]

        self.plugin_flags = ['--plugin', self.plugin] if self.plugin != 'None' else []


def get_opts(args):
    return Opts(args)


def run(*args):
    return subprocess.check_output(list(args), shell=False).strip()


def gen_renames_1(d):
    for l in d.split('\n'):
        l = l.strip()

        if ' ' in l:
            yield l.split(' ')[-1]


def have_prefix(l, p):
    for x in l:
        if not x.startswith(p):
            return False

    return True


def gen_renames_2(p, d):
    l = list(gen_renames_1(d))
    a = have_prefix(l, '_')

    for s in l:
        if 'asan_globals' in s:
            continue

        if s in ['HMAC', 'SHA1', 'SHA256', 'SHA256', 'SHA512', 'RC4', 'MD5', 'SHA384']:
            continue

        if a and s[1:] in ['HMAC', 'SHA1', 'SHA256', 'SHA256', 'SHA512', 'RC4', 'MD5', 'SHA384']:
            continue

        if a:
            yield s + ' _' + p + s[1:]
        else:
            yield s + ' ' + p + s


def gen_renames(p, d):
    return '\n'.join(gen_renames_2(p, d)).strip() + '\n'


def rename_syms(where, ret):
    p = 'v1_'

    # find symbols to rename
    syms = run(where + 'llvm-nm', '--extern-only', '--defined-only', '-A', ret)

    # prepare rename plan
    renames = gen_renames(p, syms)
    tmp = ret + '.syms'

    with open(tmp, 'w') as f:
        f.write(renames)

    # rename symbols
    run(where + 'llvm-objcopy', '--redefine-syms=' + tmp, ret)
    os.unlink(tmp)


if __name__ == "__main__":
    opts = get_opts(sys.argv[1:])

    # There is a bug in llvm-ar. Some files with size slightly greater 2^32
    # still have GNU format instead of GNU64 and cause link problems.
    # Workaround just lowers llvm-ar's GNU64 threshold to 2^31.
    if opts.arch_type == 'LLVM_AR':
        os.environ['SYM64_THRESHOLD'] = '31'

    def call():
        try:
            p = subprocess.Popen(cmd, stdin=stdin, cwd=opts.build_root)
            rc = p.wait()
            return rc
        except OSError as e:
            raise Exception('while running %s: %s' % (' '.join(cmd), e))

    try:
        os.unlink(opts.output)
    except OSError:
        pass

    if not opts.need_modify:
        cmd = [opts.archiver] + opts.create_flags + opts.plugin_flags + opts.extra_args + opts.output_opts + opts.objs
        stdin = None
        exit_code = call()
    elif len(opts.objs) == 0 and len(opts.libs) == 1:
        shutil.copy(opts.libs[0], opts.output)
        exit_code = 0
    else:
        temp = tempfile.NamedTemporaryFile(dir=os.path.dirname(opts.output), delete=False)

        with open(temp.name, 'w') as tmp:
            tmp.write('CREATE {0}\n'.format(opts.output))
            for lib in opts.libs:
                tmp.write('ADDLIB {0}\n'.format(lib))
            for obj in opts.objs:
                tmp.write('ADDMOD {0}\n'.format(obj))
            tmp.write('SAVE\n')
            tmp.write('END\n')
        cmd = [opts.archiver] + opts.modify_flags + opts.plugin_flags
        stdin = open(temp.name)
        exit_code = call()
        os.remove(temp.name)

    if exit_code != 0:
        raise Exception('{0} returned non-zero exit code {1}. Stop.'.format(' '.join(cmd), exit_code))

    if opts.enable_openssl3 != 'no-openssl3' and os.path.basename(opts.output) in ['libcontrib-libs-openssl.a', 'liblibs-openssl-crypto.a']:
        rename_syms(os.path.dirname(opts.archiver) + '/', opts.output)
