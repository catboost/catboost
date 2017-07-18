import sys
import subprocess
import tempfile
import os


class Opts(object):
    def __init__(self, args):
        self.archiver = args[0]
        self.arch_type = args[1]
        self.build_root = args[2]
        self.plugin = args[3]
        self.output = args[4]
        auto_input = args[5:]

        if self.arch_type == 'AR':
            self.create_flags = ['rcs']
            self.modify_flags = ['-M']
        elif self.arch_type == 'LIBTOOL':
            self.create_flags = ['-static', '-o']
            self.modify_flags = []

        need_modify = self.arch_type == 'AR' and any(item.endswith('.a') for item in auto_input)
        if need_modify:
            self.objs = filter(lambda x: x.endswith('.o'), auto_input)
            self.libs = filter(lambda x: x.endswith('.a'), auto_input)
        else:
            self.objs = auto_input
            self.libs = []

        self.plugin_flags = ['--plugin', self.plugin] if self.plugin != 'None' else []


def get_opts(args):
    return Opts(args)


if __name__ == "__main__":
    opts = get_opts(sys.argv[1:])

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

    if not opts.libs:
        cmd = [opts.archiver] + opts.create_flags + opts.plugin_flags + [opts.output] + opts.objs
        stdin = None
        exit_code = call()
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
