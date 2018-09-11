from __future__ import print_function
import argparse
import os
import re
import shutil
import subprocess as sp
import sys
import tempfile


def _execute(cmd, **kwargs):
    print('{}> {}'.format(os.getcwd(), ' '.join(cmd)))
    if kwargs:
        assert 0 == sp.check_call(cmd, **kwargs)
    else:
        assert 0 == sp.check_call(cmd, stdout=open(os.devnull, 'wb'))


def _host_os_eq(target_os):
    return os.name == target_os


class Cwd(object):
    def __init__(self, dir):
        self.target_dir = dir

    def __enter__(self):
        self.save_dir = os.getcwd()
        os.chdir(self.target_dir)

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.chdir(self.save_dir)
        if exc_val:
            raise


class RPackager(object):
    def __init__(self, r_dir, target_os, keep_temp):
        self.target_os = target_os
        self.r_dir = os.path.abspath(r_dir)
        self.version = [l for l in open(self.r_dir + '/DESCRIPTION').readlines() if 'Version:' in l][0].split()[1]
        self.build_dir = tempfile.mkdtemp()
        self.stem = 'catboost-R-{}-{}'.format(self.target_os, self.version)
        self.keep_temp = keep_temp

    def __del__(self):
        if self.keep_temp:
            print('Keep temp directory {}'.format(self.build_dir))
        else:
            import shutil  # gc may have discarded it
            shutil.rmtree(self.build_dir)

    # @return path to the package
    def build(self, store_dir):
        package_dir = os.path.abspath(self.build_dir + '/catboost')
        ya = os.path.abspath(self.r_dir + '/../../ya')

        os.makedirs(package_dir)
        export_with = 'svn'
        for obj in ['DESCRIPTION', 'NAMESPACE', 'README.md', 'R', 'inst', 'man', 'tests']:
            src = os.path.join(self.r_dir, obj)
            dst = os.path.join(package_dir, obj)
            if export_with == 'svn':
                try:
                    _execute([ya, 'svn', 'export', src, dst])
                except sp.CalledProcessError:
                    export_with = 'git'
            if export_with == 'git':
                tmp = tempfile.mkstemp('.tar.gz')[1]
                with Cwd(os.path.dirname(src)):
                    _execute(['git', 'archive', '--output', tmp, 'HEAD', os.path.basename(src)])
                with Cwd(os.path.dirname(dst)):
                    _execute(['tar', '-xvpf', tmp])
                os.unlink(tmp)

        _execute([ya, 'make', '-r', self.r_dir + '/src'] + ([] if _host_os_eq(self.target_os) else ['--target-platform={}'.format(self.target_os)]))
        if self.target_os == 'Windows':
            src = self.r_dir + '/src/libcatboostr.dll'
            dst = package_dir + '/inst/libs/x64/libcatboostr.dll'
        else:
            src = self.r_dir + '/src/libcatboostr.so'
            dst = package_dir + '/inst/libs/libcatboostr.so'
        os.makedirs(os.path.dirname(dst))
        shutil.copy2(src, dst)

        # Create the package
        result = os.path.join(os.path.abspath(store_dir), self.stem + '.tgz')
        if not os.path.exists(os.path.dirname(result)):
            os.makedirs(os.path.dirname(result))
        with Cwd(self.build_dir):
            _execute(['tar', '-cvzf', result, 'catboost'])
        return result

    # @return path to the package
    def build_and_install_with_r(self, store_dir):
        if not _host_os_eq(self.target_os):
            raise ValueError('Cannot run R: host {}, target_os {}'.format(os.uname()[0], self.target_os))
        cmd = ['R', 'CMD', 'INSTALL', self.r_dir, '--build', '--install-tests', '--no-multiarch', '--with-keep.source']
        print('EXECUTING {}'.format(' '.join(cmd)))
        r = sp.Popen(cmd, stderr=sp.PIPE, universal_newlines=True)
        for line in r.stderr.readlines():
            sys.stdout.write(line)
            m = re.match(r"packaged installation of .* as .*(cat.*[z]).*", line)
            if m:
                installation = m.group(1)
        status = r.wait()
        assert status == 0, "Command failed with exit status " + str(status)
        src = os.path.join(os.getcwd(), installation)
        dst = os.path.join(store_dir, installation)
        if not os.path.samefile(src, dst):
            shutil.move(src, dst)
        return dst

    def check_with_r(self):
        if not _host_os_eq(self.target_os):
            raise ValueError('Cannot run R: host {}, target_os {}'.format(os.uname()[0], self.target_os))
        seen_errors = False
        cmd = ['R', 'CMD', 'check', self.r_dir, '--no-manual', '--no-examples', '--no-multiarch']
        print('EXECUTING {}'.format(' '.join(cmd)))
        r = sp.Popen(cmd, stderr=sp.PIPE, universal_newlines=True)
        for line in r.stderr.readlines():
            sys.stdout.write(line)
            m = re.match(r".*ERROR.*", line)
            if m:
                seen_errors = True
        status = r.wait()
        assert status == 0, "Command failed with exit status " + str(status)
        assert not seen_errors, "Command completed with errors"

    def generate_doc_with_r(self):
        if not _host_os_eq(self.target_os):
            raise ValueError('Cannot run R: host {}, target_os {}'.format(os.uname()[0], self.target_os))
        seen_errors = False
        cmd = ['R', '-e', 'devtools::document("{}")'.format(self.r_dir)]
        print('EXECUTING {}'.format(' '.join(cmd)))
        r = sp.Popen(cmd, stderr=sp.PIPE, universal_newlines=True)
        for line in r.stderr.readlines():
            sys.stdout.write(line)
            m = re.match(r".*ERROR.*", line)
            if m:
                seen_errors = True
        status = r.wait()
        assert status == 0, "Command failed with exit status " + str(status)
        assert not seen_errors, "Command completed with errors"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target', metavar='OS', help='Target operating system', choices=['Windows', 'Linux', 'Darwin'], type=str, action='store',
                        default=os.name)
    parser.add_argument('--catboost-r-dir', metavar='PATH', help='Catboost R-package dir', type=str, action='store',
                        default=os.path.dirname(sys.argv[0]))
    parser.add_argument('--store-dir', metavar='PATH', help='Where to put the package', type=str, action='store',
                        default='.')
    parser.add_argument('--generate-doc-with-r', help='Use R to regenerate documentation', action='store_true',
                        default=False)
    parser.add_argument('--check-with-r', help='Use R to check the package before build', action='store_true',
                        default=False)
    parser.add_argument('--build', help='Create the package', action='store_true',
                        default=False)
    parser.add_argument('--build-with-r', help='Use R to build the package', action='store_true',
                        default=False)
    parser.add_argument('--keep-temp', help='Do not remove temporary directory', action='store_true',
                        default=False)
    args = parser.parse_args()

    rpackager = RPackager(args.catboost_r_dir, args.target, args.keep_temp)

    if args.generate_doc_with_r and _host_os_eq(args.target):
        rpackager.generate_doc_with_r()

    if args.check_with_r and _host_os_eq(args.target):
        rpackager.check_with_r()

    if args.build_with_r:
        result = rpackager.build_and_install_with_r(args.store_dir)
        print('Built {}'.format(result))

    if args.build:
        result = rpackager.build(args.store_dir)
        print('Built {}'.format(result))
