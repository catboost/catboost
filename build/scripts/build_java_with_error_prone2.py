import sys
import os
import re
import subprocess
import platform


ERROR_PRONE_FLAGS = [
    '-Xep:FunctionalInterfaceMethodChanged:WARN',
    '-Xep:ReturnValueIgnored:WARN',
]

JAVA10_EXPORTS = [
    '--add-exports=jdk.compiler/com.sun.tools.javac.api=ALL-UNNAMED',
    '--add-exports=jdk.compiler/com.sun.tools.javac.util=ALL-UNNAMED',
    '--add-exports=jdk.compiler/com.sun.tools.javac.tree=ALL-UNNAMED',
    '--add-exports=jdk.compiler/com.sun.tools.javac.main=ALL-UNNAMED',
    '--add-exports=jdk.compiler/com.sun.tools.javac.code=ALL-UNNAMED',
    '--add-exports=jdk.compiler/com.sun.tools.javac.processing=ALL-UNNAMED',
    '--add-exports=jdk.compiler/com.sun.tools.javac.parser=ALL-UNNAMED',
    '--add-exports=jdk.compiler/com.sun.tools.javac.comp=ALL-UNNAMED'
]


def get_java_version(exe):
    p = subprocess.Popen([exe, '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = p.communicate()
    for line in ((out or '').strip() + (err or '').strip()).split("\n"):
        m = re.match('java version "(.+)"', line)
        if m:
            parts = m.groups()[0].split(".")
            return parts[1] if parts[0] == "1" else parts[0]
        m = re.match('openjdk version "(\d+).*"', line)
        if m:
            parts = m.groups()[0].split(".")
            return parts[0]
    return None


def get_classpath(cmd):
    for i, part in enumerate(cmd):
        if part == '-classpath':
            i += 1
            if i < len(cmd):
                return cmd[i]
            else:
                return None
    return None


def just_do_it(argv):
    java, javac, error_prone_tool, javac_cmd = argv[0], argv[1], argv[2], argv[3:]
    ver = get_java_version(java)
    if not ver:
        raise Exception("Can't determine java version")
    if int(ver) >= 10:
        for f in javac_cmd:
            if f.startswith('-Xep:'):
                ERROR_PRONE_FLAGS.append(f)
        for f in ERROR_PRONE_FLAGS:
            if f in javac_cmd:
                javac_cmd.remove(f)
        if '-processor' in javac_cmd:
            classpath = get_classpath(javac_cmd)
            if classpath:
                error_prone_tool = error_prone_tool + os.pathsep + classpath
        cmd = [javac] + JAVA10_EXPORTS + ['-processorpath', error_prone_tool, '-XDcompilePolicy=byfile'] + [(' '.join(['-Xplugin:ErrorProne'] + ERROR_PRONE_FLAGS))] + javac_cmd
    else:
        cmd = [java, '-Xbootclasspath/p:' + error_prone_tool, 'com.google.errorprone.ErrorProneCompiler'] + ERROR_PRONE_FLAGS + javac_cmd
    if platform.system() == 'Windows':
        sys.exit(subprocess.Popen(cmd).wait())
    else:
        os.execv(cmd[0], cmd)



if __name__ == '__main__':
    just_do_it(sys.argv[1:])
