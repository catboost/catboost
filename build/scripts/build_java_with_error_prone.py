import sys
import os

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


def just_do_it(argv):
    java, error_prone_tool, javac_cmd = argv[0], argv[1], argv[2:]
    if java.endswith('javac') or java.endswith('javac.exe'):
        for f in javac_cmd:
            if f.startswith('-Xep:'):
                ERROR_PRONE_FLAGS.append(f)
        for f in ERROR_PRONE_FLAGS:
            if f in javac_cmd:
                javac_cmd.remove(f)
        os.execv(java, [java] + JAVA10_EXPORTS + ['-processorpath', error_prone_tool, '-XDcompilePolicy=byfile'] + [(' '.join(['-Xplugin:ErrorProne'] + ERROR_PRONE_FLAGS))] + javac_cmd)
    else:
        os.execv(java, [java, '-Xbootclasspath/p:' + error_prone_tool, 'com.google.errorprone.ErrorProneCompiler'] + ERROR_PRONE_FLAGS + javac_cmd)


if __name__ == '__main__':
    just_do_it(sys.argv[1:])
