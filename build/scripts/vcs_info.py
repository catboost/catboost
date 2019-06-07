import sys
import os


def merge_java_mf(out, infile):
    manifest = os.path.join(infile, 'META-INF', 'MANIFEST.MF')
    if not os.path.isfile(manifest):
        cont = 'Manifest-Version: 1.0'
    else:
        with open(manifest, 'r') as f:
            cont = f.read().rstrip()

    with open(out, 'w') as f:
        f.write(cont + '\n')

    append_vca_info_to_java_mf(out)


def append_vca_info_to_java_mf(manifest):
    with open(manifest, 'a') as f:
        f.write('Vcs-Placeholder: 123\n\n')


if __name__ == "__main__":
    merge_java_mf(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.curdir)
