import os
import shutil
import stat
import struct
import subprocess
import sys

import container


def main(output_path, entry_path, input_paths, squashfs_path):
    output_tmp_path = output_path + '.tmp'
    shutil.copy2(entry_path, output_tmp_path)
    st = os.stat(output_tmp_path)
    os.chmod(output_tmp_path, st.st_mode | stat.S_IWUSR)

    layer_paths = []
    other_paths = []
    for input_path in input_paths:
        (layer_paths if input_path.endswith('.container_layer') else other_paths).append(input_path)

    if len(other_paths) == 0:
        raise Exception('No program in container dependencies')

    if len(other_paths) > 1:
        raise Exception('Multiple non-layer inputs')

    program_path = other_paths[0]
    program_container_path = os.path.basename(program_path)

    os.symlink(program_container_path, 'entry')
    add_cmd = [ os.path.join(squashfs_path, 'mksquashfs') ]
    add_cmd.extend([program_path, 'entry', 'program_layer'])
    subprocess.run(add_cmd)

    layer_paths.append('program_layer')

    container.join_layers(layer_paths, 'container_data', squashfs_path)

    size = 0
    block_size = 1024 * 1024

    with open(output_tmp_path, 'ab') as output:
        with open('container_data', 'rb') as input_:
            while True:
                data = input_.read(block_size)
                output.write(data)
                size += len(data)

                if len(data) < block_size:
                    break

        with open(os.path.join(squashfs_path, 'unsquashfs'), 'rb') as input_:
            while True:
                data = input_.read(block_size)
                output.write(data)
                size += len(data)

                if len(data) == 0:
                    break


        output.write(struct.pack('<Q', size))

    os.rename(output_tmp_path, output_path)


def entry():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-s', '--squashfs-path', required=True)
    parser.add_argument('input', nargs='*')

    args = parser.parse_args()

    def is_container_entry(path):
        return os.path.basename(path) == '_container_entry'

    input_paths = []
    entry_paths = []

    for input_path in args.input:
        (entry_paths if is_container_entry(input_path) else input_paths).append(input_path)

    if len(entry_paths) != 1:
        raise Exception('Could not select container entry from {}'.format(entry_paths))

    return main(args.output, entry_paths[0], input_paths, args.squashfs_path)


if __name__ == '__main__':
    sys.exit(entry())
