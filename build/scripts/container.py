import subprocess
import os
import shutil


class ContainerError(Exception):
    pass


def join_layers(input_paths, output_path, squashfs_path):

    if len(input_paths) == 1:
        shutil.copy2(input_paths[0], output_path)

    else:
        # We cannot use appending here as it doesn't allow replacing files
        for input_path in input_paths:
            unpack_cmd = [ os.path.join(squashfs_path, 'unsquashfs') ]
            unpack_cmd.extend([ '-f', input_path ])
            subprocess.run(unpack_cmd)

        pack_cmd = [ os.path.join(squashfs_path, 'mksquashfs') ]
        pack_cmd.append(os.path.join(os.curdir, 'squashfs-root'))
        pack_cmd.append(output_path)
        subprocess.run(pack_cmd)

        shutil.rmtree(os.path.join(os.curdir, 'squashfs-root'))

    return 0
