import sys
import tarfile
import copy
import os
import uuid


def main(args):
    output_file, args = args[0], args[1:]
    # heretic@: Splits files on which could be merged( files ) and which should not be merged( expendables )
    # expendables will be in output_file in form {name}{ordinal number of archive in args[]}.{extension}
    try:
        split_i = args.index('-no-merge')
    except ValueError:
        split_i = len(args)
    files, expendables = args[:split_i], args[split_i + 1:]

    with tarfile.open(output_file, 'w') as outf:
        for x in files:
            with tarfile.open(x) as tf:
                for tarinfo in tf:
                    new_tarinfo = copy.deepcopy(tarinfo)
                    if new_tarinfo.name in expendables:
                        dirname, basename = os.path.split(new_tarinfo.name)
                        basename_parts = basename.split('.', 1)
                        new_basename = '.'.join([basename_parts[0] + str(uuid.uuid4())] + basename_parts[1:])
                        new_tarinfo.name = os.path.join(dirname, new_basename)
                    outf.addfile(new_tarinfo, tf.extractfile(tarinfo))


if __name__ == '__main__':
    main(sys.argv[1:])
