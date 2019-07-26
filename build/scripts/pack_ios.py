import argparse
import os
import shutil
import subprocess
import sys
import tarfile


def just_do_it():
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", required=True, help="executable file")
    parser.add_argument("--target", required=True, help="target archive path")
    parser.add_argument("--temp-dir", required=True, help="temp dir")
    parser.add_argument("peers", nargs='*')
    args = parser.parse_args()
    app_tar = [p for p in args.peers if p.endswith('.ios.interface')]
    if not app_tar:
        print >> sys.stderr, 'No one IOS_INTERFACE module found'
        shutil.copyfile(args.binary, os.path.join(args.temp_dir, 'bin'))
        if os.path.exists(args.target):
            os.remove(args.target)
        with tarfile.open(args.target, 'w') as tf:
            tf.add(os.path.join(args.temp_dir, 'bin'), arcname=os.path.join(os.path.basename(args.binary) + '.app', 'bin'))
        return
    if len(app_tar) > 1:
        app_tar = [p for p in args.peers if not p.endswith('.default.ios.interface')]
    if len(app_tar) > 1:
        print >> sys.stderr, 'Many IOS_INTERFACE modules found, {} will be used'.format(app_tar[-1])
    app_tar = app_tar[-1]
    with tarfile.open(app_tar) as tf:
        tf.extractall(args.temp_dir)
    tar_suffix = '.default.ios.interface' if app_tar.endswith('.default.ios.interface') else '.ios.interface'
    app_unpacked_path = os.path.join(args.temp_dir, os.path.basename(app_tar)[:-len(tar_suffix)] + '.app')
    if not os.path.exists(app_unpacked_path):
        raise Exception('Bad IOS_INTERFACE resource: {}'.format(app_tar))
    shutil.copyfile(args.binary, os.path.join(app_unpacked_path, 'bin'))
    subprocess.check_call(['/usr/bin/codesign', '--force', '--sign', '-', app_unpacked_path])
    if os.path.exists(args.target):
        os.remove(args.target)
    binary_origin_name = os.path.basename(args.binary)
    while os.path.splitext(binary_origin_name)[1]:
        binary_origin_name = os.path.splitext(binary_origin_name)[0]
    with tarfile.open(args.target, 'w') as tf:
        tf.add(app_unpacked_path, arcname=binary_origin_name + '.app', recursive=True)


if __name__ == '__main__':
    just_do_it()
