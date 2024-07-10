import os
import tarfile


CACHE_DIR_NAME='thinlto_cache_dir'


def add_options(parser):
    parser.add_option('--thinlto-cache')
    parser.add_option('--thinlto-cache-write', action='store_true')

def preprocess(opts, cmd):
    if opts.thinlto_cache:
        cache_dir = os.path.join(opts.build_root, CACHE_DIR_NAME)
        cmd +=['-Wl,--thinlto-cache-dir={}'.format(cache_dir)]
        if opts.thinlto_cache_write:
            os.mkdir(cache_dir)
        else:
            with tarfile.open(opts.thinlto_cache, 'r') as tar:
                tar.extractall(opts.build_root)

def postprocess(opts):
    if opts.thinlto_cache:
        cache_dir = os.path.join(opts.build_root, CACHE_DIR_NAME)
        if opts.thinlto_cache_write:
            with tarfile.open(opts.thinlto_cache, 'w:gz') as tar:
                tar.add(cache_dir, arcname=os.path.basename(cache_dir))
