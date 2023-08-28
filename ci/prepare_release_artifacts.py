import argparse
import os
import shutil
import tarfile
import tempfile


def get_sys_arch_list():
    return [
        ('darwin', ['universal2']),
        ('linux', ['aarch64', 'x86_64']),
        ('windows', ['x86_64']),
    ]


def prepare_app(src_dir: str, dst_dir: str, release_version: str):
    for system, archs in get_sys_arch_list():
        bin_suff = '.exe' if system == 'windows' else ''
        for arch in archs:
            os.rename(
                os.path.join(src_dir, f'bin_{system}-{arch}', 'catboost' + bin_suff),
                os.path.join(dst_dir, f'catboost-{system}-{arch}-{release_version}{bin_suff}')
            )

    # legacy names for compatibility
    shutil.copy2(
        os.path.join(dst_dir, f'catboost-windows-x86_64-{release_version}.exe'),
        os.path.join(dst_dir, f'catboost-{release_version}.exe')
    )
    shutil.copy2(
        os.path.join(dst_dir, f'catboost-darwin-universal2-{release_version}'),
        os.path.join(dst_dir, f'catboost-darwin-{release_version}')
    )
    shutil.copy2(
        os.path.join(dst_dir, f'catboost-linux-x86_64-{release_version}'),
        os.path.join(dst_dir, f'catboost-linux-{release_version}')
    )

def extract_file_from_tgz(src_tgz_path: str, tar_member: str, dst_path: str, tmp_dir: str):
    with tarfile.open(src_tgz_path, "r:gz") as tar:
        needed_members = [m for m in tar.getmembers() if m.name == tar_member]
        tar.extractall(members=needed_members, path=tmp_dir)
        os.rename(os.path.join(tmp_dir, tar_member), dst_path)

def prepare_R_package(src_dir: str, dst_dir: str, release_version: str, tmp_dir: str):
    for system, archs in get_sys_arch_list():
        for arch in archs:
            dst_tgz = os.path.join(dst_dir, f'catboost-R-{system}-{arch}-{release_version}.tgz')
            os.rename(os.path.join(src_dir, 'R', f'catboost-R-{system}-{arch}.tgz'), dst_tgz)

            # extract so only
            if system == 'windows':
                so_path_in_tgz = 'catboost/inst/libs/x64/libcatboostr.dll'
                dst_suffix = '.dll'
            else:
                so_path_in_tgz = 'catboost/inst/libs/libcatboostr.so'

                # there's a confusion what is the proper suffix for R packages binaries on Darwin (.so or .dylib),
                # but R-package/R/install.R uses .dylib to fetch artifacts
                dst_suffix = '.dylib' if system == 'darwin' else '.so'

            extract_file_from_tgz(
                dst_tgz,
                so_path_in_tgz,
                os.path.join(dst_dir, f'libcatboostr-{system}-{arch}-v{release_version}{dst_suffix}'),
                tmp_dir
            )


    # legacy names for compatibility
    shutil.copy2(
        os.path.join(dst_dir, f'catboost-R-darwin-universal2-{release_version}.tgz'),
        os.path.join(dst_dir, f'catboost-R-Darwin-{release_version}.tgz')
    )
    shutil.copy2(
        os.path.join(dst_dir, f'catboost-R-linux-x86_64-{release_version}.tgz'),
        os.path.join(dst_dir, f'catboost-R-Linux-{release_version}.tgz')
    )
    shutil.copy2(
        os.path.join(dst_dir, f'catboost-R-windows-x86_64-{release_version}.tgz'),
        os.path.join(dst_dir, f'catboost-R-Windows-{release_version}.tgz')
    )

    shutil.copy2(
        os.path.join(dst_dir, f'libcatboostr-darwin-universal2-v{release_version}.dylib'),
        os.path.join(dst_dir, f'libcatboostr-darwin.dylib')
    )
    # copy with different suffix for compatibility
    shutil.copy2(
        os.path.join(dst_dir, f'libcatboostr-darwin.dylib'),
        os.path.join(dst_dir, f'libcatboostr-darwin.so')
    )

    shutil.copy2(
        os.path.join(dst_dir, f'libcatboostr-linux-x86_64-v{release_version}.so'),
        os.path.join(dst_dir, f'libcatboostr-linux.so')
    )
    shutil.copy2(
        os.path.join(dst_dir, f'libcatboostr-windows-x86_64-v{release_version}.dll'),
        os.path.join(dst_dir, f'libcatboostr.dll')
    )

def prepare_catboostmodel_lib(src_dir: str, dst_dir: str, release_version: str):
    for system, archs in get_sys_arch_list():
        lib_prefix = '' if system == 'windows' else 'lib'
        lib_suffixes = {
            'darwin': ['.dylib'],
            'linux': ['.so'],
            'windows': ['.lib', '.dll']
        }[system]

        for arch in archs:
            for lib_suffix in lib_suffixes:
                os.rename(
                    os.path.join(src_dir, f'model_interface_{system}-{arch}', f'{lib_prefix}catboostmodel{lib_suffix}'),
                    os.path.join(dst_dir, f'{lib_prefix}catboostmodel-{system}-{arch}-{release_version}{lib_suffix}')
                )

    # legacy names for compatibility
    for lib_suffix in ['.lib', '.dll']:
        shutil.copy2(
            os.path.join(dst_dir, f'catboostmodel-windows-x86_64-{release_version}{lib_suffix}'),
            os.path.join(dst_dir, f'catboostmodel{lib_suffix}')
        )
    shutil.copy2(
        os.path.join(dst_dir, f'libcatboostmodel-darwin-universal2-{release_version}.dylib'),
        os.path.join(dst_dir, 'libcatboostmodel.dylib')
    )
    shutil.copy2(
        os.path.join(dst_dir, f'libcatboostmodel-linux-x86_64-{release_version}.so'),
        os.path.join(dst_dir, 'libcatboostmodel.so')
    )


def prepare_artifacts(src_dir: str, dst_dir: str, release_version: str):
    os.mkdir(dst_dir)

    with tempfile.TemporaryDirectory() as tmp_dir:
        prepare_app(src_dir, dst_dir, release_version)
        prepare_R_package(src_dir, dst_dir, release_version, tmp_dir)
        prepare_catboostmodel_lib(src_dir, dst_dir, release_version)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src-dir', help='directory with fetched artifacts', default='./')
    parser.add_argument('--dst-dir', help='directory with artifacts ready for upload', default='./upload')
    parser.add_argument('--release-version', help='release version', required=True)
    parsed_args = parser.parse_args()

    prepare_artifacts(**vars(parsed_args))
