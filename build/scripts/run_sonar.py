import os
import sys
import zipfile
import tarfile
import subprocess as sp
import optparse
import shutil
import cStringIO


def parse_args():
    parser = optparse.OptionParser()
    parser.add_option(
        '--classes-jar-path',
        dest='classes_jar_paths',
        action='append',
        default=[],
    )
    parser.add_option('--sources-jar-path')
    parser.add_option('--sonar-scanner-jar-path')
    parser.add_option('--sonar-scanner-main-class')
    parser.add_option('--java-coverage-merged-tar')
    parser.add_option('--java-binary-path')
    parser.add_option('--log-path')
    return parser.parse_args()


def extract_zip_file(zip_file_path, dest_dir):
    with zipfile.ZipFile(zip_file_path) as arch:
        arch.extractall(dest_dir)


def main(opts, props_args):
    sources_dir = os.path.abspath('src')
    os.mkdir(sources_dir)
    extract_zip_file(opts.sources_jar_path, sources_dir)

    base_props_args = ['-Dsonar.sources=' + sources_dir]

    if opts.classes_jar_paths:
        classes_dir = os.path.abspath('cls')
        os.mkdir(classes_dir)

        for classes_jar_path in opts.classes_jar_paths:
            extract_zip_file(classes_jar_path, classes_dir)

        base_props_args.append('-Dsonar.java.binaries=' + classes_dir)

    if opts.java_coverage_merged_tar:
        jacoco_report_path = os.path.abspath('jacoco.exec')
        with open(jacoco_report_path, 'w') as dest:
            with tarfile.open(opts.java_coverage_merged_tar) as tar:
                for src in tar:
                    extracted = tar.extractfile(src)
                    if extracted is not None:
                        shutil.copyfileobj(extracted, dest)

        base_props_args += [
            '-Dsonar.core.codeCoveragePlugin=jacoco',
            '-Dsonar.jacoco.reportPath=' + jacoco_report_path
        ]

    sonar_cmd = [
        opts.java_binary_path,
        '-classpath',
        opts.sonar_scanner_jar_path,
    ] + base_props_args + props_args + [opts.sonar_scanner_main_class, '-X']

    p = sp.Popen(sonar_cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
    out, _ = p.communicate()

    sys.stderr.write(out)
    with open(opts.log_path, 'w') as f:
        f.write(out)

    sys.exit(p.returncode)


if __name__ == '__main__':
    opts, args = parse_args()
    props_args = ['-D' + arg for arg in args]
    main(opts, props_args)
