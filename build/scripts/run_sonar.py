import os
import sys
import zipfile
import tarfile
import subprocess as sp
import optparse
import shutil
import xml.etree.ElementTree as et


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
    parser.add_option('--gcov-report-path')
    parser.add_option('--source-root')
    parser.add_option('--java-args', action='append', default=[])
    return parser.parse_args()


def extract_zip_file(zip_file_path, dest_dir):
    with zipfile.ZipFile(zip_file_path) as arch:
        arch.extractall(dest_dir)


def get_source_real_path(source_root, path):
    parts = os.path.normpath(path).split(os.path.sep)
    for i in xrange(len(parts)):
        if os.path.exists(os.path.join(source_root, *parts[i:])):
            return os.path.join(*parts[i:])
    return None


def collect_cpp_sources(report, source_root, destination):
    sources = set()
    with open(report) as f:
        root = et.fromstring(f.read())
    for f in root.findall('.//class[@filename]'):
        real_filename = get_source_real_path(source_root, f.attrib['filename'])
        if real_filename:
            f.attrib['filename'] = real_filename
            sources.add(real_filename)
    with open(report, 'w') as f:
        pref = '''<?xml version="1.0" ?>
<!DOCTYPE coverage
  SYSTEM 'http://cobertura.sourceforge.net/xml/coverage-03.dtd'>\n'''
        f.write(pref + et.tostring(root, encoding='utf-8') + '\n\n')
    for src in sources:
        dst = os.path.join(destination, src)
        src = os.path.join(source_root, src)
        if os.path.isfile(src):
            if not os.path.exists(os.path.dirname(dst)):
                os.makedirs(os.path.dirname(dst))
            os.link(src, dst)


def main(opts, props_args):
    sources_dir = os.path.abspath('src')
    base_props_args = ['-Dsonar.sources=' + sources_dir]
    os.mkdir(sources_dir)
    if opts.sources_jar_path:
        extract_zip_file(opts.sources_jar_path, sources_dir)
    if opts.gcov_report_path:
        collect_cpp_sources(opts.gcov_report_path, opts.source_root, sources_dir)
        base_props_args += ['-Dsonar.projectBaseDir=' + sources_dir, '-Dsonar.cxx.coverage.reportPath=' + opts.gcov_report_path]

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
    java_args = ['-{}'.format(i) for i in opts.java_args] + ['-Djava.net.preferIPv6Addresses=true', '-Djava.net.preferIPv4Addresses=false']

    sonar_cmd = [
        opts.java_binary_path,
    ] + java_args + [
        '-classpath',
        opts.sonar_scanner_jar_path,
    ] + base_props_args + props_args + [opts.sonar_scanner_main_class, '-X']

    p = sp.Popen(sonar_cmd, stdout=sp.PIPE, stderr=sp.STDOUT)
    out, _ = p.communicate()

    sys.stderr.write(out)
    with open(opts.log_path, 'a') as f:
        f.write(out)

    sys.exit(p.returncode)


if __name__ == '__main__':
    opts, args = parse_args()
    props_args = ['-D' + arg for arg in args]
    main(opts, props_args)
