import argparse
import tarfile
import zipfile
import os
import sys
import subprocess


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


def main(source, output, java, prefix_filter, exclude_filter, jars_list):
    reports_dir = 'jacoco_reports_dir'
    mkdir_p(reports_dir)
    with tarfile.open(source) as tf:
        tf.extractall(reports_dir)
    reports = [os.path.join(reports_dir, fname) for fname in os.listdir(reports_dir)]

    with open(jars_list) as f:
        jars = f.read().strip().split()

    src_dir = 'sources_dir'
    cls_dir = 'classes_dir'

    mkdir_p(src_dir)
    mkdir_p(cls_dir)

    agent_disposition = None
    for jar in jars:
        if jar.endswith('devtools-jacoco-agent.jar'):
            agent_disposition = jar

        with zipfile.ZipFile(jar) as jf:
            for entry in jf.infolist():
                if entry.filename.endswith('.java'):
                    dest = src_dir

                elif entry.filename.endswith('.class'):
                    dest = cls_dir

                else:
                    continue

                jf.extract(entry, dest)

    if not agent_disposition:
        print>>sys.stderr, 'Can\'t find jacoco agent. Will not generate html report for java coverage.'

    report_dir = 'java.report.temp'
    mkdir_p(report_dir)

    if agent_disposition:
        agent_cmd = [java, '-jar', agent_disposition, src_dir, cls_dir, prefix_filter or '.', exclude_filter or '__no_exclude__', report_dir]
        agent_cmd += reports
        subprocess.check_call(agent_cmd)

    with tarfile.open(output, 'w') as outf:
        outf.add(report_dir, arcname='.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source', action='store')
    parser.add_argument('--output', action='store')
    parser.add_argument('--java', action='store')
    parser.add_argument('--prefix-filter', action='store')
    parser.add_argument('--exclude-filter', action='store')
    parser.add_argument('--jars-list', action='store')
    args = parser.parse_args()
    main(**vars(args))
