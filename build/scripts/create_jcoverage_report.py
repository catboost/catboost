import argparse
import tarfile
import zipfile
import os
import sys
import time
import subprocess


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError:
        pass


class Timer(object):

    def __init__(self):
        self.start = time.time()

    def step(self, msg):
        sys.stderr.write("{} ({}s)\n".format(msg, int(time.time() - self.start)))
        self.start = time.time()


def main(source, output, java, prefix_filter, exclude_filter, jars_list, output_format, tar_output, agent_disposition, runners_paths):
    timer = Timer()
    reports_dir = 'jacoco_reports_dir'
    mkdir_p(reports_dir)
    with tarfile.open(source) as tf:
        tf.extractall(reports_dir)
    timer.step("Coverage data extracted")
    reports = [os.path.join(reports_dir, fname) for fname in os.listdir(reports_dir)]

    with open(jars_list) as f:
        jars = f.read().strip().split()
    if jars and runners_paths:
        for r in runners_paths:
            try:
                jars.remove(r)
            except ValueError:
                pass

    src_dir = 'sources_dir'
    cls_dir = 'classes_dir'

    mkdir_p(src_dir)
    mkdir_p(cls_dir)

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

                entry.filename = entry.filename.encode('utf-8')
                jf.extract(entry, dest)
    timer.step("Jar files extracted")

    if not agent_disposition:
        print>>sys.stderr, 'Can\'t find jacoco agent. Will not generate html report for java coverage.'

    if tar_output:
        report_dir = 'java.report.temp'
    else:
        report_dir = output
    mkdir_p(report_dir)

    if agent_disposition:
        agent_cmd = [java, '-jar', agent_disposition, src_dir, cls_dir, prefix_filter or '.', exclude_filter or '__no_exclude__', report_dir, output_format]
        agent_cmd += reports
        subprocess.check_call(agent_cmd)
        timer.step("Jacoco finished")

    if tar_output:
        with tarfile.open(output, 'w') as outf:
            outf.add(report_dir, arcname='.')


if __name__ == '__main__':
    if 'LC_ALL' in os.environ:
        if os.environ['LC_ALL'] == 'C':
            os.environ['LC_ALL'] = 'en_GB.UTF-8'

    parser = argparse.ArgumentParser()

    parser.add_argument('--source', action='store')
    parser.add_argument('--output', action='store')
    parser.add_argument('--java', action='store')
    parser.add_argument('--prefix-filter', action='store')
    parser.add_argument('--exclude-filter', action='store')
    parser.add_argument('--jars-list', action='store')
    parser.add_argument('--output-format', action='store', default="html")
    parser.add_argument('--raw-output', dest='tar_output', action='store_false', default=True)
    parser.add_argument('--agent-disposition', action='store')
    parser.add_argument('--runner-path', dest='runners_paths', action='append', default=[])
    args = parser.parse_args()
    main(**vars(args))
