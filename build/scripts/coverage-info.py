import argparse
import os
import sys
import tarfile
import collections
import subprocess
import re


GCDA_EXT = '.gcda'
GCNO_EXT = '.gcno'


def suffixes(path):
    """
    >>> list(suffixes('/a/b/c'))
    ['c', 'b/c', '/a/b/c']
    >>> list(suffixes('/a/b/c/'))
    ['c', 'b/c', '/a/b/c']
    >>> list(suffixes('/a'))
    ['/a']
    >>> list(suffixes('/a/'))
    ['/a']
    >>> list(suffixes('/'))
    []
    """
    path = os.path.normpath(path)

    def up_dirs(cur_path):
        while os.path.dirname(cur_path) != cur_path:
            cur_path = os.path.dirname(cur_path)
            yield cur_path

    for x in up_dirs(path):
        yield path.replace(x + os.path.sep, '')


def recast(in_file, out_file, probe_path, update_stat):
    PREFIX = 'SF:'

    probed_path = None

    any_payload = False

    with open(in_file, 'r') as input, open(out_file, 'w') as output:
        active = True
        for line in input:
            line = line.rstrip('\n')
            if line.startswith('TN:'):
                output.write(line + '\n')
            elif line.startswith(PREFIX):
                path = line[len(PREFIX):]
                probed_path = probe_path(path)
                if probed_path:
                    output.write(PREFIX + probed_path + '\n')
                active = bool(probed_path)
            else:
                if active:
                    update_stat(probed_path, line)
                    output.write(line + '\n')
                    any_payload = True

    return any_payload


def print_stat(da, fnda, teamcity_stat_output):
    lines_hit = sum(map(bool, da.values()))
    lines_total = len(da.values())
    lines_coverage = 100.0 * lines_hit / lines_total if lines_total else 0

    func_hit = sum(map(bool, fnda.values()))
    func_total = len(fnda.values())
    func_coverage = 100.0 * func_hit / func_total if func_total else 0

    print >>sys.stderr, '[[imp]]Lines[[rst]]     {: >16} {: >16} {: >16.1f}%'.format(lines_hit, lines_total, lines_coverage)
    print >>sys.stderr, '[[imp]]Functions[[rst]] {: >16} {: >16} {: >16.1f}%'.format(func_hit, func_total, func_coverage)

    if teamcity_stat_output:
        with open(teamcity_stat_output, 'w') as tc_file:
            tc_file.write("##teamcity[blockOpened name='Code Coverage Summary']\n")
            tc_file.write("##teamcity[buildStatisticValue key=\'CodeCoverageAbsLTotal\' value='{}']\n".format(lines_total))
            tc_file.write("##teamcity[buildStatisticValue key=\'CodeCoverageAbsLCovered\' value='{}']\n".format(lines_hit))
            tc_file.write("##teamcity[buildStatisticValue key=\'CodeCoverageAbsMTotal\' value='{}']\n".format(func_total))
            tc_file.write("##teamcity[buildStatisticValue key=\'CodeCoverageAbsMCovered\' value='{}']\n".format(func_hit))
            tc_file.write("##teamcity[blockClosed name='Code Coverage Summary']\n")


def chunks(l, n):
    """
    >>> list(chunks(range(10), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    >>> list(chunks(range(10), 5))
    [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    """
    for i in xrange(0, len(l), n):
        yield l[i:i + n]


def combine_info_files(lcov, files, out_file):
    chunk_size = 50
    files = list(set(files))

    for chunk in chunks(files, chunk_size):
        combine_cmd = [lcov]
        if os.path.exists(out_file):
            chunk.append(out_file)
        for trace in chunk:
            assert os.path.exists(trace), "Trace file does not exist: {} (cwd={})".format(trace, os.getcwd())
            combine_cmd += ["-a", os.path.abspath(trace)]
        print >>sys.stderr, '## lcov', ' '.join(combine_cmd[1:])
        out_file_tmp = "combined.tmp"
        with open(out_file_tmp, "w") as stdout:
            subprocess.check_call(combine_cmd, stdout=stdout)
        if os.path.exists(out_file):
            os.remove(out_file)
        os.rename(out_file_tmp, out_file)


def probe_path_global(path, source_root, prefix_filter, exclude_files):
    if path.endswith('_ut.cpp'):
        return None

    for suff in reversed(list(suffixes(path))):
        if (not prefix_filter or suff.startswith(prefix_filter)) and (not exclude_files or not exclude_files.match(suff)):
            full_path = source_root + os.sep + suff
            if os.path.isfile(full_path):
                return full_path

    return None


def update_stat_global(src_file, line, fnda, da):
    if line.startswith("FNDA:"):
        visits, func_name = line[len("FNDA:"):].split(',')
        fnda[src_file + func_name] += int(visits)

    if line.startswith("DA"):
        line_number, visits = line[len("DA:"):].split(',')
        if visits == '=====':
            visits = 0

        da[src_file + line_number] += int(visits)


def gen_info_global(cmd, cov_info, probe_path, update_stat, lcov_args):
    print >>sys.stderr, '## geninfo', ' '.join(cmd)
    subprocess.check_call(cmd)
    if recast(cov_info + '.tmp', cov_info, probe_path, update_stat):
        lcov_args.append(cov_info)


def init_all_coverage_files(gcno_archive, fname2gcno, fname2info, geninfo_executable, gcov_tool, gen_info, prefix_filter, exclude_files):
    with tarfile.open(gcno_archive) as gcno_tf:
        for gcno_item in gcno_tf:
            if gcno_item.isfile() and gcno_item.name.endswith(GCNO_EXT):
                gcno_tf.extract(gcno_item)

                gcno_name = gcno_item.name
                source_fname = gcno_name[:-len(GCNO_EXT)]
                if prefix_filter and not source_fname.startswith(prefix_filter):
                    sys.stderr.write("Skipping {} (doesn't match prefix '{}')\n".format(source_fname, prefix_filter))
                    continue
                if exclude_files and exclude_files.search(source_fname):
                    sys.stderr.write("Skipping {} (matched exclude pattern '{}')\n".format(source_fname, exclude_files.pattern))
                    continue

                fname2gcno[source_fname] = gcno_name

                if os.path.getsize(gcno_name) > 0:
                    coverage_info = source_fname + '.' + str(len(fname2info[source_fname])) + '.info'
                    fname2info[source_fname].append(coverage_info)
                    geninfo_cmd = [
                        geninfo_executable,
                        '--gcov-tool', gcov_tool,
                        '-i', gcno_name,
                        '-o', coverage_info + '.tmp'
                    ]
                    gen_info(geninfo_cmd, coverage_info)


def process_all_coverage_files(gcda_archive, fname2gcno, fname2info, geninfo_executable, gcov_tool, gen_info):
    with tarfile.open(gcda_archive) as gcda_tf:
        for gcda_item in gcda_tf:
            if gcda_item.isfile() and gcda_item.name.endswith(GCDA_EXT):
                gcda_name = gcda_item.name
                source_fname = gcda_name[:-len(GCDA_EXT)]
                for suff in suffixes(source_fname):
                    if suff in fname2gcno:
                        gcda_new_name = suff + GCDA_EXT
                        gcda_item.name = gcda_new_name
                        gcda_tf.extract(gcda_item)
                        if os.path.getsize(gcda_new_name) > 0:
                            coverage_info = suff + '.' + str(len(fname2info[suff])) + '.info'
                            fname2info[suff].append(coverage_info)
                            geninfo_cmd = [
                                geninfo_executable,
                                '--gcov-tool', gcov_tool,
                                gcda_new_name,
                                '-o', coverage_info + '.tmp'
                            ]
                            gen_info(geninfo_cmd, coverage_info)


def main(source_root, output, gcno_archive, gcda_archive, gcov_tool, prefix_filter, exclude_regexp, teamcity_stat_output, coverage_report_path):
    exclude_files = re.compile(exclude_regexp) if exclude_regexp else None

    fname2gcno = {}
    fname2info = collections.defaultdict(list)
    lcov_args = []
    geninfo_executable = os.path.join(source_root, 'devtools', 'lcov', 'geninfo')

    def probe_path(path):
        return probe_path_global(path, source_root, prefix_filter, exclude_files)

    fnda = collections.defaultdict(int)
    da = collections.defaultdict(int)

    def update_stat(src_file, line):
        update_stat_global(src_file, line, da, fnda)

    def gen_info(cmd, cov_info):
        gen_info_global(cmd, cov_info, probe_path, update_stat, lcov_args)

    init_all_coverage_files(gcno_archive, fname2gcno, fname2info, geninfo_executable, gcov_tool, gen_info, prefix_filter, exclude_files)
    process_all_coverage_files(gcda_archive, fname2gcno, fname2info, geninfo_executable, gcov_tool, gen_info)

    if coverage_report_path:
        output_dir = coverage_report_path
    else:
        output_dir = output + '.dir'
    os.makedirs(output_dir)

    teamcity_stat_file = None
    if teamcity_stat_output:
        teamcity_stat_file = os.path.join(output_dir, 'teamcity.out')
    print_stat(da, fnda, teamcity_stat_file)

    if lcov_args:
        output_trace = "combined.info"
        combine_info_files(os.path.join(source_root, 'devtools', 'lcov', 'lcov'), lcov_args, output_trace)
        cmd = [os.path.join(source_root, 'devtools', 'lcov', 'genhtml'), '-p', source_root, '--ignore-errors', 'source', '-o', output_dir, output_trace]
        print >>sys.stderr, '## genhtml', ' '.join(cmd)
        subprocess.check_call(cmd)

    with tarfile.open(output, 'w') as tar:
        tar.add(output_dir, arcname='.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--source-root', action='store')
    parser.add_argument('--output', action='store')
    parser.add_argument('--gcno-archive', action='store')
    parser.add_argument('--gcda-archive', action='store')
    parser.add_argument('--gcov-tool', action='store')
    parser.add_argument('--prefix-filter', action='store')
    parser.add_argument('--exclude-regexp', action='store')
    parser.add_argument('--teamcity-stat-output', action='store_const', const=True)
    parser.add_argument('--coverage-report-path', action='store')

    args = parser.parse_args()
    main(**vars(args))
