import os
import argparse
import re
import sys


def list_all_files(directory, prefix='/', hidden_files=False):
    result = []
    if os.path.exists(directory):
        for i in os.listdir(directory):
            abs_path = os.path.join(directory, i)
            result += list_all_files(os.path.join(directory, abs_path), prefix + i + '/', hidden_files) \
                if os.path.isdir(abs_path) else ([prefix + i] if (hidden_files or not i.startswith('.')) else [])
    return result


def pattern_to_regexp(p):
    return '^' + \
           ('/' if not p.startswith('**') else '') + \
           re.escape(p).replace(
               r'\*\*\/', '[_DIR_]'
           ).replace(
               r'\*', '[_FILE_]'
           ).replace(
               '[_DIR_]', '(.*/)?'
           ).replace(
               '[_FILE_]', '([^/]*)'
           ) + '$'


def resolve_java_srcs(srcdir, include_patterns, exclude_patterns, all_resources, resolve_kotlin=False, resolve_groovy=False):
    result = {'java': [], 'not_java': [], 'kotlin': [], 'groovy': []}
    include_patterns_normal, include_patterns_hidden, exclude_patterns_normal, exclude_patterns_hidden = [], [], [], []
    for vis, hid, patterns in ((include_patterns_normal, include_patterns_hidden, include_patterns), (exclude_patterns_normal, exclude_patterns_hidden, exclude_patterns),):
        for pattern in patterns:
            if (pattern if pattern.find('/') == -1 else pattern.rsplit('/', 1)[1]).startswith('.'):
                hid.append(pattern)
            else:
                vis.append(pattern)
        re_patterns = map(pattern_to_regexp, vis + hid)
        if sys.platform in ('win32', 'darwin'):
            re_patterns = [re.compile(i, re.IGNORECASE) for i in re_patterns]
        else:
            re_patterns = [re.compile(i) for i in re_patterns]
        vis[:], hid[:] = re_patterns[:len(vis)], re_patterns[len(vis):]

    for inc_patterns, exc_patterns, with_hidden_files in (
        (include_patterns_normal, exclude_patterns_normal, False),
        (include_patterns_hidden, exclude_patterns_hidden, True),
    ):
        for f in list_all_files(srcdir, hidden_files=with_hidden_files):
            excluded = False

            for exc_re in exc_patterns:
                if exc_re.match(f):
                    excluded = True
                    break

            if excluded:
                continue

            for inc_re in inc_patterns:
                if inc_re.match(f):
                    s = os.path.normpath(f[1:])
                    if all_resources or not (f.endswith('.java') or f.endswith('.kt') or f.endswith('.groovy')):
                        result['not_java'].append(s)
                    elif f.endswith('.java'):
                        result['java'].append(os.path.join(srcdir, s))
                    elif f.endswith('.kt') and resolve_kotlin:
                        result['kotlin'].append(os.path.join(srcdir, s))
                    elif f.endswith('.groovy') and resolve_groovy:
                        result['groovy'].append(os.path.join(srcdir, s))
                    else:
                        result['not_java'].append(s)
                    break

    return sorted(result['java']), sorted(result['not_java']), sorted(result['kotlin']), sorted(result['groovy'])


def do_it(directory, sources_file, resources_file, kotlin_sources_file, groovy_sources_file, include_patterns, exclude_patterns, resolve_kotlin, resolve_groovy, append, all_resources):
    j, r, k, g = resolve_java_srcs(directory, include_patterns, exclude_patterns, all_resources, resolve_kotlin, resolve_groovy)
    mode = 'a' if append else 'w'
    open(sources_file, mode).writelines(i + '\n' for i in j)
    open(resources_file, mode).writelines(i + '\n' for i in r)
    if kotlin_sources_file:
        open(kotlin_sources_file, mode).writelines(i + '\n' for i in k + j)
    if groovy_sources_file:
        open(groovy_sources_file, mode).writelines(i + '\n' for i in g + j)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True)
    parser.add_argument('-s', '--sources-file', required=True)
    parser.add_argument('-r', '--resources-file', required=True)
    parser.add_argument('-k', '--kotlin-sources-file', default=None)
    parser.add_argument('-g', '--groovy-sources-file', default=None)
    parser.add_argument('--append', action='store_true', default=False)
    parser.add_argument('--all-resources',  action='store_true', default=False)
    parser.add_argument('--resolve-kotlin',  action='store_true', default=False)
    parser.add_argument('--resolve-groovy', action='store_true', default=False)
    parser.add_argument('--include-patterns', nargs='*', default=[])
    parser.add_argument('--exclude-patterns', nargs='*', default=[])
    args = parser.parse_args()

    do_it(**vars(args))
