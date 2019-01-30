#!/usr/bin/env python

import locale
import subprocess
import time
import os
import sys
import re
import socket
import json

indent = "    "


def print_c_header(result):
    result.write("#pragma once\n")


def print_c_footer(result):
    pass


def print_java_header(result):
    result.write('\n'.join([
        "package ru.yandex.library.svnversion;\n",
        "import java.util.HashMap;\n",
        "final class SvnConstants {",
        indent + "private SvnConstants() {}",
        indent + "private static HashMap<String, String> data;",
        indent + "static {",
        indent * 2 + "data = new HashMap<String, String>();",
        ''
    ]))


def print_java_footer(result):
    result.write('\n'.join([
        indent + "}",
        indent + "static String GetProperty(String property, String defaultValue) {",
        indent * 2 + "return data.containsKey(property) ? data.get(property) : defaultValue;",
        indent + "}",
        "}",
        ''
    ]))


def escape_special_symbols(strval):
    retval = ""
    for c in strval:
        if c in ("\\", "\""):
            retval += "\\" + c
        elif ord(c) < ord(' '):
            retval += c.encode("string_escape")
        else:
            retval += c
    return retval


def escape_special_symbols_java(strval):
    return escape_special_symbols(strval).decode('utf-8', 'replace').encode('utf-8')


def escape_line_feed(strval):
    return re.sub(r'\\n', r'\\n"\\\n    "', strval)


def escape_line_feed_java(strval):
    return re.sub(r'\\n', r'\\n"\n' + indent * 3 + r'+ "', strval)


def escaped_define(strkey, strval):
    return "#define " + strkey + " \"" + escape_line_feed(escape_special_symbols(strval)) + "\""


def escaped_constant(strkey, strval):
    return indent * 2 + "data.put(\"" + strkey + "\", \"" + escape_line_feed_java(escape_special_symbols_java(strval)) + "\");"


def system_command_call(command, shell=True):
    if shell and isinstance(command, list):
        command = subprocess.list2cmdline(command)
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=shell)
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            sys.stderr.write('{}\nRunning {} failed with exit code {}\n'.format(stderr, command, process.returncode))
            return None
        return stdout if isinstance(stdout, str) else stdout.decode()
    except OSError as e:
        msg = e.strerror
        errcodes = 'error {}'.format(e.errno)
        if is_windows_host() and isinstance(e, WindowsError):
            errcodes += ', win-error {}'.format(e.winerror)
            try:
                import ctypes
                msg = unicode(ctypes.FormatError(e.winerror), locale.getdefaultlocale()[1]).encode('utf-8')
            except ImportError:
                pass
        sys.stderr.write('System command call {} failed [{}]: {}\n'.format(command, errcodes, msg))
        return None


def get_svn_info_cmd(arc_root, python_cmd=[sys.executable]):
    ya_path = os.path.join(arc_root, 'ya')
    svn_cmd = python_cmd + [ya_path, '-v', '--no-report', 'svn', 'info', '--buffered', '--detect']
    if not is_windows_host():
        svn_cmd = ['LANG=C'] + svn_cmd
    return svn_cmd


def get_svn_field(svn_info, field):
    match = re.search(field + ": (.*)\n", svn_info)
    if match:
        return match.group(1)
    return ''


def get_svn_dict(fpath, arc_root, python_cmd=[sys.executable]):
    svn_info_file = os.path.join(arc_root, '__SVNVERSION__')  # use on the distbuild
    if os.path.exists(svn_info_file) and os.path.isfile(svn_info_file):
        with open(svn_info_file) as fp:
            svn_info = json.load(fp)
            return {
                'rev': str(svn_info.get('revision')),
                'author': str(svn_info.get('author')),
                'lastchg': str(svn_info.get('last_revision')),
                'url': str(svn_info.get('repository')),
                'date': str(svn_info.get('date')),
            }

    svn_info = system_command_call(get_svn_info_cmd(arc_root, python_cmd) + [fpath])
    info = {}
    if svn_info:
        info['rev'] = get_svn_field(svn_info, 'Revision')
        info['author'] = get_svn_field(svn_info, 'Last Changed Author')
        info['lastchg'] = get_svn_field(svn_info, 'Last Changed Rev')
        info['url'] = get_svn_field(svn_info, 'URL')
        info['date'] = get_svn_field(svn_info, 'Last Changed Date')
    return info


def get_svn_scm_data(info):
    scm_data = "Svn info:\n"
    scm_data += indent + "URL: " + info['url'] + "\n"
    scm_data += indent + "Last Changed Rev: " + info['lastchg'] + "\n"
    scm_data += indent + "Last Changed Author: " + info['author'] + "\n"
    scm_data += indent + "Last Changed Date: " + info['date'] + "\n"
    return scm_data


def get_hg_info_cmd(arc_root, python_cmd=[sys.executable]):
    ya_path = os.path.join(arc_root, 'ya')
    hg_cmd = python_cmd + [ya_path, '-v', '--no-report', 'tool', 'hg', '-R', arc_root]
    # suppress user .hgrc settings
    hg_cmd += ['--config', 'alias.log=log', '--config', 'defaults.log=']
    hg_cmd += ['log', '-r', '.']
    return hg_cmd


def get_hg_field(hg_info, field):
    match = re.search(field + ": (.*)\n", hg_info)
    if match:
        return match.group(1).strip()
    return ''


def get_hg_dict(arc_root, python_cmd=[sys.executable]):
    hg_info_file = os.path.join(arc_root, '__SVNVERSION__')  # use on fuse mounted repository
    if os.path.exists(hg_info_file) and os.path.isfile(hg_info_file):
        with open(hg_info_file) as fp:
            hg_info = json.load(fp)
            return {
                'branch': str(hg_info.get('branch')),
                'hash': str(hg_info.get('hash')),
                'author': str(hg_info.get('author')),
                'date': str(hg_info.get('date')),
            }
    info = {}
    hg_info = system_command_call(get_hg_info_cmd(arc_root, python_cmd=python_cmd))
    if hg_info:
        info['branch'] = get_hg_field(hg_info, 'branch')
        info['hash'] = get_hg_field(hg_info, 'changeset')
        info['author'] = get_hg_field(hg_info, 'user')
        info['date'] = get_hg_field(hg_info, 'date')
    return info


def get_hg_scm_data(info):
    scm_data = "Hg info:\n"
    scm_data += indent + "Branch: " + info.get('branch', '') + "\n"
    scm_data += indent + "Last Changed Rev: " + info.get('hash', '') + "\n"
    scm_data += indent + "Last Changed Author: " + info.get('author', '') + "\n"
    scm_data += indent + "Last Changed Date: " + info.get('date', '') + "\n"
    return scm_data


def git_call(fpath, git_arg):
    return system_command_call(["git", "--git-dir", fpath + "/.git"] + git_arg)


def get_git_dict(fpath):
    info = {}
    git_test = git_call(fpath, ["rev-parse", "HEAD"]).strip()
    if not git_test or len(git_test) != 40:
        return info
    info['rev'] = git_test
    info['author'] = git_call(fpath, ["log", "-1", "--format='format:%an <%ae>'", git_test])
    info['summary'] = git_call(fpath, ["log", "-1", "--format='format:%s'", git_test])

    body = git_call(fpath, ["log", "-1", "--grep='^git-svn-id: '", "--format='format:%b'"])
    if body:
        url = re.match("git?-svn?-id: (.*)@", body)
        rev = re.search('@(.*?) ', body)
        if url and rev:
            info['url'] = url.group(1)
            info['lastchg'] = rev.group(1)
    return info


def get_git_scm_data(info):
    scm_data = "Git info:\n"
    scm_data += indent + "Commit: " + info['rev'] + "\n"
    scm_data += indent + "Author: " + info['author'] + "\n"
    scm_data += indent + "Summary: " + info['summary'] + "\n"
    if 'lastchg' in info and 'url' in info:
        scm_data += indent + "git-svn info:\n"
        scm_data += indent + "URL: " + info['url'] + "\n"
        scm_data += indent + "Last Changed Rev: " + info['lastchg'] + "\n"
    return scm_data


def get_arc_info_cmd(arc_root, python_cmd=[sys.executable]):
    ya_path = os.path.join(arc_root, 'ya')
    arc_cmd = python_cmd + [ya_path, '-v', '--no-report', 'tool', 'arc', '--work-tree', arc_root, 'info', '--json']
    return arc_cmd


def get_arc_dict(arc_root, python_cmd=[sys.executable]):
    info = {}
    result = system_command_call(get_arc_info_cmd(arc_root, python_cmd=python_cmd))
    if result:
        arc_info = json.loads(result)
        info['branch'] = arc_info.get('branch', '').encode('utf8')
        info['hash'] = arc_info.get('hash', '').encode('utf8')
        info['author'] = arc_info.get('author', '').encode('utf8')
        info['date'] = arc_info.get('date', '').encode('utf8')
        info['summary'] = arc_info.get('summary', '').encode('utf8')
    return info


def get_arc_scm_data(info):
    scm_data = "Arc info:\n"
    scm_data += indent + "Branch: " + info.get('branch', '') + "\n"
    scm_data += indent + "Commit: " + info.get('hash', '') + "\n"
    scm_data += indent + "Author: " + info.get('author', '') + "\n"
    scm_data += indent + "Summary: " + info.get('summary', '') + "\n"
    return scm_data


def get_local_data(src_dir, data_file):
    local_ymake = ""
    fymake = os.path.join(src_dir, data_file)
    if os.path.exists(fymake):
        local_ymake_file = open(fymake, "r")
        if local_ymake_file:
            local_ymake = indent + data_file + ":\n"
            for line in local_ymake_file:
                local_ymake += indent + indent + line
    return local_ymake


def get_user():
    sys_user = os.environ.get("USER")
    if not sys_user:
        sys_user = os.environ.get("USERNAME")
        if not sys_user:
            sys_user = "Unknown user"
    return sys_user


def get_hostname():
    hostname = socket.gethostname()
    if not hostname:
        hostname = "No host information"
    return hostname


def get_date():
    return time.strftime("%Y-%m-%d %H:%M:%S %Z")


def is_windows_host():
    windows_marker = os.environ.get("ComSpec")
    if not windows_marker:
        return False
    return True


def get_host_info():
    if not is_windows_host():
        host_info = system_command_call("uname -a")[:-1]
    else:
        host_info = system_command_call("VER")  # XXX: check shell from cygwin to call VER this way!
    return indent + indent + host_info if host_info else ""


def get_other_data(src_dir, build_dir, data_file):

    other_data = "Other info:\n"
    other_data += indent + "Build by: " + get_user() + "\n"
    other_data += indent + "Top src dir: " + src_dir + "\n"
    other_data += indent + "Top build dir: " + build_dir + '\n'
    # other_data += indent + "Build date: " + get_date() + "\n"
    other_data += indent + "Hostname: " + get_hostname() + "\n"
    other_data += indent + "Host information: \n" + get_host_info() + "\n"

    other_data += indent + get_local_data(src_dir, data_file)  # to remove later?

    return other_data


def get_scm(svnversion_file):
    try:
        svnversion = json.load(open(svnversion_file))
    except Exception:
        return 'subversion'

    return svnversion.get('repository_vcs', 'subversion')


def is_svn(arc_root):
    sv = os.path.join(arc_root, '__SVNVERSION__')
    return os.path.exists(sv) and get_scm(sv) == 'subversion' or os.path.isdir(os.path.join(arc_root, '.svn'))


def is_hg(arc_root):
    sv = os.path.join(arc_root, '__SVNVERSION__')
    return os.path.exists(sv) and get_scm(sv) == 'mercurial' or os.path.isdir(os.path.join(arc_root, '.hg'))


def is_git(arc_root):
    return os.path.isdir(os.path.join(arc_root, '.git')) and os.path.exists(os.path.join(arc_root, ".git", "config"))


def is_arc(arc_root):
    return os.path.isdir(os.path.join(arc_root, '.arc'))


def main(header, footer, line):
    if len(sys.argv) != 5:
        sys.stderr.write("Usage: svn_version_gen.py <output file> <source root> <build root> <python command>\n")
        sys.exit(1)
    arc_root = sys.argv[2]
    build_root = sys.argv[3]
    local_data_file = "local.ymake"

    python_cmd = sys.argv[4].split()

    if is_svn(arc_root):
        rev_dict = get_svn_dict(arc_root, arc_root, python_cmd=python_cmd)
        if rev_dict:
            rev_dict['vcs'] = 'svn'
            scm_data = get_svn_scm_data(rev_dict)
        else:
            scm_data = "Svn info:\n" + indent + "no svn info\n"
    elif is_git(arc_root):
        rev_dict = get_git_dict(arc_root)
        if rev_dict:
            rev_dict['vcs'] = 'git'
            scm_data = get_git_scm_data(rev_dict)
        else:
            scm_data = "Git info:\n" + indent + "no git info\n"
    elif is_hg(arc_root):
        rev_dict = get_hg_dict(arc_root, python_cmd=python_cmd)
        if rev_dict:
            rev_dict['vcs'] = 'hg'
            scm_data = get_hg_scm_data(rev_dict)
        else:
            scm_data = "Hg info:\n" + indent + "no hg info\n"
    elif is_arc(arc_root):
        rev_dict = get_arc_dict(arc_root, python_cmd=python_cmd)
        if rev_dict:
            rev_dict['vcs'] = 'arc'
            scm_data = get_arc_scm_data(rev_dict)
        else:
            scm_data = "Arc info:\n" + indent + "no arc info\n"
    else:
        rev_dict = get_svn_dict(arc_root, arc_root, python_cmd=python_cmd) or {}
        if rev_dict:
            rev_dict['vcs'] = 'svn'
            scm_data = get_svn_scm_data(rev_dict)
        else:
            scm_data = "Svn info:\n" + indent + "no svn info\n"

    result = open(sys.argv[1], 'w')
    header(result)
    result.write('\n'.join([
        line("PROGRAM_VERSION", scm_data + "\n" + get_other_data(arc_root, build_root, local_data_file)),
        line("SCM_DATA", scm_data),
        line("ARCADIA_SOURCE_PATH", arc_root),
        line("ARCADIA_SOURCE_URL", rev_dict.get('url', '')),
        line("ARCADIA_SOURCE_REVISION", rev_dict.get('rev', '-1')),
        line("ARCADIA_SOURCE_HG_HASH", rev_dict.get('hash', '')),
        line("ARCADIA_SOURCE_LAST_CHANGE", rev_dict.get('lastchg', '')),
        line("ARCADIA_SOURCE_LAST_AUTHOR", rev_dict.get('author', '')),
        line("BUILD_USER", get_user()),
        line("BUILD_HOST", get_hostname()),
        line("VCS", rev_dict.get('vcs', '')),
        line("BRANCH", rev_dict.get('branch', '')),
        ''
    ]))

    if 'url' in rev_dict:
        result.write('\n'.join([
            line("SVN_REVISION", rev_dict.get('rev', '-1')),
            line("SVN_ARCROOT", rev_dict.get('url', '')),
            line("SVN_TIME", rev_dict.get('date', '')),
            ''
        ]))
    result.write(line("BUILD_DATE", get_date()) + '\n')
    footer(result)

if __name__ == "__main__":
    if 'output-java-class' in sys.argv:
        sys.argv.remove('output-java-class')
        main(print_java_header, print_java_footer, escaped_constant)
    else:
        main(print_c_header, print_c_footer, escaped_define)
