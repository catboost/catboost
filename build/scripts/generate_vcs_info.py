# coding: utf-8
import json
import locale
import re
import os
import subprocess
import sys
import time


INDENT = " " * 4


def _get_vcs_dictionary(vcs_type, *arg):
    if vcs_type == 'git':
        return _GitVersion.parse(*arg)
    else:
        raise Exception("Unknown VCS type {}".format(str(vcs_type)))


def _get_user_locale():
    try:
        return [locale.getencoding()]
    except Exception:
        return []


class _GitVersion:
    @classmethod
    def parse(cls, commit_hash, author_info, summary_info, body_info, tag_info, branch_info, depth=None):
        r"""Parses output of
        git rev-parse HEAD
        git log -1 --format='format:%an <%ae>'
        git log -1 --format='format:%s'
        git log -1 --grep='^git-svn-id: ' --format='format:%b' or
        git log -1 --grep='^Revision: r?\d*' --format='format:%b
        git describe --exact-match --tags HEAD
        git describe --exact-match --all HEAD
        and depth as computed by _get_git_depth
        '"""

        info = {}
        info['hash'] = commit_hash
        info['commit_author'] = _SystemInfo._to_text(author_info)
        info['summary'] = _SystemInfo._to_text(summary_info)

        if 'svn_commit_revision' not in info:
            url = re.search("git?-svn?-id: (.*)@(\\d*).*", body_info)
            if url:
                info['svn_url'] = url.group(1)
                info['svn_commit_revision'] = int(url.group(2))

        if 'svn_commit_revision' not in info:
            rev = re.search('Revision: r?(\\d*).*', body_info)
            if rev:
                info['svn_commit_revision'] = int(rev.group(1))

        info['tag'] = tag_info
        info['branch'] = branch_info
        info['scm_text'] = cls._format_scm_data(info)
        info['vcs'] = 'git'

        if depth:
            info['patch_number'] = int(depth)
        return info

    @staticmethod
    def _format_scm_data(info):
        scm_data = "Git info:\n"
        scm_data += INDENT + "Commit: " + info['hash'] + "\n"
        scm_data += INDENT + "Branch: " + info['branch'] + "\n"
        scm_data += INDENT + "Author: " + info['commit_author'] + "\n"
        scm_data += INDENT + "Summary: " + info['summary'] + "\n"
        if 'svn_commit_revision' in info or 'svn_url' in info:
            scm_data += INDENT + "git-svn info:\n"
        if 'svn_url' in info:
            scm_data += INDENT + "URL: " + info['svn_url'] + "\n"
        if 'svn_commit_revision' in info:
            scm_data += INDENT + "Last Changed Rev: " + str(info['svn_commit_revision']) + "\n"
        return scm_data

    @staticmethod
    def external_data(arc_root):
        env = os.environ.copy()
        env['TZ'] = ''

        hash_args = ['rev-parse', 'HEAD']
        author_args = ['log', '-1', '--format=format:%an <%ae>']
        summary_args = ['log', '-1', '--format=format:%s']
        svn_args = ['log', '-1', '--grep=^git-svn-id: ', '--format=format:%b']
        svn_args_alt = ['log', '-1', '--grep=^Revision: r\\?\\d*', '--format=format:%b']
        tag_args = ['describe', '--exact-match', '--tags', 'HEAD']
        branch_args = ['describe', '--exact-match', '--all', 'HEAD']

        # using local 'Popen' wrapper
        commit = _SystemInfo._system_command_call(['git'] + hash_args, env=env, cwd=arc_root).rstrip()
        author = _SystemInfo._system_command_call(['git'] + author_args, env=env, cwd=arc_root)
        commit = _SystemInfo._system_command_call(['git'] + hash_args, env=env, cwd=arc_root).rstrip()
        author = _SystemInfo._system_command_call(['git'] + author_args, env=env, cwd=arc_root)
        summary = _SystemInfo._system_command_call(['git'] + summary_args, env=env, cwd=arc_root)
        svn_id = _SystemInfo._system_command_call(['git'] + svn_args, env=env, cwd=arc_root)
        if not svn_id:
            svn_id = _SystemInfo._system_command_call(['git'] + svn_args_alt, env=env, cwd=arc_root)

        try:
            tag_info = _SystemInfo._system_command_call(['git'] + tag_args, env=env, cwd=arc_root).splitlines()
        except Exception:
            tag_info = [''.encode('utf-8')]

        try:
            branch_info = _SystemInfo._system_command_call(['git'] + branch_args, env=env, cwd=arc_root).splitlines()
        except Exception:
            branch_info = [''.encode('utf-8')]

        depth = str(_GitVersion._get_git_depth(env, arc_root)).encode('utf-8')

        # logger.debug('Git info commit:{}, author:{}, summary:{}, svn_id:{}'.format(commit, author, summary, svn_id))
        return [commit, author, summary, svn_id, tag_info[0], branch_info[0], depth]

    # YT's patch number.
    @staticmethod
    def _get_git_depth(env, arc_root):
        graph = {}
        full_history_args = ["log", "--full-history", "--format=%H %P", "HEAD"]
        history = _SystemInfo._system_command_call(['git'] + full_history_args, env=env, cwd=arc_root).decode('utf-8')

        head = None
        for line in history.splitlines():
            values = line.split()
            if values:
                if head is None:
                    head = values[0]
                graph[values[0]] = values[1:]

        assert head
        cache = {}
        stack = [(head, None, False)]
        while stack:
            commit, child, calculated = stack.pop()
            if commit in cache:
                calculated = True
            if calculated:
                if child is not None:
                    cache[child] = max(cache.get(child, 0), cache[commit] + 1)
            else:
                stack.append((commit, child, True))
                parents = graph[commit]
                if not parents:
                    cache[commit] = 0
                else:
                    for parent in parents:
                        stack.append((parent, commit, False))
        return cache[head]


class _SystemInfo:
    LOCALE_LIST = _get_user_locale() + [sys.getfilesystemencoding(), 'utf-8']

    @classmethod
    def get_locale(cls):
        import codecs

        for i in cls.LOCALE_LIST:
            if not i:
                continue
            try:
                codecs.lookup(i)
                return i
            except LookupError:
                continue

    @staticmethod
    def _to_text(s):
        if isinstance(s, bytes):
            return s.decode(_SystemInfo.get_locale(), errors='replace')
        return s

    @staticmethod
    def get_user():
        sys_user = os.environ.get("USER")
        if not sys_user:
            sys_user = os.environ.get("USERNAME")
        if not sys_user:
            sys_user = os.environ.get("LOGNAME")
        if not sys_user:
            sys_user = "Unknown user"
        return sys_user

    @staticmethod
    def get_date(stamp=None):
        # Format compatible with SVN-xml format.
        return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime(stamp))

    @staticmethod
    def get_timestamp():
        # Unix timestamp.
        return int(time.time())

    @staticmethod
    def get_other_data(src_dir, data_file='local.ymake'):
        other_data = "Other info:\n"
        other_data += INDENT + "Build by: " + _SystemInfo.get_user() + "\n"
        other_data += INDENT + "Top src dir: {}\n".format(src_dir)

        # logger.debug("Other data: %s", other_data)

        return other_data

    @staticmethod
    def _get_host_info(fake_build_info=False):
        if fake_build_info:
            host_info = '*sys localhost 1.0.0 #dummy information '
        elif not on_win():
            host_info = ' '.join(os.uname())
        else:
            host_info = _SystemInfo._system_command_call("VER")  # XXX: check shell from cygwin to call VER this way!
        return INDENT + INDENT + host_info.strip() + "\n" if host_info else ""

    @staticmethod
    def _system_command_call(command, **kwargs):
        if isinstance(command, list):
            command = subprocess.list2cmdline(command)
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, **kwargs)
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                # logger.debug('{}\nRunning {} failed with exit code {}\n'.format(stderr, command, process.returncode))
                raise get_svn_exception()(stdout=stdout, stderr=stderr, rc=process.returncode, cmd=[command])
            return stdout
        except OSError as e:
            msg = e.strerror
            errcodes = 'error {}'.format(e.errno)
            if on_win() and isinstance(e, WindowsError):
                errcodes += ', win-error {}'.format(e.winerror)
                try:
                    import ctypes

                    msg = str(ctypes.FormatError(e.winerror), _SystemInfo.get_locale()).encode('utf-8')
                except ImportError:
                    pass
            # logger.debug('System command call {} failed [{}]: {}\n'.format(command, errcodes, msg))
            return None


def _get_raw_data(vcs_type, vcs_root):
    lines = []
    if vcs_type == 'git':
        lines = _GitVersion.external_data(vcs_root)

    return [l.decode('utf-8') for l in lines]


def _get_json(vcs_root):
    try:
        vcs_type = "git"
        info = _get_vcs_dictionary(vcs_type, *_get_raw_data(vcs_type, vcs_root))
        return info, vcs_root
    except Exception:
        return None, ""


def _dump_json(
    arc_root,
    info,
    other_data=None,
    build_user=None,
    build_date=None,
    build_timestamp=0,
    custom_version='',
):
    j = {}
    j['PROGRAM_VERSION'] = info['scm_text'] + "\n" + _SystemInfo._to_text(other_data)
    j['CUSTOM_VERSION'] = str(_SystemInfo._to_text(custom_version))
    j['SCM_DATA'] = info['scm_text']
    j['ARCADIA_SOURCE_PATH'] = _SystemInfo._to_text(arc_root)
    j['ARCADIA_SOURCE_URL'] = info.get('url', info.get('svn_url', ''))
    j['ARCADIA_SOURCE_REVISION'] = info.get('revision', -1)
    j['ARCADIA_SOURCE_HG_HASH'] = info.get('hash', '')
    j['ARCADIA_SOURCE_LAST_CHANGE'] = info.get('commit_revision', info.get('svn_commit_revision', -1))
    j['ARCADIA_SOURCE_LAST_AUTHOR'] = info.get('commit_author', '')
    j['ARCADIA_PATCH_NUMBER'] = info.get('patch_number', 0)
    j['BUILD_USER'] = _SystemInfo._to_text(build_user)
    j['VCS'] = info.get('vcs', '')
    j['BRANCH'] = info.get('branch', '')
    j['ARCADIA_TAG'] = info.get('tag', '')
    j['DIRTY'] = info.get('dirty', '')

    if 'url' in info or 'svn_url' in info:
        j['SVN_REVISION'] = info.get('svn_commit_revision', info.get('revision', -1))
        j['SVN_ARCROOT'] = info.get('url', info.get('svn_url', ''))
        j['SVN_TIME'] = info.get('commit_date', info.get('svn_commit_date', ''))

    j['BUILD_DATE'] = build_date
    j['BUILD_TIMESTAMP'] = build_timestamp

    return json.dumps(j, sort_keys=True, indent=4, separators=(',', ': '))


def get_version_info(arc_root, custom_version=""):
    info, vcs_root = _get_json(arc_root)
    if info is None:
        return ""

    return _dump_json(
        vcs_root,
        info,
        other_data=_SystemInfo.get_other_data(
            src_dir=vcs_root,
        ),
        build_user=_SystemInfo.get_user(),
        build_date=_SystemInfo.get_date(None),
        build_timestamp=_SystemInfo.get_timestamp(),
        custom_version=custom_version,
    )


if __name__ == '__main__':
    with open(sys.argv[1], 'wt', encoding="utf-8") as f:
        f.write(get_version_info(sys.argv[2]))
