import base64
import json
import os
import sys
import textwrap


class _Formatting(object):
    @staticmethod
    def escape_special_symbols(strval):
        c_str = strval.encode('utf-8') if isinstance(strval, unicode) else strval
        retval = ""
        for c in c_str:
            if c in ("\\", "\""):
                retval += "\\" + c
            elif ord(c) < ord(' '):
                retval += c.encode("string_escape")
            else:
                retval += c
        return retval.decode('utf-8') if isinstance(strval, unicode) else retval

    @staticmethod
    def escape_line_feed(strval, indent='    ', cont=True):
        return strval.replace(r'\n', '\\n"\\\n' + indent + '"' if cont else '\\n"\n' + indent + '"')

    @staticmethod
    def escaped_define(strkey, val):
        name = "#define " + strkey + " "
        if isinstance(val, basestring):
            define = "\"" + _Formatting.escape_line_feed(
                _Formatting.escape_special_symbols(val)) + "\""
        else:
            define = str(val)
        return name + define

    @staticmethod
    def escaped_go_map_key(strkey, strval):
        if isinstance(strval, basestring):
            return '    ' + '"' + strkey + '": "' + _Formatting.escape_line_feed(_Formatting.escape_special_symbols(strval), '      ', False) + '",'
        else:
            return '    ' + '"' + strkey + '": "' + str(strval) + '",'


def get_default_json():
    return json.loads('''{
    "ARCADIA_SOURCE_HG_HASH": "0577215664901532860606512090082402431042",
    "ARCADIA_SOURCE_LAST_AUTHOR": "ordinal",
    "ARCADIA_SOURCE_LAST_CHANGE": -1,
    "ARCADIA_SOURCE_PATH": "/",
    "ARCADIA_SOURCE_REVISION": -1,
    "ARCADIA_SOURCE_URL": "",
    "BRANCH": "math",
    "BUILD_DATE": "",
    "BUILD_HOST": "localhost",
    "BUILD_USER": "nobody",
    "PROGRAM_VERSION": "Arc info:\\n    Branch: math\\n    Commit: 0577215664901532860606512090082402431042\\n    Author: ordinal\\n    Summary: No VCS\\n\\n",
    "SCM_DATA": "Arc info:\\n    Branch: math\\n    Commit: 0577215664901532860606512090082402431042\\n    Author: ordinal\\n    Summary: No VCS\\n",
    "VCS": "arc"
}''')


def get_json(file_name):
    try:
        with open(file_name, 'r') as f:
            out = json.load(f)

        # TODO: check 'tar+svn' parsing
        for i in ['ARCADIA_SOURCE_REVISION', 'ARCADIA_SOURCE_LAST_CHANGE', 'SVN_REVISION']:
            if i in out and isinstance(out[i], basestring):
                try:
                    out[i] = int(out[i])
                except:
                    out[i] = -1
        return out
    except:
        return get_default_json()


def print_c(json_file, output_file, argv):
    """ params:
            json file
            output file
            $(SOURCE_ROOT)/build/scripts/c_templates/svn_interface.c.template"""
    def gen_header(info):
        lines = []
        for k, v in info.items():
            lines.append(_Formatting.escaped_define(k, v))
        return lines

    interface = argv[0]

    with open(interface) as c:
        c_file = c.read()
    with open(output_file, 'w') as f:
        f.write('\n'.join(gen_header(json_file)).encode('utf-8') + '\n' + c_file)


def merge_java_mf(json_file, out_manifest, input_dir):
    manifest = os.path.join(input_dir, 'META-INF', 'MANIFEST.MF')
    if not os.path.isfile(manifest):
        cont = 'Manifest-Version: 1.0'
    else:
        with open(manifest, 'r') as f:
            cont = f.read().rstrip()

    with open(out_manifest, 'w') as f:
        f.write(cont + '\n')

    with open(out_manifest, 'a') as f:
        f.write('\n'.join(print_java_mf(json_file)) + '\n\n')


def print_java_mf(info):
    wrapper = textwrap.TextWrapper(subsequent_indent=' ')

    def wrap(key, val):
        if not val:
            return []
        return wrapper.wrap(key + val)

    lines = wrap('Program-Version-String: ', base64.b64encode(info['PROGRAM_VERSION'].encode('utf-8')))
    lines += wrap('SCM-String: ', base64.b64encode(info['SCM_DATA'].encode('utf-8')))
    lines += wrap('Arcadia-Source-Path: ', info['ARCADIA_SOURCE_PATH'])
    lines += wrap('Arcadia-Source-URL: ', info['ARCADIA_SOURCE_URL'])
    lines += wrap('Arcadia-Source-Revision: ', str(info['ARCADIA_SOURCE_REVISION']))
    lines += wrap('Arcadia-Source-Hg-Hash: ', info['ARCADIA_SOURCE_HG_HASH'])
    lines += wrap('Arcadia-Source-Last-Change: ', str(info['ARCADIA_SOURCE_LAST_CHANGE']))
    lines += wrap('Arcadia-Source-Last-Author: ', info['ARCADIA_SOURCE_LAST_AUTHOR'])
    lines += wrap('Build-User: ', info['BUILD_USER'])
    lines += wrap('Build-Host: ', info['BUILD_HOST'])
    lines += wrap('Version-Control-System: ', info['VCS'])
    lines += wrap('Branch: ', info['BRANCH'])
    if 'SVN_REVISION' in info:
        lines += wrap('SVN-Revision: ', str(info['SVN_REVISION']))
        lines += wrap('SVN-Arcroot: ', info['SVN_ARCROOT'])
        lines += wrap('SVN-Time: ', info['SVN_TIME'])
    lines += wrap('Build-Date: ', info['BUILD_DATE'])
    return lines


def print_java(json_file, output_file, argv):
    """ params:
            json file
            output file
            file"""
    input_dir = argv[0] if argv else os.curdir
    merge_java_mf(json_file, output_file, input_dir)


def print_go(json_file, output_file):
    def gen_map(info):
        lines = []
        for k, v in info.items():
            lines.append(_Formatting.escaped_go_map_key(k, v))
        return lines

    with open(output_file, 'w') as f:
        f.write('\n'.join([
            'package main',
            'var buildinfo = map[string]string{'] + gen_map(json_file) + ['}']).encode('utf-8') + '\n')


if __name__ == '__main__':
    if 'output-go' in sys.argv:
        lang = 'Go'
        sys.argv.remove('output-go')
    elif 'output-java' in sys.argv:
        lang = 'Java'
        sys.argv.remove('output-java')
    else:
        lang = 'C'

    if 'no-vcs' in sys.argv:
        sys.argv.remove('no-vcs')
        json_file = get_default_json()
    else:
        json_name = sys.argv[1]
        json_file = get_json(json_name)

    if lang == 'Go':
        print_go(json_file, sys.argv[2])
    elif lang == 'Java':
        print_java(json_file, sys.argv[2], sys.argv[3:])
    else:
        print_c(json_file, sys.argv[2], sys.argv[3:])
