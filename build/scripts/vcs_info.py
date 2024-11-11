import base64
import json
import os
import re
import sys
import shutil
import tempfile
import textwrap
import zipfile


class _Formatting(object):
    @staticmethod
    def is_str(strval):
        return isinstance(strval, (bytes, str))

    @staticmethod
    def encoding_needed(strval):
        return isinstance(strval, str)

    @staticmethod
    def escape_special_symbols(strval):
        encoding_needed = _Formatting.encoding_needed(strval)
        c_str = strval.encode('utf-8') if encoding_needed else strval
        retval = b""
        for c in c_str:
            c = bytes([c])
            if c in (b"\\", b"\""):
                retval += b"\\" + c
            elif ord(c) < ord(' '):
                retval += c.decode('latin-1').encode('unicode_escape')
            else:
                retval += c
        return retval.decode('utf-8') if encoding_needed else retval

    @staticmethod
    def escape_line_feed(strval, indent='    '):
        return strval.replace(r'\n', '\\n"\\\n' + indent + '"')

    @staticmethod
    def escape_trigraphs(strval):
        return strval.replace(r'?', '\\?')

    @staticmethod
    def escaped_define(strkey, val):
        name = "#define " + strkey + " "
        if _Formatting.is_str(val):
            define = (
                "\""
                + _Formatting.escape_line_feed(_Formatting.escape_trigraphs(_Formatting.escape_special_symbols(val)))
                + "\""
            )
        else:
            define = str(val)
        return name + define

    @staticmethod
    def escaped_go_map_key(strkey, strval):
        if _Formatting.is_str(strval):
            return '    ' + '"' + strkey + '": "' + _Formatting.escape_special_symbols(strval) + '",'
        else:
            return '    ' + '"' + strkey + '": "' + str(strval) + '",'


def get_default_json():
    return json.loads(
        '''{
    "ARCADIA_SOURCE_HG_HASH": "0000000000000000000000000000000000000000",
    "ARCADIA_SOURCE_LAST_AUTHOR": "<UNKNOWN>",
    "ARCADIA_SOURCE_LAST_CHANGE": -1,
    "ARCADIA_SOURCE_PATH": "/",
    "ARCADIA_SOURCE_REVISION": -1,
    "ARCADIA_SOURCE_URL": "",
    "BRANCH": "unknown-vcs-branch",
    "BUILD_DATE": "",
    "BUILD_TIMESTAMP": 0,
    "BUILD_HOST": "localhost",
    "BUILD_USER": "nobody",
    "CUSTOM_VERSION": "",
    "RELEASE_VERSION": "",
    "PROGRAM_VERSION": "Arc info:\\n    Branch: unknown-vcs-branch\\n    Commit: 0000000000000000000000000000000000000000\\n    Author: <UNKNOWN>\\n    Summary: No VCS\\n\\n",
    "SCM_DATA": "Arc info:\\n    Branch: unknown-vcs-branch\\n    Commit: 0000000000000000000000000000000000000000\\n    Author: <UNKNOWN>\\n    Summary: No VCS\\n",
    "VCS": "arc",
    "ARCADIA_PATCH_NUMBER": 0,
    "ARCADIA_TAG": ""
}'''
    )


def get_json(file_name):
    try:
        with open(file_name, 'rt', encoding="utf-8") as f:
            out = json.load(f)

        # TODO: check 'tar+svn' parsing
        for num_var in ['ARCADIA_SOURCE_REVISION', 'ARCADIA_SOURCE_LAST_CHANGE', 'SVN_REVISION']:
            if num_var in out and _Formatting.is_str(out[num_var]):
                try:
                    out[num_var] = int(out[num_var])
                except:
                    out[num_var] = -1
        return out
    except:
        return get_default_json()


def print_c(json_file, output_file, argv):
    """params:
    json file
    output file
    $(SOURCE_ROOT)/build/scripts/c_templates/svn_interface.c"""
    interface = argv[0]

    with open(interface, 'rt', encoding="utf-8") as c:
        c_file = c.read()
    with open(output_file, 'wt', encoding="utf-8") as f:
        header = '\n'.join(_Formatting.escaped_define(k, v) for k, v in json_file.items())
        f.write(header + '\n' + c_file)


def merge_java_content(old_content, json_file):
    new_content, names = print_java_mf(json_file)

    def split_to_sections(content):
        sections = []
        cur_section = []
        for l in content:
            if l.rstrip():
                cur_section.append(l)
            else:
                sections.append(cur_section)
                cur_section = []

        if cur_section:  # should not be needed according to format specification
            sections.append(cur_section)

        return sections

    def drop_duplicate_entries(main_section, names):
        header = re.compile('^([A-Za-z0-9][A-Za-z0-9_-]*): .*$')
        new_main_section = []
        for l in main_section:
            match = header.match(l)
            # duplicate entry
            if match:
                skip = match.group(1) in names

            if not skip:
                new_main_section.append(l)
        return new_main_section

    if old_content:
        sections = split_to_sections(old_content)
        sections[0] = drop_duplicate_entries(sections[0], names)
    else:
        sections = [['Manifest-Version: 1.0\n']]

    sections[0].extend(map(lambda x: x + '\n', new_content))

    return ''.join(map(lambda x: ''.join(x), sections)) + '\n'


def merge_java_mf_jar(json_file, out_manifest, jar_file):
    try:
        temp_dir = tempfile.mkdtemp()
        try:
            with zipfile.ZipFile(jar_file, 'r') as jar:
                jar.extract(os.path.join('META-INF', 'MANIFEST.MF'), path=temp_dir)
        except KeyError:
            pass

        merge_java_mf_dir(json_file, out_manifest, temp_dir)
    finally:
        shutil.rmtree(temp_dir)


def merge_java_mf_dir(json_file, out_manifest, input_dir):
    manifest = os.path.join(input_dir, 'META-INF', 'MANIFEST.MF')

    old_lines = []
    if os.path.isfile(manifest):
        with open(manifest, 'rt', encoding="utf-8") as f:
            old_lines = f.readlines()

    with open(out_manifest, 'wt', encoding="utf-8") as f:
        f.write(merge_java_content(old_lines, json_file))


def merge_java_mf(json_file, out_manifest, input):
    if zipfile.is_zipfile(input):
        merge_java_mf_jar(json_file, out_manifest, input)
    elif os.path.isdir(input):
        merge_java_mf_dir(json_file, out_manifest, input)


def print_java_mf(info):
    wrapper = textwrap.TextWrapper(
        subsequent_indent=' ', break_long_words=True, replace_whitespace=False, drop_whitespace=False
    )
    names = set()

    def wrap(key, val):
        names.add(key[:-2])
        if not val:
            return []
        return wrapper.wrap(key + val)

    lines = wrap('Program-Version-String: ', base64.b64encode(info['PROGRAM_VERSION'].encode('utf-8')).decode('utf-8'))
    lines += wrap('SCM-String: ', base64.b64encode(info['SCM_DATA'].encode('utf-8')).decode('utf-8'))
    lines += wrap('Arcadia-Source-Path: ', info['ARCADIA_SOURCE_PATH'])
    lines += wrap('Arcadia-Source-URL: ', info['ARCADIA_SOURCE_URL'])
    lines += wrap('Arcadia-Source-Revision: ', str(info['ARCADIA_SOURCE_REVISION']))
    lines += wrap('Arcadia-Source-Hash: ', info['ARCADIA_SOURCE_HG_HASH'].rstrip())
    lines += wrap('Arcadia-Source-Last-Change: ', str(info['ARCADIA_SOURCE_LAST_CHANGE']))
    lines += wrap('Arcadia-Source-Last-Author: ', info['ARCADIA_SOURCE_LAST_AUTHOR'])
    lines += wrap('Build-User: ', info['BUILD_USER'])
    lines += wrap('Build-Host: ', info['BUILD_HOST'])
    lines += wrap('Version-Control-System: ', info['VCS'])
    lines += wrap('Branch: ', info['BRANCH'])
    lines += wrap('Arcadia-Tag: ', info.get('ARCADIA_TAG', ''))
    lines += wrap('Arcadia-Patch-Number: ', str(info.get('ARCADIA_PATCH_NUMBER', 42)))
    if 'SVN_REVISION' in info:
        lines += wrap('SVN-Revision: ', str(info['SVN_REVISION']))
        lines += wrap('SVN-Arcroot: ', info['SVN_ARCROOT'])
        lines += wrap('SVN-Time: ', info['SVN_TIME'])
    lines += wrap('Build-Date: ', info['BUILD_DATE'])
    if 'BUILD_TIMESTAMP' in info:
        lines += wrap('Build-Timestamp: ', str(info['BUILD_TIMESTAMP']))
    if 'CUSTOM_VERSION' in info:
        lines += wrap(
            'Custom-Version-String: ', base64.b64encode(info['CUSTOM_VERSION'].encode('utf-8')).decode('utf-8')
        )
    if 'RELEASE_VERSION' in info:
        lines += wrap(
            'Release-Version-String: ', base64.b64encode(info['RELEASE_VERSION'].encode('utf-8')).decode('utf-8')
        )
    return lines, names


def print_java(json_file, output_file, argv):
    """params:
    json file
    output file
    file"""
    input = argv[0] if argv else os.curdir
    merge_java_mf(json_file, output_file, input)


def print_go(json_file, output_file, arc_project_prefix):
    def gen_map(info):
        lines = []
        for k, v in info.items():
            lines.append(_Formatting.escaped_go_map_key(k, v))
        return lines

    with open(output_file, 'wt', encoding="utf-8") as f:
        f.write(
            '\n'.join(
                [
                    '// Code generated by vcs_info.py; DO NOT EDIT.',
                    '',
                    'package main',
                    'import "{}library/go/core/buildinfo"'.format(arc_project_prefix),
                    'func init() {',
                    '   buildinfo.InitBuildInfo(map[string]string {',
                ]
                + gen_map(json_file)
                + ['})', '}']
            )
            + '\n'
        )


def print_json(json_file, output_file):
    MANDATOTRY_FIELDS_MAP = {
        'ARCADIA_TAG': 'Arcadia-Tag',
        'ARCADIA_PATCH_NUMBER': 'Arcadia-Patch-Number',
        'ARCADIA_SOURCE_URL': 'Arcadia-Source-URL',
        'ARCADIA_SOURCE_REVISION': 'Arcadia-Source-Revision',
        'ARCADIA_SOURCE_HG_HASH': 'Arcadia-Source-Hash',
        'ARCADIA_SOURCE_LAST_CHANGE': 'Arcadia-Source-Last-Change',
        'ARCADIA_SOURCE_LAST_AUTHOR': 'Arcadia-Source-Last-Author',
        'BRANCH': 'Branch',
        'BUILD_HOST': 'Build-Host',
        'BUILD_USER': 'Build-User',
        'PROGRAM_VERSION': 'Program-Version-String',
        'SCM_DATA': 'SCM-String',
        'VCS': 'Version-Control-System',
    }

    SVN_REVISION = 'SVN_REVISION'

    SVN_FIELDS_MAP = {
        SVN_REVISION: 'SVN-Revision',
        'SVN_ARCROOT': 'SVN-Arcroot',
        'SVN_TIME': 'SVN-Time',
    }

    OPTIONAL_FIELDS_MAP = {
        'BUILD_TIMESTAMP': 'Build-Timestamp',
        'CUSTOM_VERSION': 'Custom-Version-String',
        'RELEASE_VERSION': 'Release-Version-String',
        'DIRTY': 'Working-Copy-State',
    }

    ext_json = {}

    for k in MANDATOTRY_FIELDS_MAP:
        ext_json[MANDATOTRY_FIELDS_MAP[k]] = json_file[k]

    if SVN_REVISION in json_file:
        for k in SVN_FIELDS_MAP:
            ext_json[SVN_FIELDS_MAP[k]] = json_file[k]

    for k in OPTIONAL_FIELDS_MAP:
        if k in json_file and json_file[k]:
            ext_json[OPTIONAL_FIELDS_MAP[k]] = json_file[k]

    with open(output_file, 'wt', encoding="utf-8") as f:
        json.dump(ext_json, f, sort_keys=True, indent=4)


if __name__ == '__main__':
    if 'output-go' in sys.argv:
        lang = 'Go'
        sys.argv.remove('output-go')
    elif 'output-java' in sys.argv:
        lang = 'Java'
        sys.argv.remove('output-java')
    elif 'output-json' in sys.argv:
        lang = 'JSON'
        sys.argv.remove('output-json')
    else:
        lang = 'C'

    if 'no-vcs' in sys.argv:
        sys.argv.remove('no-vcs')
        json_file = get_default_json()
    else:
        json_name = sys.argv[1]
        json_file = get_json(json_name)

    if lang == 'Go':
        print_go(json_file, sys.argv[2], sys.argv[3])
    elif lang == 'Java':
        print_java(json_file, sys.argv[2], sys.argv[3:])
    elif lang == 'JSON':
        print_json(json_file, sys.argv[2])
    else:
        print_c(json_file, sys.argv[2], sys.argv[3:])
