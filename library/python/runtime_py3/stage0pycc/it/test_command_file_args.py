import marshal
import os
import subprocess

import pytest
import yatest.common as yac


def binary():
    return yac.binary_path('library/python/runtime_py3/stage0pycc/stage0pycc' + ('.exe' if os.name == 'nt' else ''))


def read_code(output, module):
    code = marshal.loads(output.read_bytes())
    assert code.co_filename == module
    namespace = {}
    exec(code, namespace)
    assert namespace['VALUE'] == 7


@pytest.mark.parametrize('newline', [b'\n', b'\r\n'])
def test_command_file_matches_argv(tmp_path, newline):
    source = tmp_path / 'source with spaces.py'
    source.write_text('VALUE = 7\n', encoding='utf-8')
    direct = tmp_path / 'direct output.pyc'
    response_output = tmp_path / 'response output.pyc'
    module = 'virtual/module with spaces.py'
    subprocess.run(
        [binary(), '--ya-start-command-file', 'mod=' + module, str(source), str(direct), '--ya-end-command-file'],
        check=True,
    )
    response = tmp_path / 'arguments file.rsp'
    arguments = ['mod=' + module, str(source), str(response_output)]
    response.write_bytes(newline.join(arg.encode('utf-8') for arg in arguments) + newline)

    subprocess.run([binary(), '@' + str(response)], check=True)

    assert response_output.read_bytes() == direct.read_bytes()
    read_code(response_output, module)


def test_rejects_incomplete_command_file(tmp_path):
    response = tmp_path / 'incomplete.rsp'
    response.write_text('mod=module.py\nsource.py\n', encoding='utf-8')

    result = subprocess.run([binary(), '@' + str(response)], capture_output=True)

    assert result.returncode == 1
    assert b'Usage:' in result.stderr


def test_reports_missing_command_file(tmp_path):
    response = tmp_path / 'missing.rsp'

    result = subprocess.run([binary(), '@' + str(response)], capture_output=True)

    assert result.returncode == 1
    assert b"doesn't exist" in result.stderr
    assert response.name.encode() in result.stderr


def test_compiles_batch_larger_than_windows_command_line_limit(tmp_path):
    source = tmp_path / 'source with spaces.py'
    source.write_text('VALUE = 7\n', encoding='utf-8')
    arguments = []
    outputs = []
    while len(subprocess.list2cmdline([binary()] + arguments).encode('utf-16-le')) // 2 + 1 <= 32767:
        index = len(outputs)
        output = tmp_path / f'bytecode-{index}.pyc'
        module = f'virtual/module_{index}.py'
        arguments.extend(['mod=' + module, str(source), str(output)])
        outputs.append((output, module))
    response = tmp_path / 'arguments file.rsp'
    response.write_text('\n'.join(arguments) + '\n', encoding='utf-8')

    subprocess.run([binary(), '@' + str(response)], check=True, timeout=120)

    for output, module in outputs:
        read_code(output, module)


def test_ignores_empty_lines_and_reads_unterminated_last_line(tmp_path):
    source = tmp_path / 'source with spaces.py'
    source.write_text('VALUE = 7\n', encoding='utf-8')
    output = tmp_path / 'output with spaces.pyc'
    module = 'virtual/module with spaces.py'
    arguments = ['mod=' + module, str(source), str(output)]
    command_file = tmp_path / 'arguments.args'
    command_file.write_bytes(b'\n\r\n' + b'\n'.join(argument.encode() for argument in arguments))

    subprocess.run([binary(), '@' + str(command_file)], check=True)

    read_code(output, module)


def test_command_file_contents_are_not_expanded_recursively(tmp_path):
    (tmp_path / '@literal.py').write_text('VALUE = 7\n', encoding='utf-8')
    command_file = tmp_path / 'arguments.args'
    command_file.write_text('mod=module.py\n@literal.py\noutput.pyc\n', encoding='utf-8')

    subprocess.run([binary(), '@' + str(command_file)], cwd=tmp_path, check=True)

    read_code(tmp_path / 'output.pyc', 'module.py')


def test_rejects_nul_in_command_file(tmp_path):
    command_file = tmp_path / 'embedded-nul.args'
    command_file.write_bytes(b'mod=module.py\nsource\x00truncated.py\noutput.pyc\n')

    result = subprocess.run([binary(), '@' + str(command_file)], capture_output=True)

    assert result.returncode == 1
    assert b'contains a NUL byte' in result.stderr
    assert not (tmp_path / 'output.pyc').exists()
