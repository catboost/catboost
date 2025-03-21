from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re
import typing

base_path = pathlib.Path(__file__).parent.parent
src_path = base_path / 'portalocker'
dist_path = base_path / 'dist'
_default_output_path = base_path / 'dist' / 'portalocker.py'

_NAMES_RE = re.compile(r'(?P<names>[^()]+)$')
_RELATIVE_IMPORT_RE = re.compile(
    r'^from \.(?P<from>.*?) import (?P<paren>\(?)(?P<names>[^()]+)$',
)
_USELESS_ASSIGNMENT_RE = re.compile(r'^(?P<name>\w+) = \1\n$')

_TEXT_TEMPLATE = """'''
{}
'''

"""

logger = logging.getLogger(__name__)


def main(argv: typing.Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(required=True)
    combine_parser = subparsers.add_parser(
        'combine',
        help='Combine all Python files into a single unified `portalocker.py` '
        'file for easy distribution',
    )
    combine_parser.add_argument(
        '--output-file',
        '-o',
        type=argparse.FileType('w'),
        default=str(_default_output_path),
    )

    combine_parser.set_defaults(func=combine)
    args = parser.parse_args(argv)
    args.func(args)


def _read_file(
    path: pathlib.Path,
    seen_files: set[pathlib.Path],
) -> typing.Iterator[str]:
    if path in seen_files:
        return

    names: set[str] = set()
    seen_files.add(path)
    paren = False
    from_ = None
    for line in path.open():
        if '__future__' in line:
            continue

        if paren:
            if ')' in line:
                line = line.split(')', 1)[1]
                paren = False
                continue

            match = _NAMES_RE.match(line)
        else:
            match = _RELATIVE_IMPORT_RE.match(line)

        if match:
            if not paren:
                paren = bool(match.group('paren'))
                from_ = match.group('from')

            if from_:
                names.add(from_)
                yield from _read_file(src_path / f'{from_}.py', seen_files)
            else:
                for name in match.group('names').split(','):
                    name = name.strip()
                    names.add(name)
                    yield from _read_file(src_path / f'{name}.py', seen_files)
        else:
            yield _clean_line(line, names)


def _clean_line(line: str, names: set[str]) -> str:
    # Replace `some_import.spam` with `spam`
    if names:
        joined_names = '|'.join(names)
        line = re.sub(rf'\b({joined_names})\.', '', line)

    # Replace useless assignments (e.g. `spam = spam`)
    return _USELESS_ASSIGNMENT_RE.sub('', line)


def combine(args: argparse.Namespace) -> None:
    output_file = args.output_file
    pathlib.Path(output_file.name).parent.mkdir(parents=True, exist_ok=True)

    # We're handling this separately because it has to be the first import.
    output_file.write('from __future__ import annotations\n')

    output_file.write(
        _TEXT_TEMPLATE.format((base_path / 'README.rst').read_text()),
    )
    output_file.write(
        _TEXT_TEMPLATE.format((base_path / 'LICENSE').read_text()),
    )

    seen_files: set[pathlib.Path] = set()
    for line in _read_file(src_path / '__init__.py', seen_files):
        output_file.write(line)

    output_file.flush()
    output_file.close()

    logger.info(f'Wrote combined file to {output_file.name}')
    # Run black and ruff if available. If not then just run the file.
    os.system(f'black {output_file.name}')
    os.system(f'ruff format {output_file.name}')
    os.system(f'ruff check --fix --fix-only {output_file.name}')
    os.system(f'python3 {output_file.name}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()
