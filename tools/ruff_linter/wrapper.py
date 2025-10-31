import logging
import os
import subprocess
import time
import tomllib
from pathlib import Path

from build.plugins.lib.test_const import RUFF_RESOURCE
from library.python.testing.custom_linter_util import linter_params, reporter
from library.python.testing.style import rules


logger = logging.getLogger(__name__)

FORMAT_SNIPPET_LINES_LIMIT = 100


def get_ruff_bin(params) -> str:
    ruff_root = params.global_resources[RUFF_RESOURCE]
    return os.path.join(ruff_root, 'bin', 'ruff')


def check_extend_option_present(config: Path) -> bool:
    with config.open('rb') as afile:
        cfg = tomllib.load(afile)
    if config.name == 'pyproject.toml' and cfg.get('tool', {}).get('ruff'):
        return 'extend' in cfg['tool']['ruff']
    elif config.name == 'ruff.toml':
        return 'extend' in cfg
    raise RuntimeError(f'Unknown config type: {config.name}')


def run_ruff(ruff_bin: str, cmd_args: list[str], filename: str, config: Path) -> list[str]:
    # XXX: `--no-cache` is important when we run ruff in source root and don't want to pollute arcadia
    cmd = [ruff_bin, *cmd_args, '--no-cache', '--config', config, filename]
    res = subprocess.run(
        cmd,
        capture_output=True,
        encoding='utf8',
        errors='replace',
        env=dict(os.environ.copy(), RUFF_OUTPUT_FORMAT='concise'),
        # When config is passed through `--config`, `exclude` starts searching from cwd
        # so set cwd to config cwd to mimic behavior of autodiscovery.
        # Note that it stops being accurate when `extend` is used.
        # https://docs.astral.sh/ruff/configuration/#config-file-discovery
        cwd=os.path.dirname(config),
    )
    return res.stdout.splitlines(keepends=True) if res.returncode else []


def process_file(
    orig_filename: str, ruff_bin: str, orig_config: Path, source_root: str, check_format: bool, run_in_source_root: bool
) -> str:
    logger.debug('Check %s with config %s', orig_filename, orig_config)

    file_path = os.path.relpath(orig_filename, source_root)

    if run_in_source_root:
        filename = os.path.realpath(orig_filename) if os.path.islink(orig_filename) else orig_filename
        config = orig_config.resolve() if orig_config.is_symlink() else orig_config
    else:
        filename = orig_filename
        config = orig_config

    if check_format:
        ruff_format_check_out = run_ruff(ruff_bin, ['format', '--diff'], filename, config)
        if len(ruff_format_check_out) > FORMAT_SNIPPET_LINES_LIMIT:
            ruff_format_check_out = ruff_format_check_out[:FORMAT_SNIPPET_LINES_LIMIT]
            ruff_format_check_out.append('[truncated]...\n')
        # first two lines are absolute file paths, replace with relative ones
        if ruff_format_check_out:
            ruff_format_check_out[0] = f'--- [[imp]]{file_path}[[rst]]\n'
            ruff_format_check_out[1] = f'+++ [[imp]]{file_path}[[rst]]\n'
    else:
        ruff_format_check_out = []

    ruff_check_out = run_ruff(ruff_bin, ['check', '-q'], filename, config)
    # Every line starts with an absolute path to a file, replace with relative one
    for idx, line in enumerate(ruff_check_out):
        ruff_check_out[idx] = f'[[imp]]{file_path}[[rst]]:{line.split(':', 1)[-1]}'

    msg = ''
    if ruff_format_check_out:
        msg += '[[bad]]Formatting errors[[rst]]:\n'
        msg += ''.join(ruff_format_check_out)

    if ruff_format_check_out and ruff_check_out:
        msg += '\n'

    if ruff_check_out:
        msg += '[[bad]]Linting errors[[rst]]:\n'
        msg += ''.join(ruff_check_out)

    return msg


def main():
    params = linter_params.get_params()

    style_config_path = Path(params.configs[0])

    # TODO: Ideally, to enable `extend` we first should move execution to build root (src files + configs)
    # otherwise we risk allowing to steal from arcadia. To do that we need to mark modules 1st-party/3rd-party
    # in pyproject.toml.
    # UPD: it turned out to be more complicated for uservices-like projects because they use TOP_LEVEL / NAMESPACES
    extend_option_present = check_extend_option_present(style_config_path)

    ruff_bin = get_ruff_bin(params)

    report = reporter.LintReport()
    for file_name in params.files:
        start_time = time.perf_counter()

        if extend_option_present:
            elapsed = time.perf_counter() - start_time
            report.add(
                file_name,
                reporter.LintStatus.FAIL,
                "`extend` option in not supported in ruff config files for now. Modify your configs not to have it.",
                elapsed=elapsed,
            )
            continue

        skip_reason = rules.get_skip_reason(file_name, Path(file_name).read_text(), skip_links=False)
        if skip_reason:
            elapsed = time.perf_counter() - start_time
            report.add(
                file_name,
                reporter.LintStatus.SKIPPED,
                f"Style check is omitted: {skip_reason}",
                elapsed=elapsed,
            )
            continue

        error = process_file(
            file_name,
            ruff_bin,
            style_config_path,
            params.source_root,
            params.extra_params.get('check_format') == 'yes',
            params.extra_params.get('run_in_source_root') == 'yes',
        )
        elapsed = time.perf_counter() - start_time

        if error:
            rel_file_name = os.path.relpath(file_name, params.source_root)
            message = f'Run [[imp]]ya style --ruff {rel_file_name}[[rst]] to fix format\n{error}'
            status = reporter.LintStatus.FAIL
        else:
            message = ''
            status = reporter.LintStatus.GOOD
        report.add(file_name, status, message, elapsed=elapsed)

    report.dump(params.report_file)


if __name__ == '__main__':
    main()
