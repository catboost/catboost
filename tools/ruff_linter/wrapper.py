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
    # XXX: `--no-cache` is important because we run ruff in source root and don't want to pollute arcadia
    cmd = [ruff_bin, *cmd_args, '--no-cache', '--config', config, filename]
    res = subprocess.run(
        cmd,
        capture_output=True,
        encoding='utf8',
        errors='replace',
        env=dict(os.environ.copy(), RUFF_OUTPUT_FORMAT='concise'),
        cwd=os.path.dirname(config),  # for correct support of `exclude`
    )
    return res.stdout.splitlines(keepends=True) if res.returncode else []


def process_file(orig_filename: str, ruff_bin: str, orig_config: Path, source_root: str, check_format: bool) -> str:
    logger.debug('Check %s with config %s', orig_filename, orig_config)

    # In order for `exclude` option (pyproject.toml) to work we have two options:
    # 1. Run ruff with files AND config in build root
    # 2. Run ruff with files AND config in source root
    # For these two options there are differences in how 1st party vs 3rd party libraries are detected: in source root
    # we have access to the whole arcadia. Because of that PEERDIR'ed arcadia libraries are considered 1st party.
    # In build root, on the contrary, PEERDIR'ed libraries are not available so they are considered 3rd party.
    # In order to match `ya style` behavior which is executed as in the second case we have to go with the second option.
    # Then we MUST make sure we don't write anything to arcadia and don't "steal" files from arcadia.
    # From the model point of view first option is more correct but it would require marking all PEERDIR'ed libraries
    # 3rd party in pyproject.toml (`known-third-party`).

    file_path = os.path.relpath(orig_filename, source_root)

    if file_path.startswith(('fintech/uservices', 'taxi', 'sdg', 'electro')):
        # TODO(alevitskii) TPS-28865, TPS-31380. Run checks for fintech and taxi in build root too.
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

    # TODO To enable `extend` we first need to fully move execution to build root (src files + configs)
    # otherwise we risk allowing to steal from arcadia. To do that we need to mark modules 1st-party/3rd-party
    # in pyproject.toml.
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
            file_name, ruff_bin, style_config_path, params.source_root, params.extra_params.get('check_format') == 'yes'
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
