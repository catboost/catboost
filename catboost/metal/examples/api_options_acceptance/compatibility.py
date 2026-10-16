"""Replay every preceding check and the explicitly inventoried API additions."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    current = Path(__file__).resolve().parent
    metal = current.parents[1]
    evidence = metal / '.build/api-options-acceptance'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('snapshots', 'cli', 'smoke'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--standalone', type=Path, default=metal / 'python')
    parser.add_argument('--cli', type=Path)
    parser.add_argument('--prior-fixtures', type=Path, default=evidence / 'preceding-244-fixtures')
    parser.add_argument('--baseline', type=Path, default=evidence / 'old-000929Z-baselines')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    previous = metal / 'examples/training_modes_acceptance/compatibility.py'
    common = ['--package', str(args.package)]
    prior_common = ['--standalone', str(args.standalone), *common]
    inventory = {}
    if args.mode == 'snapshots':
        suites = [
            ('prior-snapshots', 244, previous, ['snapshots', '--prior-fixtures',
                str(args.prior_fixtures / 'preceding-214-fixtures'), '--baseline',
                str(args.prior_fixtures / 'old-220233Z-baselines'), *prior_common]),
            ('baseline-000929Z', 106, current / 'baselines.py',
                ['replay', '--baseline', str(args.baseline), *common]),
        ]
    else:
        inventory = json.loads(subprocess.check_output(
            [sys.executable, str(current / 'smoke.py'), 'list'], text=True))
        new_count = inventory.get('expected_cases_' + args.mode, inventory.get('expected_cases'))
        assert isinstance(new_count, int) and new_count > 0
        cli = []
        if args.mode == 'cli':
            if args.cli is None:
                parser.error('cli mode requires --cli')
            cli = ['--cli', str(args.cli)]
        suites = [
            ('prior-' + args.mode, 287 if args.mode == 'cli' else 134, previous,
                [args.mode, *cli, *prior_common]),
            ('api-options-' + args.mode, new_count, current / 'smoke.py',
                [args.mode, *cli, *common]),
        ]
    report = dict(mode=args.mode, status='running', cases=[], suites=[], started_unix=time.time(),
                  expected_cases=sum(case[1] for case in suites), runner_sha256=sha256(__file__),
                  package=str(args.package), python=sys.executable, new_smoke_inventory=inventory)
    try:
        for name, expected, script, options in suites:
            target = args.output_dir / name
            command = [sys.executable, str(script), *options, '--output-dir', str(target)]
            record = dict(name=name, command=command, runner_sha256=sha256(script), expected_cases=expected)
            report['suites'].append(record)
            with (args.output_dir / (name + '.log')).open('w') as output:
                completed = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT)
            record['exit_code'] = completed.returncode
            assert completed.returncode == 0, str(args.output_dir / (name + '.log'))
            child = json.loads((target / 'report.json').read_text())
            assert child['status'] == 'passed' and child['expected_cases'] == len(child['cases']) == expected
            assert sha256(script) == record['runner_sha256'], 'helper changed during acceptance'
            record['report_sha256'] = sha256(target / 'report.json')
            report['cases'].extend(dict(case, release_suite=name) for case in child['cases'])
            print(f'PASS {name}: {expected} cases', flush=True)
        assert len(report['cases']) == report['expected_cases']
        report['status'] = 'passed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{report['status']}: {len(report['cases'])}/{report['expected_cases']} cases", flush=True)


if __name__ == '__main__':
    main()
