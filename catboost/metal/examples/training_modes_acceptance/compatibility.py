"""Preserve prior release checks and add explicitly inventoried training modes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

from run import sha256


def main():
    metal = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('snapshots', 'cli', 'smoke'))
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--standalone', type=Path, default=metal / 'python')
    parser.add_argument('--cli', type=Path)
    parser.add_argument('--older-release', type=Path, default=metal / '.build/releases/20260913T155047Z')
    parser.add_argument('--prior-fixtures', type=Path, default=metal / '.build/training-modes-acceptance/preceding-214-fixtures')
    parser.add_argument('--baseline', type=Path, default=metal / '.build/training-modes-acceptance/old-220233Z-baselines')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    previous = metal / 'examples/compound_ctr_acceptance'
    current = Path(__file__).resolve().parent
    common = ['--package', str(args.package)]
    inventory = json.loads(subprocess.check_output([sys.executable, str(current / 'smoke.py'), 'list'], text=True))
    new_count = inventory['expected_cases']
    if args.mode == 'snapshots':
        suites = [
            ('prior-snapshots', 214, previous / 'compatibility.py', ['snapshots', '--baseline',
                str(args.prior_fixtures / 'old-195124Z-baselines'), '--release',
                str(args.prior_fixtures / 'preceding-202-fixtures'), '--standalone', str(args.standalone), *common]),
            ('baseline-220233Z', 30, current / 'run.py', ['baseline-replay', '--baseline', str(args.baseline), *common]),
        ]
    elif args.mode == 'cli':
        if args.cli is None:
            parser.error('cli mode requires --cli')
        suites = [
            ('prior-cli', 187, previous / 'compatibility.py', ['cli', '--older-release', str(args.older_release),
                '--standalone', str(args.standalone), '--cli', str(args.cli), *common]),
            ('training-mode-cli', new_count, current / 'smoke.py', ['cli', '--cli', str(args.cli), *common]),
        ]
    else:
        suites = [
            ('prior-smoke', 34, previous / 'compatibility.py', ['smoke', *common]),
            ('training-mode-smoke', new_count, current / 'smoke.py', ['smoke', *common]),
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
            assert child['status'] == 'passed' and len(child['cases']) == expected, name
            assert sha256(script) == record['runner_sha256'], 'helper changed during acceptance'
            record['report_sha256'] = sha256(target / 'report.json')
            report['cases'].extend(dict(release_suite=name, **case) for case in child['cases'])
            print(f'PASS {name}: {expected} cases', flush=True)
        assert len(report['cases']) == report['expected_cases']
        report['status'] = 'passed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{report['status']}: {len(report['cases'])}/{report['expected_cases']} cases; {args.output_dir / 'report.json'}", flush=True)


if __name__ == '__main__':
    main()
