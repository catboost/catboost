"""Rerun the preceding Metal acceptance suites without changing old fixtures."""

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
    parser.add_argument('--release', type=Path, default=metal / '.build/releases/20260913T195124Z')
    parser.add_argument('--older-release', type=Path, default=metal / '.build/releases/20260913T155047Z')
    parser.add_argument('--baseline', type=Path, default=metal / '.build/compound-ctr-acceptance/old-195124Z-baselines')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.absolute())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    previous = metal / 'examples/greedy_pairlogit_acceptance'
    current = Path(__file__).resolve().parent
    common = ['--package', str(args.package)]
    suites = []
    if args.mode == 'snapshots':
        suites = [
            ('baseline-195124Z', 12, current / 'run.py', ['baseline-replay', '--baseline', str(args.baseline), *common]),
            ('baseline-155047Z', 8, previous / 'run.py', ['baseline-replay', '--baseline',
                str(args.release / 'old-155047Z-baselines'), *common]),
            ('legacy-151643Z', 194, previous / 'run.py', ['legacy-replay', '--release',
                str(args.release / 'legacy-snapshot-fixtures'), '--standalone', str(args.standalone), *common]),
        ]
    elif args.mode == 'cli':
        if args.cli is None:
            parser.error('cli mode requires --cli')
        suites = [
            ('prior-cli', 153, previous / 'prior_cli.py', ['--release', str(args.older_release),
                '--standalone', str(args.standalone), '--cli', str(args.cli), *common]),
            ('pairlogit-cli', 18, previous / 'run.py', ['cli', '--cli', str(args.cli), *common]),
            ('compound-cli', 16, current / 'smoke.py', ['cli', '--cli', str(args.cli), *common]),
        ]
    else:
        suites = [
            ('pairlogit-smoke', 18, previous / 'run.py', ['smoke', *common]),
            ('compound-smoke', 16, current / 'smoke.py', ['smoke', *common]),
        ]
    report = dict(mode=args.mode, status='running', cases=[], suites=[], started_unix=time.time(),
                  runner_sha256=sha256(__file__), package=str(args.package), python=sys.executable)
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
            report['cases'].extend(dict(acceptance_suite=name, **case) for case in child['cases'])
            print(f'PASS {name}: {expected} cases', flush=True)
        report['status'] = 'passed'
    except BaseException:
        report.update(status='failed', error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{report['status']}: {len(report['cases'])} cases; {args.output_dir / 'report.json'}", flush=True)


if __name__ == '__main__':
    main()
