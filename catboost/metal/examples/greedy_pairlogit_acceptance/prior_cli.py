"""Replay the preceding 153 CLI checks in a fresh, confined staging folder.

Only the 14 reachable scripts and two seed data files are copied. Their test
logic is unchanged; literal executable/output/test-helper paths are remapped.
"""

import argparse
import ast
import json
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import time
import traceback

from run import sha256


ENTRY = 'catbooster-greedy-query-cli.py'
OLD_ROOT = '/Users/jefferypowell/Desktop/Catbooster'
OLD_CLI = '/tmp/catbooster-native-build/catboost/app/catboost'
SEED = 'catbooster-greedy-query-base-base-base-base-base-base-final-base-cli-numeric'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    metal = Path(__file__).resolve().parents[2]
    parser.add_argument('--release', type=Path, default=metal / '.build/releases/20260913T155047Z')
    parser.add_argument('--package', type=Path, required=True)
    parser.add_argument('--standalone', type=Path, default=metal / 'python')
    parser.add_argument('--cli', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    for key, value in vars(args).items():
        setattr(args, key, value.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=False)
    # Imported fixture helpers live in the preserved release archive.
    # Keep Python from adding cache files to that read-only input tree.
    sys.dont_write_bytecode = True
    report = dict(status='running', cases=[], archive=str(args.release), source_sha256={},
                  runner_sha256=sha256(__file__), cli=str(args.cli), cli_sha256=sha256(args.cli), started_unix=time.time())
    try:
        scripts = {}
        def discover(name):
            if name in scripts:
                return
            assert '/' not in name and name.endswith('.py')
            path = args.release / name
            tree = ast.parse(path.read_text())
            scripts[name] = tree
            report['source_sha256'][str(path)] = sha256(path)
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'run_path':
                    assert isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str)
                    discover(Path(node.args[0].value).name)
        discover(ENTRY)
        assert len(scripts) == 14
        source_root = args.release / 'untracked-source'
        assert (source_root / 'catboost/metal/tests').is_dir()

        class Remap(ast.NodeTransformer):
            def visit_Constant(self, node):
                value = node.value
                if not isinstance(value, str) or not value.startswith('/'):
                    return node
                if value == OLD_ROOT:
                    target = source_root
                elif value == OLD_CLI:
                    target = args.cli
                elif value.startswith('/tmp/'):
                    assert len(Path(value).parts) == 3, f'unexpected temporary path: {value}'
                    target = args.output_dir / Path(value).name
                else:
                    raise AssertionError(f'unrecognized absolute script path: {value}')
                return ast.copy_location(ast.Constant(str(target)), node)

        for name, tree in scripts.items():
            transformed = ast.fix_missing_locations(Remap().visit(tree))
            (args.output_dir / name).write_text(ast.unparse(transformed) + '\n')
        seed_output = args.output_dir / 'catbooster-qce-cli'
        seed_output.mkdir()
        for name in ('learn.tsv', 'columns.cd'):
            source = args.release / SEED / name
            report['source_sha256'][str(source)] = sha256(source)
            shutil.copy2(source, seed_output / name)

        sys.path[:0] = [str(args.package), str(args.standalone)]
        import numpy as np
        import catboost
        assert Path(catboost.__file__).resolve().is_relative_to(args.package)
        report['package'] = str(Path(catboost.__file__).resolve())
        report['extension_sha256'] = sha256(Path(catboost.__file__).parent / '_catboost.so')
        original_run = subprocess.run
        original_fit = catboost.CatBoost._fit
        original_predict = catboost.CatBoost._predict

        def forbidden_fit(*values, **kwargs):
            raise AssertionError('the CLI replay must not fit through Python')

        def checked_prediction(model, *values, **kwargs):
            assert model.get_metadata()['metal_backend'] == 'METAL'
            result = original_predict(model, *values, **kwargs)
            assert np.isfinite(result).all(), 'nonfinite CLI model prediction'
            return result

        def checked_run(command, *values, **kwargs):
            assert isinstance(command, list) and command[:2] == [str(args.cli), 'fit']
            assert command[command.index('--task-type') + 1] == 'GPU'
            assert command[command.index('--allow-writing-files') + 1] == 'false'
            assert not kwargs.get('shell')
            for option in ('--model-file', '--learn-set', '--test-set', '--column-description'):
                if option in command:
                    assert Path(command[command.index(option) + 1]).resolve().is_relative_to(args.output_dir)
            result = original_run(command, *values, **kwargs)
            assert result.returncode == 0
            report['cases'].append(dict(model=command[command.index('--model-file') + 1], command=command))
            return result

        subprocess.run = checked_run
        catboost.CatBoost._fit = forbidden_fit
        catboost.CatBoost._predict = checked_prediction
        try:
            runpy.run_path(str(args.output_dir / ENTRY), run_name='__main__')
        finally:
            subprocess.run = original_run
            catboost.CatBoost._fit = original_fit
            catboost.CatBoost._predict = original_predict
        assert len(report['cases']) == 153, f"expected 153 CLI fits, received {len(report['cases'])}"
        assert all(sha256(path) == digest for path, digest in report['source_sha256'].items())
        report['status'] = 'passed'
    except BaseException:
        report['status'] = 'failed'
        report['error'] = traceback.format_exc()
        raise
    finally:
        report['elapsed_seconds'] = time.time() - report['started_unix']
        (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{report['status']}: {len(report['cases'])} prior CLI variants; {args.output_dir / 'report.json'}", flush=True)


if __name__ == '__main__':
    main()
