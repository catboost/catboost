import argparse
import sys, os
import base64
import hashlib
import tempfile
import shutil
import subprocess
from collections import namedtuple


def call(cmd, cwd=None, env=None):
    return subprocess.check_output(cmd, stdin=None, stderr=subprocess.STDOUT, cwd=cwd, env=env)


class LLVMResourceInserter:
    def __init__(self, args, obj_output):
        self.cxx = args.compiler
        self.objcopy = args.objcopy
        self.rescompiler = args.rescompiler
        self.compressor = args.compressor
        self.obj_out = obj_output

        has_target = args.target or args.target != '__unknown_target__'
        self.target_flags = [f"--target={args.target}"] if has_target else []

        seed = os.path.basename(obj_output)
        self.mangler = lambda key: 'R' + hashlib.sha256((seed + key).encode()).hexdigest()

        self.tmpdir = tempfile.mkdtemp()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        shutil.rmtree(self.tmpdir)

    PackedArgs = namedtuple("PackedArgs", ["infile_path", "comressed_file_path", "symbol_name"])

    def _insert_files(self, infos: list[PackedArgs], output_obj: str) -> None:
        if len(infos) == 0:
            return

        # Compress resources
        cmd = [self.compressor, "--compress-only"]
        for inserted_file, compressed_file, _ in infos:
            cmd.extend([inserted_file, compressed_file])
        call(cmd)

        # Put resources into .o file
        infos = [(zipped_file, os.path.getsize(zipped_file), symbol_name) for (_, zipped_file, symbol_name) in infos]
        cmd = [self.objcopy]
        # NOTE: objcopy does not distinguish the order of arguments.
        for fname, fsize, sym in infos:
            section_name = f'.rodata.{sym}'
            cmd.extend(
                [
                    f'--add-section={section_name}={fname}',
                    f'--set-section-flags={section_name}=readonly',
                    f'--add-symbol={sym}={section_name}:0,global',
                    f'--add-symbol={sym}_end={section_name}:{fsize},global',
                ]
            )
        cmd.extend([output_obj])
        call(cmd)

    @staticmethod
    def flat_merge_cpp(outs, output):
        if len(outs) == 1:
            shutil.move(outs[0], output)
            return

        with open(output, 'w') as fout:
            for fname in outs:
                with open(fname, 'r') as fin:
                    shutil.copyfileobj(fin, fout)
        return

    def insert_resources(self, kv_files, kv_strings):
        kv_files = list(kv_files)

        # Generate resource registration cpp code & compile it
        with tempfile.NamedTemporaryFile(suffix='.cc') as dummy_src:
            cmd = [self.rescompiler, dummy_src.name, '--use-sections']
            for path, key in kv_files + list(('-', k) for k in kv_strings):
                if path != '-':
                    path = self.mangler(key)
                cmd.extend([path, key])
            call(cmd)

            # Compile
            call([self.cxx, dummy_src.name, *self.target_flags, '-c', '-o', self.obj_out])

        # Put files
        infos = [[]]
        estimated_cmd_len = 0
        LIMIT = 6000
        for idx, (path, key) in enumerate(kv_files):
            packed_args = (path, os.path.join(self.tmpdir, f'{idx}.zstd'), self.mangler(key))
            infos[-1].append(packed_args)
            estimated_cmd_len += len(path)
            if estimated_cmd_len > LIMIT:
                infos.append([])
                estimated_cmd_len = 0
        for packed_args in infos:
            self._insert_files(packed_args, self.obj_out)

        return self.obj_out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', required=True)
    parser.add_argument('--objcopy', required=True)
    parser.add_argument('--compressor', required=True)
    parser.add_argument('--rescompiler', required=True)
    parser.add_argument('--output_obj', required=True)
    parser.add_argument('--inputs', nargs='+', required=False, default=[])
    parser.add_argument('--keys', nargs='+', required=False, default=[])
    parser.add_argument('--kvs', nargs='+', required=False, default=[])
    parser.add_argument('--target', required=True)

    args = parser.parse_args()

    # Decode hex to original string
    args.keys = list(base64.b64decode(it).decode("utf-8") for it in args.keys)

    return args, args.inputs, args.keys


def main():
    args, inputs, keys = parse_args()

    with LLVMResourceInserter(args, args.output_obj) as inserter:
        inserter.insert_resources(zip(inputs, keys), args.kvs)
    return


if __name__ == '__main__':
    main()
