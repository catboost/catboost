import argparse
import sys, os
import base64
import hashlib
import tempfile
import shutil
import subprocess
from collections import namedtuple
from unittest.mock import patch


def call(cmd, cwd=None, env=None):
    return subprocess.check_output(cmd, stdin=None, stderr=sys.stderr, cwd=cwd, env=env)


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

        self.SECTION_NAME = '__DATA,__res_holder' if 'apple' in args.target else '.rodata.__res_holder'

    def __enter__(self):
        return self

    def __exit__(self, *_):
        shutil.rmtree(self.tmpdir)

    PackedArgs = namedtuple("PackedArgs", ["infile_path", "comressed_file_path", "symbol_name"])

    def _compress_files(self, infos: list[PackedArgs]) -> list[int]:
        if len(infos) == 0:
            return [0]
        # Compress resources
        cmd = [self.compressor, "--compress-only"]
        for inserted_file, compressed_file, _ in infos:
            cmd.extend([inserted_file, compressed_file])
        call(cmd)

        # Return sizes of compressed files
        return [os.path.getsize(zipped_file) for (_, zipped_file, _) in infos]

    @staticmethod
    def flat_merge_obj(outs, output):
        if len(outs) == 1:
            shutil.move(outs[0], output)
            return

        with open(output, 'wb') as fout:
            for fname in outs:
                with open(fname, 'rb') as fin:
                    shutil.copyfileobj(fin, fout)
        return

    def _insert_files(self, infos: list[PackedArgs], output_obj: str) -> None:
        if len(infos) == 0:
            return

        # Merge files into one
        compressed_files = [compressed_file for _, compressed_file, _ in infos]
        merged_output = os.path.join(self.tmpdir, 'merged.zstd')
        self.flat_merge_obj(compressed_files, merged_output)

        # Update section content
        cmd = [self.objcopy, f'--update-section={self.SECTION_NAME}={merged_output}', output_obj]
        call(cmd)

    def _gen_prefix(self, infos: list[PackedArgs], f_sizes: list[int]) -> str:
        TOTAL_SIZE = sum(f_sizes)
        if len(infos) == 0 or TOTAL_SIZE == 0:
            return ''

        syms = []
        for packed_args in infos:
            syms += [sym for (_, _, sym) in packed_args]

        cumulative_pos = [0] * (len(f_sizes) + 1)
        for i, sz in enumerate(f_sizes):
            cumulative_pos[i + 1] = cumulative_pos[i] + sz
        cumulative_pos.pop()

        SECTION_ATTR = f'__attribute__((section("{self.SECTION_NAME}")))'
        HOLDER_VARIABLE = 'resource_holder'
        generated_return = f'static {SECTION_ATTR} const char {HOLDER_VARIABLE}[{TOTAL_SIZE}] = {{0}};\n'
        template = '#define {sym}      ({var} + {begin_offset})\n'
        template += '#define {sym}_end  ({var} + {end_offset})\n'
        for sym, start_offset, size in zip(syms, cumulative_pos, f_sizes):
            generated_return += template.format(
                sym=sym, begin_offset=start_offset, end_offset=start_offset + size, var=HOLDER_VARIABLE
            )

        return generated_return

    def insert_resources(self, kv_files, kv_strings):
        kv_files = list(kv_files)

        # Step 1: Compress files and save its sizes
        file_sizes = []
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
            file_sizes += self._compress_files(packed_args)

        # Introduce custom tmp_filename generator for keeping object-file hash the same
        def tmp_file_generator():
            idx = 0
            base = os.path.basename(self.obj_out)
            while True:
                yield f'{base}_{str(idx)}'
                idx += 1

        # Step 2: Generate resource registration cpp code & compile it
        with patch.object(tempfile, "_get_candidate_names", tmp_file_generator), tempfile.NamedTemporaryFile(
            suffix='.cc'
        ) as dummy_src:
            cmd = [self.rescompiler, dummy_src.name, '--use-sections']
            for path, key in kv_files + list(('-', k) for k in kv_strings):
                if path != '-':
                    path = self.mangler(key)
                cmd.extend([path, key])
            call(cmd)

            # Generate code for real char arrays, that will hold the resource
            generated_source_text = ''
            with open(dummy_src.name, 'rt') as f:
                generated_source_text = f.read()
            generated_source_text = generated_source_text.replace('extern "C" const char', '// extern "C" const char')
            generated_source_text = self._gen_prefix(infos, file_sizes) + generated_source_text
            with open(dummy_src.name, 'wt') as f:
                f.write(generated_source_text)

            # Compile
            call([self.cxx, dummy_src.name, *self.target_flags, '-c', '-o', self.obj_out])

        # Step 3: Put files into object
        if sum(file_sizes) > 0:
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
