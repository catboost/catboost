import argparse
import datetime
import os
import subprocess
import sys
import tarfile


def extract_kindexes(kindexes):
    for kindex in kindexes:
        with tarfile.TarFile(kindex) as tf:
            for fname in [i for i in tf.getnames() if i.endswith('.kzip')]:
                tf.extract(fname)
                yield fname


def just_do_it(java, kythe, kythe_to_proto, out_name, binding_only, kindexes):
    temp_out_name = out_name + '.temp'
    kindex_inputs = list(extract_kindexes(kindexes))
    open(temp_out_name, 'w').close()
    start = datetime.datetime.now()
    for kindex in kindex_inputs:
        print >> sys.stderr, '[INFO] Processing:', kindex
        indexer_start = datetime.datetime.now()
        p = subprocess.Popen([java, '-jar', os.path.join(kythe, 'indexers/java_indexer.jar'), kindex], stdout=subprocess.PIPE)
        indexer_out, _ = p.communicate()
        print >> sys.stderr, '[INFO] Indexer execution time:', (datetime.datetime.now() - indexer_start).total_seconds(), 'seconds'
        if p.returncode:
            raise Exception('java_indexer failed with exit code {}'.format(p.returncode))
        dedup_start = datetime.datetime.now()
        p = subprocess.Popen([os.path.join(kythe, 'tools/dedup_stream')], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        dedup_out, _ = p.communicate(indexer_out)
        print >> sys.stderr, '[INFO] Dedup execution time:', (datetime.datetime.now() - dedup_start).total_seconds(), 'seconds'
        if p.returncode:
            raise Exception('dedup_stream failed with exit code {}'.format(p.returncode))
        entrystream_start = datetime.datetime.now()
        p = subprocess.Popen([os.path.join(kythe, 'tools/entrystream'), '--write_json'], stdin=subprocess.PIPE, stdout=open(temp_out_name, 'a'))
        p.communicate(dedup_out)
        if p.returncode:
            raise Exception('entrystream failed with exit code {}'.format(p.returncode))
        print >> sys.stderr, '[INFO] Entrystream execution time:', (datetime.datetime.now() - entrystream_start).total_seconds(), 'seconds'
    preprocess_start = datetime.datetime.now()
    subprocess.check_call([kythe_to_proto, '--preprocess-entry', '--entries', temp_out_name, '--out', out_name] + (['--only-binding-data'] if binding_only else []))
    print >> sys.stderr, '[INFO] Preprocessing execution time:', (datetime.datetime.now() - preprocess_start).total_seconds(), 'seconds'
    print >> sys.stderr, '[INFO] Total execution time:', (datetime.datetime.now() - start).total_seconds(), 'seconds'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--java", help="java path")
    parser.add_argument("--kythe", help="kythe path")
    parser.add_argument("--kythe-to-proto", help="kythe_to_proto tool path")
    parser.add_argument("--out-name", help="entries json out name")
    parser.add_argument("--binding-only", action="store_true", default=False, help="filter only binding data")
    parser.add_argument("kindexes", nargs='*')
    args = parser.parse_args()
    just_do_it(args.java, args.kythe, args.kythe_to_proto, args.out_name, args.binding_only, args.kindexes)
