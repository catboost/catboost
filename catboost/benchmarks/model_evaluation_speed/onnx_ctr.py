"""Compare native and ONNX CTR inference, including category preparation.

Requires the CatBoost build under test, numpy, onnx, and onnxruntime (or
onnxruntime-gpu). The input NPZ contains two non-object arrays: ``categorical``
(strings/integers) and ``numerical`` (float32), ordered as in the model within
each feature type. Omit ``numerical`` for categorical-only models.

Example:
    python onnx_ctr.py --model model.cbm --data inputs.npz --provider CUDA

An optional --baseline-onnx file compares an earlier export of the same model.
Session creation is excluded; CUDA sessions explicitly allow CPU fallback.
"""

import argparse
import json
import time

import numpy as np
import onnxruntime as ort

from catboost import CatBoost, Pool
from catboost.utils import convert_to_onnx_object, get_cat_feature_hash, get_cat_feature_hashes


def measure(function, repetitions):
    for _ in range(3):
        function()
    samples = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        function()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {'median_ms': float(np.median(samples)), 'p90_ms': float(np.percentile(samples, 90))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--baseline-onnx')
    parser.add_argument('--provider', choices=['CPU', 'CUDA'], default='CPU')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--repetitions', type=int, default=21)
    parser.add_argument('--batch-sizes', type=int, nargs='+', default=[1, 1024, 16384])
    args = parser.parse_args()
    if min(args.threads, args.repetitions, *args.batch_sizes) <= 0:
        parser.error('threads, repetitions, and batch sizes must be positive')

    model = CatBoost().load_model(args.model)
    cat_indices = model.get_cat_feature_indices()
    num_indices = [i for i in range(len(model.feature_names_)) if i not in cat_indices]
    with np.load(args.data, allow_pickle=False) as inputs:
        cats = inputs['categorical']
        nums = inputs['numerical'] if num_indices else np.empty((len(cats), 0), dtype=np.float32)
    if (cats.ndim != 2 or nums.ndim != 2 or len(cats) == 0 or len(cats) != len(nums)
            or cats.shape[1] != len(cat_indices) or nums.shape[1] != len(num_indices)):
        parser.error('input arrays must have matching nonempty rows and model feature counts')

    provider = args.provider + 'ExecutionProvider'
    if args.provider == 'CUDA' and hasattr(ort, 'preload_dlls'):
        ort.preload_dlls()
    if provider not in ort.get_available_providers():
        parser.error(provider + ' is unavailable')
    options = ort.SessionOptions()
    options.intra_op_num_threads = args.threads
    providers = [provider] if args.provider == 'CPU' else [provider, 'CPUExecutionProvider']
    proto = convert_to_onnx_object(model, {'onnx_cat_features_input_type': 'hashed'})
    sessions = {'current': ort.InferenceSession(proto.SerializeToString(), options, providers=providers)}
    if args.baseline_onnx:
        sessions['baseline'] = ort.InferenceSession(args.baseline_onnx, options, providers=providers)
    for session in sessions.values():
        assert session.get_providers()[0] == provider, session.get_providers()
    classifier = sessions['current'].get_outputs()[0].name == 'label'
    prediction_type = 'Probability' if classifier else 'RawFormulaVal'

    print(json.dumps({'onnxruntime': ort.__version__, 'provider': provider, 'cpu_threads': args.threads,
                      'repetitions': args.repetitions, 'warmups': 3, 'nodes': len(proto.graph.node),
                      'model_bytes': proto.ByteSize()}), flush=True)
    for batch in args.batch_sizes:
        rows = np.arange(batch) % len(cats)
        batch_cats, batch_nums = cats[rows], nums[rows].astype(np.float32)
        data = np.empty((batch, len(model.feature_names_)), dtype=object)
        data[:, cat_indices] = batch_cats
        data[:, num_indices] = batch_nums

        def make_pool():
            return Pool(data, cat_features=cat_indices, thread_count=args.threads)

        def make_feed(session, bulk):
            result = {}
            for value in session.get_inputs():
                if value.name == 'num_features':
                    result[value.name] = batch_nums
                else:
                    if value.name in model.feature_names_:
                        flat_index = model.feature_names_.index(value.name)
                    else:
                        flat_index = int(value.name[len('cat_feature_'):])
                    column = batch_cats[:, cat_indices.index(flat_index)]
                    hashed = get_cat_feature_hashes(column) if bulk else np.array(
                        [get_cat_feature_hash(v) for v in column], dtype=np.int64)
                    result[value.name] = hashed.reshape(-1, 1)
            return result

        pool = make_pool()
        expected = model.predict(pool, prediction_type=prediction_type, thread_count=args.threads)
        result = {'batch': batch}
        result['native_prepared'] = measure(
            lambda: model.predict(pool, prediction_type=prediction_type, thread_count=args.threads), args.repetitions)
        result['native_with_preparation'] = measure(
            lambda: model.predict(make_pool(), prediction_type=prediction_type, thread_count=args.threads), args.repetitions)
        for name, session in sessions.items():
            bulk = name == 'current'
            feed = make_feed(session, bulk)
            actual = session.run(None, feed)
            if classifier:
                np.testing.assert_array_equal(actual[0], model.predict(pool, prediction_type='Class').ravel())
                actual = np.array([[row[label] for label in model.classes_] for row in actual[1]])
            else:
                actual = actual[0].ravel()
            np.testing.assert_allclose(actual, expected, atol=2e-4, rtol=2e-6)
            result[name + '_max_error'] = float(np.max(np.abs(actual - expected)))
            result[name + '_prepared'] = measure(lambda: session.run(None, feed), args.repetitions)
            result[name + '_with_preparation'] = measure(
                lambda: session.run(None, make_feed(session, bulk)), args.repetitions)
        print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
