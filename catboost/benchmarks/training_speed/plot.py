import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt

from log_parser import read_results

FONT_DICT = {'fontsize': 20}
FIGURE_SIZE = (10, 5)


def plot_time_per_iter(tracks, figsize=FIGURE_SIZE, title=None, save_path='time_per_iter.png'):
    fig = plt.figure(figsize=figsize)

    time_per_iters = []
    algs = tracks.keys()

    for alg_name in algs:
        time_per_iter_alg = []
        for track in tracks[alg_name]:
            # aggregating statistic over different tracks
            time_per_iter = track.get_time_per_iter()
            time_per_iter_alg.extend(time_per_iter)

        time_per_iters.append(time_per_iter_alg)

    if title is not None:
        plt.title(title, FONT_DICT)

    for i, alg_name in enumerate(algs):
        print(alg_name)
        print(np.median(time_per_iters[i]))

    plt.ylabel('Seconds', FONT_DICT)
    plt.boxplot(time_per_iters, labels=algs)

    if os.path.exists(save_path):
        print('WARNING: file ' + save_path + ' already exists')

    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_quality(tracks, from_iter, to_iter, figsize=FIGURE_SIZE, title=None, save_path='quality.png'):
    fig = plt.figure(figsize=figsize)

    if title is not None:
        plt.title(title, FONT_DICT)

    flat_tracks = []
    for alg in tracks.keys():
        flat_tracks += tracks[alg]

    first_track = flat_tracks[0]
    task_type = first_track.task_type
    metric = 'Error' if task_type == 'Classification' or task_type == 'Multiclass' else 'RMSE'

    plt.xlabel('iteration', FONT_DICT)
    plt.ylabel(metric, FONT_DICT)

    lines = []
    names = []

    for track in flat_tracks:
        _, values = track.get_series()

        cur_to_iter = to_iter
        if to_iter is None or to_iter > track.get_fit_iterations():
            cur_to_iter = track.get_fit_iterations()

        values = values[from_iter:cur_to_iter]
        x_values = np.arange(from_iter, cur_to_iter)

        line, = plt.plot(x_values, values)
        lines.append(line)
        names.append(str(track))

    plt.legend(lines, names, prop={'size': 9})

    if os.path.exists(save_path):
        print('WARNING: file ' + save_path + ' already exists')

    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def plot_quality_vs_time(tracks, best_quality, low_percent=0.8, num_bins=100, only_min=False,
                         figsize=FIGURE_SIZE, title=None, save_path='time_distr.png'):
    fig = plt.figure(figsize=figsize)

    if title is not None:
        plt.title(title, FONT_DICT)

    plt.xlabel('Quality (%)', FONT_DICT)
    plt.ylabel('Time to obtain (sec)', FONT_DICT)

    algs = tracks.keys()
    up_percent = 1. - low_percent

    for i, alg_name in enumerate(algs):
        bins = [[] for j in range(num_bins + 1)]

        for track in tracks[alg_name]:
            time_series, values = track.get_series()
            time_series = time_series - time_series[0]

            for time, value in zip(time_series, values):
                percent = value / best_quality - 1.

                if percent > up_percent:
                    continue

                idx = int(np.round(num_bins * percent / up_percent))
                bins[idx].append(time)

        time_median = []
        time_q2 = []
        time_min = []
        x_values = []

        for k, times in enumerate(bins):
            if len(times) > 0:
                time_median.append(np.median(times))
                time_q2.append(np.quantile(times, 0.75))
                time_min.append(np.min(times))

                x_values.append(float(k) / num_bins * up_percent)

        cur_min = time_min[0]
        for t in range(1, len(time_min)):
            if time_min[t] > cur_min:
                time_min[t] = cur_min
            else:
                cur_min = time_min[t]

        error_plus = np.array(time_q2) - np.array(time_median)
        error_minus = np.array(time_median) - np.array(time_min)

        x_values = np.array(x_values) - (float(i) - 1.) * up_percent / num_bins / 4.
        x_values = 1. - x_values

        if only_min:
            plt.plot(x_values, time_min, label=alg_name)
        else:
            plt.errorbar(x=x_values, y=time_median, yerr=[error_minus, error_plus], fmt='o-', barsabove=True,
                         capsize=2, linewidth=2, label=alg_name)

    plt.legend(fontsize='large')

    if os.path.exists(save_path):
        print('WARNING: file ' + save_path + ' already exists')

    plt.savefig(save_path, dpi=100)
    plt.close(fig)


def params_to_str(params):
    return ''.join(map(lambda (key, value): '{}{}'.format(key, str(value)), params.items()))


def get_best(tracks, top=1):
    algorithms = tracks.keys()
    best_tracks = {}

    for algorithm_name in algorithms:
        best_scores = map(lambda track: track.get_best_score(), tracks[algorithm_name])
        idx_best = np.argsort(best_scores)[:top]
        best_tracks[algorithm_name] = map(lambda idx: tracks[algorithm_name][idx], idx_best)

    return best_tracks


def filter_tracks(tracks, params_cases):
    filtered_tracks = {}

    for alg in tracks.keys():
        filtered_tracks[alg] = []

        for track in tracks[alg]:
            if all([track.params_dict[param_name] in params_cases[param_name] for param_name in params_cases.keys()]):
                filtered_tracks[alg].append(track)

    return filtered_tracks


ONLY_TYPES = {
    'cat-cpu': ['catboost-CPU'],
    'xgb-cpu': ['xgboost-CPU'],
    'lgb-cpu': ['lightgbm-CPU'],
    'cat-gpu': ['catboost-GPU'],
    'xgb-gpu': ['xgboost-GPU'],
    'lgb-gpu': ['lightgbm-GPU'],
    'cat': ['catboost-CPU', 'catboost-GPU'],
    'xgb': ['xgboost-CPU', 'xgboost-GPU'],
    'lgb': ['lightgbm-CPU', 'lightgbm-GPU'],
    'cpu': ['catboost-CPU', 'xgboost-CPU', 'lightgbm-CPU'],
    'gpu': ['catboost-GPU', 'xgboost-GPU', 'lightgbm-GPU']
}


def get_default_file_name(plot_type, params):
    default_file_names = {
        'best': 'best_quality.png',
        'quality-vs-time': 'quality_vs_time.png',
        'time-per-iter': 'time_per_iter.png'
    }

    if plot_type in default_file_names.keys():
        return default_file_names[plot_type]

    if plot_type == 'custom':
        return params_to_str(params) + '.png'


def plot_experiment(tracks, experiment_name, args):
    file_name = args.file_name if args.file_name else get_default_file_name(args.type, args.params_cases)
    save_dir = os.path.join(args.out_dir, experiment_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, file_name)

    if args.only:
        filtered_tracks = {}

        for only_type in args.only:
            for alg_name in ONLY_TYPES[only_type]:
                filtered_tracks[alg_name] = tracks[alg_name]

        tracks = filtered_tracks

    if args.params_cases:
        with open(args.params_cases) as f:
            params_cases = json.load(f)

        tracks = filter_tracks(tracks, params_cases)

    if args.type == 'quality-vs-time':
        best_tracks = get_best(tracks)
        best_quality = min(map(lambda tracks: tracks[0].get_best_score(), best_tracks.values()))
        print(best_quality)

        if args.top:
            tracks = get_best(tracks, top=args.top)

        plot_quality_vs_time(tracks, best_quality=best_quality, low_percent=args.low_percent, only_min=args.only_min,
                             figsize=args.fig_size, num_bins=args.num_bins, save_path=save_path)

    if args.type == 'best':
        best_tracks = get_best(tracks, top=args.top)
        for alg in best_tracks.keys():
            for track in best_tracks[alg]:
                print(track)
                print(track.get_best_score())

        plot_quality(best_tracks, args.from_iter, args.to_iter, figsize=args.fig_size,
                     title=args.title, save_path=save_path)

    if args.type == 'custom':
        plot_quality(tracks, args.from_iter, args.to_iter,
                     figsize=args.fig_size, title=args.title, save_path=save_path)

    if args.type == 'time-per-iter':
        plot_time_per_iter(tracks, figsize=args.fig_size, title=args.title, save_path=save_path)


def main():
    plot_functions = {
        'time-per-iter': plot_time_per_iter,
        'best': plot_quality,
        'quality-vs-time': plot_quality_vs_time,
        'custom': plot_quality
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', choices=plot_functions.keys(), required=True)
    parser.add_argument('--only', nargs='+', choices=ONLY_TYPES.keys(), required=False)
    parser.add_argument('-i', '--results-file', required=True)
    parser.add_argument('-t', '--title')
    parser.add_argument('-f', '--fig-size', nargs=2, type=int, default=FIGURE_SIZE)
    parser.add_argument('-o', '--out-dir', default='plots')
    parser.add_argument('--params-cases', help='draw plots only with those params (tracks filtering)'
                                               ' path to json file, each line corresponds to learner '
                                               'parameter (e.g. max_depth) and list of its values')
    parser.add_argument('--from-iter', type=int, default=0, help='only custom, best modes')
    parser.add_argument('--to-iter', type=int, default=None, help='only custom, best modes')
    parser.add_argument('--low-percent', type=float, default=0.9, help='only quality-vs-time mode')
    parser.add_argument('--num-bins', type=int, default=200, help='only quality-vs-time mode')
    parser.add_argument('--only-min', action='store_true', help='only quality-vs-time mode')
    parser.add_argument('--top', type=int, default=3, help='only best mode')
    args = parser.parse_args()

    tracks = read_results(args.results_file)

    for experiment_name in tracks:
        plot_experiment(tracks[experiment_name], experiment_name, args)


if __name__ == '__main__':
    main()
