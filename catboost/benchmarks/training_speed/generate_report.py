# coding=utf-8
import argparse
import json

import numpy as np
import pandas as pd

from log_parser import read_results


def calculate_statistics(tracks, niter):
    niter -= 1

    best_track = None
    best_quality = np.inf
    best_iter = -1

    median = []
    low = []
    high = []
    total = []

    for track in tracks:
        cur_quality = track.get_best_score()
        time_per_iter = track.get_time_per_iter()
        if not len(time_per_iter) or time_per_iter.shape[0] < niter:
            continue

        median.append(np.median(time_per_iter))
        low.append(np.quantile(time_per_iter, 0.25))
        high.append(np.quantile(time_per_iter, 0.75))
        time_series = track.time_series
        total.append(time_series[niter] - time_series[0])

        if best_quality > cur_quality:
            best_quality = cur_quality
            best_iter = np.argmin(track.get_series()[1])
            best_track = track

    if best_track is None:
        return {}

    print(best_track)

    median = sum(median) / len(median)
    low = sum(low) / len(low)
    high = sum(high) / len(high)
    dev = max(median - low, high - median)
    total = np.median(total)

    return {
        "Best": best_quality,
        "Iter": best_iter,
        "MedianTime": median,
        "Deviation": dev,
        "TotalTime": total
    }


def get_experiment_stats(results_file, gpu, niter):
    stats = {}
    tracks = read_results(results_file)

    for experiment_name in tracks:
        stats[experiment_name] = {}

        experiment_tracks = tracks[experiment_name]
        experiment_tracks = dict(filter(lambda track: gpu == ('GPU' in track[0]), experiment_tracks.items()))

        for algorithm_name in experiment_tracks:
            stats[experiment_name][algorithm_name] = {}
            table_tracks = split_tracks(experiment_tracks[algorithm_name])

            for params, cur_tracks in table_tracks.items():
                stat = calculate_statistics(cur_tracks, niter)
                if stat == {}:
                    continue
                stats[experiment_name][algorithm_name][params] = stat

    return stats


def get_table_header(experiment_stats):
    parameter_set = None

    for algorithm_name in experiment_stats:
        alg_parameter_set = set(experiment_stats[algorithm_name].keys())
        if parameter_set is None:
            parameter_set = alg_parameter_set
        else:
            parameter_set &= alg_parameter_set

    return sorted(list(parameter_set))


def get_median_str(stat):
    median = np.round(stat["MedianTime"], 3)
    dev = np.round(stat["Deviation"], 3)

    median_str = str(median)
    if abs(dev) > 0:
        median_str += u' +/- ' + str(dev)

    return median_str


def print_all_in_one_table(stats, gpu, params, output):
    median_table = []
    total_table = []

    index = ["catboost", "xgboost", "lightgbm"]

    for algorithm_name in index:
        median_row = []
        total_row = []

        if gpu:
            algorithm_name += "-GPU"
        else:
            algorithm_name += "-CPU"

        for experiment_name in stats:
            experiment_stats = stats[experiment_name]

            if algorithm_name not in experiment_stats:
                continue
            if params not in experiment_stats[algorithm_name]:
                median_row.append(0.)
                total_row.append(0.)
                continue

            cur_stat = experiment_stats[algorithm_name][params]
            median_row.append(get_median_str(cur_stat))
            total_row.append(np.round(cur_stat["TotalTime"], 3))

        median_table.append(median_row)
        total_table.append(total_row)

    median_table = pd.DataFrame(median_table, index=index, columns=stats.keys())
    total_table = pd.DataFrame(total_table, index=index, columns=stats.keys())

    with open(output, 'w') as f:
        f.write('Median time per iter, sec')
        f.write('\n')
        f.write(median_table.to_string())
        f.write('\n')
        f.write('Total time, sec')
        f.write('\n')
        f.write(total_table.to_string())
        f.write('\n')


def print_experiment_table(stats, output):
    for experiment_name in stats:
        experiment_stats = stats[experiment_name]

        header = get_table_header(experiment_stats)

        median_table = []
        total_table = []

        for algorithm_name in experiment_stats:
            algorithm_stats = experiment_stats[algorithm_name]
            median_row = []
            total_row = []

            for parameter in header:
                cur_stat = algorithm_stats[parameter]

                total = np.round(cur_stat["TotalTime"], 3)

                median_row.append(get_median_str(cur_stat))
                total_row.append(total)

            median_table.append(median_row)
            total_table.append(total_row)

        index = experiment_stats.keys()
        median_table = pd.DataFrame(median_table, index=index, columns=header)
        total_table = pd.DataFrame(total_table, index=index, columns=header)

        with open(experiment_name + output, 'w') as f:
            f.write('Median time per iter, sec')
            f.write('\n')
            f.write(median_table.to_string())
            f.write('\n')
            f.write('Total time, sec')
            f.write('\n')
            f.write(total_table.to_string())
            f.write('\n')


def split_tracks(tracks):
    depths = []
    samples = []

    for track in tracks:
        depths.append(track.params.max_depth)

        if "subsample" not in track.params_dict.keys():
            samples.append(1.0)
            continue

        samples.append(track.params.subsample)

    depths = set(depths)
    samples = set(samples)

    table_tracks = {(depth, subsample): [] for depth in depths for subsample in samples}

    for track in tracks:
        subsample = track.params.subsample if "subsample" in track.params_dict.keys() else 1.
        table_tracks[(track.params.max_depth, subsample)].append(track)

    return table_tracks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--result', default='./results.json')
    parser.add_argument('-o', '--output')
    parser.add_argument('-t', '--type', choices=['common-table', 'by-depth-table', 'json'], default='common-table')
    parser.add_argument('-f', '--filter', choices=['only-gpu', 'only-cpu'], default='only-gpu')
    parser.add_argument('-p', '--params', default=(6.0, 1.0))
    parser.add_argument('-niter', type=int, default=999)
    args = parser.parse_args()

    on_gpu = args.filter == 'only-gpu'
    stats = get_experiment_stats(args.result, on_gpu, niter=args.niter)

    output = args.output
    if args.output is None:
        output = args.type + '.txt'

    if args.type == 'common-table':
        print_all_in_one_table(stats, on_gpu, params=args.params, output=output)
        return

    if args.type == 'by-depth-table':
        print_experiment_table(stats, output)
        return

    if args.type == 'json':
        while len(stats.keys()) == 1:
            stats = stats.values()[0]

        with open(output, 'w') as f:
            json.dump(stats, f)

        return


if __name__ == "__main__":
    main()
