from enum import Enum


class LabelMode(Enum):
    AddFeature = "Feature"
    IgnoreFeature = "Ignore"


class FactorUtils(object):

    @staticmethod
    def extract_factors(string_factors):
        string_factors = string_factors.strip()
        if not string_factors:
            return {}

        list_factors = []
        eval_factor_units = string_factors.split(':')
        for eval_factor_unit in eval_factor_units:
            try:
                factor = int(eval_factor_unit)
                list_factors.append(factor)
            except ValueError:
                string_bounds = eval_factor_unit.split('-')
                if len(string_bounds) != 2:
                    raise AttributeError('Range need to contain exactly two numbers!')
                begin_range = int(string_bounds[0])
                end_range = int(string_bounds[1])
                list_factors += list(range(begin_range, end_range + 1))

        return set(list_factors)

    @staticmethod
    def factors_to_string(factors):
        if len(factors) == 0:
            return ''
        parts = []
        factors_list = sorted(factors)
        begin = factors_list[0]
        for i in range(1, len(factors_list)):
            if factors_list[i] != factors_list[i - 1] + 1:
                end = factors_list[i - 1]
                if begin != end:
                    parts.append('{}-{}'.format(begin, end))
                else:
                    parts.append(str(begin))
                begin = factors_list[i]
        end = len(factors_list) - 1
        if begin != factors_list[end]:
            parts.append('{}-{}'.format(begin, factors_list[end]))
        else:
            parts.append(str(begin))
        return ':'.join(parts)

    @staticmethod
    def compress_string_factors(string_factors):
        factors = FactorUtils.extract_factors(string_factors)
        compressed_string_factors = FactorUtils.factors_to_string(factors)
        return compressed_string_factors

    @staticmethod
    def single_range_to_string(left, right):
        if left != right:
            return "{}-{}".format(left, right)
        else:
            return "{}".format(left)

    @staticmethod
    def group_factors_by_range(factors_set):
        factors = sorted(list(factors_set))
        is_start = []
        for i in range(0, len(factors)):
            is_start.append(i == 0 or (factors[i] != factors[i - 1] + 1))

        grouped_factors = []
        i = 0
        while i < len(factors):
            if is_start[i]:
                grouped_factors.append([])
            grouped_factors[-1].append(factors[i])
            i += 1
        return grouped_factors

    @staticmethod
    def factors_to_ranges_string(factors_set):
        if factors_set is None or len(factors_set) == 0:
            return "None"
        grouped_factors = FactorUtils.group_factors_by_range(factors_set)

        return ':'.join([FactorUtils.single_range_to_string(min(x), max(x)) for x in grouped_factors])

    @staticmethod
    def create_label(all_eval_features, removed_features, label_mode):
        eval_features = set(all_eval_features)
        if label_mode == LabelMode.AddFeature:
            add_features = eval_features - set(removed_features)
            return "Features: {}".format(FactorUtils.factors_to_ranges_string(add_features))
        else:
            return "Ignore: {}".format(FactorUtils.factors_to_ranges_string(set(removed_features)))
