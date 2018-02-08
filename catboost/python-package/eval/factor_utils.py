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
    def factors_to_ranges_string(factors_set):
        import numpy as np
        if factors_set is None or len(factors_set) == 0:
            return "None"
        factors = np.sort(np.array(factors_set))
        return ':'.join(
            [FactorUtils.single_range_to_string(min(x), max(x)) for x in
             np.split(factors, np.array(np.where(np.diff(factors) > 1)[0]) + 1)])
