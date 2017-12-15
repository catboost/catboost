import _test_const as consts


def check_cpu(suite_cpu_requirements, test_size):
    min_cpu_requirements = consts.TestRequirementsConstants.MinCpu
    max_cpu_requirements = consts.TestSize.get_max_requirements(test_size).get(consts.TestRequirements.Cpu)
    if isinstance(suite_cpu_requirements, str) and consts.TestRequirementsConstants.is_all_cpu(suite_cpu_requirements):
        if not consts.TestRequirementsConstants.is_all_cpu(max_cpu_requirements):
            return ["Wrong 'cpu' requirements: {}, should be in [{}..{}] for {}-size tests".format(suite_cpu_requirements, min_cpu_requirements, max_cpu_requirements, test_size)]
        return []

    if not isinstance(suite_cpu_requirements, int):
        return ["Wrong 'cpu' requirements: {}, should be integer".format(suite_cpu_requirements)]

    if suite_cpu_requirements < min_cpu_requirements or suite_cpu_requirements > consts.TestRequirementsConstants.get_cpu_value(max_cpu_requirements):
        return ["Wrong 'cpu' requirement: {}, should be in [{}..{}] for {}-size tests".format(suite_cpu_requirements, min_cpu_requirements, max_cpu_requirements, test_size)]

    return []


def is_power_of_two(num):
    return num > 0 and ((num & (num - 1)) == 0)


def check_ram(suite_ram_requirements, test_size):
    if not isinstance(suite_ram_requirements, int):
        return ["Wrong 'ram' requirements: {}, should be integer".format(suite_ram_requirements)]
    min_ram_requirements = consts.TestRequirementsConstants.MinRam
    max_ram_requirements = consts.TestSize.get_max_requirements(test_size).get(consts.TestRequirements.Ram)
    if suite_ram_requirements < min_ram_requirements or suite_ram_requirements > max_ram_requirements:
        return ["Wrong 'ram' requirements: {}, should be in [{}..{}] for {}-size tests".format(suite_ram_requirements, min_ram_requirements, max_ram_requirements, test_size)]
    # TODO: Remove this part of rule when dafault_ram_requirement becomes power of 2
    if suite_ram_requirements != max_ram_requirements and not is_power_of_two(suite_ram_requirements):
        return ["Wrong 'ram' requirements: {}, should be power of 2".format(suite_ram_requirements)]

    return []
