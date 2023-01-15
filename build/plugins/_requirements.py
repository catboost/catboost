import _test_const as consts


def check_cpu(suite_cpu_requirements, test_size, is_kvm=False):
    min_cpu_requirements = consts.TestRequirementsConstants.MinCpu
    max_cpu_requirements = consts.TestSize.get_max_requirements(test_size).get(consts.TestRequirements.Cpu)
    if isinstance(suite_cpu_requirements, str):
        if all(consts.TestRequirementsConstants.is_all_cpu(req) for req in (max_cpu_requirements, suite_cpu_requirements)):
            return None
        return "Wrong 'cpu' requirements: {}, should be in [{}..{}] for {}-size tests".format(suite_cpu_requirements, min_cpu_requirements, max_cpu_requirements, test_size)

    if not isinstance(suite_cpu_requirements, int):
        return "Wrong 'cpu' requirements: {}, should be integer".format(suite_cpu_requirements)

    if suite_cpu_requirements < min_cpu_requirements or suite_cpu_requirements > consts.TestRequirementsConstants.get_cpu_value(max_cpu_requirements):
        return "Wrong 'cpu' requirement: {}, should be in [{}..{}] for {}-size tests".format(suite_cpu_requirements, min_cpu_requirements, max_cpu_requirements, test_size)

    return None


def is_power_of_two(num):
    return num > 0 and ((num & (num - 1)) == 0)


# TODO: Remove is_kvm param when there will be guarantees on RAM
def check_ram(suite_ram_requirements, test_size, is_kvm=False):
    if not isinstance(suite_ram_requirements, int):
        return "Wrong 'ram' requirements: {}, should be integer".format(suite_ram_requirements)
    min_ram_requirements = consts.TestRequirementsConstants.MinRam
    max_ram_requirements = consts.MAX_RAM_REQUIREMENTS_FOR_KVM if is_kvm else consts.TestSize.get_max_requirements(test_size).get(consts.TestRequirements.Ram)
    if suite_ram_requirements < min_ram_requirements or suite_ram_requirements > max_ram_requirements:
        err_msg = "Wrong 'ram' requirements: {}, should be in [{}..{}] for {}-size tests".format(suite_ram_requirements, min_ram_requirements, max_ram_requirements, test_size)
        if is_kvm:
            err_msg += ' with kvm requirements'
        return err_msg
    # TODO: Remove this part of rule when dafault_ram_requirement becomes power of 2
    if suite_ram_requirements != max_ram_requirements and not is_power_of_two(suite_ram_requirements):
        return "Wrong 'ram' requirements: {}, should be power of 2".format(suite_ram_requirements)

    return None


def check_ram_disk(suite_ram_disk, test_size, is_kvm=False):
    min_ram_disk = consts.TestRequirementsConstants.MinRamDisk
    max_ram_disk = consts.TestSize.get_max_requirements(test_size).get(consts.TestRequirements.RamDisk)
    if isinstance(suite_ram_disk, str):
        if all(consts.TestRequirementsConstants.is_all_ram_disk(req) for req in (max_ram_disk, suite_ram_disk)):
            return None
        return "Wrong 'ram_disk' requirements: {}, should be in [{}..{}] for {}-size tests".format(suite_ram_disk, 0, max_ram_disk, test_size)

    if not isinstance(suite_ram_disk, int):
        return "Wrong 'ram_disk' requirements: {}, should be integer".format(suite_ram_disk)

    if suite_ram_disk < min_ram_disk or suite_ram_disk > consts.TestRequirementsConstants.get_ram_disk_value(max_ram_disk):
        return "Wrong 'ram_disk' requirement: {}, should be in [{}..{}] for {}-size tests".format(suite_ram_disk, min_ram_disk, max_ram_disk, test_size)

    return None
