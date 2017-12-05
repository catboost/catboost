import pytest

import build.plugins._requirements as requirements
import build.plugins._test_const as consts


class TestRequirements(object):
    @pytest.mark.parametrize('test_size', consts.TestSize.sizes())
    def test_cpu(self, test_size):
        max_cpu = consts.TestSize.get_max_requirements(test_size).get(consts.TestRequirements.Cpu)
        min_cpu = consts.TestRequirementsConstants.MinCpu
        assert requirements.check_cpu(min_cpu - 1, test_size)
        assert requirements.check_cpu("unknown", test_size)
        assert not requirements.check_cpu(1, test_size)
        assert not requirements.check_cpu(3, test_size)
        if max_cpu != consts.TestRequirementsConstants.AllCpu:
            assert requirements.check_cpu(max_cpu + 1, test_size)
            assert requirements.check_cpu(max_cpu + 4, test_size)
            assert requirements.check_cpu(consts.TestRequirementsConstants.AllCpu, test_size)
        else:
            assert not requirements.check_cpu(consts.TestRequirementsConstants.AllCpu, test_size)

    @pytest.mark.parametrize('test_size', consts.TestSize.sizes())
    def test_ram(self, test_size):
        max_ram = consts.TestSize.get_max_requirements(test_size).get(consts.TestRequirements.Ram)
        min_ram = consts.TestRequirementsConstants.MinRam
        assert requirements.check_ram(min_ram - 1, test_size)
        assert requirements.check_ram(max_ram + 1, test_size)
        assert requirements.check_ram(5, test_size)
        assert not requirements.check_ram(1, test_size)
        assert not requirements.check_ram(4, test_size)
        if test_size in (consts.TestSize.Large, consts.TestSize.Fat):
            assert not requirements.check_ram(32, test_size)
        else:
            assert requirements.check_ram(32, test_size)
        assert requirements.check_ram(48, test_size)
