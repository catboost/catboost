import subprocess

import yatest


def _run_cmd(cmd, out):
    with open(out, "w") as f:
        subprocess.check_call(cmd, stdout=f)

    return out


def test_R_version(links):
    links.set('R --version', _run_cmd(['R', '--version'], yatest.common.output_path('r_version.out')))


def test_cmd_build(links):
    links.set('R CMD build', _run_cmd(['R', 'CMD', 'build', '.'], yatest.common.output_path('r_cmd_build.out')))


def test_cmd_install(links):
    links.set('R CMD INSTALL', _run_cmd(['R', 'CMD', 'INSTALL', '.'], yatest.common.output_path('r_cmd_install.out')))


def test_cmd_check(links):
    links.set('R CMD check', _run_cmd(['R', 'CMD', 'check', '.', '--no-manual', '--no-examples'], yatest.common.output_path('r_cmd_check.out')))
