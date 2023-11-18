# _build_config.py.in is converted into _build_config.py during the meson build process.

from __future__ import annotations


def build_config() -> dict[str, str]:
    """
    Return a dictionary containing build configuration settings.

    All dictionary keys and values are strings, for example ``False`` is
    returned as ``"False"``.

        .. versionadded:: 1.1.0
    """
    return dict(
        # Python settings
        python_version="@python_version@",
        python_install_dir=r"@python_install_dir@",
        python_path=r"@python_path@",

        # Package versions
        contourpy_version="@contourpy_version@",
        meson_version="@meson_version@",
        mesonpy_version="@mesonpy_version@",
        pybind11_version="@pybind11_version@",

        # Misc meson settings
        meson_backend="@meson_backend@",
        build_dir=r"@build_dir@",
        source_dir=r"@source_dir@",
        cross_build="@cross_build@",

        # Build options
        build_options=r"@build_options@",
        buildtype="@buildtype@",
        cpp_std="@cpp_std@",
        debug="@debug@",
        optimization="@optimization@",
        vsenv="@vsenv@",
        b_ndebug="@b_ndebug@",
        b_vscrt="@b_vscrt@",

        # C++ compiler
        compiler_name="@compiler_name@",
        compiler_version="@compiler_version@",
        linker_id="@linker_id@",
        compile_command="@compile_command@",

        # Host machine
        host_cpu="@host_cpu@",
        host_cpu_family="@host_cpu_family@",
        host_cpu_endian="@host_cpu_endian@",
        host_cpu_system="@host_cpu_system@",

        # Build machine, same as host machine if not a cross_build
        build_cpu="@build_cpu@",
        build_cpu_family="@build_cpu_family@",
        build_cpu_endian="@build_cpu_endian@",
        build_cpu_system="@build_cpu_system@",
    )
