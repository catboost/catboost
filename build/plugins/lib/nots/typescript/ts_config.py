import copy
import os
import json

from .ts_errors import TsError, TsValidationError

from ..package_manager.base import utils

DEFAULT_TS_CONFIG_FILE = "tsconfig.json"


class TsConfig(object):
    @classmethod
    def load(cls, path):
        """
        :param path: tsconfig.json path
        :type path: str
        :rtype: TsConfig
        """
        tsconfig = cls(path)
        tsconfig.read()

        return tsconfig

    def __init__(self, path):
        if not os.path.isabs(path):
            raise TypeError("Absolute path required, given: {}".format(path))

        self.path = path
        self.data = {}

    def read(self):
        try:
            with open(self.path) as f:
                self.data = json.load(f)
        except Exception as e:
            raise TsError("Failed to read tsconfig {}: {}".format(self.path, e))

    def merge(self, rel_path, base_tsconfig):
        """
        :param rel_path: relative path to the configuration file we are merging in.
        It is required to set the relative paths correctly.
        :type rel_path: str
        :param base_tsconfig: base TsConfig we are merging with our TsConfig instance
        :type base_tsconfig: dict
        """
        if not base_tsconfig.data:
            return

        def relative_path(p):
            return os.path.normpath(os.path.join(rel_path, p))

        base_config_data = copy.deepcopy(base_tsconfig.data)

        parameter_section_labels = ["compilerOptions", "typeAcquisition", "watchOptions"]
        for opt_label in parameter_section_labels:
            base_options = base_config_data.get(opt_label)
            if not base_options:
                continue

            new_options = self.data.get(opt_label)
            for key in base_options:
                val = base_options[key]

                # lists of paths
                if key in ["extends", "outDir", "rootDir", "baseUrl", "include"]:
                    val = relative_path(val)

                # path string
                elif key in ["rootDirs", "excludeDirectories", "excludeFiles"]:
                    val = map(relative_path, val)

                # dicts having paths as values
                elif key in ["paths"]:
                    new_paths = new_options.get(key)
                    val = map(relative_path, val) + (new_paths if new_paths else [])

                base_options[key] = val

            if new_options and base_options:
                base_options.update(new_options)
                self.data[opt_label] = base_options

        base_config_data.update(self.data)
        self.data = base_config_data

    def inline_extend(self, dep_paths):
        """
        Merges the tsconfig parameters from configuration file referred by "extends" if any.
        Relative paths are adjusted, current parameter values are prioritized higer than
        those coming from extension file (according to TSC mergin rules).
        Returns list of file paths for config files merged into the current configuration
        :param dep_paths: dict of dependency names to their paths
        :type dep_paths: dict
        :rtype: list of str
        """
        ext_value = self.data.get("extends")
        if not ext_value:
            return []

        if ext_value.startswith("."):
            base_config_path = ext_value

        else:
            dep_name = utils.extract_package_name_from_path(ext_value)
            # the rest part is the ext config path
            file_path_start = len(dep_name) + 1
            file_path = ext_value[file_path_start:]
            dep_path = dep_paths.get(dep_name)
            if dep_path is None:
                raise Exception(
                    "referenceing from {}, data: {}\n: Dependency '{}' not found in dep_paths: {}"
                    .format(self.path, str(self.data), dep_name, dep_paths)
                )
            base_config_path = os.path.join(dep_path, file_path)

        rel_path = os.path.dirname(base_config_path)
        tsconfig_curdir_path = os.path.join(os.path.dirname(self.path), base_config_path)
        if os.path.isdir(tsconfig_curdir_path):
            base_config_path = os.path.join(base_config_path, DEFAULT_TS_CONFIG_FILE)

        # processing the base file recursively
        base_config = TsConfig.load(os.path.join(os.path.dirname(self.path), base_config_path))
        paths = [base_config_path] + base_config.inline_extend(dep_paths)

        self.merge(rel_path, base_config)
        del self.data["extends"]

        return paths

    def get_or_create_compiler_options(self):
        """
        Returns ref to the "compilerOptions" dict.
        :rtype: dict
        """
        opts = self.data.get("compilerOptions")
        if opts is None:
            opts = {}
            self.data["compilerOptions"] = opts

        return opts

    def compiler_option(self, name, default=None):
        """
        :param name: option key
        :type name: str
        :param default: default value
        :type default: mixed
        :rtype: mixed
        """
        return self.get_or_create_compiler_options().get(name, default)

    def inject_plugin(self, plugin):
        """
        :param plugin: plugin dict (ts-patch compatible, see https://github.com/nonara/ts-patch)
        :type plugin: dict of str
        """
        opts = self.get_or_create_compiler_options()
        if not opts.get("plugins"):
            opts["plugins"] = []
        opts["plugins"].append(plugin)

    def validate(self):
        """
        Checks whether the config is compatible with current toolchain.
        """
        opts = self.get_or_create_compiler_options()
        errors = []
        root_dir = opts.get("rootDir")
        out_dir = opts.get("outDir")
        config_dir = os.path.dirname(self.path)

        def is_mod_subdir(p):
            return not os.path.isabs(p) and os.path.normpath(os.path.join(config_dir, p)).startswith(config_dir)

        if root_dir is None:
            errors.append("'rootDir' option is required")

        if out_dir is None:
            errors.append("'outDir' option is required")
        elif not is_mod_subdir(out_dir):
            errors.append("'outDir' should be a subdirectory of the module")

        if opts.get("outFile") is not None:
            errors.append("'outFile' option is not supported")

        if opts.get("preserveSymlinks"):
            errors.append("'preserveSymlinks' option is not supported due to pnpm limitations")

        if opts.get("rootDirs") is not None:
            errors.append("'rootDirs' option is not supported, relative imports should have single root")

        if self.data.get("files") is not None:
            errors.append("'files' option is not supported, use 'include'")

        if self.data.get("references") is not None:
            errors.append("composite builds are not supported, use peerdirs in ya.make instead of 'references' option")

        if len(errors):
            raise TsValidationError(self.path, errors)

    def transform_paths(self, build_path, sources_path, package_rel_path):
        """
        Updates config with correct abs paths.
        All source files/dirs will be mapped to `sources_path`, output files/dirs will be mapped to `build_path`.
        :param build_path: module's build root
        :type build_path: str
        :param sources_path: module's source root
        :type sources_path: str
        :param package_rel_path: module's rel path to package root
        :type package_rel_path: str
        """
        opts = self.get_or_create_compiler_options()

        def sources_path_rel(x):
            return os.path.normpath(os.path.join(sources_path, x))

        def build_path_rel(x):
            return os.path.normpath(os.path.join(build_path, x))

        root_dir = opts["rootDir"]
        out_dir = opts["outDir"]

        opts["rootDir"] = sources_path_rel(root_dir)
        opts["outDir"] = build_path_rel(out_dir)

        if opts.get("typeRoots"):
            opts["typeRoots"] = list(map(sources_path_rel, opts["typeRoots"])) + list(map(build_path_rel, opts["typeRoots"]))

        if opts.get("paths") is None:
            opts["paths"] = {}

        # See: https://st.yandex-team.ru/FBP-47#62b4750775525b18f08205c7
        opts["paths"]["*"] = ["*", "./@types/*"]

        opts["baseUrl"] = os.path.normpath(os.path.join(package_rel_path, "node_modules"))

        include_dir_list = self.data.get("include")
        if include_dir_list:
            self.data["include"] = list(map(sources_path_rel, include_dir_list))

        exclude_dir_list = self.data.get("exclude")
        if exclude_dir_list:
            self.data["exclude"] = list(map(sources_path_rel, exclude_dir_list))

        if opts.get("sourceMap"):
            opts["sourceRoot"] = os.path.relpath(root_dir, out_dir)

        opts["skipLibCheck"] = True

    def write(self, path=None):
        """
        :param path: tsconfig path, defaults to original path
        :type path: str
        """
        if path is None:
            path = self.path

        with open(path, "w") as f:
            json.dump(self.data, f)
