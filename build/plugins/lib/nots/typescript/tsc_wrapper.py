import os
import json
import shutil
import subprocess
import tarfile

from ..package_manager import constants


class TsError(RuntimeError):
    pass


class TsValidationError(TsError):
    def __init__(self, path, errors):
        self.path = path
        self.errors = errors

        super(TsValidationError, self).__init__("Invalid tsconfig {}:\n{}".format(path, "\n".join(errors)))


class TsCompilationError(TsError):
    def __init__(self, code, stdout, stderr):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr

        super(TsCompilationError, self).__init__("tsc exited with code {}:\n{}\n{}".format(code, stdout, stderr))


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

    def validate(self):
        """
        Checks whether the config is compatible with current toolchain.
        """
        opts = self.get_or_create_compiler_options()
        errors = []
        root_dir = opts.get("rootDir")
        out_dir = opts.get("outDir")
        config_dir = os.path.dirname(self.path)
        is_mod_subdir = lambda p: not os.path.isabs(p) and os.path.normpath(os.path.join(config_dir, p)).startswith(config_dir)

        if root_dir is None:
            errors.append("'rootDir' option is required")
        elif not is_mod_subdir(root_dir):
            errors.append("'rootDir' should be a subdirectory of the module")

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

    def transform_paths(self, build_path, sources_path):
        """
        Updates config with correct abs paths.
        All source files/dirs will be mapped to `sources_path`, output files/dirs will be mapped to `build_path`.
        :param build_path: module's build root
        :type build_path: str
        :param sources_path: module's source root
        :type sources_path: str
        """
        opts = self.get_or_create_compiler_options()

        sources_path_rel = lambda x: os.path.normpath(os.path.join(sources_path, x))
        build_path_rel = lambda x: os.path.normpath(os.path.join(build_path, x))

        root_dir = opts["rootDir"]
        out_dir = opts["outDir"]

        opts["rootDir"] = sources_path_rel(root_dir)
        opts["outDir"] = build_path_rel(out_dir)

        if opts.get("typeRoots"):
            opts["typeRoots"] = list(map(sources_path_rel, opts["typeRoots"])) + list(map(build_path_rel, opts["typeRoots"]))

        opts["baseUrl"] = build_path_rel("node_modules")

        self.data["include"] = list(map(sources_path_rel, self.data.get("include", [])))
        self.data["exclude"] = list(map(sources_path_rel, self.data.get("exclude", [])))

        if opts.get("sourceMap"):
            opts["sourceRoot"] = os.path.relpath(root_dir, out_dir)

    def write(self, path=None):
        """
        :param path: tsconfig path, defaults to original path
        :type path: str
        """
        if path is None:
            path = self.path

        with open(path, "w") as f:
            json.dump(self.data, f)


class TscWrapper(object):
    _TSCONFIG_FILENAME = "tsconfig.json"

    def __init__(self, build_root, build_path, sources_path, nodejs_bin_path, script_path, config_path):
        self.build_root = build_root
        self.build_path = build_path
        self.sources_path = sources_path
        self.nodejs_bin_path = nodejs_bin_path
        self.script_path = script_path
        self.config_path = config_path

    def compile(self):
        self._prepare_dependencies()
        config = self._build_config()
        self._exec_tsc(["--build", config.path])

    def _prepare_dependencies(self):
        self._copy_package_json()
        self._unpack_node_modules()

    def _copy_package_json(self):
        # TODO: Validate "main" and "files" - they should include files from the output directory.
        shutil.copyfile(
            os.path.join(self.sources_path, constants.PACKAGE_JSON_FILENAME),
            os.path.join(self.build_path, constants.PACKAGE_JSON_FILENAME),
        )

    def _unpack_node_modules(self):
        nm_bundle_path = os.path.join(self.build_path, constants.NODE_MODULES_BUNDLE_FILENAME)
        if os.path.isfile(nm_bundle_path):
            with tarfile.open(nm_bundle_path) as tf:
                tf.extractall(os.path.join(self.build_path, "node_modules"))

    def _build_config(self):
        config = TsConfig.load(self.config_path)
        config.validate()
        config.transform_paths(
            build_path=self.build_path,
            sources_path=self.sources_path,
        )

        config.path = os.path.join(self.build_path, self._TSCONFIG_FILENAME)
        config.write()

        return config

    def _exec_tsc(self, args):
        p = subprocess.Popen(
            [self.nodejs_bin_path, self.script_path] + args,
            cwd=self.build_path,
            stdin=None,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = p.communicate()

        if p.returncode != 0:
            raise TsCompilationError(p.returncode, stdout.decode("utf-8"), stderr.decode("utf-8"))
