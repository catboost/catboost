from conan import ConanFile


class App(ConanFile):

    settings = "os", "compiler", "build_type", "arch"

    options = {}

    tool_requires = "ragel/6.10", "swig/4.0.2", "yasm/1.3.0"

    def requirements(self):
        requires = [ "openssl/1.1.1t" ]
        for require in requires:
            if ("linux-headers" in require) and (self.settings.os != "Linux"):
                continue
            self.requires(require)

    generators = "cmake_find_package", "cmake_paths"

    def imports(self):
        self.copy(pattern="*swig*", src="bin", dst="./bin")
        self.copy(pattern="*yasm*", src="bin", dst="./bin")
        self.copy(pattern="ragel*", src="bin", dst="./bin")
        self.copy(pattern="ytasm*", src="bin", dst="./bin")
