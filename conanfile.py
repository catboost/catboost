from conan import ConanFile


class App(ConanFile):

    settings = "os", "compiler", "build_type", "arch"

    options = {}

    def requirements(self):
        self.requires("openssl/1.1.1t")

    def build_requirements(self):
        self.tool_requires("ragel/6.10")
        self.tool_requires("swig/4.0.2")
        self.tool_requires("yasm/1.3.0")

    generators = "cmake_find_package", "cmake_paths"

    def imports(self):
        self.copy(pattern="*swig*", src="bin", dst="./bin")
        self.copy(pattern="*yasm*", src="bin", dst="./bin")
        self.copy(pattern="ragel*", src="bin", dst="./bin")
        self.copy(pattern="ytasm*", src="bin", dst="./bin")
