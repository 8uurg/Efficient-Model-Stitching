from conan import ConanFile
from conan.tools import Meson

class EALibConan(ConanFile):
    generators = ["pkg_config", "virtualenv"]
    name = "EALib"
    version = "0.0.1"
    # thrift/0.16.0, capnproto/0.9.1@
    requires = ["cmake/3.23.2@", "protobuf/3.21.1", "grpc/1.47.1@", "openssl/1.1.1q"]
    settings = "os", "compiler", "build_type", "arch"

    def build(self):
        meson = Meson(self)
        meson.configure(build_folder="build")
        meson.build()
