# Change directory to the directory this file is located in.
cd $(dirname $(readlink -f "$0")) >/dev/null
# Ask conan to install dependency packages
conan install . --output-folder=build-conan --build=missing
# Use meson to configure & setup build - using dependencies provided by conan
# Tell meson to detect whether we are using a venv or not (otherwise it will try to install it in the system python installation)
meson setup build-debug --buildtype=debug --native-file build-conan/conan_meson_native.ini 
meson configure build-debug --python.install-env=auto
meson setup build --buildtype=debugoptimized --native-file build-conan/conan_meson_native.ini
meson configure build --python.install-env=auto
meson setup build-release --buildtype=release --native-file build-conan/conan_meson_native.ini
meson configure build-release --python.install-env=auto
# Note that conda environements set quite a few flags using environment variables, overriding meson's, unset them.
# unset CFLAGS
# unset CXXFLAGS
# Compiling the package should now be as easy as
meson compile -C build
# And installation (which will also recompile, if necessary)
meson install -C build
