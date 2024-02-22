# Change directory to the directory this file is located in.
cd $(dirname $(readlink -f "$0")) >/dev/null
# Then: switch to conan_u
cd conan_debug
conan install ../conanfile.py -s build_type=Debug
conan build ../conanfile.py
meson install -C build
