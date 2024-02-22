"""
Due to an issue, conan may not inform meson about the right gcc & g++ compilers.
Tthis breaks compilation, or import, of the library.
This script patches the conan_meson_native.ini file as to ensure the right *full* path is always
provided, such that meson uses the right compiler.
"""

import subprocess

file_to_patch = "build-conan/conan_meson_native.ini"

# Obtain GCC path
gcc_path = subprocess.check_output(["which", "gcc"])
gcc_full_path = subprocess.check_output(["realpath", gcc_path]).decode().strip()
gxx_path = subprocess.check_output(["which", "g++"])
gxx_full_path = subprocess.check_output(["realpath", gxx_path]).decode().strip()

with open(file_to_patch) as f:
    content = f.read()

content = content.replace("'gcc'", f"'{gcc_full_path}'")
content = content.replace("'g++'", f"'{gxx_full_path}'")

with open(file_to_patch, 'w') as f:
    f.write(content)
