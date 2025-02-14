#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# Change directory to the directory this file is located in.
cd $(dirname $(readlink -f "$0")) >/dev/null
# Then: switch to conan_u
cd conan_debug
conan install ../conanfile.py -s build_type=Debug
conan build ../conanfile.py
meson install -C build
