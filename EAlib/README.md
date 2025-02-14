# EAlib

## Installation

We have tested this using GCC 12. Other compilers may work, but have not been tested. 
Requires at least a recent compiler that supports C++17 features, e.g.:
* GCC 9
* MSVC 19 or greater
* clang 10 or greater.

Note, while the conda environment contains a recent version of GCC, which should work, we have noticed that in certain cases this compiler is NOT selected. This is potentially due to g++ being missing, or some other hiccups, causing conan to select the wrong compiler, overriding meson's compiler selection. Installing `g++` explicitly into the conda environment may help. If this does not work, take a look at `meson-native-patch.py` which explicitly updates these paths.

Build is done through meson. If not installed, install meson and ninja using `pip install meson ninja`.

### Compling
- Follow the commands in `install.sh` for the initial setup.
- After initial setup `meson compile -C <dir>` or `meson install -C <dir>` should suffice for incremental recompilation.

# Credit
 DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability

This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.

Project leaders: Peter A.N. Bosman, Tanja Alderliesten
Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
Main code developer: Arthur Guijt
