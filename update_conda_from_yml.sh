# Create or update the recombnet conda environment from the environment.yml file
mamba env create -q -n recombnet -f environment.yml || mamba env update -q -n recombnet -f environment.yml