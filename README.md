# Efficient Model Stitching - Source Code
This repository contains the code pertaining to the work 'Exploring the Search Space of Neural Network Combinations obtained with Efficient Model Stitching' - to be presented at GECCO 2024 - Workshop: Neuroevolution at Work.

[Paper](https://dl.acm.org/doi/abs/10.1145/3638530.3664131) | [Code (Zenodo - Archived)](https://zenodo.org/doi/10.5281/zenodo.11120073) | [Data (Zenodo - Archived)](https://zenodo.org/doi/10.5281/zenodo.11120102)

## Usage
- Clone this repository
- Set up the conda environment `recombnet` by executing `update_conda_from_yml.sh`.
- On the node running the EAs, C++ code need to be compiled and installed, see `EALib/README.md` for details.
- Download the prerequisite datasets (ImageNet, VOC), and ensure they are unpacked into a dataset folder. Note the path to this folder.
- Update the path to the dataset folder by replacing `<add-dataset-folder>` with the relevant path.
- Similarly, ensure the tmp directory is large enough. Many logs may be written, e.g. by tools like ray, causing runs to terminate prematurely if this directory has no space available. If need be, relocate the tmp directory.
- Download stitched networks from (tbd), or use the `*Stitching-Pretrained*` notebooks to stitch one yourself.
- Running the commands from `experiment-*.txt` files should, given that everything has been set up correctly, perform runs in identical configuration. Output is, due to the asynchronous nature of the EAs employed, not deterministic.
    - Experiments for VOC were ran on a SLURM cluster. The batch file used here sets up a ray cluster for the use of this experiment prior to running the script, if a ray cluster is set up already - only the latter command should be necessary.
    - Experiments for ImageNet require a configured ray cluster to be set up, with the controller node being the node scripts are ran on. As we set up a new cluster for each individual run for SLURM, one may want to refer to `2023-12-25-run-exp-voc-final.sh`, or alternatively, refer to the ray documentation.
    - Data should be present on all nodes - not just the main node.
    - Environment needs to be set up on all nodes - not just the main nodes. However, the conda environment alone should suffice here: compiling and installing EAlib may be skipped.
- After running the experiments `2024-01-02-process-run-data.ipynb` can be used to process data from an individual run, and find the approximation front.
- The solutions on the front will be reevaluated on the test set with the commands in `2024-01-03-reevaluate-cmds.txt` (and `2024-01-15-reeval-imagenet-b.txt`).
- Finally, `2024-01-03-process-reeval-data.ipynb` can be used to create relevant plots.
