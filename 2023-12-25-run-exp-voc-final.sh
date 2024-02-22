#!/bin/bash
#SBATCH --job-name=check-popsize
# Experiments are bounded by a 1-day time limit.
#SBATCH --time=1-00:00:00
# Note - we use ray to manage workers - and each worker is a single task!
#SBATCH --tasks-per-node=1
# Now - the number of nodes necessary may differ - depending on the hardware of the cluster
# we can opt to have all resources for a single node. Or choose to have more nodes with
# fewer GPUs.
# In any case, use the GPU partition (change if it has a different name elsewhere.)
#SBATCH -p gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=72
#SBATCH --gpus-per-task=4
# CPUs should not shared between worker nodes.
#SBATCH --exclusive

# Set up mamba - note - this may differ from HPC system to hpc system
# and we assume that dependencies have already been setup & installed.
module load 2023
module load Mamba/23.1.0-4

# Activate environment
source ~/.bashrc
mamba activate recombnet

# Following the guide at
# https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-network-ray

# Find head node
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
IFS=' ' read -ra ADDR <<<"$head_node_ip"
if [[ ${#ADDR[0]} -gt 16 ]]; then
  head_node_ip=${ADDR[1]}
else
  head_node_ip=${ADDR[0]}
fi
echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi

# Start head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &

echo "Started head"
# Start the ray worker nodes
# number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))

for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --num-cpus "${SLURM_CPUS_PER_TASK}" --num-gpus "${SLURM_GPUS_PER_TASK}" --block &
    sleep 5
done

echo "Started workers"

echo "Starting task..."
# Now that the ray cluster for this task is up and running - we can start the main job.
# The first argument to this script is the approach, the second the population size.
python -u 2023-12-04-optimize-stitch.py \
  --problem=voc \
  --stitchnet-path ./stitched-voc-a-deeplab-mobilenetv3-b-deeplab-resnet50.th \
  --log-folder=./2023-12-19/popsize-voc-a \
  --dataset-path <add-dataset-folder> \
  --approach $1 \
  --seed=$2 \
  --population-size=$3 \
  --batch-size=4 \
  --num-gpu-per-worker=0.1 \
  --metric-0=accuracy \
  --init-p=0.055 \
  --adaptive-steering=0.5 \
  --sample-limit=64 \
  --tmp-dir None \
  --ray-head-node auto
