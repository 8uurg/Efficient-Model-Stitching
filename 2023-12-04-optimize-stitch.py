# First - the problem & the approaches 
import recomb.problems as problems
import os

from optimizestitchutils import get_approach, problem_constructor
import argparse
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--problem", required=True)
parser.add_argument("--stitchnet-path", required=True)
parser.add_argument("--approach", required=True)
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--log-folder", type=Path, default=".")
parser.add_argument("--evaluation-budget", default=200000, type=int)
parser.add_argument("--population-size", default=256, type=int)
parser.add_argument("--dataset-path", default="<add-dataset-folder>")
# Note: if the temp dir is extremely small runs may fail - ensure it has sufficient capacity for logs
# like the ones ray produces.
parser.add_argument("--tmp-dir", default="None") 
parser.add_argument("--number-of-workers", default=None, type=int)
parser.add_argument("--num-gpu-per-worker", default=1.0, type=float)
parser.add_argument("--batch-size", default=8, type=int)
parser.add_argument("--metric-0", default="loss", type=str, choices=["loss", "accuracy"])
parser.add_argument("--adaptive-steering", default=False)
parser.add_argument("--sample-limit", type=int)
parser.add_argument("--ray-head-node", default="auto")

# note - if either of these limits has been hit - the training stops,
# as such, whichever limit is stricter will dominate over the other.
parser.add_argument("--train-epochs", default=0, type=int)
parser.add_argument("--train-batches", default=1000, type=int)

parser.add_argument("--train-lr", default=1e-4, type=float)
parser.add_argument("--unfreeze-weights", action='store_true')
parser.add_argument("--dl-workers", default=2, type=int)
parser.add_argument("--init-p", default=0.10, type=float)
parsed = parser.parse_args()

# Either loss or accuracy (for now)
metric_0 = parsed.metric_0

problem_name = parsed.problem
approach = parsed.approach
seed = parsed.seed
sample_limit = parsed.sample_limit
# Note - should probably be a fixed value.
dataset_path = parsed.dataset_path
tmpdir = parsed.tmp_dir if parsed.tmp_dir != "None" else None
## %
number_of_workers = parsed.number_of_workers # determine by #gpus in cluster
num_gpu_per_worker = parsed.num_gpu_per_worker
batch_size = parsed.batch_size
# stitched_network_path = "./stitched-imagenet.th"
stitched_network_path = parsed.stitchnet_path # "./stitched-imagenet-balanced.th"
init_p = parsed.init_p
log_folder = parsed.log_folder

dl_workers = parsed.dl_workers
# Note - before we do anything else.
os.environ["RECOMB_NUM_DATALOADER_WORKERS"] = str(dl_workers)
ray_head_node = parsed.ray_head_node

evaluation_budget = parsed.evaluation_budget
population_size = parsed.population_size
num_clusters = 5

# For how many epochs do we train the offspring network?
train_offspring_epochs = parsed.train_epochs
# Note - a single epoch for imagenet is very large - limit the number of batches used.
train_offspring_batches = parsed.train_batches
# Do we unfreeze the network?
unfreeze_all_weights = parsed.unfreeze_weights
train_lr = parsed.train_lr

adaptive_steering_target_th = float(parsed.adaptive_steering) if isinstance(parsed.adaptive_steering, str) else parsed.adaptive_steering
approach_underscore = approach.replace("-", "_")

base_out_path = ("es"
                 f"-{approach_underscore}"
                 f"-{problem_name}"
                 f"-s{seed}"
                 f"-b{evaluation_budget}"
                 f"-P{population_size}"
                 f"-t{train_offspring_epochs}"
                 f"-f{'1' if unfreeze_all_weights else '0'}"
                 f"-m0{metric_0}"
                 f"-as{adaptive_steering_target_th if adaptive_steering_target_th != False else ''}")

log_folder.mkdir(parents=True, exist_ok=True)
eval_out_path = str(log_folder / (base_out_path + ".arrow"))
log_out_path = ("./optimize_logs/" + base_out_path + "tbl")
print(f"Saving evaluated solutions to {eval_out_path}")
# %% Create an event loop for the approach to run in.
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# %% Construct the problem
problem = problem_constructor(problem_name, dataset_path)

# %%
import numpy as np
import torch
import torchvision
import torchvision.datasets as thd
import torchvision.models.segmentation as segmentation_models
import recomb.layers as ly
import igraph as ig
import matplotlib.pyplot as plt
import polars as pl

import recomb.cx as cx
import recomb.layers as ly
import recomb.problems as problems
from recomb.cx import forward_get_all_feature_maps, construct_trained_cx_network_stitching
# %% Get a batch of data from the dataset - so that we can assess some things
from torch.utils.data import DataLoader
d_train = problem.get_dataset_train()
dl_train = DataLoader(d_train)
it_train = iter(dl_train)
for _ in range(5):
    X, Y = next(it_train)

# %% Load stitch
stitched_simpl = torch.load(stitched_network_path)
stitchnet, stitchinfo = stitched_simpl

# %% As we are transferring data across machines & pointer preservation
# apparently fails sometimes - use module ordering & indexing instead.
# Also, assign some data for quick active variable detection.

from recomb.cx import (
    compute_cx_module_ordering_and_out_switch,
    determine_variable_impact,
    determine_active_variables,
    set_genotype_module_ordering,
    get_genotype_module_ordering)

module_ordering, out_switch_idx = compute_cx_module_ordering_and_out_switch(stitchnet, stitchinfo)
determine_variable_impact(stitchnet)

# %%
# Embed computational cost info
import torchinfo
import recomb.eval_costs as ec
cost_summary = torchinfo.summary(stitchnet, input_data=[X], verbose=0)
ec.embed_cost_stats_in_model(cost_summary)

# %% Store preprocessed stitch network (with extra metadata)
active_stitch = (stitchnet, module_ordering, out_switch_idx)
torch.save(active_stitch, "active-stitch.th")

# %%
print(("The stitched neural network has "
       f"{len(stitchinfo.joiners)} matches"
       ))

# %%
# GOMEA et al.
import ealib
# Distributed Compute - used in this case for fitness evaluations.
# Note - start ray with a different temporary file directory: it can get quite large,
# and on the cluster we are using /tmp is at most 3GB - most of which is already used
# by the OS.
import ray
# td = Path(tmpdir)
ray.init(address=ray_head_node, _temp_dir=tmpdir)
# ray.init(_temp_dir=str(td))

# Create the evaluation log file.
assert not Path(f"{eval_out_path}.jsonl").exists(), "File already exists - please move or remove prior to rerunning an experiment"
with open(f"{eval_out_path}.jsonl", 'w') as f:
    f.write("")

# %% Set up function evaluations using ray
# We create a pool of async actors - each of which is returned to the pool once an evaluation completes.

NeuralNetIndividual = problems.NeuralNetIndividual

# AsyncActorPool
class AsyncActorPool:
    def __init__(self, actors):
        # Create a queue for actors.
        self.queue = asyncio.Queue()
        for actor in actors:
            self.queue.put_nowait(actor)

    async def perform(self, fn, *args):
        # Wait for an actor to be available
        # print("Got request for pool - waiting for availability")
        actor = await self.queue.get()
        # Perform the action - whatever fn does using actor
        # and wait until it returns (and the actor is available again)
        # print("Actor was made available - waiting for completed subtask")
        r = await fn(actor, *args)
        # Return actor to queue
        # print("Actor completed subtask - returning back to queue and finishing up.")
        await self.queue.put(actor)
        # print("Done!")
        return r

import time
import torch.utils.tensorboard
log_dir_tb = Path(log_out_path)
log_dir_tb.mkdir(parents=True, exist_ok=True)
# progress_log = torch.utils.tensorboard.SummaryWriter(log_dir=log_out_path)

class StatusLogger:

    def __init__(self, log_out_path):
        self.progress_log = torch.utils.tensorboard.SummaryWriter(log_dir=log_out_path)
        self.num_evaluated = 0
        self.num_reports = 0
        self.num_pending = 0
        self.num_evaluating = 0

    def on_start_eval(self):
        self.num_evaluating += 1
        self.progress_log.add_scalar("evaluations/num_evaluating", self.num_evaluating, self.num_reports)

        self.num_reports += 1

    def on_compl_eval(self):
        self.num_evaluating -= 1
        self.num_evaluated += 1
        self.progress_log.add_scalar("evaluations/num_evaluating", self.num_evaluating, self.num_reports)
        
        self.num_reports += 1

    def on_start_eval_req(self):
        self.num_pending += 1
        self.progress_log.add_scalar("evaluations/num_pending", self.num_pending, self.num_reports)
        
        self.num_reports += 1

    def on_compl_eval_req(self):
        self.num_pending -= 1
        self.progress_log.add_scalar("evaluations/num_pending", self.num_pending, self.num_reports)
        
        self.num_reports += 1

    

@ray.remote
class RayActorPool:
    """
    AsyncActorPool, but in ray - so that it has its own thread (hopefully buffering & saturating
    computational resources)
    """

    def __init__(self, actors, status_logger_factory):
        #
        self.status_logger = status_logger_factory()
        # Create a queue for actors.
        self.queue = asyncio.Queue()
        for actor in actors:
            self.queue.put_nowait(actor)

    async def perform(self, fn, *args):
        if self.status_logger is not None:
            self.status_logger.on_start_eval_req()
        # Wait for an actor to be available
        actor = await self.queue.get()
        # Perform the action - whatever fn does using actor
        # and wait until it returns (and the actor is available again)
        
        if self.status_logger is not None:
            self.status_logger.on_start_eval()
        r = await fn(actor, *args)
        
        if self.status_logger is not None:
            self.status_logger.on_compl_eval()
            self.status_logger.on_compl_eval_req()
        # Return actor to queue
        await self.queue.put(actor)
        return r

from copy import deepcopy

# Note - if we want to evaluate multiple solutions on the same GPU, increase this.
# However, responsibility with minimizing GPU load lies solely with the user - ray
# does not manage this.
@ray.remote(num_gpus=num_gpu_per_worker)
class RayStitchEvaluator:

    def __init__(self, problem_name, dataset_path, stitch_to_load, unfreeze_all_weights = False, train_offspring_epochs=0, train_offspring_batches=None, sample_limit=None, device="cuda"):
        self.device = torch.device(device)

        assert device != "cuda" or torch.cuda.is_available(), "CUDA should be available if requested. Yet it was not."

        if isinstance(stitch_to_load, str):
            self.stitchnet, self.module_ordering, self.out_switch_idx = torch.load(stitch_to_load)
        else:
            self.stitchnet, self.module_ordering, self.out_switch_idx = deepcopy(stitch_to_load)
        #
        self.sample_limit = sample_limit
        # 
        self.unfreeze_all_weights = unfreeze_all_weights
        self.train_offspring_epochs = train_offspring_epochs
        self.train_offspring_batches = train_offspring_batches
        #
        assert Path(dataset_path).exists(), "Dataset path should exist - otherwise data cannot be loaded (and we cannot proceed.)"
        self.problem = problem_constructor(problem_name, dataset_path)
        # Note - on every node ensure that the dataset is available prior to starting the experiment
        # e.g. by using shared storage.
        # This script does not automatically synchronize this here as there may be multiple workers
        # on the same machine. Downloading simultaneously would cause corruption & other issues.

        for m_idx in self.module_ordering:
            m = self.stitchnet.submodules[m_idx]
            m.active = 0
            m.simplify = True

        self.output_switch = self.stitchnet.submodules[self.out_switch_idx]
        self.output_switch.active = 2
        # Note - we may want to set this to false if we wish to train branches of the network seperately.
        self.output_switch.simplify = True

        # Rewrite graph - add an extra module (FeatureMapStacker) to the output switch to train
        # the ensemble efficiently.
        fm_stacker = cx.FeatureMapStacker()
        # Override - act as if this module does not exist in practice (because it would be removed /
        # unused after training)
        fm_stacker.total_macs = 0.0
        fm_stacker.total_all_bytes = 0.0
        fm_stacker_idx = len(self.stitchnet.submodules)
        self.stitchnet.submodules.append(fm_stacker)
        g = self.stitchnet.graph
        # Add stacker to graph
        v_fm_stacker = g.add_vertex(
            module=fm_stacker_idx
        )
        # Find output node in graph
        output_node_module_id = self.module_ordering[-1]
        output_switch_vertex_indices = [idx for
            idx, module_idx in enumerate(self.stitchnet.graph.vs["module"])
            if module_idx == output_node_module_id
        ]
        v_output = self.stitchnet.graph.vs[output_switch_vertex_indices[0]]
        # We will connect the stacker to this socket of the output node.
        self.stacking_output_socket = 3
        # Ad edges
        edges_to_add = []
        edges_to_add_socket = []
        # Add edges from inputs of v_output (except 2)
        for e in v_output.in_edges():
            if e["socket"] == 2: continue
            edges_to_add.append((e.source, v_fm_stacker.index))
            edges_to_add_socket.append(e["socket"])
        # Add edge from stacker to output node on socket self.stacking_output_socket
        edges_to_add.append((v_fm_stacker.index, v_output.index))
        edges_to_add_socket.append(self.stacking_output_socket)
        # Actually modify graph
        g.add_edges(edges_to_add, attributes=dict(socket=edges_to_add_socket))
        # Redetermine ordering after modification.
        self.stitchnet.determine_sorting()

        # initial_genotype = get_genotype_module_ordering(self.stitchnet, self.module_ordering)
        # print(f"Set up evaluator - f{initial_genotype}")


    
    def evaluate_genotype(self, genotype):
        # print("Applying genotype")
        # Apply genotype
        set_genotype_module_ordering(self.stitchnet, self.module_ordering, genotype)
        # Hardcoded: For output switch, do not simplify if active = 2.
        # This choice picks between the ensemble & individual networks, and if we pick
        # the ensemble, we wish to be able to train both branches seperately.
        self.output_switch.simplify = self.output_switch.active != 2

        assert (list(get_genotype_module_ordering(self.stitchnet, self.module_ordering)) == list(genotype)), "could not apply genotype - disassociated info & network."
        # Prune graph
        stitchnet_pruned = self.stitchnet.to_graph()
        stitchnet_pruned.prune_unused()
        active_variables = determine_active_variables(stitchnet_pruned)

        # Get compute & memory requirements
        # s = torchinfo.summary(stitchnet_pruned, input_data=[X])
        # print("Computing computational cost")
        total_mult_adds, total_bytes = ec.evaluate_compute_cost(stitchnet_pruned)

        neti_os = NeuralNetIndividual(stitchnet_pruned)
        # print("Computing accuracy")
        # TODO: Add sample limit here - currently running experiment has the limit applied elsewhere...
        eval_failed = ""
        try:
            untrained_accuracy, untrained_loss = self.problem.evaluate_network(self.device, neti_os, batch_size=batch_size, objective="both")
            accuracy, loss = untrained_accuracy, untrained_loss
        except Exception as e:
            # print(f"Evaluation failed due to {e}")
            accuracy = 0.0
            loss = np.inf
            eval_failed = str(e)

        result = {
            "untrained_accuracy": untrained_accuracy,
            "accuracy": accuracy,
            "untrained_loss": untrained_loss,
            "loss": loss,
            "total_memory_bytes": total_bytes,
            "total_mult_adds": total_mult_adds,
            "genotype": get_genotype_module_ordering(self.stitchnet, self.module_ordering),
            "active_variables": active_variables,
            "eval_failed": eval_failed,
        }

        if self.train_offspring_epochs > 0:
            # Copy modules for training so that we do not make further modifications to the original
            # network (We could skip this - but then there are dependencies on which node you train
            # in terms of accuracy - so if you train on a node that has trained using this stitch before
            # this would be beneficial)
            for i in range(len(stitchnet_pruned.submodules)):
                submod = stitchnet_pruned.submodules[i]
                # skip cloning switches so that the modifications to the output switch are preserved
                if isinstance(submod, cx.CXN): continue
                stitchnet_pruned.submodules[i] = deepcopy(submod)

            # Unfreeze weights if requested.
            if self.unfreeze_all_weights:
                for p in stitchnet_pruned.parameters():
                    p.requires_grad_(True)

            # Hardcoded - train an ensemble by stacking batches.
            # Note: for this to work the labels do potentially need to be
            # repeated accordingly! Check the problem definition if things fail
            # after this point on evaluating the loss.
            if self.output_switch.active == 2:
                self.output_switch.active = self.stacking_output_socket

            # Train the network
            training_failed = False
            try:
                self.problem.train_network(self.device,
                                    neti_os,
                                    batch_size=batch_size,
                                    lr=train_lr,
                                    num_epochs=self.train_offspring_epochs,
                                    num_batches=self.train_offspring_batches,
                                    seed=42,
                                    verbose=False)
                # , summarywriter=summarywriter
            except Exception as e:
                training_failed = True
            no_gradient_in_any_branch = stitchnet_pruned.had_no_gradients

            # Revert
            if self.output_switch.active == self.stacking_output_socket:
                self.output_switch.active = 2

            # Evaluate network after training - if there were any gradients.
            if not (no_gradient_in_any_branch or training_failed):
                trained_accuracy, trained_loss = self.problem.evaluate_network(self.device, neti_os, batch_size=batch_size, objective="both")

                # Use best of before and after training?
                accuracy = max(accuracy, trained_accuracy)
                loss = min(loss, trained_loss)

                # Update result.
                result["trained_accuracy"] = trained_accuracy
                result["trained_loss"] = trained_loss
                result["accuracy"] = accuracy
                result["loss"] = loss

        # print("Done - returning...")
        return result

# %%
alphabet_size = cx.get_stitcher_genotype_cardinality(stitchnet, stitchinfo)
l = len(alphabet_size)

# %%
# The scheduler is the object that manages evaluations, and how they are 
# performed in parallel. Note, generally quite simple by deferring to other
# scheduler implementations (e.g. python's asyncio loop & ray)
# In this case we do defer to pythons async loop. 

# Python AsyncIO Event Queue
scheduler, optimizer, set_adaptive_steering_threshold = get_approach(loop, 
    {
        "name": approach,
        "population_size": population_size,
        # Only really used for single-objective directions...
        "num_clusters": 5,
        "adaptive_steering": True if adaptive_steering_target_th is not False else False,
        "init_p": init_p,
    },
    {
        "alphabet_size": alphabet_size,
    })

# %%

def noop(genotype):
    return 0.0

print(alphabet_size)
problem_template = ealib.DiscreteObjectiveFunction(noop, l, alphabet_size, 0)

active_stitch_ray = ray.put(active_stitch)

if number_of_workers is None:
    ray_cluster_resources = ray.cluster_resources()
    # print(ray_cluster_resources)
    number_of_workers = int(ray_cluster_resources["GPU"] / num_gpu_per_worker)
    print(f"Using {number_of_workers} parallel evaluators (#gpus available / resources req)")

# Note 
eval_pool = RayActorPool.options(
    scheduling_strategy=ray.util.scheduling_strategies.NodeAffinitySchedulingStrategy(
        node_id=ray.get_runtime_context().get_node_id(),
        soft=False,
    )
).remote([
    RayStitchEvaluator.remote(problem_name,
                              dataset_path,
                              active_stitch_ray,
                              unfreeze_all_weights=unfreeze_all_weights,
                              train_offspring_epochs=train_offspring_epochs,
                              train_offspring_batches=train_offspring_batches,
                              sample_limit=sample_limit)
    for _ in range(number_of_workers)
], lambda: StatusLogger(log_out_path=log_out_path))

async def evaluate_genotype_using_RayStitchEvaluator(e: RayStitchEvaluator, g: np.array):
    time_eval_start = time.time()
    result = await e.evaluate_genotype.remote(g)
    time_eval_end = time.time()

    result["time_eval_start"] = time_eval_start
    result["time_eval_end"] = time_eval_end
    return result

evaluate_genotype_using_RayStitchEvaluator_rp = ray.put(evaluate_genotype_using_RayStitchEvaluator)

num_objective_evaluations = 0
num_evaluated = 0
evaluated_solutions = []

import json
from recomb.utils import SmartEncoder

# smwr = torch.utils.tensorboard.SummaryWriter(log_dir=log_out_path)
def update_adaptive_steering():
    global num_evaluated
    # For now, linearly increase the threshold until half of the evaluation budget
    # at that point, maintain the threshold value.
    progress = min(0.5 * num_evaluated / evaluation_budget, 1.0)
    # Note accuracy is flipped around
    new_th = -adaptive_steering_target_th * progress
    # Note, one may want to trottle this, as pruning the archive is potentially costly.
    set_adaptive_steering_threshold(new_th)

    # smwr.add_scalar("evaluations/accuracy_threshold", -new_th, num_evaluated)
    # global pop
    # smwr.add_scalar("solutions/num_active", pop.get_active(), num_evaluated)
    # smwr.add_scalar("solutions/num_max", pop.get_size(), num_evaluated)

async def evaluate_objective(p: ealib.Population, i: ealib.Individual):
    global evaluated_solutions
    global num_objective_evaluations
    global num_evaluated
    curr_eval_idx = num_objective_evaluations
    num_objective_evaluations += 1

    genotype = p.getData(ealib.GENOTYPECATEGORICAL, i)
    genotype_array = np.asarray(genotype)

    time_eval_sched_start = time.time()

    pyd = p.getData(ealib.PYDATA, i)
    if "active_variables" in pyd.data and "previous_genotype" in pyd.data:
        previous_genotype = pyd.data["previous_genotype"]
        changed_variables = {i for i, (v_a, v_b) in enumerate(zip(genotype_array, previous_genotype)) if v_a != v_b}

        active_variables = pyd.data["active_variables"]
        changed_and_active = changed_variables.intersection(active_variables)
        # If no active variables have changed - the evaluation will not have changed either.
        if len(changed_and_active) == 0:
            # Unquote for verbose logging of skipped evals
            # print(f"Changed variables are: {changed_variables}.")
            # print(f"Active variables are: {active_variables}.")
            # print(f"Intersection therefore is: {changed_and_active}, which is empty.")
            # print("Skipping evaluation of solution: no changes to active variables.")

            # Add row to indicate reevaluation.
            reeval_log_item = { "i": num_objective_evaluations, "reevaluates": pyd.data["eval_idx"] }
            # evaluated_solutions.append()
            with open(f"{eval_out_path}.jsonl", 'a') as f:
                json.dump(reeval_log_item, f, cls=SmartEncoder)
                f.write("\n")
            return

    # Update genotype for the next time we do a check :)
    pyd.data["previous_genotype"] = np.copy(genotype_array)

    # print(f"Got a request to evaluate {genotype_array} - adding to queue")

    # print(f"Got genotype {genotype_array}")
    # if num_evaluated > evaluation_budget:
    #     print("Budget reached!")
    #     raise ealib.EvaluationLimit()

    # ask ray to evaluate it over a pool of actors

    # print(f"Requesting resources from eval pool {genotype_array}")
    # print(f"requesting evaluation of {curr_eval_idx}")
    try:
        result = await eval_pool.perform.remote(evaluate_genotype_using_RayStitchEvaluator_rp, genotype_array)
    except Exception as e:
        print(f"Exception {e} was thrown - evaluation incomplete.")
    # print(f"completed evaluation of {curr_eval_idx}")
    # print(f"Pool has completed work for {genotype_array}")
    result["i"] = num_objective_evaluations
    # Set evaluation index
    num_evaluated += 1
    update_adaptive_steering()
    pyd.data["eval_idx"] = num_evaluated
    result["eval_idx"] = num_evaluated
    # Update set of active variables
    pyd.data["active_variables"] = result["active_variables"]
    # Turn into list for adding to df - sets aren't a natively supported datatype.
    # A list is close enough!
    result["active_variables"] = list(result["active_variables"])

    time_eval_sched_end = time.time()
    result["time_eval_sched_start"] = time_eval_sched_start
    result["time_eval_sched_end"] = time_eval_sched_end

    # evaluated_solutions.append(result)
    # Dump in json format instead.
    with open(f"{eval_out_path}.jsonl", 'a') as f:
        json.dump(result, f, cls=SmartEncoder)
        f.write("\n")

    # Update dataframe to include new evaluated solution
    # Sidenote - we might want to do this incrementally.
    # pl.DataFrame(evaluated_solutions).write_ipc(eval_out_path)
    
    # print(f"Increasing number of evaluations")
    num_evaluated += 1

    # result contains "accuracy" "loss" "total_mult_adds" "total_memory_bytes"

    # print(f"Setting objective values")
    # Store result - note - minimization is uniformly assumed.
    objective = p.getData(ealib.OBJECTIVE, i)
    # accuracy should be maximized (flip sign)
    if metric_0 == "accuracy":
        objective.set_objective(0, -result[metric_0])
    else:
        objective.set_objective(0, result[metric_0])
    # cost should be minimized (keep as-is)
    objective.set_objective(1, result["total_mult_adds"])

    # print(f"Evaluation completed for {genotype_array} - left: {num_pending}")

    # if num_evaluated > evaluation_budget:
    #     print("Budget reached!")
    #     raise ealib.EvaluationLimit()

import datetime

objective_fn = ealib.PyAsyncObjectiveFunction(scheduler, problem_template, evaluate_objective)
objective_fn = ealib.Limiter(objective_fn, evaluation_limit=evaluation_budget, time_limit=datetime.timedelta(days=1))
# %%

stepper = ealib.TerminationStepper((lambda : optimizer), None)
f = ealib.SimpleConfigurator(objective_fn, stepper, seed)
# This experiment utilizes extra python data to track active variables & which variables have changed
# as to skip changes involving no active variables.
# This is not registered by default - so request it :)
pop = f.getPopulation()
pop.registerData(ealib.PYDATA)

f.run()

torch.save(evaluated_solutions, "evaluated_solutions.pckl")
pl.DataFrame(evaluated_solutions).write_ipc("evaluated_solutions_gomea.arrow")
