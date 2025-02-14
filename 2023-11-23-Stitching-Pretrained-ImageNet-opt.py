#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
#
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
#
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

# %% [markdown]
# # Stitching Segmentation Models
# 
# Quoting from "Rethinking Atrous Convolution for Semantic Image Segmentation":
# > We evaluate the proposed models on the PASCAL VOC 2012 semantic segmentation benchmark [ 20] which contains 20 foreground object classes and one background class. The original dataset contains 1, 464 (train), 1, 449 (val), and
# 1, 456 (test) pixel-level labeled images for training, validation, and testing, respectively. The dataset is augmented by
# the extra annotations provided by [ 29 ], resulting in 10, 582 (trainaug) training images. The performance is measured in
# terms of pixel intersection-over-union (IOU) averaged across the 21 classes.
# 
# So the VOC dataset may be used.
# 
# Furthermore, the pretrained models seem to be trained on COCO-using-VOC-labels, we might want to figure that out, too.
# 
# Alternative: MONAI for Medical Decathlon?

## %
number_of_workers = None # determine by #gpus in cluster
num_gpu_per_worker = 1.0
batch_size = 8
dataset_path = "/export/scratch2/constellation-data/arthur/"
# stitched_network_path = "./stitched-imagenet.th"
stitched_network_path = "./stitched-imagenet-balanced.th"

evaluation_budget = 200000
population_size = 256
num_clusters = 5

# For how many epochs do we train the offspring network?
train_offspring_epochs = 0
# Note - a single epoch for imagenet is very large - limit the number of batches used.
train_offspring_batches = 1000
# Do we unfreeze the network?
unfreeze_all_weights = False
train_lr = 1e-4

eval_out_path = ("es-lkgomea"
                 "-imagenet"
                 f"-b{evaluation_budget}"
                 f"-P{population_size}"
                 f"-t{train_offspring_epochs}"
                 f"-f{'1' if unfreeze_all_weights else '0'}"
                 ".arrow")
print(f"Saving evaluated solutions to {eval_out_path}")
# %% Create an event loop
import asyncio
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)

# %%

import recomb.problems as problems
problem = problems.ImageNetProblem(root=dataset_path, validation_sample_limit=1000)
# problem.ensure_downloaded()

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

from pathlib import Path

import recomb.cx as cx
import recomb.layers as ly
import recomb.problems as problems
from recomb.cx import forward_get_all_feature_maps, construct_trained_cx_network_stitching
import os
os.environ["RECOMB_NUM_DATALOADER_WORKERS"] = "2"
# %%
# trsf = segmentation_models.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1.transforms()
from torch.utils.data import DataLoader
# d_train, _, _ = problem.load_problem_dataset(transform_train=trsf, transform_validation=trsf)
d_train = problem.get_dataset_train()
# d_train.transform = trsf
dl_train = DataLoader(d_train)
it_train = iter(dl_train)
for _ in range(5):
    X, Y = next(it_train)

# %%
in_shape = X.shape
in_shape

# %%
out_shape = Y.shape
out_shape

# %%
stitched_simpl = torch.load(stitched_network_path)

# %%
stitchnet, stitchinfo = stitched_simpl

# %% 

def compute_cx_module_ordering_and_out_switch(stitchnet, stitchinfo):
    c = 0
    for js in stitchinfo.joiners:
        for j in js:
            j.variable_pos = c
            c += 1
    stitchinfo.output_switch.variable_pos = c
    c += 1

    # we have c variables, which modules should be set / fetched?
    r = [None for _ in range(c)]
    
    for v in stitchnet.graph.vs:
        module_idx = v["module"]
        if module_idx < 0: continue
        md = stitchnet.submodules[module_idx]
        if not isinstance(md, cx.CXN): continue
        r[md.variable_pos] = module_idx
    
    return r, r[-1]

def determine_variable_impact(stitchnet: ly.NeuralNetGraph):
    """
    Determine which module is affected by which variable (so that a modules' presence indicates the activity of a variable)
    """
    # Initialize.
    for m in stitchnet.submodules:
        m.active_variables = set()
    # From the output determine which switches variables directly impact this modules' activity
    # Resetting whenever encountering a switch itself.
    for o in stitchnet.ord:
        v = stitchnet.graph.vs[o]
        # Input & output are always active.
        if v["module"] < 0: continue
        m = stitchnet.submodules[v["module"]]
        # CXN layers have variables associated with them.
        if isinstance(m, cx.CXN):
            m.active_variables.add(m.variable_pos)
            # Edge case - what if we have this output immidiately go into another switch?
            # Is it fair to say that there is no direct impact - need some kind of intermediate layer
            # in this case to signal things... This should work for now - but using the CX graph
            # to find which variables are active might be more effective.
            continue
        # Other layers aggregate the elements over their inputs
        for e in v.in_edges():
            tv = stitchnet.graph.vs[e.source]
            if tv["module"] < 0: continue
            tm = stitchnet.submodules[tv["module"]]
            m.active_variables.update(tm.active_variables)

def determine_active_variables(graphnet):
    active = set()
    for m in graphnet.submodules:
        if hasattr(m, "active_variables"):
            active.update(m.active_variables)
    return active

def set_genotype_module_ordering(stitchnet, mo, genotype):
    for i, md in enumerate(mo):
        stitchnet.submodules[md].active = genotype[i]

def get_genotype_module_ordering(stitchnet, mo):
    return [
        stitchnet.submodules[md].active
        for md in mo
    ]

module_ordering, out_switch_idx = compute_cx_module_ordering_and_out_switch(stitchnet, stitchinfo)
determine_variable_impact(stitchnet)

# %%
def get_cx_connectivity_graph(cx_net: ly.NeuralNetGraph):
    g = cx_net.graph.copy()
    o = g.topological_sorting()
    idxs_to_remove = []
    edges_to_add = []
    for i in o:
        vi = g.vs[i]

        is_edge_case = False
        if len(vi.in_edges()) == 0:
            # edge case - input node
            vi["cxs"] = set([vi.index])
            is_edge_case = True
        
        if len(vi.out_edges())  == 0 and vi["module"] < 0:
            # edge case - output node
            vi["cxs"] = set([vi.index])
            is_edge_case = True
        
        is_cxn = isinstance(cx_net.submodules[vi["module"]], cx.CXN)
        cxs_in = set()
        # i.e., where does this CXN link to via what node?
        affinity_mappings = {}
        for e in vi.in_edges():
            cxs_in.update(g.vs[e.source]["cxs"])
            if is_cxn:
                affinity_set = affinity_mappings.get(e["socket"], set())
                affinity_set.update(g.vs[e.source]["cxs"])
                affinity_mappings[e["socket"]] = affinity_set
        if is_cxn:
            edges_to_add += [(s, i, socket) for (socket, cxn_set) in affinity_mappings.items() for s in cxn_set]
            vi["cxs"] = set([i])
        elif not is_edge_case:
            vi["cxs"] = cxs_in
            idxs_to_remove.append(i)

    g.add_edges([t[:2] for t in edges_to_add], attributes=dict(socket=[t[2] for t in edges_to_add]))
    g.delete_vertices(idxs_to_remove)
    return g

# %%
cxg = get_cx_connectivity_graph(stitchnet)
# fig, ax = plt.subplots()
# graph_layout = cxg.layout("sugiyama")
# graph_layout.rotate(-90)
# ig.plot(cxg, target=ax, layout=graph_layout)

# %%
def compute_primary_line_membership(cx_graph, verbose=False):
    """
    Recover for each vertex in the crossover point graph to which networks they originally belonged
    assuming that input 0 to each crossover point maintains the original graph.

    (Note that this method may be skipped by tracking the original origin during stitching and assigning
     membership accordingly.)
    """
    # Initialize graph membership to each on their own
    cx_graph.vs["og"] = range(len(cx_graph.vs))
    # For input & output add a placeholder
    cx_graph.vs[0]["og"] = -1
    cx_graph.vs[1]["og"] = -1
    
    # Go over the graph in a topological order
    ordering = cx_graph.topological_sorting()
    for o in ordering:
        v = cx_graph.vs[o]
        # skip over input & output nodes
        if v["og"] == -1: continue
        # Loop over the elements with a similar affinity set and get their corresponding assignment.
        new_og = v["og"]
        same_origin_nodes = [e.source for e in v.in_edges() if e["socket"] == 0]
        if verbose: print(f"same origin: {same_origin_nodes}")
        # Merge identities in union-find like structure.
        for set_elem in same_origin_nodes:
            # print(f"visiting {set_elem}")
            v_other = cx_graph.vs[set_elem]
            if verbose: print(f"incident edge og is {v_other['og']}")
            if v_other["og"] == -1: continue
            new_og = min(new_og, v_other["og"])
        if verbose: print(f"og was {v['og']} should update to {new_og}")
        v["og"] = new_og
        for set_elem in same_origin_nodes:
            # print(f"updating {set_elem}")
            v_other = cx_graph.vs[set_elem]
            if v_other["og"] == -1: continue
            v_id = cx_graph.vs[v_other["og"]]
            v_id["og"] = new_og
        if verbose: print(f"og is now {v['og']} updated to {new_og}")
    # iterate union find for each element in the graph.
    for o in ordering:
        v = cx_graph.vs[o]
        # if special case or identical, skip
        if v["og"] == -1: continue
        if v["og"] == o: continue
        # otherwise track down the first identical element
        og = v["og"]
        while True:
            v_potential_og = cx_graph.vs[og]
            # found it?
            if v_potential_og["og"] == -1: continue
            if v_potential_og["og"] == og: break
            # otherwise continue down the line
            og = v_potential_og["og"]
        # go down the line again, updating the og value accordingly.
        l = v["og"]
        v["og"] = og
        while True:
            v_other = cx_graph.vs[l]
            if v_other["og"] == -1: continue
            l = v_other["og"]
            v_other["og"] = og
            if v_other["og"] == l: break
    return cx_graph


# %%
compute_primary_line_membership(cxg);

# %%
def compute_parallel_set(cxg, i):
    s = set(range(len(cxg.vs))) 
    s -= set(cxg.subcomponent(i, mode='out'))
    s -= set(cxg.subcomponent(i, mode='in'))
    # s.add(i)
    return s

def compute_all_parallel_set(cxg):
    return [compute_parallel_set(cxg, i) for i in range(len(cxg.vs))]

# %%
parallel_cxs = compute_all_parallel_set(cxg)


# %%
# Embed computational cost info
import torchinfo
import recomb.eval_costs as ec
cost_summary = torchinfo.summary(stitchnet, input_data=[X], verbose=0)
ec.embed_cost_stats_in_model(cost_summary)

# Embed 

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
import tempfile
import ray
# td = Path("<add-tmp-dir>")
# ray.init(address="auto", _temp_dir=str(td))
# ray.init(_temp_dir=str(td))
ray.init()


import asyncio

# %% Set up function evaluations using ray
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

from copy import deepcopy
# Note - if we want to evaluate multiple solutions on the same GPU, increase this.
# However, responsibility with minimizing GPU load lies solely with the user - ray
# does not manage this.
@ray.remote(num_gpus=num_gpu_per_worker)
class RayStitchEvaluator:

    def __init__(self, stitch_to_load, unfreeze_all_weights = False, train_offspring_epochs=0, train_offspring_batches=None, device="cuda"):
        self.device = torch.device(device)
        if isinstance(stitch_to_load, str):
            self.stitchnet, self.module_ordering, self.out_switch_idx = torch.load(stitch_to_load)
        else:
            self.stitchnet, self.module_ordering, self.out_switch_idx = deepcopy(stitch_to_load)
        # 
        self.unfreeze_all_weights = unfreeze_all_weights
        self.train_offspring_epochs = train_offspring_epochs
        self.train_offspring_batches = train_offspring_batches
        # This is hardcoded for now
        self.problem = problems.ImageNetProblem(root=dataset_path, validation_sample_limit=1000)
        self.problem.load_dataset()
        # Note - on every node ensure that the dataset is available.
        # self.problem.ensure_downloaded()

        for m_idx in self.module_ordering:
            m = self.stitchnet.submodules[m_idx]
            m.active = 0
            m.simplify = True

        self.output_switch = self.stitchnet.submodules[out_switch_idx]
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
        untrained_accuracy, untrained_loss = self.problem.evaluate_network(self.device, neti_os, batch_size=batch_size, objective="both")
        accuracy, loss = untrained_accuracy, untrained_loss

        result = {
            "untrained_accuracy": untrained_accuracy,
            "accuracy": accuracy,
            "untrained_loss": untrained_loss,
            "loss": loss,
            "total_memory_bytes": total_bytes,
            "total_mult_adds": total_mult_adds,
            "genotype": get_genotype_module_ordering(self.stitchnet, self.module_ordering),
            "active_variables": active_variables,
        }

        if train_offspring_epochs > 0:
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
eq = ealib.PythonAsyncIOEQ(loop)
scheduler = ealib.Scheduler(eq)
# Note - moved to top.
# population_size = 256
# num_clusters = 5
# # For how many epochs do we train the offspring network?
# train_offspring_epochs = 0
# # Do we unfreeze the network?
# unfreeze_all_weights = False


objective_indices = [0, 1]
# initializer = ealib.CategoricalUniformInitializer()
initializer = ealib.CategoricalProbabilisticallyCompleteInitializer(
    [[0.9, 0.1] if a == 2 else [1.0, 1.0, 1.0]
     for a in alphabet_size]
)
fos_learner = ealib.CategoricalLinkageTree(
    ealib.NMI(),
    ealib.FoSOrdering.Random,
    prune_root=True,
)

# acceptance_criterion = ealib.DominationObjectiveAcceptanceCriterion(objective_indices)
acceptance_criterion = ealib.ScalarizationAcceptanceCriterion(
    ealib.TschebysheffObjectiveScalarizer(objective_indices)
)
archive = ealib.BruteforceArchive(objective_indices)

# gomea = ealib.DistributedAsynchronousGOMEA(scheduler, population_size, num_clusters, objective_indices, initializer, fos_learner, acceptance_criterion, archive )
gomea = ealib.DistributedAsynchronousKernelGOMEA(scheduler, population_size, num_clusters, objective_indices, initializer, fos_learner, acceptance_criterion, archive,
   lkg_strategy=ealib.LKStrategy.RAND_ASYM )

# %%

def noop(genotype):
    return 0.0

print(alphabet_size)
problem_template = ealib.DiscreteObjectiveFunction(noop, l, alphabet_size, 0)

if number_of_workers is None:
    ray_cluster_resources = ray.cluster_resources()
    # print(ray_cluster_resources)
    number_of_workers = int(ray_cluster_resources["GPU"] / num_gpu_per_worker)
    print(f"Using {number_of_workers} parallel evaluators (#gpus available / resources req)")

eval_pool = AsyncActorPool([
    RayStitchEvaluator.remote(active_stitch,
                              unfreeze_all_weights=unfreeze_all_weights,
                              train_offspring_epochs=train_offspring_epochs,
                              train_offspring_batches=train_offspring_batches)
    for _ in range(number_of_workers)
])

num_pending = 0
num_evaluated = 0

async def evaluate_genotype_using_RayStitchEvaluator(e: RayStitchEvaluator, g: np.array):
    # print(f"Evaluating {g} - actor allocated")
    result = await e.evaluate_genotype.remote(g)
    print(f"Evaluated {g} - {result}")
    return result

evaluated_solutions = []

async def evaluate_objective(p: ealib.Population, i: ealib.Individual):
    global num_evaluated
    global num_pending
    global evaluated_solutions

    genotype = p.getData(ealib.GENOTYPECATEGORICAL, i)
    genotype_array = np.asarray(genotype)

    pyd = p.getData(ealib.PYDATA, i)
    if "active_variables" in pyd.data and "previous_genotype" in pyd.data:
        previous_genotype = pyd.data["previous_genotype"]
        changed_variables = {i for i, (v_a, v_b) in enumerate(zip(genotype_array, previous_genotype)) if v_a != v_b}

        active_variables = pyd.data["active_variables"]
        changed_and_active = changed_variables.intersection(active_variables)
        # If no active variables have changed - the evaluation will not have changed either.
        if len(changed_and_active) == 0:
            print(f"Changed variables are: {changed_variables}.")
            print(f"Active variables are: {active_variables}.")
            print(f"Intersection therefore is: {changed_and_active}, which is empty.")
            print("Skipping evaluation of solution: no changes to active variables.")
            return

    # Update genotype for the next time we do a check :)
    pyd.data["previous_genotype"] = np.copy(genotype_array)

    # print(f"Got a request to evaluate {genotype_array} - adding to queue")

    # print(f"Got genotype {genotype_array}")
    # if num_evaluated > evaluation_budget:
    #     print("Budget reached!")
    #     raise ealib.EvaluationLimit()

    # ask ray to evaluate it over a pool of actors
    num_pending += 1
    # print(f"Requesting resources from eval pool {genotype_array}")
    result = await eval_pool.perform(evaluate_genotype_using_RayStitchEvaluator, genotype_array)
    # print(f"Pool has completed work for {genotype_array}")
    # Update set of active variables
    pyd.data["active_variables"] = result["active_variables"]
    # Turn into list for adding to df - sets aren't a natively supported datatype.
    # A list is close enough!
    result["active_variables"] = list(result["active_variables"])
    evaluated_solutions.append(result)

    # Update dataframe to include new evaluated solution
    # Sidenote - we might want to do this incrementally.
    pl.DataFrame(evaluated_solutions).write_ipc(eval_out_path)
    
    # print(f"Increasing number of evaluations")
    num_evaluated += 1

    # result contains "accuracy" "loss" "total_mult_adds" "total_memory_bytes"

    # print(f"Setting objective values")
    # Store result - note - minimization is uniformly assumed.
    objective = p.getData(ealib.OBJECTIVE, i)
    # accuracy should be maximized (flip sign)
    objective.set_objective(0, result["loss"])
    # cost should be minimized (keep as-is)
    objective.set_objective(1, result["total_mult_adds"])

    # print("Reducing num pending")
    num_pending -= 1

    # print(f"Evaluation completed for {genotype_array} - left: {num_pending}")

    # if num_evaluated > evaluation_budget:
    #     print("Budget reached!")
    #     raise ealib.EvaluationLimit()

objective_fn = ealib.PyAsyncObjectiveFunction(scheduler, problem_template, evaluate_objective)
objective_fn = ealib.Limiter(objective_fn, evaluation_limit=evaluation_budget)
# %%

seed = 42

stepper = ealib.TerminationStepper((lambda : gomea), None)
f = ealib.SimpleConfigurator(objective_fn, stepper, seed)
# This experiment utilizes extra python data to track active variables & which variables have changed
# as to skip changes involving no active variables.
# This is not registered by default - so request it :)
pop = f.getPopulation()
pop.registerData(ealib.PYDATA)

f.run()

torch.save(evaluated_solutions, "evaluated_solutions.pckl")
pl.DataFrame(evaluated_solutions).write_ipc("evaluated_solutions_gomea.arrow")
