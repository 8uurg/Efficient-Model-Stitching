import torch
import recomb.problems as problems
from recomb.problems import NeuralNetIndividual
import recomb.cx as cx
import recomb.eval_costs as ec
import numpy as np
from pathlib import Path
from copy import deepcopy
import polars as pl

reference_output = "./stitched-imagenet-b-a-resnet50-b-resnext50_32x4d-reference.arrow"
problem_constructor = lambda : problems.ImageNetProblem("/export/scratch2/constellation-data/arthur/", 1000)
problem = problem_constructor()

number_of_workers = None # determine by #gpus in cluster
num_gpu_per_worker = 1.0
batch_size = 8
dataset_path = "/export/scratch2/constellation-data/arthur/"
# stitched_network_path = "./stitched-imagenet.th"
stitched_network_path = "./stitched-imagenet-b-a-resnet50-b-resnext50_32x4d.th"

# For how many epochs do we train the offspring network?
train_offspring_epochs = 0 # 1?
# Note - a single epoch for imagenet is very large - limit the number of batches used.
train_offspring_batches = 1000
# Do we unfreeze the network?
unfreeze_all_weights = True
train_lr = 1e-4

# Load a sample for determining computational costs.
from torch.utils.data import DataLoader
# d_train, _, _ = problem.load_problem_dataset(transform_train=trsf, transform_validation=trsf)
d_train = problem.get_dataset_train()
# d_train.transform = trsf
dl_train = DataLoader(d_train)
it_train = iter(dl_train)
X, Y = next(it_train)

# Load stitch
stitchnet, stitchinfo = torch.load(stitched_network_path)

# Determine module ordering
module_ordering, out_switch_idx = cx.compute_cx_module_ordering_and_out_switch(stitchnet, stitchinfo)
# Embed computational costs
import torchinfo
import recomb.eval_costs as ec
cost_summary = torchinfo.summary(stitchnet, input_data=[X], verbose=0)
ec.embed_cost_stats_in_model(cost_summary)
# Construct stitch with embedded info.
active_stitch = (stitchnet, module_ordering, out_switch_idx)

# Start up ray.
import ray
# td = Path("<add-tmp-dir>")
# ray.init(address="auto", _temp_dir=str(td))
# ray.init(_temp_dir=str(td))
ray.init()

if number_of_workers is None:
    ray_cluster_resources = ray.cluster_resources()
    # print(ray_cluster_resources)
    number_of_workers = int(ray_cluster_resources["GPU"] / num_gpu_per_worker)
    print(f"Using {number_of_workers} parallel evaluators (#gpus available / resources req)")

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
        self.problem = problem_constructor()
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
        print(f"Evaluating {genotype}")
        cx.set_genotype_module_ordering(self.stitchnet, self.module_ordering, genotype)
        # Hardcoded: For output switch, do not simplify if active = 2.
        # This choice picks between the ensemble & individual networks, and if we pick
        # the ensemble, we wish to be able to train both branches seperately.
        self.output_switch.simplify = self.output_switch.active != 2

        assert (list(cx.get_genotype_module_ordering(self.stitchnet, self.module_ordering)) == list(genotype)), "could not apply genotype - disassociated info & network."
        # Prune graph
        stitchnet_pruned = self.stitchnet.to_graph()
        stitchnet_pruned.prune_unused()

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
            "genotype": cx.get_genotype_module_ordering(self.stitchnet, self.module_ordering)
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

alphabet_size = cx.get_stitcher_genotype_cardinality(stitchnet, stitchinfo)
l = len(alphabet_size)

# Reference solutions to evaluate
reference_solutions = []
for i in range(2 + 1):
    s = np.zeros(l, int)
    s[-1] = i
    reference_solutions.append(s)

# AsyncActorPool
import asyncio
from recomb.evaltools import AsyncActorPool

eval_pool = AsyncActorPool([
    RayStitchEvaluator.remote(active_stitch,
                              unfreeze_all_weights=unfreeze_all_weights,
                              train_offspring_epochs=train_offspring_epochs,
                              train_offspring_batches=train_offspring_batches)
    for _ in range(number_of_workers)
])

async def evaluate_genotype_using_RayStitchEvaluator(e: RayStitchEvaluator, g: np.array):
    result = await e.evaluate_genotype.remote(g)
    return result

evaluated_solutions = []
async def evaluate_objective(genotype):
    global evaluated_solutions

    genotype_array = genotype

    # evaluate it over a pool of actors
    result = await eval_pool.perform(evaluate_genotype_using_RayStitchEvaluator, genotype_array)
    # print(f"Pool has completed work for {genotype_array}")
    evaluated_solutions.append(result)
    # Update dataframe to include new evaluated solution
    pl.DataFrame(evaluated_solutions).write_ipc(reference_output)

# Wait...
async def main(): 
    o = [evaluate_objective(r) for r in reference_solutions]
    o = await asyncio.gather(*o)
asyncio.run(main())

