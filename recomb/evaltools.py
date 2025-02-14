#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

import asyncio

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


import ray
import torch
from copy import deepcopy
from . import cx
from . import eval_costs as ec
from .problems import NeuralNetIndividual
# Note - configure the number of gpus required when creating a pool
@ray.remote(num_gpus=1)
class RayStitchEvaluator:
    def __init__(self,
                 problem_constructor,
                 stitch_to_load,
                 unfreeze_all_weights = False,
                 train_offspring_epochs=0,
                 train_offspring_batches=None,
                 train_lr=1e-4,
                 eval_batch_size=1,
                 train_batch_size=1,
                 device="cuda",
                 ):
        self.device = torch.device(device)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_lr = train_lr
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
        untrained_accuracy, untrained_loss = self.problem.evaluate_network(self.device, neti_os, batch_size=self.eval_batch_size, objective="both")
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
                                    batch_size=self.train_batch_size,
                                    lr=self.train_lr,
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
                trained_accuracy, trained_loss = self.problem.evaluate_network(self.device, neti_os, batch_size=self.eval_batch_size, objective="both")

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

async def evaluate_genotype_using_RayStitchEvaluator(e: RayStitchEvaluator, g):
    result = await e.evaluate_genotype.remote(g)
    return result