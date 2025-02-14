#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

import os
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import ray
import torch
from ray import ObjectRef
from torch.utils.data import DataLoader

from . import cx, problems

default_num_gpus_per_task = 0.50
reuse_gpu_workers = os.environ.get("RECOMB_REUSE_GPU_WORKERS", "0") == "1"

max_calls_gpu_task = 0 if reuse_gpu_workers else 1



@ray.remote(num_cpus=0)
class Population:
    def __init__(self, population_size: int, log_updates: Optional[Path] =None):
        self.population = [None for _ in range(population_size)]
        self.rng = np.random.default_rng(seed=42)
        self.log_updates = log_updates
        if self.log_updates is not None:
            self.log_updates.mkdir(parents=True, exist_ok=True)
        self.update_counter = 0
        self.initial_time = time.time()
        self.num_evaluations_started = 0
        self.num_evaluations_completed = 0

    # hack to track # of evaluations in some sort of fashion.
    def start_eval(self):
        self.num_evaluations_started += 1

    def complete_eval(self, fitness):
        self.num_evaluations_completed += 1
        if self.log_updates is not None:
            with open(self.log_updates / "evaluations.jsonl", 'a') as f:
                f.write((
                    "{"
                    f'"es": {self.num_evaluations_started}, '
                    f'"ec": {self.num_evaluations_completed}, '
                    f'"f": {fitness}'
                    "}\n"
                ))

    def set(self, idx, value):
        self.population[idx] = value
        if self.log_updates is not None:
            with open(self.log_updates / "population.jsonl", 'a') as f:
                value.log_update(idx, self.num_evaluations_started, self.num_evaluations_completed, self.update_counter, f, self.log_updates, self.initial_time)
        # assign indices to accepted population changes
        self.population[idx].pop_uid = self.update_counter
        self.update_counter += 1

    def append(self, value):
        self.population.append(value)

    def get(self, idx, accessor=None):
        if accessor is None:
            return self.population[idx]
        else:
            return accessor(self.population[idx])

    def len(self):
        return len(self.population)

    def get_ref(self, idx: int):
        return ray.put(self.population[idx])

    def get_current_refs(self):
        return [ray.put(e) for e in self.population]

    def get_random(self):
        random_index = self.rng.integers(len(self.population))
        return random_index, self.population[random_index]

    def act_lock(self, act: "PopulationAction", value):
        # Perform action on population as part of the actor, effectly locking the entire population from any changes.
        return act.act_lock(self, value)

PopulationActor = ray.ObjectRef# [Population]

class PopulationAction(ABC):
    @abstractmethod
    def act(self, population: PopulationActor, value):
        raise NotImplementedError

    @abstractmethod
    def act_lock(self, population: Population, value): # type: ignore
        raise NotImplementedError


def simple_comparator(a, b):
    if a == b:
        return 0b00
    if a is None:
        return 0b10
    if b is None:
        return 0b01
    return 2 - (a > b)

class PAReplaceAtIdxIfBetter(PopulationAction):
    def __init__(self, idx: int, fitness_comparator=None, verbose=False):
        self.idx = idx
        self.verbose = verbose
        # fitness_comparator()
        # 0b00 - equal
        # 0b10 - a is better
        # 0b01 - b is better
        # 0b11 - incomparable
        if fitness_comparator is not None:
            self.fc = fitness_comparator
        else:
            self.fc = simple_comparator

    def act(self, population: PopulationActor, value):
        # This action needs to perform locking we do not want replacement to happen in between.
        population.act_lock.remote(self, value) # type: ignore

    def act_lock(self, population: Population, value): # type: ignore
        # Compare fitnesses and replace if better.
        # - Annoyingly enough, this transfers value even when not necessary...
        # - Maybe a get & compare under act above is not a bad idea to save a larger
        #   data transfer as an optimisation.
        current_fitness = population.get(self.idx).fitness # type: ignore
        new_fitness = value.fitness
        if self.verbose: print(f"got solution with fitness {new_fitness} to potentially replace {current_fitness}")
        if self.fc(current_fitness, new_fitness) == 0b10:
            population.set(self.idx, value)

class PAReplaceWorstIdxIfBetter(PopulationAction):
    def __init__(self, idxs: List[int], fitness_comparator=None, verbose=False):
        self.idxs = idxs
        self.verbose = verbose
        # fitness_comparator()
        # 0b00 - equal
        # 0b10 - a is better
        # 0b01 - b is better
        # 0b11 - incomparable
        if fitness_comparator is not None:
            self.fc = fitness_comparator
        else:
            self.fc = simple_comparator

    def act(self, population: PopulationActor, value):
        # This action needs to perform locking we do not want replacement to happen in between.
        population.act_lock.remote(self, value) # type: ignore

    def act_lock(self, population: Population, value): # type: ignore
        # Compare fitnesses and replace if better.
        # - Annoyingly enough, this transfers value even when not necessary...
        # - Maybe a get & compare under act above is not a bad idea to save a larger
        #   data transfer as an optimisation.

        worst_idx = None
        worst_fitness = None
        idx_fitnesses = []
        for idx in self.idxs:
            current_fitness = population.get(idx).fitness
            idx_fitnesses.append(current_fitness)
            if worst_fitness is None or self.fc(current_fitness, worst_fitness) == 0b10:
                worst_idx = idx
                worst_fitness = current_fitness

        new_fitness = value.fitness
        if self.verbose: print(f"got solution with fitness {new_fitness} to potentially replace {worst_fitness} out of {idx_fitnesses}")
        if self.fc(worst_fitness, new_fitness) == 0b10:
            population.set(worst_idx, value)

class NoOpPopulationAction(PopulationAction):
    def act(self, population: PopulationActor, value):
        pass
    def act_lock(self, population: Population, value): # type: ignore
        pass

def clean_gpu(dev):
    if dev.type == "cuda":
        # clear cublas workspaces - tends to be large after inference & backprop.
        torch._C._cuda_clearCublasWorkspaces() # type: ignore
        # empty cache - free all of the memory (apart from required bookkeeping for CUDA)
        torch.cuda.empty_cache()

@ray.remote(num_gpus=default_num_gpus_per_task, max_calls=max_calls_gpu_task)
def train_network_and_act_when_trained(
    problem: problems.NASProblem,
    population: PopulationActor,
    act: PopulationAction,
    net: problems.NeuralNetIndividual,
    seed: Optional[int]=None,
    params: dict = {},
    verbose = False,
    allow_training_layer_kind="cx",
    shrink_perturb=False,
    force_train=True,
    profile_cost=True,
):
    if shrink_perturb == True:
        # Let w be the original weights and w' be reinitialized weights
        # then given (f_s, f_p), the resulting weights should be
        # w_new = f_s * w + f_p * w'
        shrink_perturb = (0.4, 0.1)
    if isinstance(shrink_perturb, Tuple):
        # as the perturb action below effectively multiplies f_s by (1 - f_p)
        # correct for that here.
        f_shrink = shrink_perturb[0] / (1 - shrink_perturb[1])
        f_perturb = shrink_perturb[1]

    # (default probabilities)
    p_merge_identical_submodules = 1.0
    p_merge_mergable_operations = 1.0
    if net.is_trained and not force_train:
        # network is already trained, act immidiately
        act.act(population, net)
        return net
    
    # potentially allow more parameters to be trained at this point
    cx.allow_training_kind(net.net, kind=allow_training_layer_kind)

    # shrink-perturb parameters for trainable layers
    if not shrink_perturb == False:
        sm = net.net.submodules
        for m in sm:
            if not cx.has_gradients(m): continue
            with torch.no_grad():
                for p in m.parameters():
                    p *= f_shrink # type: ignore

            m.perturb_self(f_perturb) # type: ignore

    if verbose: print(f"training network...")
    
    t_start = time.time()
    dev = torch.device("cuda")
    # Training hyperparameters could be introduced here, potentially as part of 'net' itself.
    train_args_params = {"num_batches", "num_epochs", "lr", "weight_decay", "optimizer", "batch_size"}
    training_params = {a: b for a, b in params.items() if a in train_args_params}
    try:
        net_trained = problem.train_network(dev, net, seed=seed, **training_params)
    except Exception as e:
        if verbose: print(f"training failed with {e}")
        # if training fails... maybe no gradients?
        net_trained = net.net.cpu()

    
    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"training network took {td}s")

    t_start = time.time()
    rng = np.random.default_rng(seed=seed)
    p_merge_identical_submodules = params.get('p_merge_identical_submodules', p_merge_identical_submodules)
    p_merge_mergable_operations = params.get('p_merge_mergable_operations', p_merge_mergable_operations)
    do_module_remap = params.get('merge_identical_submodules_remap_modules', False)
    if rng.random() < p_merge_identical_submodules:
        net_trained = cx.merge_identical_submodules(net_trained, remap_modules=do_module_remap)
        net_trained.cpu()
    if rng.random() < p_merge_mergable_operations:
        # :D
        net_trained = cx.merge_mergable_operations(net_trained, multipass=True)
        # really - do this on the cpu.
        net_trained.cpu()

    trained_individual = net.as_trained(net_trained, True)
    
    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"simplifying network took {td}s")

    act.act(population, trained_individual)

    clean_gpu(dev)

    return trained_individual


@ray.remote(num_gpus=default_num_gpus_per_task, max_calls=max_calls_gpu_task)
def evaluate_network_and_act(
    problem: problems.NASProblem,
    population: PopulationActor,
    act: PopulationAction,
    net: problems.NeuralNetIndividual,
    verbose = False,
    profile_cost = True,
    force_evaluation=True,
):
    if net.fitness is not None and not force_evaluation:
        # Network has already been evaluated...
        return net

    if verbose: print("evaluating network...")
    dev = torch.device("cuda")
    t_start = time.time()
    fitness = problem.evaluate_network(dev, net)
    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"evaluating network took {td}s")

    evaluated_neural_net = net.as_evaluated(fitness)

    act.act(population, evaluated_neural_net)

    clean_gpu(dev)

    return evaluated_neural_net

@ray.remote(num_gpus=default_num_gpus_per_task, max_calls=max_calls_gpu_task, num_returns=2)
def evaluate_network_and_bireturn(
    problem: problems.NASProblem,
    net: problems.NeuralNetIndividual,
    verbose = False,
    profile_cost = True,
):
    if verbose: print("evaluating network...")
    dev = torch.device("cuda")
    t_start = time.time()
    fitness = problem.evaluate_network(dev, net)
    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"evaluating network took {td}s")

    evaluated_neural_net = net.as_evaluated(fitness)

    clean_gpu(dev)

    return fitness, evaluated_neural_net


@ray.remote(num_gpus=1, max_calls=max_calls_gpu_task)
def train_cx_network_and_sample_act(
    problem: problems.NASProblem,
    population: PopulationActor,
    net_a: problems.NeuralNetIndividual, # ray.ObjectRef[NeuralNetIndividual]
    net_b: problems.NeuralNetIndividual, # ray.ObjectRef[NeuralNetIndividual]
    seed: int,
    num_offspring=10,
    is_offspring_trained=False,
    feature_shape_should_match=False,
    cx_style="ux",
    train_cx_network=True,
    parent_idxs=[],
    verbose=False,
    lazy_mode=False,
    profile_cost=True,
):
    dev = torch.device("cuda")
    #
    d_train = problem.get_dataset_train()
    X_in_many, y_out_many = next(iter(DataLoader(d_train, batch_size=128)))

    # print(f"fetching networks for recombination...")
    # Request the networks (download if objectref)
    # - note that this works fine, despite what mypy is saying.
    # net_a, net_b = ray.get([net_a_maybe, net_b_maybe]) # type: ignore

    if verbose: print(f"recombining networks with fitnesses {net_a.fitness} & {net_b.fitness}")

    
    # Train & construct CX network.
    t_start = time.time()

    compute_similarity = cx.compute_linear_cka_feature_map_similarity
    num_X_samples = None
    # alternative
    if lazy_mode:
        compute_similarity = cx.compute_mock_similarity
        num_X_samples = 1

    cx_net_m, cx_net = cx.construct_trained_cx_network(
        d_train,
        dev,
        net_a.net,
        net_b.net,
        X_in_many=X_in_many,
        filter_has_parameters=False,
        feature_shape_should_match=feature_shape_should_match,
        train_cx_network=train_cx_network,
        parent_idxs=parent_idxs,
        compute_similarity=compute_similarity,
        num_X_samples=num_X_samples,
    )
    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"constructing cx network took {td}s")

    rng = np.random.default_rng(seed=seed)

    if verbose: print(f"sampling offspring...")

    num_offspring_generated = 0

    while num_offspring_generated < num_offspring:

        genotypes = cx_net.get_random_genotypes(rng, cx_style=cx_style)
        for genotype in genotypes:
            # early exit if we already have sufficient offspring
            if not num_offspring_generated < num_offspring: break

            net_o = cx_net.get_offspring_from_genotype(genotype)
            # print(f"sampling offspring #{oid}")
            # note - using population uids instead
            net_o_obj = NeuralNetIndividual(net_o, None, is_offspring_trained, parent_idxs=[net_a.pop_uid, net_b.pop_uid]) # type: ignore
            num_offspring_generated += 1

            yield net_o_obj

    del cx_net_m
    del cx_net

    clean_gpu(dev)

    # print(f"done sampling!")
    # Wait for all of the evaluations to complete to call this task complete.
    # Annoyingly - this causes a deadlock as we are still consuming this tasks' resources.
    # ray.wait(tasks, num_returns=len(tasks), fetch_local=False)
    # ray.get

@ray.remote(num_cpus=0)
def await_returned_results(tasks):
    tasks = list(tasks)
    # tasks, _ = ray.wait(tasks, num_returns=len(tasks), fetch_local=False)
    # print(f"waiting...")
    while len(tasks) > 0:
        ready, tasks = ray.wait(tasks)
        ray.get(ready)

@ray.remote
def truncate_preselect(num_select, fitnesses, networks):
    fitnesses = ray.get(fitnesses)
    # find the ordering that sorts the fitnesses
    sorted_idxs = np.argsort(fitnesses)
    # since we want to go from large to small, reverse the order
    sorted_idxs = sorted_idxs[::-1]
    if num_select == 1:
        return ray.get(networks[sorted_idxs[0]])
    return tuple([ray.get(networks[i]) for i in sorted_idxs[:num_select]])

class RayTracker(problems.BaseTracker):

    def __init__(self, remote_tracker):
        self.remote_tracker = remote_tracker

    def start_eval(self):
        ray.get(self.remote_tracker.start_eval.remote())

    def complete_eval(self, state):
        ray.get(self.remote_tracker.start_eval.remote(state))

class NASGA:
    """
    A simple genetic algorithm for neural architecture search.
    """

    def __init__(
        self,
        problem: problems.NASProblem,
        rng: np.random.Generator,
        population_size: int,
        sync=False,
        max_pending_tasks: Optional[int] = None,
        num_offspring = 2,
        id: Optional[int] = None,
        cx_style = "1px",
        config: Dict[str, Any] = {},
        verbose = False,
    ):
        self.problem = problem
        self.rng = rng
        self.population_size = population_size
        self.max_pending_tasks = (
            population_size if max_pending_tasks is None else max_pending_tasks
        )
        self.num_pending_tasks = 0
        self.sync = sync

        self.initialized = False

        # Generate a random id if none provided.
        if id is None:
            self.id = self.rng.integers(np.iinfo(np.int32).max)
        else:
            self.id = id

        self.config = config
        # Can be more or less, with the current approach, more is generally favourable.
        self.num_offspring = num_offspring

        self.initial_train_params = config.get("initial_train_params", {})
        self.cx_train_params = config.get("cx_train_params", {})

        self.cx_style = cx_style
        self.shrink_perturb = config.get("shrink_perturb", False)
        self.verbose = verbose


    def train_evaluate(self, idx: int, net: problems.NeuralNetIndividual):
        # get a seed value
        seed = self.rng.integers(np.iinfo(np.int32).max)
        # Set the individual to get started
        self.population.set.remote(idx, net)
        # Note the remote here - this requires a GPU, and needs to be performed
        # separately, this function no-ops if no training is required.
        f_net_trained = train_network_and_act_when_trained.remote(
            self.problem,
            self.population,
            PAReplaceAtIdxIfBetter(idx),
            net,
            seed=seed, # type: ignore
            params=self.initial_train_params, # type: ignore
            shrink_perturb=self.shrink_perturb, # type: ignore
        )
        f_net_evaluated = evaluate_network_and_act.remote(
            self.problem, self.population, PAReplaceAtIdxIfBetter(idx), f_net_trained
        )
        return f_net_evaluated

    def wait_for_task_limit(self):
        # Without backpressure we will simply end up generating tasks continuously based on outdated data
        # never waiting for anything to complete... Instead, we opt to wait until the number of pending
        # tasks drops below a particular level.
        while len(self.pending_tasks) >= self.max_pending_tasks:
            ready_tasks, remaining_pending_tasks = ray.wait(self.pending_tasks)
            self.pending_tasks = remaining_pending_tasks

            for t in ready_tasks:
                v = ray.get(t)
                if self.verbose: print(f"completed task {t}: {v}")
            # self.num_pending_tasks -= len(ready_tasks)

    def synchronize(self):
        while len(self.pending_tasks) > 0:
            ready, self.pending_tasks = ray.wait(
                self.pending_tasks,
                # fetch_local=False,
            )
            for t in ready:
                v = ray.get(t)
                if self.verbose: print(f"completed task {t}: {v}")
        # self.pending_tasks.clear()

    def maybe_synchronize(self):
        if self.sync:
            self.synchronize()

    def generate_initial_solution(self, idx):
        net = self.problem.sample_architecture(self.rng)
        return self.train_evaluate(idx, net)

    def initialize(self):
        self.population = Population.options(name=f"population-{self.id}").remote( # type: ignore
            self.population_size,
            log_updates=self.config.get("expdir")
        )

        self.problem.set_tracker(RayTracker(self.population))
        
        # Start initializing.
        self.pending_tasks = [
            self.generate_initial_solution(idx)
            for idx in range(self.population_size)
        ]
        # self.num_pending_tasks += len(self.pending_tasks)

        # Set initialized to true
        self.initialized = True
        # - note that this is somewhat misleading though!

        # Note that we continue asynchronously without waiting for the actual training & evaluation to complete.
        # To operate synchronously, we wait on /all/ pending tasks to complete here.
        self.maybe_synchronize()

    def recombine_train_evaluate(self):
        rng = self.rng
        # rng = np.random.default_rng(seed=seed)
        idxs = rng.permutation(ray.get(self.population.len.remote()))

        for s in range(0, len(idxs), 2):
            self.wait_for_task_limit()
            idx_a, idx_b = idxs[s], idxs[s + 1]
            replacement_act = PAReplaceWorstIdxIfBetter([idx_a, idx_b])
            # get network references
            net_a = self.population.get.remote(idx_a)
            net_b = self.population.get.remote(idx_b)
            seed = self.rng.integers(np.iinfo(np.int32).max)

            tasks = self.recombine(rng, idx_a, idx_b, replacement_act, net_a, net_b, seed)
            self.pending_tasks.append(await_returned_results.remote(tasks)) # type: ignore

    def recombine(self, rng, idx_a, idx_b, replacement_act, net_a, net_b, seed):
        if self.cx_style == "psa" or self.cx_style == "psb" or self.cx_style == "psr":
            # 'parent select' crossover - no constructing a CX network here.
            parent_net = None
            if self.cx_style == "psa": parent_net = net_a
            if self.cx_style == "psb": parent_net = net_b
            if self.cx_style == "psr": parent_net = self.rng.choice([net_a, net_b])

            # Always create a singular offspring for this mode - the others would simply be redundant copies
            net_o_obj = deepcopy(parent_net)
            net_o_obj.parent_idxs = [parent_net.pop_uid] # type: ignore

            # always use all in this configuration - there is no cx point here.
            allow_training_layer_kind_options = ["all"]
            # allow_training_layer_kind_options = self.config.get("allow_training_layer_kind_options", allow_training_layer_kind_options)
            # allow_training_layer_kind = rng.choice(allow_training_layer_kind_options)
            allow_training_layer_kind = allow_training_layer_kind_options[0]

            # (note, forcing training & evaluation as we have a literal copy here, though potentially on a different data split)
            net_o_obj = train_network_and_act_when_trained.remote(
                self.problem,
                self.population,
                NoOpPopulationAction(),
                net_o_obj,
                seed=seed, # type: ignore
                params=self.cx_train_params, # type: ignore
                allow_training_layer_kind=allow_training_layer_kind, # type: ignore
                shrink_perturb=self.shrink_perturb, # type: ignore
                force_train=True, # type: ignore
                )
            net_o_obj = evaluate_network_and_act.remote(self.problem, self.population, replacement_act, net_o_obj, force_evaluation=True) # type: ignore

            return [net_o_obj]
            
        tasks = []
        train_cx_only_first=self.config.get("train_cx_only_first", False)
        sample_train_num_epochs=self.config.get("sample_train_num_epochs", 0)
        lazy_mode=self.config.get("lazy_mode", False)
        do_retrain_offspring = sample_train_num_epochs > 0
        offspring = train_cx_network_and_sample_act.options(num_returns=self.num_offspring).remote(self.problem, self.population, net_a, net_b, seed, self.num_offspring, is_offspring_trained=not do_retrain_offspring, cx_style=self.cx_style, parent_idxs=[idx_a, idx_b], lazy_mode=lazy_mode) # type: ignore
        if self.num_offspring == 1:
            # wrap in a list so that the loop works...
            offspring = [offspring]

        number_preselected = self.config.get("number_preselected", None)
        # if preselection is none, don't perform any.
        if number_preselected is not None:
            # prior to training & evaluating networks, first ONLY evaluate them
            f_fitnesses = []
            f_networks = []
            for net_o_obj in offspring:
                f_fitness, f_network = evaluate_network_and_bireturn.remote(self.problem, net_o_obj) # type: ignore
                f_fitnesses.append(f_fitness)
                f_networks.append(f_network)
            offspring = truncate_preselect.options(num_returns=number_preselected).remote(number_preselected, f_fitnesses, f_networks ) # type: ignore
            if number_preselected == 1:
                offspring = [offspring]

        for net_o_obj in offspring:
            seed = self.rng.integers(np.iinfo(np.int32).max)
            if train_cx_only_first and sample_train_num_epochs > 0:
                allow_training_layer_kind = "cx"
                net_o_obj = train_network_and_act_when_trained.remote(self.problem, self.population, NoOpPopulationAction(), net_o_obj, seed=seed, params=self.cx_train_params, allow_training_layer_kind=allow_training_layer_kind, shrink_perturb=self.shrink_perturb, force_train=True) # type: ignore

            if sample_train_num_epochs > 0:
                allow_training_layer_kind_options = ["cx"]
                allow_training_layer_kind_options = self.config.get("allow_training_layer_kind_options", allow_training_layer_kind_options)
                allow_training_layer_kind = rng.choice(allow_training_layer_kind_options)
                net_o_obj = train_network_and_act_when_trained.remote(self.problem, self.population, NoOpPopulationAction(), net_o_obj, seed=seed, params=self.cx_train_params, allow_training_layer_kind=allow_training_layer_kind, shrink_perturb=self.shrink_perturb, force_train=True) # type: ignore

            net_o_obj = evaluate_network_and_act.remote(self.problem, self.population, replacement_act, net_o_obj, force_evaluation=True) # type: ignore
            tasks.append(net_o_obj)
        return tasks

    def generation(self):
        self.recombine_train_evaluate()

        # Note - if asynchronous, generations will overlap!
        if self.sync:
            self.synchronize()

    def step(self):
        if self.initialized:
            self.generation()
        else:
            self.initialize()

    

class MultiDatasetNASGA:
    """
    A simple GA for NAS, which deals with distributed training, too.
    """

    def __init__(
        self,
        problems: List[problems.NASProblem],
        rng: np.random.Generator,
        population_size: int,
        sync=False,
        max_pending_tasks: Optional[int] = None,
        num_offspring = 2,
        id: Optional[int] = None,
        cx_style = "1px",
        config: Dict[str, Any] = {},
        verbose = False,
    ):
        self.problems = problems
        self.rng = rng
        self.population_size = population_size
        self.max_pending_tasks = (
            population_size if max_pending_tasks is None else max_pending_tasks
        )
        self.num_pending_tasks = 0
        self.sync = sync

        self.initialized = False

        # Generate a random id if none provided.
        if id is None:
            self.id = self.rng.integers(np.iinfo(np.int32).max)
        else:
            self.id = id

        self.config = config
        # Can be more or less, with the current approach, more is generally favourable
        # as the overhead of preparing a CX network is relatively high.
        self.num_offspring = num_offspring

        self.initial_train_params = config.get("initial_train_params", {})
        self.cx_train_params = config.get("cx_train_params", {})
        self.cx_only_train_params = config.get("cx_only_train_params", self.cx_train_params)

        self.cx_style = cx_style
        self.verbose = verbose
        self.shrink_perturb = config.get("shrink_perturb", False)


    def train_evaluate(self, idx: int, net: problems.NeuralNetIndividual):
        # get a seed value
        seed = self.rng.integers(np.iinfo(np.int32).max)
        # task_idx = seed % len(self.problems)
        task_idx = idx % len(self.problems)
        problem = self.problems[task_idx]
        # Set the individual to get started
        self.population.set.remote(idx, net)
        # Note the remote here - this requires a GPU, and needs to be performed
        # separately, this function no-ops if no training is required.
        f_net_trained = train_network_and_act_when_trained.remote(
            problem, self.population, PAReplaceAtIdxIfBetter(idx), net, seed=seed, params=self.initial_train_params, shrink_perturb=self.shrink_perturb # type: ignore
        )
        f_net_evaluated = evaluate_network_and_act.remote(
            problem, self.population, PAReplaceAtIdxIfBetter(idx), f_net_trained
        )
        return f_net_evaluated

    def wait_for_task_limit(self):
        # Without backpressure we will simply end up generating tasks continuously based on outdated data
        # never waiting for anything to complete... Instead, we opt to wait until the number of pending
        # tasks drops below a particular level.
        while len(self.pending_tasks) >= self.max_pending_tasks:
            ready_tasks, remaining_pending_tasks = ray.wait(self.pending_tasks)
            self.pending_tasks = remaining_pending_tasks

            for t in ready_tasks:
                v = ray.get(t)
                if self.verbose: print(f"completed task {t}: {v}")
            # self.num_pending_tasks -= len(ready_tasks)

    def synchronize(self):
        while len(self.pending_tasks) > 0:
            ready, self.pending_tasks = ray.wait(
                self.pending_tasks,
                # fetch_local=False,
            )
            for t in ready:
                v = ray.get(t)
                if self.verbose: print(f"completed task {t}: {v}")
        # self.pending_tasks.clear()

    def maybe_synchronize(self):
        if self.sync:
            self.synchronize()

    def generate_initial_solution(self, idx):
        # find task
        task_idx = idx % len(self.problems)
        problem = self.problems[task_idx]
        # sample the network
        net = problem.sample_architecture(self.rng)
        return self.train_evaluate(idx, net)

    def initialize(self):
        self.population = Population.options(name=f"population-{self.id}").remote( # type: ignore
            self.population_size,
            log_updates=self.config.get("expdir")
        )

        for problem in self.problems:
            problem.set_tracker(RayTracker(self.population))
        
        # Start initializing.
        self.pending_tasks = [
            self.generate_initial_solution(idx)
            for idx in range(self.population_size)
        ]
        # self.num_pending_tasks += len(self.pending_tasks)

        # Set initialized to true
        self.initialized = True
        # - note that this is somewhat misleading though!

        # Note that we continue asynchronously without waiting for the actual training & evaluation to complete.
        # To operate synchronously, we wait on /all/ pending tasks to complete here.
        self.maybe_synchronize()

    def recombine_train_evaluate(self):
        rng = self.rng
        # rng = np.random.default_rng(seed=seed)
        idxs = rng.permutation(ray.get(self.population.len.remote()))

        for s in range(0, len(idxs), 2):
            self.wait_for_task_limit()
            idx_a, idx_b = idxs[s], idxs[s + 1]

            replacement_act = PAReplaceWorstIdxIfBetter([idx_a]) # , idx_b
            # get network references
            net_a = self.population.get.remote(idx_a)
            net_b = self.population.get.remote(idx_b)
            seed = self.rng.integers(np.iinfo(np.int32).max)

            tasks = self.recombine(rng, idx_a, idx_b, replacement_act, net_a, net_b, seed)
            self.pending_tasks.append(await_returned_results.remote(tasks)) # type: ignore

    def recombine(self, rng, idx_a, idx_b, replacement_act, net_a, net_b, seed):
        if self.cx_style == "psa" or self.cx_style == "psb" or self.cx_style == "psr":
            # 'parent select' crossover - no constructing a CX network here.
            parent_net = None
            if self.cx_style == "psa": parent_net = net_a
            if self.cx_style == "psb": parent_net = net_b
            if self.cx_style == "psr": parent_net = self.rng.choice([net_a, net_b])

            # Always create a singular offspring for this mode - the others would simply be redundant copies
            net_o_obj = deepcopy(parent_net)

            task_idx = idx_a % len(self.problems)
            problem = self.problems[task_idx]
            # always use all in this configuration - there is no cx point here.
            allow_training_layer_kind_options = ["all"]
            # allow_training_layer_kind_options = self.config.get("allow_training_layer_kind_options", allow_training_layer_kind_options)
            # allow_training_layer_kind = rng.choice(allow_training_layer_kind_options)
            allow_training_layer_kind = allow_training_layer_kind_options[0]

            net_o_obj = train_network_and_act_when_trained.remote(problem, self.population, NoOpPopulationAction(), net_o_obj, seed=seed, params=self.cx_train_params, allow_training_layer_kind=allow_training_layer_kind, shrink_perturb=self.shrink_perturb, force_train=True) # type: ignore
            net_o_obj = evaluate_network_and_act.remote(problem, self.population, replacement_act, net_o_obj, force_evaluation=True) # type: ignore

            return [net_o_obj]

        tasks = []
        train_cx_only_first=self.config.get("train_cx_only_first", False)
        sample_train_num_epochs=self.config.get("sample_train_num_epochs", 0)
        lazy_mode=self.config.get("lazy_mode", False)
        do_retrain_offspring = sample_train_num_epochs > 0
        # evaluating
        # task_idx = seed % len(self.problems)
        task_idx = idx_a % len(self.problems)
        problem = self.problems[task_idx]
        offspring = train_cx_network_and_sample_act.options(num_returns=self.num_offspring).remote(problem, self.population, net_a, net_b, seed, self.num_offspring, is_offspring_trained=not do_retrain_offspring, cx_style=self.cx_style, lazy_mode=lazy_mode, parent_idxs=[idx_a, idx_b]) # type: ignore
        if self.num_offspring == 1:
            # wrap in a list so that the loop works...
            offspring = [offspring]

        number_preselected = self.config.get("number_preselected", None)
        # if preselection is none, don't perform any.
        if number_preselected is not None:
            # prior to training & evaluating networks, first ONLY evaluate them
            f_fitnesses = []
            f_networks = []
            for net_o_obj in offspring:
                f_fitness, f_network = evaluate_network_and_bireturn.remote(problem, net_o_obj) # type: ignore
                f_fitnesses.append(f_fitness)
                f_networks.append(f_network)
            offspring = truncate_preselect.options(num_returns=number_preselected).remote(number_preselected, f_fitnesses, f_networks ) # type: ignore
            if number_preselected == 1:
                offspring = [offspring]

        for net_o_obj in offspring:
            seed = self.rng.integers(np.iinfo(np.int32).max)

            if train_cx_only_first and sample_train_num_epochs > 0:
                allow_training_layer_kind = "cx"
                net_o_obj = train_network_and_act_when_trained.remote(problem, self.population, NoOpPopulationAction(), net_o_obj, seed=seed, params=self.cx_only_train_params, allow_training_layer_kind=allow_training_layer_kind, shrink_perturb=self.shrink_perturb, force_train=True) # type: ignore

            if sample_train_num_epochs > 0:
                allow_training_layer_kind_options = ["cx"]
                allow_training_layer_kind_options = self.config.get("allow_training_layer_kind_options", allow_training_layer_kind_options)
                allow_training_layer_kind = rng.choice(allow_training_layer_kind_options)
                net_o_obj = train_network_and_act_when_trained.remote(problem, self.population, NoOpPopulationAction(), net_o_obj, seed=seed, params=self.cx_train_params, allow_training_layer_kind=allow_training_layer_kind, shrink_perturb=self.shrink_perturb, force_train=True) # type: ignore

            net_o_obj = evaluate_network_and_act.remote(problem, self.population, replacement_act, net_o_obj, force_evaluation=True) # type: ignore
            tasks.append(net_o_obj)
        return tasks


    def generation(self):
        # for _ in range(self.num_crossover):
        self.recombine_train_evaluate()

        # Note - if asynchronous, generations will overlap!
        if self.sync:
            self.synchronize()

    def step(self):
        if self.initialized:
            self.generation()
        else:
            self.initialize()

