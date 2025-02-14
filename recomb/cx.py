#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

import hashlib
import math
import os
import time
from abc import ABC, abstractmethod
from collections import Counter
from copy import deepcopy
from enum import Enum
from functools import partial
from itertools import chain, count, cycle, islice
from typing import Any, Dict, List, Optional, Tuple

import igraph as ig
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from recomb.pqueue import KeyedPriorityQueue

from . import layers as ly
from .fmsimilarity import (compute_linear_cka_feature_map_similarity,
                           compute_mock_similarity,
                           compute_pca_feature_map_similarity)
from .pqueue import KeyedPriorityQueue


def detach_hooked_tensors(t):
    if torch.is_grad_enabled():
        if isinstance(t, torch.Tensor):
            return t.detach()
        elif isinstance(t, list) or isinstance(t, tuple):
            return [detach_hooked_tensors(e) for e in t]
        elif isinstance(t, dict):
            return {k: detach_hooked_tensors(v) for k, v in t.items()}
        else:
            # assume not a tensor, or something that contains tensors...
            return t
    else:
        return t 

def forward_get_all_feature_maps(net: ly.ModuleT, X, return_points=False, with_grad=True):
    # it is assumed that net is a graph...
    points = list(net.enumerate_points())

    hooks = []

    features = [None for _ in points]

    def hook(module, input, output, index):
        
        features[index] = detach_hooked_tensors(output)

    try:
        # Register hooks
        for idx, point in enumerate(points):
            hooks.append(
                net.get_module_by_point(point).register_forward_hook(
                    partial(hook, index=idx)
                )
            )
        # forward pass
        if with_grad:
            y_pred = net(X)
        else:
            with torch.no_grad():
                y_pred = net(X)
    finally:
        # Remove hooks
        for h in hooks:
            h.remove()

    if return_points:
        return features, points
    else:
        return features

class StitchingLib(ABC):
    """
    This class (or subclasses) characterizes feature maps and
    generates stitching layers where possible / defined.
    """

    def characterize_graph(self, g):
        pass

    def characterize_fm(self, v, fm, graph=None):
        pass

    @abstractmethod
    def can_stitch(self, a, b):
        pass

    def assess_stitch_quality(self, similarity_measure, a, b, fm_a, fm_b):
        # Original implementation only uses feature maps, impl. for compat.
        # Note that 'compute similarity matrix' actually computes a cost matrix
        return 1 - similarity_measure(fm_a, fm_b)

    @abstractmethod
    def create_stitch(self, a, b):
        pass

class CVStitchingLib(StitchingLib):

    def __init__(self, image_shape_should_match, feature_shape_should_match):
        self.image_shape_should_match = image_shape_should_match
        self.feature_shape_should_match = feature_shape_should_match

    def characterize_fm(self, v, fm, graph=None):
        if isinstance(fm, torch.Tensor):
            v["ty"] = "tensor"
            v["sh"] = list(fm.shape)
        else:
            v["ty"] = "unk"

    def can_stitch(self, a, b):
        # If types are unknown - do not allow stitching at these points.
        if a["ty"] == "unk": return False
        if b["ty"] == "unk": return False

        if a["ty"] == "tensor" and b["ty"] == "tensor":
            sh_a = a["sh"]
            sh_b = b["sh"]

            # if fm_a.shape[0] != fm_b.shape[0]:
            #     continue
            if self.image_shape_should_match and not (sh_a[2:] == sh_b[2:]):
                return False
            if self.feature_shape_should_match and not (sh_a[1] == sh_b[1]):
                return False
            return True
        
        # Cascade through
        return False

    def create_stitch(self, a, b):
        if a["ty"] == "tensor" and b["ty"] == "tensor":
            sh_a = a["sh"]
            sh_b = b["sh"]

            num_features_in = sh_a[1]
            num_features_out = sh_b[1]
            if len(sh_a) == 4 and len(sh_b) == 4 and sh_a[2:] == sh_b[2:]:
                return ly.Conv2d(
                    num_features_in, num_features_out, kernel_size=(1, 1)
                )
            elif len(sh_a) == 2 and len(sh_b) == 2:
                return ly.Linear(num_features_in, num_features_out)
            else:
                raise Exception(
                    f"cannot join items. No merging layer defined for shapes a: {sh_a} b: {sh_b}"
                )
        raise Exception(
                    f"cannot join items. No stitching layer defined between layers from type {a['ty']} to {b['ty']}"
                )

class BalancingCVStitchingLib(CVStitchingLib):
    def characterize_graph(self, g: ig.Graph):
        # compute distance from input
        d_in = np.array(g.distances(source=0)).ravel()
        # compute distance to output.
        d_out = np.array(g.distances(target=1)).ravel()
        # How far along the shortest path from input to output is this vertex?
        # Assumption input != output: i.e. resulting in d_in + d_out > 0.
        shortest_path_via_dst = d_in + d_out
        g.vs["dst"] = d_in / (shortest_path_via_dst)
        # Some alternatives using the maximum diameter
        # max_diameter = np.max(shortest_path_via_dst)
        # g.vs["dst_f"] = d_in / max_diameter
        # g.vs["dst_b"] = 1 - (d_out / max_diameter)

    def assess_stitch_quality(self, similarity_measure, a, b, fm_a, fm_b):
        # Dampen similarity if numbers are far apart
        # return similarity_measure(fm_a, fm_b) * (1 - (a["dst"] - b["dst"])**2)
        # Actually - similarity is lower = more similar
        return 0.5 + 0.5 * (1 - similarity_measure(fm_a, fm_b)) * (1 - (a["dst"] - b["dst"])**2)

default_stitching_lib = CVStitchingLib(True, False)

def compute_pairwise_similarities(
    g_a,
    fms_a,
    points_a,
    g_b,
    fms_b,
    points_b,
    compute_similarity,
    stitching_library: StitchingLib,
    fm_sh_match = False,
):
    simmap = np.full((len(g_a.vs), len(g_b.vs)), np.nan)

    num_maps_a = len(fms_a)
    num_maps_b = len(fms_b)
    for a in range(num_maps_a):
        if fms_a[a] is None:
            continue
        c_a = points_a[a][1]
        # update_progress(a * num_maps_b)
        # Assume preprocessed, prev: fms_to_flat_samples(fms_a[a], space=space).numpy()
        fm_a = fms_a[a]
        node_a = g_a.vs[c_a]

        for b in range(num_maps_b):
            if fms_b[b] is None:
                continue
            c_b = points_b[b][1]
            # assume preprocessed =  fms_to_flat_samples(fms_b[b], space=space).numpy()
            fm_b = fms_b[b]
            node_b = g_b.vs[c_b]
            if fm_sh_match and fm_a.shape[0] != fm_b.shape[0]:
                # Not entirely sure why I did this.
                continue

            if not stitching_library.can_stitch(node_a, node_b):
                continue
            # try:
            simmap[c_a, c_b] = stitching_library.assess_stitch_quality(compute_similarity, node_a, node_b, fm_a, fm_b)
            # except:
            #     pass
            # update_progress(a * num_maps_b + (b + 1))
    return simmap


def fms_to_flat_samples(f, space="feature"):
    if space == "feature":
        return f.movedim(1, -1).reshape(-1, f.shape[1]).detach()
    elif space == "sample":
        return f.reshape(f.shape[0], -1).detach()


def dag_max_distance_from_roots(g: ig.Graph):
    ord = g.topological_sorting(mode="out")
    g.vs["o"] = 0
    for i in ord:
        v = g.vs[i]
        for n in v.neighbors(mode="out"):
            n["o"] = max(n["o"], v["o"] + 1)


def find_ordered_matching(
    S: np.ndarray,
    graph_a: ig.Graph,
    graph_b: ig.Graph,
    verbose=False,
    max_alignments=np.inf,
):
    starting_time = time.time()
    if max_alignments >= min(len(graph_a.vs), len(graph_b.vs)):
        g_m = ig.Graph.Incidence(list(S), weighted=True)
        # use a different algorithm
        w, matching = cycle_cut(graph_a, graph_b, g_m, verbose=verbose)
        l_ga = len(graph_a.vs)

        completion_time = time.time()
        print(f"matching took {completion_time - starting_time}s using cc!")
        return w, {e.source: e.target - l_ga for e in matching.edges()}, np.inf
    else:
        r = find_ordered_matching_dp(S=S, graph_a=graph_a, graph_b=graph_b, verbose=verbose, max_alignments=max_alignments)
        completion_time = time.time()
        print(f"matching took {completion_time - starting_time}s using dp!")
        return r

def find_ordered_matching_dp(
    S: np.ndarray,
    graph_a: ig.Graph,
    graph_b: ig.Graph,
    verbose=False,
    max_alignments=np.inf,
):
    """
    Given a pairwise similarity matrix for two graphs, a & b, find a matching between the two networks
    va_i <-> v_b_j, such that a graph that merges these two nodes preserves the property of being a
    directed acyclic graph (DAG) while maximizing similarity.
    """
    # Note - is duplicated in fmsimilarity, too...
    # check whether we can remove one or the other.

    # Make copies, to be sure
    g_a = graph_a.copy()
    g_b = graph_b.copy()

    # Derive maximum distance from root - as to ensure ordering is valid.
    dag_max_distance_from_roots(g_a)
    dag_max_distance_from_roots(g_b)

    # Sidenote: It might be better to use a particular heuristic and apply A*...
    #
    def determine_priority(vs_a, vs_b, value):
        # Priority, we want to visit states which could potentially visit the other states first.
        os_a = g_a.vs[vs_a]["o"]
        os_b = g_b.vs[vs_b]["o"]
        os = os_a + os_b
        os.sort()
        return os

    # Start off with all the input nodes.
    inputs_a = [v.index for v in g_a.vs if v.indegree() == 0]
    inputs_b = [v.index for v in g_b.vs if v.indegree() == 0]
    initial = (frozenset(inputs_a), frozenset(inputs_b))
    v: Dict[Tuple[frozenset[int], frozenset[int]], Any] = {
        initial: (0.0, {}, max_alignments)
    }

    # Set up priority queue with initial point, and store the corresponding key, for completeness
    q = KeyedPriorityQueue([(determine_priority(inputs_a, inputs_b, 0), initial)])
    # m = {initial: 0}

    def update(state, value):
        if state not in v:
            k = q.add(determine_priority(state[0], state[1], value), state)
            # m[state] = k
            v[state] = value
            if verbose:
                print(f"new state: {state} -> {value}")
        else:
            current_value = v.get(state)
            if current_value[0] < value[0]:
                v[state] = value
                if verbose:
                    print(f"improved state: {state} -> {value}, was {current_value}")

    # Optional[Tuple[value, dict, alignments_left]]
    best_s = None

    while not q.is_empty():
        _, _, state = q.popmin()
        if verbose:
            print(f"visiting state {state}")
        # del m[v[1]]
        s = v[state]
        del v[state]
        if len(state[0]) == 0 or len(state[1]) == 0:
            if best_s is None or best_s[0] < s[0]:
                best_s = s

            # Skip over the remainder of this function - as one of the sets is empty
            # there are no further possible matches!
            continue

        v_a, v_b = state

        # We can either proceed (skip) for any over the elements in a, not increasing similarity or adding matchings.
        for a in v_a:
            vnew_a = frozenset(g_a.neighbors(a, mode="out")).union(v_a) - frozenset(
                g_a.subcomponent(a, mode="in")
            )
            for x in vnew_a:
                assert (
                    x < S.shape[0]
                ), f"tried adding node for graph index {x} but matrix is of size {S.shape[0]}"
            update((vnew_a, v_b), s)

        # Similarly, for b.
        for b in v_b:
            vnew_b = frozenset(g_b.neighbors(b, mode="out")).union(v_b) - frozenset(
                g_b.subcomponent(b, mode="in")
            )
            for x in vnew_b:
                assert (
                    x < S.shape[1]
                ), f"tried adding node for graph index {x} but matrix is of size {S.shape[1]}"
            update((v_a, vnew_b), s)

        # Alternatively, we can match any of the two in a and b.
        if s[2] > 0:
            for a in v_a:
                vnew_a = frozenset(g_a.neighbors(a, mode="out")).union(v_a) - frozenset(
                    g_a.subcomponent(a, mode="in")
                )
                for x in vnew_a:
                    assert (
                        x < S.shape[0]
                    ), f"tried adding node for graph index {x} but matrix is of size {S.shape[0]}"
                for b in v_b:
                    # Skip nan values
                    if np.isnan(S[a, b]):
                        continue
                    vnew_b = frozenset(g_b.neighbors(b, mode="out")).union(
                        v_b
                    ) - frozenset(g_b.subcomponent(b, mode="in"))
                    for x in vnew_b:
                        assert (
                            x < S.shape[1]
                        ), f"tried adding node for graph index {x} but matrix is of size {S.shape[1]}"
                    d = {a: b}
                    d.update(s[1])
                    update((vnew_a, vnew_b), (s[0] + S[a, b], d, s[2] - 1))

    return best_s

# def sort_tuple(t):
#     t = list(t)
#     t.sort()
#     return tuple(t)

def find_conflicting_matches(g, v_from, early_exit=True, verbose=False):
    s = [g.vs[v_from]]
    g.vs["visited"] = False
    g.vs["stack_idx"] = -1
    g.vs["matches"] = -1
    conflicts = []
    while len(s) > 0:
        if verbose: print(f"visiting {s[-1].index} with stack {[q.index for q in s]}")
        v = s[-1]
        next_idx = None
        
        v["visited"] = True
        v["stack_idx"] = len(s) - 1
        for e in v.out_edges():
            # skip over backwards edges - these cycles exist, but we wish to find a cycle excluding them.
            is_back_edge = len(s) > 1 and e.target == s[-2].index
            if is_back_edge:
                o = g.vs[e.target]
                if verbose: print(f"got back edge to {o.index}")
                # this edge forms a cycle, but also completes a matching pair.
                # set o's stack index to v's
                v["stack_idx"] = o["stack_idx"]
                # track that we have done this :)
                o["matches"] = v.index
                continue

            # are we on the stack? If we are - we found a cycle.
            fs = g.vs[e.target]["stack_idx"]
            if verbose: print(f"| got neighbor {e.target} stack_idx? {fs}")
            if fs > -1:
                vertices_cycle = s[fs:]
                seq_pairs = zip(vertices_cycle, islice(cycle(vertices_cycle), 1, None))
                matched_vertices_conflict = [(vert_a.index, vert_b.index) for vert_a, vert_b in seq_pairs if vert_a["type"] != vert_b["type"]]
                assert len(matched_vertices_conflict) > 0, f"got cycle {[v.index for v in vertices_cycle]}, but no matching edges appeared."
                # print(f"non-self cycle involving matchings: {matched_vertices_conflict}!")
                conflicts.append(matched_vertices_conflict)
                if early_exit: return conflicts

            # Have we visited this node at any point? If not: we have found our next node.
            if next_idx is None and not g.vs[e.target]["visited"]:
                next_idx = e.target

        while next_idx is None:
            # current node has no unvisited adjacent nodes, go up the stack until we do.
            rem = s.pop()
            if verbose: print(f"no further unvisited nodes for {rem.index} (stack idx: {rem['stack_idx']} / {len(s)}): going up...")
            if rem["stack_idx"] == len(s):
                if verbose: print("got correct stack index - removing stack index")
                # remove stack index - if it wasn't altered due to matching.
                rem["stack_idx"] = -1
                # if we have a matching index - reset that one too, now.
                if rem["matches"] != -1:
                    if verbose: print(f"{rem.index} was matched - removing stack index")
                    g.vs[rem["matches"]]["stack_idx"] = -1
            else:
                if verbose: print(f"mismatched stack index - postponed reset")
            if len(s) == 0: break
            v = s[-1]
            if verbose: print(f"checking adjacent nodes for {v.index}")
            for e in v.out_edges():
                if not g.vs[e.target]["visited"]:
                    next_idx = e.target
                    if verbose: print(f"found {next_idx}")
                    break
        
        if next_idx is not None:
            s.append(g.vs[next_idx])

    return conflicts


def cycle_cut(g_a:ig.Graph, g_b:ig.Graph, matching_graph:ig.Graph, verbose:bool=False, lim=np.inf, prune_queue=True):
    # copy, as assignments below may alias
    g_a = g_a.copy()
    g_b = g_b.copy()

    l_ga = len(g_a.vs)
    g_a.vs["type"] = False
    g_b.vs["type"] = True
    g_ab = g_a + g_b
    # prepare shared root.
    v_from = len(g_ab.vs)
    g_ab.add_vertices(1)
    g_ab.add_edges((v_from, t) for t in g_ab.vs if t.indegree() == 0 and t.index != v_from)
    matching = matching_graph.maximum_bipartite_matching(weights="weight")
    matching_graph.es["kv"] = range(len(matching_graph.es))
    sv = np.full(len(matching_graph.es), True)

    w = sum(e["weight"] for e in matching.edges())
    # (relaxation, actual, matching)
    cc_best = (-np.inf, None)
    if verbose: print(f"initial graph has {len(matching_graph.es)} edges and relaxation {w}")

    queue = KeyedPriorityQueue([(-w, (w, sv, matching_graph, matching))])

    def get_subcomponents(vx, vy):
        # note that these sets are static, and could theoretically be cached.
        # this is however not very expensive (especially given that the subcomponent call uses C via igraph)
        if vx < l_ga:
            assert vy >= l_ga
            # vx is in a, vy is in b
            return set(g_a.subcomponent(vx, mode='out')), set(b + l_ga for b in g_b.subcomponent(vy - l_ga, mode='in'))
        else:
            assert vy < l_ga
            # vx is in b, vy is in a
            return set(g_a.subcomponent(vy, mode='in')), set(b + l_ga for b in g_b.subcomponent(vx - l_ga, mode='out'))

    num_steps = 0
    while not queue.is_empty() and num_steps < lim:
        num_steps += 1
        _, _, (w, sv, matching_graph, matching) = queue.popmin()

        # queue pruning - remove problems that are strictly subproblems of the current task
        # since we will be solving this problem, these subsets are strictly covered too
        # therefore there is no need to solve them.
        if prune_queue:
            strict_subproblem_keys = []
            for (_w, ik, it) in queue.items:
                svo = it[1]
                # is there any element which would be a counterexample to svo being a subset?
                if (~sv & svo).any():
                    strict_subproblem_keys.append(ik)
            for ik in strict_subproblem_keys:
                queue.remove(ik)

        if w <= cc_best[0]:
            if verbose: print(f"relaxation of remaining branches at most as good ({w}) than best found ({cc_best[0]}). terminating...")
            break

        if verbose: print(f"visiting graph consisting of {len(matching_graph.es)} edges with relaxation {w}")

        # create graph incl. matching
        g_ab.add_edges((e.source, e.target) for e in matching.edges())
        g_ab.add_edges((e.target, e.source) for e in matching.edges())

        conflicting_matches = find_conflicting_matches(g_ab, v_from=v_from, early_exit=False)

        if len(conflicting_matches) == 0:
            # new valid matching!
            cc_best = max(cc_best, (w, matching), key=lambda v: v[0])
            if verbose: print(f"new matching with quality {w} has no conflicts. current best: {cc_best[0]}")
            continue
        

        if verbose: print(f"found conflicts: {conflicting_matches}")
        
        # return graph to original state
        g_ab.delete_edges((e.source, e.target) for e in matching.edges())
        g_ab.delete_edges((e.target, e.source) for e in matching.edges())

        best_branching = (-np.inf, None)
        
        # Edge preserve/cut rule
        # In case of preserve does not guarantee progress on constraints (a cycle through a parallel branch does not get cut)
        # But often reduces the number of edges and therefore possible matches that are in conflict significantly,
        # much more so than a cycle cut. As the edge case of a cycle through a parallel branch is comparitively rare,
        # this shoudn't be too problematic either.
        # Furthermore, as edges can only be removed once - and this choice is only picked if reduces problem complexity the most
        # if it does not actually cut the cycle - it is only going to be preserved once.
        # (and if preserve then cut, quality usually degrades to the point that branch & bound deals with it)
        edge_conflict_counter = Counter(chain(*conflicting_matches))
        if verbose: print(f"matching conflict counter: {edge_conflict_counter}")
        for (vx, vy), c in edge_conflict_counter.items():
            Ra_a, Rb_a = get_subcomponents(vx, vy)
            Ra_b, Rb_b = get_subcomponents(vy, vx)

            num_edges_removed = 0
            branching_set = []

            # Keep matching edge
            edges_to_remove = [
                e for e in matching_graph.es if
                # Edge that conflicts with (vx, vy)
                (e.source in Ra_a and e.target in Rb_a or
                e.source in Ra_b and e.target in Rb_b) and
                # Unless it is (vx, vy) itself.
                not (e.source in (vx, vy) and e.target in (vx, vy))]
            num_edges_removed += len(edges_to_remove)
            if verbose: print(f"| removes {len(edges_to_remove)} edges")
            if len(edges_to_remove) == 0: continue
            branching_set.append(edges_to_remove)

            # Remove matching edge (original rule)
            edges_to_remove = [
                e for e in matching_graph.es if
                (e.source in Ra_a and e.target in Rb_a)]
            num_edges_removed += len(edges_to_remove)
            if verbose: print(f"| removes {len(edges_to_remove)} edges")
            if len(edges_to_remove) == 0: continue
            branching_set.append(edges_to_remove)

            num_edges_total = len(matching_graph.es) * 2
            score = num_edges_removed / num_edges_total
            best_branching = max(best_branching, (score, branching_set), key=lambda v: v[0])
            if verbose: print(f"edge preserve/cut on {(vx, vy)} point scores {score} (best: {best_branching[0]})")

        # Cycle-cut rule
        # Guarantees progress, but generally worse than the previous rule.
        # Only use if the previous rule did not provide any beneficial steps.
        if best_branching[0] < 0.0:
            for conflicting_match in conflicting_matches:
                num_edges_removed = 0
                branching_set = []
                for (vx, vy) in conflicting_match:
                    Ra, Rb = get_subcomponents(vx, vy)
                    # print(f"{Ra} | {Rb}")
                    edges_to_remove = [e for e in matching_graph.es if e.source in Ra and e.target in Rb]
                    if verbose: print(f"| removes {len(edges_to_remove)} edges")
                    num_edges_removed += len(edges_to_remove)# ** 2
                    branching_set.append(edges_to_remove)
                num_edges_total = len(matching_graph.es) * len(conflicting_match)
                score = num_edges_removed / num_edges_total
                best_branching = max(best_branching, (score, branching_set), key=lambda v: v[0])
                if verbose: print(f"cycle cut on {conflicting_match} point scores {score} (best: {best_branching[0]})")

        for edges_to_remove in best_branching[1]:            
            matching_graph_c = matching_graph.copy()
            matching_graph_c.delete_edges([(e.source, e.target) for e in edges_to_remove])
            sv: np.array
            svc = np.copy(sv)
            for e in edges_to_remove:
                svc[e["kv"]] = False
            matching = matching_graph_c.maximum_bipartite_matching(weights="weight")
            w = sum(e["weight"] for e in matching.edges())
            worse_than_best = w <= cc_best[0]
            if verbose: print(f"got subproblem with {len(matching_graph_c.es)} edges and relaxation {w} {'(<= best - skipping)' if worse_than_best else ''}")
            if worse_than_best: continue
            queue.add(-w, (w, svc, matching_graph_c, matching))

    if verbose: print(f"terminated: {len(queue.items)} branches left")
    # Return best of two branches
    # cc_best = max(cc_keep, cc_omit, key=lambda v:v[0])
    # if verbose: print(f"branches have weights {cc_keep[0]} and {cc_omit[0]}, as such best: {cc_best[0]}")
    return cc_best

class CXN(ly.ModuleT):
    pass

class NNCombiner:
    pass

class FeatureMapStacker(ly.ModuleT):

    def stack(self, *X):
        fX = X[0]
        first_ty = type(fX)
        for i in X:
            assert isinstance(i, first_ty), "In order to be able to stack at all, types need to be equal"
        # Stacking cases.
        if isinstance(fX, torch.Tensor):
            # stack tensors in batch dimension.
            return torch.cat(X, dim=0)
        if isinstance(fX, dict):
            # Dictionaries are stacked by value
            return {
                k: self.stack(*[v[k] for v in X])
                for k in fX.keys()
            }
        raise ValueError(f"Unable to stack type {first_ty} - please define a rule in FeatureMapStacker")

    def forward(self, *X):
        return self.stack(*X)

class SwitchCXN(CXN):
    
    def __init__(self, agg=None):
        super().__init__()
        self.agg = agg
        self.compute_agg = False
        self.agg_v = None
        self.active = 0
        self.simplify = False
        self.randomize_per_sample = True
        # None = uniform
        self.p = None
    
    def determine_p(self):
        if isinstance(self.active, (str, int)):
            self.p = None
            return
        if self.p == None:
            return
        else:
            assert len(self.active) == len(self.p)
        self.p = np.cumsum(self.p)
        self.p /= self.p[-1]

    def sample_idx(self, choices):
        if self.p is None:
            return choices[torch.randint(0, len(choices), (1,)).item()]
        else:
            idx = np.searchsorted(self.p, torch.rand(1).item(), side='left')
            if self.active is None:
                return idx
            else:
                return choices[idx]

    def propagate_output(self):
        return True

    def forward(self, *X):
        if self.compute_agg:
            self.agg_v = self.agg(self, X)

        if np.issubdtype(type(self.active), np.integer):
            # Easy, 'default' case.
            return X[self.active]
        else:
            # Note - this behavior is not preserved if simplify = True (!)
            # and things error out instead when trying to convert in this case.
            possible_choices = self.active
            # note: special case - None = any at random.
            if self.active == None:
                possible_choices = range(len(X))
            
            if self.randomize_per_sample:
                # Construct a new feature map
                sh = X[0].shape
                return torch.stack([
                    X[self.sample_idx(possible_choices)][idx, ...]
                    for idx in range(sh[0])
                ])
            else:
                # Return the entire feature map of one of the two inputs.
                return X[self.sample_idx(possible_choices)]

    def to_subgraph(self, gc: "ly.GraphConstructor", feature_inputs):
        if not self.simplify:
            return super().to_subgraph(gc, feature_inputs)
        else:
            # do not introduce node, simplify.
            dfi = dict(feature_inputs)
            return dfi[self.active]

def agg_pairwise_mse(cxn, X):
    xs = len(X)
    l = 0.0
    lfn = nn.MSELoss()
    for i in range(xs):
        for j in range(i + 1, xs):
            l += lfn(X[i], X[j])
    return l

def filter_has_params(net, fms, points):
    fil = []
    for point in points:
        has_parameters = False
        if point[1] >= len(net.submodules):
            has_parameters = True
            m = None
        else:
            m = net.submodules[point[1]]
            for p in m.parameters():
                if p.requires_grad_:
                    has_parameters = True
                    break
        fil.append(has_parameters)
    fms = [fm for fm, k in zip(fms, fil) if k]
    points = [p for p, k in zip(points, fil) if k]
    return fms, points


def batcher(it, n):
    it = iter(it)
    while True:
        sl = list(islice(it, n))
        if sl is None or len(sl) == 0: return
        yield sl
    

class NNCombinerStitching:
    # Recombine neural networks by (potentially) performing a stitch between two networks.

    def __init__(
        self,
        graphnet_a: ly.NeuralNetGraph,
        fms_a: List[Any],
        graphnet_b: ly.NeuralNetGraph,
        fms_b: List[Any],
        matching_points: List[Tuple[Any, Any]],
        parent_idxs: List[int],
        ensemblers: List[ly.ModuleT],
        stitching_library: StitchingLib,
    ):
        self.graphnet_a = graphnet_a
        self.fms_a = fms_a
        self.graphnet_b = graphnet_b
        self.fms_b = fms_b
        self.matching_points = matching_points

        self.stitching_library = stitching_library
        self.ensemblers = ensemblers

        self.g_a = self.graphnet_a.graph.copy()
        self.g_b = self.graphnet_b.graph.copy()

        # By default - no nodes have been initialized and assigned an index yet.
        self.g_a.vs["ni"] = None
        self.g_b.vs["ni"] = None
        # Except for input & output, which are also to be shared
        self.g_a.vs[0]["ni"] = 0
        self.g_a.vs[1]["ni"] = 1
        self.g_b.vs[0]["ni"] = 0
        self.g_b.vs[1]["ni"] = 1
        
        # We are constructing a new graph
        self.gc = ly.GraphConstructor()

        self._create_joiners()
        self._construct_branches(self.graphnet_a, self.g_a)
        self._construct_branches(self.graphnet_b, self.g_b)

        try:
            self.cx_net = self.gc.to_neural_net_graph()
        except:
            print(f"exception occurred for recombining networks {parent_idxs}")
            # reraise
            raise

        # Construct cx_graph.
        self.cx_graph = get_cx_connectivity_graph(self.cx_net)

    def _create_joiners(self):
        self.joiners = []

        self.g_a.vs["j"] = None
        self.g_a.vs["l"] = None
        self.g_b.vs["j"] = None
        self.g_b.vs["l"] = None

        e_a = self.g_a.vs[1].neighbors(mode="in")[0]
        e_b = self.g_b.vs[1].neighbors(mode="in")[0]
        for point_a, point_b in self.matching_points:
            
            # todo - pass layers to transform a -> b and b -> a.
            idx_a = point_a[1]
            idx_b = point_b[1]
            # skip input node
            if idx_a == 0 or idx_b == 0:
                continue
            # skip output node
            if idx_a == 1 or idx_b == 1:
                continue
            # skip nodes connected to output node (special cased, below.)
            if idx_a == e_a.index or idx_b == e_b.index:
                continue

            node_a = self.g_a.vs[idx_a]
            node_b = self.g_b.vs[idx_b]

            # Create network switches.
            joiner_a = SwitchCXN(agg_pairwise_mse)
            joiner_b = SwitchCXN(agg_pairwise_mse)
            self.joiners.append((joiner_a, joiner_b))
            switch_a = self.gc.new_node(joiner_a)
            switch_b = self.gc.new_node(joiner_b)
            # Create trainable adaptors
            adaptor_a_to_b = self.stitching_library.create_stitch(node_a, node_b)
            adaptor_b_to_a = self.stitching_library.create_stitch(node_b, node_a)
            adaptor_a_to_b_n = self.gc.new_node(adaptor_a_to_b)
            adaptor_b_to_a_n = self.gc.new_node(adaptor_b_to_a)
            # connect to second socket.
            self.gc.new_edge(adaptor_a_to_b_n, switch_b, 1)
            self.gc.new_edge(adaptor_b_to_a_n, switch_a, 1)

            # Assign switches.
            self.g_a.vs[idx_a]["j"] = switch_a
            self.g_a.vs[idx_a]["l"] = [(switch_a, 0), (adaptor_a_to_b_n, 0)]
            self.g_b.vs[idx_b]["j"] = switch_b
            self.g_b.vs[idx_b]["l"] = [(switch_b, 0), (adaptor_b_to_a_n, 0)]

        # Output should already be aligned, with no necessary transformation.
        output_switch = SwitchCXN(agg=lambda Xs: torch.stack(Xs).mean(dim=0))
        self.output_switch = output_switch
        output_switch_n = self.gc.new_node(output_switch)
        # 
        out_0 = [(output_switch_n, 0)]
        out_1 = [(output_switch_n, 1)]
        for idx, ensembler in enumerate(self.ensemblers):
            # Add ensembler
            ensembler_n = self.gc.new_node(ensembler)
            self.gc.new_edge(ensembler_n, output_switch_n, idx + 2)
            out_0.append((ensembler_n, 0))
            out_1.append((ensembler_n, 1))

        # Request rewriting of output representation
        # j = what should references to this node be replaced with?
        # l = where do we connect this node to?
        self.g_a.vs[e_a.index]["j"] = output_switch_n
        self.g_a.vs[e_a.index]["l"] = out_0
        self.g_b.vs[e_b.index]["j"] = output_switch_n
        self.g_b.vs[e_b.index]["l"] = out_1

        # connect output switch to output
        self.gc.new_edge(output_switch_n, 1, 0)

    def _construct_branches(self, net: ly.NeuralNetGraph, g: ig.Graph):
        o = g.topological_sorting()
        # Copy over nodes
        for i in o:
            v = g.vs[i]
            if v["ni"] is None:
                mod_idx = v["module"]
                m = net.submodules[mod_idx]
                # copy the module
                #! hack: replace relu with activation layer.
                if str(m).startswith("ReLU"):
                    m = ly.ActivationFunction()
                else:
                    m = deepcopy(net.submodules[mod_idx])

                # put original nodes in evaluation mode & freeze their gradients.
                m.eval()
                for p in m.parameters():
                    p.requires_grad_(False)

                # if self.perturb is not None:
                #     m.perturb(self.perturb, m.point_start(), m.point_end())
                # if self.reinit:
                #     m.reinit(m.point_start(), m.point_end())
                # collect input indices
                inputs = [(e["socket"], g.vs[e.source]["ni"]) for e in v.in_edges()]
                for inp in inputs:
                    assert inp[1] >= 0
                v["ni"] = m.to_subgraph(self.gc, inputs)

            if v["l"] is not None:
                # this node should link its output to a cx point
                # rather than the original point.
                assert v["ni"] is not None and v["ni"] >= 0
                for (j, s) in v["l"]:
                    assert (
                        j > 0
                    ), f"edge must be positive, and also not into the input. {v}"
                    self.gc.new_edge(v["ni"], j, s)

            if v["j"] is not None:
                # further places use j as a replacement.
                v["ni"] = v["j"]
        
        # Use remapping (idx -> g[idx]["ni"]) to update "consumed_by" for each of the nodes.
        for i in o:
            v = g.vs[i]
            original_consumer = v["consumed_by"]
            if original_consumer is None: continue
            new_idx = v["ni"]
            new_consumer_idx = g.vs[original_consumer]["ni"]
            self.gc.set_consumed(new_idx, new_consumer_idx)

    def get_genotype_spec(self):
        return (
            [(0, 1)] + # which of the two output networks
            [(0, 1) for _ in range(2 * len(self.joiners))] # perform crossover a point?
        )

    def get_random_genotypes(self, rng, cx_style="1px", order_by_task=False):
        if cx_style == "ux": # uniform crossover
            gt_a = [rng.choice(spec) for spec in self.get_genotype_spec()] # type: ignore
            gt_b = [v for v in gt_a]
            gt_b[0] = 1 - gt_b[0]

            # If we request ordering by task, we ensure that the output of network A is part
            # of the first genotype, and the output of network B of the second genotype.
            # This is to ensure that the output of these offspring matches the tasks of network
            # A and B, in that order.
            if order_by_task:
                gt_a[0] = 0
                gt_b[0] = 1

            return [gt_a, gt_b]
        elif cx_style == "1px":
            genotype = [rng.choice((0, 1))] + [0 for _ in range(len(self.joiners) * 2)]
            i = rng.integers(0, len(self.joiners))
            genotype[i + 1] = 1
            genotype[i + 2] = 1

            gt_a = genotype
            gt_b = [v for v in genotype]
            gt_b[0] = 1 - gt_b[0]

            # See note on "order_by_task" under uniform crossover.
            if order_by_task:
                gt_a[0] = 0
                gt_b[0] = 1

            return [gt_a, gt_b]
        elif cx_style == "1cx":
            pg = self.cx_graph.permute_vertices(list(rng.permutation(len(self.cx_graph.vs))))
            cxg_o = pg.topological_sorting()
            cutoff = rng.integers(0, len(self.joiners))
            # 'color' the graph
            pg.vs["c"] = 1
            pg.vs[cxg_o[:cutoff]]["c"] = 0
            # if node -> output is different, this node
            for v in pg.vs:
                es = list(v.out_edges())
                if len(es) == 0: continue
                self.cx_net.submodules[v["module"]].v = int(any(v["c"] != pg.vs[ch.target]["c"] for ch in es))
            genotype = [rng.choice((0, 1))]
            p = [int(any(jx.v > 0 for jx in js)) for js in self.joiners for j in js]
            genotype += p
            
            gt_a = genotype
            gt_b = [v for v in genotype]
            gt_b[0] = 1 - gt_b[0]

            # See note on "order_by_task" under uniform crossover.
            if order_by_task:
                gt_a[0] = 0
                gt_b[0] = 1

            return [gt_a, gt_b]

    def get_offspring_from_genotype(self, genotype):
        # configure        
        for js, vs in zip(self.joiners, batcher(genotype[1:], 2)):
            for j, v in zip(js, vs):
                j.active = v
                j.simplify = True
        self.output_switch.active = genotype[0]
        self.output_switch.simplify = True

        # extract
        offspring_net = self.cx_net.to_graph()
        # simplify
        offspring_net.prune_unused()

        for js, v in zip(self.joiners, genotype[1:]):
            for j in js:
                j.active = 0
                j.simplify = False
        
        self.output_switch.active = 0
        self.output_switch.simplify = False

        return offspring_net

from torch.utils.tensorboard import SummaryWriter


def train_cx_net_stitch(dataset, cx_net, cx_net_m, classification_loss=False, extra_loss_iter=lambda: [], lr_scheduler=None, prof=None, iter_step_fn=None, batch_idx_start=0, dev=None, seed=42, num_epochs=1, num_samples=None, num_batches=None, lr=1e-3, weight_decay=1e-2, batch_size=128,
                        summarywriter: Optional[SummaryWriter]=None):
    
    cx_net_m = cx_net_m.to(dev)
    rng = torch.manual_seed(seed)
    num_dl_workers = int(os.environ.get("RECOMB_NUM_DATALOADER_WORKERS", "0"))
    if num_samples is not None and num_batches is None:
        num_batches = int(math.ceil(num_samples / batch_size))
    dl = DataLoader(dataset, batch_size=batch_size, generator=rng, shuffle=True, num_workers=num_dl_workers)
    if num_epochs is None and num_batches is not None:
        batches_per_epoch = len(dl) / batch_size
        # Note - integer division of num_batches / batches_per_epoch, rounded up.
        num_epochs = int(math.floor((num_batches + batches_per_epoch - 1) / batches_per_epoch))
    optim = torch.optim.AdamW(cx_net_m.parameters(), lr=lr, weight_decay=weight_decay)
    # end_loss_fn = nn.CrossEntropyLoss()
    batchwise_lr_schedule = True
    if lr_scheduler is not None:
        num_iter = num_epochs
        if batchwise_lr_schedule:
            if num_batches is None:
                num_batches = len(dl)
            num_iter *= min(len(dl), num_batches)
        lr_scheduler = lr_scheduler(optim, num_iter)

    batch_idx = batch_idx_start

    # print("training cx net...")
    for epoch in range(num_epochs):
        # print(f"start epoch {epoch}")
        epoch_loss = 0.0
        cur_num_batches = 0
        if num_batches is None:
            idxr = count()
        else:
            idxr = range(num_batches)
            
        for idx, (X, y) in zip(idxr, dl):
            X = X.to(dev)
            # cx_net_m.train_restore()

            # Note - we do not use the output, as at this stage this is not determined by trainable parameters.
            y_out = cx_net_m(X)
            # not using this...
            del y_out
            optim.zero_grad(set_to_none=True)

            y = y.to(dev)
            loss = torch.tensor(0.0, device=dev)
            for l in extra_loss_iter():
                loss += l()

            if summarywriter:
                summarywriter.add_scalar("Total Loss/Batch", loss.detach().item(), batch_idx)

            # if classification_loss:
            #     loss_classification = end_loss_fn(y_out, y)
            #     loss += loss_classification
            loss.backward()

            optim.step()
            # print(f"batch loss: {loss}")
            epoch_loss += loss.detach()

            if iter_step_fn is not None:
                iter_step_fn(batch_idx)

            # cx_net_m.store_state_eval()

            cur_num_batches += 1
            batch_idx += 1
            if prof is not None:
                prof.step()
            
            if lr_scheduler is not None and batchwise_lr_schedule:
                lr_scheduler.step()
        if summarywriter:
            summarywriter.add_scalar("Total Loss/Epoch", epoch_loss, batch_idx)
        if lr_scheduler is not None and not batchwise_lr_schedule:
                lr_scheduler.step()
    
    # print("trained cx net.")
    return batch_idx

def maybe_to_cpu(v):
    if isinstance(v, list) or isinstance(v, tuple):
        return [maybe_to_cpu(e) for e in v]
    elif isinstance(v, dict):
        return {k: maybe_to_cpu(e) for k, e in v.items()}
    try:
        return v.cpu()
    except:
        pass
    return v

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
        
        cxs_in = set()
        for e in vi.in_edges():
            cxs_in.update(g.vs[e.source]["cxs"])
        if isinstance(cx_net.submodules[vi["module"]], CXN):
            edges_to_add += [(s, i) for s in cxs_in]
            vi["cxs"] = set([i])
        elif not is_edge_case:
            vi["cxs"] = cxs_in
            idxs_to_remove.append(i)

    g.add_edges(edges_to_add)
    g.delete_vertices(idxs_to_remove)
    return g

def hash_array(a):
    hashv = hashlib.blake2b(a.tobytes(), digest_size=20)
    for dim in a.shape:
        hashv.update(dim.to_bytes(4, byteorder='big'))
    return hash(hashv.digest())

def hash_parameters(p):
    with torch.no_grad():
        return hash(tuple(hash_array(pr.numpy()) for pr in p))

def module_equals(m_a: 'ly.ModuleT', m_b: 'ly.ModuleT'):
    # Different types? No match
    if type(m_b) != type(m_a): return False

    # Match state dicts
    sda = m_a.state_dict()
    sdb = m_b.state_dict()
    for k in sda:
        if not k in sdb: return False
        if not (sda[k] == sdb[k]).all(): return False

    return True

def merge_identical_submodules(net, remap_modules=False):
    # move to cpu (so that I do not have to bother with devices...)
    net.cpu()

    param_hashids = {}
    # hash submodules
    for i, s in enumerate(net.submodules):
        k = hash_parameters(s.parameters())
        param_hashids[k] = param_hashids.get(k, []) + [i]

    # create submodule remapping
    # i.e. find parameter level matches.
    remap_mod = {}
    for b in param_hashids.values():
        while len(b) > 0:
            f = b.pop()
            nb = []
            while len(b) > 0:
                n = b.pop()
                # different types - mismatch
                if not module_equals(net.submodules[f], net.submodules[n]):
                    nb.append(n)
                    continue
                # match! remap n to f.
                remap_mod[n] = f
            b = nb

    # find for each module which nodes use it (under remapping)
    g = net.graph.copy()
    module_to_node_idx = {}
    for vidx, v in enumerate(g.vs):
        mod = v["module"]
        mod = remap_mod.get(mod, mod)
        module_to_node_idx[mod] = module_to_node_idx.get(mod, []) + [vidx]

    # how do we detect node level identities (i.e. identical computations?).
    # we assume that elements are not recursively remapped (i.e. a -> b, b -> c).
    # in order for two nodes to compute the same output feature map reliably they need to
    # 1. contain the same module (i.e. their modules = each other, or are equal under remap_mod)
    # 2. have the same input
    # Note that (2) indicates that this property recurses until the start of the graph.
    o = g.topological_sorting()
    # at first - assume every vertex is matched to only themselves.
    g.vs["mm"] = range(len(g.vs))
    g.vs["matched"] = False
    remap_nodes = {}

    for a in o:
        # skip over vertices we have already visited.
        if g.vs[a]["matched"]: continue
        # what is my module & remap if it is remapped
        a_mod = g.vs[a]["module"]
        a_mod = remap_mod.get(a_mod, a_mod)
        # which vertices are shared and thus matching candidates?
        similar_vertices = module_to_node_idx[a_mod]
        a_in = [g.vs[e.source]["mm"] for e in g.vs[a].in_edges()]
        a_in.sort()

        # visit *all* of these vertices and see if they match
        for b in similar_vertices:
            # note that, as o is a topological ordering, the element prior to it
            # and the elements similar to it have already been visited.
            # all we need to do is check is whether our inputs are matched too.
            b_in = [g.vs[e.source]["mm"] for e in g.vs[b].in_edges()]
            b_in.sort()

            if a_in == b_in:
                g.vs[b]["mm"] = a
                remap_nodes[b] = a
                g.vs[b]["matched"] = True
                        
    del g.vs["mm"]    
    del g.vs["matched"]

    # rewire graph nodes
    edges_to_remove = []
    edges_to_add = []
    attrs_to_add = {}
    for e in g.es:
        source = e.source
        target = e.target
        source_remapped = source in remap_nodes
        if source_remapped:
            edges_to_remove.append((source, target))
            source = remap_nodes.get(source, source)
            edges_to_add.append((source, target))
            for attr, v in e.attributes().items():
                x = attrs_to_add.get(attr, [])
                x.append(v)
                attrs_to_add[attr] = x
    
    g.delete_edges(edges_to_remove)
    g.add_edges(edges_to_add, attributes=attrs_to_add)
    
    if remap_modules:
        # (do not merge identical modules?)
        # print(remap_mod)
        g.vs["module"] = [remap_mod.get(v, v) for v in g.vs["module"]]

    net_simpl = ly.NeuralNetGraph(net.submodules, graph=g)
    net_simpl.prune_unused()
    
    return net_simpl

def prod(x):
    r = 1
    for v in x:
        r *= v
    return r

# Layer Merging
# - linear layer merging
def pack_linear_weights(linear_layer):
    # pack weights & biases into a matrix
    #  [W; b]
    #  [0; 1]
    W = linear_layer.weight
    b = linear_layer.bias
    if b is None:
        b = torch.zeros(W.shape[0])
    b = b.reshape(-1, 1)
    x = torch.concat([W, b], dim=1)
    y = torch.zeros((1, x.shape[1]))
    y[-1, -1] = 1
    return torch.concat([x, y], dim=0).T

def combine_linear_packed_weights(a, b):
    with torch.no_grad():
        return torch.matmul(a, b)

def packed_linear_weights_to_layer(packed_weights):
    bias = packed_weights[-1, :-1]
    weight = packed_weights[:-1, :-1].T

    layer = torch.nn.Linear(weight.shape[0], weight.shape[1])

    layer.weight = torch.nn.Parameter(weight)
    layer.bias = torch.nn.Parameter(bias)
        
    return layer

def combine_linear_layers(a, b):
    Pa = pack_linear_weights(a)
    Pb = pack_linear_weights(b)
    Pab = combine_linear_packed_weights(Pa, Pb)
    return packed_linear_weights_to_layer(Pab)

def combine_linear_layers_ly(a, b):
    linear_combined = combine_linear_layers(a.layer, b.layer)
    linear_ly = ly.Linear(linear_combined.in_features, linear_combined.out_features)
    return linear_ly

# - conv layer merging
# todo - make this work for my custom layers and with attributes.
def combine_conv1d(la, lb):
    assert la.groups == 1, "No support for grouped conv layers yet"
    assert la.dilation == 1, "No support for dilated conv layers yet"
    assert la.stride == 1, "No support for conv layers with different stride yet"
    assert lb.padding == 0, "No padding for b"
    assert lb.groups == 1, "No support for grouped conv layers yet"
    assert lb.dilation == 1, "No support for dilated conv layers yet"
    assert lb.stride == 1, "No support for conv layers with different stride yet"

    padding = [max(0, bsh - 1) for ash, bsh in zip(la.weight.shape[2:], lb.weight.shape[2:])]
    # note - flip, as torch does not implement convolution but cross correlation.
    W = torch.conv1d(la.weight.movedim(1, 0), torch.flip(lb.weight, dims=(2,3)), padding=padding).movedim(0, 1)
    b = None
    if la.bias is not None:
        # repeat pixels?
        # 
        b = torch.conv1d(la.bias.reshape(1, -1, 1) * torch.ones_like(lb.weight), lb.weight)[0, :, 0]
    if lb.bias is not None:
        if b is not None:
            b += lb.bias
        else:
            b = lb.bias

    # todo - use ly
    l = torch.nn.Conv1d(W.shape[1], W.shape[0], W.shape[2], padding=la.padding, padding_mode=la.padding_mode)
    l.weight = torch.nn.Parameter(W)
    if b is not None:
        l.bias = torch.nn.Parameter(b)
    else:
        l.bias = None
    return l

def combine_conv2d(la, lb):
    assert la.groups == 1, "No support for grouped conv layers yet"
    assert la.dilation == (1, 1), "No support for dilated conv layers yet"
    assert la.stride == (1, 1), "No support for conv layers with different stride yet"
    assert lb.padding == (0, 0), "No padding for b"
    assert lb.groups == 1, "No support for grouped conv layers yet"
    assert lb.dilation == (1, 1), "No support for dilated conv layers yet"
    assert lb.stride == (1, 1), "No support for conv layers with different stride yet"

    padding = [max(0, bsh - 1) for ash, bsh in zip(la.weight.shape[2:], lb.weight.shape[2:])]
    # note - flip, as torch does not implement convolution but cross correlation.
    W = torch.conv2d(la.weight.movedim(1, 0), torch.flip(lb.weight, dims=(2,3)), padding=padding).movedim(0, 1)
    b = None
    if la.bias is not None:
        # repeat pixels?
        # 
        b = torch.conv2d(la.bias.reshape(1, -1, 1, 1) * torch.ones_like(lb.weight), lb.weight)[0, :, 0, 0]
    if lb.bias is not None:
        if b is not None:
            b += lb.bias
        else:
            b = lb.bias
    
    l = ly.Conv2d(W.shape[1], W.shape[0], tuple(W.shape[2:]), padding=la.padding, padding_mode=la.padding_mode)
    l.layer.weight = torch.nn.Parameter(W)
    if b is not None:
        l.layer.bias = torch.nn.Parameter(b)
    else:
        l.layer.bias = None
    return l

def combine_conv3d(la, lb):
    assert la.groups == 1, "No support for grouped conv layers yet"
    assert la.dilation == (1, 1, 1), "No support for dilated conv layers yet"
    assert la.stride == (1, 1, 1), "No support for conv layers with different stride yet"
    assert lb.padding == (0, 0, 0), "No padding for b"
    assert lb.groups == 1, "No support for grouped conv layers yet"
    assert lb.dilation == (1, 1, 1), "No support for dilated conv layers yet"
    assert lb.stride == (1, 1, 1), "No support for conv layers with different stride yet"

    # extra padding to make this work.
    padding = [max(0, bsh - 1) for ash, bsh in zip(la.weight.shape[2:], lb.weight.shape[2:])]
    # note - flip, as torch does not implement convolution but cross correlation.
    W = torch.conv3d(la.weight.movedim(1, 0), torch.flip(lb.weight, dims=(2,3)), padding=padding).movedim(0, 1)
    b = None
    if la.bias is not None:
        # repeat pixels?
        # 
        b = torch.conv3d(la.bias.reshape(1, -1, 1, 1, 1) * torch.ones_like(lb.weight), lb.weight)[0, :, 0, 0, 0]
    if lb.bias is not None:
        if b is not None:
            b += lb.bias
        else:
            b = lb.bias

    # todo - use ly
    l = torch.nn.Conv3d(W.shape[1], W.shape[0], tuple(W.shape[2:]), padding=la.padding, padding_mode=la.padding_mode)
    l.weight = torch.nn.Parameter(W)
    if b is not None:
        l.bias = torch.nn.Parameter(b)
    else:
        l.bias = None
    return l

def merge_mergable_operations(net, stringify_types=False, prune=True, multipass=False, verbose=False):
    # locate operations that match a particular pattern & merge them if beneficial.
    # To determine if merging is beneficial we use the heuristic that number-of-parameters-after-merge < number-of-parameters-before-merge
    # in order for a merge to proceed.
    # - Can be generally computed (without operator specific information, like # flops)
    # - Fewer parameters is cheaper & usually applies restrictions (i.e. matrix must be of lower rank).

    # Currently this function is hardcoded for sequential merges of the same type with only a single input
    # (no sum-combinations allowed!)

    # move to cpu (so that I do not have to bother with devices...)
    net.cpu()

    def maybe_stringify(a):
        if stringify_types:
            return str(a)
        else:
            return a

    tymerge = {
        maybe_stringify(ly.Linear): combine_linear_layers,
        maybe_stringify(ly.Conv2d): combine_conv2d,
    }

    g = net.graph.copy()
    edges_to_remove = []
    edges_to_add = []
    edges_to_add_attrs = {}
    submodules = list(deepcopy(net.submodules))
    # immidiately update? required for one-pass to work with recursive merges.
    immidiately_update = False
    reiter = True
    while reiter:
        ord = g.topological_sorting()
        l = len(submodules)
        reiter = False
        for a_idx in ord:
            va = g.vs[a_idx]
            midx = va["module"]
            # special module - skip
            if midx < 0: continue

            m = submodules[midx]
            mty = maybe_stringify(type(m))
            if mty in tymerge:
                out_edges = list(va.out_edges())
                for eo in out_edges:
                    vb = g.vs[eo.target]
                    midxb = vb["module"]
                    # if special layer, skip here too.
                    if midxb < 0: continue

                    mb = submodules[midxb]
                    mbty = maybe_stringify(type(mb))
                    # if different kind of layer: skip.
                    if mbty != mty: continue

                    # we can compute a merged layer (note - sublayer)
                    try:
                        m_merged = tymerge[mty](m.layer, mb.layer)
                    except:
                        # if we fail here, the merge couldn't be performed in hindsight
                        # i.e. due to operator specific reasons.
                        # This for example happens in the current conv merge implementation
                        # as it does not deal well with padding and groups right now.
                        continue
                    # but do we actually merge?
                    
                    num_params_a = sum(prod(p.shape) for p in m.parameters())
                    num_params_b = sum(prod(p.shape) for p in mb.parameters())
                    # if `a` has multiple outputs - we would not be saving the parameters of `a`
                    # as they are still used for other operations.
                    # it may be of interest to expand this 
                    if len(out_edges) == 1:
                        num_params_separate = num_params_a + num_params_b
                    else:
                        num_params_separate = num_params_b
                    num_params_merged = sum(prod(p.shape) for p in m_merged.parameters())

                    # if #parameters ends up increasing, do not perform the replacement.
                    if num_params_merged > num_params_separate: continue

                    # Do not allow recursive merges if we are not immidiately updating
                    # note that this has the additional benefit of higher numerical accuracy when performing
                    # merges as we are performing merges in tree-like fashion.
                    if not immidiately_update and l <= va['module']: continue
                    if not immidiately_update and l <= vb['module']: continue

                    if verbose: print(f"merging {va.index} ({va['module']}) and {vb.index} ({vb['module']})")
                    # note - we replace layer b with the merged one to it to ensure that the number of vertices does not change
                    merged_midx = len(submodules)
                    submodules.append(m_merged)

                    vb["module"] = merged_midx

                    # unwire original input to the second layer
                    for e in vb.in_edges():
                        edges_to_remove.append((e.source, e.target))

                    # repeat again if multipass is true
                    reiter = multipass
                    
                    # finally, rewiring the input from layer a, so that the merged module has the expected input.
                    for e in va.in_edges():
                        # (note the replaced target node)
                        edges_to_add.append((e.source, eo.target))
                        for attr, v in e.attributes().items():
                            edges_to_add_attrs[attr] = edges_to_add_attrs.get(attr, []) + [v]

                    # Immidiately update.
                    if immidiately_update:
                        # Update graph - otherwise rewriting below will be incorrect
                        g.delete_edges(edges_to_remove)
                        g.add_edges(edges_to_add, attributes=edges_to_add_attrs)
                        # Reset...
                        edges_to_remove = []
                        edges_to_add = []
                        edges_to_add_attrs = {}

        # Update (deferred)
        g.delete_edges(edges_to_remove)
        g.add_edges(edges_to_add, attributes=edges_to_add_attrs)
        edges_to_remove = []
        edges_to_add = []
        edges_to_add_attrs = {}

    
    net_simpl = ly.NeuralNetGraph(submodules, graph=g)
    if prune:
        net_simpl.prune_unused()

    return net_simpl


def allow_training_all(g: ly.NeuralNetGraph):
    # Re-enable gradients for all layers
    for p in g.submodules.parameters():
        p.requires_grad_(True)

    # Set network fully back into training mode.
    g.train()

def has_gradients(m):
    r = False
    for p in m.parameters():
        r = r or p.requires_grad
        if r: break
    return r

def allow_training_kind(g: ly.NeuralNetGraph, kind="cx", clone_modules_if_required=False):
    # Don't clone for now...
    if clone_modules_if_required: raise NotImplemented()

    if kind == "cx":
        return

    if kind == "all":
        allow_training_all(g)
        return

    # Note - these kinds do not unfreeze parallel branches
    # negate_selection inverts such that parallel branches can be unfrozen
    # as a choice in the future (i.e., psuffix => kind = "in" & negate)
    if kind == "suffix":
        subcomponent_kind = "out"
        negate_selection = False
    elif kind == "prefix":
        subcomponent_kind = "in"
        negate_selection = False
    else:
        raise NotImplemented()

    # This method assumes that the only modules initially capable of training
    # are the crossover modules, and that the gradients have been disabled
    # for the pretraining procedure.
    cx_submodules = {i for i, m in enumerate(g.submodules) if has_gradients(m)}

    gg = g.graph.copy()
    gg.vs["reenable"] = [mi in cx_submodules for mi in gg.vs["module"]]
    
    # Find suffix or prefix nodes
    o = g.ord
    if subcomponent_kind == "in":
        # iterate in reverse order for prefix
        o = o[::-1]
    for idx in o:
        # skip over non-reenableting nodes
        if not gg.vs["reenable"]: continue
        # propagate reenable to suffix or prefix
        for n in gg.neighbors(idx, mode=subcomponent_kind):
            gg.vs[n]["reenable"] = True
    
    for v in gg.vs:
        if v["module"] < 0: continue

        vm = g.submodules[v["module"]]
        if v["reenable"]:
            # reenable gradients
            for p in vm.parameters():
                p.requires_grad_(True)
            # configure back into training-by-default
            # (frozen networks are in eval-by-default)
            vm.train()

def create_cx_network(
    net_a,
    points_a,
    fms_a,
    net_b,
    points_b,
    fms_b,
    stitching_library: StitchingLib = default_stitching_lib,
    compute_similarity=compute_linear_cka_feature_map_similarity,
    args_nnc=None,
    NNCombinerC: NNCombiner = None,
    parent_idxs=[],
    profile_cost=True,
):
    if args_nnc is None:
        args_nnc = {"reinit": True}

    # Old reshaping for Kernel Similarity Measurement
    # We don't use this anymore, but the old implementation assumes that this is present
    # if something breaks - we will probably need to move this to the similarity computation.
    # fms_ax = [
    #     fms_to_flat_samples(fm, space="sample").detach().numpy()
    #     if fm is not None
    #     else None
    #     for fm in fms_a
    # ]
    # fms_bx = [
    #     fms_to_flat_samples(fm, space="sample").detach().numpy()
    #     if fm is not None
    #     else None
    #     for fm in fms_b
    # ]
    points_a_v = [p for (p, fm) in zip(points_a, fms_a) if fm is not None]
    fms_a_v = [fm for fm in fms_a if fm is not None]
    points_b_v = [p for (p, fm) in zip(points_b, fms_b) if fm is not None]
    fms_b_v = [fm for fm in fms_b if fm is not None]

    # Characterize graphs
    stitching_library.characterize_graph(net_a.graph)
    stitching_library.characterize_graph(net_b.graph)

    # Store feature map shapes
    for fmidx, (p, fm) in enumerate(zip(points_a, fms_a)):
        if fm is None:
            continue
        # net_a.graph.vs[p[1]]["sh"] = list(fm.shape)
        stitching_library.characterize_fm(net_a.graph.vs[p[1]], fm, net_a.graph)
        net_a.graph.vs[p[1]]["fmidx"] = fmidx
    for fmidx, (p, fm) in enumerate(zip(points_b, fms_b)):
        if fm is None:
            continue
        # net_b.graph.vs[p[1]]["sh"] = list(fm.shape)
        stitching_library.characterize_fm(net_b.graph.vs[p[1]], fm, net_b.graph)
        net_b.graph.vs[p[1]]["fmidx"] = fmidx

    
    t_start = time.time()
    S = compute_pairwise_similarities(
        net_a.graph,
        fms_a_v,
        points_a_v,
        net_b.graph,
        fms_b_v,
        points_b_v,
        compute_similarity=compute_similarity,
        stitching_library=stitching_library,
    )
    # Input & output override
    S[0, 0] = 1
    S[1, 1] = 1

    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"[cxc] computing similarity matrix took {td}s")

    # plt.imshow(S)
    t_start = time.time()
    matching = [
        ((net_a, a), (net_b, b))
        for (a, b) in find_ordered_matching(S, net_a.graph, net_b.graph)[1].items()
    ]
    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"[cxc] computing matching took {td}s")

    try:
        r = NNCombinerC(net_a, fms_a, net_b, fms_b, matching, parent_idxs=parent_idxs, stitching_library=stitching_library, **args_nnc)
        return r
    except:
        
        S.tofile("S.dat")
        torch.save(net_a.graph, "g_a.dat")
        torch.save(net_b.graph, "g_b.dat")

        print(f"failure after matching similarity matrix for parents {parent_idxs}")
        print(f"graph a written to g_a.dat")
        print(f"graph b written to g_b.dat")
        print(f"similarity written to S.dat")
        raise

from functools import wraps


def do_immediately_backprop(func):
    @wraps(func)
    def nagg(*args, **kwargs):
        # Get the loss value for this element
        v = func(*args, **kwargs)
        # Immediately backpropagate - for the case of cx-stitch-training
        # this shouldn't be an issue.
        # Furthermore - the graphs should not overlap here :)
        v.backward()

        # Explicitly: we do not store the value here.
        return None
    return nagg

def construct_trained_cx_network_stitching(
    dataset,
    dev: torch.device,
    net_a: ly.NeuralNetGraph,
    net_b: ly.NeuralNetGraph,
    X_in_many,
    filter_has_parameters=False,
    image_shape_should_match=True,
    feature_shape_should_match=True,
    train_cx_network=True,
    pretrain_cx_network=True,
    train_all_cx_network=False,
    num_samples_pretrain=16384,
    num_epochs_pretrain=None,
    num_samples_train_stitch=0,
    lr_pretrain=1e-2,
    weight_decay_pretrain=0.0,
    batch_size=128,
    parent_idxs=[],
    compute_similarity=compute_linear_cka_feature_map_similarity,
    num_X_samples=None,
    profile_cost=True,
    use_gpu=True,
    stitching_library: StitchingLib = default_stitching_lib,
    ensemblers = [],
    summarywriter: Optional[SummaryWriter]= None,
    immediately_backprop=False,
):
    # note:
    # immediately_backprop is as efficient and saves memory, and works if in place ops are used
    # but will throw errors if we have multiple sequential stitches in use.

    # Allow for reducing # of samples to speed things up if allowed.
    if num_X_samples is not None:
        X_in_many = X_in_many[:num_X_samples].detach()
    
    t_start = time.time()

    net_a.store_state_eval()
    net_b.store_state_eval()

    if use_gpu:
        net_b.to(dev)
        net_a.to(dev)
        X_in_many = X_in_many.to(dev)
    # Set up reference networks.
    fms_a, points_a = forward_get_all_feature_maps(net_a, X_in_many, return_points=True)
    if filter_has_parameters:
        fms_a, points_a = filter_has_params(net_a, fms_a, points_a)

    fms_b, points_b = forward_get_all_feature_maps(net_b, X_in_many, return_points=True)
    if filter_has_parameters:
        fms_b, points_b = filter_has_params(net_b, fms_b, points_b)

    net_a.train_restore()
    net_b.train_restore()
    
    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"[cxc] getting feature maps took {td}s")

    # Seed torch's RNG - we may want to use this value for future reference.
    seed = torch.seed()

    if use_gpu:
        # Move back to cpu.
        net_a.cpu()
        fms_a = [maybe_to_cpu(v) for v in fms_a]
        net_b.cpu()
        fms_b = [maybe_to_cpu(v) for v in fms_b]
        X_in_many = X_in_many.cpu()
    
    t_start = time.time()
    # perform cx
    cx_net = create_cx_network(
        net_a,
        points_a,
        fms_a,
        net_b,
        points_b,
        fms_b,
        compute_similarity=compute_similarity,
        args_nnc={},
        # image_shape_should_match=image_shape_should_match,
        # feature_shape_should_match=feature_shape_should_match,
        stitching_library=stitching_library,
        NNCombinerC=partial(NNCombinerStitching, ensemblers=ensemblers),
        parent_idxs=parent_idxs,
    )
    cx_net_m = cx_net.cx_net
    
    t_end = time.time()
    td = t_end - t_start
    if profile_cost: print(f"[cxc] creating cx network took {td}s")

    # ensure conditions
    pretrain_cx_network = pretrain_cx_network and train_cx_network
    train_all_cx_network = train_all_cx_network and train_cx_network

    # pretrain?
    if pretrain_cx_network:
        t_start = time.time()
        # pretrain network such that activations are 'close'
        for js in cx_net.joiners:
            for j in js:
                j.compute_agg = True
                if immediately_backprop:
                    j.orig_agg = j.agg
                    j.agg = do_immediately_backprop(j.agg)

        batch_idx = 0
        def loss_iter_fn():
            nonlocal batch_idx
            batch_idx += 1
            # No separate losses necessary here.
            if immediately_backprop:
                return
            for join_idx, js in enumerate(cx_net.joiners):
                for j in js:
                    if j.agg_v is not None:
                        if summarywriter:
                            summarywriter.add_scalar(f"Loss/join-{join_idx}", j.agg_v.detach().item(), batch_idx)
                        yield lambda: j.agg_v
                        # save memory -> dealloc
                        j.agg_v = None

        # lr_scheduler = None
        lr_scheduler = lambda optim, num_steps: torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_steps)
        train_cx_net_stitch(dataset, cx_net, cx_net_m, dev=dev, lr=lr_pretrain, weight_decay=weight_decay_pretrain, batch_size=batch_size, extra_loss_iter=loss_iter_fn, classification_loss=False, lr_scheduler=lr_scheduler, num_epochs=num_epochs_pretrain, num_samples=num_samples_pretrain,
                            summarywriter=summarywriter)

        t_end = time.time()
        td = t_end - t_start
        if profile_cost: print(f"[cxc] pretraining network took {td}s")

        for js in cx_net.joiners:
            for j in js:
                j.compute_agg = False
                if immediately_backprop:
                    j.agg = j.orig_agg

    # train all CX layers?
    # note - disabled - is task specific.
    # if train_all_cx_network:
    #     # train all crossovers - note that this can be postponed (!) only training the layers that are actually used.
    #     os = cx_net.output_switch
    #     for i in range(len(cx_net.joiners)):
    #         for js in cx_net.joiners:
    #             for j in js:
    #                 j.active = 0

    #         ja, jb = cx_net.joiners[i]
    #         ja.active = 1
    #         jb.active = 1
    #         os.active = 0

    #         lr_scheduler = lambda optim, num_steps: torch.optim.lr_scheduler.CosineAnnealingLR(optim, num_steps)
    #         # lr_scheduler = None
    #         train_cx_net_stitch(dataset, cx_net, cx_net_m, dev=dev, lr=5e-3, lr_scheduler=lr_scheduler, num_batches=num_batches_train_stitch)
    #         os.active = 1
    #         train_cx_net_stitch(dataset, cx_net, cx_net_m, dev=dev, lr=5e-3, lr_scheduler=lr_scheduler, num_batches=num_batches_train_stitch)

    cx_net_m.cpu()
    return cx_net_m, cx_net


def construct_trained_cx_network(
    *x, cx_network_method="stitch", **y
):
    return construct_trained_cx_network_stitching(*x, **y)

# Utility methods

class SitchingInfo:
    """
    A more compact representation of the commonly used information
    from a stitching network.

    Mostly useful to keep pickle files containing this info small
    as the full object contains a lot more information about the
    intermediate steps of the stitching process.
    """
    def __init__(self, joiners, output_switch):
        self.joiners = joiners
        self.output_switch = output_switch

def get_stitcher_genotype_cardinality(stitchnet, stitcher):

    for v in stitchnet.graph.vs:
        if v["module"] < 0: continue
        md = stitchnet.submodules[v["module"]]
        if not isinstance(md, CXN): continue
        sockets = np.unique([e["socket"] for e in v.in_edges()])
        md.input_sockets = sockets
        md.num_input_sockets = len(sockets)

    o = []
    for js in stitcher.joiners:
        for j in js:
            o.append(j.num_input_sockets)
    o.append(stitcher.output_switch.num_input_sockets)
    return o

def convert_stitcher_to_genotype(stitcher, stringify=True):
    o = []
    for js in stitcher.joiners:
        for j in js:
            o.append(j.active)
    o.append(stitcher.output_switch.active)
    if stringify:
        return str(o)
    else:
        return o

def apply_genotype_to_stitcher(stitcher, genotype):
    import json
    if isinstance(genotype, str):
        gt = json.loads(genotype)
    else:
        gt = genotype
    gt = iter(gt)
    for js in stitcher.joiners:
        for j in js:
            j.active = next(gt)
    stitcher.output_switch.active = next(gt)

def get_extended_cx_connectivity_graph(cx_net: ly.NeuralNetGraph):
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
        
        is_cxn = isinstance(cx_net.submodules[vi["module"]], CXN)
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

def compute_parallel_set(cxg, i):
    s = set(range(len(cxg.vs))) 
    s -= set(cxg.subcomponent(i, mode='out'))
    s -= set(cxg.subcomponent(i, mode='in'))
    # s.add(i)
    return s

def compute_all_parallel_set(cxg):
    return [compute_parallel_set(cxg, i) for i in range(len(cxg.vs))]

from copy import copy
from functools import partial


def enumerate_parallel_set(g, set_idx, parallel_sets, i, verbose=False):
    """
    Iterate over parallel movements to stitch from one network to another.
    Accounting for parallel elements.
    """
    def call_funcs(lfn):
        for fn in lfn:
            fn()

    for (set_list, restore_list) in enumerate_parallel_set_recur(g, set_idx, parallel_sets, i, None, set(), verbose=verbose):
        yield (lambda: call_funcs(set_list)), (lambda: call_funcs(restore_list))

def enumerate_parallel_set_recur(g, set_idx, parallel_sets, i, current_set=None, unpickable: set=set(), ref_og=None, verbose=False):
    if current_set is None:
        # Initial case - current set is the parallel set of the index we are starting with.
        current_set = copy(parallel_sets[i])
        ref_og = g.vs[i]["og"]
        # filter current set based on a match
        current_set = {a for a in current_set if g.vs[a]["og"] == ref_og}
    else:
        # Otherwise, update the set of uncovered elements.
        current_set = current_set.intersection(parallel_sets[i])

    if len(current_set) == 0:
        if verbose: print(f"base case - no other choices necessary after picking {i}")
        # yield setter for configuring and unconfiguring i. No other configurations necessary
        # as there are no other branches.
        yield [partial(set_idx, i, 1)], [partial(set_idx, i, 0)]
        # return - as there are no more elements in the neighborhood.
        return

    # Obtain a fixed ordering of the set of leftover elements to be picked.
    ordering = list(current_set - unpickable)
    
    # Find current reverse cumulative intersection.
    # The intersection of sets picked so far provides knowledge of elements that may need
    # to be picked to cover all branches.
    # If we perform this operation cumulatively from the right the elements left in the
    # set allow us to identify necessary picks.
    # if we have the set with fixed ordering [ 1, 2, 3]
    # and the set corresponding here are 1 -> {0, 2, 3}, 2 -> {0, 1, 3}, 3 -> {0, 1, 2}
    # (note: the index itself is never contained within its own parallel set)
    # In this case the sequence of sets would be
    # [{}, {1}, {1, 2}]
    # as the only set that does not contain {1} is the set corresponding to {1}, 1 must be picked.
    cumulative_sets_rl = [None for _ in range(len(ordering))]
    cumulative_sets_rl[-1] = current_set.intersection(parallel_sets[ordering[-1]])
    required_right = {ordering[-1]}
    for i in range(len(ordering) - 1, 0, -1):
        el = ordering[i - 1]
        cumulative_sets_rl[i - 1] = cumulative_sets_rl[i].intersection(parallel_sets[el])
        # note - if an ordering[i - 1] is in cumulative_sets_rl[i], el needs to be picked if we do not
        # pick any of the preceding elements as there are no further elements to cover this branch.
        if ordering[i - 1] in cumulative_sets_rl[i]:
            required_right.add(el)
    # If we do it the other direction we can do the same thing for any following elements.
    cumulative_sets_lr = [None for _ in range(len(ordering))]
    cumulative_sets_lr[0] = current_set.intersection(parallel_sets[ordering[0]])
    required_left = {ordering[0]}
    for i in range(0, len(ordering) - 1):
        el = ordering[i + 1]
        cumulative_sets_lr[i + 1] = cumulative_sets_lr[i].intersection(parallel_sets[el])
        # similar reasoning - if we pick none of the elements after this one, there would be
        # a uncovered branch
        if ordering[i + 1] in cumulative_sets_lr[i]:
            required_left.add(el)
    # Elements that are in both required sets are always to be taken.
    always_required = required_left.intersection(required_right)

    # For future additions: - if one skips elements that have already been investigated previously (i.e., 
    # elsewhere in the ordering, another indicator is important to keep track of:
    # cumulative_sets_lr[-1] and cumulative_sets_rl[0] should always be empty sets - if they are not
    # there exists an element that is not optional that was excluded.
    # Probably shouldn't happen since we force always required, but just in case, handle this edge case.
    if len(cumulative_sets_lr[-1]) != 0 or len(cumulative_sets_rl[0]) != 0:
        if verbose: print("forbidden case - no choices cover all branches anymore...")
        return

    fixed_set = [partial(set_idx, i, 1)]
    fixed_restore = [partial(set_idx, i, 0)]
    
    if verbose: print(f"in this case to cover all branches {always_required} are required")
    for a in always_required:
        current_set.intersection_update(parallel_sets[a])
        fixed_set.append(partial(set_idx, a, 1))
        fixed_restore.append(partial(set_idx, a, 0))

    if len(current_set) == 0:
        if verbose: print(f"fixed case - no more free choices left to make after picking {i}")
        yield fixed_set, fixed_restore
    else:
        if verbose: print(f"recursive case for {i}")
        for e in current_set:
            # consider the cases where we pick it
            
            if verbose: print(f"considering picking {e}")
            for (set_list, restore_list) in enumerate_parallel_set_recur(g, set_idx, parallel_sets, e, current_set, unpickable=unpickable, ref_og=ref_og):
                yield (fixed_set + set_list), (fixed_restore + restore_list)
            if verbose: print(f"no longer considering picking {e}")
            # now - for the following picks consider the case where we not allow e to be picked anymore.
            unpickable.add(e)
        # to avoid issues with branching allow picking these elements again if another branch investigates them.
        for e in current_set:
            unpickable.remove(e)

def interactive_plot_extended_cx_graph(cxg):
    import plotly.graph_objects as go
    import numpy as np

    graph_layout, cxge = cxg.layout_sugiyama(return_extended_graph=True)
    # Transform graph layout to coords
    layout_coords = np.array(graph_layout)
    vertex_coords = np.array([layout_coords[v.index] for v in cxg.vs])
    # edge_coords = [[layout_coords[e.source, :], layout_coords[e.target, :], [np.nan, np.nan]] for e in cxg.es]
    edge_coords = [[
        layout_coords[e.source, :],
        layout_coords[e.target, :],
        [np.nan, np.nan],
        ]
        for e in cxge.es]
    edge_coords = np.array(edge_coords).reshape(-1, 2)
    is_arrow_end = np.array([[0, 0, 0] if edge.target >= num_orig_nodes else [0, 10, 0] for edge in cxge.es]).ravel()

    num_orig_nodes = len(cxg.vs)
    # edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=lineWidth, color=lineColor), hoverinfo='none', mode='lines')
    edge_trace = go.Scatter(
        # note: coords are transposed.
        x=edge_coords[:, 1],
        y=edge_coords[:, 0],
        hoverinfo='none',
        mode='lines+markers',
        marker=dict(
            size=10,
            angleref="previous",
            symbol="arrow",
            ),
        marker_size = is_arrow_end,
    )

    def color_table(ogv):
        if ogv == 2: return 'red'
        if ogv == 3: return 'blue'
        return 'white'

    # node_trace = go.Scatter(x=node_x, y=node_y, mode='markers', hoverinfo='text', marker=dict(showscale=False, color = nodeColor, size=nodeSize))
    node_trace = go.Scatter(
        # note: coords are transposed.
        x=vertex_coords[:, 1],
        y=vertex_coords[:, 0],
        marker_color = [color_table(og) for og in cxg.vs["og"]],
        mode='markers',
        # hoverinfo='text',
        marker=dict(showscale=False)
    )

    fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    showlegend=False,
                    # hovermode='closest',
                    margin=dict(b=20,l=5,r=5,t=40),
                    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                    )
    return fig

# Rather than using pointers & the fact that pickles preserve pointers, store some indexing data.
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
        if not isinstance(md, CXN): continue
        r[md.variable_pos] = module_idx
    
    return r, r[-1]

# Helpers for working with the ordering data
def set_genotype_module_ordering(stitchnet, mo, genotype):
    for i, md in enumerate(mo):
        stitchnet.submodules[md].active = genotype[i]

def get_genotype_module_ordering(stitchnet, mo):
    return [
        stitchnet.submodules[md].active
        for md in mo
    ]

# Active variable detection

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
        if isinstance(m, CXN):
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