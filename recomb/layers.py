from collections import OrderedDict
from collections.abc import Iterator
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union
import torch
import torch.nn as nn
from dataclasses import dataclass, field
import numpy as np
import igraph as ig
import textwrap
from copy import deepcopy
from abc import ABC, abstractmethod
import traceback

# Wrappers around torch.nn layers & other utilities for ease of manipulation (i.e. changing number of channels)
# navigating around a network.

def perturb(original_named_parameters, reinit_named_parameters, w):
    original_named_parameters = dict(original_named_parameters)
    reinit_named_parameters = dict(reinit_named_parameters)
    with torch.no_grad():
        for name in original_named_parameters:
            original_named_parameters[name] *= (1 - w)
            original_named_parameters[name] += reinit_named_parameters[name] * w

# ABC
class ModuleT(nn.Module):
    def alters_shape(self):
        """
        Can this layer alter the shape of a feature map?
        """
        return False

    def controls_num_features(self):
        """
        Can this layer alter the number of features?
        """
        return False

    def requires_num_features(self):
        """
        Does this layer require the number of features to be known?

        Distinct from `alters_num_features` as some layers do not alter, but are parameterized by the number of features.
        """
        return False
    
    def propagate_output(self):
        # Do we propagate setting the output type to inputs?
        # Generally, no.
        return False

    # @abstractmethod
    def get_reconstructor(self):
        raise NotImplementedError

    def reinit_self(self):
        # For most layers there are no such parameters
        pass

    def reinit(self, start_point, end_point):
        """
        Reinitialize trainable parameters of network
        """
        if end_point == "self": end_point = self
        if start_point is not None or end_point is not self: return
        
        self.reinit_self()

    def perturb_self(self, w):
        pass

    def perturb(self, w, start_point=None, end_point="self"):
        """
        Perturb the weights of the network.

        Resulting weights are W * (1 - w) + I * w, where W are the current weights
        and I are weights from the modules being reinitialized.
        """
        if end_point == "self": end_point = self
        if start_point is not None or end_point is not self: return
        self.perturb_self(w)

    def __init__(self):
        super().__init__()

    # @abstractmethod
    def shape_in(self):
        """
        What is the shape [b, ..., f] at input - excluding batch?
        """
        raise NotImplementedError

    # @abstractmethod
    def shape_out(self):
        """
        What is the shape [b, ..., f] at output - excluding batch?
        """
        raise NotImplementedError

    def set_standard_activation_function(self, fn):
        pass

    def enumerate_points(self):
        """
        Enumerate output vector locations.
        """
        # Default points generator. This point represents state after applying this layer.
        yield self

    def get_module_by_point(self, point):
        # Simple enough if there is no special case.
        assert point is self or point is None
        return point

    def point_start(self):
        """Give the point for the start of this layer."""
        return None

    def point_end(self):
        """Give the point for the end of this layer."""
        return self

    def find_feature_change_start(self, point):
        """
        Locate the nearest layers for which `controls_num_features` returns true, which provide input to `point`.
        Returns a point if such a layer exists within the subnetwork spanned by this node. None if this is dependent on the input.
        """
        if point is None:
            # Point is input - cannot change.
            return None
        if self.controls_num_features():
            return self
        return None

    def find_feature_change_end(self, point):
        """
        Locate the next layers for which `controls_num_features` returns true, which provide input to `point`.
        Returns a point if such a layer exists within the subnetwork spanned by this node. None if this is dependent on the output.

        """
        if point is self:
            # This position is past this node - it depends on the output instead.
            return None
        if self.controls_num_features():
            return self
        return None

    def to_graph(self):
        gc = GraphConstructor()
        out = self.to_subgraph(gc, [(0, 0)])
        gc.new_edge(out, 1, 0)
        return gc.to_neural_net_graph()

    def to_subgraph(self, gc: 'GraphConstructor', feature_inputs: list[tuple]):
        self_idx = gc.new_node(self)
        for socket, feature_input in feature_inputs:
            gc.new_edge(feature_input, self_idx, socket)
        return self_idx

    def in_shapes_to_out_shape(self, shapes_in):
        # Lazy implementation that simply uses the forward method
        was_training = self.training
        self.eval()
        shape_out = self.forward(torch.zeros(size=shapes_in)).shape
        self.train(size=was_training)
        return shape_out

    def adjust_in_shapes_for_out_shape(self, shapes_in, shapes_in_fixed, shape_out):
        assert not self.alters_shape()
        for sh_in, is_fixed in zip(shapes_in, shapes_in_fixed):
            if is_fixed:
                assert sh_in == shape_out, "conflict - cannot adjust shapes to match fixed value"
        return [shape_out for _ in shapes_in]
    
    def clone_for_shape(self, original_shapes_in, new_shapes_in, new_shape_out=None):
        # Simple default impl - use deepclone.
        # Assumes that the shapes are fine as-is or don't matter for this layer.
        return deepcopy(self)

@dataclass
class GraphConstructor:
    # A list of modules
    modules: List[ModuleT] = field(default_factory=lambda: [])
    # an item for each node, refers to the index of the module
    # negative indices are used for special cases
    nodes: List[int] = field(default_factory=lambda: [-2, -1])
    # edges between nodes
    edges: List[Tuple[int, int]] = field(default_factory=lambda: [])
    # sockets - for each edge, to what socket does it link (if a node has multiple inputs)
    sockets: List[int] = field(default_factory=lambda: [])
    # consumed - register an operator to be output consuming (e.g. in-place)
    consumptions: List[int] = field(default_factory=lambda: [None, None])

    def new_node(self, module: ModuleT) -> int:
        try:
            mi = self.modules.index(module)
        except:
            mi = len(self.modules)
            self.modules.append(module)

        i = len(self.nodes)
        self.nodes.append(mi)
        self.consumptions.append(None)
        return i
    
    def set_consumed(self, node_idx, node_idx_by):
        self.consumptions[node_idx] = node_idx_by

    def new_edge(self, from_idx: int, to_idx: int, socket: int):
        assert from_idx >= 0
        assert to_idx >= 0
        self.edges.append((from_idx, to_idx))
        self.sockets.append(socket)

    def to_neural_net_graph(self) -> 'NeuralNetGraph':
        return NeuralNetGraph(self.modules, self.nodes, self.edges, self.sockets, self.consumptions)

class Sequential(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def __init__(self, *modules):
        super().__init__()
        self.layer = nn.Sequential(*modules)

    def get_reconstructor(self):
        reconstructors = [m.get_reconstructor() for m in self.layer]
        return lambda: Sequential([r() for r in reconstructors])

    def forward(self, x):
        return self.layer(x)

    def enumerate_points(self, include_input=False):
        if include_input:
            yield None
        for idx, l in enumerate(self.layer):
            for p in l.enumerate_points():
                yield (self, idx, p)

    def point_start(self):
        """Give the point for the start of this layer."""
        return (self, -1, None)

    def point_end(self):
        """Give the point for the end of this layer."""
        num_layers = len(self.layer)
        return (self, num_layers - 1, self.layer[num_layers - 1].point_end())

    def validate_point(self, point):
        if point is None:
            return  # None represents the input to this layer
        assert len(point) == 3, "Point should be 3-element tuple for Sequential"
        assert point[0] == self, "First element to point should be reference to self."
        assert point[1] < len(self.layer), "idx should be in range."

    def to_subgraph(self, gc: GraphConstructor, features_in):
        curr = features_in[0]
        for l_idx in range(len(self.layer)):
            new_idx = self.layer[l_idx].to_subgraph(gc, [curr])
            curr = (0, new_idx)
        return curr

    def find_feature_change_start(self, point):
        self.validate_point(point)
        # First - request from the subnode.
        fcs = self.layer[point[1]].find_feature_change_start(point[2])
        if fcs is not None:
            # If it is not none - subnode has provided a point
            return (self, point[1], fcs)
        # If it is none, it is dependent on the input.
        # -> check preceding nodes
        for idx in range(point[1] - 1, 0 - 1, -1):
            l = self.layer[idx]
            # Output of this node is the input of the previous node
            # As such, search for a changing node starting at the output of this node.
            fcs = l.find_feature_change_start(l.point_end())
            if fcs is not None:
                # If it is not none - subnode has provided a point
                #
                return (self, idx, fcs)
            # Otherwise we continue.
        # If we have gone all of the nodes - the starting point precedes this layer.
        return None

    def find_feature_change_end(self, point):
        self.validate_point(point)
        # First - request from the subnode.
        fcs = self.layer[point[1]].find_feature_change_end(point[2])
        if fcs is not None:
            # If it is not none - subnode has provided a point
            return (self, point[1], fcs)
        # If it is none, the number of features must be unalterable until later.
        #
        # -> check following nodes
        for idx in range(point[1] + 1, len(self.layer)):
            l = self.layer[idx]
            # Input of this node is the output of the previous node
            # As such, search for a changing node starting at the output of this node.
            fcs = l.find_feature_change_end(l.point_start())
            if fcs is not None:
                # If it is not none - subnode has provided a point
                return (self, idx, fcs)
            # Otherwise we continue.
        # If we have gone all of the nodes - the starting point precedes this layer.
        return None
    
    def get_module_by_point(self, point):
        self.validate_point(point)
        # Special case...
        if point is None: return None
        
        point_submodule = self.layer[point[1]].get_module_by_point(point[2])
        if point_submodule is None and point[1] == 0: return None
        if point_submodule is None:
            prev_module = self.layer[point[1] - 1]
            return prev_module.get_module_by_point(prev_module.point_end())
        else:
            return point_submodule


    def reinit(self, point_start = None, point_end = "self"):
        """
        Reinitialize trainable parameters of network
        """
        if point_end == "self": point_end = self.point_end()
        self.validate_point(point_start)
        self.validate_point(point_end)

        idx_start = 0
        subpoint_start = None
        idx_end = 0
        subpoint_end = None

        if point_start is not None:
            idx_start = point_start[1]
            subpoint_start = point_start[2]

        if point_end is not None:
            idx_end = point_end[1] + 1
            subpoint_end = point_end[2]

        assert idx_end >= idx_start
        # Edge case - no nodes to reinit
        if idx_end - idx_start == 0: return
        # Edge case, reinit is entirely contained within subnode
        if idx_end - idx_start == 1:
            l_0 = self.layer[idx_start]
            l_0.reinit(subpoint_start, subpoint_end)
            return

        # Reinit starting layer potentially partially.
        l_0 = self.layer[idx_start]
        l_0.reinit(subpoint_start, l_0.point_end())
        # Fully reinit intermediate.
        for l in self.layer[idx_start + 1 : idx_end - 1]:
            l.reinit(l.point_start(), l.point_end())
        # Reinit ending layer potentially partially.
        l_e = self.layer[idx_end - 1]
        l_e.reinit(l_e.point_start(), subpoint_end)
    
    def perturb(self, w, point_start = None, point_end = "self"):
        """
        Perturb trainable parameters of network
        """
        if point_end == "self": point_end = self.point_end()
        self.validate_point(point_start)
        self.validate_point(point_end)

        idx_start = 0
        subpoint_start = None
        idx_end = 0
        subpoint_end = None

        if point_start is not None:
            idx_start = point_start[1]
            subpoint_start = point_start[2]

        if point_end is not None:
            idx_end = point_end[1] + 1
            subpoint_end = point_end[2]

        assert idx_end >= idx_start
        # Edge case - no nodes to reinit
        if idx_end - idx_start == 0: return
        # Edge case, reinit is entirely contained within subnode
        if idx_end - idx_start == 1:
            l_0 = self.layer[idx_start]
            l_0.perturb(w, subpoint_start, subpoint_end)
            return

        # Reinit starting layer potentially partially.
        l_0 = self.layer[idx_start]
        l_0.perturb(w, subpoint_start, l_0.point_end())
        # Fully reinit intermediate.
        for l in self.layer[idx_start + 1 : idx_end - 1]:
            l.perturb(w, l.point_start(), l.point_end())
        # Reinit ending layer potentially partially.
        l_e = self.layer[idx_end - 1]
        l_e.perturb(w, l_e.point_start(), subpoint_end)

    def clone_for_shape(self, original_shapes_in, new_shapes_in=None, new_shape_out=None):
        assert original_shapes_in is None or len(original_shapes_in) == 1, "Sequence is a input 1-parameter layer"
        assert new_shapes_in is None or len(new_shapes_in) == 1, "Sequence is a input 1-parameter layer"
        # Determine the original shape.
        current_shape = original_shapes_in[0]
        original_shapes = [current_shape]
        for layer in self.layer:
            current_shape = layer.in_shapes_to_out_shape([current_shape])
            original_shapes.append(current_shape)
        
        new_shapes = [s for s in original_shapes]
        fixed = np.full(len(new_shapes), False)
        if new_shapes_in is not None:
            new_shapes[0] = new_shapes_in[0]
            fixed[0] = True
            # Perform forward pass
            if fixed[0]:
                current_shape = new_shapes[0]
                for idx, layer in enumerate(self.layer, 1):
                    if not layer.controls_num_features():
                        current_shape = layer.in_shapes_to_out_shape([current_shape])
                        new_shapes[idx] = current_shape
                        fixed[idx] = True
                    else:
                        break
        if new_shape_out is not None:
            new_shapes[-1] = new_shape_out
            fixed[-1] = True
            # Backward pass
            layers = list(self.layer)
            current_out_shape = new_shape_out
            for idx, layer in zip(range(len(layers), 0, -1), layers[::-1]):
                new_in_shapes = layer.adjust_in_shapes_for_out_shape([new_shapes[idx]], [fixed[idx]], current_out_shape)
                current_out_shape = new_in_shapes[0]
                new_shapes[idx] = current_out_shape


class Concat(ModuleT):
    def get_reconstructor(self):
        return lambda: Concat()

    def forward(self, *x):
        return torch.cat(x, dim=1)

    def alters_shape(self):
        """
        Does this layer alter the shape of a feature map?
        """
        return True
    
    def in_shapes_to_out_shape(self, shapes_in):
        # Note, shape is assumed to include sample index!
        r = shapes_in[0]
        fs = r[1]
        for sh in shapes_in:
            # Assert shapes are compatible.
            assert sh[0] == r[0]
            assert sh[2:] == r[2:]
            fs += sh[1]
        # if compatible. shape is:
        return [r[0], fs] + r[2:]

    def adjust_in_shapes_for_out_shape(self, shapes_in, shapes_in_fixed, shape_out):
        # annoyingly enough - this cannot be performed atomically - we need to perform some kind of unification algorithm
        # to figure out what is (and isn't) possible. If nothing is possible, no matter what we do, we would have to avoid
        # adjusting the shapes to the preferred input shapes.
        for sh_in, is_fixed in zip(shapes_in, shapes_in_fixed):
            if is_fixed:
                assert sh_in == shape_out, "conflict - cannot adjust shapes to match fixed value"
        return [shape_out for _ in shapes_in]

class Concatenate(ModuleT):
    def __init__(self, *submodules):
        super().__init__()
        self.submodules = nn.ModuleList(submodules)
    
    def get_reconstructor(self):
        reconstructors = [m.get_reconstructor() for m in self.submodules]
        return lambda: Concatenate([r() for r in reconstructors])

    def forward(self, x):
        return torch.cat([m(x) for m in self.submodules], dim=-1)

    def point_start(self):
        """Give the point for the start of this layer."""
        return None

    def point_end(self):
        """Give the point for the end of this layer."""
        # special case: concatenation merges things together.
        return (self, -1, self)

    def validate_point(self, point):
        if point is None:
            return  # None represents the input to this layer
        assert len(point) == 3, "Point should be 3-element tuple for Concatenate"
        assert point[0] == self, "First element to point should be reference to self."
        assert point[1] < len(self.submodules), "idx should be in range."

    def reinit(self, point_start=None, point_end="self"):
        """
        Reinitialize trainable parameters of network
        """
        if point_end == "self": point_end = self.point_end()
        self.validate_point(point_start)
        self.validate_point(point_end)

        point_start_is_in_subnet = (
            point_start is not None and point_start[2] is not self
        )
        point_end_is_in_subnet = point_end is not None and point_end[2] is not self

        # If reinitializing a subnetwork - forward it.
        # Points MUST be in the same subnetwork as otherwise the range is not bounded properly
        if point_start_is_in_subnet and point_end_is_in_subnet:
            assert point_start[1] == point_end[1], "Range defined by points is not properly bounded."
            self.submodules[point_start[1]].reinit(point_start[2], point_end[2])
            return
        # If neither is in the subnetwork, reinitialize all subnetworks - if start is None and end is self
        if not point_end_is_in_subnet and not point_start_is_in_subnet:
            if point_start is not None or point_end is not self: return
            for subnet in self.submodules:
                subnet.reinit(subnet.point_start(), subnet.point_end())
        # Otherwise reinitialize the subnetwork in which the point lies
        if point_start_is_in_subnet:
            subnet = self.submodules[point_start[1]]
            subnet.reinit(point_start[2], subnet.point_end())
        if point_end_is_in_subnet:
            subnet = self.submodules[point_end[1]]
            subnet.reinit(subnet.point_start(), point_end[2])

    def perturb(self, w, point_start=None, point_end="self"):
        """
        Perturb trainable parameters of network
        """
        if point_end == "self": point_end = self.point_end()
        self.validate_point(point_start)
        self.validate_point(point_end)

        point_start_is_in_subnet = (
            point_start is not None and point_start[2] is not self
        )
        point_end_is_in_subnet = point_end is not None and point_end[2] is not self

        # If reinitializing a subnetwork - forward it.
        # Points MUST be in the same subnetwork as otherwise the range is not bounded properly
        if point_start_is_in_subnet and point_end_is_in_subnet:
            assert point_start[1] == point_end[1], "Range defined by points is not properly bounded."
            self.submodules[point_start[1]].perturb(w, point_start[2], point_end[2])
            return
        # If neither is in the subnetwork, reinitialize all subnetworks - if start is None and end is self
        if not point_end_is_in_subnet and not point_start_is_in_subnet:
            if point_start is not None or point_end is not self: return
            for subnet in self.submodules:
                subnet.perturb(w, subnet.point_start(), subnet.point_end())
        # Otherwise reinitialize the subnetwork in which the point lies
        if point_start_is_in_subnet:
            subnet = self.submodules[point_start[1]]
            subnet.perturb(w, point_start[2], subnet.point_end())
        if point_end_is_in_subnet:
            subnet = self.submodules[point_end[1]]
            subnet.perturb(w, subnet.point_start(), point_end[2])

    def to_subgraph(self, gc: 'GraphConstructor', feature_inputs):
        concat_inputs = []
        for subnet in self.submodules:
            out = subnet.to_subgraph(gc, feature_inputs)
            concat_inputs.append((len(concat_inputs), out))
        
        concat = Concat()
        out = concat.to_subgraph(gc, concat_inputs)

        return out

    def enumerate_points(self, include_input=False):
        if include_input:
            yield None
        for idx, l in enumerate(self.submodules):
            for p in l.enumerate_points(include_input=include_input):
                yield (self, idx, p)

    def get_module_by_point(self, point):
            self.validate_point(point)
            # Special case...
            if point is None: return None
            
            point_submodule = self.submodules[point[1]].get_module_by_point(point[2])
            return point_submodule

class ReLU(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def get_reconstructor(self):
        return lambda: ReLU()

    def __init__(self):
        super().__init__()
        self.layer = nn.ReLU()

    def forward(self, x):
        return self.layer(x)

class ActivationFunction(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def set_standard_activation_function(self, fn):
        self.layer = fn

    def get_reconstructor(self):
        layer = self.layer
        # note - we actually pickle a module here.
        return lambda: ActivationFunction(fn=layer)

    def __init__(self, fn=nn.ReLU()):
        super().__init__()
        self.layer = fn

    def forward(self, x):
        return self.layer(x)

class Identity(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def get_reconstructor(self):
        return lambda: Identity()

    def __init__(self):
        super().__init__()
        self.layer = nn.Identity()

    def forward(self, x):
        return self.layer(x)


class MaxPool2d(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def __init__(self, *x, **y):
        super().__init__()
        self.params = (x, y)
        self.layer = nn.MaxPool2d(*x, **y)

    def get_reconstructor(self):
        x, y = self.params
        return lambda: MaxPool2d(*x, **y)

    def forward(self, x):
        return self.layer(x)


class Conv2d(ModuleT):
    def controls_num_features(self):
        return True

    def requires_num_features(self):
        return True

    def __init__(self, in_features, out_features, *x, pad_or_trim_tensor=True, **y):
        super().__init__()
        self.params = (in_features, out_features, x, y)
        self.layer = nn.Conv2d(in_features, out_features, *x, **y)
        self.pad_or_trim_tensor = pad_or_trim_tensor
        self.in_features = in_features

    def get_reconstructor(self):
        in_features, out_features, x, y = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: Conv2d(in_features, out_features, *x, pad_or_trim_tensor, **y)

    def forward(self, x):
        assert isinstance(x, torch.Tensor), "x should be a tensor"
        if self.pad_or_trim_tensor and x.shape[1] != self.in_features:
            if x.shape[1] > self.in_features:
                x = x[:, :self.in_features, ...]
            else:
                padding = [0, 0, 0, 0, 0, self.in_features - x.shape[1]]
                x = torch.nn.functional.pad(x, padding)

        return self.layer(x)

    def get_reinit_layer(self):
        in_features, out_features, x, y = self.params
        return nn.Conv2d(in_features, out_features, *x, **y)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)


class Linear(ModuleT):
    def controls_num_features(self):
        return True

    def requires_num_features(self):
        return True

    def __init__(self, in_features, out_features, *x, pad_or_trim_tensor = True, **y):
        super().__init__()
        self.params = (in_features, out_features, x, y)
        self.layer = nn.Linear(in_features, out_features, *x, **y)
        self.pad_or_trim_tensor = pad_or_trim_tensor
        self.in_features = in_features

    def forward(self, x):
        if self.pad_or_trim_tensor and x.shape[1] != self.in_features:
            if x.shape[1] > self.in_features:
                x = x[:, :self.in_features]
            else:
                padding = [0, self.in_features - x.shape[1]]
                x = torch.nn.functional.pad(x, padding)
        return self.layer(x)

    def get_reconstructor(self):
        in_features, out_features, x, y = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: Linear(in_features, out_features, *x, pad_or_trim_tensor, **y)

    def get_reinit_layer(self):
        in_features, out_features, x, y = self.params
        return nn.Linear(in_features, out_features, *x, **y)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)


class BatchNorm1d(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return True

    def __init__(self, num_features, pad_or_trim_tensor=True, **x):
        super().__init__()
        self.params = (num_features, x)
        self.layer = nn.BatchNorm1d(num_features, **x)
        self.pad_or_trim_tensor = pad_or_trim_tensor
        self.in_features = num_features

    def forward(self, x):
        if self.pad_or_trim_tensor and x.shape[1] != self.in_features:
            if x.shape[1] > self.in_features:
                x = x[:, :self.in_features, ...]
            else:
                padding = [0 for _ in range(2 * (len(x.shape) - 2))] + [0, self.in_features - x.shape[1]]
                x = torch.nn.functional.pad(x, padding)

        return self.layer(x)

    def get_reconstructor(self):
        num_features, x = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: BatchNorm1d(num_features, pad_or_trim_tensor, **x)

    def get_reinit_layer(self):
        num_features, x = self.params
        return nn.BatchNorm1d(num_features, **x)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)


class BatchNorm2d(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return True

    def __init__(self, num_features, pad_or_trim_tensor=True, **x):
        super().__init__()
        self.params = (num_features, x)
        self.layer = nn.BatchNorm2d(num_features, **x)
        self.pad_or_trim_tensor = pad_or_trim_tensor
        self.in_features = num_features

    def forward(self, x):
        if self.pad_or_trim_tensor and x.shape[1] != self.in_features:
            if x.shape[1] > self.in_features:
                x = x[:, :self.in_features, ...]
            else:
                padding = [0, 0, 0, 0, 0, self.in_features - x.shape[1]]
                x = torch.nn.functional.pad(x, padding)

        return self.layer(x)
    
    def get_reinit_layer(self):
        num_features, x = self.params
        return nn.BatchNorm2d(num_features, **x)

    def get_reconstructor(self):
        num_features, x = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: BatchNorm2d(num_features, pad_or_trim_tensor, **x)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)

class LayerNorm(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return True

    def __init__(self, num_features, pad_or_trim_tensor=True, **x):
        super().__init__()
        self.params = (num_features, x)
        self.layer = nn.LayerNorm([num_features], **x)
        self.pad_or_trim_tensor = pad_or_trim_tensor
        self.in_features = num_features

    def forward(self, x):
        if self.pad_or_trim_tensor and x.shape[1] != self.in_features:
            if x.shape[1] > self.in_features:
                x = x[:, :self.in_features, ...]
            else:
                padding = [self.in_features - x.shape[1]]
                x = torch.nn.functional.pad(x, padding)

        return self.layer(x)
    
    def get_reinit_layer(self):
        num_features, x = self.params
        return nn.LayerNorm([num_features], **x)

    def get_reconstructor(self):
        num_features, x = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: LayerNorm(num_features, pad_or_trim_tensor, **x)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)

class InstanceNorm1d(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return True

    def __init__(self, num_features, pad_or_trim_tensor=True, **x):
        super().__init__()
        self.params = (num_features, x)
        self.layer = nn.InstanceNorm1d(num_features, **x)
        self.pad_or_trim_tensor = pad_or_trim_tensor
        self.in_features = num_features

    def forward(self, x):
        if self.pad_or_trim_tensor and x.shape[1] != self.in_features:
            if x.shape[1] > self.in_features:
                x = x[:, :self.in_features, ...]
            else:
                padding = [0, 0, 0, self.in_features - x.shape[1]]
                x = torch.nn.functional.pad(x, padding)

        return self.layer(x)
    
    def get_reinit_layer(self):
        num_features, x = self.params
        return nn.InstanceNorm1d(num_features, **x)

    def get_reconstructor(self):
        num_features, x = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: InstanceNorm1d(num_features, pad_or_trim_tensor, **x)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)

class InstanceNorm2d(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return True

    def __init__(self, num_features, pad_or_trim_tensor=True, **x):
        super().__init__()
        self.params = (num_features, x)
        self.layer = nn.InstanceNorm2d(num_features, **x)
        self.pad_or_trim_tensor = pad_or_trim_tensor
        self.in_features = num_features

    def forward(self, x):
        if self.pad_or_trim_tensor and x.shape[1] != self.in_features:
            if x.shape[1] > self.in_features:
                x = x[:, :self.in_features, ...]
            else:
                padding = [0, 0, 0, 0, 0, self.in_features - x.shape[1]]
                x = torch.nn.functional.pad(x, padding)

        return self.layer(x)
    
    def get_reinit_layer(self):
        num_features, x = self.params
        return nn.InstanceNorm2d(num_features, **x)

    def get_reconstructor(self):
        num_features, x = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: InstanceNorm2d(num_features, pad_or_trim_tensor, **x)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)

class Dropout(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def __init__(self, p=0.5, pad_or_trim_tensor=True, **x):
        super().__init__()
        self.params = (p, x)
        self.layer = nn.Dropout(p, **x)

    def forward(self, x):
        return self.layer(x)
    
    def get_reinit_layer(self):
        p, x = self.params
        return nn.Dropout(p, **x)

    def get_reconstructor(self):
        p, x = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: Dropout(p, pad_or_trim_tensor, **x)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)

class Dropout1d(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def __init__(self, p=0.5, pad_or_trim_tensor=True, **x):
        super().__init__()
        self.params = (p, x)
        self.layer = nn.Dropout1d(p, **x)

    def forward(self, x):
        return self.layer(x)
    
    def get_reinit_layer(self):
        p, x = self.params
        return nn.Dropout1d(p, **x)

    def get_reconstructor(self):
        p, x = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: Dropout1d(p, pad_or_trim_tensor, **x)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)

class Dropout2d(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def __init__(self, p=0.5, pad_or_trim_tensor=True, **x):
        super().__init__()
        self.params = (p, x)
        self.layer = nn.Dropout2d(p, **x)

    def forward(self, x):
        return self.layer(x)
    
    def get_reinit_layer(self):
        p, x = self.params
        return nn.Dropout2d(p, **x)

    def get_reconstructor(self):
        p, x = self.params
        pad_or_trim_tensor = self.pad_or_trim_tensor
        return lambda: Dropout2d(p, pad_or_trim_tensor, **x)

    def reinit_self(self):
        self.layer = self.get_reinit_layer()

    def perturb_self(self, w):
        reinit_layer = self.get_reinit_layer()
        reinit_params = reinit_layer.named_parameters()
        current_params = self.layer.named_parameters()
        perturb(current_params, reinit_params, w)

class Flatten(ModuleT):
    def controls_num_features(self):
        # Well. It does - but not in a way you can control or set...
        # For now: assume that we are flattening something of shape [b, 1..., f].
        return False

    def requires_num_features(self):
        return False

    def get_reconstructor(self):
        return lambda: Flatten()

    def __init__(self, *x, **y):
        super().__init__()
        self.layer = nn.Flatten(*x, **y)

    def forward(self, x):
        return self.layer(x)


class Softmax(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Softmax(dim)

    def get_reconstructor(self):
        return lambda: Softmax()

    def forward(self, x):
        return self.layer(x)
    
class SoftmaxOutput(ModuleT):
    def controls_num_features(self):
        return False

    def requires_num_features(self):
        return False

    def __init__(self, dim):
        super().__init__()
        self.layer = nn.Softmax(dim)
        self.logit_mode = False

    def get_reconstructor(self):
        return lambda: SoftmaxOutput()

    def forward(self, x):
        if self.logit_mode:
            return x
        else:
            return self.layer(x)

class LinearCombine(ModuleT):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws

    def get_reconstructor(self):
        ws = self.ws
        return lambda: LinearCombine(ws)
    
    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], List):
            xs = xs[0]
        assert len(xs) == len(self.ws)
        return sum(x * w for x, w in zip(xs, self.ws))

class BayesCombine(ModuleT):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws
        self.logit_mode = False
    
    def get_reconstructor(self):
        ws = self.ws
        return lambda: BayesCombine(ws)
        
    def propagate_output(self):
        # If set to logit mode - please set output layers that provide input to logit mode too.
        return True

    def forward(self, *xs):
        if len(xs) == 1 and isinstance(xs[0], List):
            xs = xs[0]
        assert len(xs) == len(self.ws)
        if not self.logit_mode:
            r = torch.prod(torch.stack([x ** w for x, w in zip(xs, self.ws)]), dim=0)
            r /= r.sum(keepdims=True, dim=1)
        else:
            r = sum(x * w for x, w in zip(xs, self.ws))
            # Note - logit output - not normalizing here.
        return r

class NeuralNetGraphNode(nn.Module):
    def __init__(self, graph, module_idx):
        super().__init__()
        self.graph = graph
        self.module_idx = module_idx

    def forward(self, *x, **y):
        if self.module_idx < 0:
            assert len(x) == 1
            return x[0]
        return self.graph.submodules[self.module_idx](*x, **y)

def set_or_update_has_multiple_occurences(r, to):
    # Tag top
    if hasattr("single_occurence"):
        # Update
        r.single_occurence &= to
    else:
        try:
            r.single_occurence = to
        except:
            pass
    
    # Tag recursively
    if isinstance(r, list) or isinstance(r, tuple):
        for e in r:
            set_or_update_has_multiple_occurences(e, to)
    elif isinstance(r, dict):
        for v in r.values():
            set_or_update_has_multiple_occurences(v, to)

def get_has_multiple_occurences(v):
    # note: maybe try using something like sys.getrefcount?
    # however, we might expect more than few duplicates in here:
    # +1 for the call to sys.getrefcount
    # +1 for the call to this method
    # +1 for the call to the forward function of the module itself
    # +1 for the current wrapper
    # +1 for storage w.r.t. the forward pass (i.e. in the list)
    try:
        return v.single_occurence
    except:
        # if cannot check, assume there are multiple occurences.
        return True

class NeuralNetGraph(ModuleT):

    def __init__(self, modules=[], nodes=[-2, -1], edges=[], sockets=[], consumptions=[], graph=None):
        """
        Create a neural-net graph.
        """
        super().__init__()
        self.end_node = nodes.index(-1)
        self.submodules = nn.ModuleList(modules)
        if graph is None:
            self.graph = ig.Graph(n=len(nodes), directed=True)
            self.graph.add_edges(edges, attributes=dict(socket=sockets))
            # special cases: -2 is input, -1 is output
            self.graph.vs["module"] = nodes
        else:
            self.graph = graph

        self.graph.vs["consumed_by"] = consumptions

        # Note - should always reflect self.graph.vs["module"], used to allow for
        # usage of pytorch's hooks within the graph.
        self.determine_wrapper_nodes()

        self.tag_occurences = False
        self.determine_sorting()

    def tag_occurences(self, b):
        self.tag_occurences = b

    def determine_sorting(self):
        # construct a copy of the graph as to account for consumptions
        g = self.graph.copy()
        # add 'consumption' edges to nodes using the output
        # to force the topological sorting account for the consumption.
        edges_to_add = []
        for v in g.vs:
            # Is this output consumed?
            c = v["consumed_by"]
            # If no, skip
            if c == None: continue
            # If yes, add edges from each node to which there is an outgoing edge
            # as their input is potentially consumed (and as such, need to 'allow'
            # the consumption to happen)
            for e in v.out_edges():
                # however, the consuming node itself does not need to allow
                # (we would otherwise get a self-loop)
                if e.target == c: continue
                edges_to_add.append((e.target, c))
        g.add_edges(edges_to_add)

        # compute the topological sorting
        self.ord = g.topological_sorting()

    def determine_wrapper_nodes(self):
        self.wrapper_nodes = [NeuralNetGraphNode(self, v["module"]) for v in self.graph.vs]

    def generate_closed_forward_fn(self):
        values = [{} for _ in range(len(self.graph.vs))]

        lines = ["def gen_nng_forward(wrapper_nodes, X):"]

        for idx in self.ord:
            c = self.graph.vs[idx]
            if c["module"] == -2:
                # Push initial value - does not generate any code
                for edge in c.out_edges():
                    values[edge.target][edge["socket"]] = "X"
                continue
            if c["module"] == -1:
                lines.append(f"\treturn {values[idx][0]}")
                return "\n".join(lines)
                # eval("\n".join(lines), {}, {"submodules": self.submodules})
                # gen_nng_forward should be defined by eval above
                # return gen_nng_forward

            # Use module - gather inputs and apply it
            # Split by key-type - first positional
            inputs_posargs = [(k, v) for (k, v) in values[idx].items() if isinstance(k, int)]
            inputs_posargs.sort()
            inputs_posargs = [v for (k, v) in inputs_posargs]
            # Then keywords
            inputs_kwargs = [(k, v) for (k, v) in values[idx].items() if isinstance(k, str)]
            # inputs = [v for _k, v in sorted()]
            # sidenote - pytorch might not like this...

            lines.append((f"\tv{idx} = wrapper_nodes[{idx}]({', '.join(inputs_posargs)}"
                          "".join(f"{k}={v}," for k, v in inputs_kwargs)))
            
            # Reset - this node has been used.
            values[idx] = None
            # Propagate resulting values.
            for edge in c.out_edges():
                values[edge.target][edge["socket"]] = f"v{idx}"
        pass

    def get_reconstructor(self):
        modules_reconstruct = [m.get_reconstructor() for m in self.submodules]
        graph = self.graph

        return lambda: NeuralNetGraph(modules=[m() for m in modules_reconstruct], graph=graph)

    def forward(self, X):
        values = [{} for _ in range(len(self.graph.vs))]

        for pos, idx in enumerate(self.ord):
            c = self.graph.vs[idx]
            if c["module"] == -2:
                # Push initial value
                for edge in c.out_edges():
                    values[edge.target][edge["socket"]] = self.wrapper_nodes[idx](X)
                continue
            if c["module"] == -1:
                return values[idx][0]

            # Use module - gather inputs and apply it
            # Split by key-type - first positional
            inputs_posargs = [(k, v) for (k, v) in values[idx].items() if isinstance(k, int)]
            inputs_posargs.sort()
            inputs_posargs = [v for (k, v) in inputs_posargs]
            # Then keywords
            inputs_kwargs = {k: v for (k, v) in values[idx].items() if isinstance(k, str)}

            # sidenote - pytorch might not like this...
            try:
                r = self.wrapper_nodes[idx](*inputs_posargs, **inputs_kwargs)
            except Exception as e:
                input_shapes = ','.join(f'{t.shape}' for t in inputs_posargs)
                # note - on error kwargs are currently excluded - if ever necessary: include
                print("Original Exception:")
                traceback.print_exc()
                print(f"Has keyword arguments {list(inputs_kwargs.keys())}")
                raise Exception(f"Forwarding layer {idx} @ {pos} (module {c['module']}) failed: {e}. Input shapes: {input_shapes}") from e
            # Reset - this node has been used.
            values[idx] = None
            if self.tag_occurences:
                # Single-occurence tagging - i.e. does this value occur more than once?
                # important to determine whether in-place modification may occur.
                # If multiple occurences exist - we must clone if backprop is potentially used
                # (and even if not - we need to be able to ensure that the current occurence is
                #  the last one - which we can never really fully guarantee, unless we track this
                #  more closely)
                # Note that this relies closely on a node only having a single output in which
                # the content is not duplicated (as this would not be detected).
                # and that the output of any node is allowed to be modified in-place, assuming
                # no external duplicates. This is not necessarily the case when a node internally duplicates
                # and returns this.
                # For now, we assume that this is not something we need to worry about, and that the output
                # of every module can be reused without recourse.
                is_not_splitting = len(c.out_edges()) <= 1
                set_or_update_has_multiple_occurences(r, is_not_splitting)
            # Propagate resulting values.
            for edge in c.out_edges():
                values[edge.target][edge["socket"]] = r

        # If we end up here - somehow the end-node is missing...
        # That should not happen.
        raise ValueError("End node is missing in the current graph.")

    def set_standard_activation_function(self, fn):
        for m in self.submodules:
            m.set_standard_activation_function(fn)

    def configure_output(self, f):
        # 'output' node
        vs_to_propagate_from = [1]
        seen = set()
        while len(vs_to_propagate_from) > 0:
            vi = vs_to_propagate_from.pop()
            v = self.graph.vs[vi] 
            # assert v["module"] == -1
            for e in v.in_edges():
                v = self.graph.vs[e.source]
                m = self.submodules[v["module"]]
                f(m)
                if m.propagate_output() and e.source not in seen:
                    seen.add(e.source)
                    vs_to_propagate_from.append(e.source)

        # for m in self.submodules:
        #     if m is None: continue
        #     if not m.is_output(): continue
        #     f(m)
 
    def enumerate_points(self, include_input=False):
        for idx, v in enumerate(self.graph.vs):
            if v["module"] < -1 and not include_input:
                continue
            yield (self, idx)

    def train_restore(self):
        """Restore training state to what it was prior to calling eval_ss"""
        for m in self.modules():
            try:
                m.train(m.was_training)
            except:
                # Did not have was_training attribute - assume
                # new & training mode should already be enabled.
                pass

    def store_state_eval(self):
        for m in self.modules():
            m.was_training = m.training
        self.eval()

    def prune_unused(self, cleanup_modules=True, clone_modules=False, return_mapping=False):
        # Determines which nodes are reachable, removing everything else.
        determines_output = self.graph.subcomponent(1, mode='in')
        
        not_determines_output = set(range(len(self.graph.vs))) - set(determines_output)
        self.graph.delete_vertices(not_determines_output)
        self.determine_sorting()

        module_idx_mapping = {}
        if cleanup_modules:
            # Some of the modules will end up being unused. Figure out which ones are used
            # and compact the list of used modules.
            new_modules = []
            for v in self.graph.vs:
                midx = v["module"]
                if midx < 0: continue
                new_midx = module_idx_mapping.get(midx)
                if new_midx is None:
                    new_midx = len(new_modules)
                    module_idx_mapping[midx] = new_midx
                    module = self.submodules[midx]
                    if clone_modules:
                        module = deepcopy(module)
                    new_modules.append(module)
                v["module"] = new_midx
            self.submodules = nn.ModuleList(new_modules)
        # reinit wrapper nodes - otherwise the update to the module does not apply,
        # and these nodes have become incorrect.
        self.determine_wrapper_nodes()

        if return_mapping:
            return module_idx_mapping

    def to_subgraph_recurse(self, gc: 'GraphConstructor', feature_inputs, return_nodemap=False):

        ordering = self.graph.topological_sorting()
        nodemap = list(self.graph.vs["module"])
        out = None
        for original_nidx in ordering:
            vertex = self.graph.vs[original_nidx]
            mod_idx = vertex["module"]
            
            def map_original_idx_to_new_idx(idx):
                nidx = nodemap[idx]
                # simple case - we have already converted this edge.
                if nidx >= 0: return nidx
                # -1 is output, should not be used as an input
                if nidx == -2: return feature_inputs[0][1]
                raise Exception()

            if mod_idx < 0:
                # special case - either input or output.
                # if input, we don't need to do anything.
                # if output. figure out who is incident.
                if mod_idx == -1:
                    for e in vertex.in_edges():
                        out = map_original_idx_to_new_idx(e.source)
                continue
            input_new_idxs = [(edge["socket"], map_original_idx_to_new_idx(edge.source)) for edge in vertex.in_edges()]
            # input_new_idxs.sort()
            # input_new_idxs = [i for (s, i) in input_new_idxs]
            module = self.submodules[mod_idx]
            module.to_subgraph(gc, input_new_idxs)

        if return_nodemap:
            return out, nodemap
        return out

    def to_subgraph_copy(self, gc: 'GraphConstructor', feature_inputs, return_nodemap=False):
        # old version - 1-for-1 copy without using to_subgraph
        nodemap = [gc.new_node(self.submodules[mod_idx]) if mod_idx >= 0 else mod_idx for mod_idx in self.graph.vs["module"]]
        out = None
        for edge in self.graph.es:
            source = nodemap[edge.source]
            target = nodemap[edge.target]
            socket = edge["socket"]
            # Remap input to the feature input provided
            if source == -2: source = feature_inputs[0]
            # If output, mark source as output and skip.
            if target == -1:
                out = source
                continue
            gc.new_edge(source, target, socket)
        if return_nodemap:
            return out, nodemap
        return out

    def to_subgraph(self, gc: 'GraphConstructor', feature_inputs: list[tuple], return_nodemap=False):

        ordering = self.graph.topological_sorting()
        nodemap = list(self.graph.vs["module"])
        out = None
        for original_nidx in ordering:
            vertex = self.graph.vs[original_nidx]
            mod_idx = vertex["module"]
            
            def map_original_idx_to_new_idx(idx):
                nidx = nodemap[idx]
                # simple case - we have already converted this edge.
                if nidx >= 0: return nidx
                # -1 is output, should not be used as an input
                if nidx == -2: return feature_inputs[0][1]
                raise Exception()

            if mod_idx < 0:
                # special case - either input or output.
                # if input, we don't need to do anything.
                # if output. figure out who is incident.
                if mod_idx == -1:
                    for e in vertex.in_edges():
                        out = map_original_idx_to_new_idx(e.source)
                continue
            input_new_idxs = [(edge["socket"], map_original_idx_to_new_idx(edge.source)) for edge in vertex.in_edges()]
            # input_new_idxs.sort(key=lambda v: f"ZZZ{v[0]:05d}" if isinstance(v[0], int) else v[0])
            # input_new_idxs = [i for (s, i) in input_new_idxs]
            module = self.submodules[mod_idx]
            new_node_id = module.to_subgraph(gc, input_new_idxs)
            assert isinstance(new_node_id, int)
            nodemap[original_nidx] = new_node_id

        if return_nodemap:
            return out, nodemap
        return out

    def point_start(self):
        # Special case: symbolizes all nodes with 'modules' < -1
        return None

    def point_end(self):
        # Hardcoded end node.
        return (self, self.end_node)

    def get_module_by_point(self, point):
        assert point[0] is self, "First element of graph must be the graph itself"
        return self.wrapper_nodes[point[1]]
    
    def find_spanned_nodes(self, points_start, points_end, include_start=False):
        # Handle edge case
        if points_start is None:
            points_start = [v for v in self.graph.vs if v["module"] < -1]

        def check_self_and_return_idx(point):
            assert point[0] is self
            return point[1]

        # Normalize to iterables
        try:
            points_start = [check_self_and_return_idx(v) for v in iter(points_start)]
        except TypeError:
            assert points_start[0] is self, "First element of graph must be the graph itself"
            points_start = [points_start[1]]
        try:
            points_end = [check_self_and_return_idx(v) for v in iter(points_end)]
        except TypeError:
            assert points_end[0] is self, "First element of graph must be the graph itself"
            points_end = [points_end]
        
        # Find nodes of whose the current value 
        reachable_from_start = set()
        for v in points_start:
            reachable_from_start += set(self.graph.subcomponent(v, mode='out'))

        reachable_from_end = set()
        for v in points_end:
            reachable_from_end += set(self.graph.subcomponent(v, mode='in'))

        reachable_from_both = reachable_from_start.intersection(reachable_from_end)
        if include_start:
            return reachable_from_both
        else:
            return reachable_from_both - set(points_start)

    def reinit(self, start_point, end_point):
        """
        Reinitialize the weights of the network.
        """
        idxs = self.find_spanned_nodes(start_point, end_point)
        module_idxs = np.array([v["module"] for v in self.graph.vs[idxs]])

        for m_idx in module_idxs:
            subm = self.submodules[m_idx]
            subm.reinit()
    
    def perturb(self, w, start_point, end_point):
        """
        Perturb the weights of the network.

        Resulting weights are W * (1 - w) + I * w, where W are the current weights
        and I are weights from the modules being reinitialized.
        """
        idxs = self.find_spanned_nodes(start_point, end_point)
        module_idxs = np.array([v["module"] for v in self.graph.vs[idxs]])
        # if a module is referred to more than once, do we perturb multiple times?
        # currently, yes. but we may want to deduplicate and see if that is better.

        for m_idx in module_idxs:
            subm = self.submodules[m_idx]
            subm.perturb(w)

    def to_dot(self, out, include_ord_label=False):
        # note - for now assume a fully encompassed graph (with no subgraphs)
        if include_ord_label:
            ordloc = np.argsort(self.ord)

        def convert_node(idx):
            graph_vertex = self.graph.vs[idx]
            module_idx = graph_vertex["module"]
            if module_idx >= 0:
                module_str = str(self.submodules[module_idx])
                module_str = '\\n'.join(textwrap.wrap(module_str, width=30))
                module_str = f"(M{module_idx}) {module_str}"
            elif module_idx == -1:
                module_str = "Out"
            elif module_idx < -1:
                module_str = f"In {-(module_idx + 1)}"

            in_labels = " | ".join((f'<s{e["socket"]}> {e["socket"]}' for e in graph_vertex.in_edges()))
            in_labels = f"{{{in_labels}}}"
            
            pos_label = ""
            if include_ord_label:
                pos_label = f" @ {ordloc[idx]}"

            return f'n{idx}[label="{{ {in_labels} | {module_str}{pos_label} | <so> Out }}"];\n'

        def convert_edge(idx):
            graph_edge = self.graph.es[idx]
            
            return f'n{graph_edge.source}:so -> n{graph_edge.target}:s{graph_edge["socket"]};\n'

        out.write('digraph {\n')
        out.write('graph [rankdir = LR];\n')
        out.write('node[shape=record];\n')

        for vidx in range(len(self.graph.vs)):
            out.write(convert_node(vidx))

        for eidx in range(len(self.graph.es)):
            out.write(convert_edge(eidx))

        out.write('}\n')

class LinearAggregate(ModuleT):
    def __init__(self, ws):
        super().__init__()
        
        self.ws = nn.Parameter(torch.tensor(ws).reshape(-1), requires_grad=False)

    def get_reconstructor(self):
        ws = self.ws
        return lambda: LinearAggregate(ws)

    def forward(self, *x):
        assert len(x) == len(self.ws)
        stacked = torch.stack(x, dim=0)
        return torch.sum(stacked * self.ws.reshape([-1 if x == 0 else 1 for x in range(len(stacked.shape))]), dim=0)      

class LinearEnsemble(ModuleT):
    def __init__(self, submodules, ws):
        super().__init__()
        self.submodules = nn.ModuleList(submodules)
        self.ws = ws

    def get_reconstructor(self):
        ws = self.ws
        submodules_reconstructor = [m.get_reconstructor() for m in self.submodules]
        return lambda: LinearEnsemble([m() for m in submodules_reconstructor], ws)
    
    def forward(self, x):
        r = None
        for sm, w in zip(self.submodules, self.ws):
            if r is None:
                r = sm(x) * w
            else:
                r += sm(x) * w
        return r

    def to_subgraph(self, gc: 'GraphConstructor', feature_inputs):
        agg = LinearAggregate(self.ws)
        agg_inputs = [(i, sm.to_subgraph(gc, feature_inputs)) for i, sm in enumerate(self.submodules)]
        out = agg.to_subgraph(gc, agg_inputs)
        return out


class BayesAggregate(ModuleT):
    def __init__(self, ws):
        super().__init__()
        self.ws = ws

    def get_reconstructor(self):
        ws = self.ws
        return lambda: BayesAggregate(ws)
    
    def forward(self, *X):
        r = torch.prod(torch.stack([x ** w for x, w in zip(X, self.ws)]), dim=0)
        r /= r.sum(keepdims=True, dim=1)
        return r

class BayesEnsemble(ModuleT):
    def __init__(self, submodules, ws):
        super().__init__()
        self.submodules = nn.ModuleList(submodules)
        self.ws = ws

    def get_reconstructor(self):
        ws = self.ws
        submodules_reconstructor = [m.get_reconstructor() for m in self.submodules]
        return lambda: BayesEnsemble([m() for m in submodules_reconstructor], ws)
    
    def to_subgraph(self, gc: 'GraphConstructor', feature_inputs):
        agg = BayesAggregate(self.ws)
        agg_inputs = [(i, sm.to_subgraph(gc, feature_inputs)) for i, sm in enumerate(self.submodules)]
        out = agg.to_subgraph(gc, agg_inputs)
        return out
    
    def forward(self, x):
        r = torch.prod(torch.stack([sm(x) ** w for sm, w in zip(self.submodules, self.ws)]), dim=0)
        r /= r.sum(keepdims=True, dim=1)
        return r

# Assuming dictionary output
class LinearDictAggregate(ModuleT):
    def __init__(self, ws, in_place=False):
        super().__init__()
        self.in_place = in_place
        self.ws = torch.nn.Parameter(torch.tensor(ws).reshape(-1), requires_grad=False)

    def get_reconstructor(self):
        ws = self.ws
        return lambda: LinearDictAggregate(ws)
    
    def forward_key(self, x, k):
        
        assert len(x) == len(self.ws)
        # original implementation, inefficient.
        # stacked = torch.stack([xv[k] for xv in x], dim=0)
        # return torch.sum(stacked * self.ws.reshape([-1 if x == 0 else 1 for x in range(len(stacked.shape))]), dim=0)
        
        r = x[0][k]
        if not self.in_place:
            r = torch.clone(r)
        for xv in x[1:]:
            r += xv[k]
        return r

    def forward(self, *x):
        # Note - assumes set of keys is always identical.
        return {
            k: self.forward_key(x, k)
            for k in x[0].keys()
        }

class LinearDictEnsemble(ModuleT):
    def __init__(self, submodules, ws):
        super().__init__()
        self.submodules = torch.nn.ModuleList(submodules)
        self.ws = ws

    def get_reconstructor(self):
        ws = self.ws
        submodules_reconstructor = [m.get_reconstructor() for m in self.submodules]
        return lambda: LinearDictEnsemble([m() for m in submodules_reconstructor], ws)
    
    def forward(self, x):
        r = None
        for sm, w in zip(self.submodules, self.ws):
            o = sm(x)
            if r is None:
                r = {k: o[k] * w for k in o.keys()}
            else:
                for k in o.keys():
                    r[k] += o[k] * w
        return r

    def to_subgraph(self, gc: GraphConstructor, feature_inputs):
        agg = LinearDictAggregate(self.ws)
        agg_inputs = [(i, sm.to_subgraph(gc, feature_inputs)) for i, sm in enumerate(self.submodules)]
        out = agg.to_subgraph(gc, agg_inputs)
        return out

def pad_feature_maps_to_common_size(xs):
    num_features = max(x.shape[1] for x in xs)
    num_dims = len(xs[0].shape)
    for x in xs:
        assert num_dims == len(x.shape)
    # image features, no padding, for now...
    suffix_padding = [0 for _ in range(2 * (num_dims - 2))]
    # 
    return [nn.functional.pad(x, suffix_padding + [0, num_features - x.shape[1]]) for x in xs]

class Add(ModuleT):
    def __init__(self, pad_or_trim_tensor=True):
        super().__init__()
        self.pad_or_trim_tensor = pad_or_trim_tensor

    def get_reconstructor(self):
        return lambda: Add()

    def forward(self, *x):
        # pad to common feature size.
        if self.pad_or_trim_tensor:
            x = pad_feature_maps_to_common_size(x)

        return torch.sum(torch.stack(x, dim=0), dim=0)

class Mean(ModuleT):

    def __init__(self, pad_or_trim_tensor=True):
        super().__init__()
        self.pad_or_trim_tensor = pad_or_trim_tensor

    def get_reconstructor(self):
        return lambda: Mean()

    def forward(self, *x):
        # pad to common feature size.
        if self.pad_or_trim_tensor:
            x = pad_feature_maps_to_common_size(x)

        return torch.mean(torch.stack(x, dim=0), dim=0)

class Residual(ModuleT):
    def controls_num_features(self):
        # Quite the opposite! Forces input = output.
        return False

    def requires_num_features(self):
        return False

    def __init__(self, layer, bypass=None):
        super().__init__()
        self.layer = layer
        self.bypass = bypass
        # Note: this 'layer' is decomposed when adding it to a NeuralNetGraph

    def get_reconstructor(self):
        layer_c = self.layer.get_reconstructor()
        return lambda: Residual(layer_c())
    
    def _add_identity_proxy(self, gc, fi):
        idt = Identity()
        return idt.to_subgraph(gc, [(0, fi[1])])

    def to_subgraph(self, gc: 'GraphConstructor', feature_inputs):
        # - direct 1-to-1
        # fi = self.layer.to_subgraph(gc, feature_inputs)
        # add_mod = Add()
        # out = add_mod.to_subgraph(gc, feature_inputs + [fi])
        
        fi = self.layer.to_subgraph(gc, feature_inputs)
        add_mod = Add()
        # add identity operation?
        # allows for branches to be indexed seperately & replaced.
        add_identity_op = True
        if self.bypass is not None:
            feature_inputs_proxy = [self.bypass.to_subgraph(gc, feature_inputs)]
        elif add_identity_op:
            feature_inputs_proxy = [self._add_identity_proxy(gc, fix) for fix in feature_inputs]
        else:
            feature_inputs_proxy = feature_inputs
        out = add_mod.to_subgraph(gc, feature_inputs_proxy + [(len(feature_inputs_proxy), fi)])
        return out

    def forward(self, x):
        if self.bypass is None:
            return x + self.layer(x)
        else:
            return self.bypass(x) + self.layer(x)

    def point_start(self):
        """Give the point for the start of this layer."""
        return None

    def point_end(self):
        """Give the point for the end of this layer."""
        # special case: concatenation merges things together.
        return (self, self)

    def validate_point(self, point):
        if point is None:
            return  # None represents the input to this layer
        assert len(point) == 2, "Point should be 2-element tuple for Residual"
        assert point[0] == self, "First element to point should be reference to self."

    def reinit(self, point_start=None, point_end="self"):
        if point_end == "self": point_end = self.point_end()
        # If the start point is the endpoint, or the endpoint is the starting point
        # We do not reinit at all.
        if point_start is not None and point_start[1] is self: return
        if point_end is None: return

        # This residual layer does not have any side branches, so simply
        # reinit the sublayer as is.
        if point_start is None: point_start = self.layer.point_start()
        if point_end[1] is self: point_end = self.layer.point_end()
        self.layer.reinit(point_start, point_end)

    
    def enumerate_points(self, include_input=False):
        if include_input:
            yield None
        for p in self.layer.enumerate_points(include_input=include_input):
            yield (self, p)
        yield (self, self)

    def get_module_by_point(self, point):
            self.validate_point(point)
            # Special cases
            if point is None: return None
            if point[1] is self: return self

            return self.layer.get_module_by_point(point[1])
    

def maybe_dropout(kind, dropout_when, dropout_p=0.5):
    if kind != dropout_when:
        return []
    return [Dropout(dropout_p)]

def maybe_dropout1d(kind, dropout_when, dropout_p=0.5):
    if kind != dropout_when:
        return []
    return [Dropout1d(dropout_p)]

def normalization_layer(kind, sh):
    # sh is the shape excluding batch dimension
    nc = sh[0]
    if kind == "batch":
        if len(sh) == 1 or len(sh) == 2:
            return [BatchNorm1d(nc)]
        if len(sh) == 3:
            return [BatchNorm2d(nc)]
    elif kind == "instance":
        # note: we are sometimes using it iin the same position as batchnorm
        # which allows for an affine transform
        if len(sh) == 1:
            return [LayerNorm(nc, elementwise_affine=True)]
        if len(sh) == 2:
            return [InstanceNorm1d(nc, affine=True)]
        if len(sh) == 3:
            return [InstanceNorm2d(nc, affine=True)]
    # No layer otherwise
    return []

def maybe_dropout2d(kind, dropout_when, dropout_p=0.5):
    if kind != dropout_when:
        return []
    return [Dropout2d(dropout_p)]

def get_simple_convnet(
        in_channels: int,
        num_classes: int,
        num_channels:int = 16,
        filter_size=3,
        with_dropout=None,
        conv_padding_mode="zeros",
        norm_kind="batch",
        capacity_multiplier=2,
    ):
    # conv_padding_mode original value was reflect
    dropout_when = with_dropout

    nc = num_channels
    nc1 = int(nc * capacity_multiplier ** 0)
    nc2 = int(nc * capacity_multiplier ** 1)
    nc3 = int(nc * capacity_multiplier ** 2)
    nc4 = int(nc * capacity_multiplier ** 3)

    net = Sequential(# input = 0
        Conv2d(in_channels, nc1, filter_size, padding="same", padding_mode=conv_padding_mode),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc1, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        Conv2d(nc1, nc2, filter_size, padding="same", padding_mode=conv_padding_mode),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc2 * capacity_multiplier ** 1, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        MaxPool2d(2),

        Conv2d(nc2, nc2, filter_size, padding="same", padding_mode=conv_padding_mode),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc2, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        Conv2d(nc2, nc3, filter_size, padding="same", padding_mode=conv_padding_mode),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc3, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        MaxPool2d(2),

        Conv2d(nc3, nc3, filter_size, padding="same", padding_mode=conv_padding_mode),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc3, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        Conv2d(nc3, nc3, filter_size, padding="same", padding_mode=conv_padding_mode),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc3, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        Conv2d(nc3, nc4, filter_size, padding="same", padding_mode=conv_padding_mode),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc4, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        MaxPool2d(2),

        Conv2d(nc4, nc4, 3),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc4, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        Flatten(),
        Linear(nc4, nc4),
        *maybe_dropout1d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc4]),
        *maybe_dropout1d("after_norm", dropout_when),
        ReLU(),
        Linear(nc4, num_classes),
        SoftmaxOutput(dim=1),
    )
    net.train(True)
    return net

def get_residual_convnet(
        in_channels: int,
        num_classes: int,
        num_channels:int = 16,
        filter_size=3,
        with_dropout=None,
        conv_padding_mode="zeros",
        norm_kind="batch",
        capacity_multiplier = 2,
    ):
    nc = num_channels
    nc1 = int(nc * capacity_multiplier ** 0)
    nc2 = int(nc * capacity_multiplier ** 1)
    nc3 = int(nc * capacity_multiplier ** 2)
    nc4 = int(nc * capacity_multiplier ** 3)

    dropout_when = with_dropout

    net = Sequential(
        Conv2d(in_channels, nc1, filter_size, padding="same", padding_mode=conv_padding_mode),
        Residual(Sequential(
            Conv2d(nc1, nc1, filter_size, padding="same", padding_mode=conv_padding_mode),
            *maybe_dropout2d("before_norm", dropout_when),
            *normalization_layer(norm_kind, [nc1, -1, -1]),
            *maybe_dropout2d("after_norm", dropout_when),
            ReLU(),
            *maybe_dropout2d("after_activation", dropout_when),
            Conv2d(nc1, nc2, filter_size, padding="same", padding_mode=conv_padding_mode),
            *maybe_dropout2d("before_norm", dropout_when),
            *normalization_layer(norm_kind, [nc2, -1, -1]),
            *maybe_dropout2d("after_norm", dropout_when),
            ReLU(),
            *maybe_dropout2d("after_activation", dropout_when),
        ), Conv2d(nc1, nc2, (1, 1))),
        MaxPool2d(2),
        Residual(Sequential(
            Conv2d(nc2, nc2, filter_size, padding="same", padding_mode=conv_padding_mode),
            *maybe_dropout2d("before_norm", dropout_when),
            *normalization_layer(norm_kind, [nc2, -1, -1]),
            *maybe_dropout2d("after_norm", dropout_when),
            ReLU(),
            *maybe_dropout2d("after_activation", dropout_when),
            Conv2d(nc2, nc3, filter_size, padding="same", padding_mode=conv_padding_mode),
            *maybe_dropout2d("before_norm", dropout_when),
            *normalization_layer(norm_kind, [nc3, -1, -1]),
            *maybe_dropout2d("after_norm", dropout_when),
            ReLU(),
            *maybe_dropout2d("after_activation", dropout_when),
        ), Conv2d(nc2, nc3, (1, 1))),
        MaxPool2d(2),
        Residual(Sequential(
            Conv2d(nc3, nc3, filter_size, padding="same", padding_mode=conv_padding_mode),
            *maybe_dropout2d("before_norm", dropout_when),
            *normalization_layer(norm_kind, [nc3, -1, -1]),
            *maybe_dropout2d("after_norm", dropout_when),
            ReLU(),
            *maybe_dropout2d("after_activation", dropout_when),
            Conv2d(nc3, nc4, filter_size, padding="same", padding_mode=conv_padding_mode),
            *maybe_dropout2d("before_norm", dropout_when),
            *normalization_layer(norm_kind, [nc4, -1, -1]),
            *maybe_dropout2d("after_norm", dropout_when),
            ReLU(),
            *maybe_dropout2d("after_activation", dropout_when),
        ), Conv2d(nc3, nc4, (1, 1))),
        MaxPool2d(2),
        Conv2d(nc4, nc4, 3),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc4, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        *maybe_dropout2d("after_activation", dropout_when),
        Flatten(),
        Linear(nc4, nc4),
        *maybe_dropout1d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc4]),
        *maybe_dropout1d("after_norm", dropout_when),
        ReLU(),
        Linear(nc4, num_classes),
        SoftmaxOutput(dim=1),
    )
    net.train(True)
    return net

def cdv(net):
    with torch.no_grad():
        # Relu gain
        net.layer.weight /= np.sqrt(2)
        # weird extra gain that pytorch uses by default
        net.layer.weight /= np.sqrt(5)
    return net

def cdvc(net):
    # with torch.no_grad():
    #     net.layer.weight /= np.sqrt(5)
    return net

def get_separable_convnet(
        in_channels: int,
        num_classes: int,
        num_channels = 16,
        filter_size=3,
        with_dropout=None,
        conv_padding_mode="zeros",
        norm_kind="batch",
        capacity_multiplier = 2,
    ):
    dropout_when = with_dropout

    nc = num_channels
    nc1 = int(nc * capacity_multiplier ** 0)
    nc2 = int(nc * capacity_multiplier ** 1)
    nc3 = int(nc * capacity_multiplier ** 2)
    nc4 = int(nc * capacity_multiplier ** 3)

    net = Sequential(
        cdv(Conv2d(in_channels, nc1, (filter_size, 1), padding="same", padding_mode=conv_padding_mode, bias=False)),
        cdvc(Conv2d(nc1, nc1, (1, filter_size), padding="same", padding_mode=conv_padding_mode)),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc2, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        MaxPool2d(2),
        cdv(Conv2d(nc2, nc2, (filter_size, 1), padding="same", padding_mode=conv_padding_mode, bias=False)),
        cdvc(Conv2d(nc2, nc3, (1, filter_size), padding="same", padding_mode=conv_padding_mode)),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc3, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        MaxPool2d(2),
        cdv(Conv2d(nc3, nc3, (filter_size, 1), padding="same", padding_mode=conv_padding_mode, bias=False)),
        cdvc(Conv2d(nc3, nc4, (1, filter_size), padding="same", padding_mode=conv_padding_mode)),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc4, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        MaxPool2d(2),
        Conv2d(nc4, nc4, 3),
        *maybe_dropout2d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc4, -1, -1]),
        *maybe_dropout2d("after_norm", dropout_when),
        ReLU(),
        Flatten(),
        Linear(nc4, nc4),
        *maybe_dropout1d("before_norm", dropout_when),
        *normalization_layer(norm_kind, [nc4]),
        *maybe_dropout1d("after_norm", dropout_when),
        ReLU(),
        Linear(nc4, num_classes),
        SoftmaxOutput(dim=1),
    )
    net.train(True)
    return net

class FixedArgsDict(nn.Module):
    """
    Similar to torch' ParameterDict, but does not convert tensors that do not require grad
    to parameters - and stores them as buffers instead. Also does not require the workarounds
    below. :)
    
    The key reason being that some fixed arguments found during tracing do not require gradients
    are often just buffers. Registering them as parameters will potentially lead to the buffers
    being interpreted as parameters - and gradients being turned on - while the operation may not
    necessarily be differentiable w.r.t. the buffer.

    See https://github.com/pytorch/pytorch/blob/fa1ccc34c4f65756bc50c3e3ab135c88b175b18c/torch/nn/modules/container.py#L682
    for the original implementation of ParameterDict - of which this is a modification.
    """

    def __init__(self, args: Any = None) -> None:
        super().__init__()
        self._keys: Dict[str, None] = {}
        if args is not None:
            self.update(args)

    def _key_to_attr(self, key: Any) -> str:
        # Change: allow integer keys too by converting to str.
        return str(key)

    def __setitem__(self, key: str, value: Any) -> None:
        # 
        self._keys[key] = None
        attr = self._key_to_attr(key)
        register_as_buffer = False
        if isinstance(value, torch.Tensor) and not isinstance(value, torch.nn.Parameter):
            if value.requires_grad:
                # Convert to parameter
                value = torch.nn.Parameter(value)
            else:
                # Edit: Do not convert to parameter, register as buffer instead.
                register_as_buffer = True
        if register_as_buffer:
            self.register_buffer(attr, value)
        else:
            setattr(self, attr, value)

    def __getitem__(self, key: str) -> Any:
        attr = self._key_to_attr(key)
        return getattr(self, attr)

    def __delitem__(self, key: str) -> None:
        del self._keys[key]
        attr = self._key_to_attr(key)
        delattr(self, attr)

    def __len__(self) -> int:
        return len(self._keys)

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __reversed__(self) -> Iterator[str]:
        return reversed(list(self._keys))

    def copy(self) -> 'FixedArgsDict':
        return FixedArgsDict(OrderedDict((k, self[k]) for k in self._keys))

    def __contains__(self, key: str) -> bool:
        return key in self._keys

    def setdefault(self, key: str, default: Optional[Any] = None) -> Any:
        if key not in self:
            self[key] = default
        return self[key]

    def clear(self) -> None:
        for k in self._keys.copy():
            del self[k]

    def pop(self, key: str) -> Any:
        v = self[key]
        del self[key]
        return v

    def popitem(self) -> Tuple[str, Any]:
        k, _ = self._keys.popitem()
        # We need the key in the _keys to be able to access/del
        self._keys[k] = None
        val = self[k]
        del self[k]
        return k, val

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        return self[key] if key in self else default

    def fromkeys(self, keys: Iterable[str], default: Optional[Any] = None) -> 'FixedArgsDict':
        return FixedArgsDict((k, default) for k in keys)

    def keys(self) -> Iterable[str]:
        return self._keys.keys()

    def items(self) -> Iterable[Tuple[str, Any]]:
        return ((k, self[k]) for k in self._keys)

    def values(self) -> Iterable[Any]:
        return (self[k] for k in self._keys)

    def update(self, parameters: Union[Mapping[str, Any], 'FixedArgsDict']) -> None:
        if isinstance(parameters, OrderedDict):
            for key, parameter in parameters.items():
                self[key] = parameter
        elif isinstance(parameters, dict):
            for key, parameter in sorted(parameters.items()):
                self[key] = parameter
        else:
            for j, p in enumerate(parameters):
                
                self[p[0]] = p[1]  # type: ignore[assignment]

    def extra_repr(self) -> str:
        child_lines = []
        for k, p in self.items():
            if isinstance(p, torch.Tensor):
                size_str = 'x'.join(str(size) for size in p.size())
                if p.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
                    device_str = f' ({p.device})'
                else:
                    device_str = ''
                parastr = '{} containing: [{} of size {}{}]'.format(
                    "Parameter" if isinstance(p, torch.nn.Parameter) else "Tensor",
                    torch.typename(p), size_str, device_str)
                child_lines.append('  (' + str(k) + '): ' + parastr)
            else:
                child_lines.append('  (' + str(k) + '): Object of type: ' + type(p).__name__)
        tmpstr = '\n'.join(child_lines)
        return tmpstr

    def __call__(self, input):
        raise RuntimeError('FixedArgsDict should not be called.')

    def __or__(self, other: 'FixedArgsDict') -> 'FixedArgsDict':
        copy = self.copy()
        copy.update(other)
        return copy

    def __ror__(self, other: 'FixedArgsDict') -> 'FixedArgsDict':
        copy = other.copy()
        copy.update(self)
        return copy

    def __ior__(self, other : 'FixedArgsDict') -> 'FixedArgsDict':
        self.update(other)
        return self

# 'Tracing' Based Conversion
# Note that we additionally have generic in-place disable
# functionality here (as in-place modification only works
# in specific cases)
class WrappedLayer(ModuleT):
    def __init__(self,
                 layer,
                 args_in_place=[],
                 kwargs_in_place=[],
                 fixed_posargs={},
                 fixed_kwargs={},
                ):
        super().__init__()
        self.layer = layer

        # Deal with fixed arguments
        self.max_posarg = max(fixed_posargs.keys(), default=-1)
        self.posargs_free = set(range(self.max_posarg)) - set(fixed_posargs.keys())
        
        # Workaround: ParameterDict only allows for string keys for now.
        # therefore: stringify them and store the integers separately.
        # `construct_base_vector` is modified too to set the right items.
        # self.fixed_posargs = torch.nn.ParameterDict({str(k): v for k, v in fixed_posargs.items()})
        # While we are now using FixedArgsDict, note that attributes are still only allowed to be strings
        # so we maintain the original list of indices.
        self.fixed_posargs = FixedArgsDict(fixed_posargs)
        self.fixed_posargs_idxs = list(fixed_posargs.keys())
        # set gradient requirements (as conversion to nn.Parameter resets this...)
        # for k, v in fixed_posargs.items():
        #     if isinstance(v, torch.Tensor):
        #         self.fixed_posargs[str(k)].requires_grad_(v.requires_grad)

        self.fixed_kwargs = FixedArgsDict(fixed_kwargs)
        # for k, v in fixed_kwargs.items():
        #     if isinstance(v, torch.Tensor):
        #         self.fixed_kwargs[k].requires_grad_(v.requires_grad)

        self.has_fixed_args = len(fixed_posargs) > 0 or len(fixed_kwargs) > 0

        # Deal with in-place modification of arguments
        self.args_in_place = args_in_place
        self.kwargs_in_place = kwargs_in_place
        self.in_place_op = len(args_in_place) > 0 or len(kwargs_in_place) > 0

    def construct_base_vector(self):
        # integer keys are positional and are kept fixed - with holes
        # filled in according to any argument
        x = [None for _ in range(self.max_posarg + 1)]
        for k in self.fixed_posargs_idxs:
            x[k] = self.fixed_posargs[str(k)]
        return x

    def forward_layer(self, x, y):
        # Fast path
        if not self.has_fixed_args:
            return self.layer(*x, **y)
        
        # Otherwise, we need to do some argument wrangling
        # fill in free positions in posargs
        x_r = self.construct_base_vector()
        for p, v in zip(self.posargs_free, x):
            x_r[p] = v
        # add remainder
        x = x_r + list(x)[len(self.posargs_free):]

        # add in kwargs
        for k, v in self.fixed_kwargs.items():
            y[k] = v

        return self.layer(*x, **y)

    def __repr__(self):
        return f"WrappedLayer-{self.layer.__repr__()}"

    def forward(self, *x, **y):
        # First of all, if the op is not in place, there is no problem
        # to simply call the method.
        if not self.in_place_op:
            return self.forward_layer(x, y)
        
        # If the op is in place we need to ensure that the original value is not being used elsewhere anymore.
        # If backprop is disabled, the call to this module needs to occur last.
        # If backprop is enabled, this needs to be the only value in existence (
        # even in the backpropagation graph)
        # If this is not the case, we need to clone - to capture any changes from
        # propagating to the original tensor.
        x = list(x)
        for x_idx in self.args_in_place:
            # nothing to clone - argument was not passed - probably does not happen
            # but was included as a bug caused an off-by-one error
            # if len(x) <= x_idx: continue
            
            # If this is the only occurence, we can reuse it. 
            if not get_has_multiple_occurences(x[x_idx]): continue

            # Otherwise, an argument's value is still necessary somewhere, while an in-place
            # op would destroy it, clone the input before passing it.
            x[x_idx] = x[x_idx].clone()
        for y_name in self.kwargs_in_place:
            if y_name not in y: continue
            y[y_name] = y[y_name].clone()

        return self.forward_layer(x, y)
    
class WrappedFn(WrappedLayer):

    def __repr__(self):
        return f"WrappedFn-{self.layer.__name__}"

def try_get_origin(t: 'TracedTensor'):
    try:
        return t.origin
    except:
        return None

def get_tensor_version(t):
    try:
        return t._version
    except:
        return None

def mklist(*args):
    return args

def is_dynamic_value(t):
    if isinstance(t, TracedTensor): return True
    if hasattr(t, "origin"): return True
    if isinstance(t, list) or isinstance(t, tuple):
        return any(is_dynamic_value(e) for e in t)
    if isinstance(t, dict):
        # Note - keys of a dictionary are always assumed fixed.
        return any(is_dynamic_value(e) for e in t.values())
    # Note - will be assumed fixed
    return False

def link_dynamic_value(gc: GraphConstructor, arg, function_id, arg_idf):
    # arg_idf may be an integer (positional argument) or a string (kwarg)
    # Skip over elements whose value is static
    if not is_dynamic_value(arg): return False

    if isinstance(arg, TracedTensor) or hasattr(arg, "origin"):
        gc.new_edge(arg.origin, function_id, arg_idf)
        return True
    elif isinstance(arg, list) or isinstance(arg, tuple):
        # Argument is a list (or tuple) - if there are any values with origin set contained within
        # construct a list with these elements determined dynamically.

        # Note that this assumes that lists in this case will be of fixed size - if this
        # is not the case the assumption that lower level modules only describe dynamic behavior
        # is violated. Rewrite the model code so that the violating portion is encapsulated within
        # a module.

        # Note - list does not modify its arguments in place, nor has kwargs
        fixed_posargs = {}
        dynamic_args = []
        for list_idx, subarg in enumerate(arg):
            if is_dynamic_value(subarg):
                dynamic_args.append((list_idx, subarg))
            else:
                fixed_posargs[list_idx] = subarg        

        # Create a list constructing node
        m_func = WrappedFn(mklist,
                           fixed_posargs=fixed_posargs)
        list_function_id = gc.new_node(m_func)

        # Link dynamic values to this node (recursively)
        for (list_idx, subarg) in dynamic_args:
            link_dynamic_value(gc, subarg, list_function_id, list_idx)

        # Link new list-creating node
        gc.new_edge(list_function_id, function_id, arg_idf)
        return True
    elif isinstance(arg, dict):
        # Similar to list, but for a dictionary. Here positional arguments are considered absent.
        fixed_kwargs = {}
        dynamic_args = []
        for dict_key, subarg in arg.items():
            if is_dynamic_value(subarg):
                dynamic_args.append((dict_key, subarg))
            else:
                # print("creating dict with fixed value???")
                fixed_kwargs[dict_key] = subarg     

        # print(f"creating a dict with fixed arguments for keys {list(fixed_kwargs.keys())}")
        # print(f"and dynamic arguments for keys {[k for (k, _v) in dynamic_args]}")

        m_func = WrappedFn(dict,
                           fixed_kwargs=fixed_kwargs)
        dict_function_id = gc.new_node(m_func)

        # Link dynamic values to this node (recursively)
        for (dict_key, subarg) in dynamic_args:
            link_dynamic_value(gc, subarg, dict_function_id, dict_key)

        # Link new dict-creating node
        gc.new_edge(dict_function_id, function_id, arg_idf)
        return True

def untrace(v):
    if isinstance(v, TracedInt):
        if not hasattr(v, "origin"): print("Untracing TracedInt without origin, this is a bug.")
        return v.i
    elif isinstance(v, TracedSize):
        if not hasattr(v, "origin"): print("Untracing TracedSize without origin, this is a bug.")
        return tuple(v)
    else:
        return v

def is_tracing():
    try:
        return TracedTensor.tracing_calls
    except:
        return False

class MetaTracedInt(type):

    _do_not_wrap_fn = {"__new__", "__init__", "__str__", "__repr__", "__hash__", "__getattribute__", "__init_subclass__", "__subclasshook__",
    "__reduce_ex__", "__getnewargs__", "__format__", "__sizeof__", "__doc__", "__class__"}
    
    def __new__(typ, name, bases, attrs, base_type, do_not_wrap={}):
        # base_type = bases[0]
        base_members = set(dir(base_type))
        wrapped = base_members - set(attrs) - MetaTracedInt._do_not_wrap_fn - do_not_wrap

        cls = type.__new__(typ, name, bases, attrs)

        for member in wrapped:
            obj = object.__getattribute__(base_type, member)
            if callable(obj):
                wrapped = cls.wrapper(obj)
                setattr(cls, member, wrapped)
        
        return cls

import functools
class TracedInt(metaclass=MetaTracedInt, base_type=int, do_not_wrap={"__setattr__"}):
    def __init__(self, i):
        self.i = i

    def __int__(self):
        return self.i

    @classmethod
    def wrapper(cls, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # TODO: trace argument & kwargs?
            # print(f"performing op {func.__name__} with args {args} ({','.join(str(type(a)) for a in args)}) and kwargs {kwargs}")
            o = func(*[untrace(a) for a in args], **kwargs)
            if isinstance(o, int):
                return TracedInt(o)
            else:
                return o
        return wrapper

def get_size(t):
    return t.size()

def get_item(tuple_like, *args):
    return tuple_like.__getitem__(*args)

class TracedSize(Tuple[int, ...]):
    # Note, this implementation is tied to TracedTensor's state to avoid duplication.
    # it may be apt to have a 'TracingState' class to deal with this info instead.

    def __getitem__(self, key: int) -> int:
        # if is_tracing():
        # Hardcoded assumption on argument types
        # Key is assumed static, 'self' - the size tensor is not.
        fixed_posargs = {1: key}

        # Create requested transformation node
        # getty_that_item <-> torch.Size.__getitem__
        m_func = WrappedFn(get_item,
                            fixed_posargs=fixed_posargs)
        id_func = TracedTensor.gc.new_node(m_func)
        assert hasattr(self, "origin"), f"{self} does not have origin set while it is a TracedSize"
        link_dynamic_value(TracedTensor.gc, self, id_func, 0)
        
        o = super().__getitem__(key)
        if isinstance(o, int):
            # print("creating TracedInt")
            o = TracedInt(o)
        else:
            # print("creating TracedSize")
            o = TracedSize(o)
        
        o.origin = id_func

        return o

    def __iter__(self) -> Iterator[int]:
        return super().__iter__()

    def numel(self) -> int:
        print("getting number of dimensions of size")
        return super().__len__()

# Trace module. - note, only works if all operations are defined
# via the lowest level modules - if this is not the case, behavior
# may differ. Test before usage.
class TracedTensor(torch.Tensor):
    # Roughly follows the documentation at
    # https://pytorch.org/docs/stable/notes/extending.html#subclassing-torch-tensor
    # for creating a 'LoggingTensor'

    @staticmethod
    def __new__(cls, x, *args, **kwargs):
        # Is __torch_function__ of this class currently being called?
        o = super().__new__(cls, x, *args, **kwargs)
        return o
    
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        
        orig_versions = [get_tensor_version(arg) for arg in args]

        args_untraced = tuple(untrace(a) for a in args)
        if kwargs is not None:
            kwargs_untraced = {k: untrace(v) for (k, v) in kwargs.items()}
        else:
            kwargs_untraced = None

        o = super().__torch_function__(func, types, args_untraced, kwargs_untraced)
        
        if cls.tracing_calls and (isinstance(o, TracedTensor) or isinstance(o, torch.Size)):
            if isinstance(o, torch.Size):
                # HACK: For some reason there is no info that tells me it is trying to
                # get a tensor's shape. Rather, it seems to be performing a __get__ operation with no additional context.
                # This substitutes the operation here.
                func = get_size
                o = TracedSize(o)

            # Figure out in-place modified arguments
            in_place_modified_args = [arg_idx for arg_idx, (arg, orig_version) in enumerate(zip(args, orig_versions)) if get_tensor_version(arg) != orig_version]
            # Figure out internal parameters likely to be a normal Tensor
            # rather than a TracedTensor. Or a list containing TracedTensors
            # See - is_dynamic_value
            fixed_posargs = {i: v for i, v in enumerate(args) if not is_dynamic_value(v)}

            if kwargs is not None:
                fixed_kwargs = {k: v for k, v in kwargs.items() if not is_dynamic_value(v)}
            else:
                fixed_kwargs = {}

            # print(f"func {func} was called with args {args}")
            m_func = WrappedFn(func,
                               args_in_place=in_place_modified_args,
                               fixed_posargs=fixed_posargs,
                               fixed_kwargs=fixed_kwargs)
            id_func = TracedTensor.gc.new_node(m_func)

            if try_get_origin(o) != None:
                # An in-place update has occurred as a tensor has been reused.
                # Mark as such.
                # print(f"detected in-place update of output of {o.origin}")
                TracedTensor.gc.set_consumed(o.origin, id_func)

            for arg_idx, arg in enumerate(args):
                link_dynamic_value(TracedTensor.gc, arg, id_func, arg_idx)
            
            for arg_kw, arg in kwargs.items():
                link_dynamic_value(TracedTensor.gc, arg, id_func, arg_kw)

            o.origin = id_func
        elif cls.tracing_calls:
            import warnings
            warnings.warn((f"Tracing while interacting with type {type(o)} which is not a Tensor.\n"
                            "Output may be incorrect. Consider dealing with this type to correct the output."))

        return o
TracedTensor.verbose = False

def trace_network(model, input_shape, verbose=False):
    X_trace = TracedTensor(torch.zeros(input_shape))
    X_trace.origin = 0
    
    gc = GraphConstructor()
    TracedTensor.gc = gc
    TracedTensor.tracing_calls = True
    hooks = []

    num_total = 0
    num_fails = 0

    def untrace_inputs_during_module_pass(m, i):
        # Do not trace module internals - already covered.
        TracedTensor.tracing_calls = False
        m.original_versions = [get_tensor_version(ii) for ii in i]


    def trace_module_pass(m, i, o):
        nonlocal num_total
        nonlocal num_fails

        from_ids = []
        # Reenable tracing once module pass is complete
        TracedTensor.tracing_calls = True
        for ii in i:
            if not isinstance(ii, TracedTensor):
                print(f"got argument of type {ii}")
            num_total += 1
            try:
                from_id = ii.origin
            except Exception as e:
                # This happens when a non-lower level layer performs operations
                # 
                if verbose:
                    print(f"note - input was not tagged for an input of {type(m)} - {e}")
                num_fails += 1
                from_id = 1
            from_ids.append(from_id)

        in_place_changed_args = [ii_idx for ii_idx, (ii, v_orig) in enumerate(zip(i, m.original_versions)) if get_tensor_version(ii) != v_orig]
        # kwargs?

        # Note: current assumption is that modules encapsulate their fixed arguments
        # (so we do not need to capture them). If this turns out to be false, use
        # determine and use the `fixed_args` argument of WrappedLayer.
        to_id = gc.new_node(WrappedLayer(m, args_in_place=in_place_changed_args))
        for s, ii in enumerate(from_ids):
            gc.new_edge(ii, to_id, s)
            if verbose:
                print(f"[{to_id}] socket {s} originates from {ii}")

        if isinstance(o, TracedTensor) and try_get_origin(o) != None:
            # An in-place update has occurred as a tensor has been reused.
            # Mark as such.
            # print(f"detected in-place update of output of {o.origin}")
            gc.set_consumed(o.origin, to_id)

        if not isinstance(o, TracedTensor):
            import warnings
            warnings.warn((f"Tracing while interacting with type {type(o)} which is not a TracedTensor.\n"
                            "Output may be incorrect. Consider dealing with this type to correct the output."))


        new_o = TracedTensor(o)
        new_o.origin = to_id

        del m.original_versions

        return new_o
    
    # For each lowest level
    for (_n, m) in model.named_modules():
        # Count # of submodules (note: 1 is self.)
        # We don't need to know whether there are more than 2.
        c = 0
        for _sm in m.modules():
            c += 1
            if c == 2: break
        
        # Skip higher level nodes - do not need to be added to graph,
        # probably. If they do perform some operations that are unlisted
        # we might be in a place of hurt.
        if c == 2: continue
        
        hook_x = m.register_forward_pre_hook(untrace_inputs_during_module_pass)
        hooks.append(hook_x)
        hook = m.register_forward_hook(trace_module_pass)
        hooks.append(hook)

        # print(f"{n}: {type(m)} / {num_submodules}")
    
    try:
        out = model(X_trace)

        if verbose:
            print(f"linking up failed in {num_fails} / {num_total} links")
            print("linking up output")
        
        # Link up output
        link_dynamic_value(gc, out, 1, 0)

    finally:
        for hook in hooks:
            hook.remove()
        TracedTensor.gc = None

    return gc