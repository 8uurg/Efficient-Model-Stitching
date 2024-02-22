import os
import time
from abc import ABC, abstractmethod
from itertools import count
from pathlib import Path
from typing import Optional
import math

import numpy as np
import polars as pl
import torch
import torch.nn as nn
import torchvision.io
import torchvision.transforms as tvt
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader, Dataset

from . import cx as cx
from . import datasets as dt
from . import layers as ly

# from PIL import Image
# def loadimage_pil(path):
#     return Image.open(path).convert("RGB")

# import cv2
# def loadimage_cv(path):
#     img = cv2.imread(path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     return img

def loadimage_torchio(path):
    return torchvision.io.read_image(path, torchvision.io.ImageReadMode.RGB)

class BaseTracker(ABC):
    """
    Base Class for a tracker that monitors a particular bit of state during training,
    evaluation, or another long running process.
    """

    @abstractmethod
    def start_eval():
        raise NotImplementedError()

    @abstractmethod
    def complete_eval(state):
        raise NotImplementedError()

class NeuralNetIndividual:
    def __init__(self, net: ly.NeuralNetGraph, fitness=None, is_trained=False, parent_idxs=[], meta={}):
        self.net = net
        self.fitness = fitness
        self.is_trained = is_trained
        self.parent_idxs = parent_idxs
        self.meta = meta

    def as_trained(self, net, is_trained):
        if net is None: net = self.net
        return NeuralNetIndividual(
            net=net,
            fitness=self.fitness,
            is_trained=is_trained,
            parent_idxs=self.parent_idxs,
            meta=self.meta
        )

    def as_evaluated(self, fitness):
        return NeuralNetIndividual(
            net=self.net,
            fitness=fitness,
            is_trained=self.is_trained,
            parent_idxs=self.parent_idxs,
            meta=self.meta
        )

    def log_update(self, idx, num_evaluations_started, num_evaluations_completed, update_counter, f, d, initial_time):
        # dump net to file.
        fn = f"net_{update_counter}.th"
        dp = d / fn
        td = time.time() - initial_time
        torch.save(self.net, dp)
        f.write((
            '{'
                f'"n": {update_counter}, '
                f'"i": {idx}, '
                f'"es": {num_evaluations_started}, '
                f'"ec": {num_evaluations_completed}, '
                f'"td": {td}, '
                f'"f": {self.fitness if self.fitness is not None else "null"}, '
                f'"tr": {"true" if self.is_trained else "false"},'
                f'"parent_idxs": {self.parent_idxs}, "net": "{fn}"'
            '}\n'))


class NASProblem(ABC):
    """
    A NAS Problem

    Note that this class is potentially transferred between machines, reliance
    on the presence of data in the filesystem is dangerous. It is recommended
    that state that depends on the filesystem is not pickled alongside the remainder
    and omitted using __getstate__ and __setstate__.
    """

    @abstractmethod
    def sample_architecture(self, rng: np.random.Generator) -> NeuralNetIndividual:
        """Sample an architecture that is suitable for the given problem."""
        raise NotImplementedError
    
    def set_tracker(self, tracker):
        self.tracker = tracker

    @abstractmethod
    def evaluate_network(
        self,
        dev: torch.device,
        net: NeuralNetIndividual,
        dataset="validation",
        objective="accuracy",
    ) -> float:
        raise NotImplementedError

    @abstractmethod
    def train_network(
        self, dev: torch.device, net: NeuralNetIndividual, lr=1e-3, weight_decay=1e-2, seed=None, return_to_cpu=True, num_epochs=5, num_batches=None,
    ) -> ly.NeuralNetGraph:
        raise NotImplementedError

    @abstractmethod
    def get_dataset_train(self):
        raise NotImplementedError

    @abstractmethod
    def get_dataset_validation(self):
        raise NotImplementedError

    @abstractmethod
    def get_dataset_test(self):
        raise NotImplementedError

def check_output_grad(o):
    """
    Checks whether the current output has any gradients in
    any of its parts
    """
    if isinstance(o, torch.Tensor):
        return o.requires_grad
    elif isinstance(o, list):
        return any(check_output_grad(e) for e in o)
    elif isinstance(o, dict):
        return any(check_output_grad(e) for e in o.values())
    # unrecognized.
    return False

class ClassificationNASProblem(NASProblem):
    def __init__(self, verbose=False, loss_fn=None):
        self.networks = None
        self.tracker = None
        self.verbose = verbose
        self.dataset = None
        self.loss_fn = nn.CrossEntropyLoss() if loss_fn is None else loss_fn

    def load_problem_dataset(self) -> tuple: # type: ignore
        pass

    def load_dataset(self) -> tuple:
        if self.dataset is None:
            self.dataset = self.load_problem_dataset()
        return self.dataset
    
    def compute_batch_loss(self, a, b):
        # note if the batch size of b is a multiple, we are training multiple outputs
        # stacked together (i.e. an ensemble) tile the output
        if a.shape[0] != b.shape[0]:
            sh = [a.shape[0] // b.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
            b = b.repeat(*sh)
        return self.loss_fn(a, b)

    def compute_batch_correct(self, a, b):
        return (a.argmax(dim=1) == b).sum().detach().cpu().item()

    def train_network(
        self,
        dev: torch.device,
        neti: NeuralNetIndividual,
        lr=1e-3,
        weight_decay=1e-2,
        seed=None,
        return_to_cpu=True,
        num_epochs=None,
        num_batches=None,
        termination_criterion=None,
        lr_scheduler=None,
        per_batch_lr_schedule=False,
        optimizer="nm",
        batch_size=128,
        use_amp=True,
        gradscaler=True,
        shuffle=True,
        raise_on_nan_loss=True,
        epoch_start_idx=None,
        profiler=None,
        on_batch_end=None,
        minout_nan=False,
        stop_on_no_gradients=True,
        summarywriter=None,
        num_dl_workers=None,
        verbose=False,
    ):
        # note - transferring a network this way may be somewhat inefficient.
        # might be worth making this a bit more complicated

        # Load the dataset on this node - not sure how expensive this is exactly
        # but we have to do this here, there is no guarantee that this node already
        # has access to this dataset.
        # [ ] - use ray data instead?
        d_train, d_validation, d_test = self.load_dataset()

        if num_dl_workers is None:
            num_dl_workers = int(os.environ.get("RECOMB_NUM_DATALOADER_WORKERS", "0"))

        if gradscaler and use_amp:
            gradscaler = torch.cuda.amp.GradScaler() # type: ignore
        else:
            gradscaler = None

        # seed the rng
        if seed is not None:
            torch.manual_seed(seed)

        if isinstance(neti, NeuralNetIndividual):
            # Extract network component
            net = neti.net.to(dev)
        else:
            # Assume module
            net = neti.to(dev)

        # train
        dl = DataLoader(d_train, batch_size=batch_size, num_workers=num_dl_workers, shuffle=shuffle)
        # optim = torch.optim.SGD(net.parameters(), lr=lr, nesterov=False, momentum=0.9, weight_decay=weight_decay)
        # optim = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
        if optimizer == "nm":
            optim = torch.optim.SGD(net.parameters(), lr=lr, nesterov=True, momentum=0.9, weight_decay=weight_decay)
        elif optimizer == "na":
            optim = torch.optim.NAdam(net.parameters(), lr=lr, weight_decay=weight_decay)
        elif isinstance(optimizer, torch.optim.Optimizer):
            # use the provided optimizer if this is provided instead
            # note that we do assume that the optimizer is already configured to use the correct parameters
            # if this is not the case, the network will not be trained :)
            optim = optimizer
            # optimizer's state in potentially on the wrong device, this is a workaround
            # that apparently works due to the way pytorch handles state dicts.
            # https://github.com/pytorch/pytorch/issues/8741#issuecomment-496907204
            optim.load_state_dict(optim.state_dict())

        if num_epochs is None and num_batches is None:
            # if no arguments, default to 5 epochs.
            num_epochs = 5
        elif num_epochs is None:
            # Compute the number of required epochs to pass this many batches
            num_batches_per_epoch = len(dl)
            num_epochs = np.ceil(num_batches / num_batches_per_epoch) # type: ignore
        # Otherwise num_epochs is set
        # - If num_batches is set too, strictest limit stops.
        # - If num_batches is not set - no batch limit is set up & all epochs are guaranteed to complete
        #   fully (unless errors).

        lr_schedule = None
        if lr_scheduler is not None:
            lr_schedule = lr_scheduler(optim, num_epochs * len(dl) if per_batch_lr_schedule else num_epochs)

        batch_iter = count()
        batch_idx = 0

        def enable_logit_output(l):
            l.logit_mode = True
        def disable_logit_output(l):
            l.logit_mode = False

        had_nan_loss = False

        net.train_restore()
        net.had_no_gradients = False
        skip_amp = False
        no_gradients = False
        if profiler: profiler.start()
        for epoch in range(num_epochs):
            if profiler: profiler.step()
            # Sometimes training is split into multiple calls to train_network.
            # epoch resets to 0 each time, but we want to keep track of the actual
            # #epochs that have passed (e.g. for logging)
            actual_epoch = epoch + (0 if epoch_start_idx is None else epoch_start_idx)

            if termination_criterion is not None and termination_criterion({
                "epoch": epoch,
                "actual_epoch": actual_epoch,
                "neti": neti,
            }): break
            if had_nan_loss: break

            epoch_loss = 0.0
            num_samples = 0
            for batch_idx, (batch_X, batch_true_y) in zip(batch_iter, dl):
                if num_batches is not None and batch_idx >= num_batches: break
                batch_X, batch_true_y = batch_X.to(dev), batch_true_y.to(dev)
                optim.zero_grad(set_to_none=True)
                net.configure_output(enable_logit_output)

                with torch.autocast(dev.type, enabled=use_amp and not skip_amp): # type: ignore
                    batch_pred_y = net(batch_X)
                    if minout_nan:
                        # Note: detach - because this is a hacky way to replace a value (and keep things working)
                        # alternatively, we could drop the sample.
                        replv = torch.min(torch.nan_to_num(batch_pred_y, torch.inf)).detach()
                        batch_pred_y = torch.where(torch.isnan(batch_pred_y), replv, batch_pred_y)
                    loss = self.compute_batch_loss(batch_pred_y, batch_true_y)
                    if not loss.requires_grad:
                        if check_output_grad(batch_pred_y):
                            print("No loss gradient, however, the output does contain elements that require grad")
                            print("i.e. there is an unused output that has been modified.") 
                        no_gradients = True
                        if stop_on_no_gradients:
                            break
                    else:
                        no_gradients = False

                    if verbose: print(f"batch loss: {loss}")
                    if torch.isnan(loss).all() and not had_nan_loss:
                        had_nan_loss = True
                        print(f"Individual {neti} with metadata {neti.meta} trained with nan loss")
                    
                    mean_batch_loss = loss.detach().item()
                    batch_loss = (mean_batch_loss * batch_X.shape[0])

                    #     print("loss is nan")
                    #     np.savetxt("nangrad_predictions_logits.txt", batch_pred_y.detach().cpu().numpy())
                    #     np.savetxt("nangrad_predictions_probs.txt", batch_pred_y.detach().softmax(dim=1).detach().cpu().numpy())
                    #     np.savetxt("nangrad_targets.txt", batch_true_y.detach().cpu().numpy())

                skip_amp = False
                
                if not had_nan_loss and not no_gradients:
                    if gradscaler is not None:
                        # if torch.isnan(loss):
                        # skip_amp = True
                        gradscaler.scale(loss).backward() # type: ignore
                        gradscaler.step(optim)
                        gradscaler.update()
                    else:
                        loss.backward()
                        optim.step()

                net.configure_output(disable_logit_output)

                if summarywriter:
                    summarywriter.add_scalar("Batch Loss", mean_batch_loss, batch_idx)
                
                if not had_nan_loss:
                    epoch_loss += batch_loss
                    num_samples += batch_X.shape[0]
                
                # print(f"batch_loss: {batch_loss}")
                if lr_schedule is not None and per_batch_lr_schedule:
                    lr_schedule.step()
                if on_batch_end is not None:
                    on_batch_end(batch_loss)

                # stop early on nan - if we are not ignoring it.
                if minout_nan:
                    # we replacing nan losses, just continue & ignore
                    had_nan_loss = False
                if had_nan_loss: break
            
            if no_gradients and stop_on_no_gradients:
                print(f"stopping as gradients are absent")
                net.had_no_gradients = True
                break
            
            # This shouldn't happen... but. just in case
            if num_samples != 0:
                epoch_loss = epoch_loss / num_samples
            else:
                print("number of samples was 0 for whatever reason")

            if summarywriter:
                summarywriter.add_scalar("Epoch Loss", epoch_loss, batch_idx)
            if self.verbose: print(f"epoch train loss @ {actual_epoch}: {epoch_loss}")

            if num_batches is not None and batch_idx >= num_batches: break
            if lr_scheduler is not None and not per_batch_lr_schedule:
                lr_scheduler.step()

        if profiler: profiler.stop()
        #
        if return_to_cpu:
            net = net.cpu()

        if raise_on_nan_loss and had_nan_loss:
            raise ValueError("Network trained with nan loss")
        return net

    def evaluate_network(
        self,
        dev: torch.device,
        neti: NeuralNetIndividual,
        dataset="validation",
        objective="accuracy",
        batch_size=128,
        return_to_cpu=True,
        profiler=None,
        sample_limit=None,
        num_dl_workers=None,
        custom_metric_batch=None,
    ):
        # return_logits_and_true was added to return raw predictions, to compute other metrics from without
        # having to load & run the neural network over and over again...
        # increment evaluation start counter
        if self.tracker is not None:
            # wait?
            vf = self.tracker.start_eval()

        # Load the dataset on this node - not sure how expensive this is exactly
        # but we have to do this here, there is no guarantee that this node already
        # has access to this dataset.
        # [ ] - use ray data instead?
        d_train, d_validation, d_test = self.load_dataset()

        if isinstance(neti, NeuralNetIndividual):
            # Extract network component
            net = neti.net.to(dev)
        else:
            # Assume module
            net = neti.to(dev)

        # Evaluation can be done on varying datasets
        if dataset == "train":
            dataset = d_train
        elif dataset == "validation":
            dataset = d_validation
        elif dataset == "test":
            dataset = d_test
        else:
            raise Exception()
        
        def enable_logit_output(l):
            l.logit_mode = True
        def disable_logit_output(l):
            l.logit_mode = False
        if num_dl_workers is None:
            num_dl_workers = int(os.environ.get("RECOMB_NUM_DATALOADER_WORKERS", "0"))
        dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_dl_workers)
        try:
            net.configure_output(enable_logit_output)
        except:
            pass

        try:
            net.store_state_eval()
        except:
            pass
        dataset_loss = 0.0
        dataset_correct = 0
        num_samples = 0
        custom_metric_state = {}

        if sample_limit is not None:
            batch_limit = int(math.ceil(sample_limit / batch_size))
            batch_idx_itr = range(batch_limit)
        else:
            batch_idx_itr = count()

        # Do not compute gradients when evaluating a network
        with torch.no_grad():
            if profiler: profiler.start()
            for batch_idx, (batch_X, batch_true_y) in zip(batch_idx_itr, dl):
                if profiler: profiler.step()
                batch_X, batch_true_y = batch_X.to(dev), batch_true_y.to(dev)
                batch_pred_y = net(batch_X)
                loss = self.compute_batch_loss(batch_pred_y, batch_true_y)

                if custom_metric_batch is not None:
                    custom_metric_batch(custom_metric_state, batch_pred_y.detach().cpu().numpy(), batch_true_y.cpu().numpy())

                dataset_loss += (loss * batch_X.shape[0]).detach().cpu().item()
                dataset_correct += self.compute_batch_correct(batch_pred_y, batch_true_y)
                num_samples += batch_X.shape[0]
            if profiler: profiler.stop()

        dataset_accuracy = dataset_correct / num_samples
        dataset_loss = dataset_loss / num_samples
        try:
            net.train_restore()
            net.configure_output(disable_logit_output)
        except:
            pass
        if return_to_cpu:
            net.cpu()
        
        #
        if objective == "accuracy":
            if self.verbose: print(f"evaluated network: {dataset_accuracy} (loss: {dataset_loss})")
            if self.tracker is not None:
                # wait?
                vf = self.tracker.complete_eval(dataset_accuracy)
            if custom_metric_batch is not None:
                return dataset_accuracy, custom_metric_state
            return dataset_accuracy
        elif objective == "loss":
            if self.verbose: print(f"evaluated network: {dataset_loss} (accuracy: {dataset_accuracy})")
            if self.tracker is not None:
                # wait?
                vf = self.tracker.complete_eval(dataset_loss)
            if custom_metric_batch is not None:
                return dataset_loss, custom_metric_state
            return dataset_loss
        elif objective == "both":
            if self.verbose: print(f"evaluated network: (accuracy = {dataset_accuracy}, loss = {dataset_loss})")
            if self.tracker is not None:
                # wait?
                vf = self.tracker.complete_eval((dataset_accuracy, dataset_loss))
            if custom_metric_batch is not None:
                return (dataset_accuracy, dataset_loss), custom_metric_state
            return dataset_accuracy, dataset_loss

    def compute_feature_maps_and_points_for_network(self, net: NeuralNetIndividual):
        d_train, d_validation, d_test = self.load_dataset()
        dl = DataLoader(d_train, batch_size=128)
        X_in_many, y_in_many = next(iter(dl))

        cx.forward_get_all_feature_maps(net.net, X_in_many, return_points=True)

    def sample_architecture_from_file(self, rng: np.random.Generator):
        pass

    def sample_random_architecture(self, rng: np.random.Generator):
        pass

    def sample_architecture(self, rng: np.random.Generator):
        """
        Sample a architecture that is suitable for the given problem.

        i.e. the architecture is valid, it accepts the input data in the format for this problem, and provides output
        in the right format for this problem.
        """

        return self.sample_architecture_from_file(rng)

    def get_dataset_train(self):
        d_train, d_validation, d_test = self.load_dataset()
        return d_train

    def get_dataset_validation(self):
        d_train, d_validation, d_test = self.load_dataset()
        return d_validation

    def get_dataset_test(self):
        d_train, d_validation, d_test = self.load_dataset()
        return d_test

class FashionMNISTImageClassificationNASProblem(ClassificationNASProblem):
    def __init__(self, verbose=False, pretrained_nets_dir="./trained-nets-fmnist", net_sampling_config={}):
        super().__init__(verbose=verbose)
        self.pretrained_nets_dir = pretrained_nets_dir
        self.net_sampling_config = net_sampling_config
        if isinstance(pretrained_nets_dir, str):
            self.pretrained_nets_dir = Path(pretrained_nets_dir)
        if isinstance(self.pretrained_nets_dir, Path) and not self.pretrained_nets_dir.exists():
            if verbose: print(f"pretrained networks not found - training from scratch")
            self.pretrained_nets_dir = None

    def load_problem_dataset(self):
        return dt.load_FashionMNIST()
    
    def sample_architecture_from_file(self, rng: np.random.Generator):
        networks = list(Path(self.pretrained_nets_dir).glob("*.th")) # type: ignore

        net = torch.load(rng.choice(networks)) # type: ignore

        # These networks should be trained, but not yet evaluated.
        return NeuralNetIndividual(net, None, is_trained=True)

    def sample_random_architecture(self, rng: np.random.Generator):
        norm_kinds = ["batch", "instance"]
        norm_kinds = self.net_sampling_config.get("norm_kinds", norm_kinds)

        filter_sizes = [3, 5, 7]
        filter_sizes = self.net_sampling_config.get("filter_sizes", filter_sizes)

        channel_widths = [8, 16, 24, 32]
        channel_widths = self.net_sampling_config.get("channel_widths", channel_widths)

        with_dropout_choices = [None, "before_norm", "after_norm"]
        with_dropout_choices = self.net_sampling_config.get("with_dropout_choices", with_dropout_choices)

        kind = rng.integers(3)
        num_channels = rng.choice(channel_widths)
        with_dropout = rng.choice(with_dropout_choices)
        filter_size = rng.choice(filter_sizes)
        norm_kind = rng.choice(norm_kinds)

        if kind == 0:
            net = ly.get_simple_convnet(1, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 1:
            net = ly.get_separable_convnet(1, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 2:
            net = ly.get_residual_convnet(1, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        else:
            raise ValueError()

        # These networks are unevaluated (None) and untrained (False)
        r = NeuralNetIndividual(net, None, is_trained=False)
        r.meta = dict(
            kind = kind,
            num_channels = num_channels,
            with_dropout = with_dropout,
            filter_size = filter_size,
            norm_kind = norm_kind
        )
        return r

    def sample_architecture(self, rng: np.random.Generator):
        """
        Sample a architecture that is suitable for the given problem.

        i.e. the architecture is valid, it accepts the input data in the format for this problem, and provides output
        in the right format for this problem.
        """

        return self.sample_architecture_from_file(rng)

class CIFAR10ClassificationNASProblem(ClassificationNASProblem):
    def __init__(self, verbose=False, pretrained_nets_dir:Optional[str]="./trained-nets-cifar10", net_sampling_config={}, augment=False):
        super().__init__(verbose=verbose)
        self.pretrained_nets_dir = pretrained_nets_dir
        self.net_sampling_config = net_sampling_config
        if isinstance(pretrained_nets_dir, str):
            self.pretrained_nets_dir = Path(pretrained_nets_dir)
        if isinstance(self.pretrained_nets_dir, Path) and not self.pretrained_nets_dir.exists():
            if verbose: print(f"pretrained networks not found - training from scratch")
            self.pretrained_nets_dir = None
        self.augment = augment

    def load_problem_dataset(self):
        return dt.load_CIFAR10(augment=self.augment)
    
    def sample_architecture_from_file(self, rng: np.random.Generator):
        networks = list(Path(self.pretrained_nets_dir).glob("*.th")) # type: ignore

        net = torch.load(rng.choice(networks)) # type: ignore

        # These networks should be trained, but not yet evaluated.
        return NeuralNetIndividual(net, None, is_trained=True)

    def sample_random_architecture(self, rng: np.random.Generator):
        norm_kinds = ["batch", "instance"]
        norm_kinds = self.net_sampling_config.get("norm_kinds", norm_kinds)

        filter_sizes = [3, 5, 7]
        filter_sizes = self.net_sampling_config.get("filter_sizes", filter_sizes)

        # channel_widths = [8, 16, 24, 32, 64, 128, 256, 512]
        channel_widths = [32, 64, 128, 256, 512]
        channel_widths = self.net_sampling_config.get("channel_widths", channel_widths)

        with_dropout_choices = [None, "before_norm", "after_norm"]
        with_dropout_choices = self.net_sampling_config.get("with_dropout_choices", with_dropout_choices)    
        
        kind = rng.integers(3)
        num_channels = rng.choice(channel_widths)
        with_dropout = rng.choice(with_dropout_choices)
        filter_size = rng.choice(filter_sizes)
        norm_kind = rng.choice(norm_kinds)
        kind_name = "unknown"

        if kind == 0:
            kind_name = "convnet"
            net = ly.get_simple_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 1:
            kind_name = "seperable convnet"
            net = ly.get_separable_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 2:
            kind_name = "residual convnet"
            net = ly.get_residual_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        else:
            raise ValueError()

        # These networks are unevaluated (None) and untrained (False)
        r = NeuralNetIndividual(net, None, is_trained=False) 
        r.meta = dict(
            kind = kind,
            kind_name = kind_name,
            num_channels = num_channels,
            with_dropout = with_dropout,
            filter_size = filter_size,
            norm_kind = norm_kind
        )
        return r

    def sample_architecture(self, rng: np.random.Generator):
        """
        Sample a architecture that is suitable for the given problem.

        i.e. the architecture is valid, it accepts the input data in the format for this problem, and provides output
        in the right format for this problem.
        """

        return self.sample_architecture_from_file(rng)

class CIFAR100ClassificationNASProblem(ClassificationNASProblem):
    def __init__(self, verbose=False, pretrained_nets_dir="./trained-nets-cifar100", net_sampling_config={}, augment=False):
        super().__init__(verbose=verbose)
        self.pretrained_nets_dir = pretrained_nets_dir
        self.net_sampling_config=net_sampling_config
        if isinstance(pretrained_nets_dir, str):
            self.pretrained_nets_dir = Path(pretrained_nets_dir)
        if isinstance(self.pretrained_nets_dir, Path) and not self.pretrained_nets_dir.exists():
            if verbose: print(f"pretrained networks not found - training from scratch")
            self.pretrained_nets_dir = None
        self.augment = augment

    def load_problem_dataset(self):
        return dt.load_CIFAR100(augment=self.augment)
    
    def sample_architecture_from_file(self, rng: np.random.Generator):
        networks = list(Path(self.pretrained_nets_dir).glob("*.th")) # type: ignore

        net = torch.load(rng.choice(networks)) # type: ignore

        # These networks should be trained, but not yet evaluated.
        return NeuralNetIndividual(net, None, is_trained=True)

    def sample_random_architecture(self, rng: np.random.Generator):
        norm_kinds = ["batch", "instance"]
        norm_kinds = self.net_sampling_config.get("norm_kinds", norm_kinds)

        filter_sizes = [3, 5, 7]
        filter_sizes = self.net_sampling_config.get("filter_sizes", filter_sizes)

        # channel_widths = [8, 16, 24, 32, 64, 128, 256, 512]
        channel_widths = [32, 64, 128, 256, 512]
        channel_widths = self.net_sampling_config.get("channel_widths", channel_widths)
        
        with_dropout_choices = [None, "before_norm", "after_norm"]
        with_dropout_choices = self.net_sampling_config.get("with_dropout_choices", with_dropout_choices)
        
        kind = rng.integers(3)
        num_channels = rng.choice(channel_widths)
        with_dropout = rng.choice(with_dropout_choices)
        filter_size = rng.choice(filter_sizes)
        norm_kind = rng.choice(norm_kinds)
        if kind == 0:
            kind_name = "convnet"
            net = ly.get_simple_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 1:
            kind_name = "seperable convnet"
            net = ly.get_separable_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 2:
            kind_name = "residual convnet"
            net = ly.get_residual_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        else:
            raise ValueError()

        # These networks are unevaluated (None) and untrained (False)
        r = NeuralNetIndividual(net, None, is_trained=False)
        r.meta = dict(
            kind = kind,
            kind_name = kind_name,
            num_channels = num_channels,
            with_dropout = with_dropout,
            filter_size = filter_size,
            norm_kind = norm_kind
        )
        return r

    def sample_architecture(self, rng: np.random.Generator):
        """
        Sample a architecture that is suitable for the given problem.

        i.e. the architecture is valid, it accepts the input data in the format for this problem, and provides output
        in the right format for this problem.
        """

        return self.sample_architecture_from_file(rng)

class CIFAR10SplitTaskClassificationNASProblem(CIFAR10ClassificationNASProblem):
    def __init__(self, verbose=False, pretrained_nets_dir=None, subtask=0, num_split=3, net_sampling_config={}, augment=False):
        pretrained_nets_dir = "./trained-nets-cifar10split" if pretrained_nets_dir is None else pretrained_nets_dir
        super().__init__(verbose=verbose, pretrained_nets_dir=pretrained_nets_dir, net_sampling_config=net_sampling_config, augment=augment)
        self.num_split = num_split
        self.subtask = subtask

    def load_problem_dataset(self):
        d_train, d_val, d_test = dt.load_CIFAR10(augment=self.augment)
        rng_split2 = torch.manual_seed(43)
        d_trains = dt.random_split(d_train, [1/self.num_split for _ in range(self.num_split)], generator=rng_split2)
        d_vals = dt.random_split(d_val, [1/self.num_split for _ in range(self.num_split)], generator=rng_split2)
        return d_trains[self.subtask], d_vals[self.subtask], d_test
    
    def sample_architecture_from_file(self, rng: np.random.Generator):
        # without annotation
        networks = list(Path(self.pretrained_nets_dir).glob(f"*-t{self.subtask}.th")) # type: ignore
        # with annotation
        networks +=  list(Path(self.pretrained_nets_dir).glob(f"*-t{self.subtask}-*.th")) # type: ignore
        net = torch.load(rng.choice(networks)) # type: ignore

        # These networks should be trained, but not yet evaluated.
        return NeuralNetIndividual(net, None, is_trained=True)

    def sample_random_architecture(self, rng: np.random.Generator):
        norm_kinds = ["batch", "instance"]
        norm_kinds = self.net_sampling_config.get("norm_kinds", norm_kinds)

        filter_sizes = [3, 5, 7]
        filter_sizes = self.net_sampling_config.get("filter_sizes", filter_sizes)

        # channel_widths = [8, 16, 24, 32, 64, 128, 256, 512]
        channel_widths = [32, 64, 128, 256, 512]
        channel_widths = self.net_sampling_config.get("channel_widths", channel_widths)

        with_dropout_choices = [None, "before_norm", "after_norm"]
        with_dropout_choices = self.net_sampling_config.get("with_dropout_choices", with_dropout_choices)    
        
        kind = rng.integers(3)
        num_channels = rng.choice(channel_widths)
        with_dropout = rng.choice(with_dropout_choices)
        filter_size = rng.choice(filter_sizes)
        norm_kind = rng.choice(norm_kinds)

        if kind == 0:
            kind_name = "convnet"
            net = ly.get_simple_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 1:
            kind_name = "seperable convnet"
            net = ly.get_separable_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 2:
            kind_name = "residual convnet"
            net = ly.get_residual_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        else:
            raise ValueError()

        # These networks are unevaluated (None) and untrained (False)
        r = NeuralNetIndividual(net, None, is_trained=False)
        r.meta = dict(
            kind = kind,
            kind_name = kind_name,
            num_channels = num_channels,
            with_dropout = with_dropout,
            filter_size = filter_size,
            norm_kind = norm_kind,
            subtask=self.subtask,
            num_subtasks=self.num_split,
        )
        return r

    def sample_architecture(self, rng: np.random.Generator):
        """
        Sample a architecture that is suitable for the given problem.

        i.e. the architecture is valid, it accepts the input data in the format for this problem, and provides output
        in the right format for this problem.
        """

        return self.sample_architecture_from_file(rng)


def sample_cumulative_transform(seed, b1, b2, minimal_value, maximal_value, num_channels):
    rng = np.random.default_rng(seed=seed)
    # use a beta distribution to allow for a stronger posterization effect in some cases.
    ctlut = rng.beta(b1, b2, size=(maximal_value - minimal_value + 1 + 2, num_channels))
    # compute cumulative sum and normalize to the range [minimal_value, maximal_value]
    # round them into integers.
    # We include extra entries at the start and end, and remove these entries to ensure
    # that the minimum and maximum values can deviate slightly.
    ctlut = np.cumsum(ctlut, axis=0)
    ctlut = ctlut / ctlut[[-1], :]
    ctlut = np.round(ctlut * (maximal_value - minimal_value) + minimal_value).astype(int)
    ctlut = ctlut[1:-1, :]
    return ctlut

def transform_image_numpy(img, ctlut):
    num_channels = ctlut.shape[-1]
    return np.stack([ctlut[img[:, i, ...], i] for i in range(num_channels)], axis=0).swapaxes(0, 1)

def transform_image_torch(img, ctlut):
    # ensure img is a batched tensor
    # Torch's dataloading infrastructure does not actually use
    # the fact that you can pass a batched tensor to the transform
    # though.
    is_unbatched = len(img.shape) == 3
    if is_unbatched:
        img = img.unsqueeze(0)

    num_channels = ctlut.shape[-1]

    # ensure ctlut is on the same device as img
    ctlut = ctlut.to(img.device)
    def map_channel(image_channel, channel):
        # annoyingly enough, we cannot use bytes directly because classically
        # bytes were used in place of bools. This behavior is now deprecated
        # but this still disallows us from indexing using the smaller type.
        image_channel_lutted = ctlut[image_channel.ravel().to(torch.int), channel].reshape(image_channel.shape)
        assert image_channel_lutted.shape == image_channel.shape
        return image_channel_lutted

    # apply the lut - lut is defined for each channel separately.
    # as such we apply each of the per-channel luts on each channel separately
    # and then stack the results into a [batch, channel, ...] tensor. 
    mapped_channels = [map_channel(img[:, i, ...], i) for i in range(num_channels)]
    r = torch.stack(mapped_channels, dim=1)
    assert r.shape == img.shape

    if is_unbatched:
        r = r.squeeze(0)
    return r

class CumulativeTransform(torch.nn.Module):
    
    def __init__(self, ctlut, untransform=True):
        super().__init__()
        self.ctlut = ctlut
        self.untransform = untransform

    def forward(self, x):
        with torch.no_grad():
            if self.untransform:
                x = (x * 255).round().clamp(0, 255).byte()

            r = transform_image_torch(x, self.ctlut)

            if self.untransform:
                r = r.to(torch.float32) / 255.0
        
        return r

class CIFAR10SplitTaskDifferentAugmentationClassificationNASProblem(CIFAR10SplitTaskClassificationNASProblem):

    def __init__(self, verbose=False, pretrained_nets_dir=None, subtask=0, num_split=3, net_sampling_config={}, augment=False, b1=0.05, b2=1.0):
        super().__init__(verbose=verbose, pretrained_nets_dir=pretrained_nets_dir, subtask=subtask, num_split=num_split, net_sampling_config=net_sampling_config, augment=augment)
        # note - problem specific arguments, 'parse' just to be sure
        # in case they were provided from the command line (i.e., as strings).
        self.b1 = float(b1)
        self.b2 = float(b2)

    def get_task_augmentation(self):
        ctlut = sample_cumulative_transform(self.subtask + 42, self.b1, self.b2, 0, 255, 3)
        return [CumulativeTransform(torch.tensor(ctlut))]
        
        # transform_id = self.subtask % 8
        # if transform_id == 0:
        #     # task 0 is as-is
        #     return None
        # elif transform_id == 1:
        #     # task 1 is autoconstrasted
        #     return [tvt.RandomAutocontrast(1.0)]
        # elif transform_id == 2:
        #     # task 2 is sharpened
        #     return [tvt.RandomAdjustSharpness(1.2 ,1.0)]
        # elif transform_id == 3:
        #     # task 3 is jittered
        #     return [tvt.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02)]
        # elif transform_id == 4:
        #     # task 4 is blurred slightly 
        #     return [tvt.GaussianBlur(3)]
        # elif transform_id == 5:
        #     # task 5 is autocontrasted randomly 
        #     return [tvt.RandomAutocontrast(0.5)]
        # elif transform_id == 6:
        #     # task 6 is sharpened & blurred
        #     return [tvt.RandomAdjustSharpness(1.2 ,1.0), tvt.GaussianBlur(3)]
        # elif transform_id == 7:
        #     # task 7 is sharpened & jittered
        #     return [tvt.RandomAdjustSharpness(1.2 ,1.0), tvt.ColorJitter(brightness=0.02, contrast=0.02, saturation=0.02)]

    def load_problem_dataset(self):
        d_train, d_val, d_test = dt.load_CIFAR10_preaug(self.get_task_augmentation(), augment=self.augment)
        rng_split2 = torch.manual_seed(43)
        d_trains = dt.random_split(d_train, [1/self.num_split for _ in range(self.num_split)], generator=rng_split2)
        d_vals = dt.random_split(d_val, [1/self.num_split for _ in range(self.num_split)], generator=rng_split2)
        return d_trains[self.subtask], d_vals[self.subtask], d_test
    

class CIFAR100VariedSplitTaskClassificationNASProblem(CIFAR10ClassificationNASProblem):
    
    def __init__(self, verbose=False, pretrained_nets_dir:Optional[str]=None, subtask=0, num_split=3, net_sampling_config={}, augment=False):
        self.pretrained_nets_dir = "./trained-nets-cifar10split" if pretrained_nets_dir is None else pretrained_nets_dir
        super().__init__(verbose=verbose, pretrained_nets_dir=pretrained_nets_dir, net_sampling_config=net_sampling_config, augment=augment)
        self.num_split = num_split
        self.subtask = subtask
    
    def load_problem_dataset(self):
        d_train, d_val, d_test = dt.load_CIFAR100presplit(self.subtask, augment=self.augment)
        # already split, so no need to split further in this case :)
        return d_train, d_val, d_test
    
    def sample_architecture_from_file(self, rng: np.random.Generator):
        # without annotation
        networks = list(Path(self.pretrained_nets_dir).glob(f"*-t{self.subtask}.th"))
        # with annotation
        networks +=  list(Path(self.pretrained_nets_dir).glob(f"*-t{self.subtask}-*.th"))
        net = torch.load(rng.choice(networks)) # type: ignore

        # These networks should be trained, but not yet evaluated.
        return NeuralNetIndividual(net, None, is_trained=True)

    def sample_random_architecture(self, rng: np.random.Generator):
        norm_kinds = ["batch", "instance"]
        norm_kinds = self.net_sampling_config.get("norm_kinds", norm_kinds)

        filter_sizes = [3, 5, 7]
        filter_sizes = self.net_sampling_config.get("filter_sizes", filter_sizes)

        # channel_widths = [8, 16, 24, 32, 64, 128, 256, 512]
        channel_widths = [32, 64, 128, 256, 512]
        channel_widths = self.net_sampling_config.get("channel_widths", channel_widths)

        with_dropout_choices = [None, "before_norm", "after_norm"]
        with_dropout_choices = self.net_sampling_config.get("with_dropout_choices", with_dropout_choices)    
        
        kind = rng.integers(3)
        num_channels = rng.choice(channel_widths)
        with_dropout = rng.choice(with_dropout_choices)
        filter_size = rng.choice(filter_sizes)
        norm_kind = rng.choice(norm_kinds)

        if kind == 0:
            kind_name = "convnet"
            net = ly.get_simple_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 1:
            kind_name = "seperable convnet"
            net = ly.get_separable_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        elif kind == 2:
            kind_name = "residual convnet"
            net = ly.get_residual_convnet(3, 10, num_channels=num_channels, with_dropout=with_dropout, filter_size=filter_size, norm_kind=norm_kind).to_graph()
        else:
            raise ValueError()
        
        # These networks are unevaluated (None) and untrained (False)
        r = NeuralNetIndividual(net, None, is_trained=False)
        r.meta = dict(
            kind = kind,
            kind_name = kind_name,
            num_channels = num_channels,
            with_dropout = with_dropout,
            filter_size = filter_size,
            norm_kind = norm_kind,
            subtask=self.subtask,
            num_subtasks=self.num_split,
        )
        return r

    def sample_architecture(self, rng: np.random.Generator):
        """
        Sample a architecture that is suitable for the given problem.

        i.e. the architecture is valid, it accepts the input data in the format for this problem, and provides output
        in the right format for this problem.
        """

        return self.sample_architecture_from_file(rng)

def get_imagenet_normalize():
    return transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

def get_imagenet_training_transform():
    normalize = get_imagenet_normalize()
    return transforms.Compose([
                    # Convert from PIL
                    transforms.ToImage(), # Alternatively transforms.ToImageTensor()
                    transforms.ToDtype(torch.uint8, scale=True),
                    # Default opts
                    transforms.RandomResizedCrop(224, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # To float format
                    transforms.ToDtype(torch.float32, scale=True),
                    normalize,
                ])

def get_imagenet_validation_transform():
    normalize = get_imagenet_normalize()
    return transforms.Compose([
                    # Convert from PIL - new since v2
                    transforms.ToImage(), # Alternatively transforms.ToImageTensor()
                    transforms.ToDtype(torch.uint8, scale=True),
                    transforms.Resize(256, antialias=True),
                    transforms.CenterCrop(224),
                    # transforms.ToTensor(),
                    # To float format
                    transforms.ToDtype(torch.float32, scale=True),
                    normalize,
                ])

class ImageNetKG(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        dataset_path = Path(self.root)

        datafile = None

        if split == "val":
            # the official validation set of ImageNet, commonly used
            # as test set.
            datafile = dataset_path / "ImageNet_validation.arrow"
            # Set default transformation for validation
            if transform is None:
                transform = get_imagenet_validation_transform()
        elif split == "train":
            datafile = dataset_path / "ImageNet_train.arrow"
            # Set default transformation for train
            if transform is None:
                transform = get_imagenet_training_transform()
                
        if not datafile.exists():
            self.process_mappings()

        self.dt = pl.read_ipc(datafile)
        self.info = pl.read_ipc(dataset_path / "LOC_synset_mapping.arrow")

        self.transform = transform

    def process_mappings(self):
        dataset_path = Path(self.root)
        # Convert class mapping to arrow ipc format.
        numerical_id_class_id_name_mapping = (
            pl.read_csv(
                # Not actually a CSV - but it is a line-separated
                # text file, so close enough.
                dataset_path / "LOC_synset_mapping.txt",
                # This file does not have an header
                has_header=False,
                # Give the singular column a name `r`, for row.
                new_columns=["r"],
                # Just a placeholder that does not occur, so that 
                # we read separate lines without splitting.
                separator="|")
            .lazy()
            # The text files contains two columns, "folder" and "class",
            # The class id is implicit via the row number.
            # The following expression performs this conversion.
            .select(
                pl.col("r").str.splitn(" ", 2)
                .struct.rename_fields(["Class", "Description"])
            )
            .unnest("r")
            .with_row_count(name="cid", offset=0)
        ).collect()

        # Write synset mapping to arrow file
        numerical_id_class_id_name_mapping.write_ipc(dataset_path / "LOC_synset_mapping.arrow")

        # Convert solution mappings
        def process_solution_mapping(file):
            return (pl.read_csv(file)
                .lazy()
                .select([
                    pl.col("ImageId"),
                    pl.col("PredictionString").str.split_exact(" ", 0)
                    .struct.rename_fields(["Class"])
                ])
                .unnest("PredictionString")
                .join(numerical_id_class_id_name_mapping.lazy(), on="Class")
                .select([
                    "ImageId", "Class", "cid"
                ]))
        
        train_solution = (process_solution_mapping(dataset_path / "LOC_train_solution.csv")
            .select([
                # Construct path relative to root folder
                (pl.lit("ILSVRC/Data/CLS-LOC/train/") + pl.col("Class") + "/" + pl.col("ImageId") + ".JPEG").alias("Path"),
                pl.col("cid")
            ])
        ).collect()
        train_solution.write_ipc(dataset_path / "ImageNet_train.arrow")
        validation_solution = (process_solution_mapping(dataset_path / "LOC_val_solution.csv")
            .select([
                # Construct path relative to root folder
                (pl.lit("ILSVRC/Data/CLS-LOC/val/") + pl.col("ImageId") + ".JPEG").alias("Path"),
                pl.col("cid")
            ])
        ).collect()
        validation_solution.write_ipc(dataset_path / "ImageNet_validation.arrow")

    def __len__(self):
        return len(self.dt)
    
    def class_name(self, idx):
        return self.info[idx]["Description"]

    def __getitem__(self, idx):
        r = self.dt[idx]

        x = loadimage_torchio(os.path.join(self.root, r["Path"].item()))
        if self.transform:
            x = self.transform(x)
        return x, r["cid"].item()

class ImageNetV2(Dataset):
    """
    ImageNet V2 - downloaded from
        https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main
    
    See also:
        Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. 2019.
        Do ImageNet classifiers generalize to ImageNet?. In International Conference on
        Machine Learning. PMLR, 5389-5400.
    """

    def __init__(self, root, split, transform=None, sample_limit=None):
        self.root = root
        self.sample_limit = sample_limit
        dataset_path = Path(self.root)

        # split can be one of the following
        possible_splits = {
            "matched-frequency": [dataset_path / "imagenetv2-matched-frequency-format-val"],
            "threshold0.7": [dataset_path / "imagenetv2-threshold0.7-format-val"],
            "top-images": [dataset_path / "imagenetv2-top-images-format-val"],
            "merged": [dataset_path / "imagenetv2-matched-frequency-format-val",
                       dataset_path / "imagenetv2-threshold0.7-format-val",
                       dataset_path / "imagenetv2-top-images-format-val"],
        }

        datafile = dataset_path / f"imagenetv2-{split}-index.arrow"

        if not datafile.exists():
            self.create_index(possible_splits[split])\
                .write_ipc(datafile)
            
        self.dt = pl.read_ipc(datafile)

        self.info = None

        if transform is None:
            transform = get_imagenet_validation_transform()
            
        self.transform = transform

    def create_index(self, base_directories):

        def create_directory_index(base_dir):
            files = base_dir.glob("*/*.jpeg")
            return pl.DataFrame([[str(f.relative_to(self.root)), f.name, f.parent.name] for f in files], schema=["Path", "ImageId", "cid"])

        dfs = (
            pl.concat([
                create_directory_index(base_dir).lazy()
                for base_dir in base_directories])
            .unique("ImageId")
            .select([
                pl.col("Path"),
                pl.col("cid").cast(int),
            ])
        ).collect()

        return dfs

    def get_info(self, idx):
        if self.info is None:
            dataset_path = Path(self.root)
            self.info = pl.read_ipc(dataset_path / "LOC_synset_mapping.arrow")

        return self.info[idx]["Description"]

    def __len__(self):
        if self.sample_limit:
            return min(self.sample_limit, len(self.dt))
        return len(self.dt)
    
    def class_name(self, idx):
        return self.info[idx]["Description"]

    def __getitem__(self, idx):
        r = self.dt[idx]

        x = loadimage_torchio(os.path.join(self.root, r["Path"].item()))

        if self.transform:
            x = self.transform(x)
        return x, r["cid"].item()

class ImageNetProblem(ClassificationNASProblem):
    
    def __init__(self, root, validation_sample_limit=None, verbose=False):
        super().__init__(verbose=verbose)
        self.root = root
        self.validation_sample_limit = validation_sample_limit

    def load_problem_dataset(self) -> tuple: # type: ignore
        d_train = ImageNetKG(self.root, "train", None)

        # Validation set will be ImageNet V2 - downloaded from
        # https://huggingface.co/datasets/vaishaal/ImageNetV2/tree/main
        # See also:
        # Benjamin Recht, Rebecca Roelofs, Ludwig Schmidt, and Vaishaal Shankar. 2019.
        # Do ImageNet classifiers generalize to ImageNet?. In International Conference on
        # Machine Learning. PMLR, 5389–5400.
        d_validation = ImageNetV2(self.root, "merged", None, sample_limit=self.validation_sample_limit)
        # Use the original validation set as test set - as
        # is traditionally the case.
        d_test = ImageNetKG(self.root, "val", None)

        return d_train, d_validation, d_test


def get_imagenet_training_transform():
    normalize = get_imagenet_normalize()
    return transforms.Compose([
                    # Convert from PIL
                    transforms.ToImage(), # Alternatively transforms.ToImageTensor()
                    transforms.ToDtype(torch.uint8, scale=True),
                    # Default opts
                    transforms.RandomResizedCrop(224, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    # transforms.ToTensor(),
                    # To float format
                    transforms.ToDtype(torch.float32, scale=True),
                    normalize,
                ])


class VOCSegmentationProblem(ClassificationNASProblem):
    # Note: might have to write a SegmentationNASProblem superclass instead.
    # We will have to see whether the currently used routines work for non
    # classification tasks.

    def __init__(self, root, validation_sample_limit=None, verbose=False, loss_fn=None, batched_validation=False):
        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=255)
        super().__init__(verbose=verbose, loss_fn=loss_fn)
        self.root = root
        self.validation_sample_limit = validation_sample_limit
        self.batched_validation = batched_validation

    def compute_batch_loss(self, a, b):
        a = a["out"]
        b = b[:, 0, ...].long()
        # note if the batch size of b is a multiple, we are training multiple outputs
        # stacked together (i.e. an ensemble) tile the output
        if a.shape[0] != b.shape[0]:
            sh = [a.shape[0] // b.shape[0]] + [1 for _ in range(len(b.shape) - 1)]
            b = b = b.repeat(*sh)
        return self.loss_fn(a, b)

    def compute_batch_correct(self, a, b):
        a = a["out"]
        b = b[:, 0, ...]
        # compute # correct
        num_correct = (a.argmax(dim=1) == b.long()).sum(dim=(1,2))
        num_unignored = (255 != b.long()).sum(dim=(1,2))
        return torch.nan_to_num(num_correct / num_unignored, 1).sum().detach().cpu().item()

    def ensure_downloaded(self):
        self.load_problem_dataset(download=True)

    def get_default_normalize(self):
        return transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])

    def get_default_training_transforms(self):
        normalize = self.get_default_normalize()
        return transforms.Compose([
                    # Convert from PIL
                    transforms.ToImage(), # Alternatively transforms.ToImageTensor()
                    transforms.ToDtype(torch.uint8, scale=True),
                    # Default opts
                    transforms.RandomResizedCrop(520, antialias=True),
                    transforms.RandomHorizontalFlip(),
                    # Convert to float
                    transforms.ToDtype(torch.float32, scale=True),
                    normalize,
            ])
    
    def get_validation_resize(self):
        if not self.batched_validation:
            return (transforms.Resize(520, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True), )
            
        return (
            transforms.Resize(520, interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
            transforms.CenterCrop(520)
        )

    def get_default_validation_transforms(self):
        normalize = self.get_default_normalize()
        return transforms.Compose([
                    # Convert from PIL
                    transforms.ToImage(), # Alternatively transforms.ToImageTensor()
                    transforms.ToDtype(torch.uint8, scale=True),
                    # Default opts
                    *self.get_validation_resize(),
                    # Convert to float
                    transforms.ToDtype(torch.float32, scale=True),
                    normalize,
                ])

    def load_problem_dataset(self, download=False, transforms_train=None, transforms_validation=None) -> tuple: # type: ignore
        from torchvision.datasets import wrap_dataset_for_transforms_v2
        if transforms_train is None: transforms_train = self.get_default_training_transforms()
        if transforms_validation is None: transforms_validation = self.get_default_validation_transforms()

        # Problem: originally pretrained networks are trained using /both/ these datasets.
        # Therefore we CANNOT evaluate the performance of these networks properly
        # as the corresponding test set is not available.
        d_train = torchvision.datasets.VOCSegmentation(self.root, "2012", "train", download=download, transforms=transforms_train)
        d_validation = torchvision.datasets.VOCSegmentation(self.root, "2012", "val", download=download, transforms=transforms_validation)
        # We use 2007's public test set for local evaluation, if used, for now.
        # We may want to use VOC's evaluation server in the end once we are happy. 
        d_test = torchvision.datasets.VOCSegmentation(self.root, "2007", "test", download=download, transforms=transforms_validation)

        # Wrap datasets for v2 transforms
        d_train = wrap_dataset_for_transforms_v2(d_train)
        d_validation = wrap_dataset_for_transforms_v2(d_validation)
        d_test = wrap_dataset_for_transforms_v2(d_test)

        return d_train, d_validation, d_test


def get_problem_strs():
    return {"fmnist", "cifar10", "cifar100", "cifar10split", "cifar10splitdaug"}

def get_problem_by_str(problem: str, **args):
    if problem == "fmnist":
        return FashionMNISTImageClassificationNASProblem(**args)
    elif problem == "cifar10":
        return CIFAR10ClassificationNASProblem(**args)
    elif problem == "cifar100":
        return CIFAR100ClassificationNASProblem(**args)
    elif problem == "cifar10split":
        return CIFAR10SplitTaskClassificationNASProblem(**args)
    elif problem == "cifar10splitdaug":
        return CIFAR10SplitTaskDifferentAugmentationClassificationNASProblem(**args)

    raise ValueError()
