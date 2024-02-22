import recomb.layers as ly
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from copy import deepcopy
from tqdm.auto import tqdm
from sklearn.decomposition import PCA

def verbosity_helper(verbose=True):
    if verbose:
        def get_tqdm(*x, **y):
            it = tqdm(*x, **y)
            return ((v, it.set_postfix) for v in it)
        return get_tqdm
    else:
        return lambda x, *_, **__: ((v, print) for v in x)

def train_network(net: ly.ModuleT, dataset, batch_size=1024, seed=42, num_epochs=100, early_stopping=None, dev=None, verbose=True):
    # Early stopping is (number_of_failing_steps_allowed, validation_dataset)
    # Seed rng with a reasonable seed (i.e. not 42 directly, but one wrangled through a prng)
    rngx = np.random.default_rng(seed=seed)
    rng = torch.manual_seed(rngx.integers(np.iinfo(np.int64).max))

    progress_bar = verbosity_helper(verbose=verbose)
    # Sidenote - we will continue running training until performance has regressed

    net.train()

    optimizer = torch.optim.Adam(net.parameters())
    loss_fn = nn.CrossEntropyLoss()

    is_cpu = dev is None or str(dev) == "cpu"
    # print(f"Training in CPU mode: {is_cpu}")

    dl = DataLoader(dataset, 
        batch_size=batch_size,
        shuffle=True,
        generator=rng,
        pin_memory=not is_cpu)
        
    # pin_memory_device=("" if is_cpu else str(dev))

    best_val_err = np.inf
    num_iter_failed = 0
    if dev is not None:
        loss_fn = loss_fn.to(dev)
        net = net.to(dev)

    for epoch, info in progress_bar(range(num_epochs)):
        cumulative_loss = 0.0
        num_samples = 0
        # Disable for now...
        for (X, y_true), _ in progress_bar(dl, leave=False, disable=True):
            X, y_true = X.to(dev), y_true.to(dev)
            
            optimizer.zero_grad()
            y_pred = net(X)
            loss = loss_fn(y_pred, y_true)
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.detach() * X.shape[0]
            num_samples += X.shape[0]
        
        iter_info = {
            "train": f"loss: {(cumulative_loss / num_samples):.3f}"
        }
        if early_stopping is not None:
            current_val_loss, current_val_acc = evaluate_network(net, early_stopping[1], batch_size=batch_size, alter_mode=False, dev=dev, verbose=False)
            iter_info["validation"] = f"loss: {current_val_loss:.3f} - acc: {current_val_acc:.2f}"
            val_err = current_val_loss
            if best_val_err < val_err:
                if num_iter_failed > early_stopping[0]:
                    #
                    net.load_state_dict(torch.load( "training.th"))
                    info(iter_info)
                    break
                else:
                    num_iter_failed += 1
            else:
                torch.save(net.state_dict(), "training.th")
                best_val_err = val_err
                num_iter_failed = 0
        info(iter_info)


def posttrain_network(net: ly.ModuleT, dataset, batch_size=4, num_batches=16, seed=42, num_epochs=4, dev=None):
    """
    Go through the network with a small batch size...
    Seems to help with batchnorm not transferring to eval mode.
    """
    # Seed rng with a reasonable seed (i.e. not 42 directly, but one wrangled through a prng)
    rngx = np.random.default_rng(seed=seed)
    rng = torch.manual_seed(rngx.integers(np.iinfo(np.int64).max))

    net.train()

    if dev is not None:
        net = net.to(dev)

    is_cpu = dev is None or str(dev) == "cpu"
    dl = DataLoader(dataset, 
        batch_size=batch_size,
        shuffle=True,
        generator=rng,
        pin_memory=not is_cpu)
    # pin_memory_device=("" if dev is None else str(dev))

    for (X, y_true), _ in zip(dl, range(num_batches)):
        X, y_true = X.to(dev), y_true.to(dev)
        
        y_pred = net(X)

def evaluate_network(net: ly.ModuleT, dataset, batch_size=1024, seed=42, alter_mode=True, dev=None, verbose=True):
    # Seed rng with a reasonable seed (i.e. not 42 directly, but one wrangled through a prng)
    rngx = np.random.default_rng(seed=seed)
    rng = torch.manual_seed(rngx.integers(np.iinfo(np.int64).max))
    progress_bar = verbosity_helper(verbose=verbose)

    if alter_mode:
        was_training = net.training
        net.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()
    is_cpu = dev is None or str(dev) == "cpu"
    dl = DataLoader(dataset, 
        batch_size=batch_size,
        shuffle=True,
        generator=rng,
        pin_memory=not is_cpu)
    # pin_memory_device=("" if dev is None else str(dev))

    if dev is not None:
        loss_fn = loss_fn.to(dev)
        net = net.to(dev)

    for (X, y_true), _ in progress_bar(dl, leave=False):
        X, y_true = X.to(dev), y_true.to(dev)
        
        y_pred = net(X)
        loss = loss_fn(y_pred, y_true)
        total_loss += loss.detach() * X.shape[0] # correct for batch size.
        total_correct += (torch.argmax(y_pred.detach(), -1) == y_true).sum().detach()
        total_samples += X.shape[0]

    if alter_mode:
        net.train(was_training)
    
    return total_loss / total_samples, total_correct / total_samples

def evaluate_confusionmatrix_network(net: ly.ModuleT, dataset, num_classes, batch_size=1024, seed=42, alter_mode=True, dev=None, verbose=True):
    # Seed rng with a reasonable seed (i.e. not 42 directly, but one wrangled through a prng)
    rngx = np.random.default_rng(seed=seed)
    rng = torch.manual_seed(rngx.integers(np.iinfo(np.int64).max))
    progress_bar = verbosity_helper(verbose=verbose)

    if alter_mode:
        was_training = net.training
        net.eval()

    cx = torch.zeros((num_classes, num_classes)).to(dev)
    total_samples = 0
    loss_fn = nn.CrossEntropyLoss()
    is_cpu = dev is None or str(dev) == "cpu"
    dl = DataLoader(dataset, 
        batch_size=batch_size,
        shuffle=True,
        generator=rng,
        pin_memory=not is_cpu)
    # pin_memory_device=("" if dev is None else str(dev))

    if dev is not None:
        loss_fn = loss_fn.to(dev)
        net = net.to(dev)

    for (X, y_true), _ in progress_bar(dl):
        X, y_true = X.to(dev), y_true.to(dev)
        
        y_pred = net(X)
        loss = loss_fn(y_pred, y_true)
        nn.functional.one_hot
        # total_loss += loss.detach() * X.shape[0] # correct for batch size.
        # total_correct += (torch.argmax(y_pred.detach(), -1) == y_true).sum().detach()
        ohx = nn.functional.one_hot(torch.argmax(y_pred.detach(), -1)).reshape(-1, 1, num_classes)
        ohy = nn.functional.one_hot(y_true).reshape(-1, num_classes, 1)

        cx += (ohx * ohy).sum(dim=0) 
        total_samples += X.shape[0]

    if alter_mode:
        net.train(was_training)
    
    return cx / total_samples, total_samples

def interpolate(a: ly.ModuleT, b: ly.ModuleT, w: float):
    r = deepcopy(a)
    state_a = a.state_dict()
    state_b = b.state_dict()
    state_r = r.state_dict()

    for k, v in state_a.items():
        try:
            state_r[k] = state_b[k] * (1.0 - w) + v * w
        except:
            pass

    r.load_state_dict(state_r)

    return r


def distill_network(net_student: ly.ModuleT, net_teacher: ly.ModuleT, dataset, w_teacher, batch_size=1024, seed=42, num_epochs=100, early_stopping=None, dev=None, verbose=True):
    # Early stopping is (number_of_failing_steps_allowed, validation_dataset)
    # Seed rng with a reasonable seed (i.e. not 42 directly, but one wrangled through a prng)
    rngx = np.random.default_rng(seed=seed)
    rng = torch.manual_seed(rngx.integers(np.iinfo(np.int64).max))

    progress_bar = verbosity_helper(verbose=verbose)
    # Sidenote - we will continue running training until performance has regressed

    net_student.train()

    optimizer = torch.optim.Adam(net_student.parameters())
    loss_fn = nn.CrossEntropyLoss()

    is_cpu = dev is None or str(dev) == "cpu"
    # print(f"Training in CPU mode: {is_cpu}")

    dl = DataLoader(dataset, 
        batch_size=batch_size,
        shuffle=True,
        generator=rng,
        pin_memory=not is_cpu)
        
    # pin_memory_device=("" if is_cpu else str(dev))

    best_val_err = np.inf
    num_iter_failed = 0
    if dev is not None:
        loss_fn = loss_fn.to(dev)
        net_student = net_student.to(dev)

    for epoch, info in progress_bar(range(num_epochs)):
        cumulative_loss = 0.0
        num_samples = 0
        # Disable for now...
        for (X, y_true), _ in progress_bar(dl, leave=False, disable=True):
            X, y_true = X.to(dev), y_true.to(dev)
            
            optimizer.zero_grad()
            y_pred_teacher = net_teacher(X).detach()
            y_pred = net_student(X)
            loss = (loss_fn(y_pred, y_true) + loss_fn(y_pred, y_pred_teacher) * w_teacher) / (w_teacher + 1)
            loss.backward()
            optimizer.step()
            cumulative_loss += loss.detach() * X.shape[0]
            num_samples += X.shape[0]
        
        iter_info = {
            "train": f"loss: {(cumulative_loss / num_samples):.3f}"
        }
        if early_stopping is not None:
            current_val_loss, current_val_acc = evaluate_network(net_student, early_stopping[1], batch_size=batch_size, alter_mode=False, dev=dev, verbose=False)
            iter_info["validation"] = f"loss: {current_val_loss:.3f} - acc: {current_val_acc:.2f}"
            val_err = current_val_loss
            if best_val_err < val_err:
                if num_iter_failed > early_stopping[0]:
                    #
                    net_student.load_state_dict(torch.load( "training.th"))
                    info(iter_info)
                    break
                else:
                    num_iter_failed += 1
            else:
                torch.save(net_student.state_dict(), "training.th")
                best_val_err = val_err
                num_iter_failed = 0
        info(iter_info)

def remapping_loss_fn(x_pred, x_true, base_loss_fn):
    eps = 1e-7
    dim = tuple(i for i in range(len(x_true.shape)) if i != 1)
    ms = x_true.mean(dim=dim, keepdim=True).detach()
    ms_pred = x_pred.mean(dim=dim, keepdim=True)
    dv = x_true.std(dim=dim, keepdim=True).detach() + eps
    dv_a = (1 / dv).mean().detach()

    # Differences from the means
    x_pred_s = (x_pred - ms_pred.detach()) / dv
    x_true_s = (x_true - ms) / dv

    # First loss only regards offsets from the means.
    loss_corr = base_loss_fn(x_pred_s, x_true_s) / dv_a
    # Second loss is for the offset itself.
    loss_offset = base_loss_fn(ms_pred, ms)
    # if not printed:
    #     # printed = True
    #     print(f"dv_a: {dv_a}")
    #     print(f"Loss corr: {loss_corr} | Loss offset: {loss_offset}")
    # Final loss is the sum of these two.
    final_loss = loss_corr + loss_offset
    return final_loss

def imitate_network(net: ly.ModuleT, net_to_imitate: ly.ModuleT, dataset, alignments, batch_size=1024, num_epochs=20, seed=42, p_replace = 1.0, detach_pass=False, dev=None, verbose=True):
    loss = 0.0
    num_alignment_points = len(alignments)
    
    # w_place = 0.0
    hooks = []

    # Early stopping is (number_of_failing_steps_allowed, validation_dataset)
    # Seed rng with a reasonable seed (i.e. not 42 directly, but one wrangled through a prng)
    rngx = np.random.default_rng(seed=seed)
    rng = torch.manual_seed(rngx.integers(np.iinfo(np.int64).max))

    progress_bar = verbosity_helper(verbose=verbose)
    # Sidenote - we will continue running training until performance has regressed

    # net_to_imitate.eval()
    net_to_imitate.train()
    net.train()

    # optimizer = torch.optim.Adam(net.parameters(), lr=1.0)
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-1)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
    # optimizer = torch.optim.SGD(net.parameters(), lr=1e-3, momentum=0.1, nesterov=True)
    # printed = False
    base_loss_fn = nn.MSELoss(reduction='mean')
    loss_fn = lambda x_pred, x_true: remapping_loss_fn(x_pred, x_true, base_loss_fn)
    # loss_fn = base_loss_fn
    
    # loss_fn_end = nn.CrossEntropyLoss()
    # loss_fn = nn.MSELoss()
    dl = DataLoader(dataset, 
        batch_size=batch_size,
        shuffle=True,
        generator=rng,
        pin_memory=dev is not None, 
        pin_memory_device=("" if dev is None else str(dev)))

    best_val_err = np.inf
    num_iter_failed = 0
    if dev is not None:
        # loss_fn = loss_fn.to(dev)
        net = net.to(dev)
        net_to_imitate = net_to_imitate.to(dev)

    def attach_hook_for(module_training: nn.Module, module_reference: nn.Module):
        nonlocal loss
        nonlocal p_replace
        nonlocal dev
        nonlocal detach_pass
        value = None
        def hook_reference(module, input, output):
            nonlocal value
            # For a reference network, save the current value.
            # We are not training this network: detach gradients.
            value = output.detach()
        def hook_training(module, input, output):
            nonlocal value
            nonlocal loss
            nonlocal p_replace
            nonlocal dev
            nonlocal detach_pass
            # For the network under training,
            # compute the loss of the original output against the value of the reference network.
            local_loss = loss_fn(output, value)
            loss += local_loss
            assert local_loss >= 0.0, f"Local loss is {local_loss} for module {module}"
            # Replace output value with our current value.
            # return value
            # Return output value probabilistically.
            s = torch.rand(tuple([output.shape[0]] + [1 for _ in range(len(output.shape) - 1)])).to(dev)
            if detach_pass:
                # Detach the output gradient in any case - we don't want the weights before this point
                # to change depending on errors made later on.
                return torch.where(s < p_replace, value, output.detach())
            return torch.where(s < p_replace, value, output)

        hook_ref = module_reference.register_forward_hook(hook_reference)
        hook_training = module_training.register_forward_hook(hook_training)
        hooks.append(hook_ref)
        hooks.append(hook_training)

    for (point_in_net, point_in_net_to_imitate) in alignments:
        module_net = net.get_module_by_point(point_in_net)
        module_net_to_imitate = net_to_imitate.get_module_by_point(point_in_net_to_imitate)
        attach_hook_for(module_net, module_net_to_imitate)

    try:
        for epoch, info in progress_bar(range(num_epochs)):
            cumulative_loss = 0.0
            num_samples = 0
            # p_replace = 1 - (epoch / num_epochs)
            # Disable sub-progressbar for now...
            for (X, y_true), _ in progress_bar(dl, leave=False, disable=True):
                X, y_true = X.to(dev), y_true.to(dev)
                
                optimizer.zero_grad()
                # Reset loss.
                loss = 0.0
                # Set values using the hooks above!
                y_ref = net_to_imitate(X)
                # Note - loss is changed due to the hooks above!
                y_pred = net(X)
                # Include a loss term for the final result
                # loss += loss_fn_end(y_pred, y_ref)
                # 
                loss.backward()
                optimizer.step()
                cumulative_loss += loss.detach() * X.shape[0]
                num_samples += X.shape[0]
            
            iter_info = {
                "train": f"loss: {(cumulative_loss / (num_samples * num_alignment_points)):.10f}"
            }
            info(iter_info)
    finally:
        for hook in hooks:
            hook.remove()

def forward_get_all_feature_maps(net: ly.ModuleT, X):
    # it is assumed that net is a graph...
    points = list(net.enumerate_points())
    
    hooks = []

    features = [None for _ in points]
    def hook(module, input, output, index):
        features[index] = output.detach()

    try:
        # Register hooks
        for idx, point in enumerate(points):
            hooks.append(net.get_module_by_point(point).register_forward_hook(partial(hook, index=idx)))

        y_pred = net(X)
    finally:
        # Remove hooks
        for h in hooks:
            h.remove()
    
    return features

