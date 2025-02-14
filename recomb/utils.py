#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

import torch
import json
import ConfigSpace as cslib
from . import problems
from pathlib import Path
import numpy as np

# Validation-based Early Stopping
class NoImprovementTerminator():
    def __init__(self, dev, problem: problems.NASProblem, lim=5, measure="accuracy", backup=None, lim_total=None, impatient=True):
        self.s = 0
        self.steps_total = 0
        self.dev = dev
        self.problem = problem
        self.impatient = impatient
        self.measure = measure
        # should return true if b is better than a.
        self.ord = lambda a, b: a < b
        self.lim = lim
        self.lim_total = lim_total
        self.backup = backup

        # For keeping track of best accuracy & loss
        self.q = 0.0
        self.q_l = np.inf
        # For keeping track of last loss
        self.q_last = 0.0
        self.q_last_l = np.inf
        # And for a given span (i.e. set of training epochs with specific hyperparameters)
        self.q_span = 0.0
        self.q_l_span = np.inf

        self.reverted_optimizer = False

    def cmp_measure(self, a, b):
        # a and b are both tuples of (accuracy, loss) or (None, None)
        # should return True if b is better than a.
        if a[0] is None:
            return True

        if self.measure == "accuracy":
            return a[0] < b[0]
        elif self.measure == "loss":
            return a[1] > b[1]
        else:
            raise Exception("Invalid measure")

    def __call__(self, data: dict):
        neti: problems.NeuralNetIndividual = data["neti"]
        new_q, new_q_loss = self.problem.evaluate_network(dev=self.dev, neti=neti, objective="both", return_to_cpu=False)
        # Use np.nan_to_num to convert nans to np.inf for loss (so they are never the best)
        new_q_loss = np.nan_to_num(new_q_loss, np.inf)

        # Track last evaluation
        self.q_last = new_q
        self.q_last_l = new_q_loss

        self.steps_total += 1
        
        if self.cmp_measure((self.q_span, self.q_l_span), (new_q, new_q_loss)):
            self.q_span = new_q
            self.q_l_span = new_q_loss

        if self.cmp_measure((self.q, self.q_l), (new_q, new_q_loss)):
            self.q = new_q
            self.q_l = new_q_loss
            self.s = 0
            # if we are backing up - save to file
            if self.backup is not None:
                backup_state = {
                    "network_state": neti.net.state_dict(),
                }
                optimizer = data.get("optimizer")
                if optimizer is not None:
                    backup_state["optimizer_state"] = optimizer.state_dict()
                torch.save(backup_state, self.backup)
        else:
            self.s += 1

        # if we are impatient - we immidiately terminate.
        # if we are not impatient - we continue still, but keep track of the
        # fact that we are not improving - for HPO, this might be a good signal
        # to stop the HPO process.

        self.patience_limit_hit = (self.s >= self.lim)
        self.total_limit_hit = (self.lim_total is not None and (self.steps_total >= self.lim_total))

        terminate = (self.impatient and self.patience_limit_hit) or self.total_limit_hit
        
        # note - class that has access to optimizer & network should probably try to revert :)
        # if terminate:
        #     self.try_revert()
            
        return terminate
    
    def try_revert(self, neti, optimizer = None):
        if self.backup is not None:
            # revert state
            # if we backup to a file - reset
            backup_state = torch.load(self.backup)
            neti.net.load_state_dict(backup_state["network_state"])
            optimizer_state = backup_state.get("optimizer_state")
            if optimizer is not None and optimizer_state is not None:
                self.reverted_optimizer = False
                try:
                    optimizer.load_state_dict(optimizer_state)
                    self.reverted_optimizer = True
                except:
                    # if failed (i.e. different kind of optimizer), ignore.
                    print("tried to revert optimizer state, but failed.")
                    pass
            # also, set q_last, and q_last_l to the values corresponding
            # to the backup being loaded.
            self.q_last = self.q
            self.q_last_l = self.q_l
            self.q_span = self.q
            self.q_l_span = self.q_l

    def cleanup(self):
        if self.backup is not None:
            # remove, or not if already removed.
            Path(self.backup).unlink(missing_ok=True)

    def reset_span(self):
        self.q_span = 0.0
        self.q_l_span = np.inf

    def get_best_span_validation_accuracy(self):
        return self.q_span

    def get_best_span_validation_loss(self):
        return self.q_l_span

    def get_best_validation_accuracy(self):
        return self.q
    
    def get_best_validation_loss(self):
        return self.q_l
    
    def get_validation_accuracy(self):
        return self.q_last
    
    def get_validation_loss(self):
        return self.q_last_l
    
# Following https://stackoverflow.com/a/57915246
class SmartEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, cslib.Configuration):
            return super(SmartEncoder, self).default(dict(obj))
        if isinstance(obj, cslib.hyperparameters.Hyperparameter):
            return str(obj)
        return super(SmartEncoder, self).default(obj)