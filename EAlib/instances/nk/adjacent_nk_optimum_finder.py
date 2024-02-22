from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import List
import numpy as np
import argparse

@dataclass
class LUTFunction:
    variables: np.ndarray
    lut: np.ndarray

    def get_fitnesses(self, x: np.ndarray) -> int:
        wlut = np.flip(np.power(2, np.arange(len(self.variables)))).reshape(1, -1)
        return self.lut[(x[:, self.variables] * wlut).sum(axis=1)]

@dataclass
class NKLandscape:
    subfunctions: List[LUTFunction]
    l: int

    def get_fitnesses(self, x: np.ndarray):
        return np.sum([s.get_fitnesses(x) for s in self.subfunctions], axis=0)

def parse_nklandscape(p: Path) -> NKLandscape:
    with p.open() as f:
        next_line = f.readline().split()
        l = int(next_line[0])
        num_subfunctions = int(next_line[1])
        next_line = f.readline()
        luts = []
        while next_line:
            next_line = next_line.split()
            num_vars = int(next_line[0])
            variables = np.array([int(a) for a in next_line[1:]])
            # Assumption: entries are in power order
            lut = np.array([float(f.readline().split()[1]) for idx in range(2**num_vars)])
            luts.append(LUTFunction(variables, lut))
            next_line = f.readline()
        return NKLandscape(luts, l)

def solve_well_ordered_adjacent_nklandscape(nkl: NKLandscape):
    # assumes a lot of stuff about the ordering of subfunctions, and which
    # variables overlap. Variables are assumed to be binary.
    sfn1 = nkl.subfunctions[0]
    sfn2 = nkl.subfunctions[1]

    # Infer block size & overlap between sequential subfunctions.
    n = len(sfn1.variables)
    o = sum((a in sfn2.variables) for a in sfn1.variables)
    s = n - o
    
    # Create other lookup tables for particular operations
    # wlut = np.flip(np.power(2, np.arange(n))).reshape(1, -1)
    vlut = np.array(list(list(a) for a in product([0, 1], repeat=n - o))).astype(int) # table for argmax -> values
    vlut_s = np.array(list(list(a) for a in product([0, 1], repeat=o))).astype(int) # table for argmax -> values

    # lut is a 2^n table. the last s variables overlap with the next set, which will need to be accounted for.
    # To start with. aggregate for each possible set of the last 2 variables
    shp_a = (2**o, -1)
    shp_b = (-1, 2**o)
    shp_t = (2**s, -1)
    olut = sfn1.lut.reshape(shp_b)
    contributions = olut.max(axis=0).T # Aggregate over the first axis
    
    variable_values = vlut[olut.argmax(axis=0), :] # Find which variable assignments maximized for each subpair of variables.


    for subfn in nkl.subfunctions[1:]:
        # add the contributions from the previous subfunction
        olut = subfn.lut.reshape(shp_a) + contributions.reshape(-1, 1)
        # prepare for aggregation
        olut = olut.reshape(shp_b)
        # Compute contributions for next subfuntion
        contributions = olut.max(axis=0).T
        # Decide on the next set of (s) variable values.
        indices = olut.argmax(axis=0)
        # So now we have the variable values corresponding to
        # i_0, ..., i_s                (via indices)
        #               p_0, ..., p_o  (via positioning)
        # <-------o---------->          for variables from previous lut
        # hence the cell we need to lookup the previous value can be found in
        vs = (np.arange(2**o) >> s) + (indices << (o - s))
        variable_values = np.hstack([variable_values[vs], vlut[indices, :]])

    # Finally pick the contribution with maximum value
    opt = contributions.max()
    idx = contributions.argmax()
    opt_sol = np.hstack([variable_values[idx], vlut_s[idx, :]])

    return opt, opt_sol


parser = argparse.ArgumentParser()
parser.add_argument("instance", type=Path)
r = parser.parse_args()

path = r.instance
# path = Path("./instances/n4_s1/L20/2.txt")
# path = Path("./instances/nk/instances/n4_s1/L20/1.txt")
nkl = parse_nklandscape(path)
opt, opt_sol = solve_well_ordered_adjacent_nklandscape(nkl)
# Note: unused variables (i.e. at the end) are NOT included in the optimal solution.

check_opt = nkl.get_fitnesses(opt_sol.reshape(1, -1))[0]

with path.with_suffix(".txt.opt").open("w") as f:
    f.write(f"{opt}\t{' '.join(str(a) for a in opt_sol)}")

# print(f"Found solution with fitness {opt} / {check_opt}: {opt_sol}")
