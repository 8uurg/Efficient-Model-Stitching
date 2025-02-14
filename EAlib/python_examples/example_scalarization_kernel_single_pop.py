#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

import ealib
import datetime
import numpy as np
from collections import Counter

population_size = 128
number_of_clusters = 7
objective_indices = [0, 1]

archive = ealib.BruteforceArchive(objective_indices)
init = ealib.CategoricalUniformInitializer()
foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.Random, False, False, True)
scalarizer = ealib.TschebysheffObjectiveScalarizer(objective_indices)

plugin = ealib.HoangScalarizationScheme(scalarizer, objective_indices)
criterion = ealib.ScalarizationAcceptanceCriterion(scalarizer)

linkagekernels = False
# gomea = ealib.MO_GOMEA(population_size, number_of_clusters, objective_indices, init, foslearner, criterion, archive)
gomea = ealib.KernelGOMEA(population_size, number_of_clusters, objective_indices, init, foslearner, criterion, archive, plugin)
# linkagekernels = True
stepper = ealib.TerminationStepper((lambda : gomea), 10)

om = ealib.OneMax(100, index=0)
zm = ealib.ZeroMax(100, index=1)


problem = ealib.Compose([om, zm])
problem_limited = ealib.Limiter(problem, 500000, datetime.timedelta(minutes=2))
# problem_monitored = ealib.ElitistMonitor(problem_limited, criterion)
f = ealib.SimpleConfigurator(problem_limited, stepper, 42)

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
for gen in range(10):
    f.step()

    pop = f.getPopulation()
    # print(problem_limited.get_time_spent_ms())
    # print(problem_limited.get_num_evaluations())


    solutions = gomea.getSolutionPopulation()

    solution_i_to_index = {
        s.i: idx
        for idx, s in enumerate(solutions)
    }
    remap_solution_i = np.vectorize(solution_i_to_index.get)
    # print(solution_i_to_index)

    print(f"Population: {len(solutions)}")
    objectives = []
    sod = []
    neighborhood = []
    for e in solutions:
        # print(f"elitist: {e}")
        # print(pop.getData(ealib.GENOTYPECATEGORICAL, e).genotype)
        # print(pop.getData(ealib.OBJECTIVE, e).objective)
        objectives.append(np.array(pop.getData(ealib.OBJECTIVE, e), copy=False))
        sod.append(pop.getData(ealib.USESINGLEOBJECTIVE, e).index)
        if linkagekernels:
            lk = pop.getData(ealib.LINKAGEKERNEL, e)
            neighborhood.append([ii.i for ii in lk.pop_neighborhood])


    print(f"Found {len(archive.get_archived())} points on the front!")
    objectives_archive = []
    for e in archive.get_archived():
        # print(f"elitist: {e}")
        # print(pop.getData(ealib.GENOTYPECATEGORICAL, e).genotype)
        # print(pop.getData(ealib.OBJECTIVE, e).objective)
        objectives_archive.append(np.array(pop.getData(ealib.OBJECTIVE, e), copy=False))

    objectives_archive = np.array(objectives_archive)
    objectives = np.array(objectives)
    sod = np.array(sod)
    print(Counter(sod))

    fig.clear()

    plt.scatter(objectives[:, 0], objectives[:, 1], marker='o', alpha=0.1, label="Population")
    plt.scatter(objectives[sod == 0, 0], objectives[sod == 0, 1], marker='1', label="Single Objective (O0)")
    plt.scatter(objectives[sod == 1, 0], objectives[sod == 1, 1], marker='2', label="Single Objective (O1)")
    plt.scatter(objectives_archive[:, 0], objectives_archive[:, 1], s=0.5, marker='x', label="Archive")

    # Plot neighborhood of item 0
    if len(neighborhood) > 0 and len(neighborhood[0]) != 0:
        nb = neighborhood[0]
        print(remap_solution_i(nb))
        print(nb)
        xs = np.empty(3 * len(nb))
        xs[::3] = objectives[0, 0]
        xs[1::3] = objectives[remap_solution_i(nb), 0]
        xs[2::3] = np.nan
        
        ys = np.empty(3 * len(nb))
        ys[::3] = objectives[0, 1]
        ys[1::3] = objectives[remap_solution_i(nb), 1]
        ys[2::3] = np.nan

        plt.plot(xs, ys, label="Neighbors of first element", alpha=0.1)
        plt.scatter([objectives[0, 0]], [objectives[0, 1]], label="First", marker="*")
    plt.legend()
    ax = plt.gca()
    ax.set_xlabel("<- Objective 0")
    ax.set_ylabel("<- Objective 1")
    plt.savefig(f"test{gen}.png")
