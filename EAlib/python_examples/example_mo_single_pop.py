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

population_size = 16
number_of_clusters = 3
objective_indices = [0, 1]

archive = ealib.BruteforceArchive(objective_indices)
init = ealib.CategoricalUniformInitializer()
foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.AsIs)

criterion = ealib.MOGAcceptanceCriterion(ealib.WrappedOrSingleSolutionPerformanceCriterion(ealib.DominationObjectiveAcceptanceCriterion(objective_indices)), archive)

gomea = ealib.MO_GOMEA(population_size, number_of_clusters, objective_indices, init, foslearner, criterion, archive)
stepper = ealib.TerminationStepper((lambda : gomea), 10)

om = ealib.OneMax(100, index=0)
zm = ealib.ZeroMax(100, index=1)


problem = ealib.Compose([om, zm])
problem_limited = ealib.Limiter(problem, 20000, datetime.timedelta(seconds=1))
# problem_monitored = ealib.ElitistMonitor(problem_limited, criterion)
f = ealib.SimpleConfigurator(problem_limited, stepper, 42)
f.run()

pop = f.getPopulation()
print(problem_limited.get_time_spent_ms())
print(problem_limited.get_num_evaluations())

objectives = []

print(f"Found {len(archive.get_archived())} points on the front!")
for e in archive.get_archived():
    # print(f"elitist: {e}")
    # print(pop.getData(ealib.GENOTYPECATEGORICAL, e).genotype)
    # print(pop.getData(ealib.OBJECTIVE, e).objective)
    objectives.append(np.array(pop.getData(ealib.OBJECTIVE, e), copy=False))

import numpy as np
import matplotlib.pyplot as plt

print(np.unique(objectives, axis=0).shape[0])

objectives = np.array(objectives)
plt.scatter(objectives[:, 0], objectives[:, 1])
plt.savefig("test.png")
