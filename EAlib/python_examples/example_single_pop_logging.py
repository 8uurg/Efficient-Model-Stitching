#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

import ealib
import numpy as np
import datetime

l = 3
seed = 42
population_size = 16
budget_e = 1_000_000
budget_t = datetime.timedelta(seconds=10)

init = ealib.CategoricalUniformInitializer()
foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.AsIs)
criterion = ealib.SingleObjectiveAcceptanceCriterion()


a = ealib.BruteforceArchive([0])
archive = ealib.LoggingArchive(a, ealib.CSVLogger(
    "elitist.csv",
    ealib.SequencedItemLogger([
        ealib.NumEvaluationsLogger(),
        ealib.ObjectiveLogger(),
        ealib.GenotypeCategoricalLogger(),
    ])))
gomea = ealib.GOMEA(population_size, init, foslearner, criterion, archive)
stepper = ealib.TerminationStepper((lambda : gomea), 10)
problem = ealib.OneMax(l)
problem_limited = ealib.Limiter(problem, budget_e, budget_t)
problem_monitored = ealib.ElitistMonitor(problem_limited, criterion)
f = ealib.SimpleConfigurator(problem_monitored, stepper, seed)
f.run()

pop = f.getPopulation()
print(problem_limited.get_time_spent_ms())
print(problem_limited.get_num_evaluations())
elitist = archive.get_archived()[0]
print(f"elitist: {elitist}")
print(np.array(pop.getData(ealib.GENOTYPECATEGORICAL, elitist), copy=False))
print(np.array(pop.getData(ealib.OBJECTIVE, elitist), copy=False))


