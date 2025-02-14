#  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
# 
# This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
# 
# Project leaders: Peter A.N. Bosman, Tanja Alderliesten
# Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
# Main code developer: Arthur Guijt

import ealib
import datetime

init = ealib.CategoricalUniformInitializer()
foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.AsIs)
criterion = ealib.SingleObjectiveAcceptanceCriterion()
archive = ealib.BruteforceArchive([0])
stepper = ealib.InterleavedMultistartScheme((lambda p : ealib.GOMEA(p, init, foslearner, criterion, archive)), ealib.AverageFitnessComparator())

problem = ealib.OneMax(100)
problem_limited = ealib.Limiter(problem, 20000, datetime.timedelta(seconds=1))
problem_monitored = ealib.ElitistMonitor(problem_limited, criterion)
f = ealib.SimpleConfigurator(problem_monitored, stepper, 42)
f.run()

pop = f.getPopulation()
print(problem_limited.get_time_spent_ms())
print(problem_limited.get_num_evaluations())
elitist = archive.get_archived()[0]
print(f"elitist: {elitist}")
print(pop.getData(ealib.GENOTYPECATEGORICAL, elitist).as_list())
print(pop.getData(ealib.OBJECTIVE, elitist).as_list())
