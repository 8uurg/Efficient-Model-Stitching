import ealib
import numpy as np
import datetime


init = ealib.CategoricalUniformInitializer()
foslearner = ealib.CategoricalLinkageTree(ealib.NMI(), ealib.FoSOrdering.AsIs)
criterion = ealib.ObjectiveAcceptanceCriterion()

archive = ealib.BruteforceArchive([0])
gomea = ealib.GOMEA(16, init, foslearner, criterion, archive)
stepper = ealib.TerminationStepper((lambda : gomea), 10)
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
print(np.array(pop.getData(ealib.GENOTYPECATEGORICAL, elitist), copy=False))
print(np.array(pop.getData(ealib.OBJECTIVE, elitist), copy=False))


