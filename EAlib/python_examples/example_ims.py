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
