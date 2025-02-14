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

replacement_strategy = 0
population_size = 16
tournament_size = 2

initializer = ealib.CategoricalUniformInitializer()
criterion = ealib.SingleObjectiveAcceptanceCriterion()
base_archive = ealib.BruteforceArchive([0])

problem = ealib.OneMax(10)
limiter = ealib.Limiter(problem, 20000)
problem_monitored = ealib.ElitistMonitor(limiter, criterion)


def get_runtime(population, individual):
    return 1.0


problem_simulated_runtime = ealib.SimulatedFunctionRuntimeObjectiveFunction(
    limiter, get_runtime
)

issd = ealib.ECGAGreedyMarginalProduct()
selection = ealib.OrderedTournamentSelection(
    tournament_size, 1, ealib.ShuffledSequentialSelection(), criterion
)

ws = ealib.WritingSimulator("./results/ecga-sync-python/events.jsonl", 1000.0)
sim = ealib.SimulatorParameters(ws, population_size, False)

archive_logitem = ealib.ArchivedLogger()
logger = ealib.CSVLogger(
    "./results/ecga-sync-python/archive.csv",
    ealib.SequencedItemLogger(
        [
            ealib.NumEvaluationsLogger(limiter),
            ealib.SimulationTimeLogger(ws),
            ealib.ObjectiveLogger(),
            archive_logitem,
            ealib.GenotypeCategoricalLogger(),
        ]
    ),
)
archive = ealib.LoggingArchive(base_archive, logger, archive_logitem)

approach = ealib.SynchronousSimulatedECGA(
    replacement_strategy, sim, population_size, issd, initializer, selection, archive
)
stepper = ealib.TerminationStepper((lambda: approach), 100)
f = ealib.SimpleConfigurator(problem_simulated_runtime, stepper, 42)
f.run()

pop = f.getPopulation()
print(limiter.get_time_spent_ms())
print(limiter.get_num_evaluations())
elitist = archive.get_archived()[0]
print(f"elitist: {elitist}")
print(np.array(pop.getData(ealib.GENOTYPECATEGORICAL, elitist), copy=False))
print(np.array(pop.getData(ealib.OBJECTIVE, elitist), copy=False))
