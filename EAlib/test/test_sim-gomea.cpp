#include "archive.hpp"
#include "base.hpp"
#include "initializers.hpp"
#include "logging.hpp"
#include "problems.hpp"
#include "sim-gomea.hpp"
#include "sim.hpp"

#include <catch2/catch.hpp>

#include <memory>
#include <random>

TEST_CASE("Integration Test: SimParallelSynchronousGOMEA")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 16;
    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<IArchive> archive(new BruteforceArchive({0}));

    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax = std::make_shared<OneMax>(l);
    SimulatedFixedRuntimeObjectiveFunction sfr_onemax(onemax, 1.0);

    pop->registerGlobalData(GObjectiveFunction(&sfr_onemax));

    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(population_size, 30.0, false));
    auto sim = sp->simulator;

    SimParallelSynchronousGOMEA gomea(sp, population_size, initializer, foslearner, performance_criterion, archive);
    gomea.setPopulation(pop);
    sfr_onemax.setPopulation(pop);
    gomea.registerData();
    sfr_onemax.registerData();
    gomea.afterRegisterData();
    sfr_onemax.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 5; ++step)
        {
            gomea.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
}

TEST_CASE("Integration Test: SimParallelSynchronousMO_GOMEA")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 16;
    size_t number_of_clusters = 3;
    std::vector<size_t> objective_indices = {0, 1};

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    std::shared_ptr<IArchive> archive(new BruteforceArchive({0}));

    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax(new OneMax(l, 0));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    std::shared_ptr<Compose> compose(new Compose({onemax, zeromax}));

    SimulatedFixedRuntimeObjectiveFunction sfr_onemax(compose, 1.0);

    pop->registerGlobalData(GObjectiveFunction(&sfr_onemax));

    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(population_size, 30.0, false));
    auto sim = sp->simulator;

    SimParallelSynchronousMO_GOMEA gomea(sp,
                                         population_size,
                                         number_of_clusters,
                                         objective_indices,
                                         initializer,
                                         foslearner,
                                         performance_criterion,
                                         archive);
    gomea.setPopulation(pop);
    sfr_onemax.setPopulation(pop);
    gomea.registerData();
    sfr_onemax.registerData();
    gomea.afterRegisterData();
    sfr_onemax.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 5; ++step)
        {
            gomea.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
}

TEST_CASE("Integration Test: SimParallelSynchronousKernelGOMEA")
{
    // SimulatorParameters sp;
    // SimParallelSynchronousGOMEA spsg()
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 16;
    size_t number_of_clusters = 3;
    std::vector<size_t> objective_indices = {0, 1};

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    std::shared_ptr<IArchive> archive(new BruteforceArchive({0}));

    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax(new OneMax(l, 0));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    std::shared_ptr<Compose> compose(new Compose({onemax, zeromax}));

    SimulatedFixedRuntimeObjectiveFunction sfr_onemax(compose, 1.0);

    pop->registerGlobalData(GObjectiveFunction(&sfr_onemax));

    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(population_size, 30.0, false));
    auto sim = sp->simulator;

    SimParallelSynchronousKernelGOMEA gomea(sp,
                                            population_size,
                                            number_of_clusters,
                                            objective_indices,
                                            initializer,
                                            foslearner,
                                            performance_criterion,
                                            archive);
    gomea.setPopulation(pop);
    sfr_onemax.setPopulation(pop);
    gomea.registerData();
    sfr_onemax.registerData();
    gomea.afterRegisterData();
    sfr_onemax.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 5; ++step)
        {
            gomea.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
}

TEST_CASE("Integration Test: SimParallelAsynchronousKernelGOMEA - OneMax vs ZeroMax")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 16;
    size_t number_of_clusters = 3;
    std::vector<size_t> objective_indices = {0, 1};

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    std::shared_ptr<IArchive> archive(new BruteforceArchive({0}));

    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax(new OneMax(l, 0));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    std::shared_ptr<Compose> compose(new Compose({onemax, zeromax}));

    std::uniform_real_distribution<double> d(0.8, 1.4);
    SimulatedFunctionRuntimeObjectiveFunction sfr_onemax(
        compose, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });

    pop->registerGlobalData(GObjectiveFunction(&sfr_onemax));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/test-sim-gomea/events.jsonl", 100.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    SimParallelAsynchronousKernelGOMEA gomea(sp,
                                             population_size,
                                             number_of_clusters,
                                             objective_indices,
                                             initializer,
                                             foslearner,
                                             performance_criterion,
                                             archive);
    gomea.setPopulation(pop);
    sfr_onemax.setPopulation(pop);
    gomea.registerData();
    sfr_onemax.registerData();
    gomea.afterRegisterData();
    sfr_onemax.afterRegisterData();

    try
    {
        gomea.step();
        gomea.step();
        size_t few_size = pop->size();
        // Extra steps
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 9; ++step)
        {
            gomea.step();
            // Population size should not be increasing in an unbounded manner (after the first few steps)
            REQUIRE(pop->size() < 5 * few_size);
        }
    }
    catch (time_limit_reached &e)
    {
    }
}

TEST_CASE("Integration Test: SimParallelAsynchronousGOMEA - OneMax vs ZeroMax")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 16;
    size_t number_of_clusters = 3;
    std::vector<size_t> objective_indices = {0, 1};

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    std::shared_ptr<IArchive> archive(new BruteforceArchive({0}));

    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax(new OneMax(l, 0));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    std::shared_ptr<Compose> compose(new Compose({onemax, zeromax}));

    std::uniform_real_distribution<double> d(0.8, 1.4);
    SimulatedFunctionRuntimeObjectiveFunction sfr_onemax(
        compose, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });

    pop->registerGlobalData(GObjectiveFunction(&sfr_onemax));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/test-sim-gomea/events.jsonl", 100.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    SimParallelAsynchronousGOMEA gomea(sp,
                                       population_size,
                                       number_of_clusters,
                                       objective_indices,
                                       initializer,
                                       foslearner,
                                       performance_criterion,
                                       archive);
    gomea.setPopulation(pop);
    sfr_onemax.setPopulation(pop);
    gomea.registerData();
    sfr_onemax.registerData();
    gomea.afterRegisterData();
    sfr_onemax.afterRegisterData();

    try
    {
        gomea.step();
        gomea.step();
        size_t few_size = pop->size();
        // Extra steps
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 9; ++step)
        {
            gomea.step();
            // Population size should not be increasing in an unbounded manner (after the first few steps)
            REQUIRE(pop->size() < 5 * few_size);
        }
    }
    catch (time_limit_reached &e)
    {
    }
}

TEST_CASE("Integration Test: SimParallelSynchronousKernelGOMEA - Trap")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 16;
    size_t number_of_clusters = 3;

    std::vector<size_t> objective_indices = {0, 1};
    std::shared_ptr<ObjectiveFunction> bot0(
        new BestOfTraps(load_BestOfTraps("../instances/bestoftraps/k5/bot_n40k5fns1s42.txt")));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    std::shared_ptr<Compose> compose(new Compose({bot0, zeromax}));
    std::shared_ptr<Limiter> limiter(new Limiter(compose));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));

    std::shared_ptr<WritingSimulator> ws(
        new WritingSimulator("./results/test-sim-synchronous-gomea/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/test-sim-synchronous-gomea/archive.csv",
                      SequencedItemLogger::shared({NumEvaluationsLogger::shared(limiter),
                                                   SimulationTimeLogger::shared(sim),
                                                   ObjectiveLogger::shared(),
                                                   archive_logitem,
                                                   GenotypeCategoricalLogger::shared()})));

    std::shared_ptr<IArchive> archive(new LoggingArchive(base_archive, logger, archive_logitem));

    Rng rng(42);
    pop->registerGlobalData(rng);

    std::uniform_real_distribution<double> d(0.8, 1.4);
    SimulatedFunctionRuntimeObjectiveFunction sfr_obj(
        limiter, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });

    pop->registerGlobalData(GObjectiveFunction(&sfr_obj));

    SimParallelSynchronousKernelGOMEA gomea(sp,
                                            population_size,
                                            number_of_clusters,
                                            objective_indices,
                                            initializer,
                                            foslearner,
                                            performance_criterion,
                                            archive);
    gomea.setPopulation(pop);
    sfr_obj.setPopulation(pop);
    gomea.registerData();
    sfr_obj.registerData();
    gomea.afterRegisterData();
    sfr_obj.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 20; ++step)
        {
            gomea.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
}

TEST_CASE("Integration Test: SimParallelAsynchronousKernelGOMEA - Trap")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 16;
    size_t number_of_clusters = 3;

    std::vector<size_t> objective_indices = {0, 1};
    std::shared_ptr<ObjectiveFunction> bot0(
        new BestOfTraps(load_BestOfTraps("../instances/bestoftraps/k5/bot_n40k5fns1s42.txt")));
    std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    std::shared_ptr<Compose> compose(new Compose({bot0, zeromax}));
    std::shared_ptr<Limiter> limiter(new Limiter(compose));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/test-sim-gomea/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/test-sim-gomea/archive.csv",
                      SequencedItemLogger::shared({NumEvaluationsLogger::shared(limiter),
                                                   SimulationTimeLogger::shared(sim),
                                                   ObjectiveLogger::shared(),
                                                   archive_logitem,
                                                   GenotypeCategoricalLogger::shared()})));

    std::shared_ptr<IArchive> archive(new LoggingArchive(base_archive, logger, archive_logitem));

    Rng rng(42);
    pop->registerGlobalData(rng);

    std::uniform_real_distribution<double> d(0.8, 1.4);
    SimulatedFunctionRuntimeObjectiveFunction sfr_obj(
        limiter, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });

    pop->registerGlobalData(GObjectiveFunction(&sfr_obj));

    SimParallelAsynchronousKernelGOMEA gomea(sp,
                                             population_size,
                                             number_of_clusters,
                                             objective_indices,
                                             initializer,
                                             foslearner,
                                             performance_criterion,
                                             archive);
    gomea.setPopulation(pop);
    sfr_obj.setPopulation(pop);
    gomea.registerData();
    sfr_obj.registerData();
    gomea.afterRegisterData();
    sfr_obj.afterRegisterData();

    try
    {
        // Sidenote: # simulator steps / step has been notably increased.
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 10; ++step)
        {
            gomea.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
}

TEST_CASE("Integration Test: SimParallelSynchronousGOMEA - CPU Limit")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 32;
    size_t num_processors = 16;
    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive({0}));

    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax = std::make_shared<OneMax>(l);

    // std::shared_ptr<SimulatedFixedRuntimeObjectiveFunction> sfr_onemax(new
    // SimulatedFixedRuntimeObjectiveFunction(onemax, 1.0));

    std::shared_ptr<Limiter> limiter(new Limiter(onemax));

    std::uniform_real_distribution<double> d(0.8, 1.4);
    auto sfr_obj = std::make_shared<SimulatedFunctionRuntimeObjectiveFunction>(
        limiter, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });

    pop->registerGlobalData(GObjectiveFunction(sfr_obj.get()));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/gomea-sync-cpulim/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, num_processors, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/gomea-sync-cpulim/archive.csv",
                      SequencedItemLogger::shared({NumEvaluationsLogger::shared(limiter),
                                                   SimulationTimeLogger::shared(sim),
                                                   ObjectiveLogger::shared(),
                                                   archive_logitem,
                                                   GenotypeCategoricalLogger::shared()})));

    std::shared_ptr<IArchive> archive(new LoggingArchive(base_archive, logger, archive_logitem));

    SimParallelSynchronousGOMEA gomea(sp, population_size, initializer, foslearner, performance_criterion, archive);
    gomea.setPopulation(pop);
    sfr_obj->setPopulation(pop);
    gomea.registerData();
    sfr_obj->registerData();
    gomea.afterRegisterData();
    sfr_obj->afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 5; ++step)
        {
            gomea.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
}

TEST_CASE("Integration Test: SimParallelAsynchronousKernelGOMEA - CPU Limit")
{
    // SimulatorParameters sp;
    // SimParallelSynchronousGOMEA spsg()
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 32;
    size_t num_processors = 16;
    size_t number_of_clusters = 3;
    std::vector<size_t> objective_indices = {0};

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<LinkageMetric> metric(new NMI());
    std::shared_ptr<FoSLearner> foslearner(new CategoricalLinkageTree(metric));
    std::shared_ptr<IPerformanceCriterion> performance_criterion(
        new DominationObjectiveAcceptanceCriterion(objective_indices));
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive({0}));

    Rng rng(42);
    pop->registerGlobalData(rng);
    std::shared_ptr<ObjectiveFunction> onemax(new OneMax(l, 0));
    // std::shared_ptr<ObjectiveFunction> zeromax(new ZeroMax(l, 1));
    // std::shared_ptr<Compose> compose(new Compose({onemax, zeromax}));

    // std::shared_ptr<SimulatedFixedRuntimeObjectiveFunction> sfr_onemax(new
    // SimulatedFixedRuntimeObjectiveFunction(onemax, 1.0));
    std::shared_ptr<Limiter> limiter(new Limiter(onemax));

    std::uniform_real_distribution<double> d(0.8, 1.4);
    auto sfr_obj = std::make_shared<SimulatedFunctionRuntimeObjectiveFunction>(
        limiter, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });

    pop->registerGlobalData(GObjectiveFunction(sfr_obj.get()));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/gomea-async-cpulim/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, num_processors, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/gomea-async-cpulim/archive.csv",
                      SequencedItemLogger::shared({NumEvaluationsLogger::shared(limiter),
                                                   SimulationTimeLogger::shared(sim),
                                                   ObjectiveLogger::shared(),
                                                   archive_logitem,
                                                   GenotypeCategoricalLogger::shared()})));

    std::shared_ptr<IArchive> archive(new LoggingArchive(base_archive, logger, archive_logitem));

    SimParallelAsynchronousKernelGOMEA gomea(sp,
                                             population_size,
                                             number_of_clusters,
                                             objective_indices,
                                             initializer,
                                             foslearner,
                                             performance_criterion,
                                             archive);
    gomea.setPopulation(pop);
    sfr_obj->setPopulation(pop);
    gomea.registerData();
    sfr_obj->registerData();
    gomea.afterRegisterData();
    sfr_obj->afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 5; ++step)
        {
            gomea.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
}