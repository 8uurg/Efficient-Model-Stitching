
#include "acceptation_criteria.hpp"
#include "ga.hpp"
#include "sim-ga.hpp"
#include "initializers.hpp"
#include "problems.hpp"

#include <catch2/catch.hpp>

#include <filesystem>
#include <memory>
#include <random>

TEST_CASE("Integration Test: GA Synchronous - Trap")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Note: provided by instance below.
    // size_t l = 10;
    size_t population_size = 64;
    size_t offspring_size = 64;
    // size_t tournament_size = 2;
    
    std::filesystem::path f("./results/sga-sync");
    std::filesystem::create_directories(f);

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> bot0(
        new BestOfTraps(load_BestOfTraps("../instances/bestoftraps/k5/bot_n40k5fns1s42.txt")));
    std::shared_ptr<Limiter> limiter(new Limiter(bot0));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ICrossover> crossover(new UniformCrossover());
    std::shared_ptr<IMutation> mutation(new NoMutation());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> selection(new ShuffledSequentialSelection());
    // std::shared_ptr<ISelection> selection(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection, performance_criterion));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator(f / "events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger(f / "archive.csv",
                      SequencedItemLogger::shared({NumEvaluationsLogger::shared(limiter),
                                                   SimulationTimeLogger::shared(sim),
                                                   ObjectiveLogger::shared(),
                                                   archive_logitem,
                                                   GenotypeCategoricalLogger::shared()})));

    std::shared_ptr<IArchive> archive(new LoggingArchive(base_archive, logger, archive_logitem));

    Rng rng(42);
    pop->registerGlobalData(rng);
    int replacement_strategy = GENERATE(0, 1, 2, 3, 4, 5);

    DYNAMIC_SECTION("Replacement Strategy " << replacement_strategy)
    {
    std::uniform_real_distribution<double> d(0.8, 1.4);
    SimulatedFunctionRuntimeObjectiveFunction sfr_obj(
        limiter, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });

    pop->registerGlobalData(GObjectiveFunction(&sfr_obj));

    SimulatedSynchronousSimpleGA sga(sp,
                                population_size,
                                offspring_size,
                                replacement_strategy,
                                initializer,
                                crossover,
                                mutation,
                                selection,
                                performance_criterion,
                                archive);
    sga.setPopulation(pop);
    sfr_obj.setPopulation(pop);
    sga.registerData();
    sfr_obj.registerData();
    sga.afterRegisterData();
    sfr_obj.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 50; ++step)
        {
            std::cout << "Stepping step " << step << std::endl;
            sga.step();
            std::cout << "Stepped step " << step << std::endl;
        }
    }
    catch (time_limit_reached &e)
    {
        std::cout << "Time Limit Reached" << std::endl;
    }
    catch (stop_approach &e)
    {
        std::cout << "Approach indicated convergence" << std::endl;
    }
    }
}

TEST_CASE("Integration Test: GA Synchronous Generational - Trap")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Note: provided by instance below.
    // size_t l = 10;
    size_t population_size = 64;
    size_t offspring_size = 64;
    size_t tournament_size = 4;
    bool include_population = true;
    
    std::filesystem::path f("./results/sga-sync-gen");
    std::filesystem::create_directories(f);

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> bot0(
        new BestOfTraps(load_BestOfTraps("../instances/bestoftraps/k5/bot_n40k5fns1s42.txt")));
    std::shared_ptr<Limiter> limiter(new Limiter(bot0));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ICrossover> crossover(new UniformCrossover());
    std::shared_ptr<IMutation> mutation(new NoMutation());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> selection(new ShuffledSequentialSelection());
    // std::shared_ptr<ISelection> selection(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection, performance_criterion));

    std::shared_ptr<ISelection> tournament_contender_selection(new ShuffledSequentialSelection());
    std::shared_ptr<ISelection> generationalish_selection(new OrderedTournamentSelection(tournament_size, 1, tournament_contender_selection, performance_criterion));


    std::shared_ptr<WritingSimulator> ws(new WritingSimulator(f / "events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger(f / "archive.csv",
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

    SimulatedSynchronousSimpleGA sga(sp,
                                population_size,
                                offspring_size,
                                initializer,
                                crossover,
                                mutation,
                                selection,
                                performance_criterion,
                                archive,
                                include_population,
                                generationalish_selection);
    sga.setPopulation(pop);
    sfr_obj.setPopulation(pop);
    sga.registerData();
    sfr_obj.registerData();
    sga.afterRegisterData();
    sfr_obj.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 50; ++step)
        {
            std::cout << "Stepping step " << step << std::endl;
            sga.step();
            std::cout << "Stepped step " << step << std::endl;
        }
    }
    catch (time_limit_reached &e)
    {
        std::cout << "Time Limit Reached" << std::endl;
    }
    catch (stop_approach &e)
    {
        std::cout << "Approach indicated convergence" << std::endl;
    }
}

TEST_CASE("Integration Test: GA Asynchronous - Trap")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Note: provided by instance below.
    // size_t l = 10;
    size_t population_size = 64;
    size_t offspring_size = 64;
    // size_t tournament_size = 2;

    std::filesystem::path f("./results/sga-async");
    std::filesystem::create_directories(f);

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> bot0(
        new BestOfTraps(load_BestOfTraps("../instances/bestoftraps/k5/bot_n40k5fns1s42.txt")));
    std::shared_ptr<Limiter> limiter(new Limiter(bot0));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ICrossover> crossover(new UniformCrossover());
    std::shared_ptr<IMutation> mutation(new NoMutation());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> selection(new ShuffledSequentialSelection());
    // std::shared_ptr<ISelection> selection(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection, performance_criterion));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator(f / "events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger(f / "archive.csv",
                      SequencedItemLogger::shared({NumEvaluationsLogger::shared(limiter),
                                                   SimulationTimeLogger::shared(sim),
                                                   ObjectiveLogger::shared(),
                                                   archive_logitem,
                                                   GenotypeCategoricalLogger::shared()})));

    std::shared_ptr<IArchive> archive(new LoggingArchive(base_archive, logger, archive_logitem));

    Rng rng(42);
    pop->registerGlobalData(rng);
    int replacement_strategy = 0;

    std::uniform_real_distribution<double> d(0.8, 1.4);
    SimulatedFunctionRuntimeObjectiveFunction sfr_obj(
        limiter, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });

    pop->registerGlobalData(GObjectiveFunction(&sfr_obj));

    SimulatedAsynchronousSimpleGA sga(sp,
                                population_size,
                                offspring_size,
                                replacement_strategy,
                                initializer,
                                crossover,
                                mutation,
                                selection,
                                performance_criterion,
                                archive);
    sga.setPopulation(pop);
    sfr_obj.setPopulation(pop);
    sga.registerData();
    sfr_obj.registerData();
    sga.afterRegisterData();
    sfr_obj.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 50; ++step)
        {
            std::cout << "Stepping step " << step << std::endl;
            sga.step();
            std::cout << "Stepped step " << step << std::endl;
        }
    }
    catch (time_limit_reached &e)
    {
        std::cout << "Time Limit Reached" << std::endl;
    }
    catch (stop_approach &e)
    {
        std::cout << "Approach indicated convergence" << std::endl;
    }
}