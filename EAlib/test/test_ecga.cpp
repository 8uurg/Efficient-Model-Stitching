#include "base.hpp"
#include "ecga.hpp"
#include "ga.hpp"
#include "initializers.hpp"
#include "problems.hpp"

#include <algorithm>
#include <catch2/catch.hpp>
#include <random>

TEST_CASE("learnMPM: simple")
{
    SECTION("Full combo")
    {
        auto result = learnMPM(2,
                               {
                                   {0, 0},
                                   {0, 1},
                                   {1, 0},
                                   {1, 1},
                               });
        std::vector<std::vector<size_t>> expected = {{0}, {1}};
        // Note: sort each result such that the order is as expected.
        for (auto &r : result)
        {
            std::sort(r.begin(), r.end());
        }
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(expected));
    }
    SECTION("Equal variables")
    {
        auto result = learnMPM(2,
                               {
                                   {0, 0},
                                   {0, 0},
                                   {1, 1},
                                   {1, 1},
                               });
        std::vector<std::vector<size_t>> expected = {{0, 1}};
        // Note: sort each result such that the order is as expected.
        for (auto &r : result)
        {
            std::sort(r.begin(), r.end());
        }
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(expected));
    }
    SECTION("Inequal variables")
    {
        auto result = learnMPM(2,
                               {
                                   {0, 1},
                                   {0, 1},
                                   {1, 0},
                                   {1, 0},
                               });
        std::vector<std::vector<size_t>> expected = {{0, 1}};
        // Note: sort each result such that the order is as expected.
        for (auto &r : result)
        {
            std::sort(r.begin(), r.end());
        }
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(expected));
    }

    SECTION("Two pairs")
    {
        auto result = learnMPM(4,
                               {
                                   {0, 0, 0, 0},
                                   {1, 1, 1, 1},
                                   {0, 0, 1, 1},
                                   {1, 1, 0, 0},
                               });
        std::vector<std::vector<size_t>> expected = {{0, 1}, {2, 3}};
        // Note: sort each result such that the order is as expected.
        for (auto &r : result)
        {
            std::sort(r.begin(), r.end());
        }
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(expected));
    }

    SECTION("Triplet")
    {
        auto result =
            learnMPM(3,
                     {
                         {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1},
                         {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1},
                         {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1}, {0, 0, 0}, {1, 1, 1},
                     });
        std::vector<std::vector<size_t>> expected = {{0, 1, 2}};
        // Note: sort each result such that the order is as expected.
        for (auto &r : result)
        {
            std::sort(r.begin(), r.end());
        }
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(expected));
    }

    SECTION("Concatenated Deceptive Trap")
    {
        std::default_random_engine rng(42);
        std::uniform_real_distribution<double> u01(0, 1);

        size_t N = 1024;
        size_t block_size = 5;
        size_t l = 20;
        std::vector<std::vector<char>> input(N);
        for (size_t i = 0; i < N; ++i)
        {
            input[i].resize(l);
            std::fill(input[i].begin(), input[i].end(), 0);
            for (size_t s = 0; s < l; s += block_size)
            {
                if (u01(rng) < 1.0 / std::pow(2, block_size))
                {
                    std::fill(input[i].begin() + static_cast<int>(s),
                              input[i].begin() + static_cast<int>(std::min(l, s + block_size)),
                              1);
                }
            }
        }

        auto result = learnMPM(20, std::move(input));

        std::vector<std::vector<size_t>> expected = {
            {0, 1, 2, 3, 4}, {5, 6, 7, 8, 9}, {10, 11, 12, 13, 14}, {15, 16, 17, 18, 19}};
        // Note: sort each result such that the order is as expected.
        for (auto &r : result)
        {
            std::sort(r.begin(), r.end());
        }
        REQUIRE_THAT(result, Catch::Matchers::UnorderedEquals(expected));
    }
}

TEST_CASE("Integration Test: ECGA Synchronous - OneMax")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Note: provided by instance below.
    size_t l = 10;
    size_t population_size = 128;
    size_t tournament_size = 2;
    int replacement_strategy = 0;
    Rng rng(42);

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> onemax(
        new OneMax(l));
    std::shared_ptr<Limiter> limiter(new Limiter(onemax));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ISolutionSamplingDistribution> issd(new ECGAGreedyMarginalProduct());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> tournament_candidate_selection(new ShuffledSequentialSelection());
    std::shared_ptr<ISelection> selection(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection, performance_criterion));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/ecga-sync/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/ecga-sync/archive.csv",
                      SequencedItemLogger::shared({NumEvaluationsLogger::shared(limiter),
                                                   SimulationTimeLogger::shared(sim),
                                                   ObjectiveLogger::shared(),
                                                   archive_logitem,
                                                   GenotypeCategoricalLogger::shared()})));

    std::shared_ptr<IArchive> archive(new LoggingArchive(base_archive, logger, archive_logitem));

    pop->registerGlobalData(rng);

    std::uniform_real_distribution<double> d(0.8, 1.4);
    auto sfr_obj = std::make_shared<SimulatedFunctionRuntimeObjectiveFunction>(
        limiter, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });
    
    auto ovtrd = ObjectiveValuesToReachDetector(sfr_obj, {{-static_cast<double>(l)}});

    pop->registerGlobalData(GObjectiveFunction(&ovtrd));

    SynchronousSimulatedECGA ecga(replacement_strategy,
                                performance_criterion,
                                sp,
                                population_size,
                                issd,
                                initializer,
                                selection,
                                archive);
    ecga.setPopulation(pop);
    ovtrd.setPopulation(pop);
    ecga.registerData();
    ovtrd.registerData();
    ecga.afterRegisterData();
    ovtrd.afterRegisterData();
   
    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 20; ++step)
        {
            ecga.step();
        }
    }
    catch (vtr_reached &e)
    {
    }
}

TEST_CASE("Integration Test: ECGA Synchronous - Trap")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Note: provided by instance below.
    // size_t l = 10;
    size_t population_size = 16;
    size_t tournament_size = 2;

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> bot0(
        new BestOfTraps(load_BestOfTraps("../instances/bestoftraps/k5/bot_n40k5fns1s42.txt")));
    std::shared_ptr<Limiter> limiter(new Limiter(bot0));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ISolutionSamplingDistribution> issd(new ECGAGreedyMarginalProduct());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> tournament_candidate_selection(new ShuffledSequentialSelection());
    std::shared_ptr<ISelection> selection(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection, performance_criterion));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/ecga-sync/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/ecga-sync/archive.csv",
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

    SynchronousSimulatedECGA ecga(replacement_strategy, 
                                sp,
                                population_size,
                                issd,
                                initializer,
                                selection,
                                archive);
    ecga.setPopulation(pop);
    sfr_obj.setPopulation(pop);
    ecga.registerData();
    sfr_obj.registerData();
    ecga.afterRegisterData();
    sfr_obj.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 50; ++step)
        {
            ecga.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
    catch (stop_approach &e)
    {
    }
}

TEST_CASE("Integration Test: ECGA Synchronous - Trap - Diff")
{
    auto pop_ee = std::make_shared<Population>();
    auto pop_c = std::make_shared<Population>();

    // Note: provided by instance below.
    size_t l = 25;
    size_t population_size = 32;
    size_t tournament_size = 4;

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> bot_ee(
        new BestOfTraps(load_BestOfTraps("../instances/trap__l_25__k_5.txt")));
    std::shared_ptr<ObjectiveFunction> bot_c(
        new BestOfTraps(load_BestOfTraps("../instances/trap__l_25__k_5.txt")));
    std::shared_ptr<Limiter> limiter_ee(new Limiter(bot_ee));
    std::shared_ptr<Limiter> limiter_c(new Limiter(bot_c));

    std::shared_ptr<ISolutionInitializer> initializer_ee(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ISolutionInitializer> initializer_c(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ECGAGreedyMarginalProduct> issd_ee(new ECGAGreedyMarginalProduct());
    std::shared_ptr<ECGAGreedyMarginalProduct> issd_c(new ECGAGreedyMarginalProduct());
    std::shared_ptr<IArchive> base_archive_ee(new BruteforceArchive(objective_indices));
    std::shared_ptr<IArchive> base_archive_c(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion_ee(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<IPerformanceCriterion> performance_criterion_c(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> tournament_candidate_selection_ee(new ShuffledSequentialSelection());
    std::shared_ptr<ISelection> tournament_candidate_selection_c(new ShuffledSequentialSelection());
    std::shared_ptr<ISelection> selection_ee(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection_ee, performance_criterion_ee));
    std::shared_ptr<ISelection> selection_c(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection_c, performance_criterion_c));

    std::shared_ptr<Simulator> ws_ee(new Simulator(1000.0));
    std::shared_ptr<SimulatorParameters> sp_ee(new SimulatorParameters(ws_ee, population_size, false));
    
    std::shared_ptr<Simulator> ws_c(new Simulator(1000.0));
    std::shared_ptr<SimulatorParameters> sp_c(new SimulatorParameters(ws_c, population_size, false));
    
    auto sim_ee = sp_ee->simulator;

    std::shared_ptr<IArchive> archive_ee = base_archive_ee;
    std::shared_ptr<IArchive> archive_c = base_archive_c;

    Rng rng_ee(42);
    Rng rng_c(42);

    pop_ee->registerGlobalData(rng_ee);
    pop_c->registerGlobalData(rng_c);
    int replacement_strategy = 0;

    SimulatedFunctionRuntimeObjectiveFunction sfr_obj_c(
        limiter_c, [](Population & /*pop*/, Individual & /*ii*/) 
        {
            return 1.0;
            });
    SimulatedFunctionRuntimeObjectiveFunction sfr_obj_ee(
        limiter_ee, [](Population & pop, Individual & ii) 
        {
            size_t gs = 0;
            auto &g = pop.getData<GenotypeCategorical>(ii);
            for (auto gv: g.genotype)
            {
                gs += gv;
            }
            return 1.0 + static_cast<double>(gs) / static_cast<double>(g.genotype.size());
            });

    pop_ee->registerGlobalData(GObjectiveFunction(&sfr_obj_ee));
    pop_c->registerGlobalData(GObjectiveFunction(&sfr_obj_c));

    SynchronousSimulatedECGA ecga_ee(replacement_strategy, 
                                sp_ee,
                                population_size,
                                issd_ee,
                                initializer_ee,
                                selection_ee,
                                archive_ee);
    ecga_ee.setPopulation(pop_ee);
    sfr_obj_ee.setPopulation(pop_ee);
    ecga_ee.registerData();
    sfr_obj_ee.registerData();
    ecga_ee.afterRegisterData();
    sfr_obj_ee.afterRegisterData();

    SynchronousSimulatedECGA ecga_c(replacement_strategy, 
                                sp_c,
                                population_size,
                                issd_c,
                                initializer_c,
                                selection_c,
                                archive_c);
    ecga_c.setPopulation(pop_c);
    sfr_obj_c.setPopulation(pop_c);
    ecga_c.registerData();
    sfr_obj_c.registerData();
    ecga_c.afterRegisterData();
    sfr_obj_c.afterRegisterData();

    try
    {
        // Behavior for the first two generations should be the same:
        // Initialization with the same seed should cause the same population.
        // Following that, the model learnt from this population should be identical.
        // Solutions evaluated from this model should be identical.
        // The order in which they finish evaluating is different: this can now vary.
        // Solutions with shorter evaluation times will end up on top (because faster)
        // While solutions with longer evaluation times should end up on the bottom.
        // The only difference should be the ordering: sorting both populations on genotype should
        // bring them back together.
        for (size_t step = 0; step < 2; ++step)
        {
            ecga_ee.step();
            ecga_c.step();

            auto &ee_pop = ecga_ee.getSolutionPopulation();
            auto &c_pop  = ecga_c.getSolutionPopulation();
            std::sort(ee_pop.begin(), ee_pop.end(), [&pop_ee](Individual &a, Individual b){
                auto &ga = pop_ee->getData<GenotypeCategorical>(a);
                auto &gb = pop_ee->getData<GenotypeCategorical>(b);
                return std::lexicographical_compare(ga.genotype.begin(), ga.genotype.end(), gb.genotype.begin(), ga.genotype.end());
            });
            std::sort(c_pop.begin(), c_pop.end(), [&pop_c](Individual &a, Individual b){
                auto &ga = pop_c->getData<GenotypeCategorical>(a);
                auto &gb = pop_c->getData<GenotypeCategorical>(b);
                return std::lexicographical_compare(ga.genotype.begin(), ga.genotype.end(), gb.genotype.begin(), ga.genotype.end());
            });
            auto are_equal = std::equal(ee_pop.begin(), ee_pop.end(), c_pop.begin(), c_pop.end(), [&pop_c, &pop_ee](Individual &a, Individual &b) {
                auto &ga = pop_ee->getData<GenotypeCategorical>(a);
                auto &gb = pop_c->getData<GenotypeCategorical>(b);
                return std::equal(ga.genotype.begin(), ga.genotype.end(), gb.genotype.begin(), gb.genotype.end());
            });

            if (!are_equal)
            {
                for (size_t idx = 0; idx < population_size; ++idx)
                {
                    auto &g_ee = pop_ee->getData<GenotypeCategorical>(ee_pop[idx]);
                    auto &g_c = pop_c->getData<GenotypeCategorical>(c_pop[idx]);
                    for (size_t vidx = 0; vidx < l; ++vidx)
                    {
                        std::cout << static_cast<int>(g_ee.genotype[vidx]) << '|' << static_cast<int>(g_c.genotype[vidx]) << ' ';
                    }
                    std::cout << std::endl;
                }
            }

            REQUIRE(are_equal);
        }
    }
    catch (time_limit_reached &e)
    {
    }
    catch (stop_approach &e)
    {
    }
}

TEST_CASE("Integration Test: ECGA Asynchronous - Trap")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Note: provided by instance below.
    // size_t l = 10;
    size_t population_size = 16;
    size_t tournament_size = 2;

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> bot0(
        new BestOfTraps(load_BestOfTraps("../instances/bestoftraps/k5/bot_n40k5fns1s42.txt")));
    std::shared_ptr<Limiter> limiter(new Limiter(bot0));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ISolutionSamplingDistribution> issd(new ECGAGreedyMarginalProduct());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> tournament_candidate_selection(new ShuffledSequentialSelection());
    std::shared_ptr<ISelection> selection(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection, performance_criterion));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/ecga-async/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, population_size, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/ecga-async/archive.csv",
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

    AsynchronousSimulatedECGA ecga(replacement_strategy,
                                performance_criterion,
                                sp,
                                population_size,
                                issd,
                                initializer,
                                selection,
                                archive);
    ecga.setPopulation(pop);
    sfr_obj.setPopulation(pop);
    ecga.registerData();
    sfr_obj.registerData();
    ecga.afterRegisterData();
    sfr_obj.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 10000; ++step)
        {
            ecga.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
    catch (stop_approach &e)
    {
    }
}

TEST_CASE("Integration Test: ECGA Synchronous - OneMax - CPU Limit")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Note: provided by instance below.
    size_t l = 10;
    size_t population_size = 128;
    size_t num_processors = 64;
    size_t tournament_size = 2;
    int replacement_strategy = 0;
    Rng rng(42);

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> onemax(
        new OneMax(l));
    std::shared_ptr<Limiter> limiter(new Limiter(onemax));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ISolutionSamplingDistribution> issd(new ECGAGreedyMarginalProduct());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> tournament_candidate_selection(new ShuffledSequentialSelection());
    std::shared_ptr<ISelection> selection(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection, performance_criterion));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/ecga-sync-cpulim/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, num_processors, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/ecga-sync-cpulim/archive.csv",
                      SequencedItemLogger::shared({NumEvaluationsLogger::shared(limiter),
                                                   SimulationTimeLogger::shared(sim),
                                                   ObjectiveLogger::shared(),
                                                   archive_logitem,
                                                   GenotypeCategoricalLogger::shared()})));

    std::shared_ptr<IArchive> archive(new LoggingArchive(base_archive, logger, archive_logitem));

    pop->registerGlobalData(rng);

    std::uniform_real_distribution<double> d(0.8, 1.4);
    auto sfr_obj = std::make_shared<SimulatedFunctionRuntimeObjectiveFunction>(
        limiter, [&d, &rng](Population & /* pop */, Individual & /* ii */) { return d(rng.rng); });
    
    auto ovtrd = ObjectiveValuesToReachDetector(sfr_obj, {{-static_cast<double>(l)}});

    pop->registerGlobalData(GObjectiveFunction(&ovtrd));

    SynchronousSimulatedECGA ecga(replacement_strategy,
                                performance_criterion,
                                sp,
                                population_size,
                                issd,
                                initializer,
                                selection,
                                archive);
    ecga.setPopulation(pop);
    ovtrd.setPopulation(pop);
    ecga.registerData();
    ovtrd.registerData();
    ecga.afterRegisterData();
    ovtrd.afterRegisterData();
   
    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 20; ++step)
        {
            ecga.step();
        }
    }
    catch (vtr_reached &e)
    {
    }
}

TEST_CASE("Integration Test: ECGA Asynchronous - OneMax - CPU Limit")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    size_t l = 10;
    size_t population_size = 128;
    size_t num_processors = 64;
    size_t tournament_size = 2;

    std::vector<size_t> objective_indices = {0};
    std::shared_ptr<ObjectiveFunction> om(
        new OneMax(l));
    std::shared_ptr<Limiter> limiter(new Limiter(om));

    std::shared_ptr<ISolutionInitializer> initializer(new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<ISolutionSamplingDistribution> issd(new ECGAGreedyMarginalProduct());
    std::shared_ptr<IArchive> base_archive(new BruteforceArchive(objective_indices));
    
    std::shared_ptr<IPerformanceCriterion> performance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ISelection> tournament_candidate_selection(new ShuffledSequentialSelection());
    std::shared_ptr<ISelection> selection(new OrderedTournamentSelection(tournament_size, 1, tournament_candidate_selection, performance_criterion));

    std::shared_ptr<WritingSimulator> ws(new WritingSimulator("./results/ecga-async-cpulim/events.jsonl", 1000.0));
    std::shared_ptr<SimulatorParameters> sp(new SimulatorParameters(ws, num_processors, false));
    auto sim = sp->simulator;

    auto archive_logitem = ArchivedLogger::shared();

    std::shared_ptr<CSVLogger> logger(
        new CSVLogger("./results/ecga-async-cpulim/archive.csv",
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

    AsynchronousSimulatedECGA ecga(replacement_strategy,
                                performance_criterion,
                                sp,
                                population_size,
                                issd,
                                initializer,
                                selection,
                                archive);
    ecga.setPopulation(pop);
    sfr_obj.setPopulation(pop);
    ecga.registerData();
    sfr_obj.registerData();
    ecga.afterRegisterData();
    sfr_obj.afterRegisterData();

    try
    {
        // Note to self: convergence likely happens very quickly.
        for (size_t step = 0; step < 10000; ++step)
        {
            ecga.step();
        }
    }
    catch (time_limit_reached &e)
    {
    }
    catch (stop_approach &e)
    {
    }
}