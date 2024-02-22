#include "acceptation_criteria.hpp"
#include "base.hpp"
#include "ga.hpp"
#include "initializers.hpp"
#include "problems.hpp"

#include <algorithm>
#include <catch2/catch.hpp>
#include <memory>
#include <vector>

TEST_CASE("Uniform Crossover", "[Operator][Crossover]")
{
    auto pop = std::make_shared<Population>();

    UniformCrossover ux(0.5);
    ux.setPopulation(pop);
    ux.registerData();

    SECTION("Uniform Crossover requires the categorical genotype to be present - absent case")
    {
        // Note: It can also be used with a continuous genotype,
        //       but in that context generally different operators are used.
        REQUIRE_THROWS(ux.afterRegisterData());
    }

    // Register genotype
    pop->registerData<GenotypeCategorical>();

    SECTION("Uniform Crossover requires a random number generator to be present - absent case")
    {
        REQUIRE_THROWS(ux.afterRegisterData());
    }

    Rng rng(42);
    pop->registerGlobalData(rng);

    // Now things should be fine
    CHECK_NOTHROW(ux.afterRegisterData());

    // String length
    size_t l = 100;

    // Create some individuals
    Individual a = pop->newIndividual();
    Individual b = pop->newIndividual();

    auto &genotype_a = pop->getData<GenotypeCategorical>(a);
    genotype_a.genotype.resize(l);
    std::fill(genotype_a.genotype.begin(), genotype_a.genotype.end(), 0);

    auto &genotype_b = pop->getData<GenotypeCategorical>(b);
    genotype_b.genotype.resize(l);
    std::fill(genotype_b.genotype.begin(), genotype_b.genotype.end(), 1);

    SECTION("Uniform Crossover is binary")
    {
        std::vector<Individual> parents = {a, a, a};
        REQUIRE_THROWS(ux.crossover(parents));
    }

    SECTION("Uniform Crossover with itself always return copies")
    {
        std::vector<Individual> parents = {a, a};
        auto result = ux.crossover(parents);
        CHECK(result.size() == 2);
        for (auto r : result)
        {
            auto gc = pop->getData<GenotypeCategorical>(r);
            REQUIRE(gc.genotype.size() == l);
            CHECK_THAT(gc.genotype, Catch::Matchers::Equals(genotype_a.genotype));
        }
    }

    SECTION("Uniform Crossover should have the option to return both values given opposite strings")
    {
        // Note that this test has a one in 2^l chance of failing, but it should not fail consistently.

        std::vector<Individual> parents = {a, b};
        auto result = ux.crossover(parents);
        for (auto r : result)
        {
            auto gc = pop->getData<GenotypeCategorical>(r);
            REQUIRE(gc.genotype.size() == l);
            CHECK_THAT(gc.genotype, Catch::Matchers::VectorContains((char)0));
            CHECK_THAT(gc.genotype, Catch::Matchers::VectorContains((char)1));
        }
    }

    SECTION("Uniform Crossover should return offspring that are opposites if parents are opposites")
    {
        std::vector<Individual> parents = {a, b};
        auto result = ux.crossover(parents);
        auto ca = pop->getData<GenotypeCategorical>(result[0]);
        auto cb = pop->getData<GenotypeCategorical>(result[1]);
        for (size_t i = 0; i < l; ++i)
        {
            REQUIRE(ca.genotype[i] != cb.genotype[i]);
        }
    }
}

TEST_CASE("K-point Crossover", "[Operator][Crossover]")
{
    auto pop = std::make_shared<Population>();
    size_t k = GENERATE(1, 2, 4);

    KPointCrossover ux(k);
    ux.setPopulation(pop);
    ux.registerData();

    SECTION("K-point Crossover requires the categorical genotype to be present - absent case")
    {
        // Note: It can also be used with a continuous genotype,
        //       but in that context generally different operators are used.
        REQUIRE_THROWS(ux.afterRegisterData());
    }

    // Register genotype
    pop->registerData<GenotypeCategorical>();

    SECTION("K-point Crossover requires a random number generator to be present - absent case")
    {
        REQUIRE_THROWS(ux.afterRegisterData());
    }

    Rng rng(42);
    pop->registerGlobalData(rng);

    // Now things should be fine
    CHECK_NOTHROW(ux.afterRegisterData());

    // String length
    size_t l = 100;

    // Create some individuals
    Individual a = pop->newIndividual();
    auto &genotype_a = pop->getData<GenotypeCategorical>(a);
    genotype_a.genotype.resize(l);
    std::fill(genotype_a.genotype.begin(), genotype_a.genotype.end(), 0);

    Individual b = pop->newIndividual();
    auto &genotype_b = pop->getData<GenotypeCategorical>(b);
    genotype_b.genotype.resize(l);
    std::fill(genotype_b.genotype.begin(), genotype_b.genotype.end(), 1);

    SECTION("k-Point Crossover with itself always return copies")
    {
        std::vector<Individual> parents = {a, a};
        auto result = ux.crossover(parents);
        CHECK(result.size() == 2);
        for (auto r : result)
        {
            auto gc = pop->getData<GenotypeCategorical>(r);
            REQUIRE(gc.genotype.size() == l);
            CHECK_THAT(gc.genotype, Catch::Matchers::Equals(genotype_a.genotype));
        }
    }

    SECTION("k-Point Crossover should return offspring that are opposites if parents are opposites")
    {
        std::vector<Individual> parents = {a, b};
        auto result = ux.crossover(parents);
        auto ca = pop->getData<GenotypeCategorical>(result[0]);
        auto cb = pop->getData<GenotypeCategorical>(result[1]);
        for (size_t i = 0; i < l; ++i)
        {
            REQUIRE(ca.genotype[i] != cb.genotype[i]);
        }
    }

    SECTION("Results of k-Point Crossover of two opposite strings will contain k switchovers")
    {
        std::vector<Individual> parents = {a, b};
        auto result = ux.crossover(parents);
        CHECK(result.size() == 2);

        for (Individual o : result)
        {
            size_t count_switches = 0;
            char last = -1;
            auto go = pop->getData<GenotypeCategorical>(o);
            for (char g : go.genotype)
            {
                if (g != last)
                    ++count_switches;
                last = g;
            }
            // Note: initial character is counted as a switch by the code above as well
            //       so the expected count obtained is actually one higher.
            REQUIRE(count_switches == k + 1);
        }
    }
}

TEST_CASE("Subfunction Crossover", "[Operator][Crossover]")
{
    auto pop = std::make_shared<Population>();
    // String length
    size_t l = 100;

    SubfunctionCrossover ux(0.5);
    ux.setPopulation(pop);
    ux.registerData();

    SECTION("Subfunction Crossover requires the categorical genotype to be present - absent case")
    {
        // Note: It can also be used with a continuous genotype,
        //       but in that context generally different operators are used.
        REQUIRE_THROWS(ux.afterRegisterData());
    }

    // Register genotype
    pop->registerData<GenotypeCategorical>();

    // Register subfunctions to use
    Subfunctions subfns;
    size_t block_size = 5; // Assumption: l is divisible by block_size.
    for (size_t idx = 0; idx < l; idx += block_size)
    {
        std::vector<size_t> indices(block_size);
        std::iota(indices.begin(), indices.end(), idx);
        subfns.subfunctions.push_back(indices);
    }
    pop->registerGlobalData(subfns);

    SECTION("Subfunction Crossover requires a random number generator to be present - absent case")
    {
        REQUIRE_THROWS(ux.afterRegisterData());
    }

    Rng rng(42);
    pop->registerGlobalData(rng);

    // Now things should be fine
    CHECK_NOTHROW(ux.afterRegisterData());

    // Create some individuals
    Individual a = pop->newIndividual();
    Individual b = pop->newIndividual();

    auto &genotype_a = pop->getData<GenotypeCategorical>(a);
    genotype_a.genotype.resize(l);
    std::fill(genotype_a.genotype.begin(), genotype_a.genotype.end(), 0);

    auto &genotype_b = pop->getData<GenotypeCategorical>(b);
    genotype_b.genotype.resize(l);
    std::fill(genotype_b.genotype.begin(), genotype_b.genotype.end(), 1);

    SECTION("Subfunction Crossover is binary")
    {
        std::vector<Individual> parents = {a, a, a};
        REQUIRE_THROWS(ux.crossover(parents));
    }

    SECTION("Subfunction Crossover with itself always return copies")
    {
        std::vector<Individual> parents = {a, a};
        auto result = ux.crossover(parents);
        CHECK(result.size() == 2);
        for (auto r : result)
        {
            auto gc = pop->getData<GenotypeCategorical>(r);
            REQUIRE(gc.genotype.size() == l);
            CHECK_THAT(gc.genotype, Catch::Matchers::Equals(genotype_a.genotype));
        }
    }

    SECTION("Subfunction Crossover exchanges subfunctions")
    {
        // Note: this check assumes the subfunctions do not overlap
        // This becomes slightly more difficult to check otherwise...
        std::vector<Individual> parents = {a, b};
        auto result = ux.crossover(parents);
        for (auto r : result)
        {
            auto gc = pop->getData<GenotypeCategorical>(r);
            for (auto &subfn : subfns.subfunctions)
            {
                auto ref = gc.genotype[subfn[0]];
                for (auto v : subfn)
                {
                    CHECK(ref == gc.genotype[v]);
                }
            }
        }
    }

    SECTION("Subfunction Crossover should return offspring that are opposites if parents are opposites")
    {
        std::vector<Individual> parents = {a, b};
        auto result = ux.crossover(parents);
        auto ca = pop->getData<GenotypeCategorical>(result[0]);
        auto cb = pop->getData<GenotypeCategorical>(result[1]);
        for (size_t i = 0; i < l; ++i)
        {
            REQUIRE(ca.genotype[i] != cb.genotype[i]);
        }
    }
}

TEST_CASE("No Mutation", "[Operator][Mutation]")
{
    auto pop = std::make_shared<Population>();
    pop->registerData<GenotypeCategorical>();
    NoMutation nm;
    nm.setPopulation(pop);
    nm.registerData();
    CHECK_NOTHROW(nm.afterRegisterData());

    // String length
    size_t l = 100;

    // Create some individuals
    Individual a = pop->newIndividual();
    auto &genotype_a = pop->getData<GenotypeCategorical>(a);
    genotype_a.genotype.resize(l);
    std::fill(genotype_a.genotype.begin(), genotype_a.genotype.end(), 0);

    Individual b = pop->newIndividual();
    auto &genotype_b = pop->getData<GenotypeCategorical>(b);
    genotype_b.genotype.resize(l);
    std::fill(genotype_b.genotype.begin(), genotype_b.genotype.end(), 1);

    SECTION("No mutation should not alter the data associated with individuals")
    {
        std::vector<Individual> example = {a, b};
        std::vector<char> copy_genotype_a = genotype_a.genotype;
        std::vector<char> copy_genotype_b = genotype_b.genotype;
        nm.mutate(example);
        REQUIRE_THAT(genotype_a.genotype, Catch::Matchers::Equals(copy_genotype_a));
        REQUIRE_THAT(genotype_b.genotype, Catch::Matchers::Equals(copy_genotype_b));
    }
}

TEST_CASE("RandomSelection", "[Operator][Selection]")
{
    auto pop = std::make_shared<Population>();
    RandomSelection randomSelection;
    Rng rng(42);

    randomSelection.setPopulation(pop);

    randomSelection.registerData();
    SECTION("Rng should be present")
    {
        REQUIRE_THROWS(randomSelection.afterRegisterData());
    }
    pop->registerGlobalData(rng);

    randomSelection.afterRegisterData();

    // This one should not get selected
    Individual skipped = pop->newIndividual();

    // Create a population.
    size_t population_size = 16;
    std::vector<Individual> subpopulation(population_size);
    std::generate(subpopulation.begin(), subpopulation.end(), [&pop]() { return pop->newIndividual(); });

    auto selection = randomSelection.select(subpopulation, population_size);
    SECTION("Should not select skipped.")
    {
        for (auto s : selection)
        {
            REQUIRE(s.i != skipped.i);
        }
    }
    SECTION("Should select from provided subpopulation.")
    {
        for (auto s : selection)
        {
            REQUIRE((s.i >= 1 && s.i <= 16));
        }
    }
}

TEST_CASE("RandomUniqueSelection", "[Operator][Selection]")
{
    auto pop = std::make_shared<Population>();
    RandomUniqueSelection randomSelection;
    Rng rng(42);

    randomSelection.setPopulation(pop);

    SECTION("Rng should be present")
    {
        randomSelection.registerData();
        REQUIRE_THROWS(randomSelection.afterRegisterData());
    }
    pop->registerGlobalData(rng);
    randomSelection.registerData();

    randomSelection.afterRegisterData();

    // This one should not get selected
    Individual skipped = pop->newIndividual();

    // Create a population.
    size_t population_size = 16;
    std::vector<Individual> subpopulation(population_size);
    std::generate(subpopulation.begin(), subpopulation.end(), [&pop]() { return pop->newIndividual(); });

    auto selection = randomSelection.select(subpopulation, population_size);
    SECTION("Should not select skipped.")
    {
        for (auto s : selection)
        {
            REQUIRE(s.i != skipped.i);
        }
    }
    SECTION("Should select from provided subpopulation.")
    {
        for (auto s : selection)
        {
            REQUIRE((s.i >= 1 && s.i <= 16));
        }
    }
    SECTION("Selected items should be unique")
    {
        std::sort(selection.begin(), selection.end(), [](Individual &a, Individual &b) { return a.i < b.i; });
        auto new_end =
            std::unique(selection.begin(), selection.end(), [](Individual &a, Individual &b) { return a.i == b.i; });
        selection.erase(new_end, selection.end());
        REQUIRE(selection.size() == population_size);
    }
}

TEST_CASE("ShuffledSequentialSelection", "[Operator][Selection]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    ShuffledSequentialSelection selection;
    Rng rng(42);

    selection.setPopulation(pop);

    SECTION("Rng should required to be registered")
    {
        selection.registerData();
        REQUIRE_THROWS(selection.afterRegisterData());
    }

    selection.registerData();
    pop->registerGlobalData(rng);
    selection.afterRegisterData();

    size_t population_size = 8;
    std::vector<Individual> subpopulation(population_size);
    std::generate(subpopulation.begin(), subpopulation.end(), [&pop]() { return pop->newIndividual(); });

    SECTION(("should be unique across selection calls given that the population has not been exhausted"))
    {
        std::vector<size_t> count(population_size);

        for (size_t c = 0; c < population_size; c += 2)
        {
            auto subset = selection.select(subpopulation, 2);
            for (auto i : subset)
                count[i.i] += 1;
        }

        // If the step divides the population size nicely, all should be equal to 1!
        for (size_t c : count)
            REQUIRE(c == 1);
    }
    SECTION("if the population is exhaused, some samples will repeat")
    {
        size_t over = 2;
        std::vector<size_t> count(population_size);

        for (size_t c = 0; c < population_size + over; c += 2)
        {
            auto subset = selection.select(subpopulation, 2);
            for (auto i : subset)
                count[i.i] += 1;
        }

        size_t count_one = 0;
        size_t count_two = 0;
        for (size_t c : count)
        {
            REQUIRE((c == 1 || c == 2));
            if (c == 1)
                ++count_one;
            if (c == 2)
                ++count_two;
        }

        REQUIRE(count_one == population_size - over);
        REQUIRE(count_two == over);
    }
    SECTION("the order of samples should be random over each generation")
    {
        // Get a initial ordering
        auto subset_a = selection.select(subpopulation, population_size);
        // Get a second ordering
        auto subset_b = selection.select(subpopulation, population_size);
        // Note: Technically there are 8! possible permutations
        // i.e. there is a 1 / 40320 chance that we get the same ordering twice.
        // Let us assume this does not occur (unlikely if properly implemented)
        // If it does: maybe try changing the random seed above to be less unlucky.
        REQUIRE_FALSE(std::equal(
            subset_a.begin(), subset_a.end(), subset_b.begin(), subset_b.end(), [](Individual &a, Individual &b) {
                return a.i == b.i;
            }));
    }
}

TEST_CASE("TruncationSelection", "[Operator][Selection]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    pop->registerData<Objective>();
    auto tgo = pop->getDataContainer<Objective>();
    std::shared_ptr<IPerformanceCriterion> pc = std::make_shared<SingleObjectiveAcceptanceCriterion>();

    TruncationSelection selection(pc);
    Rng rng(42);

    selection.setPopulation(pop);

    SECTION("Rng should required to be registered")
    {
        selection.registerData();
        REQUIRE_THROWS(selection.afterRegisterData());
    }

    selection.registerData();
    pop->registerGlobalData(rng);
    selection.afterRegisterData();

    size_t population_size = 8;
    std::vector<Individual> subpopulation(population_size);
    std::generate(subpopulation.begin(), subpopulation.end(), [&pop]() { return pop->newIndividual(); });

    SECTION("should return the k-best solutions without duplicates (already sorted)")
    {
        for (size_t k = 1; k < population_size; ++k)
        {
            for (size_t idx = 0; idx < population_size; ++idx)
        {
            tgo.getData(subpopulation[idx]).objectives = { static_cast<double>(idx) };
        }

            std::vector<Individual> selected = selection.select(subpopulation, k);
            // Sort, as the actual ordering of the selected items is not specified.
            std::sort(selected.begin(), selected.end(), [&tgo](Individual &a , Individual &b) {
                return tgo.getData(a).objectives[0] < tgo.getData(b).objectives[0];
            });

            REQUIRE(selected.size() == k);
            for (size_t idx = 0; idx < k; ++idx)
            {
                CHECK(tgo.getData(selected[idx]).objectives[0] == idx);
            }
        }
    }

    SECTION("should return the k-best solutions without duplicates (reversed)")
    {
        for (size_t k = 1; k < population_size; ++k)
        {
            for (size_t idx = 0; idx < population_size; ++idx)
            {
                tgo.getData(subpopulation[idx]).objectives = { static_cast<double>(population_size - idx - 1) };
            }

            std::vector<Individual> selected = selection.select(subpopulation, k);
            // Sort, as the actual ordering of the selected items is not specified.
            std::sort(selected.begin(), selected.end(), [&tgo](Individual &a , Individual &b) {
                return tgo.getData(a).objectives[0] < tgo.getData(b).objectives[0];
            });

            REQUIRE(selected.size() == k);
            for (size_t idx = 0; idx < k; ++idx)
            {
                CHECK(tgo.getData(selected[idx]).objectives[0] == idx);
            }
        }
    }
}

TEST_CASE("Standard GA", "[GA]")
{
    size_t l = 10;
    size_t population_size = 16;
    auto eval_function = [](std::vector<char> &) { return 0; };
    std::vector<char> alphabet_size(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    std::shared_ptr<Population> pop = std::make_shared<Population>();
    //
    Rng rng(42);
    pop->registerGlobalData(rng);

    // Set up some parameters
    std::shared_ptr<DiscreteObjectiveFunction> objective_function(
        new DiscreteObjectiveFunction(eval_function, l, alphabet_size));
    std::shared_ptr<CategoricalProbabilisticallyCompleteInitializer> initializer(
        new CategoricalProbabilisticallyCompleteInitializer());
    std::shared_ptr<UniformCrossover> crossover(new UniformCrossover());
    std::shared_ptr<NoMutation> mutation(new NoMutation());
    std::shared_ptr<RandomUniqueSelection> parent_selection(new RandomUniqueSelection());
    std::shared_ptr<SingleObjectiveAcceptanceCriterion> acceptance_criterion(new SingleObjectiveAcceptanceCriterion());
    std::shared_ptr<ShuffledSequentialSelection> tournament_individual_selection(new ShuffledSequentialSelection());
    std::shared_ptr<OrderedTournamentSelection> population_selection(
        new OrderedTournamentSelection(2, 1, parent_selection, acceptance_criterion));

    SimpleGA ga(population_size,
                initializer,
                crossover,
                mutation,
                parent_selection,
                population_selection,
                acceptance_criterion);

    SECTION("stepping without initializing & registering should lead to an exception")
    {
        REQUIRE_THROWS(ga.step());
    }

    // Set populations
    ga.setPopulation(pop);
    objective_function->setPopulation(pop);
    ga.registerData();
    objective_function->registerData();
    ga.afterRegisterData();
    objective_function->afterRegisterData();
    pop->registerGlobalData(GObjectiveFunction(&*objective_function));

    // Perform the first step
    ga.step();

    SECTION("first step should preserve population size")
    {
        REQUIRE(ga.getSolutionPopulation().size() == population_size);
    }

    SECTION("first step should initialize")
    {
        for (auto ii : ga.getSolutionPopulation())
        {
            GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
            REQUIRE(genotype.genotype.size() == l);
        }
    }

    ga.step();

    // Second step should start recombining and create offspring.
    SECTION("second step should preserve population size")
    {
        REQUIRE(ga.getSolutionPopulation().size() == population_size);
    }

    ga.step();

    // Third step for good measure, as things may go wrong at this point.
    SECTION("third step should preserve population size")
    {
        REQUIRE(ga.getSolutionPopulation().size() == population_size);
    }
}