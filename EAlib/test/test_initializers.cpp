#include <catch2/catch.hpp>
#include <optional>
#include "initializers.hpp"

TEST_CASE("CategoricalUniformInitializer", "[Operator][Initialization]")
{
    size_t l = 10;
    size_t population_size = 128;

    std::vector<char> alphabet_size;
    alphabet_size.resize(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    auto population = std::make_shared<Population>();
    // Set objective
    auto init = CategoricalUniformInitializer();
    auto rng = Rng(42);
    // Perform registration routine.
    init.setPopulation(population);
    population->registerGlobalData(GenotypeCategoricalData { l, alphabet_size } );
    init.registerData();
    population->registerData<GenotypeCategorical>();
    population->registerGlobalData(rng);
    init.afterRegisterData();
    std::vector<Individual> iis(population_size);
    std::generate(iis.begin(), iis.end(), [&population]() { return population->newIndividual(); });
    init.initialize(iis);

    SECTION("All solutions should be of the right size")
    {
        for (auto ii : iis)
        {
            GenotypeCategorical genotype = population->getData<GenotypeCategorical>(ii);
            REQUIRE(genotype.genotype.size() == l);
        }
    }

    SECTION("All solutions should contain both values")
    {
        for (size_t v = 0; v < l; v++)
        {
            bool seen_0 = false;
            bool seen_1 = false;

            for (auto ii : iis)
            {
                GenotypeCategorical genotype = population->getData<GenotypeCategorical>(ii);
                if (genotype.genotype[v] == 0)
                    seen_0 = true;
                if (genotype.genotype[v] == 1)
                    seen_1 = true;

                REQUIRE((genotype.genotype[v] == 0) | (genotype.genotype[v] == 1));

                if (seen_0 && seen_1)
                    return;
            }

            REQUIRE(seen_0 & seen_1);
        }
    }
}

TEST_CASE("CategoricalWeightedInitializer", "[Operator][Initialization]")
{
    SECTION("Samples according to the probability distribution")
    {
        auto init = CategoricalWeightedInitializer({
            {0.5, 0.5, 0.0},
            {0.0, 0.5, 0.5},
        });

        CHECK(init.sample_gene(0.0, 0) == 0);
        CHECK(init.sample_gene(0.1, 0) == 0);
        CHECK(init.sample_gene(0.3, 0) == 0);
        CHECK(init.sample_gene(0.6, 0) == 1);
        CHECK(init.sample_gene(0.9, 0) == 1);
        CHECK(init.sample_gene(1.0, 0) == 1);
    }

    SECTION("Initializes the population")
    {
        size_t l = 2;
        size_t population_size = 128;

        std::vector<char> alphabet_size;
        alphabet_size.resize(l);
        std::fill(alphabet_size.begin(), alphabet_size.end(), 3);

        auto population = std::make_shared<Population>();
        // Set objective
        auto init = CategoricalWeightedInitializer({
            {0.5, 0.5, 0.0},
            {0.5, 0.5, 0.0},
        });
        auto rng = Rng(42);
        // Perform registration routine.
        init.setPopulation(population);
        population->registerGlobalData(GenotypeCategoricalData { l, alphabet_size } );
        init.registerData();
        population->registerData<GenotypeCategorical>();
        population->registerGlobalData(rng);
        init.afterRegisterData();
        std::vector<Individual> iis(population_size);
        std::generate(iis.begin(), iis.end(), [&population]() { return population->newIndividual(); });
        init.initialize(iis);

        SECTION("All solutions should be of the right size")
        {
            for (auto ii : iis)
            {
                GenotypeCategorical genotype = population->getData<GenotypeCategorical>(ii);
                REQUIRE(genotype.genotype.size() == l);
            }
        }

        SECTION("There should be a 0 and 1 possible for these variables, but no 2's")
        {
            for (size_t v = 0; v < l; v++)
            {
                bool seen_0 = false;
                bool seen_1 = false;
                bool seen_2 = false;

                for (auto ii : iis)
                {
                    GenotypeCategorical genotype = population->getData<GenotypeCategorical>(ii);
                    if (genotype.genotype[v] == 0)
                        seen_0 = true;
                    if (genotype.genotype[v] == 1)
                        seen_1 = true;
                    if (genotype.genotype[v] == 2)
                        seen_2 = true;
                }

                REQUIRE((seen_0 && seen_1 && !seen_2));
            }
        }
    }
}

TEST_CASE("CategoricalProbabilisticallyCompleteInitializer", "[Operator][Initialization]")
{
    SECTION("Initializes uniformly when not provided with probabilities")
    {
        size_t l = 10;
        size_t population_size = 128;

        std::vector<char> alphabet_size;
        alphabet_size.resize(l);
        std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

        auto population = std::make_shared<Population>();
        // Set objective
        population->registerGlobalData(GenotypeCategoricalData { l, alphabet_size } );
        auto init = CategoricalProbabilisticallyCompleteInitializer();
        auto rng = Rng(42);
        // Perform registration routine.
        init.setPopulation(population);
        init.registerData();
        population->registerData<GenotypeCategorical>();
        population->registerGlobalData(rng);
        init.afterRegisterData();
        std::vector<Individual> iis(population_size);
        std::generate(iis.begin(), iis.end(), [&population]() { return population->newIndividual(); });
        init.initialize(iis);

        SECTION("All solutions should be of the right size")
        {
            for (auto ii : iis)
            {
                GenotypeCategorical genotype = population->getData<GenotypeCategorical>(ii);
                REQUIRE(genotype.genotype.size() == l);
            }
        }

        SECTION(("As the alphabet size is equal to the solution size, "
                "the occurences of each letter in the alphabet should be equal as well"))
        {
            for (size_t v = 0; v < l; v++)
            {
                size_t count_0 = 0;
                size_t count_1 = 0;

                for (auto ii : iis)
                {
                    GenotypeCategorical genotype = population->getData<GenotypeCategorical>(ii);
                    if (genotype.genotype[v] == 0)
                        ++count_0;
                    if (genotype.genotype[v] == 1)
                        ++count_1;
                }

                REQUIRE(count_0 == count_1);
            }
        }
    }
    SECTION("Initializes weighted when not provided with probabilities")
    {
        size_t l = 2;
        size_t population_size = 128;

        std::vector<char> alphabet_size;
        alphabet_size.resize(l);
        std::fill(alphabet_size.begin(), alphabet_size.end(), 3);

        auto population = std::make_shared<Population>();
        // Set objective
        population->registerGlobalData(GenotypeCategoricalData { l, alphabet_size } );
        auto init = CategoricalProbabilisticallyCompleteInitializer(
            std::make_optional<probabilities>({
                {0.0, 0.5, 0.5},
                {0.5, 0.5, 0.0},
            })
        );
        auto rng = Rng(42);
        // Perform registration routine.
        init.setPopulation(population);
        init.registerData();
        population->registerData<GenotypeCategorical>();
        population->registerGlobalData(rng);
        init.afterRegisterData();
        std::vector<Individual> iis(population_size);
        std::generate(iis.begin(), iis.end(), [&population]() { return population->newIndividual(); });
        init.initialize(iis);

        SECTION("All solutions should be of the right size")
        {
            for (auto ii : iis)
            {
                GenotypeCategorical genotype = population->getData<GenotypeCategorical>(ii);
                REQUIRE(genotype.genotype.size() == l);
            }
        }

        SECTION(("As the alphabet size is equal to the solution size, "
                "the occurences of each letter in the alphabet should be equal as well"))
        {
            for (size_t v = 0; v < l; v++)
            {
                size_t count_0 = 0;
                size_t count_1 = 0;
                size_t count_2 = 0;

                for (auto ii : iis)
                {
                    GenotypeCategorical genotype = population->getData<GenotypeCategorical>(ii);
                    if (genotype.genotype[v] == 0)
                        ++count_0;
                    if (genotype.genotype[v] == 1)
                        ++count_1;
                    if (genotype.genotype[v] == 2)
                        ++count_2;
                }

                if (v == 0)
                {
                    CHECK(count_0 == 0);
                    CHECK(count_1 == count_2);
                }
                else if (v == 1)
                {
                    CHECK(count_0 == count_1);
                    CHECK(count_2 == 0);
                }
            }
        }
    }
}
