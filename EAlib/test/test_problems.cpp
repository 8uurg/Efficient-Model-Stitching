#include "base.hpp"
#include "problems.hpp"
#include "test/mocks_base_ea.hpp"

#include <catch2/catch.hpp>
#include <random>
#include <numeric>
#include <sstream>

TEST_CASE("Discrete Objective function", "[Evaluation]")
{
    size_t l = 10;
    bool called = false;
    auto testObjective = [&called](std::vector<char> &) {
        called = true;
        return 1.0;
    };

    std::vector<char> alphabet_size;
    alphabet_size.resize(l);
    std::fill(alphabet_size.begin(), alphabet_size.end(), 2);

    auto population = std::make_shared<Population>();
    auto obj = DiscreteObjectiveFunction(testObjective, l, alphabet_size);
    obj.setPopulation(population);
    obj.registerData();
    obj.afterRegisterData();

    // Problem defines the encoding used (and the corresponding string length)
    REQUIRE(population->isRegistered<GenotypeCategorical>());
    REQUIRE(population->isGlobalRegistered<GenotypeCategoricalData>());
    auto discrete_l = population->getGlobalData<GenotypeCategoricalData>();
    REQUIRE(discrete_l.get()->l == l);

    // It also defines the objective function to be used & the objective value.
    REQUIRE(population->isRegistered<Objective>());
    
    // In order to allow wrapping to occur, it shouldn't be automatically registered.
    REQUIRE(!population->isGlobalRegistered<GObjectiveFunction>());

    // Create an individual
    Individual a = population->newIndividual();
    GenotypeCategorical &a_genotype = population->getData<GenotypeCategorical>(a);
    a_genotype.genotype.resize(l);
    std::fill(a_genotype.genotype.begin(), a_genotype.genotype.end(), 0);

    // Evaluate!
    auto &evaluation_function = obj;
    evaluation_function.evaluate(a);
    REQUIRE(called);

    // Check value
    auto &a_obj = population->getData<Objective>(a);
    REQUIRE(a_obj.objectives.size() == 1);
    REQUIRE(a_obj.objectives[0] == 1.0);
}

TEST_CASE("Continuous Objective function", "[Evaluation]")
{
    size_t l = 10;
    bool called = false;
    auto testObjective = [&called](std::vector<double> &) {
        called = true;
        return 1.0;
    };

    auto population = std::make_shared<Population>();
    auto obj = ContinuousObjectiveFunction(testObjective, l);
    obj.setPopulation(population);
    obj.registerData();
    obj.afterRegisterData();

    // Problem defines the encoding used (and the corresponding string length)
    REQUIRE(population->isRegistered<GenotypeContinuous>());
    REQUIRE(population->isGlobalRegistered<GenotypeContinuousLength>());
    auto continuous_l = population->getGlobalData<GenotypeContinuousLength>();
    REQUIRE(continuous_l.get()->l == l);

    // It also defines the objective value struct.
    REQUIRE(population->isRegistered<Objective>());

    // In order to allow wrapping to occur, it shouldn't be automatically registered.
    REQUIRE(!population->isGlobalRegistered<GObjectiveFunction>());

    // Create an individual
    Individual a = population->newIndividual();
    GenotypeContinuous &a_genotype = population->getData<GenotypeContinuous>(a);
    a_genotype.genotype.resize(l);
    std::fill(a_genotype.genotype.begin(), a_genotype.genotype.end(), 0.0);

    // Evaluate!
    auto &evaluation_function = obj;
    evaluation_function.evaluate(a);
    REQUIRE(called);

    // Check value
    auto &a_obj = population->getData<Objective>(a);
    REQUIRE(a_obj.objectives.size() == 1);
    REQUIRE(a_obj.objectives[0] == 1.0);
}

TEST_CASE("OneMax", "[Problem]")
{
    size_t l = 10;
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    OneMax problem(l);

    problem.setPopulation(pop);
    problem.registerData();
    problem.afterRegisterData();

    SECTION("uses a Categorical Genotype")
    {
        REQUIRE(pop->isRegistered<GenotypeCategorical>());
    }
    SECTION("and computes an Objective value")
    {
        REQUIRE(pop->isRegistered<Objective>());
    }

    std::shared_ptr<GenotypeCategoricalData> data =
        pop->getGlobalData<GenotypeCategoricalData>();
    SECTION("problem is of specified length")
    {
        REQUIRE(data->l == l);
    }
    SECTION("problem is defined to be binary")
    {
        std::vector<char> binary(data->l);
        std::fill(binary.begin(), binary.end(), 2);
        REQUIRE_THAT(data->alphabet_size, Catch::Matchers::Equals(binary));
    }

    Individual ii = pop->newIndividual();
    GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
    genotype.genotype.resize(l);
    // Start off with all zeroes
    std::fill(genotype.genotype.begin(), genotype.genotype.end(), 0);

    SECTION("all zeroes has a fitness of 0")
    {
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == 0);
    }

    // Fitness should be negative, as this codebase assumes lower = better.
    SECTION("all ones has a fitness of -l")
    {
        std::fill(genotype.genotype.begin(), genotype.genotype.end(), 1);
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l));
    }

    // Fitness should be negative, as this codebase assumes lower = better.
    SECTION("one one has a fitness of -1")
    {
        genotype.genotype[0] = 1;
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -1);
    }
}

TEST_CASE("MaxCut", "[Problem]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Example instance
    size_t num_vertices = 2;
    size_t num_edges = 1;
    double w = GENERATE(1, 2, -1, 5);
    // String length (should be) equal to the number of vertices.
    size_t l = num_vertices;
    std::vector<Edge> edges = {Edge{0, 1, w}};
    MaxCutInstance instance = {
        num_vertices,
        num_edges,
        edges,
    };

    MaxCut problem(instance);

    problem.setPopulation(pop);
    problem.registerData();
    problem.afterRegisterData();

    SECTION("uses a Categorical Genotype")
    {
        REQUIRE(pop->isRegistered<GenotypeCategorical>());
    }
    SECTION("and computes an Objective value")
    {
        REQUIRE(pop->isRegistered<Objective>());
    }

    std::shared_ptr<GenotypeCategoricalData> data =
        pop->getGlobalData<GenotypeCategoricalData>();
    SECTION("problem is of specified length")
    {
        REQUIRE(data->l == l);
    }
    SECTION("problem is defined to be binary")
    {
        std::vector<char> binary(data->l);
        std::fill(binary.begin(), binary.end(), 2);
        REQUIRE_THAT(data->alphabet_size, Catch::Matchers::Equals(binary));
    }

    Individual ii = pop->newIndividual();
    GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
    genotype.genotype.resize(l);
    // Start off with all zeroes
    std::fill(genotype.genotype.begin(), genotype.genotype.end(), 0);

    SECTION("[0, 0] has a fitness of 0")
    {
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == 0);
    }

    SECTION("[1, 1] has a fitness of 0")
    {
        std::fill(genotype.genotype.begin(), genotype.genotype.end(), 1);
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == 0);
    }

    SECTION("[0, 1] has a fitness of -w")
    {
        genotype.genotype[1] = 1;
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -w);
    }

    SECTION("[1, 0] has a fitness of -w")
    {
        genotype.genotype[0] = 1;
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -w);
    }
}

TEST_CASE("MaxCut: Loading")
{
    SECTION("Via problem constructor")
    {
        std::filesystem::path path = "../instances/maxcut/set0a/n0000006i00.txt";
        MaxCut mc(path);
    }
    SECTION("Happy path: valid instance from file")
    {
        std::filesystem::path path = "../instances/maxcut/set0a/n0000006i00.txt";
        MaxCutInstance maxcut_instance = load_maxcut(path);
        REQUIRE(maxcut_instance.num_vertices == 6);
        REQUIRE(maxcut_instance.num_edges == 15);
    }
    SECTION("Non-existant path")
    {
        std::filesystem::path path = "thisfileshouldnotexist.txt";
        REQUIRE_THROWS(load_maxcut(path));
    }
    SECTION("Invalid instance: empty file")
    {
        std::istringstream s("");
        REQUIRE_THROWS(load_maxcut(s));
    }
    SECTION("Invalid instance: missing number")
    {
        std::istringstream s("3");
        REQUIRE_THROWS(load_maxcut(s));
    }
    SECTION("Invalid instance: missing weight")
    {
        std::istringstream s("3 3\n1 2 3\n1 2 3\n1 2");
        REQUIRE_THROWS(load_maxcut(s));
    }
}

TEST_CASE("Best-of-Traps: Single Concatenated-Permuted-Trap", "[Problem]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Example instance
    size_t k = 5;
    size_t num_blocks = 4;
    size_t l = k * num_blocks;

    std::vector<size_t> permutation(l);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<char> optimum(l);
    std::fill(optimum.begin(), optimum.end(), 1);
    ConcatenatedPermutedTrap cpt = {
        l,
        k,
        permutation,
        optimum,
    };
    BestOfTrapsInstance instance = {l, {cpt}};

    BestOfTraps problem(instance);

    problem.setPopulation(pop);
    problem.registerData();
    problem.afterRegisterData();

    SECTION("uses a Categorical Genotype")
    {
        REQUIRE(pop->isRegistered<GenotypeCategorical>());
    }
    SECTION("and computes an Objective value")
    {
        REQUIRE(pop->isRegistered<Objective>());
    }

    std::shared_ptr<GenotypeCategoricalData> data =
        pop->getGlobalData<GenotypeCategoricalData>();
    SECTION("problem is of specified length")
    {
        REQUIRE(data->l == l);
    }
    SECTION("problem is defined to be binary")
    {
        std::vector<char> binary(data->l);
        std::fill(binary.begin(), binary.end(), 2);
        REQUIRE_THAT(data->alphabet_size, Catch::Matchers::Equals(binary));
    }

    Individual ii = pop->newIndividual();
    GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
    genotype.genotype.resize(l);
    // Start off with the optimum
    std::copy(optimum.begin(), optimum.end(), genotype.genotype.begin());

    SECTION("optimum has a fitness of -l")
    {
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l));
    }

    SECTION("inverse of optimum has fitness of -(l - num_blocks)")
    {
        std::transform(genotype.genotype.begin(), genotype.genotype.end(), genotype.genotype.begin(), [](char v) {
            return 1 - v;
        });
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l - num_blocks));
    }

    SECTION("one off optimum has fitness -(l - k)")
    {
        genotype.genotype[0] = 1 - genotype.genotype[0];
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l - k));
    }
}

TEST_CASE("Best-of-Traps: Two Concatenated-Permuted-Traps", "[Problem]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Example instance
    size_t k = 5;
    size_t num_blocks = 4;
    size_t l = k * num_blocks;

    size_t seed = GENERATE(42, 43, 44, 45);
    std::mt19937 rng(seed);

    std::vector<size_t> permutation0(l);
    std::iota(permutation0.begin(), permutation0.end(), 0);
    std::shuffle(permutation0.begin(), permutation0.end(), rng);
    std::vector<char> optimum0(l);
    std::fill(optimum0.begin(), optimum0.end(), 1);

    std::vector<size_t> permutation1(l);
    std::iota(permutation1.begin(), permutation1.end(), 0);
    std::shuffle(permutation1.begin(), permutation1.end(), rng);
    std::vector<char> optimum1(l);
    std::fill(optimum1.begin(), optimum1.begin() + l / 2, 0);
    std::fill(optimum1.begin() + l / 2, optimum1.end(), 1);

    ConcatenatedPermutedTrap cpt0 = {
        l,
        k,
        permutation0,
        optimum0,
    };
    ConcatenatedPermutedTrap cpt1 = {
        l,
        k,
        permutation1,
        optimum1,
    };
    BestOfTrapsInstance instance = {l, {cpt0, cpt1}};

    BestOfTraps problem(instance);

    problem.setPopulation(pop);
    problem.registerData();
    problem.afterRegisterData();

    Individual ii = pop->newIndividual();
    GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
    genotype.genotype.resize(l);

    SECTION("optimum of function 0 has a fitness of -l")
    {
        std::copy(optimum0.begin(), optimum0.end(), genotype.genotype.begin());
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l));
    }

    SECTION("optimum of function 1 has a fitness of -l")
    {
        std::copy(optimum1.begin(), optimum1.end(), genotype.genotype.begin());
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l));
    }
}

TEST_CASE("BestOfTraps: Loading")
{
    SECTION("Via problem constructor")
    {
        std::filesystem::path path = "../instances/bestoftraps/k5/bot_n10k5fns2s42.txt";
        BestOfTraps bot(path);
    }
    SECTION("Happy path: Valid existing instance")
    {
    // Instance is a fully connected instance
        std::filesystem::path path = "../instances/bestoftraps/k5/bot_n10k5fns2s42.txt";
        BestOfTrapsInstance bot_instance = load_BestOfTraps(path);
        REQUIRE(bot_instance.l == 10);
        REQUIRE(bot_instance.concatenatedPermutedTraps.size() == 2);
    }

    SECTION("Non-existant path")
    {
        std::filesystem::path path = "thisfileshouldnotexist.txt";
        REQUIRE_THROWS(load_BestOfTraps(path));
    }
    SECTION("Invalid instance: empty file")
    {
        std::istringstream s("");
        REQUIRE_THROWS(load_BestOfTraps(s));
    }
    SECTION("Invalid instance: missing subfunctions")
    {
        std::istringstream s("1\n");
        REQUIRE_THROWS(load_BestOfTraps(s));
    }
    SECTION("Invalid instance: missing optimum and permutation")
    {
        std::istringstream s("1\n10 5\n");
        REQUIRE_THROWS(load_BestOfTraps(s));
    }
    SECTION("Invalid instance: missing permutation")
    {
        std::istringstream s("1\5 5\n0 0 0 0 0\n");
        REQUIRE_THROWS(load_BestOfTraps(s));
    }
}

TEST_CASE("Worst-of-Traps: Single Concatenated-Permuted-Trap", "[Problem]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    // Example instance
    size_t k = 5;
    size_t num_blocks = 4;
    size_t l = k * num_blocks;

    std::vector<size_t> permutation(l);
    std::iota(permutation.begin(), permutation.end(), 0);
    std::vector<char> optimum(l);
    std::fill(optimum.begin(), optimum.end(), 1);
    ConcatenatedPermutedTrap cpt = {
        l,
        k,
        permutation,
        optimum,
    };
    BestOfTrapsInstance instance = {l, {cpt}};

    WorstOfTraps problem(instance);

    problem.setPopulation(pop);
    problem.registerData();
    problem.afterRegisterData();

    SECTION("uses a Categorical Genotype")
    {
        REQUIRE(pop->isRegistered<GenotypeCategorical>());
    }
    SECTION("and computes an Objective value")
    {
        REQUIRE(pop->isRegistered<Objective>());
    }

    std::shared_ptr<GenotypeCategoricalData> data =
        pop->getGlobalData<GenotypeCategoricalData>();
    SECTION("problem is of specified length")
    {
        REQUIRE(data->l == l);
    }
    SECTION("problem is defined to be binary")
    {
        std::vector<char> binary(data->l);
        std::fill(binary.begin(), binary.end(), 2);
        REQUIRE_THAT(data->alphabet_size, Catch::Matchers::Equals(binary));
    }

    Individual ii = pop->newIndividual();
    GenotypeCategorical &genotype = pop->getData<GenotypeCategorical>(ii);
    genotype.genotype.resize(l);
    // Start off with the optimum
    std::copy(optimum.begin(), optimum.end(), genotype.genotype.begin());

    SECTION("optimum has a fitness of -l")
    {
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l));
    }

    SECTION("inverse of optimum has fitness of -(l - num_blocks)")
    {
        std::transform(genotype.genotype.begin(), genotype.genotype.end(), genotype.genotype.begin(), [](char v) {
            return 1 - v;
        });
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l - num_blocks));
    }

    SECTION("one off optimum has fitness -(l - k)")
    {
        genotype.genotype[0] = 1 - genotype.genotype[0];
        problem.evaluate(ii);
        Objective &objective = pop->getData<Objective>(ii);
        REQUIRE(objective.objectives.size() == 1);
        REQUIRE(objective.objectives[0] == -static_cast<double>(l - k));
    }
}

TEST_CASE("Compose")
{
    using trompeloeil::_;
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    Individual test = pop->newIndividual();

    auto o0 = std::make_shared<MockObjectiveFunction>();
    auto o1 = std::make_shared<MockObjectiveFunction>();

    std::vector<std::shared_ptr<ObjectiveFunction>> functions {o0, o1};

    Compose composed(functions);
    SECTION("setPopulation is passed to both")
    {
        REQUIRE_CALL(*o0, setPopulation(_));
        REQUIRE_CALL(*o1, setPopulation(_));
        composed.setPopulation(pop);
    }

    SECTION("registerData is passed to both")
    {
        REQUIRE_CALL(*o0, registerData());
        REQUIRE_CALL(*o1, registerData());
        composed.registerData();
    }

    SECTION("afterRegisterData is passed to both")
    {
        REQUIRE_CALL(*o0, afterRegisterData());
        REQUIRE_CALL(*o1, afterRegisterData());
        composed.afterRegisterData();
    }

    SECTION("evaluations are passed to both")
    {
        REQUIRE_CALL(*o0, evaluate(_));
        REQUIRE_CALL(*o1, evaluate(_));
        composed.evaluate(test);
    }
}

TEST_CASE("NK-Landscape", "[Problem]")
{
    NKLandscapeInstance nkli = load_nklandscape("../instances/nk/instances/n4_s1/L20/1.txt");

    NKLandscape nkl(nkli);

    std::shared_ptr<Population> pop = std::make_shared<Population>();
    nkl.setPopulation(pop);
    nkl.registerData();
    nkl.afterRegisterData();

    Individual n = pop->newIndividual();
    auto& d = *pop->getGlobalData<GenotypeCategoricalData>();
    auto& gc = pop->getData<GenotypeCategorical>(n);
    gc.genotype.resize(d.l);
    std::fill(gc.genotype.begin(), gc.genotype.end(), 0);

    nkl.evaluate(n);
}