#include "base.hpp"
#include "mocks_base_ea.hpp"
#include "problems.hpp"

#include <catch2/catch.hpp>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>

#include <cereal/archives/json.hpp>

struct TestData
{
    size_t a;
    std::vector<size_t> b;

    template<class Archive>
    void serialize( Archive & ar )
    {
        ar(a);
        ar(b);
    }
};

struct TestData2
{
    size_t x;
};

TEST_CASE("Population: Per Solution Data", "[Population]")
{
    auto pop = std::make_shared<Population>();
    SECTION("registering data before doing anything should work")
    {
        REQUIRE_NOTHROW(pop->registerData<TestData>());
    }

    SECTION("data should now be registerable after creation as well and should be properly backed by memory.")
    {
        auto ii = pop->newIndividual();
        REQUIRE_NOTHROW(pop->registerData<TestData>());
        auto &td = pop->getData<TestData>(ii);
        td.a = 5;
    }

    SECTION("registering data twice should ignore the second time")
    {
        REQUIRE_NOTHROW(pop->registerData<TestData>());
        REQUIRE_NOTHROW(pop->registerData<TestData>());
    }

    SECTION("creating a new individual should work")
    {
        REQUIRE_NOTHROW(pop->newIndividual());
    }

    SECTION("dropping an existing individual should work")
    {
        Individual i = pop->newIndividual();
        REQUIRE_NOTHROW(pop->dropIndividual(i));
    }

    SECTION("dropping a non-existing individual should throw an exception")
    {
        REQUIRE_THROWS(pop->dropIndividual(Individual{0, pop.get()}));
    }

    SECTION("dropping and creating a new individual should reuse instead of enlarging")
    {
        Individual i_a = pop->newIndividual();
        REQUIRE_NOTHROW(pop->dropIndividual(i_a));
        Individual i_b = pop->newIndividual();
        REQUIRE(i_a.i == i_b.i);
    }

    SECTION("getting data for an existing individual for registered data should work")
    {
        pop->registerData<TestData>();
        Individual i = pop->newIndividual();
        REQUIRE_NOTHROW(pop->getData<TestData>(i));
    }

    SECTION("getting data for a non-existing individual should throw an exception")
    {
        pop->registerData<TestData>();
        REQUIRE_THROWS(pop->getData<TestData>(Individual{0, pop.get()}));
    }

    SECTION("getting unregistered data should throw an exception")
    {
        Individual i = pop->newIndividual();
        REQUIRE_THROWS(pop->getData<TestData>(i));
    }

    SECTION(("the data obtained from different points should point to the same memory"
             "i.e. be aliased"))
    {
        pop->registerData<TestData>();
        Individual a = pop->newIndividual();
        auto &data_a = pop->getData<TestData>(a);
        data_a.a = 1;
        data_a.b = {1, 2};
        data_a.a = 2;
        data_a.b = {3, 4};
        auto &data_b = pop->getData<TestData>(a);
        CHECK(data_b.a == 2);
        std::vector<size_t> expected = {3, 4};
        CHECK_THAT(data_b.b, Catch::Matchers::Equals(expected));
    }

    SECTION("copyIndividual should copy all the data from one individual to another")
    {
        pop->registerData<TestData>();
        Individual a = pop->newIndividual();
        Individual b = pop->newIndividual();
        auto &data_a = pop->getData<TestData>(a);
        data_a.a = 1;
        data_a.b = {1, 2};
        pop->copyIndividual(a, b);
        auto &data_b = pop->getData<TestData>(b);
        CHECK(data_b.a == 1);
        std::vector<size_t> expected = {1, 2};
        CHECK_THAT(data_b.b, Catch::Matchers::Equals(expected));
    }

    SECTION("copied data should be copied, not point to the same memory")
    {
        pop->registerData<TestData>();
        Individual a = pop->newIndividual();
        Individual b = pop->newIndividual();
        auto &data_a = pop->getData<TestData>(a);
        data_a.a = 1;
        data_a.b = {1, 2};
        pop->copyIndividual(a, b);
        data_a.a = 2;
        data_a.b = {3, 4};
        auto &data_b = pop->getData<TestData>(b);
        CHECK(data_b.a == 1);
        std::vector<size_t> expected = {1, 2};
        CHECK_THAT(data_b.b, Catch::Matchers::Equals(expected));
    }

    SECTION("ALL data should be copied if there are multiple pieces of data attached")
    {
        pop->registerData<TestData>();
        pop->registerData<TestData2>();
        Individual a = pop->newIndividual();
        Individual b = pop->newIndividual();
        auto &data_a1 = pop->getData<TestData>(a);
        data_a1.a = 1;
        data_a1.b = {1, 2};
        auto &data_a2 = pop->getData<TestData2>(a);
        data_a2.x = 5;
        pop->copyIndividual(a, b);
        auto &data_b1 = pop->getData<TestData>(b);
        CHECK(data_b1.a == 1);
        std::vector<size_t> expected = {1, 2};
        CHECK_THAT(data_b1.b, Catch::Matchers::Equals(expected));
        auto &data_b2 = pop->getData<TestData2>(b);
        CHECK(data_b2.x == 5);
    }

    SECTION("GetData should work for a new solution - with allocation")
    {
        pop->registerData<TestData>();
        Individual a = pop->newIndividual();
        REQUIRE_NOTHROW(pop->getData<TestData>(a));
    }

    SECTION("GetData should work for a new solution - with reuse allocation")
    {
        pop->registerData<TestData>();
        Individual d = pop->newIndividual();
        pop->dropIndividual(d);

        Individual a = pop->newIndividual();
        REQUIRE_NOTHROW(pop->getData<TestData>(a));
    }
}

class TestGlobalData
{
  public:
    TestGlobalData(std::vector<size_t> a) : a(a){};
    std::vector<size_t> a;
};

TEST_CASE("Population: Global Data", "[Population]")
{
    auto pop = std::make_shared<Population>();
    TestGlobalData data{std::vector<size_t>{0, 1, 2}};
    REQUIRE_NOTHROW(pop->registerGlobalData(data));

    SECTION("normal usage should work, and give back the original data")
    {
        auto data_ref = pop->getGlobalData<TestGlobalData>();
        REQUIRE_THAT(data.a, Catch::Matchers::Equals(data_ref->a));
    }

    SECTION("duplicate registration should throw an exception")
    {
        REQUIRE_THROWS(pop->registerGlobalData(data));
    }
}

TEST_CASE("Rng")
{
    SECTION("should be initializable without a seed")
    {
        Rng rng = Rng(std::nullopt);
        (void) rng;
    }
}

TEST_CASE("Limiter")
{
    using trompeloeil::_;

    SECTION("Passes through evaluation calls")
    {
        auto objective = std::make_shared<MockObjectiveFunction>();
        Limiter limiter(objective);
        Individual fake { 0, NULL };
        REQUIRE_CALL(*objective, evaluate(_));
        limiter.evaluate(fake);
    }

    SECTION("Raises an exception once the evaluation limit is hit")
    {
        auto objective = std::make_shared<MockObjectiveFunction>();
        Limiter limiter(objective, 2);
        Individual fake { 0, NULL };
        REQUIRE_CALL(*objective, evaluate(_)).TIMES(2);
        limiter.evaluate(fake);
        limiter.evaluate(fake);
        REQUIRE_THROWS(limiter.evaluate(fake));
    }

    SECTION("Raises an exception once the time limit is hit")
    {
        auto objective = std::make_shared<MockObjectiveFunction>();
        Limiter limiter(objective, std::nullopt, std::chrono::milliseconds(0));
        Individual fake { 0, NULL };
        REQUIRE_THROWS(limiter.evaluate(fake));
    }
}

TEST_CASE("ElitistMonitor")
{
    using trompeloeil::_;

    std::shared_ptr<Population> pop = std::make_shared<Population>();
    pop->registerData<TestData>();
    auto objective = std::make_shared<MockObjectiveFunction>();
    auto criterion = std::make_shared<MockAcceptanceCriterion>();
    ElitistMonitor monitor(objective, criterion);

    trompeloeil::sequence obj, crit, oco, coc;

    REQUIRE_CALL(*objective, setPopulation(_)).IN_SEQUENCE(obj, oco);
    REQUIRE_CALL(*criterion, setPopulation(_)).IN_SEQUENCE(crit, coc);
    REQUIRE_CALL(*objective, registerData()).IN_SEQUENCE(obj, coc);
    REQUIRE_CALL(*criterion, registerData()).IN_SEQUENCE(crit, oco);
    REQUIRE_CALL(*objective, afterRegisterData()).IN_SEQUENCE(obj, oco);
    REQUIRE_CALL(*criterion, afterRegisterData()).IN_SEQUENCE(crit, coc);

    monitor.setPopulation(pop);
    monitor.registerData();
    monitor.afterRegisterData();

    REQUIRE(monitor.getElitist().has_value());

    Individual i = pop->newIndividual();
    TestData &i_data = pop->getData<TestData>(i);
    i_data.a = 42;
    {
        REQUIRE_CALL(*objective, evaluate(_));
        monitor.evaluate(i);
    }
    
    SECTION("First call always replaces the elitist")
    {
        TestData &elitist_data = pop->getData<TestData>(*monitor.getElitist());
        REQUIRE(elitist_data.a == 42);
    }

    i_data.a = 32;
    {
        REQUIRE_CALL(*objective, evaluate(_));
        REQUIRE_CALL(*criterion, compare(_, _)).RETURN((_1.i == i.i) ? 2 : 1);
        monitor.evaluate(i);
    }
    SECTION("Further calls follow the judgement provided")
    {
        TestData &elitist_data = pop->getData<TestData>(*monitor.getElitist());
        REQUIRE(elitist_data.a == 42);
    }
    {
        REQUIRE_CALL(*objective, evaluate(_));
        REQUIRE_CALL(*criterion, compare(_, _)).RETURN((_1.i == i.i) ? 1 : 2);
        monitor.evaluate(i);
    }
    SECTION("Further calls follow the judgement provided")
    {
        TestData &elitist_data = pop->getData<TestData>(*monitor.getElitist());
        REQUIRE(elitist_data.a == 32);
    }
}

TEST_CASE("AverageFitnessComparator")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    AverageFitnessComparator comparator;

    pop->registerData<Objective>();

    std::vector<Individual> population_a(1);
    population_a[0] = pop->newIndividual();
    Objective &p_a_0 = pop->getData<Objective>(population_a[0]);
    p_a_0.objectives = {0};
    auto approach_a = std::make_shared<MockGenerationalApproach>();
    ALLOW_CALL(*approach_a, getSolutionPopulation()).LR_RETURN(population_a);

    std::vector<Individual> population_b(1);
    population_b[0] = pop->newIndividual();
    Objective &p_b_0 = pop->getData<Objective>(population_b[0]);
    p_b_0.objectives = {-1};
    auto approach_b = std::make_shared<MockGenerationalApproach>();
    ALLOW_CALL(*approach_b, getSolutionPopulation()).LR_RETURN(population_b);

    comparator.setPopulation(pop);
    comparator.registerData();
    comparator.afterRegisterData();
    comparator.clear();

    SECTION("same => comparator returns 3")
    {
        REQUIRE(comparator.compare(approach_a, approach_a) == 3);
    }
    SECTION("b is better than a => comparator returns 2")
    {
        REQUIRE(comparator.compare(approach_a, approach_b) == 2);
    }
    SECTION("a is better than b => comparator returns 1")
    {
        REQUIRE(comparator.compare(approach_b, approach_a) == 1);
    }
}

TEST_CASE("ObjectiveValuesToReachDetector")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    double f = 0.0;
    size_t l = 0;
    std::vector<char> alphabet_size;
    std::vector<std::vector<double>> vtrs = {
        {1.0},
        {2.0},
    };

    auto &&fn = [&f](std::vector<char> &){ return f; };
    auto dof = std::make_shared<DiscreteObjectiveFunction>(fn, l, alphabet_size);
    auto vtrd = std::make_shared<ObjectiveValuesToReachDetector>(dof, vtrs);

    vtrd->setPopulation(pop);
    vtrd->registerData();
    vtrd->afterRegisterData();

    auto i = pop->newIndividual();

    REQUIRE_NOTHROW(vtrd->evaluate(i));
    f = 1.0;
    REQUIRE_NOTHROW(vtrd->evaluate(i));
    REQUIRE_NOTHROW(vtrd->evaluate(i));
    f = 2.0;
    REQUIRE_THROWS(vtrd->evaluate(i));
}

// Register the serialized container & mark as serialzable 
template <>
struct is_data_serializable<TestData> : std::true_type { };
CEREAL_REGISTER_TYPE(SubDataContainer<TestData>)
// CEREAL_REGISTER_TYPE(SubDataContainer<TestData2>)

TEST_CASE("Serializing a subpopulation")
{
    std::filesystem::path p("./test.json");

    Population pop;
    pop.registerData<TestData>();
    pop.registerData<TestData2>();

    std::vector<Individual> iis(10);
    pop.newIndividuals(iis);

    TypedGetter<TestData> tgtd = pop.getDataContainer<TestData>();
    TypedGetter<TestData2> tgtd2 = pop.getDataContainer<TestData2>();
    for (size_t idx = 0; idx < 10; ++idx)
    {
        tgtd.getData(iis[idx]).a = idx;
        tgtd2.getData(iis[idx]).x = idx + 5;
    }

    {
        // Serialize to file.
        std::ofstream test_out(p);
        cereal::JSONOutputArchive oarchive(test_out);
        auto d = pop.getSubpopulationData(iis);
        oarchive(d);
    }
    
    // Alter the data.
    for (size_t idx = 0; idx < 10; ++idx)
    {
        tgtd.getData(iis[idx]).a = idx + 1;
        tgtd2.getData(iis[idx]).x = idx + 6;
    }

    {
        // Load data
        std::ifstream test_in(p);
        cereal::JSONInputArchive iarchive(test_in);
        SubpopulationData data;
        iarchive(data);

        // Update population content.
        data.inject(pop, iis);
    }

    // Checkycheck
    for (size_t idx = 0; idx < 10; ++idx)
    {
        // Serialized data has been reverted.
        REQUIRE(tgtd.getData(iis[idx]).a == idx);
        // Unserialized data has been left untouched.
        REQUIRE(tgtd2.getData(iis[idx]).x == idx + 6);
    }
    
}