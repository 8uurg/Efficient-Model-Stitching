#include "acceptation_criteria.hpp"

#include "mocks_base_ea.hpp"
#include <catch2/catch.hpp>

TEST_CASE("Objective Acceptance Criterion", "[Operator][Acceptance]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    SingleObjectiveAcceptanceCriterion oac;
    oac.setPopulation(pop);
    oac.registerData();
    SECTION("Objective should be defined")
    {
        REQUIRE_THROWS(oac.afterRegisterData());
    }
    pop->registerData<Objective>();
    oac.afterRegisterData();

    Individual a = pop->newIndividual();
    Individual b = pop->newIndividual();
    auto &o_a = pop->getData<Objective>(a);
    auto &o_b = pop->getData<Objective>(b);

    SECTION("Should return 3 if they are equal")
    {
        o_a.objectives = {0.0};
        o_b.objectives = {0.0};
        REQUIRE(oac.compare(a, b) == 3);
    }

    SECTION("Should return 1 if a is better")
    {
        o_a.objectives = {0.0};
        o_b.objectives = {1.0};
        REQUIRE(oac.compare(a, b) == 1);
    }

    SECTION("Should return 2 if b is better")
    {
        o_a.objectives = {1.0};
        o_b.objectives = {0.0};
        REQUIRE(oac.compare(a, b) == 2);
    }

    SECTION("Should return 0 if the result cannot be determined")
    {
        o_a.objectives = {NAN};
        o_b.objectives = {NAN};
        REQUIRE(oac.compare(a, b) == 0);
    }

    SECTION("Should return 1 if b is uninitialized and a is initialized")
    {
        o_a.objectives = {1.0};
        o_b.objectives = {};
        REQUIRE(oac.compare(a, b) == 1);
    }

    SECTION("Should return 2 if a is uninitialized and b is initialized")
    {
        o_a.objectives = {};
        o_b.objectives = {1.0};
        REQUIRE(oac.compare(a, b) == 2);
    }

    SECTION("Should return 0 if both uninitialized")
    {
        o_a.objectives = {};
        o_b.objectives = {};
        REQUIRE(oac.compare(a, b) == 0);
    }

    // TODO: Different objectives
}

TEST_CASE("Objective Domination Acceptance Criterion", "[Operator][Acceptance]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    DominationObjectiveAcceptanceCriterion doac({0, 1});
    doac.setPopulation(pop);
    doac.registerData();
    SECTION("Objective should be defined")
    {
        REQUIRE_THROWS(doac.afterRegisterData());
    }
    pop->registerData<Objective>();
    doac.afterRegisterData();

    Individual a = pop->newIndividual();
    Individual b = pop->newIndividual();
    auto &o_a = pop->getData<Objective>(a);
    auto &o_b = pop->getData<Objective>(b);

    SECTION("Should return 3 if they are equal")
    {
        o_a.objectives = {0.0, 0.0};
        o_b.objectives = {0.0, 0.0};
        REQUIRE(doac.compare(a, b) == 3);
    }

    SECTION("Should return 1 if a dominates b")
    {
        o_a.objectives = {0.0, 0.0};
        o_b.objectives = {1.0, 0.0};
        REQUIRE(doac.compare(a, b) == 1);
    }

    SECTION("Should return 2 if b dominates a")
    {
        o_a.objectives = {1.0, 0.0};
        o_b.objectives = {0.0, 0.0};
        REQUIRE(doac.compare(a, b) == 2);
    }

    SECTION("Should return 0 if the result cannot be determined")
    {
        o_a.objectives = {1.0, 0.0};
        o_b.objectives = {0.0, 1.0};
        REQUIRE(doac.compare(a, b) == 0);
    }

    SECTION("Should return 1 if b is uninitialized and a is initialized")
    {
        o_a.objectives = {1.0, 1.0};
        o_b.objectives = {};
        REQUIRE(doac.compare(a, b) == 1);
    }

    SECTION("Should return 2 if a is uninitialized and b is initialized")
    {
        o_a.objectives = {};
        o_b.objectives = {1.0, 1.0};
        REQUIRE(doac.compare(a, b) == 2);
    }

    SECTION("Should return 0 if both uninitialized")
    {
        o_a.objectives = {};
        o_b.objectives = {};
        REQUIRE(doac.compare(a, b) == 0);
    }

    SECTION("Should return 0 if both partially uninitialized")
    {
        o_a.objectives = {0.5};
        o_b.objectives = {0.2};
        REQUIRE(doac.compare(a, b) == 0);
    }
}

TEST_CASE("Sequential Combine Acceptance Criterion", "[Operator][Acceptance]")
{
    using trompeloeil::_;

    bool nondeterminate_is_equal = GENERATE(false, true);

    std::shared_ptr<Population> pop = std::make_shared<Population>();

    auto c0 = std::shared_ptr<MockAcceptanceCriterion>(new MockAcceptanceCriterion());
    auto c1 = std::shared_ptr<MockAcceptanceCriterion>(new MockAcceptanceCriterion());

    SequentialCombineAcceptanceCriterion scac({c0, c1}, nondeterminate_is_equal);

    SECTION("Setting population is called on all underlying")
    {
        REQUIRE_CALL(*c0, setPopulation(_))
            .WITH(_1.get() == pop.get());
        REQUIRE_CALL(*c1, setPopulation(_))
            .WITH(_1.get() == pop.get());

        scac.setPopulation(pop);
    }

    SECTION("registerData is called on all underlying")
    {
        REQUIRE_CALL(*c0, registerData());
        REQUIRE_CALL(*c1, registerData());

        scac.registerData();
    }
    
    SECTION("afterRegisterData is called on all underlying")
    {
        REQUIRE_CALL(*c0, afterRegisterData());
        REQUIRE_CALL(*c1, afterRegisterData());

        scac.afterRegisterData();
    }

    Individual a = pop->newIndividual();
    Individual b = pop->newIndividual();

    SECTION("a better on first criterion")
    {
        REQUIRE_CALL(*c0, compare(a, b))
            .RETURN(1);

        REQUIRE(scac.compare(a, b) == 1);
    }

    SECTION("b better on first criterion")
    {
        REQUIRE_CALL(*c0, compare(a, b))
            .RETURN(2);

        REQUIRE(scac.compare(a, b) == 2);
    }

    SECTION("equal on first, a better on second")
    {
        REQUIRE_CALL(*c0, compare(a, b))
            .RETURN(3);
        REQUIRE_CALL(*c1, compare(a, b))
            .RETURN(1);

        REQUIRE(scac.compare(a, b) == 1);
    }

    SECTION("equal on first, b better on second")
    {
        REQUIRE_CALL(*c0, compare(a, b))
            .RETURN(3);
        REQUIRE_CALL(*c1, compare(a, b))
            .RETURN(2);

        REQUIRE(scac.compare(a, b) == 2);
    }

    SECTION("equal on both criteria")
    {
        REQUIRE_CALL(*c0, compare(a, b))
            .RETURN(3);
        REQUIRE_CALL(*c1, compare(a, b))
            .RETURN(3);

        REQUIRE(scac.compare(a, b) == 3);
    }

    SECTION("non-determinate first")
    {
        REQUIRE_CALL(*c0, compare(a, b))
            .RETURN(0);
        REQUIRE_CALL(*c1, compare(a, b))
            .RETURN(1)
            .TIMES(0, 1);

        if (nondeterminate_is_equal)
            REQUIRE(scac.compare(a, b) == 1);
        else
            REQUIRE(scac.compare(a, b) == 0);
    }
}

TEST_CASE("Threshold Acceptance Criterion - No Slack", "[Operator][Acceptance]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    ThresholdAcceptanceCriterion oac(0, 0.0, false);
    oac.setPopulation(pop);
    oac.registerData();
    pop->registerData<Objective>();
    oac.afterRegisterData();

    Individual a = pop->newIndividual();
    Individual b = pop->newIndividual();
    auto &o_a = pop->getData<Objective>(a);
    auto &o_b = pop->getData<Objective>(b);

    SECTION("Should return 3 if they are on the same side of the threshold")
    {
        o_a.objectives = {1.0};
        o_b.objectives = {2.0};
        REQUIRE(oac.compare(a, b) == 3);
    }

    SECTION("Should return 1 if a is below, and b is above")
    {
        o_a.objectives = {-1.0};
        o_b.objectives = {1.0};
        REQUIRE(oac.compare(a, b) == 1);
    }
    SECTION("Should return 1 if a is at threshold, and b is above")
    {
        o_a.objectives = {0.0};
        o_b.objectives = {1.0};
        REQUIRE(oac.compare(a, b) == 1);
    }

    SECTION("Should return 2 if b is below, and a is above")
    {
        o_a.objectives = {1.0};
        o_b.objectives = {-1.0};
        REQUIRE(oac.compare(a, b) == 2);
    }

    SECTION("Should return 1 if b is uninitialized and a is initialized")
    {
        o_a.objectives = {1.0, 1.0};
        o_b.objectives = {};
        REQUIRE(oac.compare(a, b) == 1);
    }

    SECTION("Should return 2 if a is uninitialized and b is initialized")
    {
        o_a.objectives = {};
        o_b.objectives = {1.0, 1.0};
        REQUIRE(oac.compare(a, b) == 2);
    }

    SECTION("Should return 0 if both uninitialized")
    {
        o_a.objectives = {};
        o_b.objectives = {};
        REQUIRE(oac.compare(a, b) == 0);
    }
}


TEST_CASE("Threshold Acceptance Criterion - With Slack", "[Operator][Acceptance]")
{
    std::shared_ptr<Population> pop = std::make_shared<Population>();
    ThresholdAcceptanceCriterion oac(0, 0.0, true);
    oac.setPopulation(pop);
    oac.registerData();
    pop->registerData<Objective>();
    oac.afterRegisterData();

    Individual a = pop->newIndividual();
    Individual b = pop->newIndividual();
    auto &o_a = pop->getData<Objective>(a);
    auto &o_b = pop->getData<Objective>(b);

    SECTION("Should return 3 if they violate the constraint equally badly.")
    {
        o_a.objectives = {1.0};
        o_b.objectives = {1.0};
        REQUIRE(oac.compare(a, b) == 3);
    }

    SECTION("Should return 1 if a violates the constraint less.")
    {
        o_a.objectives = {1.0};
        o_b.objectives = {2.0};
        REQUIRE(oac.compare(a, b) == 1);
    }
    
    SECTION("Should return 2 if b violates the constraint less.")
    {
        o_a.objectives = {2.0};
        o_b.objectives = {1.0};
        REQUIRE(oac.compare(a, b) == 2);
    }

    SECTION("Should return 1 if a is below, and b is above")
    {
        o_a.objectives = {-1.0};
        o_b.objectives = {1.0};
        REQUIRE(oac.compare(a, b) == 1);
    }
    SECTION("Should return 1 if a is at threshold, and b is above")
    {
        o_a.objectives = {0.0};
        o_b.objectives = {1.0};
        REQUIRE(oac.compare(a, b) == 1);
    }

    SECTION("Should return 2 if b is below, and a is above")
    {
        o_a.objectives = {1.0};
        o_b.objectives = {-1.0};
        REQUIRE(oac.compare(a, b) == 2);
    }

    SECTION("Should return 3 if both are below")
    {
        o_a.objectives = {-2.0};
        o_b.objectives = {-1.0};
        REQUIRE(oac.compare(a, b) == 3);
    }

    SECTION("Should return 1 if b is uninitialized and a is initialized")
    {
        o_a.objectives = {1.0, 1.0};
        o_b.objectives = {};
        REQUIRE(oac.compare(a, b) == 1);
    }

    SECTION("Should return 2 if a is uninitialized and b is initialized")
    {
        o_a.objectives = {};
        o_b.objectives = {1.0, 1.0};
        REQUIRE(oac.compare(a, b) == 2);
    }

    SECTION("Should return 0 if both uninitialized")
    {
        o_a.objectives = {};
        o_b.objectives = {};
        REQUIRE(oac.compare(a, b) == 0);
    }
}