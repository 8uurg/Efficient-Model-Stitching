#include "base.hpp"
#include "gomea.hpp"
#include "problems.hpp"
#include "running.hpp"
#include "mocks_base_ea.hpp"
#include "mocks_running.hpp"
#include "initializers.hpp"

#include <initializer_list>
#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

struct TerminationStepperTestFactory
{
    MAKE_MOCK0(fn, std::shared_ptr<GenerationalApproach>());
};

TEST_CASE("TerminationStepper")
{
    using trompeloeil::_;
    std::shared_ptr<Population> pop = std::make_shared<Population>();

    TerminationStepperTestFactory factory;

    trompeloeil::sequence setup_approach;
    
    auto approach = std::shared_ptr<MockGenerationalApproach>(new MockGenerationalApproach());
    REQUIRE_CALL(factory, fn()).RETURN(approach);
    REQUIRE_CALL(*approach, setPopulation(_)).IN_SEQUENCE(setup_approach);
    REQUIRE_CALL(*approach, registerData()).IN_SEQUENCE(setup_approach);
    REQUIRE_CALL(*approach, afterRegisterData()).IN_SEQUENCE(setup_approach);

    // Originally we were checking if the approach got freed.
    // We are now using a smart pointer, so it shouldn't be freed anymore
    // as we are still holding a reference to `approach` above!
    {
        TerminationStepper stepper([&factory]() { return factory.fn(); });

        // First of all. set up a few successful steps, followed by a step which terminates.
        trompeloeil::sequence seq_stepping;

        REQUIRE_CALL(*approach, step()).TIMES(10).IN_SEQUENCE(setup_approach, seq_stepping);
        REQUIRE_CALL(*approach, step()).THROW(stop_approach()).IN_SEQUENCE(seq_stepping);

        stepper.setPopulation(pop);
        stepper.registerData();
        stepper.afterRegisterData();

        // Stepper should absorb the exception that terminated it.
        REQUIRE_NOTHROW(stepper.run());
    }
    
}

struct InterleavedMultistartSchemeTestFactory
{
    MAKE_MOCK1(fn, std::shared_ptr<GenerationalApproach>(size_t));
};

TEST_CASE("InterleavedMultistartScheme")
{
    using trompeloeil::_;

    std::shared_ptr<Population> pop = std::make_shared<Population>();
    InterleavedMultistartSchemeTestFactory factory;

    auto approach0 = std::shared_ptr<MockGenerationalApproach>(new MockGenerationalApproach());
    auto approach1 = std::shared_ptr<MockGenerationalApproach>(new MockGenerationalApproach());
    auto approach2 = std::shared_ptr<MockGenerationalApproach>(new MockGenerationalApproach());

    // Each call to the factory should return the next approach up.
    // Furthermore, each should have been called with an increasing population size;
    trompeloeil::sequence seq, o0, o1, o2;
    
    // approach 0 is added
    REQUIRE_CALL(factory, fn(4ULL))
        .RETURN(approach0)
        .IN_SEQUENCE(seq, o0);
    // approach 1 is added
    REQUIRE_CALL(factory, fn(8ULL))
        .RETURN(approach1)
        .IN_SEQUENCE(o1);
    // approach2 is added
    REQUIRE_CALL(factory, fn(16ULL))
        .RETURN(approach2)
        .IN_SEQUENCE(o2);

    // Approaches should be set up before use
    REQUIRE_CALL(*approach0, setPopulation(_)).IN_SEQUENCE(o0);
    REQUIRE_CALL(*approach0, registerData()).IN_SEQUENCE(o0);
    REQUIRE_CALL(*approach0, afterRegisterData()).IN_SEQUENCE(o0);
    
    // Other two approaches cannot be configured this way as individuals have been created
    REQUIRE_CALL(*approach1, setPopulation(_)).IN_SEQUENCE(o1);

    REQUIRE_CALL(*approach2, setPopulation(_)).IN_SEQUENCE(o2);
        
    // Expected sequence for size 3 elements.
    // Comments starting with # are utilize internals of the
    // implementation (and are not part of the specification!)
    // # first loop - initialize a0, step a0 three times.
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq, o0);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    // # second loop - initialize a1, step a0 once, a1 once.
    // # i.e. start with current generation.
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach1, step()).IN_SEQUENCE(seq, o1);
    // # recursive call to step a0 thrice.
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    // # - 1
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach1, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq); 
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    // # - 2
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach1, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    // # - 3
    // # approach 2 is initialized.
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach1, step()).IN_SEQUENCE(seq);
    // Finally, the last call, throw an exception to stop the runner.
    REQUIRE_CALL(*approach2, step()).IN_SEQUENCE(seq, o2).THROW(stop_approach());

    // Originally we were checking if the approaches got freed.
    // We are now using a smart pointer, so it shouldn't be freed anymore
    // as we are still holding a reference to `approach0`, `approach1`, `approach2` above!
    {
        InterleavedMultistartScheme ims([&factory](size_t s){ return factory.fn(s); }, std::make_shared<GenerationalApproachComparator>(), 4, 4, 2);
        ims.setPopulation(pop);
        ims.registerData();
        ims.afterRegisterData();
        ims.run();
    }
}

// Disable on MSVC: Cannot deal with IN_SEQUENCE of more than 2 items.
#if !_MSC_VER
TEST_CASE("InterleavedMultistartScheme: With termination")
{
    using trompeloeil::_;

    std::shared_ptr<Population> pop = std::make_shared<Population>();
    InterleavedMultistartSchemeTestFactory factory;

    auto comparator = std::make_shared<MockGenerationalApproachComparator>();

    auto approach0 = std::shared_ptr<MockGenerationalApproach>(new MockGenerationalApproach());
    auto approach1 = std::shared_ptr<MockGenerationalApproach>(new MockGenerationalApproach());
    auto approach2 = std::shared_ptr<MockGenerationalApproach>(new MockGenerationalApproach());

    // Each call to the factory should return the next approach up.
    // Furthermore, each should have been called with an increasing population size;
    trompeloeil::sequence seq, o0, o1, o2, c0, c1, c2, c3, c4;
    
    // Default to the original behavior.
    ALLOW_CALL(*comparator, clear()).IN_SEQUENCE(c0);
    ALLOW_CALL(*comparator, compare(_, _)).RETURN(0).IN_SEQUENCE(c1);

    // approach 0 is added
    REQUIRE_CALL(factory, fn(4ULL))
        .RETURN(approach0)
        .IN_SEQUENCE(seq, o0);
    // approach 1 is added
    REQUIRE_CALL(factory, fn(8ULL))
        .RETURN(approach1)
        .IN_SEQUENCE(o1);
    // approach2 is added
    REQUIRE_CALL(factory, fn(16ULL))
        .RETURN(approach2)
        .IN_SEQUENCE(o2);

    // Approaches should be set up before use
    REQUIRE_CALL(*approach0, setPopulation(_)).IN_SEQUENCE(o0);
    REQUIRE_CALL(*approach0, registerData()).IN_SEQUENCE(o0);
    REQUIRE_CALL(*approach0, afterRegisterData()).IN_SEQUENCE(o0);
    
    // Other two approaches cannot be configured this way as individuals have been created
    REQUIRE_CALL(*approach1, setPopulation(_)).IN_SEQUENCE(o1);

    REQUIRE_CALL(*approach2, setPopulation(_)).IN_SEQUENCE(o2);
        
    // Expected sequence for size 3 elements.
    // Comments starting with # are utilize internals of the
    // implementation (and are not part of the specification!)
    // # first loop - initialize a0, step a0 three times.
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq, o0);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    // # second loop - initialize a1, step a0 once, a1 once.
    // # i.e. start with current generation.
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach1, step()).IN_SEQUENCE(seq, o1);
    // # recursive call to step a0 thrice.
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    // # - 1
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach1, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq); 
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    // # - 2
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach1, step()).IN_SEQUENCE(seq);
    REQUIRE_CALL(*approach0, step()).IN_SEQUENCE(seq, c0, c1, c2, c3, c4);
    // Difference: decide to have approach 0 be worse than approach 1 at this point.
    //   The comparator will tell the IMS that this is the case 
    //   and approach 0 should hence be stopped.
    ALLOW_CALL(*comparator, clear()).IN_SEQUENCE(c0);
    ALLOW_CALL(*comparator, compare(_, _))
        .WITH(_1.get() != approach0.get() || _2.get() != approach0.get())
        .RETURN(0).IN_SEQUENCE(c1);
    ALLOW_CALL(*comparator, compare(_, _))
        .WITH(_1.get() == approach0.get())
        .RETURN(2).IN_SEQUENCE(c2);
    ALLOW_CALL(*comparator, compare(_, _))
        .WITH(_2.get() == approach0.get())
        .RETURN(1).IN_SEQUENCE(c3);
    // # - 3
    // # approach 2 is initialized.
    REQUIRE_CALL(*approach1, step()).IN_SEQUENCE(seq);
    // Finally, the last call, throw an exception to stop the runner.
    REQUIRE_CALL(*approach2, step()).IN_SEQUENCE(seq, o2).THROW(stop_approach());

    // Originally we were checking if the approaches got freed.
    // We are now using a smart pointer, so it shouldn't be freed anymore
    // as we are still holding a reference to `approach0`, `approach1`, `approach2` above!
    {
        InterleavedMultistartScheme ims([&factory](size_t s){ return factory.fn(s); }, comparator, 4, 4, 2);
        ims.setPopulation(pop);
        ims.registerData();
        ims.afterRegisterData();
        ims.run();
    }
}
#endif

TEST_CASE("SimpleConfigurator")
{
    using trompeloeil::_;

    std::shared_ptr<MockRunner> rm(new MockRunner());
    std::shared_ptr<MockObjectiveFunction> of(new MockObjectiveFunction());

    trompeloeil::sequence init_approach, init_objective, i_aoa, i_oao;
    REQUIRE_CALL(*rm, setPopulation(_)).IN_SEQUENCE(init_approach, i_aoa);
    REQUIRE_CALL(*of, setPopulation(_)).IN_SEQUENCE(init_objective, i_oao);

    REQUIRE_CALL(*rm, registerData()).IN_SEQUENCE(init_approach, i_oao);
    REQUIRE_CALL(*of, registerData()).IN_SEQUENCE(init_objective, i_aoa);

    REQUIRE_CALL(*rm, afterRegisterData()).IN_SEQUENCE(init_approach, i_aoa);
    REQUIRE_CALL(*of, afterRegisterData()).IN_SEQUENCE(init_objective, i_oao);

    SimpleConfigurator configurator(of, rm, 42);
}

TEST_CASE("Integration Test: GOMEA on OneMax (via full API)")
{
    auto init = std::make_shared<CategoricalProbabilisticallyCompleteInitializer>();
    auto foslearner = std::make_shared<CategoricalLinkageTree>(std::make_shared<NMI>(), FoSOrdering::AsIs);
    auto criterion = std::make_shared<SingleObjectiveAcceptanceCriterion>();
    std::shared_ptr<IArchive> archive(new BruteforceArchive({0}));
    // auto gom = std::make_shared<GOM>(criterion);
    // auto fi = std::make_shared<FI>(criterion);
    auto gomea = std::make_shared<GOMEA>(256, init, foslearner, criterion, archive);
    auto stepper = std::make_shared<TerminationStepper>([gomea](){ return gomea; });
    
    auto problem = std::make_shared<OneMax>(100);
    auto problem_limited = std::make_shared<Limiter>(problem, 10000);
    
    auto configurator = SimpleConfigurator(problem_limited, stepper, 42);
    configurator.run();
}