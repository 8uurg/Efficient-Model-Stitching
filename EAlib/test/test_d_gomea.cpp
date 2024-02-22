#include <catch2/catch.hpp>
#include <memory>

#include "acceptation_criteria.hpp"
#include "archive.hpp"
#include "base.hpp"
#include "d-gomea.hpp"
#include "decentralized.hpp"
#include "gomea.hpp"
#include "initializers.hpp"
#include "problems.hpp"
#include "test/mocks.hpp"
#include "test/mocks_archive.hpp"
#include "test/mocks_base_ea.hpp"
#include "test/mocks_decentralized.hpp"
#include "test/mocks_gomea.hpp"
#include "trompeloeil.hpp"

TEST_CASE("StateMachineGOM", "[Decentralized]")
{
    using trompeloeil::_;

    auto mteq = std::make_shared<MockTaggedEventQueue>();
    Scheduler wd(mteq);
    auto pop = std::make_shared<Population>();

    MockObjectiveFunction mof;
    GObjectiveFunction gof(&mof);
    pop->registerGlobalData(gof);

    std::vector<size_t> evaluating_callback_tags;

    bool completed = false;
    bool changed = false;
    bool improved = false;
    bool limited = false;

    MockSamplingDistribution msd;
    // All resamplings are successful (but actually do nothing)

    MockAcceptanceCriterion mac;

    GenericMockFunction rip;
    GenericMockFunction ct;

    FoS fos = {{0}, {1}};

    Individual to_improve = pop->newIndividual();

    auto replace_in_population = [&rip](Individual &, Individual &) { rip.mock_function(); };

    size_t completion_tag = wd.get_tag(std::make_unique<FunctionalResumable>([&completed, &ct](Scheduler &) {
        completed = true;
        ct.mock_function();
        return false;
    }));

    auto sm_gom = std::make_unique<StateMachineGOM>(
        *pop, to_improve, &fos, &msd, &mac, replace_in_population, completion_tag, false, &changed, &improved, &limited);

    // With all this setup ready to go, the expected order of operations is this.
    // For each FoS Element
    // | 1: GOM initially requests the sampling distribution to replace a subset, parameterized by a fos element.
    // |  - As this is mocked: nothing actually happens.
    // | 2: Then, an evaluation is requested using the async method.
    // |    i.e. a tag is placed in the tag_stack.
    // |  - As this is mocked: the tag in the tag_stack should be added
    // |    to `evaluating_callback_tags`
    // > 3: Control should resume to this test.
    // < We invoke the callback in the scheduler.
    // | 4: Following the evaluation, the Acceptance Criterion assesses the change.
    // |  - As this too, is mocked, and simply accepts all changes.
    // | 5: If this was the final element `replace_in_population` is called
    // |    independent of prior success.
    //
    // 6: Once all FoS elements have been processed
    // the completion callback is executed.

    REQUIRE(!completed);

    {
        trompeloeil::sequence s;
        // 1
        REQUIRE_CALL(msd, apply_resample(_, _)).IN_SEQUENCE(s).RETURN(true);
        // 2
        REQUIRE_CALL(mof, evaluate(_)).IN_SEQUENCE(s).LR_SIDE_EFFECT({
            // Move the tag to our callback vector.
            evaluating_callback_tags.push_back(wd.tag_stack.back());
            wd.tag_stack.pop_back();
        });

        wd.schedule_immediately(std::move(sm_gom), 0.0);
        wd.step();
    }
    // Check if the evaluation has been requested.
    REQUIRE(evaluating_callback_tags.size() == 1);
    // Note: at this point no changes or improvements are
    //       definitive yet.
    REQUIRE(!changed);
    REQUIRE(!improved);
    REQUIRE(!completed);

    {
        trompeloeil::sequence s;
        // 4
        REQUIRE_CALL(mac, compare(_, _)).IN_SEQUENCE(s).RETURN(2);
        // 5 - skipped, not applicable.
        // Next fos element.
        // 1
        REQUIRE_CALL(msd, apply_resample(_, _)).IN_SEQUENCE(s).RETURN(true);
        // 2
        REQUIRE_CALL(mof, evaluate(_)).IN_SEQUENCE(s).LR_SIDE_EFFECT({
            // Move the tag to our callback vector.
            evaluating_callback_tags.push_back(wd.tag_stack.back());
            wd.tag_stack.pop_back();
        });
        // Continue GOM with aforementioned expectations being checked.
        wd.complete_tag(evaluating_callback_tags.back());
    }

    // Again, the next evaluation should be requested now
    REQUIRE(evaluating_callback_tags.size() == 2);
    REQUIRE(changed);
    REQUIRE(improved);
    REQUIRE(!completed);

    {
        trompeloeil::sequence s;
        // 4
        REQUIRE_CALL(mac, compare(_, _)).IN_SEQUENCE(s).RETURN(2);
        // 5
        REQUIRE_CALL(rip, mock_function()).IN_SEQUENCE(s);
        // 6: As there is no next FOS element. :)
        REQUIRE_CALL(ct, mock_function()).IN_SEQUENCE(s);
        // Finish up GOM with aforementioned expectations being checked.
        wd.complete_tag(evaluating_callback_tags.back());
    }

    REQUIRE(changed);
    REQUIRE(improved);
    REQUIRE(completed);
}

TEST_CASE("StateMachineFI", "[Decentralized]")
{
    using trompeloeil::_;

    auto mteq = std::make_shared<MockTaggedEventQueue>();
    Scheduler wd(mteq);
    auto pop = std::make_shared<Population>();

    MockObjectiveFunction mof;
    GObjectiveFunction gof(&mof);
    pop->registerGlobalData(gof);

    std::vector<size_t> evaluating_callback_tags;

    bool completed = false;
    bool changed = false;
    bool improved = false;
    bool limited = false;

    MockSamplingDistribution msd;
    // All resamplings are successful (but actually do nothing)

    MockAcceptanceCriterion mac;

    GenericMockFunction rip;
    GenericMockFunction ct;

    FoS fos = {{0}, {1}, {2}};

    Individual to_improve = pop->newIndividual();

    auto replace_in_population = [&rip](Individual &, Individual &) { rip.mock_function(); };

    size_t completion_tag = wd.get_tag(std::make_unique<FunctionalResumable>([&completed, &ct](Scheduler &) {
        completed = true;
        ct.mock_function();
        return false;
    }));

    auto sm_gom = std::make_unique<StateMachineFI>(
        *pop, to_improve, &fos, &msd, &mac, replace_in_population, completion_tag, &changed, &improved, &limited);

    // With all this setup ready to go, the expected order of operations is this.
    // For each FoS Element, until success
    // | 1: FI initially requests the sampling distribution to replace a subset, parameterized by a fos element.
    // |  - As this is mocked: nothing actually happens in reality, but FI does not know that.
    // | 2: Then, an evaluation is requested using the async method.
    // |    i.e. a tag is placed in the tag_stack.
    // |  - As this is mocked: the tag in the tag_stack should be added
    // |    to `evaluating_callback_tags`
    // > 3: Control should resume to this test.
    // < We invoke the callback in the scheduler.
    // | 4: Following the evaluation, the Acceptance Criterion assesses the change.
    // |  - As this too, is mocked. First is worse, second accepts. Third should never happen.
    // | 5: On success, `replace_in_population` is called.
    //
    // 6: Once all FoS elements have been processed or success has been obtained.
    //    the completion callback is executed.

    REQUIRE(!completed);

    SECTION("2nd step is successful")
    {
        bool pop_element = GENERATE(0, 1);
        if (pop_element)
        {
            fos.pop_back();
        }
        DYNAMIC_SECTION("Popped element: " + std::to_string(pop_element))
        {

        {
            trompeloeil::sequence s;
            // 1
            REQUIRE_CALL(msd, apply_resample(_, _)).IN_SEQUENCE(s).RETURN(true);
            // 2
            REQUIRE_CALL(mof, evaluate(_)).IN_SEQUENCE(s).LR_SIDE_EFFECT({
                // Move the tag to our callback vector.
                evaluating_callback_tags.push_back(wd.tag_stack.back());
                wd.tag_stack.pop_back();
            });

            wd.schedule_immediately(std::move(sm_gom), 0.0);
            wd.step();
        }
        // Check if the evaluation has been requested.
        REQUIRE(evaluating_callback_tags.size() == 1);
        REQUIRE(!completed);

        {
            trompeloeil::sequence s;
            // 4
            REQUIRE_CALL(mac, compare(_, _)).IN_SEQUENCE(s).RETURN(1);
            // 5 - skipped, not applicable.
            // Next fos element.
            // 1
            REQUIRE_CALL(msd, apply_resample(_, _)).IN_SEQUENCE(s).RETURN(true);
            // 2
            REQUIRE_CALL(mof, evaluate(_)).IN_SEQUENCE(s).LR_SIDE_EFFECT({
                // Move the tag to our callback vector.
                evaluating_callback_tags.push_back(wd.tag_stack.back());
                wd.tag_stack.pop_back();
            });
            // Continue FI with aforementioned expectations being checked.
            wd.complete_tag(evaluating_callback_tags.back());
        }

        // Again, the next evaluation should be requested now
        REQUIRE(evaluating_callback_tags.size() == 2);
        REQUIRE(!completed);

        {
            trompeloeil::sequence s;
            // 4
            REQUIRE_CALL(mac, compare(_, _)).IN_SEQUENCE(s).RETURN(2);
            // 5: We got success!
            REQUIRE_CALL(rip, mock_function()).IN_SEQUENCE(s);
            // 6
            REQUIRE_CALL(ct, mock_function()).IN_SEQUENCE(s);
            // Finish up GOM with aforementioned expectations being checked.
            wd.complete_tag(evaluating_callback_tags.back());
        }
        // No more evaluation requests should be performed.
        REQUIRE(evaluating_callback_tags.size() == 2);

        REQUIRE(completed);
        }
    }

    SECTION("None are successful")
    {
        {
            trompeloeil::sequence s;
            // 1
            REQUIRE_CALL(msd, apply_resample(_, _)).IN_SEQUENCE(s).RETURN(true);
            // 2
            REQUIRE_CALL(mof, evaluate(_)).IN_SEQUENCE(s).LR_SIDE_EFFECT({
                // Move the tag to our callback vector.
                evaluating_callback_tags.push_back(wd.tag_stack.back());
                wd.tag_stack.pop_back();
            });

            wd.schedule_immediately(std::move(sm_gom), 0.0);
            wd.step();
        }
        // Check if the evaluation has been requested.
        REQUIRE(evaluating_callback_tags.size() == 1);
        REQUIRE(!completed);

        {
            trompeloeil::sequence s;
            // 4
            REQUIRE_CALL(mac, compare(_, _)).IN_SEQUENCE(s).RETURN(1);
            // 5 - skipped, not applicable.
            // Next fos element.
            // 1
            REQUIRE_CALL(msd, apply_resample(_, _)).IN_SEQUENCE(s).RETURN(true);
            // 2
            REQUIRE_CALL(mof, evaluate(_)).IN_SEQUENCE(s).LR_SIDE_EFFECT({
                // Move the tag to our callback vector.
                evaluating_callback_tags.push_back(wd.tag_stack.back());
                wd.tag_stack.pop_back();
            });
            // Continue FI with aforementioned expectations being checked.
            wd.complete_tag(evaluating_callback_tags.back());
        }

        // Again, the next evaluation should be requested now
        REQUIRE(evaluating_callback_tags.size() == 2);
        REQUIRE(!completed);

        {
            trompeloeil::sequence s;
            // 4
            REQUIRE_CALL(mac, compare(_, _)).IN_SEQUENCE(s).RETURN(1);
            // 5 - skipped, not applicable.
            // Next fos element.
            // 1
            REQUIRE_CALL(msd, apply_resample(_, _)).IN_SEQUENCE(s).RETURN(true);
            // 2
            REQUIRE_CALL(mof, evaluate(_)).IN_SEQUENCE(s).LR_SIDE_EFFECT({
                // Move the tag to our callback vector.
                evaluating_callback_tags.push_back(wd.tag_stack.back());
                wd.tag_stack.pop_back();
            });
            // Continue FI with aforementioned expectations being checked.
            wd.complete_tag(evaluating_callback_tags.back());
        }

        // The third request should be made now..
        REQUIRE(evaluating_callback_tags.size() == 3);

        {
            trompeloeil::sequence s;
            // 4
            REQUIRE_CALL(mac, compare(_, _)).IN_SEQUENCE(s).RETURN(1);
            // 5 - skipped: not successful.
            // 6
            REQUIRE_CALL(ct, mock_function()).IN_SEQUENCE(s);
            // We have ran out of FoS elements
            wd.complete_tag(evaluating_callback_tags.back());
        }

        // No more requests.
        REQUIRE(evaluating_callback_tags.size() == 3);
        REQUIRE(completed);
    }
}

class MockKernelData : public trompeloeil::mock_interface<IGOMFIData>
{
    IMPLEMENT_MOCK0(getFOSForGOM);
    IMPLEMENT_MOCK0(getDistributionForGOM);
    IMPLEMENT_MOCK0(getPerformanceCriterionForGOM);
    IMPLEMENT_MOCK0(getFOSForFI);
    IMPLEMENT_MOCK0(getDistributionForFI);
    IMPLEMENT_MOCK0(getPerformanceCriterionForFI);
};

TEST_CASE("DistributedGOMThenMaybeFI", "[Decentralized]")
{
    using trompeloeil::_;

    bool has_events = true;
    auto mteq = std::make_shared<MockTaggedEventQueue>();
    ALLOW_CALL(*mteq, next(_, _)).LR_SIDE_EFFECT(has_events = false).RETURN(false);

    Scheduler wd(mteq);
    auto pop = std::make_shared<Population>();
    pop->registerData<NIS>();

    MockObjectiveFunction mof;
    GObjectiveFunction gof(&mof);
    pop->registerGlobalData(gof);

    std::vector<size_t> evaluating_callback_tags;

    bool completed_a = false;
    bool completed_b = false;

    MockSamplingDistribution msd;
    auto msd_ptr = &msd;
    MockAcceptanceCriterion mac_gom;
    auto mac_gom_ptr = &mac_gom;
    MockAcceptanceCriterion mac_fi;
    auto mac_fi_ptr = &mac_fi;

    Individual replacement = pop->newIndividual();
    size_t nis_v = 1;
    auto getNISThreshold = [&nis_v](Individual &) { return nis_v; };

    bool requested_replacement = false;
    auto getReplacementSolution = [replacement, &requested_replacement](Individual &) {
        requested_replacement = true;
        return replacement;
    };

    FoS fos = {{0}, {1}, {2}};
    auto fos_ptr = &fos;

    Individual to_improve = pop->newIndividual();

    auto replace_in_population = [](Individual &, Individual &) {};

    size_t completion_tag = wd.get_tag(std::make_unique<FunctionalResumable>([&completed_a](Scheduler &) {
        completed_a = true;
        return false;
    }));

    auto mkd = std::make_unique<MockKernelData>();
    auto &mkdr = *mkd;

    auto onCompletion = [&completed_b](Scheduler &) { completed_b = true; };

    // &fos, &msd, &mac,

    auto sm_gom = std::make_unique<DistributedGOMThenMaybeFI>(*pop,
                                                              to_improve,
                                                              completion_tag,
                                                              getNISThreshold,
                                                              getReplacementSolution,
                                                              replace_in_population,
                                                              onCompletion,
                                                              std::move(mkd),
                                                              false);

    trompeloeil::sequence s0, s1, s2;

    // Sampling is always successful
    ALLOW_CALL(msd, apply_resample(_, _)).RETURN(true);

    // Allow GOM and then FI to be performed.
    ALLOW_CALL(mkdr, getFOSForGOM()).IN_SEQUENCE(s0).RETURN(fos_ptr);
    ALLOW_CALL(mkdr, getFOSForFI()).IN_SEQUENCE(s0).RETURN(fos_ptr);
    ALLOW_CALL(mkdr, getDistributionForGOM()).IN_SEQUENCE(s1).RETURN(msd_ptr);
    ALLOW_CALL(mkdr, getDistributionForFI()).IN_SEQUENCE(s1).RETURN(msd_ptr);
    ALLOW_CALL(mkdr, getPerformanceCriterionForGOM()).IN_SEQUENCE(s2).RETURN(mac_gom_ptr);
    ALLOW_CALL(mkdr, getPerformanceCriterionForFI()).IN_SEQUENCE(s2).RETURN(mac_fi_ptr);

    ALLOW_CALL(mof, evaluate(_)).LR_SIDE_EFFECT({
        size_t tag = wd.tag_stack.back();
        wd.tag_stack.pop_back();
        wd.schedule_immediately(std::make_unique<FunctionalResumable>([tag](Scheduler &wd) {
                                    wd.complete_tag(tag);
                                    return false;
                                }),
                                0.0);
    });

    SECTION("GOM is successful")
    {
        // GOM is always successful!
        REQUIRE_CALL(mac_gom, compare(_, _)).TIMES(AT_LEAST(1)).RETURN(2);

        // FI is skipped if GOM is successful: running until termination should
        // not cause any calls to FI-related components.
        wd.schedule_immediately(std::move(sm_gom), 0.0);
        // Step-a-thon
        while (has_events)
        {
            wd.step();
        }
        // No forbidden (read: FI related) calls should occur.
        FORBID_CALL(mkdr, getFOSForFI());
        FORBID_CALL(mkdr, getDistributionForFI());
        FORBID_CALL(mkdr, getPerformanceCriterionForFI());
        REQUIRE(completed_a);
        REQUIRE(completed_b);
    }

    SECTION("GOM fails, FI is successful")
    {
        // GOM fails
        REQUIRE_CALL(mac_gom, compare(_, _)).TIMES(AT_LEAST(1)).RETURN(1);
        // FI is successful
        REQUIRE_CALL(mac_fi, compare(_, _)).TIMES(AT_LEAST(1)).RETURN(2);

        // FI is skipped if GOM is successful: running until termination should
        // not cause any calls to FI-related components.
        wd.schedule_immediately(std::move(sm_gom), 0.0);
        // Step-a-thon
        while (has_events)
        {
            wd.step();
        }
        // No forbidden (read: FI related) calls should occur.
        REQUIRE(completed_a);
        REQUIRE(completed_b);
    }
    SECTION("GOM & FI fail")
    {
        // GOM fails
        REQUIRE_CALL(mac_gom, compare(_, _)).TIMES(AT_LEAST(1)).RETURN(1);
        // FI fails similarly
        REQUIRE_CALL(mac_fi, compare(_, _)).TIMES(AT_LEAST(1)).RETURN(1);

        // FI is skipped if GOM is successful: running until termination should
        // not cause any calls to FI-related components.
        wd.schedule_immediately(std::move(sm_gom), 0.0);
        // Step-a-thon
        while (has_events)
        {
            wd.step();
        }
        // No forbidden (read: FI related) calls should occur.
        REQUIRE(completed_a);
        REQUIRE(completed_b);
        REQUIRE(requested_replacement);
    }
}

TEST_CASE("DistributedSynchronousGOMEA", "[Decentralized]")
{
    using trompeloeil::_;
    auto pop = std::make_shared<Population>();

    // Sadly, some core parts use this.
    Rng rng(42);
    pop->registerGlobalData(rng);

    const size_t population_size = 16;
    auto mteq = std::make_shared<MockTaggedEventQueue>();
    auto wd = std::make_shared<Scheduler>(mteq);

    auto mof = std::make_shared<MockObjectiveFunction>();
    pop->registerGlobalData(GObjectiveFunction{mof.get()});

    auto msi = std::make_shared<MockSolutionInitializer>();
    auto mfl = std::make_shared<MockFoSLearner>();
    auto mac = std::make_shared<MockAcceptanceCriterion>();
    auto ma = std::make_shared<MockArchive>();

    auto msd_gom = std::make_shared<MockSamplingDistribution>();
    auto *msd_gom_ptr = msd_gom.get();
    auto msd_fi = std::make_shared<MockSamplingDistribution>();
    auto *msd_fi_ptr = msd_fi.get();

    DistributedSynchronousGOMEA gomea(
        wd,
        population_size,
        msi,
        mfl,
        mac,
        ma,
        false,
        false,
        [msd_gom_ptr](Population &, GenerationalData &, BaseGOMEA *) { return msd_gom_ptr; },
        [msd_fi_ptr](Population &, GenerationalData &, BaseGOMEA *) { return msd_fi_ptr; });

    // Initialization steps!
    {
        REQUIRE_CALL(*msi, setPopulation(_)).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mfl, setPopulation(_)).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mac, setPopulation(_)).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*ma, setPopulation(_)).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mteq, setPopulation(_)).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*msi, registerData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mfl, registerData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mac, registerData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*ma, registerData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mteq, registerData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*msi, afterRegisterData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mfl, afterRegisterData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mac, afterRegisterData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*ma, afterRegisterData()).TIMES(AT_LEAST(1));
        REQUIRE_CALL(*mteq, afterRegisterData()).TIMES(AT_LEAST(1));

        gomea.setPopulation(pop);
        gomea.registerData();
        gomea.afterRegisterData();
    }

    // First step: initialization.
    size_t complete_this_tag_to_start_next_generation = 0;
    {
        trompeloeil::sequence s;
        // should initialize first
        REQUIRE_CALL(*msi, initialize(_)).WITH(_1.size() == population_size).IN_SEQUENCE(s);

        // And request evaluations.
        std::vector<size_t> request_eval_tags;
        REQUIRE_CALL(*mof, evaluate(_))
            .LR_SIDE_EFFECT({
                // Remove callback tags.
                request_eval_tags.push_back(wd->tag_stack.back());
                wd->tag_stack.pop_back();
            })
            .TIMES(population_size)
            .IN_SEQUENCE(s);

        // Stepping should cause these behaviors & return.
        gomea.step();

        REQUIRE(request_eval_tags.size() == population_size);

        // The first n - 1 completions shouldn't change anything.
        for (size_t idx = 0; idx < population_size - 1; idx++)
        {
            // Completing the evaluation results in an attempt to insert
            // the solution into the archive.
            REQUIRE_CALL(*ma, try_add(_)).RETURN(Archived{false, false, 0, {}});

            auto tag = request_eval_tags[idx];
            // 'complete' the evaluation
            wd->complete_tag(tag);
            // Stepping the scheduler shouldn't do anything.
            // The approach is waiting
            REQUIRE_CALL(*mteq, next(_, _)).RETURN(false);
            wd->step();
        }
        // The next one completes the generation however,
        // and immediately starts the next generation.
        complete_this_tag_to_start_next_generation = request_eval_tags.back();
    }

    // Now we check the following generation.
    {
        std::vector<size_t> request_eval_tags;
        // Annoyingly enough: we cannot actually provide the sampling operator.
        // That needs to be fixed, maybe?
        // - First of all, it completes the previous generation.
        ALLOW_CALL(*ma, try_add(_)).RETURN(Archived{false, false, 0, {}});
        // - Second, it learns a new linkage model.
        REQUIRE_CALL(*mfl, learnFoS(_));
        FoS fos = {{0}};
        ALLOW_CALL(*mfl, getFoS()).LR_RETURN(fos);
        // - Third, it starts the GOM & FI for all of the solutions,
        //   (Note: may not be started upon completion immediately )
        //   ( and require additional queue steps.                 )
        ALLOW_CALL(*msd_gom, apply_resample(_, _)).RETURN(true);

        REQUIRE_CALL(*mof, evaluate(_))
            .LR_SIDE_EFFECT({
                // Remove callback tags.
                request_eval_tags.push_back(wd->tag_stack.back());
                wd->tag_stack.pop_back();
            })
            .TIMES(population_size);

        ALLOW_CALL(*mac, compare(_, _)).RETURN(2);
        wd->complete_tag(complete_this_tag_to_start_next_generation);
        // Clear the queue, just in case.
        wd->stepUntilEmptyQueue();
    }
}

TEST_CASE("Integration Test: DistributedSynchronousGOMEA on OneMax", "[Decentralized][Integration]")
{
    // This test relies on convergence, and that it truly stops in such a case.
    // Sadly, setting time limits is not particularly possible. So if this takes more than 10s, something
    // is probably wrong.
    // Additional aspects under test include hitting a limit (!)

    auto pop = std::make_shared<Population>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    const size_t population_size =  GENERATE(16, 128);
    size_t budget = GENERATE(1000000, 100);
    auto mteq = std::make_shared<GRPCEventQueue>();
    auto wd = std::make_shared<Scheduler>(mteq);
    // Is called properly by remote evaluation class.
    // wd->setPopulation(pop);
    // wd->registerData();
    // wd->afterRegisterData();

    auto e_problem = std::make_shared<OneMax>(100);
    e_problem->setPopulation(pop);
    e_problem->registerData();
    e_problem->afterRegisterData();

    int port = 0;
    RemoteProblemEvaluatorService rpes(e_problem);
    auto server = rpes.start_server("127.0.0.1:0", &port);
    std::string e_host = "127.0.0.1:" + std::to_string(port);

    auto c_problem = std::make_shared<RemoteAsyncObjectiveFunction>(wd, e_host);
    auto l_problem = std::make_shared<Limiter>(c_problem, budget);
    l_problem->setPopulation(pop);
    l_problem->registerData();
    pop->registerGlobalData(GObjectiveFunction { l_problem.get() });

    auto initializer = std::make_shared<CategoricalProbabilisticallyCompleteInitializer>();
    auto foslearner = std::make_shared<CategoricalUnivariateFoS>();
    auto performance_criterion = std::make_shared<SingleObjectiveAcceptanceCriterion>();
    std::vector<size_t> indices = { 0 };
    auto archive = std::make_shared<BruteforceArchive>(indices);

    DistributedSynchronousGOMEA dsg(wd, population_size, initializer, foslearner, performance_criterion, archive);
    dsg.setPopulation(pop);
    dsg.registerData();
    l_problem->afterRegisterData();
    dsg.afterRegisterData();

    auto s = pop->getGlobalData<SharedgRPCCompletionQueue>();

    DYNAMIC_SECTION("Population Size: " << population_size << " & Budget: " << budget)
    {
        bool cause_is_initial_exception = false;
        bool cause_is_exception = false;
        try
        {
            dsg.step();
        }
        catch (evaluation_limit_reached &e)
        {
            cause_is_initial_exception = true;
            cause_is_exception = true;
        }
        
        if (budget < population_size) {
            // Only evaluate as many as we are allowed to, not fewer, not more.
            REQUIRE(s->pending == budget);
        }

        while(! wd->terminated() && ! cause_is_initial_exception)
        {
            try
            {
                dsg.step();
            }
            catch (evaluation_limit_reached &e)
            {
                cause_is_exception = true;
                break;
            }
        }

        if (budget == 100)
        {
            // As we now soft-stop in case of an exception, it shouldn't be stopped
            // by an exception. 
            REQUIRE(! cause_is_exception);
        }
        if (budget == 100 && population_size > 100)
        {
            // If budget was insufficient
            // But as in the previous case: no exceptions.
            REQUIRE(! cause_is_initial_exception);
        }

        server->Shutdown();

        auto &archived = archive->get_archived();
        if (archived.size() > 0)
        {
            std::cout << "Best objective obtained: " << pop->getData<Objective>(archived[0]).objectives[0] << std::endl;
        }
        else
        {
            std::cout << "No solutions have finished evaluating before evaluation limit was hit." << std::endl;
        }
    }
}

TEST_CASE("Integration Test: DistributedAsynchronousGOMEA on OneMax", "[Decentralized][Integration]")
{
    // This test relies on convergence, and that it truly stops in such a case.
    // Sadly, setting time limits is not particularly possible. So if this takes more than 10s, something
    // is probably wrong.
    // Additional aspects under test include hitting a limit (!)

    auto pop = std::make_shared<Population>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    // 2048
    const size_t population_size = GENERATE(16, 128);
    size_t budget = GENERATE(1000000, 100);
    auto mteq = std::make_shared<GRPCEventQueue>();
    auto wd = std::make_shared<Scheduler>(mteq);
    
    // Is called properly by remote evaluation class.
    // wd->setPopulation(pop);
    // wd->registerData();
    // wd->afterRegisterData();

    auto e_problem = std::make_shared<OneMax>(100);
    e_problem->setPopulation(pop);
    e_problem->registerData();
    e_problem->afterRegisterData();

    int port = 0;
    RemoteProblemEvaluatorService rpes(e_problem);
    auto server = rpes.start_server("127.0.0.1:0", &port);
    std::string e_host = "127.0.0.1:" + std::to_string(port);

    auto c_problem = std::make_shared<RemoteAsyncObjectiveFunction>(wd, e_host);
    auto l_problem = std::make_shared<Limiter>(c_problem, budget);

    auto initializer = std::make_shared<CategoricalProbabilisticallyCompleteInitializer>();
    auto foslearner = std::make_shared<CategoricalUnivariateFoS>();
    auto performance_criterion = std::make_shared<SingleObjectiveAcceptanceCriterion>();
    std::vector<size_t> indices = { 0 };
    auto archive = std::make_shared<BruteforceArchive>(indices);
    // 
    size_t number_of_clusters = 1;
    std::vector<size_t> objective_indices = {0};
    DistributedAsynchronousGOMEA dsg(wd, population_size, number_of_clusters, objective_indices, initializer, foslearner, performance_criterion, archive);
    dsg.setPopulation(pop);
    l_problem->setPopulation(pop);
    dsg.registerData();
    l_problem->registerData();
    pop->registerGlobalData(GObjectiveFunction { l_problem.get() });
    dsg.afterRegisterData();
    l_problem->afterRegisterData();

    size_t prev_num_ids = 0;
    int num_greater = 0;
    int num_lesser = 0;

    auto s = pop->getGlobalData<SharedgRPCCompletionQueue>();

    DYNAMIC_SECTION("Population Size: " << population_size << " & Budget: " << budget)
    {
        bool cause_is_initial_exception = false;
        bool cause_is_exception = false;
        try
        {
            dsg.step();
        }
        catch (evaluation_limit_reached &e)
        {
            cause_is_initial_exception = true;
            cause_is_exception = true;
        }

        if (budget < population_size) {
            // Only evaluate as many as we are allowed to, not fewer, not more.
            REQUIRE(s->pending == budget);
        }


        while(! wd->terminated() && ! cause_is_initial_exception)
        {
            try
            {
                dsg.step();
                if (pop->size() > prev_num_ids)
                {
                    num_greater++;
                }
                else
                {
                    num_lesser++;
                }
                prev_num_ids = pop->size();
            }
            catch (evaluation_limit_reached &e)
            {
                cause_is_exception = true;
                break;
            }
        }

        if (budget == 100)
        {
            REQUIRE(! cause_is_exception);
        }
        if (budget == 100 && population_size > 100)
        {
            REQUIRE(! cause_is_initial_exception);
        }

        // There shouldn't be a clear increase in the number of individuals over time over a long run.
        std::cout << "Got " << num_greater << " increases and " << num_lesser << " decreases." << std::endl;
        REQUIRE(static_cast<double>(num_greater) / static_cast<double>(num_greater + num_lesser) < 0.7);

        server->Shutdown();

        auto &archived = archive->get_archived();
        if (archived.size() > 0)
        {
            std::cout << "Best objective obtained: " << pop->getData<Objective>(archived[0]).objectives[0] << std::endl;
        }
        else
        {
            std::cout << "No solutions have finished evaluating before evaluation limit was hit." << std::endl;
        }
    }
}

TEST_CASE("Integration Test: DistributedAsynchronousKernelGOMEA on OneMax", "[Decentralized][Integration]")
{
    // This test relies on convergence, and that it truly stops in such a case.
    // Sadly, setting time limits is not particularly possible. So if this takes more than 10s, something
    // is probably wrong.
    // Additional aspects under test include hitting a limit (!)

    auto pop = std::make_shared<Population>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    const size_t population_size = GENERATE(16, 128);
    size_t budget = GENERATE(1000000, 100);
    auto mteq = std::make_shared<GRPCEventQueue>();
    auto wd = std::make_shared<Scheduler>(mteq);
    
    // Is called properly by remote evaluation class.
    // wd->setPopulation(pop);
    // wd->registerData();
    // wd->afterRegisterData();

    auto e_problem = std::make_shared<OneMax>(100);
    e_problem->setPopulation(pop);
    e_problem->registerData();
    e_problem->afterRegisterData();

    int port = 0;
    RemoteProblemEvaluatorService rpes(e_problem);
    auto server = rpes.start_server("127.0.0.1:0", &port);
    std::string e_host = "127.0.0.1:" + std::to_string(port);

    auto c_problem = std::make_shared<RemoteAsyncObjectiveFunction>(wd, e_host);
    auto l_problem = std::make_shared<Limiter>(c_problem, budget);

    auto initializer = std::make_shared<CategoricalProbabilisticallyCompleteInitializer>();
    auto foslearner = std::make_shared<CategoricalUnivariateFoS>();
    auto performance_criterion = std::make_shared<SingleObjectiveAcceptanceCriterion>();
    std::vector<size_t> indices = { 0 };
    auto archive = std::make_shared<BruteforceArchive>(indices);
    // 
    size_t number_of_clusters = 1;
    std::vector<size_t> objective_indices = {0};
    DistributedAsynchronousKernelGOMEA dsg(wd, population_size, number_of_clusters, objective_indices, initializer, foslearner, performance_criterion, archive);
    dsg.setPopulation(pop);
    l_problem->setPopulation(pop);
    dsg.registerData();
    l_problem->registerData();
    pop->registerGlobalData(GObjectiveFunction { l_problem.get() });
    dsg.afterRegisterData();
    l_problem->afterRegisterData();

    
    auto s = pop->getGlobalData<SharedgRPCCompletionQueue>();

    DYNAMIC_SECTION("Population Size: " << population_size << " & Budget: " << budget)
    {
        bool cause_is_initial_exception = false;
        bool cause_is_exception = false;
        try
        {
            dsg.step();
        }
        catch (evaluation_limit_reached &e)
        {
            cause_is_initial_exception = true;
            cause_is_exception = true;
        }
        
        if (budget < population_size) {
            // Only evaluate as many as we are allowed to, not fewer, not more.
            REQUIRE(s->pending == budget);
        }

        while(! wd->terminated() && ! cause_is_initial_exception)
        {
            try
            {
                dsg.step();
            }
            catch (evaluation_limit_reached &e)
            {
                cause_is_exception = true;
                break;
            }
        }

        if (budget == 100)
        {
            REQUIRE(! cause_is_exception);
        }
        if (budget == 100 && population_size > 100)
        {
            REQUIRE(! cause_is_initial_exception);
        }

        server->Shutdown();

        auto &archived = archive->get_archived();
        if (archived.size() > 0)
        {
            std::cout << "Best objective obtained: " << pop->getData<Objective>(archived[0]).objectives[0] << std::endl;
        }
        else
        {
            std::cout << "No solutions have finished evaluating before evaluation limit was hit." << std::endl;
        }
    }
}

#include "dispatcher.hpp"
#include <thread>

TEST_CASE("Integration Test: DistributedSynchronousGOMEA on ME OneMax", "[Decentralized][Integration]")
{
    // This test relies on convergence, and that it truly stops in such a case.
    // Sadly, setting time limits is not particularly possible. So if this takes more than 10s, something
    // is probably wrong.
    // Additional aspects under test include hitting a limit (!)

    auto pop = std::make_shared<Population>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    const size_t population_size = 16;
    size_t budget = 10000000;
    auto mteq = std::make_shared<GRPCEventQueue>();
    auto wd = std::make_shared<Scheduler>(mteq);
    // wd->setPopulation(pop);
    // wd->registerData();
    // wd->afterRegisterData();

    auto e_problem = std::make_shared<OneMax>(100);
    // ???
    // e_problem->setPopulation(pop);
    // e_problem->registerData();
    // e_problem->afterRegisterData();

    int port = 0;
    // Note: we need to clone a problem instance as each evaluator has a different population
    // and hence cannot be shared.
    auto problem_c_a = std::shared_ptr<ObjectiveFunction>(e_problem->clone());
    RemoteProblemEvaluatorService rpes_a(problem_c_a);
    auto rpes_a_server = rpes_a.start_server("127.0.0.1:0", &port);
    std::string rpes_a_host = "127.0.0.1:" + std::to_string(port);
    std::cout << "Started evaluator on " << rpes_a_host << std::endl;
    
    // See note for problem_c_a
    auto problem_c_b = std::shared_ptr<ObjectiveFunction>(e_problem->clone());
    RemoteProblemEvaluatorService rpes_b(problem_c_b);
    auto rpes_b_server = rpes_b.start_server("127.0.0.1:0", &port);
    std::string rpes_b_host = "127.0.0.1:" + std::to_string(port);
    std::cout << "Started evaluator on " << rpes_b_host << std::endl;

    DispatcherServer ds("127.0.0.1:0", &port);
    ds.start_server();
    std::string e_host = "127.0.0.1:" + std::to_string(port);
    std::thread server_handler([&ds](){ds.start_handling_requests();});
    std::cout << "Started distributor on " << e_host << std::endl;

    rpes_a.register_with(e_host);
    rpes_b.register_with(e_host);

    auto c_problem = std::make_shared<RemoteAsyncObjectiveFunction>(wd, e_host, e_problem);
    auto l_problem = std::make_shared<Limiter>(c_problem, budget);

    auto initializer = std::make_shared<CategoricalProbabilisticallyCompleteInitializer>();
    auto foslearner = std::make_shared<CategoricalUnivariateFoS>();
    auto performance_criterion = std::make_shared<SingleObjectiveAcceptanceCriterion>();
    std::vector<size_t> indices = { 0 };
    auto archive = std::make_shared<BruteforceArchive>(indices);

    DistributedSynchronousGOMEA dsg(wd, population_size, initializer, foslearner, performance_criterion, archive);
    l_problem->setPopulation(pop);
    dsg.setPopulation(pop);
    dsg.registerData();
    l_problem->registerData();
    dsg.afterRegisterData();
    l_problem->afterRegisterData();
    pop->registerGlobalData(GObjectiveFunction { l_problem.get() });

    dsg.step();
    while(! wd->terminated())
    {
        try
        {
            dsg.step();
        }
        catch (std::exception &e)
        {
            break;
        }
    }
    

    ds.shutdown();
    server_handler.join();
    rpes_a_server->Shutdown();
    rpes_b_server->Shutdown();
    
}
