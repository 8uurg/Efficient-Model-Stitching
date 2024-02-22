#include <memory>
#include <sstream>
#include <thread>

#include <catch2/catch.hpp>
#include <catch2/trompeloeil.hpp>

#include "base.hpp"
#include "cereal/archives/binary.hpp"
#include "decentralized.hpp"
#include "mocks_decentralized.hpp"

TEST_CASE("Scheduler", "[Scheduler]")
{
    auto mteq = std::make_shared<MockTaggedEventQueue>();
    Scheduler s(mteq);

    SECTION("IDataUser is forwarded to the TaggedEventQueue")
    {
        auto pop = std::make_shared<Population>();
        REQUIRE_CALL(*mteq.get(), setPopulation(pop));
        s.setPopulation(pop);

        REQUIRE_CALL(*mteq.get(), registerData());
        s.registerData();

        REQUIRE_CALL(*mteq.get(), afterRegisterData());
        s.afterRegisterData();
    }

    // Simple resumables.
    bool executed_fr1 = false;
    auto fr1 = std::make_unique<FunctionalResumable>([&executed_fr1](Scheduler &) {
        executed_fr1 = true;
        return false;
    });
    bool executed_fr2 = false;
    auto fr2 = std::make_unique<FunctionalResumable>([&executed_fr2](Scheduler &) {
        executed_fr2 = true;
        return false;
    });

    SECTION("Stepping the scheduler when an event is present, the highest priority event should be resumed first, then "
            "the second.")
    {
        s.schedule_immediately(std::move(fr1), 1.0);
        s.schedule_immediately(std::move(fr2), 0.0);

        // First step should execute the first, not the second.
        s.step();
        REQUIRE(executed_fr1);
        REQUIRE(!executed_fr2);

        // Second step should execute the second too.
        s.step();
        REQUIRE(executed_fr1);
        REQUIRE(executed_fr2);
    }

    SECTION("If no events are left to be performed, the tag event queue should be queried.")
    {
        using trompeloeil::_;

        // Register tag
        size_t tag = s.get_tag(std::move(fr1));
        // Next query returns this tag & is successful.
        REQUIRE_CALL(*mteq.get(), next(_, _)).SIDE_EFFECT(*_2 = tag).RETURN(true);

        s.step();

        // Stepping should have also executed this tag.
        REQUIRE(executed_fr1);
    }

    SECTION("If there is nothing to be done, nothing goes wrong")
    {
        using trompeloeil::_;
        REQUIRE_CALL(*mteq.get(), next(_, _)).RETURN(false);

        s.step();
    }

    SECTION(
        "If an event is only scheduled to happen after a message event it should only happen after a message event.")
    {
        using trompeloeil::_;
        s.schedule_after_message_completion(std::move(fr1), 0.0);
        REQUIRE_CALL(*mteq.get(), next(_, _)).RETURN(false);

        s.step();

        REQUIRE(!executed_fr1);

        // Send a message completion event.
        size_t tag = s.get_tag(std::move(fr2), true);
        s.complete_tag(tag);

        s.step();

        REQUIRE(executed_fr1);
    }

    SECTION("An event can be tagged and completed")
    {
        size_t tag1 = s.get_tag(std::move(fr1));
        size_t tag2 = s.get_tag(std::move(fr2));
        
        REQUIRE(!executed_fr1);
        REQUIRE(!executed_fr2);

        s.complete_tag(tag2);
        REQUIRE(!executed_fr1);
        REQUIRE(executed_fr2);

        s.complete_tag(tag1);
        REQUIRE(executed_fr1);
        REQUIRE(executed_fr2);
    }

    SECTION("Tags are reused")
    {
        size_t tag1 = s.get_tag(std::move(fr1));
        s.complete_tag(tag1);
        REQUIRE(executed_fr1);
        REQUIRE(!executed_fr2);

        size_t tag2 = s.get_tag(std::move(fr2));
        s.complete_tag(tag2);
        REQUIRE(executed_fr1);
        REQUIRE(executed_fr2);
        REQUIRE(tag1 == tag2);
    }
}

#include "grpc_test_helpers.hpp"

TEST_CASE("GRPCEventQueue", "[grpc][Scheduler]")
{
    auto pop = std::make_shared<Population>();
    auto eq = std::make_shared<GRPCEventQueue>();

    eq->setPopulation(pop);
    eq->registerData();
    eq->afterRegisterData();
    Scheduler wd(eq);

    // This should have registered the handler for this.
    // If it hasn't: we have a problem.
    REQUIRE(pop->isGlobalRegistered<SharedgRPCCompletionQueue>());
    auto cqw = pop->getGlobalData<SharedgRPCCompletionQueue>();

    // Start a test server.
    int test_port = 0;
    TestEvaluatorService test_service("test_service");
    auto test_server = test_service.start_local_locked_test_server("127.0.0.1:0", &test_port);

    auto ch = grpc::CreateChannel("127.0.0.1:" + std::to_string(test_port), grpc::InsecureChannelCredentials());
    Evaluator::Stub ev(ch);

    grpc::ClientContext cc;
    EvaluationEvaluateSolutionsRequest req;
    EvaluationEvaluateSolutionsResponse res;
    grpc::Status status;

    bool received_query = false;
    auto fr = std::make_unique<FunctionalResumable>([&received_query](Scheduler & /* wd */) {
        received_query = true;
        return false;
    });
    size_t tag = wd.get_tag(std::move(fr));

    // Note: impl needs a count of # of pending evaluations which is not maintained
    // automatically.
    cqw->pending++;
    auto aq = ev.AsyncEvaluateSolutions(&cc, req, cqw->cq.get());
    aq->Finish(&res, &status, (void *)tag);

    wd.step();

    REQUIRE(received_query);

    // Cleanup
    test_server->Shutdown();
}

TEST_CASE("WaitForOtherEventsResumable", "[Scheduler]")
{
    auto mteq = std::make_shared<MockTaggedEventQueue>();
    Scheduler wd(mteq);

    int complete_count = 0;
    auto completer_fn = [&complete_count](Scheduler &) {
        complete_count++;
        return false;
    };
    auto completer = std::make_unique<FunctionalResumable>(completer_fn);
    size_t completer_tag = wd.get_tag(std::move(completer));
    // wd.tag_stack.push_back(completer_tag);

    auto counting_resumable = std::make_unique<WaitForOtherEventsResumable>(completer_tag);
    counting_resumable->wait_for_one();
    counting_resumable->wait_for_one();
    counting_resumable->wait_for_one();
    size_t counting_tag = wd.get_tag(std::move(counting_resumable));

    // First two.
    wd.complete_tag(counting_tag);
    REQUIRE(complete_count == 0);
    wd.complete_tag(counting_tag);
    REQUIRE(complete_count == 0);
    // This is the final one we waited for!
    wd.complete_tag(counting_tag);
    REQUIRE(complete_count == 1);
}

TEST_CASE("GenerationalStarter", "[Scheduler]")
{
    auto mteq = std::make_shared<MockTaggedEventQueue>();
    Scheduler wd(mteq);

    int complete_count = 0;
    size_t current_tag;

    auto cfn = [&complete_count, &current_tag](Scheduler &wd) {
        complete_count++;
        current_tag = wd.tag_stack.back();
        wd.tag_stack.pop_back();
    };

    auto np = std::unique_ptr<IResumable>();
    auto gs = (std::unique_ptr<IResumable>) std::make_unique<GenerationalStarter>(cfn);
    current_tag = wd.get_tag(std::move(gs));

    wd.complete_tag(current_tag);
    wd.complete_tag(current_tag);
    wd.complete_tag(current_tag);
    wd.complete_tag(current_tag);

    REQUIRE(complete_count == 4);
}

TEST_CASE("RemoteAsyncObjectiveFunction")
{
    auto pop = std::make_shared<Population>();
    auto eq = std::make_shared<GRPCEventQueue>();
    // Note - RemoteAsyncObjectiveFunction does not register.
    auto wd = std::make_shared<Scheduler>(eq);
    wd->setPopulation(pop);
    wd->registerData();
    wd->afterRegisterData();
    TestEvaluatorService test_service("test_service");
    std::string hostname = "127.0.0.1:";
    int port = 0;
    auto server = test_service.start_local_locked_test_server(hostname + "0", &port);
    std::string evaluation_host = hostname + std::to_string(port);
    RemoteAsyncObjectiveFunction raof(wd, evaluation_host);
    raof.setPopulation(pop);
    raof.registerData();
    raof.afterRegisterData();

    bool done = false;
    auto tag = wd->get_tag(std::make_unique<FunctionalResumable>([&done](Scheduler &) { done = true; return false; }));
    wd->tag_stack.push_back(tag);

    Individual i = pop->newIndividual();
    raof.evaluate(i);

    // Should wait on response.
    wd->step();

    REQUIRE(done);

    server->Shutdown();
}

class ConstantOne : public ObjectiveFunction
{
  public:
    void evaluate(Individual a) override
    {
        auto &o = population->getData<Objective>(a);
        if (o.objectives.size() == 0)
        {
            o.objectives.resize(1);
        }
        o.objectives[0] = 1;    
    }

    void registerData() override
    {
        population->registerData<Objective>();
    }
    
    ObjectiveFunction* clone() override
    {
        return new ConstantOne();
    }
};

TEST_CASE("Integration Test: Remote Evaluations")
{
    auto pop = std::make_shared<Population>();
    auto eq = std::make_shared<GRPCEventQueue>();
    auto wd = std::make_shared<Scheduler>(eq);
    // Note - this test does not contain anything that registers the scheduler.
    wd->setPopulation(pop);
    wd->registerData();
    wd->afterRegisterData();

    auto problem = std::make_shared<ConstantOne>();
    std::shared_ptr<ObjectiveFunction> blueprint(problem->clone());
    // Note: use the problem definition to register the prerequisite data structures.
    // Note - RemoteProblemEvaluatorService already does this
    // problem->setPopulation(pop);
    // problem->registerData();
    // problem->afterRegisterData();

    RemoteProblemEvaluatorService test_service(problem);
    std::string hostname = "127.0.0.1:";
    int port = 0;
    auto server = test_service.start_server(hostname + "0", &port);
    std::string evaluation_host = hostname + std::to_string(port);
    RemoteAsyncObjectiveFunction raof(wd, evaluation_host, blueprint);
    raof.setPopulation(pop);
    raof.registerData();
    raof.afterRegisterData();

    bool done = false;
    auto tag = wd->get_tag(std::make_unique<FunctionalResumable>([&done](Scheduler &) { done = true; return false; }));
    wd->tag_stack.push_back(tag);

    Individual i = pop->newIndividual();
    raof.evaluate(i);

    // Should wait on response.
    wd->step();

    REQUIRE(done);

    Objective &o = pop->getData<Objective>(i);
    REQUIRE(o.objectives.size() == 1);
    REQUIRE(o.objectives[0] == 1);

    server->Shutdown();
}