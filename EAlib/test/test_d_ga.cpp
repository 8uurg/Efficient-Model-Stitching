#include <catch2/catch.hpp>
#include <memory>

#include "acceptation_criteria.hpp"
#include "archive.hpp"
#include "base.hpp"
#include "d-ga.hpp"
#include "decentralized.hpp"
#include "ga.hpp"
#include "initializers.hpp"
#include "problems.hpp"
#include "trompeloeil.hpp"


TEST_CASE("Integration Test: DistributedSynchronousSimpleGA on OneMax", "[Decentralized][Integration]")
{
    // This test relies on convergence, and that it truly stops in such a case.
    // Sadly, setting time limits is not particularly possible. So if this takes more than 10s, something
    // is probably wrong.
    // Additional aspects under test include hitting a limit (!)

    auto pop = std::make_shared<Population>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    const size_t population_size =  GENERATE(16, 128, 512);
    size_t budget = GENERATE(10000, 100);
    auto mteq = std::make_shared<GRPCEventQueue>();
    auto wd = std::make_shared<Scheduler>(mteq);
    // Is called properly by remote evaluation class.
    // wd->setPopulation(pop);
    // wd->registerData();
    // wd->afterRegisterData();

    size_t l = 100;
    auto e_problem = std::make_shared<OneMax>(l);
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
    auto performance_criterion = std::make_shared<SingleObjectiveAcceptanceCriterion>();
    auto crossover = std::make_shared<UniformCrossover>();
    auto mutation = std::make_shared<PerVariableInAlphabetMutation>(1.0 / static_cast<double>(l));
    auto parent_selection = std::make_shared<ShuffledSequentialSelection>();
    std::vector<size_t> indices = { 0 };
    auto archive = std::make_shared<BruteforceArchive>(indices);

    DistributedSynchronousSimpleGA dsg(
        wd, population_size, population_size, 7,
        initializer,
        crossover,
        mutation,
        parent_selection,
        performance_criterion,
        archive);
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

        while(!dsg.terminated() && ! cause_is_initial_exception)
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

TEST_CASE("Integration Test: DistributedAsynchronousSimpleGA on OneMax", "[Decentralized][Integration]")
{
    // This test relies on convergence, and that it truly stops in such a case.
    // Sadly, setting time limits is not particularly possible. So if this takes more than 10s, something
    // is probably wrong.
    // Additional aspects under test include hitting a limit (!)

    auto pop = std::make_shared<Population>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    const size_t population_size =  GENERATE(16, 128, 512);
    size_t budget = GENERATE(100, 10000);
    auto mteq = std::make_shared<GRPCEventQueue>();
    auto wd = std::make_shared<Scheduler>(mteq);
    // Is called properly by remote evaluation class.
    // wd->setPopulation(pop);
    // wd->registerData();
    // wd->afterRegisterData();

    size_t l = 100;
    auto e_problem = std::make_shared<OneMax>(l);
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
    auto performance_criterion = std::make_shared<SingleObjectiveAcceptanceCriterion>();
    auto crossover = std::make_shared<UniformCrossover>();
    auto mutation = std::make_shared<PerVariableInAlphabetMutation>(1.0 / static_cast<double>(l));
    auto parent_selection = std::make_shared<ShuffledSequentialSelection>();
    std::vector<size_t> indices = { 0 };
    auto archive = std::make_shared<BruteforceArchive>(indices);

    DistributedAsynchronousSimpleGA dsg(
        wd, population_size, population_size, 7,
        initializer,
        crossover,
        mutation,
        parent_selection,
        performance_criterion,
        archive);
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


TEST_CASE("Integration Test: DistributedRandomSearch on OneMax", "[Decentralized][Integration]")
{
    // This test relies on convergence, and that it truly stops in such a case.
    // Sadly, setting time limits is not particularly possible. So if this takes more than 10s, something
    // is probably wrong.
    // Additional aspects under test include hitting a limit (!)

    auto pop = std::make_shared<Population>();
    Rng rng(42);
    pop->registerGlobalData(rng);

    const size_t batch_size = GENERATE(16, 256);
    size_t budget = GENERATE(100, 10000);
    auto mteq = std::make_shared<GRPCEventQueue>();
    auto wd = std::make_shared<Scheduler>(mteq);
    // Is called properly by remote evaluation class.
    // wd->setPopulation(pop);
    // wd->registerData();
    // wd->afterRegisterData();

    size_t l = 100;
    auto e_problem = std::make_shared<OneMax>(l);
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

    auto initializer = std::make_shared<CategoricalProbabilisticallyCompleteInitializer>();    std::vector<size_t> indices = { 0 };
    auto archive = std::make_shared<BruteforceArchive>(indices);

    DistributedRandomSearch dsg(
        wd,
        initializer,
        archive,
        batch_size);
    dsg.setPopulation(pop);
    dsg.registerData();
    l_problem->afterRegisterData();
    dsg.afterRegisterData();

    auto s = pop->getGlobalData<SharedgRPCCompletionQueue>();

    DYNAMIC_SECTION("Population Size: " << batch_size << " & Budget: " << budget)
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
        
        if (budget < batch_size) {
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
        if (budget == 100 && batch_size > 100)
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