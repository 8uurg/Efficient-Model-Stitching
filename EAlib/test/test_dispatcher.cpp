#include "dispatcher.hpp"
#include <catch2/catch.hpp>
#include <chrono>
#include <memory>
#include <thread>

// RPC
#include "rpc.h"
#include "grpc_test_helpers.hpp"


TEST_CASE("Dispatcher", "[Dispatcher]")
{
    // Start some test servers.
    // Locks start locked: we need to release the function evaluation to see if they are spread out!
    int port_a = 0;
    TestEvaluatorService service_a("service_a");
    auto server_a = service_a.start_local_locked_test_server("127.0.0.1:0", &port_a);

    int port_b = 0;
    TestEvaluatorService service_b("service_b");
    auto server_b = service_b.start_local_locked_test_server("127.0.0.1:0", &port_b);

    int ds_port;
    DispatcherServer ds("127.0.0.1:0", &ds_port);
    // Set up server
    ds.start_server();
    // Start handling requests
    std::thread th([&ds]() { ds.start_handling_requests(); });

    // Connect to the server.
    auto channel = grpc::CreateChannel("127.0.0.1:" + std::to_string(ds_port), grpc::InsecureChannelCredentials());
    EvaluationDistributor::Stub dispatcher_client_ed(channel);
    Evaluator::Stub dispatcher_client_e(channel);

    auto connect_port = [&dispatcher_client_ed](int port){
        grpc::ClientContext context;
        EvaluationDistributorRegisterEvaluatorRequest req;
        EvaluationDistributorRegisterEvaluatorResponse res;
        Host *host = new Host;
        host->set_host("127.0.0.1:" + std::to_string(port));
        req.set_allocated_host(host);
        auto s = dispatcher_client_ed.RegisterEvaluator(&context, req, &res);
        REQUIRE(s.ok());
    };

    SECTION("spreads out workload")
    {
        using trompeloeil::_;
        // Ignore attempts with regards to using Connect & Disconnect for this test
        ALLOW_CALL(service_a.connectable, Connect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_b.connectable, Connect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_a.connectable, Disconnect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_b.connectable, Disconnect(_, _, _)).RETURN(grpc::Status::OK);

        // Use connection to register the test servers
        connect_port(port_a);
        connect_port(port_b);

        // At this point the server should be ready to process our queries!
        // (Note that the registration was done synchronously: we waited
        //  for completion, as such we are guaranteed to have these requests
        //  completed at this point. )

        // Send out two requests for evaluation. Note that synchronous
        // requests will block, so these requests are asynchronous instead.
        {
            grpc::CompletionQueue cq;

            grpc::ClientContext context1;
            context1.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
            EvaluationEvaluateSolutionsRequest req1;
            EvaluationEvaluateSolutionsResponse res1;
            grpc::Status s1;
            auto r1 = dispatcher_client_e.AsyncEvaluateSolutions(&context1, req1, &cq);
            r1->Finish(&res1, &s1, (void*) 0);

            grpc::ClientContext context2;
            context2.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
            EvaluationEvaluateSolutionsRequest req2;
            EvaluationEvaluateSolutionsResponse res2;
            grpc::Status s2;
            auto r2 = dispatcher_client_e.AsyncEvaluateSolutions(&context2, req2, &cq);
            r2->Finish(&res2, &s2, (void*) 1);

            grpc::ClientContext context3;
            context3.set_deadline(std::chrono::system_clock::now() + std::chrono::seconds(3));
            EvaluationEvaluateSolutionsRequest req3;
            EvaluationEvaluateSolutionsResponse res3;
            grpc::Status s3;
            auto r3 = dispatcher_client_e.AsyncEvaluateSolutions(&context3, req3, &cq);
            r3->Finish(&res3, &s3, (void*) 2);

            // ?: Maybe check if nothing finishes before unlocking?
            // See if we see our tags back.
            std::vector<char> seen(3);
            std::fill(seen.begin(), seen.end(), 0);

            size_t num_seen = 0;
            bool is_ok;
            void *tag;
            while (num_seen < 3)
            {
                bool is_valid = cq.Next(&tag, &is_ok);
                if (!is_valid || !is_ok)
                    break;
                seen[(size_t)tag] = 1;
                num_seen++;
            }

            for (size_t idx = 0; idx < 3; ++idx)
            {
                REQUIRE(seen[idx] == 1);
            }
        }

        // Cleanup.
        ds.shutdown();
        th.join();
        server_a->Shutdown();
        server_b->Shutdown();

        // Assert that the calls were distributed somewhat.
        REQUIRE(service_a.getNumCalls() + service_b.getNumCalls() == 3);
        REQUIRE(std::max(service_a.getNumCalls(), service_b.getNumCalls()) == 2);
        REQUIRE(std::min(service_a.getNumCalls(), service_b.getNumCalls()) == 1);

        std::cout << "Processor A processed " << service_a.getNumCalls() << " calls" << std::endl;
        std::cout << "Processor B processed " << service_b.getNumCalls() << " calls" << std::endl;
    }

    SECTION("enqueues until servers become available")
    {
        using trompeloeil::_;
        // Ignore attempts with regards to using Connect & Disconnect for this test
        ALLOW_CALL(service_a.connectable, Connect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_b.connectable, Connect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_a.connectable, Disconnect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_b.connectable, Disconnect(_, _, _)).RETURN(grpc::Status::OK);

        {
            grpc::CompletionQueue cq;

            grpc::ClientContext context1;
            EvaluationEvaluateSolutionsRequest req1;
            EvaluationEvaluateSolutionsResponse res1;
            grpc::Status s1;
            auto r1 = dispatcher_client_e.AsyncEvaluateSolutions(&context1, req1, &cq);
            r1->Finish(&res1, &s1, (void*) 0);

            grpc::ClientContext context2;
            EvaluationEvaluateSolutionsRequest req2;
            EvaluationEvaluateSolutionsResponse res2;
            grpc::Status s2;
            auto r2 = dispatcher_client_e.AsyncEvaluateSolutions(&context2, req2, &cq);
            r2->Finish(&res2, &s2, (void*) 1);

            // Late registration of processors.
            connect_port(port_a);
            connect_port(port_b);

            // See if we see our tags back when we register late.
            std::vector<char> seen(2);
            std::fill(seen.begin(), seen.end(), 0);
            size_t num_seen = 0;
            bool is_ok;
            void *tag;
            while (num_seen < 2)
            {
                bool is_valid = cq.Next(&tag, &is_ok);

                if (!is_valid || !is_ok)
                    break;
                seen[(size_t)tag] = 1;
                num_seen++;
            }

            for (size_t idx = 0; idx < 2; ++idx)
            {
                REQUIRE(seen[idx] == 1);
            }
            // No hitting deadlines here.
            REQUIRE(s1.ok());
            REQUIRE(s2.ok());
        }
        
        ds.shutdown();
        th.join();
        server_a->Shutdown();
        server_b->Shutdown();
    }

    SECTION("forwards message content")
    {
        using trompeloeil::_;
        // Ignore attempts with regards to using Connect & Disconnect for this test
        ALLOW_CALL(service_a.connectable, Connect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_b.connectable, Connect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_a.connectable, Disconnect(_, _, _)).RETURN(grpc::Status::OK);
        ALLOW_CALL(service_b.connectable, Disconnect(_, _, _)).RETURN(grpc::Status::OK);

        // Use connection to register the test servers
        // For this test it is important to know that the test server
        // echo's the original request content back.
        connect_port(port_a);
        connect_port(port_b);

        grpc::ClientContext context;
        EvaluationEvaluateSolutionsRequest req;
        req.set_key(15);
        auto ms = req.mutable_solutions();
        ms->set_cerealencoded("this is a test");
        ms->set_num(5);
        
        EvaluationEvaluateSolutionsResponse res;
        grpc::Status s1;
        dispatcher_client_e.EvaluateSolutions(&context, req, &res);

        REQUIRE(res.key() == 15);
        REQUIRE(res.solutions().num() == 5);
        REQUIRE(res.solutions().cerealencoded() == "this is a test");
        
        ds.shutdown();
        th.join();
        server_a->Shutdown();
        server_b->Shutdown();
    }

    SECTION("informs client of connection & disconnection")
    {
        using trompeloeil::_;
        {
            REQUIRE_CALL(service_a.connectable, Connect(_, _, _)).RETURN(grpc::Status::OK);
            connect_port(port_a);
        }
        {
            REQUIRE_CALL(service_b.connectable, Connect(_, _, _)).RETURN(grpc::Status::OK);
            connect_port(port_b);
        }

        {
            REQUIRE_CALL(service_a.connectable, Disconnect(_, _, _)).RETURN(grpc::Status::OK);
            REQUIRE_CALL(service_b.connectable, Disconnect(_, _, _)).RETURN(grpc::Status::OK);
            ds.shutdown();
            th.join();
        }

        // Cleanup.
        server_a->Shutdown();
        server_b->Shutdown();
    }
}