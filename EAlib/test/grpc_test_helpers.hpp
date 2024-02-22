#pragma once

#include "trompeloeil.hpp"
#include <thread>

// RPC
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#include <grpc/grpc.h>
// - Client
#include <grpcpp/client_context.h>
// - Server
#include <grpcpp/security/server_credentials.h>
#include <grpcpp/server.h>
#include <grpcpp/server_builder.h>
#include <grpcpp/server_context.h>
// - Generated
#include "ealib.grpc.pb.h"
#include "ealib.pb.h"
#pragma GCC diagnostic pop

// Helper service for testing
class TestEvaluatorService final : public Evaluator::Service
{
    std::string nickname;

    size_t num_calls = 0;

  public:
    class TestConnectableService final : public trompeloeil::mock_interface<Connectable::Service>
    {
        IMPLEMENT_MOCK3(Connect);
        // grpc::Status Connect(grpc::ServerContext *ctx, const ConnectRequest *req, ConnectResponse *res);

        IMPLEMENT_MOCK3(Disconnect);
        // grpc::Status Disconnect(grpc::ServerContext *ctx, const ConnectRequest *req, ConnectResponse *res); 
    };
    TestConnectableService connectable;

    TestEvaluatorService(std::string nickname) :
        nickname(nickname)
    {
    }
    size_t getNumCalls()
    {
        return num_calls;
    }

    grpc::Status EvaluateSolutions(grpc::ServerContext * /* context */,
                                   const EvaluationEvaluateSolutionsRequest *req,
                                   EvaluationEvaluateSolutionsResponse *res)
    {
        num_calls++;

        // Make this somewhat expensive!
        std::this_thread::sleep_for (std::chrono::milliseconds(500));

        res->set_key(req->key());
        auto *solutions = new Solutions();
        *solutions = req->solutions();
        res->set_allocated_solutions(solutions);

        return grpc::Status::OK;
    }

    std::unique_ptr<grpc::Server> start_local_locked_test_server(const std::string &uri,
                                                             int *selected_port)
    {
        grpc::ServerBuilder server_builder;
        server_builder.AddListeningPort(uri, grpc::InsecureServerCredentials(), selected_port);
        server_builder.RegisterService(this);
        server_builder.RegisterService(&connectable);
        return server_builder.BuildAndStart();
    }

};

