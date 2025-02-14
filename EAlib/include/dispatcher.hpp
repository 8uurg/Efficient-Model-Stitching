//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

// RPC
#include "ealib.grpc.pb.h"
#include "rpc.h"
#include <grpcpp/client_context.h>

// TODO: Do away with InsecureChannelCredentials

class EvaluatorClient
{
    // Note: we only send one request to each client at a time, so we can store
    // the prerequisite memory here. (If this invariant changes, move it to the
    // Handler instead.)
    grpc::ClientContext ctx_;
    EvaluationEvaluateSolutionsRequest req;
    EvaluationEvaluateSolutionsResponse res;

    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<Evaluator::Stub> stub_;
    std::unique_ptr<Connectable::Stub> connectable_stub_;

  public:
    EvaluatorClient(std::shared_ptr<grpc::Channel> channel) :
        channel(channel), stub_(Evaluator::NewStub(channel)), connectable_stub_(Connectable::NewStub(channel))
    {
    }

    Evaluator::Stub *getEvaluatorStub()
    {
        return stub_.get();
    }

    bool supports_connectable = false;
    Connectable::Stub *getConnectableStub()
    {
        return connectable_stub_.get();
    }
};

class DispatcherServer final
{
    // Asynchronous redirector.
    // Asynchronous impl is roughly based on what is described in
    // https://github.com/grpc/grpc/blob/v1.46.3/examples/cpp/helloworld/greeter_async_server.cc

    std::string server_address;
    int *selected_port;

    // Handlers

    size_t getAvailableHandlerIdx()
    {
        size_t handler_idx;
        if (handler_reuse_indices.empty())
        {
            handler_idx = handlers.size();
            handlers.push_back(nullptr);
        }
        else
        {
            handler_idx = handler_reuse_indices.back();
            handler_reuse_indices.pop_back();
        }
        return handler_idx;
    }

    void waitingCheck()
    {
        while (!waiting.empty() && !available.empty())
        {
            waiting.back()->progress();
            waiting.pop_back();
        }
    }

    class Handler
    {
      public:
        virtual ~Handler() = default;
        virtual void progress() = 0;
    };

    // Handles calls & responses to registerEvaluator.
    class RegisterEvaluatorHandler : public Handler
    {
        enum STATE
        {
            CREATED,
            HANDLING,
            SENT_CONNECTION_NOTIF,
            FINISHED
        };
        STATE state = CREATED;

        EvaluationDistributor::AsyncService *service_;
        grpc::ServerContext ctx_;
        grpc::ServerCompletionQueue *cq_;
        DispatcherServer *server_;

        grpc::ClientContext c_ctx_;
        ConnectRequest crq;
        ConnectResponse crs;
        grpc::Status c_status;

        size_t client_idx;

        EvaluationDistributorRegisterEvaluatorRequest req;
        EvaluationDistributorRegisterEvaluatorResponse res;
        grpc::ServerAsyncResponseWriter<EvaluationDistributorRegisterEvaluatorResponse> responder;

        size_t handler_idx = 0;

      public:
        RegisterEvaluatorHandler(EvaluationDistributor::AsyncService *service,
                                 grpc::ServerCompletionQueue *cq,
                                 DispatcherServer *server) :
            service_(service), cq_(cq), server_(server), responder(&ctx_)
        {
        }

        void progress() override
        {
            void *tag = (void *)handler_idx;
            switch (state)
            {
            case CREATED:
                handler_idx = server_->getAvailableHandlerIdx();
                server_->handlers[handler_idx] = this;
                // Update tag...
                tag = (void *)handler_idx;
                // The next time this function is called, we should be handling an incoming request.
                state = HANDLING;
                // Prepare this handler to handle a request.
                service_->RequestRegisterEvaluator(&ctx_, &req, &responder, cq_, cq_, tag);
                break;
            case HANDLING: {
                // Current 'RegisterEvaluatorHandler' is now handling a request.
                // Create a new one for any following requests.
                auto new_handler = (new RegisterEvaluatorHandler(service_, cq_, server_));
                // Let it handle any new requests.
                new_handler->progress();
            }
                {
                    // `req` should now be populated!
                    // Get the index for this evaluator.
                    client_idx = server_->clients.size();
                    // Create a new client.
                    auto channel = grpc::CreateChannel(req.host().host(), grpc::InsecureChannelCredentials());
                    auto new_client = std::unique_ptr<EvaluatorClient>(new EvaluatorClient(channel));
                    server_->clients.push_back(std::move(new_client));

                    // Next up: Send the connection notification to the client.
                    // Tell the evaluator we are connecting, to ensure that it can keep count,
                    // allowing it to, e.g., shut down when there are no more clients connected.
                    auto req = server_->clients[client_idx]->getConnectableStub()->AsyncConnect(&c_ctx_, crq, cq_);
                    req->Finish(&crs, &c_status, tag);

                    state = SENT_CONNECTION_NOTIF;
                }
                break;
            case SENT_CONNECTION_NOTIF: {
                // Prepare the response.
                res.set_idx(static_cast<int64_t>(client_idx));
                responder.Finish(res, grpc::Status::OK, tag);
                state = FINISHED;
            }
            break;
            case FINISHED:
                // We have sent the connection message, and the response is now received.
                // If the server implements the Connectable service, we should have an OK
                // response, otherwise this may be an error.
                if (c_status.ok())
                {
                    // Mark this evaluator as supporting connection registration.
                    server_->clients[client_idx]->supports_connectable = true;
                }

                // Finally: make the client available for evaluations to use.
                server_->available.push_back(client_idx);
                // Check if anyone is waiting and can be restarted.
                server_->waitingCheck();
                // Remove the marker
                server_->handlers[handler_idx] = nullptr;
                server_->handler_reuse_indices.push_back(handler_idx);
                // Clean up after ourselves, the newly created `RegisterEvaluatorHandler`
                // (or its own offspring!) should be handling further requests.
                delete this;
                break;
            }
        }
    };

    // Handles calls & responses to evaluateSolutions.
    class EvaluateSolutionsHandler : public Handler
    {
        enum STATE
        {
            CREATED,
            HANDLING,
            RETURNING,
            FINISHED
        };
        STATE state = CREATED;

        Evaluator::AsyncService *service_;
        grpc::ServerContext ctx_;
        grpc::ClientContext c_ctx_;
        grpc::ServerCompletionQueue *cq_;
        DispatcherServer *server_;
        size_t evaluator_idx = 0;

        size_t handler_idx = 0;

        EvaluationEvaluateSolutionsRequest req;
        EvaluationEvaluateSolutionsRequest req_copy;
        EvaluationEvaluateSolutionsResponse res;
        grpc::ServerAsyncResponseWriter<EvaluationEvaluateSolutionsResponse> responder;

        // Evaluator reply
        std::unique_ptr<grpc::ClientAsyncResponseReader<EvaluationEvaluateSolutionsResponse>> c_responder;
        grpc::Status status;

      public:
        EvaluateSolutionsHandler(Evaluator::AsyncService *service,
                                 grpc::ServerCompletionQueue *cq,
                                 DispatcherServer *server) :
            service_(service), cq_(cq), server_(server), responder(&ctx_)
        {
        }

        void progress() override
        {
            void *tag = (void *)handler_idx;
            switch (state)
            {
            case CREATED:
                handler_idx = server_->getAvailableHandlerIdx();
                server_->handlers[handler_idx] = this;
                // Update tag...
                tag = (void *)handler_idx;
                // The next time this function is called, we should be handling an incoming request.
                state = HANDLING;
                // Prepare this handler to handle a request.
                service_->RequestEvaluateSolutions(&ctx_, &req, &responder, cq_, cq_, tag);
                break;
            case HANDLING: {
                // Current 'EvaluateSolutionsHandler' is now handling a request.
                // Create a new one.
                auto new_handler = (new EvaluateSolutionsHandler(service_, cq_, server_));
                // Let it handle any new requests.
                new_handler->progress();
            }
                {
                    // At this point, our request object should be populated, we should forward it.
                    if (server_->available.empty())
                    {
                        // However: it turns out we have no available nodes for performing an evaluation,
                        // postpone until one becomes available.
                        server_->waiting.push_back(this);
                        break;
                    }
                    // an evaluator is available and can be uniquely assigned!
                    evaluator_idx = server_->available.back();
                    server_->available.pop_back();
                    // We are now going to inquire into getting this solution evaluated!
                    state = RETURNING;
                    auto &client = server_->clients[evaluator_idx];
                    // Forward the request onwards!
                    req_copy = req;
                    c_responder = client->getEvaluatorStub()->AsyncEvaluateSolutions(&c_ctx_, req_copy, cq_);
                    c_responder->Finish(&res, &status, tag);
                }
                break;
            case RETURNING: {
                // Request has finished - evaluator can be used again.
                server_->available.push_back(evaluator_idx);
                // Now res should be set, complete our own request.
                state = FINISHED;
                responder.Finish(res, status, tag);
                // Check if anyone is waiting and can be restarted.
                server_->waitingCheck();
            }
            break;
            case FINISHED:
                // Remove the marker
                server_->handlers[handler_idx] = nullptr;
                server_->handler_reuse_indices.push_back(handler_idx);
                // Clean up self.
                delete this;
                break;
            }
        }
    };

    // Currently pending requests.
    std::vector<Handler *> handlers;
    std::vector<size_t> handler_reuse_indices;

    // Data
    std::unique_ptr<grpc::ServerCompletionQueue> cq_;
    std::unique_ptr<grpc::Server> server_;

    EvaluationDistributor::AsyncService service_ed_;
    Evaluator::AsyncService service_e_;

  protected:
    std::vector<std::unique_ptr<EvaluatorClient>> clients;
    std::vector<size_t> available;
    std::vector<EvaluateSolutionsHandler *> waiting;

  public:
    DispatcherServer(std::string server_address, int *selected_port = nullptr) :
        server_address(server_address), selected_port(selected_port)
    {
    }

    void shutdown()
    {
        // Inform clients that we are shutting down and disconnecting
        for (auto& client : clients)
        {
            grpc::ClientContext c_ctx;
            DisconnectRequest drq;
            DisconnectResponse drs;
            // Note: done serially, without involvement of the async queue.
            auto status = client->getConnectableStub()->Disconnect(&c_ctx, drq, &drs);
            // Status should be OK, if we were able to send the initial connect message.
            if (client->supports_connectable && !status.ok())
            {
                std::cerr << "Could not inform evaluator of shutdown." << std::endl;
            }
        }
        // Shut down server
        server_->Shutdown();
        // Followed by the queue.
        cq_->Shutdown();
        // Empty the queue before cq is destroyed by ignoring everything.
        // Via: https://github.com/grpc/grpc/issues/23238
        void *ignored_tag;
        bool ignored_ok;
        while (cq_->Next(&ignored_tag, &ignored_ok))
        {
        }
    }

    ~DispatcherServer()
    {
        shutdown();
    }

    void start_server()
    {
        grpc::ServerBuilder builder;
        builder.AddListeningPort(server_address, grpc::InsecureServerCredentials(), selected_port);
        builder.RegisterService(&service_ed_);
        builder.RegisterService(&service_e_);
        cq_ = builder.AddCompletionQueue();
        server_ = builder.BuildAndStart();
        assert(server_ != nullptr);
        std::cout << "Server listening on " << server_address << std::endl;
    }

    void start_handling_requests()
    {
        handleRPCs();
    }

  private:
    void handleRPCs()
    {
        // Prepare initial handlers.
        (new RegisterEvaluatorHandler(&service_ed_, cq_.get(), this))->progress();
        (new EvaluateSolutionsHandler(&service_e_, cq_.get(), this))->progress();
        // Re-used memory
        bool ok = true;
        void *tag;
        grpc::Status status;

        while (true)
        {
            bool is_new_event = cq_->Next(&tag, &ok);
            if (!is_new_event || !ok)
            {
                // Queue is drained & shutting down: server is shutting down. Stop handling queries.
                // Or: something went wrong - maybe a client disconnected? -- This needs to be checked though!
                break;
            }
            auto *handler = handlers[(size_t)tag];
            handler->progress();
        }
    }
};