//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "decentralized.hpp"
#include "ealib.grpc.pb.h"
#include <exception>
#include <memory>

bool FunctionalResumable::resume(Scheduler &wd, std::unique_ptr<IResumable> &)
{
    return func(wd);
}

// Scheduler
Scheduler::Scheduler(std::shared_ptr<ITaggedEventQueue> teq) : teq(teq)
{
}
bool Scheduler::stepQueue()
{
    if (event_queue.empty())
        return false;

    auto &eq_top = event_queue.top();
    eq_top.ptr->resume(*this, eq_top.ptr);
    event_queue.pop();

    return true;
}
void Scheduler::schedule_immediately(std::unique_ptr<IResumable> resumable, double priority)
{
    event_queue.push(Event{priority, std::move(resumable)});
}
void Scheduler::schedule_after_message_completion(std::unique_ptr<IResumable> resumable, double priority)
{
    after_message_completion.push_back(Event{priority, std::move(resumable)});
}
size_t Scheduler::get_tag(std::unique_ptr<IResumable> resumable, bool message_completion)
{
    size_t tag;
    if (tags.empty())
    {
        tag = waiting_tag.size();
        waiting_tag.push_back(std::move(resumable));
        waiting_tag_flags.push_back(0);
    }
    else
    {
        tag = tags.back();
        tags.pop_back();
        waiting_tag[tag] = std::move(resumable);
    }

    num_waiting_tags++;
    waiting_tag_flags[tag] = message_completion ? 1 : 0;
    return tag;
}
void Scheduler::complete_tag(size_t tag, std::exception_ptr ex)
{
    bool pending = true;
    auto &resumable = waiting_tag[tag];

    current_ex = ex;
    if (resumable != nullptr)
    {
        pending = resumable->resume(*this, resumable);
    }
    if (!pending)
    {
        num_waiting_tags--;
        tags.push_back(tag);

        waiting_tag[tag] = nullptr;
    }
    // Tag was flagged as a message completion event.
    // Message completion events are marked as events that are guaranteed to take time -
    // Deferred scheduling operations that may immediately schedule themselves again
    // may wait for this prior to enter the event queue to avoid getting stuck in an
    // infinite loop.
    if ((waiting_tag_flags[tag] & 0b1) == 1)
    {
        // Add events for after_message_completion into the queue.
        for (auto &a : after_message_completion)
        {
            event_queue.push(std::move(a));
        }
        after_message_completion.clear();
    }
}
std::exception_ptr Scheduler::get_exception()
{
    return this->current_ex;
}
bool Scheduler::terminated()
{
    return event_queue.empty() && !not_last_step;
}
void Scheduler::terminate_waiting_tags()
{
    num_waiting_tags = 0;
}
void Scheduler::stepUntilEmptyQueue()
{
    while (stepQueue())
    {
    }
}
bool Scheduler::step()
{
    if (stepQueue())
        return true;
    size_t tag;
    if (!teq->next(*this, &tag))
    {
        not_last_step = false;
        return false;
    }
    complete_tag(tag);
    return true;
}
void Scheduler::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    this->teq->setPopulation(population);
}
void Scheduler::registerData()
{
    this->teq->registerData();
}
void Scheduler::afterRegisterData()
{
    this->teq->afterRegisterData();
}

// GenerationalStarter
GenerationalStarter::GenerationalStarter(std::function<void(Scheduler &)> func) : func(func)
{
}
bool GenerationalStarter::resume(Scheduler &wd, std::unique_ptr<IResumable> &self)
{
    size_t new_tag = wd.get_tag(std::move(self));
    wd.tag_stack.push_back(new_tag);

    func(wd);

    return false;
}

// WaitForOtherEventsResumable
WaitForOtherEventsResumable::WaitForOtherEventsResumable(size_t tag_next) : tag_next(tag_next)
{
}
void WaitForOtherEventsResumable::wait_for_one()
{
    number_left++;
}
bool WaitForOtherEventsResumable::resume(Scheduler &wd, std::unique_ptr<IResumable> & /* resumable */)
{
    number_left--;
    if (number_left <= 0)
    {
        wd.complete_tag(tag_next);
        return false;
    }

    return true;
}

// RemoteAsyncObjectiveFunction
bool RemoteAsyncObjectiveFunction::EvaluationResumable::resume(Scheduler &wd,
                                                               std::unique_ptr<IResumable> & /* resumable */)
{
    // We have been resumed, this means the evaluation has completed.
    q->pending--;
    // First. decode the response!
    auto &data = res.solutions().cerealencoded();
    std::stringstream ss(data);
    cereal::BinaryInputArchive boa(ss);
    // Again, obtain a vector of individuals & the corresponding subpopulation.
    std::vector<Individual> iis = {i};

    t_assert(status.ok(), "Request should be successfully processed.");
    t_assert(i.i == static_cast<size_t>(res.key()), "Key and solution should match");
    // Load data from archive.
    auto spd = SubpopulationData();
    boa(spd);
    // Apply spd to population
    spd.inject(*population, iis);

    // Now the solution is evaluated!
    wd.complete_tag(*completion_tag);

    // Since we have completed: we are no longer pending!
    return false;
}
RemoteAsyncObjectiveFunction::RemoteAsyncObjectiveFunction(std::shared_ptr<Scheduler> s,
                                                           std::string host,
                                                           std::shared_ptr<ObjectiveFunction> blueprint) :
    blueprint(blueprint),
    host(host),
    s(s),
    channel(grpc::CreateChannel(host, grpc::InsecureChannelCredentials())),
    evaluator_stub_(Evaluator::NewStub(channel))
{
}
void RemoteAsyncObjectiveFunction::registerData()
{
    if (blueprint != nullptr)
        blueprint->registerData();

    // Note - scheduler should be registered by something else - i.e. the actual algorithm
    // that uses it.
    // this->s->registerData();
}
void RemoteAsyncObjectiveFunction::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    // Note - scheduler should be registered by something else - i.e. the actual algorithm
    // that uses it.
    // this->s->setPopulation(population);

    if (blueprint != nullptr)
        blueprint->setPopulation(population);
}
void RemoteAsyncObjectiveFunction::afterRegisterData()
{
    // Note - scheduler should be registered by something else - i.e. the actual algorithm
    // that uses it.

    t_assert(population->isGlobalRegistered<SharedgRPCCompletionQueue>(),
             "SharedgRPCCompletionQueue should be registered.");
    scq = population->getGlobalData<SharedgRPCCompletionQueue>();

    if (blueprint != nullptr)
        blueprint->afterRegisterData();
}
void RemoteAsyncObjectiveFunction::evaluate(Individual i)
{
    t_assert(!s->tag_stack.empty(), "This method performs asynchronous evaluation.\
          Without a callback the approach will not know when an evaluation has finished.\
          The fact that the Scheduler has an empty tagstack indicates that no callbacks are expected.\
          In all likelihood, a mistake has been made. If it has not: add a tag onto the tagstack.\
          A nullpointer callback does nothing, but does stop this assertion from happening.");

    EvaluationResumable *er(new EvaluationResumable());
    // Set completion tag
    er->completion_tag = s->tag_stack.back();
    s->tag_stack.pop_back();
    // Set processing queue (such that #pending can be reduced)
    er->q = scq.get();
    // Set individual to update
    er->i = i;
    er->population = population.get();
    // Set up evaluation request
    Solutions *solutions = new Solutions();
    solutions->set_num(1);
    // Grab Solution data & encode it and add it to the request.
    std::ostringstream oss;
    std::vector<Individual> iis = {i};
    auto spd = population->getSubpopulationData(iis);
    cereal::BinaryOutputArchive boa(oss);
    spd.serialize(boa);
    solutions->set_cerealencoded(oss.str());
    er->req.set_allocated_solutions(solutions);
    er->req.set_key(static_cast<int64_t>(i.i));
    // Register & fetch a tag.
    size_t tag = s->get_tag(std::unique_ptr<IResumable>(er), true);
    // Send the request off!
    scq->pending++;
    auto x = evaluator_stub_->AsyncEvaluateSolutions(&er->ctx, er->req, scq->cq.get());
    x->Finish(&er->res, &er->status, (void *)tag);
}
ObjectiveFunction *RemoteAsyncObjectiveFunction::clone()
{
    return new RemoteAsyncObjectiveFunction(s, host);
}

// GRPCEventQueue
bool GRPCEventQueue::next(Scheduler &wd, size_t *tag)
{
    if (scq->pending == 0)
    {
        return false;
    }
    void *vtag;
    bool ok;
    bool got_event = scq->cq->Next(&vtag, &ok);
    // gRPC has shut down, no more events.
    if (!got_event)
    {
        // Terminate waiting tags: anything to do with gRPC will never be completed anymore.
        wd.terminate_waiting_tags();
        return false;
    }
    *tag = (size_t)vtag;
    return true;
}
void GRPCEventQueue::registerData()
{
    population->registerGlobalData(SharedgRPCCompletionQueue{std::make_shared<grpc::CompletionQueue>()});
    scq = population->getGlobalData<SharedgRPCCompletionQueue>();
}

// RemoteProblemEvaluatorService
RemoteProblemEvaluatorService::RemoteProblemEvaluatorService(std::shared_ptr<ObjectiveFunction> problem) :
    problem(problem)
{
    pop = std::make_shared<Population>();
    problem->setPopulation(pop);
    problem->registerData();
    problem->afterRegisterData();
    i = pop->newIndividual();
}
grpc::Status RemoteProblemEvaluatorService::EvaluateSolutions(grpc::ServerContext * /* context */,
                                                              const EvaluationEvaluateSolutionsRequest *req,
                                                              EvaluationEvaluateSolutionsResponse *res)
{
    res->set_key(req->key());

    // Decode
    auto &indata = req->solutions().cerealencoded();
    std::stringstream ss(indata);
    cereal::BinaryInputArchive bia(ss);
    SubpopulationData spd;
    bia(spd);
    std::vector<Individual> i_v = {i};

    // - From this point onwards we interact with shared memory!
    mtx.lock();
    spd.inject(*pop, i_v);

    // Evaluate
    problem->evaluate(i);

    // Re-encode
    SubpopulationData spdo = pop->getSubpopulationData(i_v);
    std::ostringstream oss;
    cereal::BinaryOutputArchive boa(oss);
    spdo.serialize(boa);
    // After serialization, we can re-use the associated memory.
    mtx.unlock();

    std::string oss_str = oss.str();

    t_assert(!oss_str.empty(), "Result should not be empty.");

    auto solutions = res->mutable_solutions();
    solutions->set_num(1);
    solutions->set_cerealencoded(oss_str);

    return grpc::Status::OK;
}
std::unique_ptr<grpc::Server> RemoteProblemEvaluatorService::start_server(const std::string &uri, int *selected_port)
{
    auto server_promise = std::promise<grpc::Server*>();
    ccss = std::make_unique<ConnectionCountShutdownService>(server_promise.get_future());

    grpc::ServerBuilder server_builder;
    server_builder.AddListeningPort(uri, grpc::InsecureServerCredentials(), selected_port);
    server_builder.RegisterService(this);

    auto server = server_builder.BuildAndStart();
    server_promise.set_value(server.get());

    my_uri = uri.substr(0, uri.find_last_of(':') + 1) + std::to_string(*selected_port);
    while (!uris_to_register_with.empty())
    {
        send_register_request(uris_to_register_with.back());
        uris_to_register_with.pop_back();
    }

    return server;
}
void RemoteProblemEvaluatorService::send_register_request(const std::string &uri)
{
    t_assert(my_uri.has_value(), "Server should be started before registering.");
    // Open a connection.
    auto channel = grpc::CreateChannel(uri, grpc::InsecureChannelCredentials());
    EvaluationDistributor::Stub dispatcher_client_ed(channel);
    Evaluator::Stub dispatcher_client_e(channel);
    // Send a request for registration.
    grpc::ClientContext context;
    EvaluationDistributorRegisterEvaluatorRequest req;
    EvaluationDistributorRegisterEvaluatorResponse res;
    Host *host = new Host;
    host->set_host(my_uri.value());
    req.set_allocated_host(host);
    auto s = dispatcher_client_ed.RegisterEvaluator(&context, req, &res);
    // Check.
    t_assert(s.ok(), "Sending request should be successful");
}
void RemoteProblemEvaluatorService::register_with(const std::string &uri)
{
    if (my_uri.has_value())
    {
        send_register_request(uri);
    }
    else
    {
        uris_to_register_with.push_back(uri);
    }
}
