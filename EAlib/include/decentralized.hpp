#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <memory>
#include <mutex>
#include <future>
#include <optional>
#include <queue>
#include <sstream>
#include <tuple>

#include "base.hpp"
#include "ealib.grpc.pb.h"
#include "ealib.pb.h"
#include "logging.hpp"

// RPC
#include "rpc.h"

// Struct for tracking solution specific time spent on evaluating
struct MeasuredTimeSpent
{
    double t;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(CEREAL_NVP(t));
    }
};
template <> struct is_data_serializable<MeasuredTimeSpent> : std::true_type
{
};
CEREAL_REGISTER_TYPE(SubDataContainer<MeasuredTimeSpent>)

const std::shared_ptr<IDataType> MEASUREDTIMESPENT =
    std::make_shared<DataType<MeasuredTimeSpent>>(typeid(MeasuredTimeSpent).name());

class Scheduler;

/**
 * @brief Part of a state machine that can be resumed.
 *
 * A key aspect of these state machines is usually that there is a wait for something.
 * For example, a particular timestamp needs to be hit, or a particular piece of IO needs to return.
 */
class IResumable
{
  public:
    virtual ~IResumable() = default;

    // Resume the resumable
    //
    // Returns true if it is still pending
    // Returns false if the resumable is exhausted or has moved itself
    virtual bool resume(Scheduler &wd, std::unique_ptr<IResumable> &resumable) = 0;
};

class FunctionalResumable : public IResumable
{
    std::function<bool(Scheduler &)> func;

  public:
    FunctionalResumable(std::function<bool(Scheduler &)> func) : func(func)
    {
    }
    virtual bool resume(Scheduler &wd, std::unique_ptr<IResumable> & /* resumable */);
};

struct Event
{
    double priority;
    mutable std::unique_ptr<IResumable> ptr;

    // This one cannot be copied due to the unique_ptr.
    // Force the use of move operators by defining them explicitly.
    // Cost of figuring this one out: 2 hours and a bit.
    Event(Event &&) = default;
    Event &operator=(Event &&) = default;

    bool operator<(const Event &o) const
    {
        return priority < o.priority;
    };
};

class ITaggedEventQueue : public IDataUser
{
  public:
    virtual ~ITaggedEventQueue() = default;
    virtual bool next(Scheduler &wd, size_t *tag) = 0;
};

class Scheduler : public IDataUser
{
    std::shared_ptr<ITaggedEventQueue> teq;

    std::vector<size_t> tags;
    std::exception_ptr current_ex = nullptr;
    std::vector<std::unique_ptr<IResumable>> waiting_tag;
    std::vector<char> waiting_tag_flags;
    int num_waiting_tags = 0;
    // If false, the event queue was empty & the task queue was empty.
    // i.e. we are 'done'.
    bool not_last_step = true;

    std::priority_queue<Event> event_queue;

    // Waits for something that performs message completion.
    // Helps avoid busy-waiting, instead waiting on the message queue to properly process messages.
    std::vector<Event> after_message_completion;

    bool stepQueue();

  public:
    Scheduler(std::shared_ptr<ITaggedEventQueue> teq);

    // A stack to add tags onto, allows for flexibility with regards to events.
    // Generally this is used to say, anything that adds a scheduled task with a tag
    // after me, when you are done handling your tag. Complete me as well.
    // This provides a means of asynchronously doing things afterwards, or more generally
    // allow asynchronous combinations of things.
    std::vector<size_t> tag_stack;

    // Annoyingly enough, sometimes we can end up in a situation where our normal queue
    // of things to do is effectively busy-looping: i.e. recombination is taking place,
    // but because solutions are identical, no evaluations (and therefore queries) are
    // taking place. Such operators generally schedule another round into the queue.
    // This results in these tasks being continuously performed, and worse.
    // Causing currently pending queries to starve.
    // This integer counts how many evaluation queries (or other RPC calls!) have been performed
    // if this integer at the start and end of an operation is the same, this operator is potentially busy
    // looping, and schedule_after_message_completion should be used instead.
    size_t num_queue_messages_handled = 0;

    /**
     * @brief Schedule a task to be completed immediately.
     *
     * Be careful! A loop of such tasks could starve the tag completion queue,
     * and stop remote calls from being performed. Ensure no such loops exist.
     * Otherwise, consider breaking the loop by using `schedule_after_tag_completion`
     * instead.
     *
     * @param resumable The resumable to be scheduled
     * @param priority The priority with which this is to be scheduled.
     */
    void schedule_immediately(std::unique_ptr<IResumable> resumable, double priority);

    /**
     * @brief Queue a task to be scheduled after a tag is completed.
     *
     * @param resumable
     * @param priority
     */
    void schedule_after_message_completion(std::unique_ptr<IResumable> resumable, double priority);

    size_t get_tag(std::unique_ptr<IResumable> resumable, bool message_completion = false);

    void complete_tag(size_t tag, std::exception_ptr = nullptr);

    /**
     * @brief This function determines whether the scheduler will keep itself going by stepping.
     *
     * We are not done if there are events to schedule.
     * We can ignore `after_tag_completion` here:
     * - If there are events in the event queue, we haven't terminated yet.
     * - If there are waiting tags, then we haven't terminated yet.
     * - If there are no waiting tags, and no events in the event queue. then
     *   no new tags will be scheduled, and `after_tag_completion`'s resumables
     *   will not be resumed
     */
    bool terminated();

    // There might still be open queries at termination
    // leading to remaining waiting tags that will never be completed.
    // This function is called when this is determined to be the case.
    void terminate_waiting_tags();

    bool step();

    std::exception_ptr get_exception();

    void stepUntilEmptyQueue();

    void setPopulation(std::shared_ptr<Population> population);
    void registerData();
    void afterRegisterData();
};

/**
 * @brief Like FunctionalResumable, but pushes itself onto the tag stack before calling the function.
 *
 * As the name implies, it is generally useful for generational algorithms.
 */
class GenerationalStarter : public IResumable
{
    std::function<void(Scheduler &)> func;

  public:
    GenerationalStarter(std::function<void(Scheduler &)> func);

    bool resume(Scheduler &wd, std::unique_ptr<IResumable> &self) override;
};

class WaitForOtherEventsResumable : public IResumable
{
  private:
    int number_left = 0;
    size_t tag_next;

  public:
    WaitForOtherEventsResumable(size_t tag_next);

    void wait_for_one();

    bool resume(Scheduler &wd, std::unique_ptr<IResumable> & /* resumable */);
};

/**
 * @brief gRPC Queue used in the event loop.
 */
struct SharedgRPCCompletionQueue
{
    std::shared_ptr<grpc::CompletionQueue> cq;
    size_t pending = 0;
};

/**
 * @brief An objective function that is evaluated asynchronously by deferring the evaluation
 * to a remote server.
 *
 * Note: the approach using this method must support asynchronous evaluation. Specifically by adding on a callback
 * to the provided Scheduler's tagstack.
 */
class RemoteAsyncObjectiveFunction : public ObjectiveFunction
{
  private:
    std::shared_ptr<ObjectiveFunction> blueprint;
    std::string host;
    std::shared_ptr<Scheduler> s;
    std::shared_ptr<grpc::Channel> channel;
    std::unique_ptr<Evaluator::Stub> evaluator_stub_;

    std::shared_ptr<SharedgRPCCompletionQueue> scq;

    struct EvaluationResumable : public IResumable
    {
        // Under evaluation
        Individual i;
        grpc::ClientContext ctx;
        EvaluationEvaluateSolutionsRequest req;

        EvaluationEvaluateSolutionsResponse res;
        grpc::Status status;

        std::optional<size_t> completion_tag;
        Population *population;
        SharedgRPCCompletionQueue *q;

        bool resume(Scheduler &wd, std::unique_ptr<IResumable> & /* resumable */);
    };

  public:
    RemoteAsyncObjectiveFunction(std::shared_ptr<Scheduler> s,
                                 std::string host,
                                 std::shared_ptr<ObjectiveFunction> blueprint = nullptr);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;
};

class ConnectionCountShutdownService final : public Connectable::Service
{
  private:
    size_t count_connected = 0;
    std::mutex mtx;
    std::promise<void> request_shutdown;
    std::optional<std::thread> th;

    class ShutdownThreadException : public std::exception {};

  public:
    ConnectionCountShutdownService(std::future<grpc::Server*> server)
    {
        auto shutdown_server_upon_request = [](std::future<void> f, std::future<grpc::Server*> server)
        {
            try
            {
                // std::cout << "Shutdown-o-matic is waiting for a reference to the server." << std::endl;
                server.wait();
                auto s = server.get();
                // std::cout << "Shutdown-o-matic is waiting shutdown to be requested." << std::endl;
                f.wait();
                f.get();
                // std::cout << "Shutdown is requested. Shutting down..." << std::endl;
                s->Shutdown();
            }
            catch (std::exception &e)
            {
                // Do nothing, an exception is thrown by server.wait
                // std::cout << "Shutdown-o-matic aborted." << std::endl;
            }
        };

        th = std::thread(shutdown_server_upon_request, request_shutdown.get_future(), std::move(server));
    }

    ~ConnectionCountShutdownService()
    {
        if (th.has_value())
        {
            // Stop the wait!
            request_shutdown.set_exception(std::make_exception_ptr(ShutdownThreadException()));
            th->join();
        }
    }

    grpc::Status Connect(grpc::ServerContext *, const ConnectRequest *, ConnectResponse *) override
    {
        mtx.lock();
        count_connected++;
        mtx.unlock();
        return grpc::Status::OK;
    }
    grpc::Status Disconnect(grpc::ServerContext *, const DisconnectRequest *, DisconnectResponse *) override
    {

        mtx.lock();
        t_assert(count_connected > 0, "Cannot disconnect more than connected. This is a bug.");
        count_connected--;
        if (count_connected == 0)
        {
            // Inform that the server should be shut down.
            request_shutdown.set_value();
        }
        mtx.unlock();
        return grpc::Status::OK;
    }
};

class RemoteProblemEvaluatorService final : public Evaluator::Service
{
    std::shared_ptr<Population> pop;
    std::shared_ptr<ObjectiveFunction> problem;
    std::mutex mtx;
    Individual i;

    std::unique_ptr<ConnectionCountShutdownService> ccss;

    std::optional<std::string> my_uri;
    std::vector<std::string> uris_to_register_with;

    void send_register_request(const std::string &uri);

  public:
    RemoteProblemEvaluatorService(std::shared_ptr<ObjectiveFunction> problem);

    grpc::Status EvaluateSolutions(grpc::ServerContext * /* context */,
                                   const EvaluationEvaluateSolutionsRequest *req,
                                   EvaluationEvaluateSolutionsResponse *res);

    std::unique_ptr<grpc::Server> start_server(const std::string &uri, int *selected_port);

    void register_with(const std::string &uri);
};

class GRPCEventQueue : public ITaggedEventQueue
{
    std::shared_ptr<SharedgRPCCompletionQueue> scq;

  public:
    bool next(Scheduler &wd, size_t *tag) override;

    void registerData() override;
};
