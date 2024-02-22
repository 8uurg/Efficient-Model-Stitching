#include "decentralized.hpp"
#include "base.hpp"
#include "decentralized_py.hpp"
#include "dispatcher.hpp"
#include "pybind11/detail/common.h"
#include "pybind11/pytypes.h"
#include "pybindi.h"
#include <string>
#include <thread>

class PyServerWrapper
{
    std::unique_ptr<grpc::Server> server;
    std::string host;
    int port;

  public:
    PyServerWrapper(std::unique_ptr<grpc::Server> server, std::string host, int port) :
        server(std::move(server)), host(host), port(port)
    {
    }

    void shutdown()
    {
        server->Shutdown();
    }

    void wait()
    {
        server->Wait();
    }

    int get_port()
    {
        return port;
    }

    std::string get_uri()
    {
        return host.substr(0, host.find_last_of(':') + 1) + std::to_string(port);
    }
};

class DispatcherServerPyWrapper
{
    std::string uri;
    DispatcherServer server;
    int port = 0;
    std::optional<std::thread> th;
    bool has_shutdown = false;

  public:
    DispatcherServerPyWrapper(std::string uri) : uri(uri), server(uri, &port)
    {
    }

    ~DispatcherServerPyWrapper()
    {
        if (!has_shutdown)
            server.shutdown();
        join();
    }

    void shutdown()
    {
        if (!has_shutdown)
        {
            server.shutdown();
            has_shutdown = true;
        }
    }

    void start()
    {
        server.start_server();
    }

    void start_handling_requests()
    {
        th = std::thread([this]() { server.start_handling_requests(); });
    }

    void join()
    {
        if (th.has_value())
            th->join();
        th.reset();
    }

    std::string get_uri()
    {
        return uri.substr(0, uri.find_last_of(':') + 1) + std::to_string(port);
    }
};

class PyResumable : public IResumable
{
    std::function<void()> fn;

  public:
    PyResumable(std::function<void()> fn) : fn(fn)
    {
    }

    bool resume(Scheduler &, std::unique_ptr<IResumable> &) override
    {
        fn();
        return false;
    }

    IResumable *clone()
    {
        return new PyResumable(fn);
    }
};

void pybind_decentralized(py::module_ &m)
{
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IResumable, std::unique_ptr<IResumable>>(m, "IResumable");

    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ITaggedEventQueue, IDataUser, std::shared_ptr<ITaggedEventQueue>>(m, "ITaggedEventQueue");
    py::class_<Scheduler, IDataUser, std::shared_ptr<Scheduler>>(m, "Scheduler")
        .def(py::init<std::shared_ptr<ITaggedEventQueue>>(), py::arg("teq"))
        .def("get_tag",
             [](Scheduler &s, PyResumable *r) {
                 // ? Note that currently it only works for PyResumables for testing purposes.
                 // In order to make it work more generally, make all IResumables cloneable.
                 return s.get_tag(std::unique_ptr<IResumable>(r->clone()));
             })
        .def("stack_tag", [](Scheduler &s, size_t tag) { s.tag_stack.push_back(tag); })
        .def("step", &Scheduler::step);

    py::class_<PyResumable, IResumable, std::unique_ptr<PyResumable>>(m, "PyResumable")
        .def(py::init<std::function<void()>>(), py::arg("func"));

    py::class_<GRPCEventQueue, ITaggedEventQueue, std::shared_ptr<GRPCEventQueue>>(m, "GRPCEventQueue")
        .def(py::init<>());

    py::class_<PythonAsyncIOEQ, ITaggedEventQueue, std::shared_ptr<PythonAsyncIOEQ>>(m, "PythonAsyncIOEQ")
        .def(py::init<py::object>(), py::arg("event_loop") = py::none());

    py::class_<PythonAsyncIOEQ::TagCompleter, std::shared_ptr<PythonAsyncIOEQ::TagCompleter>>(m, "TagCompleter")
        .def("provide_tag", &PythonAsyncIOEQ::TagCompleter::provide_tag);

    py::class_<RemoteAsyncObjectiveFunction, ObjectiveFunction, std::shared_ptr<RemoteAsyncObjectiveFunction>>(
        m, "RemoteAsyncObjectiveFunction")
        .def(py::init<std::shared_ptr<Scheduler>, std::string, std::shared_ptr<ObjectiveFunction>>(),
             py::arg("scheduler"),
             py::arg("host"),
             py::arg("blueprint") = py::none());

    py::class_<PyAsyncObjectiveFunction, ObjectiveFunction, std::shared_ptr<PyAsyncObjectiveFunction>>(
        m, "PyAsyncObjectiveFunction")
        .def(py::init<std::shared_ptr<Scheduler>, std::shared_ptr<ObjectiveFunction>, py::object>(),
             py::arg("scheduler"),
             py::arg("problem_template"),
             py::arg("async_evaluator"));

    py::class_<PyServerWrapper>(m, "EvaluatorServer")
        .def("shutdown", &PyServerWrapper::shutdown)
        .def("get_port", &PyServerWrapper::get_port)
        .def("get_uri", &PyServerWrapper::get_uri)
        .def("wait", &PyServerWrapper::wait);

    py::class_<RemoteProblemEvaluatorService>(m, "RemoteProblemEvaluatorService")
        .def(py::init<std::shared_ptr<ObjectiveFunction>>(), py::arg("problem"))
        .def(
            "start_server",
            [](RemoteProblemEvaluatorService &t, const std::string uri) {
                int port = 0;
                auto s = t.start_server(uri, &port);

                return PyServerWrapper{std::move(s), uri, port};
            },
            py::keep_alive<0, 1>() // Keep service alive while server is running.
            )
        .def("register_with", &RemoteProblemEvaluatorService::register_with);

    py::class_<DispatcherServerPyWrapper>(m, "DispatcherServer")
        .def(py::init<std::string>(), py::arg("uri"))
        .def("start", &DispatcherServerPyWrapper::start)
        .def("get_uri", &DispatcherServerPyWrapper::get_uri)
        .def("start_handling_requests", &DispatcherServerPyWrapper::start_handling_requests)
        .def("shutdown", &DispatcherServerPyWrapper::shutdown);
}