//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "pybindi.h"

#include "sim.hpp"
#include <optional>

void pybind_sim(py::module_ &m)
{
    py::class_<TimeSpent, std::shared_ptr<TimeSpent>>(m, "TimeSpent").def_readwrite("t", &TimeSpent::t);

    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ISimulator, std::shared_ptr<ISimulator>>(m, "ISimulator");
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IResumableSimulated, std::shared_ptr<IResumableSimulated>>(m, "IResumableSimulated");
    // Cannot wrap: FunctionalSimResumable will be have shared reference in any case (with Python)
    // py::class_<FunctionalSimResumable, IResumableSimulated, std::shared_ptr<FunctionalSimResumable>>(m,
    // "FunctionalSimResumable") .def(py::init<std::function<void(ISimulator &, double,
    // std::shared_ptr<IResumableSimulated> &)>>());

    py::class_<SimEvent, std::shared_ptr<SimEvent>>(m, "Event")
        .def_readwrite("t", &SimEvent::t)
        .def_readwrite("ord", &SimEvent::ord);
    py::class_<Simulator, ISimulator, std::shared_ptr<Simulator>>(m, "Simulator")
        .def(py::init<std::optional<double>>(), py::arg("time_limit") = std::nullopt)
        .def("now", &Simulator::now);
    py::class_<WritingSimulator, Simulator, std::shared_ptr<WritingSimulator>>(m, "WritingSimulator")
        .def(py::init<std::filesystem::path, std::optional<double>>(),
             py::arg("out_path"),
             py::arg("time_limit") = std::nullopt);

    py::class_<SimulatorParameters, std::shared_ptr<SimulatorParameters>>(m, "SimulatorParameters")
        .def(py::init<std::shared_ptr<Simulator>, size_t, bool>(),
             py::arg("simulator"),
             py::arg("num_workers"),
             py::arg("measure_overhead") = true)
        .def(py::init<size_t, std::optional<double>, bool>(),
             py::arg("num_workers"),
             py::arg("time_limit"),
             py::arg("measure_overhead") = true)
        .def_readwrite("num_workers", &SimulatorParameters::num_workers)
        .def_readwrite("measure_overhead", &SimulatorParameters::measure_overhead)
        .def_readwrite("simulator", &SimulatorParameters::simulator)
        .def_readwrite("num_workers_busy", &SimulatorParameters::num_workers_busy);

    py::class_<SimulatedFixedRuntimeObjectiveFunction,
               ObjectiveFunction,
               std::shared_ptr<SimulatedFixedRuntimeObjectiveFunction>>(m, "SimulatedFixedRuntimeObjectiveFunction")
        .def(py::init<std::shared_ptr<ObjectiveFunction>, double>(), py::arg("of"), py::arg("dt"));

    py::class_<SimulatedFunctionRuntimeObjectiveFunction,
               ObjectiveFunction,
               std::shared_ptr<SimulatedFunctionRuntimeObjectiveFunction>>(m,
                                                                           "SimulatedFunctionRuntimeObjectiveFunction")
        .def(py::init<std::shared_ptr<ObjectiveFunction>, std::function<double(Population &, Individual &)>>(),
             py::arg("of"),
             py::arg("dt"));

    py::class_<SimulationTimeLogger, ItemLogger, std::shared_ptr<SimulationTimeLogger>>(m, "SimulationTimeLogger")
        .def(py::init<std::shared_ptr<Simulator>>(), py::arg("simulator"));

    py::class_<SpanItemLogger, ItemLogger, std::shared_ptr<SpanItemLogger>>(m, "SpanItemLogger").def(py::init<>());

    py::class_<SpanLogger, std::shared_ptr<SpanLogger>>(m, "SpanLogger")
        .def(py::init<std::shared_ptr<BaseLogger>, std::shared_ptr<SpanItemLogger>>(),
             py::arg("logger"),
             py::arg("span_item_logger") = NULL);

    py::class_<SimulatedEvaluationSpanLoggerWrapper,
               ObjectiveFunction,
               std::shared_ptr<SimulatedEvaluationSpanLoggerWrapper>>(m, "SimulatedEvaluationSpanLoggerWrapper")
        .def(py::init<std::shared_ptr<ObjectiveFunction>,
                      std::shared_ptr<SpanLogger>,
                      std::shared_ptr<SimulationTimeLogger>>(),
             py::arg("wrapped"),
             py::arg("span_logger"),
             py::arg("sim_time_logger"));

    py::class_<TimeSpentItemLogger, ItemLogger, std::shared_ptr<TimeSpentItemLogger>>(m, "TimeSpentItemLogger")
        .def(py::init<>());
}