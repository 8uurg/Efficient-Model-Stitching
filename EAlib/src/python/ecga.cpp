#include "base.hpp"
#include "pybindi.h"

#include "ecga.hpp"
#include <memory>

void pybind_ecga(py::module_ &m)
{
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ISolutionSamplingDistribution, IDataUser, std::shared_ptr<ISolutionSamplingDistribution>>(
        m, "ISolutionSamplingDistribution");
    py::class_<ECGAGreedyMarginalProduct, ISolutionSamplingDistribution, std::shared_ptr<ECGAGreedyMarginalProduct>>(
        m, "ECGAGreedyMarginalProduct")
        .def(py::init<size_t, size_t>(), py::arg("update_pop_every_learn_call") = 1, py::arg("update_mpm_every_pop_update") = 1);

    py::class_<SynchronousSimulatedECGA, GenerationalApproach, std::shared_ptr<SynchronousSimulatedECGA>>(
        m, "SynchronousSimulatedECGA")
        .def(py::init<int,
                      std::shared_ptr<SimulatorParameters>,
                      size_t,
                      std::shared_ptr<ISolutionSamplingDistribution>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<SpanLogger>>(),
             py::arg("replacement_strategy"),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("distribution"),
             py::arg("initializer"),
             py::arg("selection"),
             py::arg("archive"),
             py::arg("sl") = py::none())
        .def(py::init<int,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<SimulatorParameters>,
                      size_t,
                      std::shared_ptr<ISolutionSamplingDistribution>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<SpanLogger>>(),
             py::arg("replacement_strategy"),
             py::arg("perf_criterion_replacement"),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("distribution"),
             py::arg("initializer"),
             py::arg("selection"),
             py::arg("archive"),
             py::arg("sl") = py::none());
    py::class_<AsynchronousSimulatedECGA, GenerationalApproach, std::shared_ptr<AsynchronousSimulatedECGA>>(
        m, "AsynchronousSimulatedECGA")
        .def(py::init<int,
                      std::shared_ptr<SimulatorParameters>,
                      size_t,
                      std::shared_ptr<ISolutionSamplingDistribution>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<SpanLogger>>(),
             py::arg("replacement_strategy"),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("distribution"),
             py::arg("initializer"),
             py::arg("selection"),
             py::arg("archive"),
             py::arg("sl") = py::none())
        .def(py::init<int,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<SimulatorParameters>,
                      size_t,
                      std::shared_ptr<ISolutionSamplingDistribution>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<SpanLogger>>(),
             py::arg("replacement_strategy"),
             py::arg("perf_criterion_replacement"),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("distribution"),
             py::arg("initializer"),
             py::arg("selection"),
             py::arg("archive"),
             py::arg("sl") = py::none());
}