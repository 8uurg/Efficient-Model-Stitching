//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "base.hpp"
#include "pybindi.h"

#include "sim-gomea.hpp"

void pybind_sim_gomea(py::module_ &m)
{
    py::class_<SimParallelSynchronousGOMEA, GOMEA, std::shared_ptr<SimParallelSynchronousGOMEA>>(
        m, "SimParallelSynchronousGOMEA")
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<SpanLogger>,
                      bool,
                      bool>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("initializer"),
             py::arg("foslearner"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("span_logger") = py::none(),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true);

    py::class_<SimParallelSynchronousMO_GOMEA, MO_GOMEA, std::shared_ptr<SimParallelSynchronousMO_GOMEA>>(
        m, "SimParallelSynchronousMO_GOMEA")
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      size_t,
                      std::vector<size_t>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<GOMEAPlugin>,
                      std::shared_ptr<SpanLogger>,
                      bool,
                      bool>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("number_of_clusters"),
             py::arg("objective_indices"),
             py::arg("initializer"),
             py::arg("foslearner"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("plugin") = std::make_shared<GOMEAPlugin>(),
             py::arg("span_logger") = py::none(),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true);

    py::class_<SimParallelSynchronousKernelGOMEA, KernelGOMEA, std::shared_ptr<SimParallelSynchronousKernelGOMEA>>(
        m, "SimParallelSynchronousKernelGOMEA")
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      size_t,
                      std::vector<size_t>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<GOMEAPlugin>,
                      std::shared_ptr<SpanLogger>,
                      bool,
                      bool>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("number_of_clusters"),
             py::arg("objective_indices"),
             py::arg("initializer"),
             py::arg("foslearner"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("plugin") = std::make_shared<GOMEAPlugin>(),
             py::arg("span_logger") = py::none(),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true);

    py::class_<SimParallelAsynchronousKernelGOMEA,
               GenerationalApproach,
               std::shared_ptr<SimParallelAsynchronousKernelGOMEA>>(m, "SimParallelAsynchronousKernelGOMEA")
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      size_t,
                      std::vector<size_t>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      //   std::shared_ptr<GOMEAPlugin>,
                      std::shared_ptr<SpanLogger>,
                      bool,
                      bool>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("number_of_clusters"),
             py::arg("objective_indices"),
             py::arg("initializer"),
             py::arg("foslearner"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             //   py::arg("plugin") = std::make_shared<GOMEAPlugin>(),
             py::arg("span_logger") = py::none(),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true);

    py::class_<SimParallelAsynchronousGOMEA,
               SimParallelAsynchronousKernelGOMEA,
               std::shared_ptr<SimParallelAsynchronousGOMEA>>(m, "SimParallelAsynchronousGOMEA")
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      size_t,
                      std::vector<size_t>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      //   std::shared_ptr<GOMEAPlugin>,
                      std::shared_ptr<SpanLogger>,
                      bool,
                      bool>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("number_of_clusters"),
             py::arg("objective_indices"),
             py::arg("initializer"),
             py::arg("foslearner"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             //   py::arg("plugin") = std::make_shared<GOMEAPlugin>(),
             py::arg("span_logger") = py::none(),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true);
}