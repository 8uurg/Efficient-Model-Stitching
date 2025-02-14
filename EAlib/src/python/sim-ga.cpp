//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "base.hpp"
#include "pybindi.h"

#include "sim-ga.hpp"

void pybind_sim_ga(py::module_ &m)
{
    py::class_<SimulatedSynchronousSimpleGA, GenerationalApproach, std::shared_ptr<SimulatedSynchronousSimpleGA>>(
        m, "SimulatedSynchronousSimpleGA")
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      size_t,
                      int,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ICrossover>,
                      std::shared_ptr<IMutation>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<SpanLogger>>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("offspring_size"),
             py::arg("replacement_strategy"),
             py::arg("initializer"),
             py::arg("crossover"),
             py::arg("mutation"),
             py::arg("parent_selection"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("sl") = py::none())
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      size_t,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ICrossover>,
                      std::shared_ptr<IMutation>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      bool,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<SpanLogger>>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("offspring_size"),
             py::arg("initializer"),
             py::arg("crossover"),
             py::arg("mutation"),
             py::arg("parent_selection"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("include_population"),
             py::arg("generationalish_selection"),
             py::arg("sl") = py::none());

    py::class_<SimulatedAsynchronousSimpleGA, SimulatedSynchronousSimpleGA, std::shared_ptr<SimulatedAsynchronousSimpleGA>>(
        m, "SimulatedAsynchronousSimpleGA")
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      size_t,
                      int,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ICrossover>,
                      std::shared_ptr<IMutation>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<SpanLogger>>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("offspring_size"),
             py::arg("replacement_strategy"),
             py::arg("initializer"),
             py::arg("crossover"),
             py::arg("mutation"),
             py::arg("parent_selection"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("sl") = py::none())
        .def(py::init<std::shared_ptr<SimulatorParameters>,
                      size_t,
                      size_t,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ICrossover>,
                      std::shared_ptr<IMutation>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      bool,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<SpanLogger>>(),
             py::arg("simulator_parameters"),
             py::arg("population_size"),
             py::arg("offspring_size"),
             py::arg("initializer"),
             py::arg("crossover"),
             py::arg("mutation"),
             py::arg("parent_selection"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("include_population"),
             py::arg("generationalish_selection"),
             py::arg("sl") = py::none());
}