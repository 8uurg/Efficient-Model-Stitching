//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "pybindi.h"

#include "running.hpp"

void pybind_running(py::module_ &m)
{
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IRunner, IDataUser, std::shared_ptr<IRunner>>(m, "IRunner");
    py::class_<TerminationStepper, IRunner, std::shared_ptr<TerminationStepper>>(m, "TerminationStepper")
        .def(py::init<std::function<std::shared_ptr<GenerationalApproach>()>, std::optional<int>>(),
             py::arg("approach_factory"),
             py::arg("step_limit"));
    py::class_<InterleavedMultistartScheme, IRunner, std::shared_ptr<InterleavedMultistartScheme>>(
        m, "InterleavedMultistartScheme")
        .def(py::init<std::function<std::shared_ptr<GenerationalApproach>(size_t)>,
                      std::shared_ptr<GenerationalApproachComparator>,
                      size_t,
                      size_t,
                      size_t>(),
             py::arg("approach_factory"),
             py::arg("comparator"),
             py::arg("steps") = 4,
             py::arg("base") = 4,
             py::arg("multiplier") = 2);
    py::class_<SimpleConfigurator>(m, "SimpleConfigurator")
        .def(py::init<std::shared_ptr<ObjectiveFunction>, std::shared_ptr<IRunner>, size_t>(),
             py::arg("objective"),
             py::arg("runner"),
             py::arg("random_seed"))
        .def("run", &SimpleConfigurator::run)
        .def("step", &SimpleConfigurator::step)
        .def("getPopulation", &SimpleConfigurator::getPopulation, py::return_value_policy::reference);
}
