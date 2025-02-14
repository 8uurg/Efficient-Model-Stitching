//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "pybind11/cast.h"
#include "pybind11/pytypes.h"
#include "pybindi.h"
#include <pybind11/stl.h>

#include "initializers.hpp"

void pybind_initializers(py::module_ &m)
{
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ISolutionInitializer, IDataUser, std::shared_ptr<ISolutionInitializer>>(m, "ISolutionInitializer");
    py::class_<CategoricalUniformInitializer, ISolutionInitializer, std::shared_ptr<CategoricalUniformInitializer>>(
        m, "CategoricalUniformInitializer")
        .def(py::init<>())
        .def(py::pickle(
            [](CategoricalUniformInitializer &) { return py::make_tuple(); },
            [](py::tuple &t) { t_assert(t.size() == 1, "Invalid pickle.") return CategoricalUniformInitializer(); }));
    py::class_<CategoricalWeightedInitializer, ISolutionInitializer, std::shared_ptr<CategoricalWeightedInitializer>>(
        m, "CategoricalWeightedInitializer")
        .def(py::init<probabilities>(), py::arg("p") = py::none())
        .def(py::pickle(
            [](CategoricalWeightedInitializer &c) { return py::make_tuple(c.p); },
            [](py::tuple &t) { t_assert(t.size() == 1, "Invalid pickle.") return CategoricalWeightedInitializer(t[0].cast<probabilities>()); }));
    py::class_<CategoricalProbabilisticallyCompleteInitializer,
               ISolutionInitializer,
               std::shared_ptr<CategoricalProbabilisticallyCompleteInitializer>>(
        m, "CategoricalProbabilisticallyCompleteInitializer")
        .def(py::init<variable_p>(), py::arg("p") = py::none())
        .def(py::pickle(
            [](CategoricalProbabilisticallyCompleteInitializer &c) { return py::make_tuple(c.p); },
            [](py::tuple &t) { t_assert(t.size() == 1, "Invalid pickle.") return CategoricalProbabilisticallyCompleteInitializer(t[0].cast<variable_p>()); }));
    // - performance criteria
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IPerformanceCriterion, IDataUser, std::shared_ptr<IPerformanceCriterion>>(m, "IPerformanceCriterion");
    // - generational approaches interface
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<GenerationalApproach, IDataUser, std::shared_ptr<GenerationalApproach>>(m, "GenerationalApproach")
        .def("step", &GenerationalApproach::step)
        .def("getSolutionPopulation", &GenerationalApproach::getSolutionPopulation);

    py::class_<GenerationalApproachComparator, IDataUser, std::shared_ptr<GenerationalApproachComparator>>(
        m, "GenerationalApproachComparator")
        .def(py::init<>());
    py::class_<AverageFitnessComparator, GenerationalApproachComparator, std::shared_ptr<AverageFitnessComparator>>(
        m, "AverageFitnessComparator")
        .def(py::init<>())
        .def(py::pickle(
            [](AverageFitnessComparator &) { return py::make_tuple(); },
            [](py::tuple &t) { t_assert(t.size() == 0, "Invalid pickle.") return AverageFitnessComparator(); }));
}