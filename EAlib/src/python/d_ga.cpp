//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "base.hpp"
#include "d-ga.hpp"
#include "pybind11/pybind11.h"
#include "pybindi.h"

void pybind_d_ga(py::module_ &m)
{
    py::class_<DistributedSynchronousSimpleGA,
        GenerationalApproach,
        std::shared_ptr<DistributedSynchronousSimpleGA>>(m, "DistributedSynchronousSimpleGA")
        .def(py::init<
            std::shared_ptr<Scheduler>,
            size_t,
            size_t,
            int,
            std::shared_ptr<ISolutionInitializer>,
            std::shared_ptr<ICrossover>,
            std::shared_ptr<IMutation>,
            std::shared_ptr<ISelection>,
            std::shared_ptr<IPerformanceCriterion>,
            std::shared_ptr<IArchive>>(),
            py::arg("scheduler"),
            py::arg("population_size"),
            py::arg("offspring_size"),
            py::arg("replacement_strategy"),
            py::arg("initializer"),
            py::arg("crossover"),
            py::arg("mutation"),
            py::arg("parent_selection"),
            py::arg("performance_criterion"),
            py::arg("archive")
        )
        .def("step", &DistributedSynchronousSimpleGA::step);
    py::class_<DistributedAsynchronousSimpleGA,
        GenerationalApproach,
        std::shared_ptr<DistributedAsynchronousSimpleGA>>(m, "DistributedAsynchronousSimpleGA")
        .def(py::init<
            std::shared_ptr<Scheduler>,
            size_t,
            size_t,
            int,
            std::shared_ptr<ISolutionInitializer>,
            std::shared_ptr<ICrossover>,
            std::shared_ptr<IMutation>,
            std::shared_ptr<ISelection>,
            std::shared_ptr<IPerformanceCriterion>,
            std::shared_ptr<IArchive>>(),
            py::arg("scheduler"),
            py::arg("population_size"),
            py::arg("offspring_size"),
            py::arg("replacement_strategy"),
            py::arg("initializer"),
            py::arg("crossover"),
            py::arg("mutation"),
            py::arg("parent_selection"),
            py::arg("performance_criterion"),
            py::arg("archive")
        )
        .def("step", &DistributedAsynchronousSimpleGA::step);

    py::class_<DistributedRandomSearch,
        GenerationalApproach,
        std::shared_ptr<DistributedRandomSearch>>(m, "DistributedRandomSearch")
        .def(py::init<
            std::shared_ptr<Scheduler>,
            std::shared_ptr<ISolutionInitializer>,
            std::shared_ptr<IArchive>,
            size_t>(),
            py::arg("scheduler"),
            py::arg("initializer"),
            py::arg("archive"),
            py::arg("max_pending")
        )
        .def("step", &DistributedRandomSearch::step);
}