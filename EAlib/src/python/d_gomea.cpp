//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "base.hpp"
#include "d-gomea.hpp"
#include "pybind11/pybind11.h"
#include "pybindi.h"
#include <limits>

void pybind_d_gomea(py::module_ &m)
{
    py::class_<DistributedSynchronousGOMEA, GOMEA, std::shared_ptr<DistributedSynchronousGOMEA>>(
        m, "DistributedSynchronousGOMEA")
        .def(py::init<std::shared_ptr<Scheduler>,
                      size_t,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      bool,
                      bool>(),
             py::arg("scheduler"),
             py::arg("population_size"),
             py::arg("initializer"),
             py::arg("fos_learner"),
             py::arg("acceptance_criterion"),
             py::arg("archive"),
             py::arg("autowrap") = true,
             py::arg("steadystate") = false)
        .def("step", &DistributedSynchronousGOMEA::step);

    // 
    py::class_<DistributedAsynchronousBaseGOMEA, GenerationalApproach, std::shared_ptr<DistributedAsynchronousBaseGOMEA>>(
        m, "DistributedAsynchronousBaselGOMEA");

    py::enum_<LKSimpleStrategy>(m, "LKSimpleStrategy")
        .value("SQRT_SYM", LKSimpleStrategy::SQRT_SYM)
        .value("SQRT_ASYM", LKSimpleStrategy::SQRT_ASYM)
        .value("RAND_SYM", LKSimpleStrategy::RAND_SYM)
        .value("RAND_ASYM", LKSimpleStrategy::RAND_ASYM)
        .value("RAND_INT_SYM", LKSimpleStrategy::RAND_INT_SYM)
        .value("RAND_INT_ASYM", LKSimpleStrategy::RAND_INT_ASYM)
        .export_values();

    py::class_<LKStrategy, std::shared_ptr<LKStrategy>>(m, "LKStrategy");
    py::class_<SimpleStrategy, LKStrategy, std::shared_ptr<SimpleStrategy>>(m, "SimpleStrategy")
        .def(py::init<LKSimpleStrategy, double>(),
            py::arg("strategy"),
            py::arg("distance_threshold") = -std::numeric_limits<double>::infinity());

    py::class_<DistributedAsynchronousKernelGOMEA, DistributedAsynchronousBaseGOMEA, std::shared_ptr<DistributedAsynchronousKernelGOMEA>>(
        m, "DistributedAsynchronousKernelGOMEA")
        .def(py::init<std::shared_ptr<Scheduler>,
                      size_t,
                      size_t,
                      std::vector<size_t>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      bool,
                      bool,
                      bool,
                      bool,
                      std::shared_ptr<LKStrategy>>(),
             py::arg("scheduler"),
             py::arg("population_size"),
             py::arg("num_clusters"),
             py::arg("objective_indices"),
             py::arg("initializer"),
             py::arg("fos_learner"),
             py::arg("acceptance_criterion"),
             py::arg("archive"),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true,
             py::arg("steadystate") = false,
             py::arg("sample_from_copies") = true,
             py::arg("lkg_strategy") = py::none()
            )
        .def("step", &DistributedAsynchronousGOMEA::step);

    py::class_<DistributedAsynchronousGOMEA, DistributedAsynchronousBaseGOMEA, std::shared_ptr<DistributedAsynchronousGOMEA>>(
        m, "DistributedAsynchronousGOMEA")
        .def(py::init<std::shared_ptr<Scheduler>,
                      size_t,
                      size_t,
                      std::vector<size_t>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      bool,
                      bool,
                      bool,
                      bool>(),
             py::arg("scheduler"),
             py::arg("population_size"),
             py::arg("num_clusters"),
             py::arg("objective_indices"),
             py::arg("initializer"),
             py::arg("fos_learner"),
             py::arg("acceptance_criterion"),
             py::arg("archive"),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true,
             py::arg("steadystate") = false,
             py::arg("sample_from_copies") = true)
        .def("step", &DistributedAsynchronousGOMEA::step);
}