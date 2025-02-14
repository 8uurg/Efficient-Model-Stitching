//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "pybindi.h"

#include "gomea.hpp"

void pybind_gomea(py::module_ &m)
{
    // - linkage metrics
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<LinkageMetric, IDataUser, std::shared_ptr<LinkageMetric>>(m, "LinkageMetric");
    py::class_<MI, LinkageMetric, std::shared_ptr<MI>>(m, "MI").def(py::init<>());
    py::class_<NMI, LinkageMetric, std::shared_ptr<NMI>>(m, "NMI").def(py::init<>());
    py::class_<RandomLinkage, LinkageMetric, std::shared_ptr<RandomLinkage>>(m, "RandomLinkage").def(py::init<>());
    py::class_<FixedLinkage, LinkageMetric, std::shared_ptr<FixedLinkage>>(m, "FixedLinkage")
        .def(py::init<SymMatrix<double>, std::optional<double>, std::optional<double>>(),
             py::arg("matrix"),
             py::arg("minimum_threshold") = std::nullopt,
             py::arg("maximum_threshold") = std::nullopt);
    // - fos learners
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<FoSLearner, IDataUser, std::shared_ptr<FoSLearner>>(m, "FoSLearner");
    py::class_<CategoricalUnivariateFoS, FoSLearner, std::shared_ptr<CategoricalUnivariateFoS>>(
        m, "CategoricalUnivariateFoS")
        .def(py::init<>());

    py::enum_<FoSOrdering>(m, "FoSOrdering")
        .value("AsIs", FoSOrdering::AsIs)
        .value("Random", FoSOrdering::Random)
        .value("SizeIncreasing", FoSOrdering::SizeIncreasing);
    py::class_<CategoricalLinkageTree, FoSLearner, std::shared_ptr<CategoricalLinkageTree>>(m, "CategoricalLinkageTree")
        .def(py::init<std::shared_ptr<LinkageMetric>, FoSOrdering, bool, bool, bool>(),
             py::arg("linkage_metric"),
             py::arg("ordering") = FoSOrdering::AsIs,
             py::arg("prune_minima") = false,
             py::arg("prune_maxima") = false,
             py::arg("prune_root") = true);
    // - sampling distributions (not currently exposed to python)
    // py::class_<ISamplingDistribution>(m, "ISamplingDistribution");
    // py::class_<CategoricalPopulationSamplingDistribution, ISamplingDistribution>(
    //     m, "CategoricalPopulationSamplingDistribution");
    // - incremental improvement operator
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IIncrementalImprovementOperator, IDataUser, std::shared_ptr<IIncrementalImprovementOperator>>(
        m, "IIncrementalImprovementOperator");
    py::class_<GOM, IIncrementalImprovementOperator, std::shared_ptr<GOM>>(m, "GOM").def(py::init<>());
    py::class_<FI, IIncrementalImprovementOperator, std::shared_ptr<FI>>(m, "FI").def(py::init<>());
    // - actual approach
    py::class_<GOMEA, GenerationalApproach, std::shared_ptr<GOMEA>>(m, "GOMEA")
        .def(py::init<size_t,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      bool>(),
             py::arg("population_size"),
             py::arg("initializer"),
             py::arg("foslearner"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("autowrap") = true)
        .def("step", &GOMEA::step)
        .def("getSolutionPopulation", &GOMEA::getSolutionPopulation);

    // - mo-gomea
    py::class_<ArchiveAcceptanceCriterion, IPerformanceCriterion, std::shared_ptr<ArchiveAcceptanceCriterion>>(
        m, "MOGAcceptanceCriterion")
        .def(py::init<std::shared_ptr<IPerformanceCriterion>, std::shared_ptr<IArchive>, bool, bool>(),
             py::arg("wrapped"),
             py::arg("archive"),
             py::arg("accept_if_added") = true,
             py::arg("accept_if_undominated") = false);
    py::class_<WrappedOrSingleSolutionPerformanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<WrappedOrSingleSolutionPerformanceCriterion>>(
        m, "WrappedOrSingleSolutionPerformanceCriterion")
        .def(py::init<std::shared_ptr<IPerformanceCriterion>>(), py::arg("wrapped"));

    py::class_<GOMEAPlugin, IDataUser, std::shared_ptr<GOMEAPlugin>>(m, "GOMEAPlugin").def(py::init());

    py::class_<HoangScalarizationScheme, GOMEAPlugin, std::shared_ptr<HoangScalarizationScheme>>(
        m, "HoangScalarizationScheme")
        .def(py::init<std::shared_ptr<Scalarizer>, std::vector<size_t>>(),
             py::arg("scalarizer"),
             py::arg("objective_indices"));

    py::class_<MO_GOMEA, GenerationalApproach, std::shared_ptr<MO_GOMEA>>(m, "MO_GOMEA")
        .def(py::init<size_t,
                      size_t,
                      std::vector<size_t>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<GOMEAPlugin>,
                      bool,
                      bool>(),
             py::arg("population_size"),
             py::arg("number_of_clusters"),
             py::arg("objective_indices"),
             py::arg("initializer"),
             py::arg("foslearner"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("plugin") = std::make_shared<GOMEAPlugin>(),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true)
        .def("step", &MO_GOMEA::step)
        .def("getSolutionPopulation", &MO_GOMEA::getSolutionPopulation);

    py::class_<KernelGOMEA, GenerationalApproach, std::shared_ptr<KernelGOMEA>>(m, "KernelGOMEA")
        .def(py::init<size_t,
                      size_t,
                      std::vector<size_t>,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<FoSLearner>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::shared_ptr<IArchive>,
                      std::shared_ptr<GOMEAPlugin>,
                      bool,
                      bool>(),
             py::arg("population_size"),
             py::arg("number_of_clusters"),
             py::arg("objective_indices"),
             py::arg("initializer"),
             py::arg("foslearner"),
             py::arg("performance_criterion"),
             py::arg("archive"),
             py::arg("plugin") = std::make_shared<GOMEAPlugin>(),
             py::arg("donor_search") = true,
             py::arg("autowrap") = true)
        .def("step", &KernelGOMEA::step)
        .def("getSolutionPopulation", &KernelGOMEA::getSolutionPopulation);

    py::class_<ClusterIndices>(m, "ClusterIndices").def_readwrite("cluster_index", &ClusterIndices::indices);
    m.attr("CLUSTERINDICES") = CLUSTERINDICES;

    py::class_<ClusterIndex>(m, "ClusterIndex").def_readwrite("cluster_index", &ClusterIndex::cluster_index);
    m.attr("CLUSTERINDEX") = CLUSTERINDEX;

    py::class_<UseSingleObjective>(m, "UseSingleObjective").def_readwrite("index", &UseSingleObjective::index);
    m.attr("USESINGLEOBJECTIVE") = USESINGLEOBJECTIVE;

    py::class_<LinkageKernel>(m, "LinkageKernel")
        .def_readwrite("neighborhood", &LinkageKernel::neighborhood)
        .def_readwrite("pop_neighborhood", &LinkageKernel::pop_neighborhood)
        .def_property_readonly("fos",
                               [](LinkageKernel &s) { return std::shared_ptr<FoSLearner>(s.fos->cloned_ptr()); });
    m.attr("LINKAGEKERNEL") = LINKAGEKERNEL;
}