#include "pybindi.h"

#include "ga.hpp"

void pybind_ga(py::module_ &m)
{
    // - crossovers
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ICrossover, IDataUser, std::shared_ptr<ICrossover>>(m, "ICrossover");
    py::class_<UniformCrossover, ICrossover, std::shared_ptr<UniformCrossover>>(m, "UniformCrossover")
        .def(py::init<double>(), py::arg("p") = 0.5);
    py::class_<KPointCrossover, ICrossover, std::shared_ptr<KPointCrossover>>(m, "KPointCrossover")
        .def(py::init<size_t>(), py::arg("k"));
    py::class_<SubfunctionCrossover, ICrossover, std::shared_ptr<SubfunctionCrossover>>(m, "SubfunctionCrossover")
        .def(py::init<double>(), py::arg("p") = 0.5);
    // - mutation
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IMutation, IDataUser, std::shared_ptr<IMutation>>(m, "IMutation");
    py::class_<NoMutation, IMutation, std::shared_ptr<NoMutation>>(m, "NoMutation").def(py::init<>());
    py::class_<PerVariableBitFlipMutation, IMutation, std::shared_ptr<PerVariableBitFlipMutation>>(
        m, "PerVariableBitFlipMutation")
        .def(py::init<double>(), py::arg("p"));
    py::class_<PerVariableInAlphabetMutation, IMutation, std::shared_ptr<PerVariableInAlphabetMutation>>(
        m, "PerVariableInAlphabetMutation")
        .def(py::init<double>(), py::arg("p"));

    // - selection
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ISelection, IDataUser, std::shared_ptr<ISelection>>(m, "ISelection");
    py::class_<RandomSelection, ISelection, std::shared_ptr<RandomSelection>>(m, "RandomSelection").def(py::init<>());
    py::class_<RandomUniqueSelection, ISelection, std::shared_ptr<RandomUniqueSelection>>(m, "RandomUniqueSelection")
        .def(py::init<>());
    py::class_<ShuffledSequentialSelection, ISelection, std::shared_ptr<ShuffledSequentialSelection>>(
        m, "ShuffledSequentialSelection")
        .def(py::init<>());
    py::class_<OrderedTournamentSelection, ISelection, std::shared_ptr<OrderedTournamentSelection>>(
        m, "OrderedTournamentSelection")
        .def(py::init<size_t, size_t, std::shared_ptr<ISelection>, std::shared_ptr<IPerformanceCriterion>>(),
             py::arg("tournament_size"),
             py::arg("samples_per_tournament"),
             py::arg("pool_selection"),
             py::arg("comparator"));
    py::class_<TruncationSelection, ISelection, std::shared_ptr<TruncationSelection>>(m, "TruncationSelection")
        .def(py::init<std::shared_ptr<IPerformanceCriterion>>(), py::arg("comparator"));
    // - approaches
    py::class_<SimpleGA, GenerationalApproach, std::shared_ptr<SimpleGA>>(m, "SimpleGA")
        .def(py::init<size_t,
                      std::shared_ptr<ISolutionInitializer>,
                      std::shared_ptr<ICrossover>,
                      std::shared_ptr<IMutation>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<ISelection>,
                      std::shared_ptr<IPerformanceCriterion>,
                      std::optional<size_t>,
                      std::optional<bool>>(),
             py::arg("population_size"),
             py::arg("initializer"),
             py::arg("crossover"),
             py::arg("mutation"),
             py::arg("parent_selection"),
             py::arg("population_selection"),
             py::arg("performance_criterion"),
             py::arg("offspring_size") = std::nullopt,
             py::arg("copy_population_to_offspring") = true)
        .def("step", &SimpleGA::step);
}