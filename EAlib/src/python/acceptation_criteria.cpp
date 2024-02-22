#include "base.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
namespace py = pybind11;

#include "acceptation_criteria.hpp"

void pybind_acceptation_criteria(py::module_ &m)
{
    py::class_<DominationObjectiveAcceptanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<DominationObjectiveAcceptanceCriterion>>(m, "DominationObjectiveAcceptanceCriterion")
        .def(py::init<std::vector<size_t>>(), py::arg("objective_indices"))
        .def(py::pickle([](DominationObjectiveAcceptanceCriterion &c) { return py::make_tuple(c.getIndices()); },
                        [](py::tuple c) {
                            t_assert(c.size() == 1, "Invalid pickle");
                            return DominationObjectiveAcceptanceCriterion(c[0].cast<std::vector<size_t>>());
                        }));

    py::class_<SingleObjectiveAcceptanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<SingleObjectiveAcceptanceCriterion>>(m, "SingleObjectiveAcceptanceCriterion")
        .def(py::init<size_t>(), py::arg("objective") = 0)
        .def(py::pickle([](SingleObjectiveAcceptanceCriterion &c) { return py::make_tuple(c.getIndex()); },
                        [](py::tuple c) {
                            t_assert(c.size() == 1, "Invalid pickle");
                            return SingleObjectiveAcceptanceCriterion(c[0].cast<size_t>());
                        }));

    py::class_<SequentialCombineAcceptanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<SequentialCombineAcceptanceCriterion>>(m, "SequentialCombineAcceptanceCriterion")
        .def(py::init<std::vector<std::shared_ptr<IPerformanceCriterion>>, bool>(),
             py::arg("criteria"),
             py::arg("nondeterminable_is_equal") = false);

    py::class_<ThresholdAcceptanceCriterion, IPerformanceCriterion, std::shared_ptr<ThresholdAcceptanceCriterion>>(
        m, "ThresholdAcceptanceCriterion")
        .def(py::init<size_t, double, bool>(), py::arg("objective"), py::arg("threshold"), py::arg("slack") = true)
        .def(py::pickle([](ThresholdAcceptanceCriterion &c) { return py::make_tuple(c.getIndex(), c.getThreshold(), c.getSlack()); },
                        [](py::tuple c) {
                            t_assert(c.size() == 2, "Invalid pickle");
                            return ThresholdAcceptanceCriterion(c[0].cast<size_t>(), c[1].cast<double>(), c[2].cast<bool>());
                        }))
        .def("get_threshold", &ThresholdAcceptanceCriterion::getThreshold)
        .def("set_threshold", &ThresholdAcceptanceCriterion::setThreshold);

    py::class_<Scalarizer, IDataUser, std::shared_ptr<Scalarizer>>(m, "Scalarizer");

    py::class_<TschebysheffObjectiveScalarizer, Scalarizer, std::shared_ptr<TschebysheffObjectiveScalarizer>>(
        m, "TschebysheffObjectiveScalarizer")
        .def(py::init<std::vector<size_t>>(), py::arg("objective_indices"))
        .def(py::pickle([](TschebysheffObjectiveScalarizer &c) { return py::make_tuple(c.get_objective_indices()); },
                        [](py::tuple c) {
                            t_assert(c.size() == 1, "Invalid pickle");
                            return TschebysheffObjectiveScalarizer(c[0].cast<std::vector<size_t>>());
                        }));

    py::class_<ScalarizationAcceptanceCriterion,
               IPerformanceCriterion,
               std::shared_ptr<ScalarizationAcceptanceCriterion>>(m, "ScalarizationAcceptanceCriterion")
        .def(py::init<std::shared_ptr<Scalarizer>, bool>(), py::arg("scalarizer"), py::arg("use_weights_of_first") = false)
        .def(py::pickle([](ScalarizationAcceptanceCriterion &c) { return py::make_tuple(c.get_scalarizer()); },
                        [](py::tuple c) {
                            t_assert(c.size() == 1, "Invalid pickle");
                            return ScalarizationAcceptanceCriterion(c[0].cast<std::shared_ptr<Scalarizer>>());
                        }));

    py::class_<FunctionAcceptanceCriterion, IPerformanceCriterion, std::shared_ptr<FunctionAcceptanceCriterion>>(
        m, "FunctionAcceptanceCriterion")
        .def(py::init<std::function<short(Population & pop, Individual & a, Individual & b)>>(), py::arg("criterion"));
}