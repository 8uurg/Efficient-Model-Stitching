#include "pybind11/detail/common.h"
#include "pybind11/stl_bind.h"
#include "pybindi.h"

#include "base.hpp"

// Python specific data-structure
struct PyData
{
    py::object data;

    PyData()
    {
        data = py::dict();
    }

    void operator=(const PyData& o)
    {
        // Ensure associated data is copied - rather than passed by reference.
        // Otherwise with all the copyIndividual's happening, they might all end
        // up referring to the same object.
        auto deepcopy = py::module::import("copy").attr("deepcopy");
        data = deepcopy(o.data);
    }
};
const std::shared_ptr<IDataType> PYDATA =
    std::make_shared<DataType<PyData>>(typeid(PyData).name());

void pybind_base(py::module_ &m)
{
    // General wrapper for a datatype
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<IDataType, std::shared_ptr<IDataType>>(m, "DataType");

    // Python Custom Data
    py::class_<PyData>(m, "PyData")
        .def_readwrite("data", &PyData::data);
    m.attr("PYDATA") = PYDATA;

    // Datatypes in the current code
    // Note: The default behaviour for vectors, like the one we are using here is to convert
    // to the respective native python type. This results in a copy, changes hence are NOT reflected
    // in the final object.
    // Instead, we provide convinience methods to manipulate these kinds of data.
    // First of all, for reading & writing, use the buffer protocol.
    // For Objectives an objective may be out of bounds (i.e. vector is not large enough)
    // as such we additionally provide set_objective(size_t index, double value), which automatically
    // resizes, and ensure_at_least_size, for upsizing.
    py::class_<Objective>(m, "Objective", py::buffer_protocol()) //
        .def("ensure_at_least_size",
             [](Objective &o, size_t size) {
                 if (o.objectives.size() < size)
                     o.objectives.resize(size);
             })
        .def("set_objective",
             [](Objective &o, size_t index, double value) {
                 if (o.objectives.size() <= index)
                     o.objectives.resize(index + 1);
                 o.objectives[index] = value;
             })
        .def("as_list",
             [](Objective &o) {
                 // This uses the automatic conversion.
                 return o.objectives;
             })
        .def_buffer([](Objective &m) -> py::buffer_info {
            return py::buffer_info(m.objectives.data(),
                                   sizeof(double),
                                   py::format_descriptor<double>::format(),
                                   1,
                                   {m.objectives.size()},
                                   {sizeof(double)});
        })
        .def(py::pickle([](Objective &m) { return py::make_tuple(m.objectives); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 1, "Invalid pickle.");
                            return Objective{t[0].cast<std::vector<double>>()};
                        }));
    m.attr("OBJECTIVE") = OBJECTIVE;

    // Similar story as for Objective, use the buffer protocol.
    py::class_<GenotypeCategorical>(m, "GenotypeCategorical", py::buffer_protocol()) //
        .def("as_list",
             [](GenotypeCategorical &gc) {
                 // This uses the automatic conversion.
                 return gc.genotype;
             })
        .def_buffer([](GenotypeCategorical &m) -> py::buffer_info {
            return py::buffer_info(m.genotype.data(),
                                   sizeof(char),
                                   py::format_descriptor<char>::format(),
                                   1,
                                   {m.genotype.size()},
                                   {sizeof(char)});
        })
        .def(py::pickle([](GenotypeCategorical &m) { return py::make_tuple(m.genotype); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 1, "Invalid pickle.");
                            return GenotypeCategorical{t[0].cast<std::vector<char>>()};
                        }));
    m.attr("GENOTYPECATEGORICAL") = GENOTYPECATEGORICAL;

    py::class_<GenotypeContinuous>(m, "GenotypeContinuous", py::buffer_protocol())
        .def("as_list",
             [](GenotypeContinuous &gc) {
                 // This uses the automatic conversion.
                 return gc.genotype;
             })
        .def_buffer([](GenotypeContinuous &m) -> py::buffer_info {
            return py::buffer_info(m.genotype.data(),
                                   sizeof(double),
                                   py::format_descriptor<double>::format(),
                                   1,
                                   {m.genotype.size()},
                                   {sizeof(double)});
        })
        .def(py::pickle([](GenotypeContinuous &m) { return py::make_tuple(m.genotype); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 1, "Invalid pickle.");
                            return GenotypeContinuous{t[0].cast<std::vector<double>>()};
                        }));
    m.attr("GENOTYPECONTINUOUS") = GENOTYPECONTINUOUS;

    py::class_<Population, std::shared_ptr<Population>>(m, "Population")
        .def(py::init<>())
        .def("getData", &Population::getDataPython, py::return_value_policy::reference)
        .def("newIndividual", &Population::newIndividual, py::return_value_policy::copy)
        .def(
            "registerData",
            [](Population &population, IDataType &datatype) { datatype.registerDataType(population); },
            py::arg("datatype"))
        .def("copyIndividual", &Population::copyIndividual)
        .def("get_capacity", &Population::capacity)
        .def("get_size", &Population::size)
        .def("get_active", &Population::active);

    py::class_<Individual>(m, "Individual").def_readonly("i", &Individual::i);

    py::class_<IDataUser, std::shared_ptr<IDataUser>>(m, "IDataUser")
        .def("setPopulation", &IDataUser::setPopulation)
        .def("registerData", &IDataUser::registerData)
        .def("afterRegisterData", &IDataUser::afterRegisterData);
    // - objective functions
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ObjectiveFunction, IDataUser, std::shared_ptr<ObjectiveFunction>>(m, "ObjectiveFunction")
        .def("clone", [](ObjectiveFunction &of) { return std::shared_ptr<ObjectiveFunction>(of.clone()); })
        .def("evaluate", &ObjectiveFunction::evaluate);

    //   (related utilities & wrappers)
    py::class_<Limiter, ObjectiveFunction, std::shared_ptr<Limiter>>(m, "Limiter")
        .def(py::init<std::shared_ptr<ObjectiveFunction>,
                      std::optional<long long>,
                      std::optional<std::chrono::duration<double>>>(),
             py::arg("wrapping"),
             py::arg("evaluation_limit") = std::nullopt,
             py::arg("time_limit") = std::nullopt)
        .def("restart", &Limiter::restart)
        .def("get_time_spent_ms", &Limiter::get_time_spent_ms)
        .def("get_num_evaluations", &Limiter::get_num_evaluations)
        .def(py::pickle(
            [](Limiter &m) { return py::make_tuple(m.get_wrapping(), m.get_evaluation_limit(), m.get_time_limit()); },
            [](py::tuple &t) {
                t_assert(t.size() == 3, "Invalid pickle.");
                return Limiter(t[0].cast<std::shared_ptr<ObjectiveFunction>>(),
                               t[1].cast<std::optional<long long>>(),
                               t[2].cast<std::optional<std::chrono::duration<double>>>());
            }));
    py::class_<ElitistMonitor, ObjectiveFunction, std::shared_ptr<ElitistMonitor>>(m, "ElitistMonitor")
        .def(py::init<std::shared_ptr<ObjectiveFunction>, std::shared_ptr<IPerformanceCriterion>>(),
             py::arg("wrapping"),
             py::arg("criterion"))
        .def("getElitist", &ElitistMonitor::getElitist)
        .def(py::pickle([](ElitistMonitor &m) { return py::make_tuple(m.get_wrapping(), m.get_criterion()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle.");
                            return ElitistMonitor(t[0].cast<std::shared_ptr<ObjectiveFunction>>(),
                                                  t[1].cast<std::shared_ptr<IPerformanceCriterion>>());
                        }));

    py::class_<ObjectiveValuesToReachDetector, ObjectiveFunction, std::shared_ptr<ObjectiveValuesToReachDetector>>(
        m, "ObjectiveValuesToReachDetector")
        .def(py::init<std::shared_ptr<ObjectiveFunction>, std::vector<std::vector<double>>>(),
             py::arg("wrapping"),
             py::arg("vtrs"))
        .def(py::pickle([](ObjectiveValuesToReachDetector &m) { return py::make_tuple(m.get_wrapping(), m.get_vtrs()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle.");
                            return ObjectiveValuesToReachDetector(t[0].cast<std::shared_ptr<ObjectiveFunction>>(),
                                                  t[1].cast<std::vector<std::vector<double>>>());
                        }));

    py::register_exception<run_complete>(m, "RunCompleted");
    py::register_exception<evaluation_limit_reached>(m, "EvaluationLimit");
    py::register_exception<time_limit_reached>(m, "TimeLimit");
    py::register_exception<vtr_reached>(m, "AllVTRFound");
}