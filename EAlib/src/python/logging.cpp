//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "pybindi.h"

#include "logging.hpp"

void pybind_logging(py::module_ &m)
{
    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<ItemLogger, IDataUser, std::shared_ptr<ItemLogger>>(m, "ItemLogger");
    py::class_<SequencedItemLogger, ItemLogger, std::shared_ptr<SequencedItemLogger>>(m, "SequencedItemLogger")
        .def(py::init<std::vector<std::shared_ptr<ItemLogger>>>(), py::arg("subloggers"))
        .def(py::pickle([](SequencedItemLogger &sil) { return py::make_tuple(sil.get_subloggers()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 1, "Invalid pickle.");
                            return SequencedItemLogger(t[0].cast<std::vector<std::shared_ptr<ItemLogger>>>());
                        }));
    ;
    py::class_<GenotypeCategoricalLogger, ItemLogger, std::shared_ptr<GenotypeCategoricalLogger>>(
        m, "GenotypeCategoricalLogger")
        .def(py::init<>())
        .def(py::pickle([](GenotypeCategoricalLogger &) { return py::make_tuple(); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 0, "Invalid pickle.");
                            return GenotypeCategoricalLogger();
                        }));
    py::class_<ObjectiveLogger, ItemLogger, std::shared_ptr<ObjectiveLogger>>(m, "ObjectiveLogger")
        .def(py::init<>())
        .def(py::pickle([](ObjectiveLogger &) { return py::make_tuple(); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 0, "Invalid pickle.");
                            return ObjectiveLogger();
                        }));
    py::class_<NumEvaluationsLogger, ItemLogger, std::shared_ptr<NumEvaluationsLogger>>(m, "NumEvaluationsLogger")
        .def(py::init<std::shared_ptr<Limiter>>(), py::arg("limiter"))
        .def(py::pickle([](NumEvaluationsLogger &nel) { return py::make_tuple(nel.get_limiter()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 1, "Invalid pickle.");
                            return NumEvaluationsLogger(t[0].cast<std::shared_ptr<Limiter>>());
                        }));
    py::class_<WallTimeLogger, ItemLogger, std::shared_ptr<WallTimeLogger>>(m, "WallTimeLogger")
        .def(py::init<std::shared_ptr<Limiter>>(), py::arg("limiter"))
        .def(py::pickle([](WallTimeLogger &wtl) { return py::make_tuple(wtl.get_limiter()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 0, "Invalid pickle.");
                            return WallTimeLogger(t[0].cast<std::shared_ptr<Limiter>>());
                        }));
    py::class_<SolutionIndexLogger, ItemLogger, std::shared_ptr<SolutionIndexLogger>>(m, "SolutionIndexLogger")
        .def(py::init<>())
        .def(py::pickle([](SolutionIndexLogger &) { return py::make_tuple(); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 0, "Invalid pickle.");
                            return SolutionIndexLogger();
                        }));

    // NOLINTNEXTLINE(bugprone-unused-raii): pybind11 uses it internally
    py::class_<BaseLogger, IDataUser, std::shared_ptr<BaseLogger>>(m, "BaseLogger");
    py::class_<CSVLogger, BaseLogger, std::shared_ptr<CSVLogger>>(m, "CSVLogger")
        .def(py::init<std::filesystem::path, std::shared_ptr<ItemLogger>>(),
             py::arg("out_path"),
             py::arg("item_logger"))
        .def(py::pickle([](CSVLogger &csvl) { return py::make_tuple(csvl.get_out_path(), csvl.get_item_logger()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle.");
                            return CSVLogger(
                                t[0].cast<std::filesystem::path>(),
                                t[1].cast<std::shared_ptr<ItemLogger>>()
                            );
                        }));
}