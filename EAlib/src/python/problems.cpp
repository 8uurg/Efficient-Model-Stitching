#include "base.hpp"
#include "pybindi.h"

#include "problems.hpp"

void pybind_problems(py::module_ &m)
{
    // General purpose functions
    py::class_<DiscreteObjectiveFunction, ObjectiveFunction, std::shared_ptr<DiscreteObjectiveFunction>>(
        m, "DiscreteObjectiveFunction")
        .def(py::init<std::function<double(std::vector<char> &)>, size_t, std::vector<char>, size_t>(),
             py::arg("evaluation_function"),
             py::arg("l"),
             py::arg("alphabet_size"),
             py::arg("index") = 0)
        .def(py::init<std::function<double(std::vector<char> &)>, size_t, std::vector<short>, size_t>(),
             py::arg("evaluation_function"),
             py::arg("l"),
             py::arg("alphabet_size"),
             py::arg("index") = 0)
        .def(py::pickle(
            [](DiscreteObjectiveFunction &dof) {
                return py::make_tuple(
                    dof.get_evaluation_function(), dof.get_l(), dof.get_alphabet_size(), dof.get_index());
            },
            [](py::tuple &t) {
                t_assert(t.size() == 4, "Invalid pickle");
                return DiscreteObjectiveFunction(t[0].cast<std::function<double(std::vector<char> &)>>(),
                                                 t[1].cast<size_t>(),
                                                 t[2].cast<std::vector<char>>(),
                                                 t[3].cast<size_t>());
            }));
    py::class_<ContinuousObjectiveFunction, ObjectiveFunction, std::shared_ptr<ContinuousObjectiveFunction>>(
        m, "ContinuousObjectiveFunction")
        .def(py::init<std::function<double(std::vector<double> &)>, size_t, size_t>(),
             py::arg("evaluation_function"),
             py::arg("l"),
             py::arg("index") = 0)
        .def(py::pickle(
            [](ContinuousObjectiveFunction &dof) {
                return py::make_tuple(dof.get_evaluation_function(), dof.get_l(), dof.get_index());
            },
            [](py::tuple &t) {
                t_assert(t.size() == 3, "Invalid pickle");
                return ContinuousObjectiveFunction(t[0].cast<std::function<double(std::vector<double> &)>>(),
                                                   t[1].cast<size_t>(),
                                                   t[3].cast<size_t>());
            }));

    // -- Benchmark Functions --

    // Onemax & Zeromax
    py::class_<OneMax, ObjectiveFunction, std::shared_ptr<OneMax>>(m, "OneMax")
        .def(py::init<size_t, size_t>(), py::arg("l"), py::arg("index") = 0)
        .def(py::pickle([](OneMax &om) { return py::make_tuple(om.get_l(), om.get_index()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle");
                            return OneMax(t[0].cast<size_t>(), t[1].cast<size_t>());
                        }));
    py::class_<ZeroMax, ObjectiveFunction, std::shared_ptr<ZeroMax>>(m, "ZeroMax")
        .def(py::init<size_t, size_t>(), py::arg("l"), py::arg("index") = 0)
        .def(py::pickle([](ZeroMax &om) { return py::make_tuple(om.get_l(), om.get_index()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle");
                            return ZeroMax(t[0].cast<size_t>(), t[1].cast<size_t>());
                        }));

    // Maxcut
    py::class_<Edge>(m, "Edge")
        .def(py::init<size_t, size_t, double>(), py::arg("i"), py::arg("j"), py::arg("w"))
        .def(py::pickle([](Edge &e) { return py::make_tuple(e.i, e.j, e.w); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 3, "Invalid pickle");
                            return Edge{t[0].cast<size_t>(), t[1].cast<size_t>(), t[2].cast<double>()};
                        }));
    py::class_<MaxCutInstance>(m, "MaxCutInstance")
        .def(py::init<size_t, size_t, std::vector<Edge>>(),
             py::arg("num_vertices"),
             py::arg("num_edges"),
             py::arg("edges"))
        .def(py::pickle(
            [](MaxCutInstance &instance) {
                return py::make_tuple(instance.num_vertices, instance.num_edges, instance.edges);
            },
            [](py::tuple &t) {
                t_assert(t.size() == 3, "Invalid pickle");
                return MaxCutInstance{t[0].cast<size_t>(), t[1].cast<size_t>(), t[2].cast<std::vector<Edge>>()};
            }));
    py::class_<MaxCut, ObjectiveFunction, std::shared_ptr<MaxCut>>(m, "MaxCut")
        .def(py::init<MaxCutInstance, size_t>(), py::arg("instance"), py::arg("index") = 0)
        .def(py::init<std::filesystem::path &, size_t>(), py::arg("path"), py::arg("index") = 0)
        .def(py::pickle([](MaxCut &mc) { return py::make_tuple(mc.get_instance(), mc.get_index()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle");
                            return MaxCut(t[0].cast<MaxCutInstance>(), t[1].cast<size_t>());
                        }));

    // NK-Landscape
    py::class_<NKSubfunction>(m, "NKSubfunction")
        .def(py::init<std::vector<size_t>, std::vector<double>>(), py::arg("variables"), py::arg("lut"))
        .def(py::pickle([](NKSubfunction &nksf) { return py::make_tuple(nksf.variables, nksf.lut); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle");
                            return NKSubfunction{t[0].cast<std::vector<size_t>>(), t[1].cast<std::vector<double>>()};
                        }));
    py::class_<NKLandscapeInstance>(m, "NKLandscapeInstance")
        .def(py::init<std::vector<NKSubfunction>, size_t>(), py::arg("subfunctions"), py::arg("l"))
        .def(py::pickle([](NKLandscapeInstance &nksf) { return py::make_tuple(nksf.subfunctions, nksf.l); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle");
                            return NKLandscapeInstance{t[0].cast<std::vector<NKSubfunction>>(), t[1].cast<size_t>()};
                        }));
    py::class_<NKLandscape, ObjectiveFunction, std::shared_ptr<NKLandscape>>(m, "NKLandscape")
        .def(py::init<std::filesystem::path &, size_t>(), py::arg("path"), py::arg("index") = 0)
        .def(py::pickle([](NKLandscape &nkl) { return py::make_tuple(nkl.get_instance(), nkl.get_index()); },
                        [](py::tuple &t) {
                            t_assert(t.size() == 2, "Invalid pickle");
                            return NKLandscape(t[0].cast<NKLandscapeInstance>(), t[1].cast<size_t>());
                        }));

    // Best-of-Traps
    py::class_<ConcatenatedPermutedTrap>(m, "ConcatenatedPermutedTrap")
        .def(py::init<size_t, size_t, std::vector<size_t>, std::vector<char>>(),
             py::arg("number_of_parameters"),
             py::arg("block_size"),
             py::arg("permutation"),
             py::arg("optimum"))
        .def(py::pickle(
            [](ConcatenatedPermutedTrap &cpt) {
                return py::make_tuple(cpt.number_of_parameters, cpt.block_size, cpt.permutation, cpt.optimum);
            },
            [](py::tuple &t) {
                t_assert(t.size() == 4, "Invalid pickle");
                return ConcatenatedPermutedTrap{t[0].cast<size_t>(),
                                                t[1].cast<size_t>(),
                                                t[2].cast<std::vector<size_t>>(),
                                                t[3].cast<std::vector<char>>()};
            }));
    py::class_<BestOfTrapsInstance>(m, "BestOfTrapsInstance")
        .def(py::init<size_t, std::vector<ConcatenatedPermutedTrap>>(),
             py::arg("l"),
             py::arg("concatenatedPermutedTraps"))
        .def(py::pickle(
            [](BestOfTrapsInstance &cpt) { return py::make_tuple(cpt.l, cpt.concatenatedPermutedTraps); },
            [](py::tuple &t) {
                t_assert(t.size() == 2, "Invalid pickle");
                return BestOfTrapsInstance{t[0].cast<size_t>(), t[1].cast<std::vector<ConcatenatedPermutedTrap>>()};
            }));

    py::class_<BestOfTraps, ObjectiveFunction, std::shared_ptr<BestOfTraps>>(m, "BestOfTraps")
        .def(py::init<BestOfTrapsInstance, size_t>(), py::arg("instance"), py::arg("index") = 0)
        .def(py::init<std::filesystem::path &, size_t>(), py::arg("path"), py::arg("index") = 0)
        .def(py::pickle(
            [](BestOfTraps &bot) { return py::make_tuple(bot.get_instance(), bot.get_index()); },
            [](py::tuple &t) {
                t_assert(t.size() == 2, "Invalid pickle");
                return BestOfTraps(t[0].cast<BestOfTrapsInstance>(), t[1].cast<size_t>());
            }));

    py::class_<WorstOfTraps, ObjectiveFunction, std::shared_ptr<WorstOfTraps>>(m, "WorstOfTraps")
        .def(py::init<BestOfTrapsInstance, size_t>(), py::arg("instance"), py::arg("index") = 0)
        .def(py::init<std::filesystem::path &, size_t>(), py::arg("path"), py::arg("index") = 0)
        .def(py::pickle(
            [](WorstOfTraps &bot) { return py::make_tuple(bot.get_instance(), bot.get_index()); },
            [](py::tuple &t) {
                t_assert(t.size() == 2, "Invalid pickle");
                return WorstOfTraps(t[0].cast<BestOfTrapsInstance>(), t[1].cast<size_t>());
            }));
    // Compose multiple functions together.
    py::class_<Compose, ObjectiveFunction, std::shared_ptr<Compose>>(m, "Compose")
        .def(py::init<std::vector<std::shared_ptr<ObjectiveFunction>>>(), py::arg("objective_functions"))
        .def(py::pickle(
            [](Compose &cp) { return py::make_tuple(cp.get_problems()); },
            [](py::tuple &t) {
                t_assert(t.size() == 1, "Invalid pickle");
                return Compose(t[0].cast<std::vector<std::shared_ptr<ObjectiveFunction>>>());
            }));
    //
    py::class_<HierarchicalDeceptiveTrap, ObjectiveFunction, std::shared_ptr<HierarchicalDeceptiveTrap>>(
        m, "HierarchicalDeceptiveTrap")
        .def(py::init<size_t, size_t, size_t>(), py::arg("l"), py::arg("k") = 3, py::arg("index") = 0)
        .def(py::pickle(
            [](HierarchicalDeceptiveTrap &hdt) { return py::make_tuple(hdt.get_l(), hdt.get_k(), hdt.get_index()); },
            [](py::tuple &t) {
                t_assert(t.size() == 3, "Invalid pickle");
                return HierarchicalDeceptiveTrap(t[0].cast<size_t>(), t[1].cast<size_t>(), t[2].cast<size_t>());
            }));

    // Exceptions
    py::register_exception<missing_file>(m, "FileNotFound");
    py::register_exception<invalid_instance>(m, "InvalidInstance");
}