//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "sim.hpp"
#include <filesystem>

// Event - Comparison
bool SimEvent::operator<(const SimEvent &o) const
{
    if (t > o.t)
        return true;
    if (t < o.t)
        return false;
    return ord > o.ord;
}

// Simulator
Simulator::Simulator()
{
}
Simulator::Simulator(std::optional<double> time_limit) : time_limit(time_limit)
{
}
void Simulator::insert_event(std::unique_ptr<IResumableSimulated> e, double at, std::optional<std::string> descriptor)
{
    event_new(ordinal, simulated_time, descriptor);
    event_queue.push(SimEvent{at, ordinal++, std::move(e)});
}
double Simulator::now()
{
    return simulated_time;
}
void Simulator::serial_simulation(double t)
{
    simulated_time += t;
}
void Simulator::serial_simulation(Stopwatch &sw)
{
    simulated_time += sw.measure();
}
void Simulator::step()
{
    // Nothing to do?
    if (event_queue.empty())
        return;

    // Remove topmost item
    double t = event_queue.top().t;
    size_t ord = event_queue.top().ord;
    std::unique_ptr<IResumableSimulated> ptr;
    ptr.swap(event_queue.top().ptr);
    event_queue.pop();

    simulated_time = std::max(simulated_time, t);
    if (time_limit.has_value() && simulated_time > time_limit)
    {
        sim_time_limit_reached(ordinal, simulated_time);
        throw time_limit_reached();
    }
    size_t ord_before = ordinal;
    size_t e_ord = ord;
    ptr->resume(*this, simulated_time, ptr);
    event_hit(e_ord, simulated_time, ord_before, ordinal);
}
void Simulator::simulate_until_end()
{
    simulate_until([]() { return false; });
}
void Simulator::event_new(size_t, double, std::optional<std::string>)
{
}
void Simulator::event_hit(size_t, double, size_t, size_t)
{
}
void Simulator::sim_time_limit_reached(size_t, double)
{
}

// WritingSimulator: Simulator, but writes to a log file.
WritingSimulator::WritingSimulator(std::filesystem::path out_path, std::optional<double> time_limit) :
    Simulator(time_limit), out_path(out_path)
{
    // Note: removes file if exists, danger!
    if (std::filesystem::exists(out_path))
    {
        std::filesystem::remove(out_path);
    }
}
void WritingSimulator::event_new(size_t ord, double t, std::optional<std::string> descriptor)
{
    std::ofstream out(out_path, std::ios::app);
    out << "{"
        << "\"kind\": \"new\", "
        << "\"ord\": " << ord << ", "
        << "\"t\": " << t << ", ";
    if (descriptor.has_value())
        out << "\"desc\": \"" << descriptor.value() << "\", ";
    out << "}" << std::endl;
}
void WritingSimulator::event_hit(size_t ord, double t, size_t ord_before, size_t ord_after)
{
    std::ofstream out(out_path, std::ios::app);
    out << "{"
        << "\"kind\": \"performed\", "
        << "\"ord\": " << ord << ", "
        << "\"t\": " << t << ", "
        << "\"ord_before\": " << ord_before << ", "
        << "\"ord_after\": " << ord_after << ", "
        << "}" << std::endl;
}
void WritingSimulator::sim_time_limit_reached(size_t ord, double t)
{
    std::ofstream out(out_path, std::ios::app);
    out << "{"
        << "\"kind\": \"time_limit\", "
        << "\"ord\": " << ord << ", "
        << "\"t\": " << t << ", "
        << "}" << std::endl;
}

// General simulator container \w global metadata
SimulatorParameters::SimulatorParameters(size_t num_workers, std::optional<double> time_limit, bool measure_overhead) :
    num_workers(num_workers), measure_overhead(measure_overhead), simulator(std::make_shared<Simulator>(time_limit))
{
}

SimulatorParameters::SimulatorParameters(std::shared_ptr<Simulator> simulator,
                                         size_t num_workers,
                                         bool measure_overhead) :
    num_workers(num_workers), measure_overhead(measure_overhead), simulator(simulator)
{
}

// SimulatedFixedRuntimeObjectiveFunction
SimulatedFixedRuntimeObjectiveFunction::SimulatedFixedRuntimeObjectiveFunction(std::shared_ptr<ObjectiveFunction> of,
                                                                               double dt) :
    of(of), dt(dt)
{
}
void SimulatedFixedRuntimeObjectiveFunction::setPopulation(std::shared_ptr<Population> population)
{
    ObjectiveFunction::setPopulation(population);
    of->setPopulation(population);
}
void SimulatedFixedRuntimeObjectiveFunction::registerData()
{
    of->registerData();

    Population &pop = *population;
    pop.registerData<TimeSpent>();
    tgts.emplace(pop.getDataContainer<TimeSpent>());
}
void SimulatedFixedRuntimeObjectiveFunction::afterRegisterData()
{
    of->afterRegisterData();
}
void SimulatedFixedRuntimeObjectiveFunction::evaluate(Individual i)
{
    of->evaluate(i);
    auto &ts_i = tgts->getData(i);
    ts_i.t += dt;
}

// SimulatedFunctionRuntimeObjectiveFunction
SimulatedFunctionRuntimeObjectiveFunction::SimulatedFunctionRuntimeObjectiveFunction(
    std::shared_ptr<ObjectiveFunction> of, std::function<double(Population &, Individual &)> dt) :
    of(of), dt(dt)
{
}
void SimulatedFunctionRuntimeObjectiveFunction::setPopulation(std::shared_ptr<Population> population)
{
    ObjectiveFunction::setPopulation(population);
    of->setPopulation(population);
}
void SimulatedFunctionRuntimeObjectiveFunction::registerData()
{
    of->registerData();

    Population &pop = *population;
    pop.registerData<TimeSpent>();
    tgts.emplace(pop.getDataContainer<TimeSpent>());
}
void SimulatedFunctionRuntimeObjectiveFunction::afterRegisterData()
{
    of->afterRegisterData();
}
void SimulatedFunctionRuntimeObjectiveFunction::evaluate(Individual i)
{
    of->evaluate(i);
    auto &ts_i = tgts->getData(i);
    ts_i.t += dt(*population, i);
}
SimulationTimeLogger::SimulationTimeLogger(std::shared_ptr<Simulator> simulator) : simulator(simulator)
{
}
void SimulationTimeLogger::header(IMapper &mapper)
{
    mapper << "simulation time (s)";
}
void SimulationTimeLogger::log(IMapper &mapper, const Individual &)
{
    mapper << std::to_string(simulator->now() + offset.value_or(0.0));
    offset.reset();
}
std::shared_ptr<SimulationTimeLogger> SimulationTimeLogger::shared(std::shared_ptr<Simulator> simulator)
{
    return std::make_shared<SimulationTimeLogger>(simulator);
}
void SimulationTimeLogger::set_one_time_offset(double offset)
{
    this->offset.emplace(offset);
}

void SimulatedEvaluationSpanLoggerWrapper::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    span_logger->setPopulation(population);
    sim_time_logger->setPopulation(population);
    wrapped->setPopulation(population);
}
void SimulatedEvaluationSpanLoggerWrapper::registerData()
{
    span_logger->registerData();
    sim_time_logger->registerData();
    wrapped->registerData();
}
void SimulatedEvaluationSpanLoggerWrapper::afterRegisterData()
{
    span_logger->afterRegisterData();
    sim_time_logger->afterRegisterData();
    wrapped->afterRegisterData();
}
void SimulatedEvaluationSpanLoggerWrapper::doCache()
{
    if (!cache.has_value())
    {
        cache.emplace(Cache{population->getDataContainer<TimeSpent>()});
    }
}
void SimulatedEvaluationSpanLoggerWrapper::evaluate(Individual i)
{
    doCache();
    // Start evaluation.
    span_logger->start_span(i.i, i, 1);

    wrapped->evaluate(i);

    // Compute end of evaluation
    Cache &ch = *this->cache;
    TimeSpent &ts = ch.tgts.getData(i);
    sim_time_logger->set_one_time_offset(ts.t);
    span_logger->end_span(i.i, i, 1);
}
SimulatedEvaluationSpanLoggerWrapper::SimulatedEvaluationSpanLoggerWrapper(
    std::shared_ptr<ObjectiveFunction> wrapped,
    std::shared_ptr<SpanLogger> span_logger,
    std::shared_ptr<SimulationTimeLogger> sim_time_logger) :
    wrapped(wrapped), span_logger(span_logger), sim_time_logger(sim_time_logger)
{
}
ObjectiveFunction *SimulatedFixedRuntimeObjectiveFunction::clone()
{

    auto of_clone = std::shared_ptr<ObjectiveFunction>(of->clone());
    return new SimulatedFixedRuntimeObjectiveFunction(of, dt);
}
ObjectiveFunction *SimulatedFunctionRuntimeObjectiveFunction::clone()
{
    auto of_clone = std::shared_ptr<ObjectiveFunction>(of->clone());
    return new SimulatedFunctionRuntimeObjectiveFunction(of, dt);
}
