//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <queue>
#include <tuple>

#include "base.hpp"
#include "logging.hpp"

// Struct for tracking solution specific time spent in.
struct TimeSpent
{
    double t;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(CEREAL_NVP(t));
    }
};
template <> struct is_data_serializable<TimeSpent> : std::true_type
{
};
CEREAL_REGISTER_TYPE(SubDataContainer<TimeSpent>)
const std::shared_ptr<IDataType> TIMESPENT = std::make_shared<DataType<TimeSpent>>(typeid(TimeSpent).name());

// -- Simulation & Timing Utilities --

/**
 * @brief A stopwatch for measuring time spent between two points.
 */
struct Stopwatch
{
    std::chrono::time_point<std::chrono::system_clock> starting_point;

    Stopwatch() : starting_point(std::chrono::system_clock::now()){};
    double measure()
    {
        auto diff = starting_point - std::chrono::system_clock::now();
        return std::chrono::duration<double>(diff).count();
    }
};

class IResumableSimulated;

/**
 * @brief A general interface for a simulator, used to let resumables emit events.
 */
class ISimulator
{
  public:
    virtual ~ISimulator() = default;
    virtual void insert_event(std::unique_ptr<IResumableSimulated> e,
                              double at,
                              std::optional<std::string> descriptor) = 0;
};

/**
 * @brief A piece of code that can be simulated.
 */
class IResumableSimulated
{
  public:
    virtual ~IResumableSimulated() = default;
    virtual void resume(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self) = 0;
};

class FunctionalSimResumable : public IResumableSimulated
{
  private:
    std::function<void(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self)> resumable;

  public:
    FunctionalSimResumable(
        std::function<void(ISimulator &, double, std::unique_ptr<IResumableSimulated> &)> resumable) :
        resumable(resumable)
    {
    }

    void resume(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self) override
    {
        resumable(simulator, at, self);
    }
};

struct SimEvent
{
    double t;
    size_t ord;
    mutable std::unique_ptr<IResumableSimulated> ptr;

    // This one cannot be copied due to the unique_ptr.
    // Force the use of move operators by defining them explicitly.
    // Cost of figuring this one out: 2 hours and a bit.
    SimEvent(SimEvent &&) = default;
    SimEvent &operator=(SimEvent &&) = default;

    bool operator<(const SimEvent &o) const;
};

class Simulator : public ISimulator
{
  public:
    std::optional<double> time_limit;
    size_t ordinal = 0;
    double simulated_time = 0.0;

    Simulator();
    Simulator(std::optional<double> time_limit);

    std::priority_queue<SimEvent> event_queue;

    void insert_event(std::unique_ptr<IResumableSimulated> e,
                      double at,
                      std::optional<std::string> descriptor) override;

    double now();

    void serial_simulation(double t);

    void serial_simulation(Stopwatch &sw);

    void step();

    virtual void event_new(size_t ord, double t, std::optional<std::string> descriptor);
    virtual void event_hit(size_t ord, double t, size_t ord_before, size_t ord_after);
    virtual void sim_time_limit_reached(size_t ord, double t);

    template <typename T> void simulate_until(T &&predicate)
    {
        while ((!predicate()) && event_queue.size() > 0)
        {
            step();
        }
    }

    void simulate_until_end();
};

class WritingSimulator : public Simulator
{
  private:
    std::filesystem::path out_path;

  public:
    WritingSimulator(std::filesystem::path out_path, std::optional<double> time_limit);

    void event_new(size_t ord, double t, std::optional<std::string> descriptor) override;
    void event_hit(size_t ord, double t, size_t ord_before, size_t ord_after) override;
    void sim_time_limit_reached(size_t ord, double t) override;
};

struct SimulatorParameters
{
    size_t num_workers;
    bool measure_overhead;

    SimulatorParameters(size_t num_workers, std::optional<double> time_limit, bool measure_overhead = true);
    SimulatorParameters(std::shared_ptr<Simulator> simulator, size_t num_workers, bool measure_overhead = true);

    std::shared_ptr<Simulator> simulator;

    size_t num_workers_busy = 0;
    // Queue of resumables that do not have an assigned processor.
    std::queue<std::tuple<double, std::unique_ptr<IResumableSimulated>, std::optional<std::string>>> processing_queue;
    // Queue of resumables that do not have an assigned processor, but that may be no-ops that add themselves again.
    // Using the previous queue would lead the simulator to get stuck due to a single converged neighborhood and
    // leftover resources.
    std::queue<std::tuple<double, std::unique_ptr<IResumableSimulated>, std::optional<std::string>>>
        processing_queue_next_t;
    double last_t_addition = 0.0;

    SimulatorParameters(SimulatorParameters &&) = default;
    SimulatorParameters(const SimulatorParameters &) = delete;
};

// -- Utilities --

/**
 * @brief A wrapper for an objective function that simulates runtime. Runtime is set to a fixed value.
 */
class SimulatedFixedRuntimeObjectiveFunction : public ObjectiveFunction
{
  private:
    // Objective function we are wrapping
    std::shared_ptr<ObjectiveFunction> of;
    // The fixed runtime of evaluating a solution.
    double dt;
    // Getter for time spent - speeds up lookups.
    std::optional<TypedGetter<TimeSpent>> tgts;

  public:
    SimulatedFixedRuntimeObjectiveFunction(std::shared_ptr<ObjectiveFunction> of, double dt);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;
};

/**
 * @brief A wrapper for an objective function that simulates runtime. Runtime is provided by an arbitrary function.
 */
class SimulatedFunctionRuntimeObjectiveFunction : public ObjectiveFunction
{
  private:
    // Objective function we are wrapping
    std::shared_ptr<ObjectiveFunction> of;
    // A function that provides the runtime
    std::function<double(Population &, Individual &)> dt;
    // Getter for time spent - speeds up lookups.
    std::optional<TypedGetter<TimeSpent>> tgts;

  public:
    SimulatedFunctionRuntimeObjectiveFunction(std::shared_ptr<ObjectiveFunction> of,
                                              std::function<double(Population &, Individual &)> dt);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;
};

class SimulationTimeLogger : public ItemLogger
{
  private:
    std::shared_ptr<Simulator> simulator;
    std::optional<double> offset;

  public:
    SimulationTimeLogger(std::shared_ptr<Simulator> simulator);
    static std::shared_ptr<SimulationTimeLogger> shared(std::shared_ptr<Simulator> simulator);

    void set_one_time_offset(double offset);

    virtual void header(IMapper &mapper);
    virtual void log(IMapper &mapper, const Individual &);
};

class SpanItemLogger : public ItemLogger
{
  private:
    bool kind = false;
    size_t id = 0;
    size_t tid = 0;

  public:
    SpanItemLogger() = default;

    void convey_item(bool kind, size_t id, size_t tid)
    {
        this->kind = kind;
        this->id = id;
        this->tid = tid;
    }
    virtual void header(IMapper &mapper)
    {
        mapper << "span_kind"
               << "span_id"
               << "span_tid";
    }
    virtual void log(IMapper &mapper, const Individual &)
    {
        mapper << std::to_string(kind) << std::to_string(id) << std::to_string(tid);
    }
};

class SpanLogger : public IDataUser
{
    // Currently Reserved/Used TIDs
    // 0 - Population replacements
    // 1 - Function Evaluations

  private:
    std::shared_ptr<BaseLogger> logger;
    std::shared_ptr<SpanItemLogger> pkl;

  public:
    SpanLogger(std::shared_ptr<BaseLogger> logger, std::shared_ptr<SpanItemLogger> pkl = NULL) :
        logger(logger), pkl(pkl)
    {
    }

    void start_span(size_t id, Individual &i, size_t tid)
    {
        if (this->pkl != NULL)
        {
            this->pkl->convey_item(false, id, tid);
        }
        logger->log(i);
    }
    void end_span(size_t id, Individual &i, size_t tid)
    {
        if (this->pkl != NULL)
        {
            this->pkl->convey_item(true, id, tid);
        }
        logger->log(i);
    }

    void setPopulation(std::shared_ptr<Population> population) override
    {
        this->population = population;
        logger->setPopulation(population);
        pkl->setPopulation(population);
    }
    void registerData() override
    {
        logger->registerData();
        pkl->registerData();
    }
    void afterRegisterData() override
    {
        logger->afterRegisterData();
        pkl->afterRegisterData();
    }
};

class SimulatedEvaluationSpanLoggerWrapper : public ObjectiveFunction
{
  private:
    std::shared_ptr<ObjectiveFunction> wrapped;
    std::shared_ptr<SpanLogger> span_logger;
    std::shared_ptr<SimulationTimeLogger> sim_time_logger;

    struct Cache
    {
        TypedGetter<TimeSpent> tgts;
    };
    std::optional<Cache> cache;

    void doCache();

  public:
    SimulatedEvaluationSpanLoggerWrapper(std::shared_ptr<ObjectiveFunction> wrapped,
                                         std::shared_ptr<SpanLogger> span_logger,
                                         std::shared_ptr<SimulationTimeLogger> sim_time_logger);

    void evaluate(Individual i) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    ObjectiveFunction *clone() override
    {
        // auto wrapped_clone = std::shared_ptr<ObjectiveFunction>(wrapped->clone());
        // SpanLogger and SimulationTimeLogger are not clonable (yet).
        // Currently, the simulator has no need for an implementation, and as such
        // this is unimplemented (for now).
        return nullptr;
    }
};

class TimeSpentItemLogger : public ItemLogger
{
  private:
    struct Cache
    {
        TypedGetter<TimeSpent> tgts;
    };
    std::optional<Cache> cache;
    void doCache()
    {
        if (!cache.has_value())
        {
            t_assert(population != NULL, "population should not be NULL");
            cache.emplace(Cache{population->getDataContainer<TimeSpent>()});
        }
    }

  public:
    virtual void header(IMapper &mapper)
    {
        mapper << "time_spent";
    }
    virtual void log(IMapper &mapper, const Individual &ii)
    {
        doCache();
        Cache &ch = *cache;
        mapper << std::to_string(ch.tgts.getData(ii).t);
    }
};