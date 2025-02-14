//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
namespace py = pybind11;

#include <cereal/archives/binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

#include "cppassert.h"

// #ifndef BOOST_STACKTRACE_USE_BACKTRACE
// #define BOOST_STACKTRACE_USE_BACKTRACE
// #endif
// #include <boost/stacktrace.hpp>
// #define POPULATION_KEEP_LOGBOOK

// template <typename T> struct reset_data
// {
//     void operator()(T &)
//     {
//     }
// };

// Data should not be assumed to be serializable by default.
template <typename T, typename = std::void_t<>> struct is_data_serializable : std::false_type
{
};

template <typename T>

constexpr bool is_data_serializable_v = is_data_serializable<T>::value;
// There are many ways in which an EA can be implemented.
// One particular part is of interest: we want to associate data with solutions.
// One annoyance here however is that the data associated with solutions can change
// depending on the approach we are employing. Leading to multiple structures that
// encode a solution for each approach. This works, but leads to a lot of reimplementation
// even if the underlying data is exceedingly similar.
// As such this codebase utilizes some components from the Entity Component System framework commonly used in game
// development. See also https://austinmorlan.com/posts/entity_component_system/. Unlike Game Development, we do however
// assume all our entities are -- apart from their associated data
// -- identical. I.e., all the components (and hence data) associated are the sample, simplifing the implementation
// Additionally, we do not need the nice linear data access either: while some algorithms could benefit from this,
// it would require an extra mapping for each data kind, which would undo the benefits either way.
//
// Alternatively, you can view this as a dataframe kind of datastructure, where we want to associate data
// with rows, but not all operations require all columns.

// We'll have a pointer to the original creator in each individual, so we can keep track
// of whether an individual belongs to the same population.
class Population;

// Simple struct identifying an individual
// (struct to avoid confusion!)
struct Individual
{
    size_t i;
    Population *creator;

    bool operator==(const Individual &b) const
    {
        return this->i == b.i;
    }
};

template <> struct std::hash<Individual>
{
    std::size_t operator()(Individual const &s) const noexcept
    {
        return std::hash<size_t>{}(s.i);
    }
};

class SubIDataContainer;

// Interface to do non-type specific interactions.
class IDataContainer
{
  public:
    virtual ~IDataContainer() = default;
    virtual void resize(size_t size) = 0;
    virtual void copy(const Individual &from, const Individual &to) = 0;
    virtual void reset(const Individual &who) = 0;
    // virtual void reset(Individual) = 0;
    virtual std::unique_ptr<SubIDataContainer> subcontainer(const std::vector<Individual> & /* subset */)
    {
        // Default = no-op container, no data is archived (other than the existence?)
        return std::make_unique<SubIDataContainer>();
    }

    // Python specific
    virtual py::object getDataPython(Individual ii) = 0;
};
CEREAL_REGISTER_TYPE(SubIDataContainer)

class SubIDataContainer
{

  public:
    virtual ~SubIDataContainer() = default;
    /**
     * @brief Replace data for individuals with contained data.
     *
     * @param population The population (i.e. data table) for the provided individuals.
     * @param individuals THe individuals to replace the data for.
     */
    virtual void inject(Population & /* population */, std::vector<Individual> & /* individuals */)
    {
        // As default serialization is a no-op, so is injection.
    }

    template <class Archive> void serialize(Archive & /* ar */)
    {
        // Default serialization: do nothing. Serialization is opt-in!
    }
};

template <typename T> class SubDataContainer;

// Actual vector data container.
template <typename T> class DataContainer : public IDataContainer
{
  private:
    const static auto BLOCK_SIZE_SHIFT = 6;
    const static auto BLOCK_SIZE = 2 << BLOCK_SIZE_SHIFT;
    using BlockType = std::array<T, BLOCK_SIZE>;

    class BlockContainer
    {
      public:
        BlockContainer()
        {
            block = std::unique_ptr<BlockType>(new BlockType());
        }

        BlockContainer(BlockContainer &&o)
        {
            block.swap(o.block);
        }

        T &at(size_t idx)
        {
            return block->at(idx);
        }
        std::unique_ptr<BlockType> block;
    };

    std::vector<BlockContainer> data;

  public:
    DataContainer(size_t size)
    {
        resize(size);
    };

    T &getData(const Individual &ii)
    {
        size_t block_idx = ii.i >> BLOCK_SIZE_SHIFT;
        size_t remainder = ii.i - (block_idx << BLOCK_SIZE_SHIFT);
        return data.at(block_idx).at(remainder);
    };
    void resize(size_t size) override
    {
        // Deny downsizing.
        if (size <= data.size())
            return;

        size_t block_idx = size >> BLOCK_SIZE_SHIFT;
        size_t remainder = size - (block_idx << BLOCK_SIZE_SHIFT);
        if (remainder > 0)
            block_idx += 1;
        data.resize(block_idx);
    };
    void copy(const Individual &from, const Individual &to) override
    {
        getData(to) = getData(from);
    };
    void reset(const Individual &who) override
    {
        getData(who) = T();
    }
    // void reset(Individual ii) override
    // {
    //     reset_data<T>{}(data.at(ii.i));
    // }

    // Python specific
    py::object getDataPython(Individual ii) override
    {
        return py::cast(&getData(ii));
    }

    virtual std::unique_ptr<SubIDataContainer> subcontainer(const std::vector<Individual> &subset) override
    {
        // If type does not serialize, return no-op serializer (alternative: nullptr and filter?).
        if (!is_data_serializable_v<T>)
        {
            // std::cout << "Type " << typeid(T).name() << " is not serializable, and was hence defaulted to a no-op" <<
            // std::endl;
            return std::unique_ptr<SubIDataContainer>(new SubIDataContainer());
        }
        // std::cout << "Type " << typeid(T).name() << " is serializable, and is hence using SubDataContainer" <<
        // std::endl;

        std::vector<T> data(subset.size());
        for (size_t idx = 0; idx < subset.size(); ++idx)
        {
            data[idx] = getData(subset[idx]);
        }
        return std::unique_ptr<SubDataContainer<T>>(new SubDataContainer<T>(data));
    }
};

class SubpopulationData
{
    std::vector<std::unique_ptr<SubIDataContainer>> data;

    friend Population;
    SubpopulationData(std::vector<std::unique_ptr<SubIDataContainer>> &&data) : data(std::move(data))
    {
    }

  public:
    SubpopulationData() = default;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(CEREAL_NVP(data));
    }

    void inject(Population &population, std::vector<Individual> &individuals)
    {
        for (auto &container : data)
        {
            container->inject(population, individuals);
        }
    }
};

class IDataType
{
  public:
    virtual const char *typeName() = 0;
    virtual void registerDataType(Population &pop) = 0;
    virtual ~IDataType() = default;
};

// For quick access to a specific piece of data.
template <typename T> class TypedGetter
{
  public:
    TypedGetter(DataContainer<T> &data) : data(data){};
    T &getData(const Individual &ii)
    {
        return data.getData(ii);
    }

  private:
    DataContainer<T> &data;
};

/**
 * The population containing individuals.
 *
 * For those familiar with ECS nomenclarure, it is an component manager -- but we assume all entities share all
 * components. The components may differ however as different algorithms associate different data with solutions.
 */
class Population
{
  public:
    // Create a new empty population.
    Population();

    // Disallow copying implicitly, as this is a surefire way to end up with issues.
    explicit Population(const Population &rhs) = default;
    Population &operator=(const Population &rhs) = delete;

    // Register a struct associating data with a solution.
    template <typename T> void registerData()
    {
        // t_assert(capacity == 0, "Can only register data containers when no individuals have been created yet.");

        const char *typeName = typeid(T).name();

        if (dataContainers.find(typeName) != dataContainers.end())
            return;

        dataContainers[typeName] = std::make_shared<DataContainer<T>>(current_capacity);
    }
    // Check if a struct associating data is registered.
    template <typename T> bool isRegistered()
    {
        const char *typeName = typeid(T).name();
        return dataContainers.find(typeName) != dataContainers.end();
    }
    bool isRegistered(IDataType &type)
    {
        const char *typeName = type.typeName();
        return dataContainers.find(typeName) != dataContainers.end();
    }

    // A getter for repeatedly getting access to data of type T for varying
    // individuals.
    template <typename T> TypedGetter<T> getDataContainer()
    {
        const char *typeName = typeid(T).name();

        bool is_registered = dataContainers.find(typeName) != dataContainers.end();
        if (! is_registered) {
            std::cerr << "Type " << typeName << " was requested, but not registered." << std::endl;
        }
        t_assert(is_registered, "Datatype should be registered.");
        return TypedGetter(*std::static_pointer_cast<DataContainer<T>>(dataContainers.at(typeName)));
    }

    // Get the data of kind T belonging to a certain individual.
    template <typename T> T &getData(Individual ii)
    {
#ifdef POPULATION_VERIFY
        t_assert(ii.creator == this, "Can only get data for individuals belonging to this population.");
        t_assert(ii.i < in_use_pool_position.size(), "Can only get data for individuals that exist: out of bounds.");
        t_assert(in_use_pool_position[ii.i] != ((size_t)-1), "Can only get data for individuals that exist: dropped.");
#endif
        const char *typeName = typeid(T).name();

        t_assert(dataContainers.find(typeName) != dataContainers.end(), "Data should be registered before use.");

        auto container = std::static_pointer_cast<DataContainer<T>>(dataContainers.at(typeName));

        auto &data = container->getData(ii);

#ifdef POPULATION_KEEP_LOGBOOK
        access_log_per_individual[ii.i].logbook.push_back(
            {Accesses::Kind::GetData, std::nullopt, boost::stacktrace::stacktrace()});
#endif

        return data;
    }

    // Python specific!
    py::object getDataPython(IDataType &type, Individual ii)
    {
        t_assert(dataContainers.find(type.typeName()) != dataContainers.end(), "Data should be registered before use.");
        return dataContainers.at(type.typeName())->getDataPython(ii);
    }

    // Additionally, one may want to have some global information associated in a similar fashion
    template <typename T> void registerGlobalData(T data)
    {
        const char *typeName = typeid(T).name();

        bool was_not_already_registered = globalData.find(typeName) == globalData.end();
        if (! was_not_already_registered) {
            std::cerr << "Type " << typeName << " was already registered." << std::endl; 
        }
        t_assert(was_not_already_registered, "A datatype should not be registered more than once.");

        globalData[typeName] = std::move(std::make_shared<T>(std::move(data)));
    }
    // Check if a certain kind of global data is present.
    template <typename T> bool isGlobalRegistered()
    {
        const char *typeName = typeid(T).name();
        return globalData.find(typeName) != globalData.end();
    }
    // Get the data of kind T belonging to a certain individual.
    template <typename T> std::shared_ptr<T> getGlobalData()
    {
        const char *typeName = typeid(T).name();

        if (globalData.find(typeName) == globalData.end())
        {
            std::cerr << "Failed to find \"" << typeName << "\"." << std::endl;
        }
        t_assert(globalData.find(typeName) != globalData.end(), "Data not registered.");

        return std::static_pointer_cast<T>(globalData.at(typeName));
    }

    // Returns a currently unused individual index.
    //
    // WARNING: Calling this may invalidate current references to any data held.
    //          ALWAYS re-obtain any references after calling this method.
    Individual newIndividual();
    void newIndividuals(std::vector<Individual> &to_init);
    // Stop using an individual index. i.e. return the memory used.
    void dropIndividual(Individual ii);
    // Reset an individual (i.e. drop without invalidating)
    void resetIndividual(Individual ii);
    // Copy a solution from a to b, preserving all associated data.
    void copyIndividual(Individual from, Individual to);

    SubpopulationData getSubpopulationData(std::vector<Individual> &individuals)
    {
        std::vector<std::unique_ptr<SubIDataContainer>> data;

        for (auto &[c, a] : dataContainers)
        {
            data.push_back(a->subcontainer(individuals));
        }

        return SubpopulationData(std::move(data));
    }

#ifdef POPULATION_KEEP_LOGBOOK
    struct Accesses
    {
        enum Kind
        {
            Created = 0,
            Reused = 1,
            Dropped = 2,
            GetData = 3,
            CopyFrom = 4,
            CopyTo = 5,
        };

        struct Copy
        {
            Individual from;
            Individual to;
        };
        using What = std::optional<Copy>;

        std::vector<std::tuple<Kind, What, boost::stacktrace::stacktrace>> logbook;
    };
    std::vector<Accesses> access_log_per_individual;
#endif

    size_t size()
    {
        return current_size;
    }

    size_t active()
    {
        return in_use_pool.size();
    }

    size_t capacity()
    {
        return current_capacity;
    }

  private:
    // Mapping from a type to the data container.
    std::unordered_map<const char *, std::shared_ptr<IDataContainer>> dataContainers;
    // Mapping from a type to global data.
    std::unordered_map<const char *, std::shared_ptr<void>> globalData;
    // Pool of items that are in use.
    std::vector<size_t> in_use_pool;
    // Mapping for each index where there are located in the pool.
    // Maximum size_t (or alternatively, -1) is used to indicate absence.
    std::vector<size_t> in_use_pool_position;

    // DEBUG
    // std::vector<boost::stacktrace::stacktrace> dropStacktraces;

    std::vector<size_t> reuse_pool;
    size_t current_capacity;
    size_t current_size;

    // Resizing the population & the associated data for each solution.
    void resize(size_t ii);

    // Get the next individual in sequence.
    Individual nextNewIndividual();
};

template <typename T> class SubDataContainer : public SubIDataContainer
{
    std::vector<T> data;

    SubDataContainer() = default;
    SubDataContainer(std::vector<T> data) : data(data){};

    friend cereal::access;
    friend DataContainer<T>;

  public:
    template <class Archive> void serialize(Archive &ar)
    {
        ar(cereal::base_class<SubIDataContainer>(this));
        ar(CEREAL_NVP(data));
    }

    virtual void inject(Population &population, std::vector<Individual> &individuals) override
    {
        t_assert(individuals.size() == data.size(), "#individuals and amount of data should be equal");
        auto tgt = population.getDataContainer<T>();
        for (size_t idx = 0; idx < individuals.size(); ++idx)
        {
            tgt.getData(individuals[idx]) = data[idx];
        }
    }
    // Note! Ensure you register CEREAL_REGISTER_TYPE(SubDataContainer<T>) for all datatypes used.
};

template <typename T> class DataType : public IDataType
{
    const char *typeNameStr;

  public:
    DataType(const char *typeNameStr) : typeNameStr(typeNameStr)
    {
    }

    const char *typeName() override
    {
        return typeNameStr;
    }

    void registerDataType(Population &pop) override
    {
        pop.registerData<T>();
    }
};

/**
 * This is a class representing a component that registers data.
 *
 * First, we have the parts of the codebase that use and require particular data to be present.
 * As part of the startup first register_data is called on all users. Then after_register_data is called
 * on all users, to get access to the relevant data locations.
 *
 * Note that a user should only register the values it provides, not the values it needs. This ensures
 * that not having a value provided for will yield the corresponding error.
 **/
class IDataUser
{
  public:
    virtual ~IDataUser() = default;
    virtual void setPopulation(std::shared_ptr<Population> population);
    virtual void registerData(){};
    virtual void afterRegisterData(){};

    std::shared_ptr<Population> population;
};

// Common Properties & associated methods
// A certain subset of data associated with solutions are quite common, and are described down below.

/**
 * The objective value of a solution.
 *
 * - Lower is better is assumed within this codebase. This may require objective functions to be inverted.
 */
struct Objective
{
  public:
    std::vector<double> objectives;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(CEREAL_NVP(objectives));
    }
};
template <> struct is_data_serializable<Objective> : std::true_type
{
};
CEREAL_REGISTER_TYPE(SubDataContainer<Objective>)
const std::shared_ptr<IDataType> OBJECTIVE = std::make_shared<DataType<Objective>>(typeid(Objective).name());

/**
 * The constraint value of a solution.
 *
 * - Lower is better is assumed in this codebase.
 * - Solutions with a positive constraint value are assumed to violate constraints.
 * - Negative values should be deemed as good as a value of zero.
 */
struct Constraint
{
    double constraint;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(CEREAL_NVP(constraint));
    }
};
template <> struct is_data_serializable<Constraint> : std::true_type
{
};
CEREAL_REGISTER_TYPE(SubDataContainer<Constraint>)
const std::shared_ptr<IDataType> CONSTRAINT = std::make_shared<DataType<Constraint>>(typeid(Constraint).name());

/**
 * A categorical genotype for encoding discrete decisions.
 */
struct GenotypeCategorical
{
    std::vector<char> genotype;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(CEREAL_NVP(genotype));
    }
};
template <> struct is_data_serializable<GenotypeCategorical> : std::true_type
{
};
CEREAL_REGISTER_TYPE(SubDataContainer<GenotypeCategorical>)
const std::shared_ptr<IDataType> GENOTYPECATEGORICAL =
    std::make_shared<DataType<GenotypeCategorical>>(typeid(GenotypeCategorical).name());

/**
 * The length of the categorical genotype
 */
struct GenotypeCategoricalData
{
    size_t l;
    std::vector<char> alphabet_size;
};

/**
 * A continuous genotype for encoding continuous numbers.
 */
struct GenotypeContinuous
{
    std::vector<double> genotype;

    template <class Archive> void serialize(Archive &ar)
    {
        ar(CEREAL_NVP(genotype));
    }
};
template <> struct is_data_serializable<GenotypeContinuous> : std::true_type
{
};
CEREAL_REGISTER_TYPE(SubDataContainer<GenotypeContinuous>)

const std::shared_ptr<IDataType> GENOTYPECONTINUOUS =
    std::make_shared<DataType<GenotypeContinuous>>(typeid(GenotypeContinuous).name());

/**
 * The length of the continuous genotype
 */
struct GenotypeContinuousLength
{
    size_t l;
};

//
class ObjectiveFunction : public IDataUser
{
  public:
    virtual ~ObjectiveFunction() = default;
    virtual void evaluate(Individual i) = 0;
    virtual ObjectiveFunction *clone() = 0;
};

// Global data
class GObjectiveFunction
{
  public:
    GObjectiveFunction(ObjectiveFunction *of) : of(of){};
    ObjectiveFunction *of;
};

// Common utilities

// Exception thrown by an objective function indicating that no further
// objective evaluations should take place.
class run_complete : public std::exception
{
  public:
    const char *what() const throw()
    {
        return "arbitrary run complete";
    }
};

class evaluation_limit_reached : public run_complete
{
  public:
    const char *what() const throw()
    {
        return "evaluation limit reached";
    }
};

class time_limit_reached : public run_complete
{
  public:
    const char *what() const throw()
    {
        return "time limit reached";
    }
};

class vtr_reached : public run_complete
{
  public:
    const char *what() const throw()
    {
        return "all values-to-reach reached";
    }
};

class stop_approach : public std::exception
{
  public:
    const char *what() const throw()
    {
        return "generic approach stop exception";
    }
};


/**
 * Random number generator
 */
class Rng
{
  public:
    Rng(std::optional<size_t> seed)
    {
        if (seed.has_value())
            rng.seed(seed.value());
    }
    std::mt19937 rng;
};

/**
 * There are multiple ways to initialize a solution, this is an interface for doing so.
 */
class ISolutionInitializer : public IDataUser
{
  public:
    void afterRegisterData() override;
    virtual void initialize(std::vector<Individual> &iis) = 0;
};

class IPerformanceCriterion : public IDataUser
{
  public:
    virtual ~IPerformanceCriterion() = default;
    // Compare two individuals return a value signifying which is better (if any)
    // 0 - inconclusive - neither is better or equal to the other
    // 1 - a is better than b
    // 2 - b is better than a
    // 3 - a and b are equal
    virtual short compare(Individual &a, Individual &b) = 0;
};

/**
 * Base class for approaches that perform Generational-style evaluations.
 */
class GenerationalApproach : public IDataUser
{
  public:
    virtual ~GenerationalApproach() = default;
    virtual void step() = 0;
    virtual std::vector<Individual> &getSolutionPopulation() = 0;
    virtual bool terminated();
};

class GenerationalApproachComparator : public IDataUser
{
  public:
    virtual void clear();
    virtual short compare(std::shared_ptr<GenerationalApproach>, std::shared_ptr<GenerationalApproach>);
};

class AverageFitnessComparator : public GenerationalApproachComparator
{
  public:
    AverageFitnessComparator(size_t index = 0) : index(index)
    {
    }

    short compare(std::shared_ptr<GenerationalApproach> a, std::shared_ptr<GenerationalApproach> b) override;

    void clear() override;

  private:
    size_t index;
    double compute_average_fitness(std::shared_ptr<GenerationalApproach> &approach);
    std::unordered_map<GenerationalApproach *, double> cache;
};

// utilities related to the evaluation function

/**
 * An evaluation count & time limiter
 *
 * Ensures that the underlying objective function is not called more
 * often than a certain amount of times, or after a certain time budget.
 * Throws an exception once either of the limits are hit.
 */
class Limiter : public ObjectiveFunction
{
  private:
    std::shared_ptr<ObjectiveFunction> wrapping;
    std::optional<long long> evaluation_limit;
    std::optional<std::chrono::duration<double>> time_limit;
    long long num_evaluations = 0;
    std::chrono::time_point<std::chrono::system_clock> start;

    std::chrono::duration<double> current_duration()
    {
        return std::chrono::system_clock::now() - start;
    }

  public:
    Limiter(std::shared_ptr<ObjectiveFunction> wrapping,
            std::optional<long long> evaluation_limit = std::nullopt,
            std::optional<std::chrono::duration<double>> time_limit = std::nullopt);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    ObjectiveFunction *clone() override;

    void restart();

    void evaluate(Individual i) override;

    long long get_time_spent_ms();

    long long get_num_evaluations();

    
    std::shared_ptr<ObjectiveFunction> get_wrapping()
    {
        return wrapping;
    }

    std::optional<long long> get_evaluation_limit()
    {
        return evaluation_limit;
    }

    std::optional<std::chrono::duration<double>> get_time_limit()
    {
        return time_limit;
    }
};

/**
 * A Value-to-Reach detector
 */
class ObjectiveValuesToReachDetector : public ObjectiveFunction
{
  private:
    std::shared_ptr<ObjectiveFunction> wrapping;
    std::vector<std::vector<double>> vtrs;

    struct Cache
    {
        TypedGetter<Objective> tg_o;
    };
    std::optional<Cache> cache;

    void doCache();

    bool checkPresent(Individual i);

  public:
    ObjectiveValuesToReachDetector(std::shared_ptr<ObjectiveFunction> wrapping, std::vector<std::vector<double>> vtrs);

    void evaluate(Individual i) override;

    ObjectiveFunction *clone() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    
    std::shared_ptr<ObjectiveFunction> get_wrapping()
    {
        return wrapping;
    }
    std::vector<std::vector<double>> get_vtrs()
    {
        return vtrs;
    }
};

/**
 * @brief Keep track of the best solution according to a criterion
 *
 * Note: in general it is better to use the archive API instead,
 * which can maintain sets of best solutions larger than one.
 */
class ElitistMonitor : public ObjectiveFunction
{
    std::optional<Individual> elitist;
    std::shared_ptr<ObjectiveFunction> wrapping;
    std::shared_ptr<IPerformanceCriterion> criterion;
    bool is_real_solution = false;

  public:
    ElitistMonitor(std::shared_ptr<ObjectiveFunction> wrapping, std::shared_ptr<IPerformanceCriterion> criterion);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    ObjectiveFunction *clone() override;

    void evaluate(Individual i) override;

    virtual void onImproved(){};

    std::optional<Individual> getElitist()
    {
        return elitist;
    };

    
    std::shared_ptr<ObjectiveFunction> get_wrapping()
    {
        return wrapping;
    }
    std::shared_ptr<IPerformanceCriterion> get_criterion()
    {
        return criterion;
    }
};

struct Subfunctions
{
    std::vector<std::vector<size_t>> subfunctions;
};

/**
 * @brief An individual that is dropped from the population when it goes out of scope.
 */
class DroppingIndividual
{
    std::optional<Individual> i;

  public:
    DroppingIndividual() = default;
    DroppingIndividual(const Individual &i) = delete;
    DroppingIndividual(DroppingIndividual &&i);
    DroppingIndividual(Individual &&i);
    void operator=(const Individual &i) = delete;
    void operator=(const Individual &&i);
    ~DroppingIndividual();

    operator Individual *();
    // operator Individual &();
};