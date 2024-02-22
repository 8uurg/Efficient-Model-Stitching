#include "base.hpp"
#include <algorithm>
#include <optional>

// Columnar Population

Population::Population()
{
    current_capacity = 0;
    current_size = 0;
}

void Population::resize(size_t size)
{
    // Resize data containers
    for (auto [_, container] : dataContainers)
    {
        container.get()->resize(size);
    }

#ifdef POPULATION_KEEP_LOGBOOK
    access_log_per_individual.resize(size);
#endif
    // DEBUG
    // dropStacktraces.resize(size);

    // Update pool position
    in_use_pool_position.resize(size);
    std::fill(in_use_pool_position.begin() + static_cast<long>(this->current_capacity), in_use_pool_position.end(), -1);
    // Capacity is the internal available size.
    this->current_capacity = size;
}

Individual Population::newIndividual()
{
    return nextNewIndividual();
}

void Population::newIndividuals(std::vector<Individual> &to_init)
{
    for (size_t i = 0; i < to_init.size(); ++i)
    {
        to_init[i] = nextNewIndividual();
    }
}

Individual Population::nextNewIndividual()
{
    // First attempt to use the reuse pool
    if (!reuse_pool.empty())
    {
        size_t ii = reuse_pool.back();
        reuse_pool.pop_back();

        // Keep track of the indices in use.
        in_use_pool_position[ii] = in_use_pool.size();
        in_use_pool.push_back(ii);

#ifdef POPULATION_KEEP_LOGBOOK
        access_log_per_individual[ii].logbook.push_back(
            {Accesses::Kind::Reused, std::nullopt, boost::stacktrace::stacktrace()});
#endif

        return Individual{ii, this};
    }
    // Check if no funky business is afoot.
    t_assert(current_capacity >= current_size, "The population should never be smaller than the capacity.");
    // If that fails, check if there is enough capacity
    if (current_capacity == current_size)
    {
        // We have hit max capacity, enlarge!
        size_t new_capacity = current_capacity * 2;
        // (If zero, we need a value to start with)
        if (new_capacity == 0)
            new_capacity = 16;
        this->resize(new_capacity);
        // Aforementioned if statement should no longer be true.
        t_assert(current_capacity != current_size, "Resizing should have provided more capacity.");
    }
    // Next index available should work!
    size_t ii = current_size;
    current_size += 1;
    // Keep track of the indices in use.
    in_use_pool_position[ii] = in_use_pool.size();
    in_use_pool.push_back(ii);

#ifdef POPULATION_KEEP_LOGBOOK
    access_log_per_individual[ii].logbook.push_back(
        {Accesses::Kind::Created, std::nullopt, boost::stacktrace::stacktrace()});
#endif

    return Individual{ii, this};
}

void Population::copyIndividual(Individual from, Individual to)
{
#ifdef POPULATION_VERIFY
    t_assert(from.creator == this, "First individual was created by a different population.");
    t_assert(to.creator == this, "Second individual was created by a different population.");
#endif
    for (auto [_, container] : dataContainers)
    {
        container.get()->copy(from, to);
    }

#ifdef POPULATION_KEEP_LOGBOOK
    access_log_per_individual[from.i].logbook.push_back(
        {Accesses::Kind::CopyFrom, Accesses::Copy{from, to}, boost::stacktrace::stacktrace()});
    access_log_per_individual[to.i].logbook.push_back(
        {Accesses::Kind::CopyTo, Accesses::Copy{from, to}, boost::stacktrace::stacktrace()});
#endif
}

void Population::dropIndividual(Individual ii)
{
#ifdef POPULATION_VERIFY
    t_assert(ii.creator == this, "Individual to be dropped was created by another population.")
#endif
        t_assert(ii.i < in_use_pool_position.size(),
                 "Individual to be dropped should exist, is out of bounds instead.");
    t_assert(in_use_pool_position.at(ii.i) != ((size_t)-1),
             "Individual to be dropped should exist, has been dropped before instead.");

    // DEBUG
    // dropStacktraces[ii.i] = boost::stacktrace::stacktrace();

    for (auto [_, container] : dataContainers)
    {
        container.get()->reset(ii);
    }

#ifdef POPULATION_KEEP_LOGBOOK
    access_log_per_individual[ii.i].logbook.push_back(
        {Accesses::Kind::Dropped, std::nullopt, boost::stacktrace::stacktrace()});
#endif

    // Add to reuse pool
    reuse_pool.push_back(ii.i);
    // And remove from the in-use pool
    // - Position to replace
    size_t pos_pool = in_use_pool_position[ii.i];
    // - We'll replace it with the last item in the pool (and update bookkeeping accordingly)
    size_t in_pool_last = in_use_pool.back();
    in_use_pool[pos_pool] = in_use_pool.back();
    in_use_pool_position[in_pool_last] = pos_pool;
    // - Invalidate position of ii.i -- as in_pool_last may be ii.i
    //   this is done after the previous step, overwriting if equal.
    in_use_pool_position[ii.i] = (size_t)-1;
    // - Finally, remove the last element of the in_use_pool
    in_use_pool.pop_back();
}

void Population::resetIndividual(Individual ii)
{
#ifdef POPULATION_VERIFY
    t_assert(ii.creator == this, "Individual to be reset was created by another population.")
#endif
    t_assert(ii.i < in_use_pool_position.size(),
                "Individual to be reset should exist, is out of bounds instead.");
    t_assert(in_use_pool_position.at(ii.i) != ((size_t)-1),
             "Individual to be reset should exist, has been dropped before instead.");

    // DEBUG
    // dropStacktraces[ii.i] = boost::stacktrace::stacktrace();

    for (auto [_, container] : dataContainers)
    {
        container.get()->reset(ii);
    }
}

void IDataUser::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
}

void ISolutionInitializer::afterRegisterData()
{
    t_assert(population != NULL, "Population should be set.");
    Population &pop = (*population);
    t_assert(pop.isGlobalRegistered<Rng>(), "Rng should be present.");
}

bool GenerationalApproach::terminated()
{
    return false;
}

void GenerationalApproachComparator::clear()
{
}
short GenerationalApproachComparator::compare(std::shared_ptr<GenerationalApproach>,
                                              std::shared_ptr<GenerationalApproach>)
{
    return 0;
}

short AverageFitnessComparator::compare(std::shared_ptr<GenerationalApproach> a,
                                        std::shared_ptr<GenerationalApproach> b)
{
    double a_avg_fitness = compute_average_fitness(a);
    double b_avg_fitness = compute_average_fitness(b);

    short result = 0;
    if (a_avg_fitness <= b_avg_fitness)
        result |= 1;
    if (b_avg_fitness <= a_avg_fitness)
        result |= 2;
    return result;
}
void AverageFitnessComparator::clear()
{
    cache.clear();
}
double AverageFitnessComparator::compute_average_fitness(std::shared_ptr<GenerationalApproach> &approach)
{
    // Use cache if entry exists
    auto cache_hit = cache.find(approach.get());
    if (cache_hit != cache.end())
        return cache_hit->second;

    Population &population = *this->population;
    // Otherwise, compute
    double sum_objective = 0.0;
    size_t count = 0;
    auto &pop = approach->getSolutionPopulation();
    for (Individual &ii : pop)
    {
        Objective &o = population.getData<Objective>(ii);
        sum_objective += o.objectives[index];
        ++count;
    }
    double avg_objective = sum_objective / static_cast<double>(count);
    cache[approach.get()] = avg_objective;
    return avg_objective;
}

// Limiter
Limiter::Limiter(std::shared_ptr<ObjectiveFunction> wrapping,
                 std::optional<long long> evaluation_limit,
                 std::optional<std::chrono::duration<double>> time_limit) :
    wrapping(wrapping), evaluation_limit(evaluation_limit), time_limit(time_limit)
{
    // Start timer at construction... Might be a tad early!
    start = std::chrono::system_clock::now();
}
void Limiter::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    wrapping->setPopulation(population);
}
void Limiter::registerData()
{
    wrapping->registerData();
}
void Limiter::afterRegisterData()
{
    wrapping->afterRegisterData();
}
void Limiter::restart()
{
    start = std::chrono::system_clock::now();
}
void Limiter::evaluate(Individual i)
{
    if (time_limit.has_value() && *time_limit < current_duration())
        throw time_limit_reached();
    if (evaluation_limit.has_value() && *evaluation_limit <= num_evaluations)
        throw evaluation_limit_reached();
    wrapping->evaluate(i);
    ++num_evaluations;
}
long long Limiter::get_time_spent_ms()
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(current_duration()).count();
}
long long Limiter::get_num_evaluations()
{
    return num_evaluations;
}
ObjectiveFunction *Limiter::clone()
{
    auto wrapping_clone = std::shared_ptr<ObjectiveFunction>(wrapping->clone());
    return new Limiter(wrapping_clone, evaluation_limit, time_limit);
}

// ElitistMonitor
ElitistMonitor::ElitistMonitor(std::shared_ptr<ObjectiveFunction> wrapping,
                               std::shared_ptr<IPerformanceCriterion> criterion) :
    wrapping(wrapping), criterion(criterion)
{
}
void ElitistMonitor::setPopulation(std::shared_ptr<Population> population)
{
    ObjectiveFunction::setPopulation(population);
    this->wrapping->setPopulation(population);
    this->criterion->setPopulation(population);
}
void ElitistMonitor::registerData()
{
    wrapping->registerData();
    criterion->registerData();
}
void ElitistMonitor::afterRegisterData()
{
    wrapping->afterRegisterData();
    criterion->afterRegisterData();

    Population &population = *this->population;
    // Create elitist
    elitist = population.newIndividual();
}
void ElitistMonitor::evaluate(Individual i)
{
    wrapping->evaluate(i);

    t_assert(elitist.has_value(), "afterRegisterData must have been called before the first evaluation.");

    if (!is_real_solution || criterion->compare(i, *elitist) == 1)
    {
        Population &population = *this->population;
        population.copyIndividual(i, *elitist);
        onImproved();
        is_real_solution = true;
    }
}
ObjectiveFunction *ElitistMonitor::clone()
{
    // auto wrapping_clone = std::shared_ptr<ObjectiveFunction>(wrapping->clone());
    //! IPerformanceCriterion is not clonable (yet!)
    // TODO: Make IPerformanceCriterion clonable.
    // TODO: Fix this cloning method.
    return nullptr;
}

#ifdef POPULATION_KEEP_LOGBOOK
void printLogBookTrace(Population &population, Individual i, size_t logidx)
{
    std::cout << "Stacktrace:\n";
    std::cout << std::get<2>(population.access_log_per_individual[i.i].logbook[logidx]) << std::endl;
}
#endif

ObjectiveValuesToReachDetector::ObjectiveValuesToReachDetector(std::shared_ptr<ObjectiveFunction> wrapping,
                                                               std::vector<std::vector<double>> vtrs) :
    wrapping(wrapping), vtrs(vtrs)
{
}
void ObjectiveValuesToReachDetector::evaluate(Individual i)
{
    wrapping->evaluate(i);
    if (checkPresent(i) && vtrs.size() == 0)
    {
        throw vtr_reached();
    }
}
void ObjectiveValuesToReachDetector::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    wrapping->setPopulation(population);
}
void ObjectiveValuesToReachDetector::registerData()
{
    wrapping->registerData();
}
void ObjectiveValuesToReachDetector::afterRegisterData()
{
    wrapping->afterRegisterData();
}
bool ObjectiveValuesToReachDetector::checkPresent(Individual i)
{
    doCache();
    bool any_removed = false;
    Objective &o = cache->tg_o.getData(i);
    auto new_end = std::remove_if(vtrs.begin(), vtrs.end(), [o, &any_removed](std::vector<double> &vtr) {
        // Note: duplicates mean that you need to hit it multiple times.
        //       You may want to cache the solutions you have already evaluated beforehand though:
        //       Evaluating the same solution multiple times will each count as a hit.
        if (vtr.size() != o.objectives.size() || any_removed)
            return false;

        any_removed = std::equal(vtr.begin(), vtr.end(), o.objectives.begin());
        return any_removed;
    });
    vtrs.erase(new_end, vtrs.end());
    return any_removed;
}

void ObjectiveValuesToReachDetector::doCache()
{
    if (cache.has_value())
        return;
    Population &pop = *population;
    cache.emplace(Cache{
        pop.getDataContainer<Objective>(),
    });
}
ObjectiveFunction *ObjectiveValuesToReachDetector::clone()
{
    auto wrapping_clone = std::shared_ptr<ObjectiveFunction>(wrapping->clone());
    return new ObjectiveValuesToReachDetector(wrapping_clone, vtrs);
}

// Self-Dropping individual
DroppingIndividual::DroppingIndividual(Individual &&i) : i(i)
{
}
DroppingIndividual::~DroppingIndividual()
{
    if (i.has_value())
    {
        auto ii = *i;
        ii.creator->dropIndividual(ii);
    }
}
DroppingIndividual::operator Individual *()
{
    return &i.value();
}
void DroppingIndividual::operator=(const Individual &&i)
{
    this->i.emplace(i);
}
DroppingIndividual::DroppingIndividual(DroppingIndividual &&i)
{
    this->i.swap(i.i);
}
