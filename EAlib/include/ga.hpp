#pragma once

#include <algorithm>
#include <base.hpp>
#include <cstddef>
#include <memory>
#include <numeric>
#include <random>

class ICrossover : public IDataUser
{
  public:
    virtual ~ICrossover() = default;
    virtual std::vector<Individual> crossover(std::vector<Individual> &parents) = 0;
    virtual size_t num_parents() = 0;
};

class IMutation : public IDataUser
{
  public:
    virtual ~IMutation() = default;
    virtual void mutate(std::vector<Individual> &offspring) = 0;
};

class ISelection : public IDataUser
{
  public:
    virtual ~ISelection() = default;
    virtual std::vector<Individual> select(std::vector<Individual> &ii_population, size_t amount) = 0;
};

/**
 * Uniform Crossover
 *
 * Takes two solutions and produces two solutions by drawing each variable independently from either parent,
 * selecting the first parent with probability p, and the second parent with probability 1 - p.
 **/
class UniformCrossover : public ICrossover
{
    struct Cache
    {
        TypedGetter<GenotypeCategorical> tggc;
        Rng *rng;
    };
    std::optional<Cache> cache;
    void doCache();

  public:
    UniformCrossover(float p = 0.5) : p(p){};

    void afterRegisterData() override;

    std::vector<Individual> crossover(std::vector<Individual> &parents) override;
    size_t num_parents() override
    {
        return 2;
    }

    float p;
};

/**
 * k-Point Crossover
 *
 * Takes two solutions and produces two solutions by recombining using k+1 contiguous strings.
 * As such the distribution is dependent on the order within the string.
 */
class KPointCrossover : public ICrossover
{
    struct Cache
    {
        TypedGetter<GenotypeCategorical> tggc;
        Rng *rng;
    };
    std::optional<Cache> cache;
    void doCache();

  public:
    KPointCrossover(size_t k);

    void afterRegisterData() override;
    std::vector<Individual> crossover(std::vector<Individual> &parents) override;
    size_t num_parents() override
    {
        return 2;
    }

    size_t k;
};

class SubfunctionCrossover : public ICrossover
{
    double p;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> tggc;
        Rng *rng;
        Subfunctions *subfns;
    };
    std::optional<Cache> cache;
    void doCache();

  public:
    SubfunctionCrossover(double p = 0.5);

    void afterRegisterData() override;
    std::vector<Individual> crossover(std::vector<Individual> &parents) override;
    size_t num_parents() override
    {
        return 2;
    }
};

// Subfunctions
/**
 * Do not perform any mutation, i.e. make mutation attempts a no-op.
 */
class NoMutation : public IMutation
{
  public:
    NoMutation();
    void mutate(std::vector<Individual> &) override;
};

class PerVariableBitFlipMutation : public IMutation
{
    double p;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> tggc;
        Rng* rng;
    };
    std::optional<Cache> cache;
    void doCache();

    void mutate_individual(Individual &i);

  public:
    PerVariableBitFlipMutation(double p);

    void mutate(std::vector<Individual> &iis) override;
};

class PerVariableInAlphabetMutation : public IMutation
{
    double p;

    struct Cache
    {
        TypedGetter<GenotypeCategorical> tggc;
        GenotypeCategoricalData* gcd;
        Rng* rng;
    };
    std::optional<Cache> cache;
    void doCache();

    void mutate_individual(Individual &i);

  public:
    PerVariableInAlphabetMutation(double p);

    void mutate(std::vector<Individual> &iis) override;

    void setPopulation(std::shared_ptr<Population> population) override;
};
/**
 * Uniformly randomly select with replacement.
 */
class RandomSelection : public ISelection
{
  public:
    RandomSelection(){};
    void afterRegisterData() override
    {
        Population &pop = *population;
        // Rng is required for randomly selecting a solution.
        t_assert(pop.isGlobalRegistered<Rng>(), "Random Selection requires a random number generator to be present.");
    };

    std::vector<Individual> select(std::vector<Individual> &ii_population, size_t amount) override;
};

/**
 * Uniformly randomly select without replacement.
 */
class RandomUniqueSelection : public ISelection
{
  public:
    RandomUniqueSelection(){};
    void afterRegisterData() override
    {
        Population &pop = *population;
        // Rng is required for randomly selecting a solution.
        t_assert(pop.isGlobalRegistered<Rng>(), "Random Selection requires a random number generator to be present.");
    };

    std::vector<Individual> select(std::vector<Individual> &ii_population, size_t amount) override;
};

/**
 * Shuffled Sequential Selection
 *
 * Resets if a different source population is used.
 */
class ShuffledSequentialSelection : public ISelection
{
  public:
    ShuffledSequentialSelection(){};
    void afterRegisterData() override
    {
        Population &pop = *population;
        // Rng is required for randomly selecting a solution.
        t_assert(pop.isGlobalRegistered<Rng>(),
                 "ShuffledSequentialSelection requires a random number generator to be present for shuffling.");
    };

    std::vector<Individual> select(std::vector<Individual> &ii_population, size_t amount) override;

  private:
    std::vector<size_t> shuffle;
    size_t idx;

    void reset(std::vector<Individual> &ii_population);
};

/**
 * OrderedTournamentSelection
 *
 * Perform tournament selection assuming the provided comparator induces a complete ordering lattice,
 * i.e. we can sort solutions in one order, such that one solution is better than or equal to the others.
 **/
class OrderedTournamentSelection : public ISelection
{
  public:
    OrderedTournamentSelection(size_t tournament_size,
                               size_t samples_per_tournament,
                               std::shared_ptr<ISelection> pool_selection,
                               std::shared_ptr<IPerformanceCriterion> comparator);

    void setPopulation(std::shared_ptr<Population> population) override
    {
        ISelection::setPopulation(population);
        pool_selection->setPopulation(population);
        comparator->setPopulation(population);
    }
    void registerData() override
    {
        pool_selection->registerData();
        comparator->registerData();
    }

    void afterRegisterData() override
    {
        Population &pop = *population;
        // Rng is required to break ties.
        t_assert(pop.isGlobalRegistered<Rng>(), "Random Selection requires a random number generator to be present.");
        pool_selection->afterRegisterData();
        comparator->afterRegisterData();
    };

    std::vector<Individual> select(std::vector<Individual> &ii_population, size_t amount) override;

  private:
    size_t tournament_size;
    size_t samples_per_tournament;
    std::shared_ptr<ISelection> pool_selection;
    std::shared_ptr<IPerformanceCriterion> comparator;
};

class TruncationSelection : public ISelection
{
  private:
    std::shared_ptr<IPerformanceCriterion> comparator;

  public:
    TruncationSelection(std::shared_ptr<IPerformanceCriterion> comparator);

    void setPopulation(std::shared_ptr<Population> population) override
    {
        ISelection::setPopulation(population);
        comparator->setPopulation(population);
    }
    void registerData() override
    {
        comparator->registerData();
    }

    void afterRegisterData() override
    {
        Population &pop = *population;
        // Rng is required to break ties.
        t_assert(pop.isGlobalRegistered<Rng>(), "Random Selection requires a random number generator to be present.");
        comparator->afterRegisterData();
    };

    std::vector<Individual> select(std::vector<Individual> &ii_population, size_t amount) override;
};

// Note: think about making it possible to use a template to initialize this specifically.

class SimpleGA : public GenerationalApproach
{
  public:
    SimpleGA(size_t population_size,
             std::shared_ptr<ISolutionInitializer> initializer,
             std::shared_ptr<ICrossover> crossover,
             std::shared_ptr<IMutation> mutation,
             std::shared_ptr<ISelection> parent_selection,
             std::shared_ptr<ISelection> population_selection,
             std::shared_ptr<IPerformanceCriterion> performance_criterion,

             std::optional<size_t> offspring_size = std::nullopt,
             std::optional<bool> copy_population_to_offspring = true);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void step() override;

    std::vector<Individual> &getSolutionPopulation() override;

  private:
    std::vector<Individual> ii_population;
    bool initialized = false;
    void initialize();

    void recombine_mutate_evaluate_select();

    const size_t population_size;
    const size_t offspring_size;
    const bool copy_population_to_offspring;
    const std::shared_ptr<ISolutionInitializer> initializer;
    const std::shared_ptr<ICrossover> crossover;
    const std::shared_ptr<IMutation> mutation;
    const std::shared_ptr<ISelection> parent_selection;
    const std::shared_ptr<ISelection> population_selection;
    const std::shared_ptr<IPerformanceCriterion> performance_criterion;
};