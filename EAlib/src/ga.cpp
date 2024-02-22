#include "ga.hpp"
#include "base.hpp"

#include <algorithm>
#include <cstddef>
#include <numeric>
#include <random>

void UniformCrossover::afterRegisterData()
{
    t_assert(population != NULL, "Population should be assigned");

    t_assert((*population).isRegistered<GenotypeCategorical>(),
             "Uniform Crossover operates on the categorical genotype");

    t_assert((*population).isGlobalRegistered<Rng>(), "Uniform Crossover requires a random number generator");
}

void UniformCrossover::doCache()
{
    if (cache.has_value())
        return;
    cache.emplace(Cache{
        population->getDataContainer<GenotypeCategorical>(),
        population->getGlobalData<Rng>().get(),
    });
}

std::vector<Individual> UniformCrossover::crossover(std::vector<Individual> &parents)
{
    t_assert(parents.size() == 2, "Uniform Crossover operates on two solutions at a time");
    doCache();

    auto &pop = (*population);
    auto &rng = *cache->rng;

    auto oi0 = pop.newIndividual();
    auto oi1 = pop.newIndividual();

    auto &a_genotype = cache->tggc.getData(parents[0]);
    auto &b_genotype = cache->tggc.getData(parents[1]);
    size_t l = a_genotype.genotype.size();
    t_assert(a_genotype.genotype.size() == l, "Parent genotype a should be initialized.");
    t_assert(b_genotype.genotype.size() == l, "Parent genotype b should be initialized.");

    auto &oi0_genotype = cache->tggc.getData(oi0);
    auto &oi1_genotype = cache->tggc.getData(oi1);
    oi0_genotype.genotype.resize(l);
    oi1_genotype.genotype.resize(l);


    std::uniform_real_distribution<float> probability(0.0, 1.0);

    for (size_t i = 0; i < l; ++i)
    {
        if (probability(rng.rng) > p)
        {
            oi0_genotype.genotype[i] = a_genotype.genotype[i];
            oi1_genotype.genotype[i] = b_genotype.genotype[i];
        }
        else
        {
            oi0_genotype.genotype[i] = b_genotype.genotype[i];
            oi1_genotype.genotype[i] = a_genotype.genotype[i];
        }
    }

    return {oi0, oi1};
}

void KPointCrossover::afterRegisterData()
{
    t_assert(population != NULL, "Population should be assigned");

    t_assert((*population).isRegistered<GenotypeCategorical>(),
             "K-point Crossover operates on the categorical genotype");

    t_assert((*population).isGlobalRegistered<Rng>(), "K-point Crossover requires a random number generator");
}

KPointCrossover::KPointCrossover(size_t k) : k(k)
{
}
void KPointCrossover::doCache()
{
    if (cache.has_value())
        return;
    cache.emplace(Cache{
        population->getDataContainer<GenotypeCategorical>(),
        population->getGlobalData<Rng>().get(),
    });
}

std::vector<Individual> KPointCrossover::crossover(std::vector<Individual> &parents)
{
    t_assert(parents.size() == 2, "k-point Crossover operates on two solutions at a time");
    doCache();
    auto &pop = (*population);
    auto &rng = *cache->rng;
    
    auto oi0 = pop.newIndividual();
    auto oi1 = pop.newIndividual();

    auto &a_genotype = cache->tggc.getData(parents[0]);
    auto &b_genotype = cache->tggc.getData(parents[1]);

    size_t l = a_genotype.genotype.size();
    auto &oi0_genotype = cache->tggc.getData(oi0);
    auto &oi1_genotype = cache->tggc.getData(oi1);
    oi0_genotype.genotype.resize(l);
    oi1_genotype.genotype.resize(l);

    std::uniform_int_distribution<size_t> position(0, l);
    std::vector<size_t> positions(k + 2);
    std::generate(positions.begin() + 1, positions.end() - 1, [&rng, &position]() { return position(rng.rng); });
    positions[0] = 0;
    positions[k + 2 - 1] = l;
    std::sort(positions.begin(), positions.end());

    for (size_t p = 0; p < positions.size() - 1; ++p)
    {
        size_t p_start = positions[p];
        size_t p_end = positions[p + 1];
        if (p % 2 == 0)
        {
            for (size_t i = p_start; i < p_end; ++i)
            {
                oi0_genotype.genotype[i] = a_genotype.genotype[i];
                oi1_genotype.genotype[i] = b_genotype.genotype[i];
            }
        }
        else
        {
            for (size_t i = p_start; i < p_end; ++i)
            {
                oi0_genotype.genotype[i] = b_genotype.genotype[i];
                oi1_genotype.genotype[i] = a_genotype.genotype[i];
            }
        }
    }

    return {oi0, oi1};
}

// Subfunction based crossover
// Note that this uses information about the objective function, comparisons
// against black-box approaches are therefore potentially misleading.
// As a reference for an 'ideal' approach, this is quite useful however.
// but note that the best way to recombine is not necessarily determined by
// the subfunctions.
SubfunctionCrossover::SubfunctionCrossover(double p) : p(p)
{
}
void SubfunctionCrossover::afterRegisterData()
{
    t_assert(population != NULL, "Population should be assigned");

    t_assert((*population).isRegistered<GenotypeCategorical>(),
             "Subfunction Crossover operates on the categorical genotype");

    t_assert((*population).isGlobalRegistered<Rng>(), "K-point Crossover requires a random number generator");
}
void SubfunctionCrossover::doCache()
{
    if (cache.has_value())
        return;
    cache.emplace(Cache{
        population->getDataContainer<GenotypeCategorical>(),
        population->getGlobalData<Rng>().get(),
        population->getGlobalData<Subfunctions>().get(),
    });
}
std::vector<Individual> SubfunctionCrossover::crossover(std::vector<Individual> &parents)
{
    t_assert(parents.size() == 2, "Subfunction Crossover operates on two solutions at a time");
    doCache();

    auto &pop = (*population);
    auto &rng = *cache->rng;

    auto oi0 = pop.newIndividual();
    auto oi1 = pop.newIndividual();

    auto &a_genotype = cache->tggc.getData(parents[0]);
    auto &b_genotype = cache->tggc.getData(parents[1]);
    size_t l = a_genotype.genotype.size();
    t_assert(a_genotype.genotype.size() == l, "Parent genotype a should be initialized.");
    t_assert(b_genotype.genotype.size() == l, "Parent genotype b should be initialized.");

    auto &oi0_genotype = cache->tggc.getData(oi0);
    auto &oi1_genotype = cache->tggc.getData(oi1);
    oi0_genotype.genotype.resize(l);
    oi1_genotype.genotype.resize(l);

    std::copy(a_genotype.genotype.begin(), a_genotype.genotype.end(), oi0_genotype.genotype.begin());
    std::copy(b_genotype.genotype.begin(), b_genotype.genotype.end(), oi1_genotype.genotype.begin());

    // Shuffle the subfunctions
    std::vector<size_t> indices(cache->subfns->subfunctions.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), rng.rng);

    std::uniform_real_distribution<double> u(0.0, 1.0);

    for (auto idx : indices)
    {
        // Only exchange with probability p
        // Skip otherwise.
        if (u(rng.rng) >= p)
            continue;

        for (auto v : cache->subfns->subfunctions[idx])
        {
            std::swap(oi0_genotype.genotype[v], oi1_genotype.genotype[v]);
        }
    }

    return {oi0, oi1};
}

// No mutation is quite simple: perform no mutation.
NoMutation::NoMutation()
{
}
void NoMutation::mutate(std::vector<Individual> &)
{
}

// Bitflip mutation - flip a bit with probability p
// Note: assumes the categorical genotype to be binary.
void PerVariableBitFlipMutation::doCache()
{
    if (cache.has_value())
        return;
    cache.emplace(Cache{
        population->getDataContainer<GenotypeCategorical>(),
        population->getGlobalData<Rng>().get(),
    });
}
void PerVariableBitFlipMutation::mutate_individual(Individual &i)
{
    auto &genotype = cache->tggc.getData(i);
    size_t l = genotype.genotype.size();
    std::uniform_real_distribution<double> u(0.0, 1.0);

    for (size_t vidx = 0; vidx < l; ++vidx)
    {
        if (u(cache->rng->rng) < p)
            genotype.genotype[vidx] = (genotype.genotype[vidx] == 0) ? 1 : 0;
    }
}
PerVariableBitFlipMutation::PerVariableBitFlipMutation(double p) : p(p)
{
}
void PerVariableBitFlipMutation::mutate(std::vector<Individual> &iis)
{
    doCache();
    for (auto &i : iis)
    {
        mutate_individual(i);
    }
}

void PerVariableInAlphabetMutation::doCache()
{
    if (cache.has_value())
        return;
    cache.emplace(Cache{
        population->getDataContainer<GenotypeCategorical>(),
        population->getGlobalData<GenotypeCategoricalData>().get(),
        population->getGlobalData<Rng>().get(),
    });
}
void PerVariableInAlphabetMutation::mutate_individual(Individual &i)
{
    auto &genotype = cache->tggc.getData(i);
    size_t l = genotype.genotype.size();
    std::uniform_real_distribution<double> u(0.0, 1.0);


    for (size_t vidx = 0; vidx < l; ++vidx)
    {
        if (u(cache->rng->rng) < p)
        {
            std::uniform_int_distribution<char> ud(0, static_cast<char>(cache->gcd->alphabet_size[vidx] - 2));
            auto v = ud(cache->rng->rng);
            if (genotype.genotype[vidx] >= v)
            {
                v += 1;
            }
            genotype.genotype[vidx] = v;
        }
    }
}
PerVariableInAlphabetMutation::PerVariableInAlphabetMutation(double p) : p(p)
{
}
void PerVariableInAlphabetMutation::mutate(std::vector<Individual> &iis)
{
    doCache();
    for (auto &i : iis)
    {
        mutate_individual(i);
    }
}
void PerVariableInAlphabetMutation::setPopulation(std::shared_ptr<Population> population)
{
    IMutation::setPopulation(population);
    cache.reset();
}

// Random Selection
std::vector<Individual> RandomSelection::select(std::vector<Individual> &ii_population, size_t amount)
{
    Population &pop = (*population);
    Rng &rng = *pop.getGlobalData<Rng>();

    std::vector<Individual> out(amount);
    std::uniform_int_distribution<size_t> index(0, ii_population.size() - 1);

    for (size_t i = 0; i < amount; ++i)
    {
        out[i] = ii_population[index(rng.rng)];
    }

    return out;
}

// Random Unique Selection
std::vector<Individual> RandomUniqueSelection::select(std::vector<Individual> &ii_population, size_t amount)
{
    Population &pop = (*population);
    Rng &rng = *pop.getGlobalData<Rng>();

    std::vector<Individual> out(amount);

    auto end = std::sample(ii_population.begin(), ii_population.end(), out.begin(), amount, rng.rng);
    out.erase(end, out.end());

    return out;
}

// Ordered Tournament Selection
//
// - Assumes that samples are well ordered, selects the top `samples_per_tournament` of a sample of `tournament_size`
//   from the population
OrderedTournamentSelection::OrderedTournamentSelection(size_t tournament_size,
                                                       size_t samples_per_tournament,
                                                       std::shared_ptr<ISelection> pool_selection,
                                                       std::shared_ptr<IPerformanceCriterion> comparator) :
    tournament_size(tournament_size),
    samples_per_tournament(samples_per_tournament),
    pool_selection(std::move(pool_selection)),
    comparator(std::move(comparator))
{
}

std::vector<Individual> OrderedTournamentSelection::select(std::vector<Individual> &ii_population, size_t amount)
{
    size_t selected = 0;
    std::vector<Individual> result(amount);

    while (selected < amount)
    {
        auto pool = pool_selection->select(ii_population, tournament_size);
        std::nth_element(pool.begin(),
                         pool.begin() + static_cast<long>(samples_per_tournament),
                         pool.end(),
                         [this](Individual &a, Individual &b) { return comparator->compare(a, b) == 1; });
        for (size_t i = 0; i < samples_per_tournament && selected < amount; ++i, ++selected)
        {
            result[selected] = pool[i];
        }
    }
    return result;
}

// Shuffled Sequential Selection
void ShuffledSequentialSelection::reset(std::vector<Individual> &ii_population)
{
    shuffle.resize(ii_population.size());
    std::iota(shuffle.begin(), shuffle.end(), 0);
    
    Rng &rng = *(*population).getGlobalData<Rng>();
    std::shuffle(shuffle.begin(), shuffle.end(), rng.rng);

    idx = 0;
}

std::vector<Individual> ShuffledSequentialSelection::select(std::vector<Individual> &ii_population, size_t amount){
    t_assert(ii_population.size() >= amount, "Cannot sample more than population size samples at once.");
    // Should we reset?
    if (ii_population.size() != shuffle.size() || // Population to sample from has different size
        idx + amount - 1 >= shuffle.size() // Out of samples
    )
    {
        reset(ii_population);
    }
    std::vector<Individual> out(amount);
    for (size_t num = 0; num < amount; ++num, ++idx)
    {
        // Note: remove the shuffling & replace shuffle[idx] with idx
        // and you get SequentialSelection (without shuffling)
        out[num] = ii_population[shuffle[idx]];
    }
    return out;
}

// TruncationSelection

TruncationSelection::TruncationSelection(std::shared_ptr<IPerformanceCriterion> comparator) : comparator(comparator)
{
}
std::vector<Individual> TruncationSelection::select(std::vector<Individual> &ii_population, size_t amount)
{
    std::vector<size_t> indices(ii_population.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::vector<int> tiebreakers(ii_population.size());
    std::fill(tiebreakers.begin(), tiebreakers.end(), -1);

    t_assert(amount <= ii_population.size(),
             "Pool that is being sampled from should be larger or equal to the amount of solutions being sampled.");

    std::uniform_int_distribution<int> d(0);

    Population &pop = *this->population;
    Rng &rng = *pop.getGlobalData<Rng>();

    std::nth_element(indices.begin(),
                     indices.begin() + static_cast<long>(amount),
                     indices.end(),
                     [&ii_population, &tiebreakers, this, &d, &rng](size_t a, size_t b) {
                         // Use comparator
                         short c = comparator->compare(ii_population[a], ii_population[b]);

                         if (c == 1)
                             return true;
                         if (c == 2)
                             return false;

                         // Tie or inconclusive -- note: act as if inconclusive is a tie here.
                         // As sorting does not necessarily work nicely this way, an alternative implementation
                         // may be warranted.

                         // In this case we assign a tiebreaker, if we haven't already assigned a value.
                         if (tiebreakers[a] == -1)
                             tiebreakers[a] = d(rng.rng);
                         if (tiebreakers[b] == -1)
                             tiebreakers[b] = d(rng.rng);

                         return a < b;
                     });

    std::vector<Individual> result(amount);
    for (size_t iidx = 0; iidx < amount; ++iidx)
    {
        result[iidx] = ii_population[indices[iidx]];
    }

    return result;
}

// Standard GA setup.
SimpleGA::SimpleGA(size_t population_size,
                   std::shared_ptr<ISolutionInitializer> initializer,
                   std::shared_ptr<ICrossover> crossover,
                   std::shared_ptr<IMutation> mutation,
                   std::shared_ptr<ISelection> parent_selection,
                   std::shared_ptr<ISelection> population_selection,
                   std::shared_ptr<IPerformanceCriterion> performance_criterion,

                   std::optional<size_t> offspring_size,
                   std::optional<bool> copy_population_to_offspring) :
    population_size(population_size),
    offspring_size(offspring_size.value_or(2 * population_size)),
    copy_population_to_offspring(copy_population_to_offspring),
    initializer(std::move(initializer)),
    crossover(std::move(crossover)),
    mutation(std::move(mutation)),
    parent_selection(std::move(parent_selection)),
    population_selection(std::move(population_selection)),
    performance_criterion(std::move(performance_criterion))
{
    t_assert(this->offspring_size >= population_size,
             "Number of offspring should be equal or larger than the population size");
}

// Cascade these operations downwards, should make things a tad bit shorter and easier.
void SimpleGA::setPopulation(std::shared_ptr<Population> population)
{
    IDataUser::setPopulation(population);
    initializer->setPopulation(population);
    crossover->setPopulation(population);
    mutation->setPopulation(population);
    parent_selection->setPopulation(population);
    population_selection->setPopulation(population);
    performance_criterion->setPopulation(population);
}
void SimpleGA::registerData()
{
    IDataUser::registerData();
    initializer->registerData();
    crossover->registerData();
    mutation->registerData();
    parent_selection->registerData();
    population_selection->registerData();
    performance_criterion->registerData();
}
void SimpleGA::afterRegisterData()
{
    IDataUser::afterRegisterData();
    initializer->afterRegisterData();
    crossover->afterRegisterData();
    mutation->afterRegisterData();
    parent_selection->afterRegisterData();
    population_selection->afterRegisterData();
    performance_criterion->afterRegisterData();
}

void SimpleGA::step()
{
    // First step will initialize
    if (!initialized)
    {
        initialize();
        initialized = true;
        return;
    }

    // Otherwise we run an interation of the recombine mutate evaluate select loop
    recombine_mutate_evaluate_select();
}

void SimpleGA::initialize()
{
    // Resize to population size
    ii_population.resize(population_size);

    t_assert(population != NULL, "Population should be registered before stepping.");
    auto &pop = (*population);

    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    std::generate(ii_population.begin(), ii_population.end(), [&pop]() { return pop.newIndividual(); });
    initializer->initialize(ii_population);
    for (auto ii: ii_population)
    {
        objective_function.of->evaluate(ii);
    }
}

void SimpleGA::recombine_mutate_evaluate_select()
{
    //
    size_t offspring_idx = 0;
    std::vector<Individual> offspring(offspring_size);
    auto &pop = (*population);
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();


    if (copy_population_to_offspring)
    {
        for (size_t i = 0; i < population_size; ++i)
        {
            offspring[i] = ii_population[i];
        }
        offspring_idx = population_size;
    }

    while (offspring_idx < offspring_size)
    {
        // While we still have insufficient offspring:
        // - select the required number of parents
        std::vector<Individual> parents = parent_selection->select(ii_population, crossover->num_parents());
        // - perform crossover (recombine)
        std::vector<Individual> parent_offspring = crossover->crossover(parents);
        // - mutate
        mutation->mutate(parent_offspring);

        // Maybe: shuffle solutions before doing this?
        // - evaluate & add solutions to offspring
        for (size_t i = 0; offspring_idx < offspring_size && i < parent_offspring.size(); ++i, ++offspring_idx)
        {
            objective_function.of->evaluate(parent_offspring[i]);
            offspring[offspring_idx] = parent_offspring[i];
        }
    }

    // If we have not copied the population, these solutions will go unused from now on.
    // Indicate this so the memory associated can be reused.
    if (!copy_population_to_offspring)
    {
        std::for_each(ii_population.begin(), ii_population.end(), [&pop](Individual ii) { pop.dropIndividual(ii); });
    }

    // Copy over the selected solutions to a set of new solutions
    auto selected = population_selection->select(offspring, population_size);
    std::generate(ii_population.begin(), ii_population.end(), [&pop]() { return pop.newIndividual(); });

    for (size_t i = 0; i < population_size; ++i)
    {
        pop.copyIndividual(selected[i], ii_population[i]);
    }

    // The offspring now go unused as well, similarly to the population when not copied.
    // (and including the population itself, if copied)
    std::for_each(offspring.begin(), offspring.end(), [&pop](Individual ii) { pop.dropIndividual(ii); });
}
std::vector<Individual> &SimpleGA::getSolutionPopulation()
{
    return ii_population;
}
