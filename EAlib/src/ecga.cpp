//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "ecga.hpp"
#include "base.hpp"
#include "utilities.hpp"
#include <algorithm>
#include <cstddef>
#include <exception>
#include <random>

void fast_index_radix_sort(
    size_t max_v, size_t current_bit, size_t *indices, size_t l_start, size_t r_start, size_t *vs)
{
    // Note: l_start is a valid index, r_start is one after the last index to sort.
    // (this is mostly to allow encoding zero length arrays, which otherwise would require negative numbers)

    // Variance in bits left is zero, we can stop:
    // the sequence has been sorted.
    if (max_v == 0)
        return;
    // Similarly sequences of size 0 & 1 are sorted.
    if (l_start == r_start || l_start + 1 == r_start)
        return;

    // While there are items left to sort for the current bit
    size_t l = l_start;
    size_t r = r_start - 1; // Actual starting point is one lower (see reasoning above)
    while (l < r)
    {
        // Check the leftmost item, is bit at current_bit 1?
        if (((vs[indices[l]] >> current_bit) & 1) > 0)
        {
            // If it is, we should relocate it to the right.
            std::swap(indices[l], indices[r]);
            // Index at r is now sorted, so decrement.
            --r;
        }
        else
        {
            // If it is zero, it is in the right position
            ++l;
        }
    }
    // Sort the two subsequences for the next bit.
    // Note that l == r at this point, and l is the index for the first 1-bit value.
    fast_index_radix_sort(max_v >> 1, current_bit + 1, indices, l_start, l, vs);
    fast_index_radix_sort(max_v >> 1, current_bit + 1, indices, l, r_start, vs);
}

struct MPCell
{
    std::vector<size_t> p_indices;
    std::vector<size_t> ord_per_p;
    std::vector<size_t> s_sizes;
    // Model complexity, excluding weight.
    double model_complexity = 1;
    // Compressed population complexity, excluding weight.
    double population_complexity = 0;
};

std::vector<std::vector<size_t>> learnMPM(size_t l, std::vector<std::vector<char>> &genotypes)
{
    size_t N = genotypes.size();
    double dN = static_cast<double>(N);
    double wMC = std::log2(dN + 1);
    double wCPC = static_cast<double>(N);
    // Initialize univariate mpm & bookkeeping utilities.
    std::vector<std::vector<size_t>> mpm(l);
    std::vector<MPCell> c(l);

    std::vector<size_t> vs(genotypes.size());
    for (size_t idx = 0; idx < l; ++idx)
    {
        mpm[idx] = {idx};
        // Ensure the vectors are correctly sized.
        c[idx].p_indices.resize(N);
        c[idx].ord_per_p.resize(N);
        std::iota(c[idx].p_indices.begin(), c[idx].p_indices.end(), 0);

        // Order indices such that population members belonging to the same bin are sequential.
        // std::sort(c[idx].p_indices.begin(), c[idx].p_indices.end(), [&genotypes, &idx](size_t idx_i, size_t idx_j) {
        //     return genotypes[idx_i][idx] < genotypes[idx_j][idx];
        // });
        size_t largest_v = 0;
        for (size_t x = 0; x < genotypes.size(); ++x)
        {
            auto v = static_cast<size_t>(static_cast<unsigned char>(genotypes[x][idx]));
            largest_v = std::max(largest_v, v);
            vs[x] = v;
        }
        fast_index_radix_sort(largest_v, 0, c[idx].p_indices.data(), 0, vs.size(), vs.data());
        // Determine initial values.
        size_t current_size = 0;
        size_t ord = 0;
        char current_char = genotypes[c[idx].p_indices[0]][idx];
        for (size_t i = 0; i < N; ++i)
        {
            size_t current = c[idx].p_indices[i];
            // New, or same, bin?
            if (genotypes[current][idx] == current_char)
            {
                current_size += 1;
            }
            else
            {
                c[idx].s_sizes.push_back(current_size);
                double p = static_cast<double>(current_size) / dN;
                c[idx].population_complexity += -p * std::log2(p);
                current_size = 1;
                ord += 1;
                current_char = genotypes[current][idx];
            }
            c[idx].ord_per_p[current] = ord;
        }
        // Complete last bin
        c[idx].s_sizes.push_back(current_size);
        double p = static_cast<double>(current_size) / dN;
        c[idx].population_complexity += -p * std::log2(p);
    }
    // Note: commented out as we are not using this.
    // double current_combined_complexity = wMC * current_model_complexity + wCPC * current_population_complexity;

    // Possible - i.e. let us determine the minimum merge without scanning over the entire matrix.
    // TODO?: Randomize in case of tie (instead of preferring largest first int?)
    std::vector<std::tuple<double, size_t, size_t>> possible_merges;
    size_t min_index = 0;

    auto evaluate_delta_merge = [wMC, &c, wCPC, dN, &mpm](size_t i, size_t j, bool update) {
        // Compute the cost of merging MPM of i & j.
        double model_complexity_contrib_i = c[i].model_complexity;
        double model_complexity_contrib_j = c[j].model_complexity;
        double model_complexity_contrib_ij = ((1 << (mpm[i].size() + mpm[j].size())) - 1);

        if (update)
        {
            // When updating, i is reused, generally speaking, as well as updated.
            c[i].model_complexity = model_complexity_contrib_ij;
        }

        double delta_model_complexity =
            model_complexity_contrib_ij - (model_complexity_contrib_i + model_complexity_contrib_j);

        // Compute population contribution for merge.
        double population_complexity_contrib_ij = 0.0;
        size_t start = 0;
        auto &sizes = c[i].s_sizes;
        auto &new_sizes = c[j].s_sizes;
        if (update)
        {
            // repurpose j's array if we are going to update!
            new_sizes.clear();
        }
        size_t ord = 0;
        for (size_t idx = 0; idx < sizes.size(); ++idx)
        {
            size_t end = start + c[i].s_sizes[idx];
            // Reorder subsequence by ord of other sequence
            // std::sort(c[i].p_indices.begin() + static_cast<int>(start),
            //           c[i].p_indices.begin() + static_cast<int>(end),
            //           [&c, &j](size_t idx_i, size_t idx_j) { return c[j].ord_per_p[idx_i] < c[j].ord_per_p[idx_j];
            //           });
            fast_index_radix_sort(c[j].s_sizes.size() - 1, 0, c[i].p_indices.data(), start, end, c[j].ord_per_p.data());

            size_t current_size = 0;
            size_t current_other_ord = c[j].ord_per_p[c[i].p_indices[start]];
            // Go over sequence, determine bins their counts
            for (size_t x = start; x < end; ++x)
            {
                size_t current = c[i].p_indices[x];
                // Still same bin?
                if (current_other_ord == c[j].ord_per_p[current])
                {
                    // if yes: increase size
                    current_size += 1;
                }
                else
                {
                    // add current size to new_sizes, if updating
                    if (update)
                    {
                        new_sizes.push_back(current_size);
                    }
                    // otherwise: add contribution of bin to value.
                    double p = static_cast<double>(current_size) / dN;
                    population_complexity_contrib_ij += -p * std::log2(p);
                    // and set size to 1
                    current_size = 1;
                    // increase ord to start a new bin
                    ord += 1;
                    // change current other ord
                    current_other_ord = c[j].ord_per_p[current];
                }
                if (update)
                {
                    c[i].ord_per_p[current] = ord;
                }
            }
            // Once done: complete tasks for final bin -- note: this always completes the bin
            if (update)
            {
                new_sizes.push_back(current_size);
            }
            // add contribution of final combined i(idx),j bin to value.
            double p = static_cast<double>(current_size) / dN;
            population_complexity_contrib_ij += -p * std::log2(p);
            // increase ord (as there will likely be bins for i + 1 as well!)
            ord += 1;

            start = end;
        }

        if (update)
        {
            // Update sizes: we no longer need the old contents!
            sizes.swap(new_sizes);
            // And corresponding size!
            c[i].population_complexity = population_complexity_contrib_ij;
            // Merge mpm & clear mpm of j
            mpm[i].insert(mpm[i].end(), mpm[j].begin(), mpm[j].end());
            mpm[j].clear();
        }

        //
        double population_complexity_contrib_i = c[i].population_complexity;
        double population_complexity_contrib_j = c[j].population_complexity;
        double delta_population_complexity =
            population_complexity_contrib_ij - (population_complexity_contrib_i + population_complexity_contrib_j);
        double delta_combined_complexity = wMC * delta_model_complexity + wCPC * delta_population_complexity;
        return delta_combined_complexity;
    };

    // Compute pairwise merging costs
    // Note! If merge cost is positive, we should NEVER perform this merge!
    for (size_t i = 0; i < l; ++i)
    {
        for (size_t j = i + 1; j < l; ++j)
        {
            double delta_combined_complexity = evaluate_delta_merge(i, j, false);
            if (delta_combined_complexity < 0.0)
            {
                // Add to list of merges
                possible_merges.push_back(std::make_tuple(delta_combined_complexity, i, j));
                if (std::get<0>(possible_merges[min_index]) > std::get<0>(possible_merges.back()))
                    min_index = possible_merges.size() - 1;
            }
        }
    }

    //
    while (possible_merges.size() > 0)
    {
        // Get merge to perform (at min_index)
        auto i = std::get<1>(possible_merges[min_index]);
        auto j = std::get<2>(possible_merges[min_index]);
        // Actually perform merge
        evaluate_delta_merge(i, j, true);

        // Remove possible merges involving i or j (or both: removes element above).
        // These merges have been invalidated.
        auto new_end_possible_merges = std::remove_if(possible_merges.begin(), possible_merges.end(), [i, j](auto e) {
            auto [delta_o, i_o, j_o] = e;
            return i_o == i || i_o == j || j_o == i || j_o == j;
        });
        possible_merges.erase(new_end_possible_merges, possible_merges.end());

        // Recompute min_index (as removal has changed indices, and certainly includes the removal of the current
        // minimum element)
        min_index = 0;
        for (size_t o = 0; o < possible_merges.size(); ++o)
        {
            if (std::get<0>(possible_merges[min_index]) > std::get<0>(possible_merges[o]))
                min_index = o;
        }

        // Evaluate new potential merges between the newly merged element i and the remaining elements.
        for (size_t j = 0; j < l; ++j)
        {
            // Cannot merge with oneself (i == j), or an already merged subset (mpm is of size 0)
            if (i == j || mpm[j].size() == 0)
                continue;
            double delta_combined_complexity = evaluate_delta_merge(i, j, false);
            if (delta_combined_complexity < 0.0)
            {
                // Add to list of merges
                possible_merges.push_back(std::make_tuple(delta_combined_complexity, i, j));
                if (std::get<0>(possible_merges[min_index]) > std::get<0>(possible_merges.back()))
                    min_index = possible_merges.size() - 1;
            }
        }
    }

    // Remove zero sized elements.
    auto new_mpm_end = std::remove_if(mpm.begin(), mpm.end(), [](auto &e) { return e.size() == 0; });
    mpm.erase(new_mpm_end, mpm.end());
    // Return resulting mpm model.
    return mpm;
}

std::vector<std::vector<size_t>> learnMPM(size_t l, std::vector<std::vector<char>> &&genotypes)
{
    return learnMPM(l, genotypes);
}

ECGAGreedyMarginalProduct::ECGAGreedyMarginalProduct(size_t update_pop_every_learn_call,
                                                     size_t update_mpm_every_pop_update) :
    update_pop_every_learn_call(update_pop_every_learn_call), update_mpm_every_pop_update(update_mpm_every_pop_update)
{
}
bool ECGAGreedyMarginalProduct::should_update()
{
    learn_calls_since_last_update++;
    if (learn_calls_since_last_update < update_pop_every_learn_call)
        return false;
    learn_calls_since_last_update = 0;
    return true;
}
void ECGAGreedyMarginalProduct::afterRegisterData()
{
    Population &pop = *population;
    cache.emplace(Cache{
        pop.getDataContainer<GenotypeCategorical>(),
        *pop.getGlobalData<Rng>(),
    });
}
void ECGAGreedyMarginalProduct::learn(std::vector<Individual> &individuals)
{
    // Copy genotypes
    size_t N = individuals.size();
    genotypes.resize(N);
    for (size_t idx = 0; idx < N; ++idx)
    {
        genotypes[idx] = cache->tggc.getData(individuals[idx]).genotype;
    }
    // Obtain length from solution
    l = genotypes[0].size();
    // Learn MPM (every `update_mpm_every` times the population was updated)
    updates_since_last_mpm_update++;
    if (updates_since_last_mpm_update >= update_mpm_every_pop_update)
    {
        updates_since_last_mpm_update = 0;
        mpm = learnMPM(l, genotypes);
    }
}
void ECGAGreedyMarginalProduct::sample(Individual &ii)
{
    auto &g = cache->tggc.getData(ii);
    std::uniform_int_distribution<size_t> g_idx(0, genotypes.size() - 1);

    // Construct a solution using the marginal product model.
    g.genotype.resize(l);
    for (auto &s : mpm)
    {
        size_t idx = g_idx(cache->rng.rng);
        for (auto e : s)
        {
            g.genotype[e] = genotypes[idx][e];
        }
    }
}
std::shared_ptr<ISolutionSamplingDistribution> ECGAGreedyMarginalProduct::clone()
{
    return std::shared_ptr<ISolutionSamplingDistribution>(new ECGAGreedyMarginalProduct(*this));
}

void SynchronousSimulatedECGA::replace_fifo(Individual ii)
{
    Population &pop = *population;
    if (sl != NULL)
    {
        sl->end_span(current_replacement_index, individuals[current_replacement_index], 0);
        sl->start_span(current_replacement_index, ii, 0);
    }
    pop.copyIndividual(ii, individuals[current_replacement_index]);
    current_replacement_index += 1;
    if (current_replacement_index >= population_size)
        current_replacement_index = 0;
}
void SynchronousSimulatedECGA::replace_idx(size_t idx, Individual ii)
{
    Population &pop = *population;
    if (sl != NULL)
    {
        sl->end_span(idx, individuals[idx], 0);
        sl->start_span(idx, ii, 0);
    }
    pop.copyIndividual(ii, individuals[idx]);
}
SynchronousSimulatedECGA::SynchronousSimulatedECGA(int replacement_strategy,
                                                   std::shared_ptr<SimulatorParameters> sim,
                                                   size_t population_size,
                                                   std::shared_ptr<ISolutionSamplingDistribution> distribution,
                                                   std::shared_ptr<ISolutionInitializer> initializer,
                                                   std::shared_ptr<ISelection> selection,
                                                   std::shared_ptr<IArchive> archive,
                                                   std::shared_ptr<SpanLogger> sl) :
    sim(sim),
    population_size(population_size),
    initializer(initializer),
    selection(selection),
    distribution(distribution),
    archive(archive),
    sl(sl),
    replacement_strategy(replacement_strategy)
{
}
SynchronousSimulatedECGA::SynchronousSimulatedECGA(int replacement_strategy,
                                                   std::shared_ptr<IPerformanceCriterion> perf_criterion,
                                                   std::shared_ptr<SimulatorParameters> sim,
                                                   size_t population_size,
                                                   std::shared_ptr<ISolutionSamplingDistribution> distribution,
                                                   std::shared_ptr<ISolutionInitializer> initializer,
                                                   std::shared_ptr<ISelection> selection,
                                                   std::shared_ptr<IArchive> archive,
                                                   std::shared_ptr<SpanLogger> sl) :
    sim(sim),
    population_size(population_size),
    initializer(initializer),
    selection(selection),
    distribution(distribution),
    archive(archive),
    sl(sl),
    replacement_strategy(replacement_strategy),
    perf_criterion(perf_criterion)
{
}
void SynchronousSimulatedECGA::setPopulation(std::shared_ptr<Population> population)
{
    GenerationalApproach::setPopulation(population);
    initializer->setPopulation(population);
    selection->setPopulation(population);
    distribution->setPopulation(population);
    archive->setPopulation(population);
    if (sl != NULL)
        sl->setPopulation(population);
}
void SynchronousSimulatedECGA::registerData()
{
    Population &pop = *population;
    pop.registerData<TimeSpent>();

    initializer->registerData();
    selection->registerData();
    distribution->registerData();
    archive->registerData();
    if (sl != NULL)
        sl->registerData();
}
void SynchronousSimulatedECGA::afterRegisterData()
{
    initializer->afterRegisterData();
    selection->afterRegisterData();
    distribution->afterRegisterData();
    archive->afterRegisterData();
    if (sl != NULL)
        sl->afterRegisterData();
}
void SynchronousSimulatedECGA::initialize()
{
    Population &population = *this->population;
    doCache();

    individuals.resize(population_size);
    population.newIndividuals(individuals);
    initializer->initialize(individuals);

    offspring.resize(population_size);
    population.newIndividuals(offspring);

    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        auto &i = individuals[idx];
        evaluate_initial_solution(idx, i);
    }

    sim->simulator->simulate_until_end();
}
void SynchronousSimulatedECGA::sample_solution(size_t /* idx */, Individual ii)
{
    distribution->sample(ii);
}
void SynchronousSimulatedECGA::generation()
{
    if (terminated())
        throw stop_approach();

    size_t to_select = population_size;
    if (distribution->should_update())
    {
        auto subset = selection->select(individuals, to_select);
        distribution->learn(subset);
    }
    for (size_t idx = 0; idx < offspring.size(); ++idx)
    {
        auto &nii = offspring[idx];
        sample_solution(idx, nii);
        evaluate_solution(idx, nii);
    }

    sim->simulator->simulate_until_end();
}
void SynchronousSimulatedECGA::place_in_population(size_t idx,
                                                   const Individual ii,
                                                   const std::optional<int> override_replacement_strategy)
{
    archive->try_add(ii);
    int replacement_strategy_to_use = replacement_strategy;
    if (override_replacement_strategy.has_value())
        replacement_strategy_to_use = *override_replacement_strategy;

    switch (replacement_strategy_to_use)
    {
    case 0:
        replace_fifo(ii);
        break;
    case 1:
        replace_idx(idx, ii);
        break;
    case 2:
        replace_uniformly(ii);
        break;
    case 3:
        replace_selection_fifo(ii);
        break;
    case 4:
        replace_selection_idx(idx, ii);
        break;
    case 5:
        replace_selection_uniformly(ii);
        break;
    }
}
void SynchronousSimulatedECGA::evaluate_initial_solution(size_t idx, Individual ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    TimeSpent &ts = cache->tgts.getData(ii);
    ts.t = 0;
    std::optional<std::exception_ptr> maybe_exception;
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
        maybe_exception = std::current_exception();
    }

    // Wait until a processor is available.
    sim->simulator->simulate_until([this]() { return sim->num_workers_busy < sim->num_workers; });

    ++sim->num_workers_busy;
    sim->simulator->insert_event(
        std::make_unique<FunctionalSimResumable>(
            [idx, ii, maybe_exception, this](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Perform replacement
                place_in_population(idx, ii, 1);
                --sim->num_workers_busy;
                if (maybe_exception.has_value())
                    std::rethrow_exception(*maybe_exception);
            }),
        sim->simulator->now() + ts.t,
        "New solution on " + std::to_string(idx));
}
void SynchronousSimulatedECGA::evaluate_solution(size_t idx, Individual ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    TimeSpent &ts = cache->tgts.getData(ii);
    ts.t = 0;
    std::optional<std::exception_ptr> maybe_exception;
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
        maybe_exception = std::current_exception();
    }

    // Wait until a processor is available.
    sim->simulator->simulate_until([this]() { return sim->num_workers_busy < sim->num_workers; });

    ++sim->num_workers_busy;
    sim->simulator->insert_event(
        std::make_unique<FunctionalSimResumable>(
            [idx, ii, maybe_exception, this](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Perform replacement
                place_in_population(idx, ii, std::nullopt);
                --sim->num_workers_busy;
                if (maybe_exception.has_value())
                    std::rethrow_exception(*maybe_exception);
            }),
        sim->simulator->now() + ts.t,
        "New solution on " + std::to_string(idx));
}
void SynchronousSimulatedECGA::step()
{
    if (!initialized)
    {
        initialize();
        initialized = true;
    }
    else
    {
        generation();
    }
}
std::vector<Individual> &SynchronousSimulatedECGA::getSolutionPopulation()
{
    return individuals;
}
bool SynchronousSimulatedECGA::terminated()
{
    // Not converged if we haven't started yet.
    if (!initialized)
        return false;

    Population &pop = *population;
    auto tggc = pop.getDataContainer<GenotypeCategorical>();
    auto &r = tggc.getData(individuals[0]);

    // This is sufficient for synchronous!
    for (Individual ii : individuals)
    {
        auto &o = tggc.getData(ii);
        if (!std::equal(r.genotype.begin(), r.genotype.end(), o.genotype.begin()))
            return false;
    }
    return true;
}
void SynchronousSimulatedECGA::doCache()
{
    if (cache.has_value())
        return;
    Population &pop = *population;
    cache.emplace(
        Cache{pop.getDataContainer<Objective>(), pop.getDataContainer<TimeSpent>(), *pop.getGlobalData<Rng>()});
}
void SynchronousSimulatedECGA::replace_uniformly(Individual ii)
{
    Population &pop = *population;
    std::uniform_int_distribution<size_t> d_pop_idx(0, population_size - 1);
    size_t idx = d_pop_idx(cache->rng.rng);
    if (sl != NULL)
    {
        sl->end_span(idx, individuals[idx], 0);
        sl->start_span(idx, ii, 0);
    }
    pop.copyIndividual(ii, individuals[idx]);
}
void SynchronousSimulatedECGA::replace_selection_fifo(Individual ii)
{
    Population &pop = *population;
    size_t idx = current_replacement_index++;
    short c = (*perf_criterion)->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        if (sl != NULL)
        {
            sl->end_span(idx, individuals[idx], 0);
            sl->start_span(idx, ii, 0);
        }
        pop.copyIndividual(ii, individuals[idx]);
    }
    if (current_replacement_index >= population_size)
        current_replacement_index = 0;
}
void SynchronousSimulatedECGA::replace_selection_idx(size_t idx, Individual ii)
{
    Population &pop = *population;
    short c = (*perf_criterion)->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        if (sl != NULL)
        {
            sl->end_span(idx, individuals[idx], 0);
            sl->start_span(idx, ii, 0);
        }
        pop.copyIndividual(ii, individuals[idx]);
    }
}
void SynchronousSimulatedECGA::replace_selection_uniformly(Individual ii)
{
    Population &pop = *population;
    std::uniform_int_distribution<size_t> d_pop_idx(0, population_size - 1);
    size_t idx = d_pop_idx(cache->rng.rng);
    short c = (*perf_criterion)->compare(individuals[idx], ii);
    if ((c == 3 && replace_if_equal) || (c == 0 && replace_if_incomparable) || (c == 2))
    {
        if (sl != NULL)
        {
            sl->end_span(idx, individuals[idx], 0);
            sl->start_span(idx, ii, 0);
        }
        pop.copyIndividual(ii, individuals[idx]);
    }
}

AsynchronousSimulatedECGA::AsynchronousSimulatedECGA(int replacement_strategy,
                                                     std::shared_ptr<SimulatorParameters> sim,
                                                     size_t population_size,
                                                     std::shared_ptr<ISolutionSamplingDistribution> distribution,
                                                     std::shared_ptr<ISolutionInitializer> initializer,
                                                     std::shared_ptr<ISelection> selection,
                                                     std::shared_ptr<IArchive> archive,
                                                     std::shared_ptr<SpanLogger> sl) :
    SynchronousSimulatedECGA(replacement_strategy, sim, population_size, distribution, initializer, selection, archive, sl)
{
}
AsynchronousSimulatedECGA::AsynchronousSimulatedECGA(int replacement_strategy,
                                                     std::shared_ptr<IPerformanceCriterion> perf_criterion,
                                                     std::shared_ptr<SimulatorParameters> sim,
                                                     size_t population_size,
                                                     std::shared_ptr<ISolutionSamplingDistribution> distribution,
                                                     std::shared_ptr<ISolutionInitializer> initializer,
                                                     std::shared_ptr<ISelection> selection,
                                                     std::shared_ptr<IArchive> archive,
                                                     std::shared_ptr<SpanLogger> sl) :
    SynchronousSimulatedECGA(
        replacement_strategy, perf_criterion, sim, population_size, distribution, initializer, selection, archive, sl)
{
}
void AsynchronousSimulatedECGA::initialize()
{
    Population &population = *this->population;

    individuals.resize(population_size);
    population.newIndividuals(individuals);
    initializer->initialize(individuals);

    offspring.resize(population_size);
    population.newIndividuals(offspring);

    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        population.copyIndividual(individuals[idx], offspring[idx]);
        // Initial evaluations should always greedily replace the original individual
        // to simulate the actual completion of evaluation of this solution.
        evaluate_initial_solution(idx, offspring[idx]);
    }

    // sim->simulator->simulate_until_end();
}
void AsynchronousSimulatedECGA::generation()
{
    if (terminated())
        throw stop_approach();

    for (size_t num = 0; num < population_size; ++num)
    {
        // t_assert(sim->num_workers_busy <= sim->num_workers,
        //          "Number of busy workers should be upper bounded by the number of workers.");
        t_assert(sim->processing_queue.size() > 0 || sim->simulator->event_queue.size() > 0,
                 "Either queue length or processing queue should have items inside.");
        while (sim->processing_queue.size() > 0 && sim->num_workers_busy < sim->num_workers)
        {
            auto &[t_cost, resumable, msg] = sim->processing_queue.front();
            // Note: we do not have a descriptive message to use here...
            // Maybe provide one in the processing queue?
            sim->simulator->insert_event(std::move(resumable), sim->simulator->now() + t_cost, msg);
            sim->processing_queue.pop();
            ++sim->num_workers_busy;
        }
        // bool bounded_before_step = sim->num_workers_busy <= sim->num_workers;
        sim->simulator->step();
        // bool bounded_after_step = sim->num_workers_busy <= sim->num_workers;
        // t_assert(bounded_after_step | !bounded_before_step,
        //          "Number of busy workers should be upper bounded by the number of workers, yet a step broke this.");

    }
}
void AsynchronousSimulatedECGA::evaluate_initial_solution(size_t idx, Individual ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();
    doCache();

    TimeSpent &ts = cache->tgts.getData(ii);
    ts.t = 0;
    std::optional<std::exception_ptr> maybe_exception;
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
        maybe_exception = std::current_exception();
    }

    // Wait until a processor is available.
    sim->simulator->simulate_until([this]() { return sim->num_workers_busy < sim->num_workers; });

    ++sim->num_workers_busy;
    sim->simulator->insert_event(
        std::make_unique<FunctionalSimResumable>(
            [idx, ii, maybe_exception, this](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Perform replacement -- initial solutions for async should specifically replace their original index.
                // Hence the override here.
                place_in_population(idx, ii, 1);
                // Process has completed
                --sim->num_workers_busy;
                // Queue up next replacement for this index.
                sim->processing_queue.push(
                    std::make_tuple(0.0,
                                    std::make_unique<FunctionalSimResumable>(
                                        [this, idx](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                                            sample_and_evaluate_new_solution(idx);
                                        }),
                                    "New solution on " + std::to_string(idx)));

                if (maybe_exception.has_value())
                    std::rethrow_exception(*maybe_exception);
            }),
        sim->simulator->now() + ts.t,
        "Evaluate initial solution " + std::to_string(idx));
}
void AsynchronousSimulatedECGA::sample_solution(size_t /* idx */, Individual ii)
{
    // Sample new solution
    size_t to_select = population_size;
    if (distribution->should_update())
    {
        auto subset = selection->select(individuals, to_select);
        distribution->learn(subset);
    }
    distribution->sample(ii);
}
void AsynchronousSimulatedECGA::sample_and_evaluate_new_solution(size_t idx)
{
    Individual ii = offspring[idx];
    sample_solution(idx, ii);
    evaluate_solution(idx, ii);
}
bool AsynchronousSimulatedECGA::terminated()
{
    // Not converged if we haven't started yet.
    if (!initialized)
        return false;

    Population &pop = *population;
    auto tggc = pop.getDataContainer<GenotypeCategorical>();
    auto &r = tggc.getData(individuals[0]);
    // This is sufficient for synchronous, but not asynchronous:
    // there could still be a pending evaluation that unconverges the population.
    for (Individual ii : individuals)
    {
        auto &o = tggc.getData(ii);
        if (!std::equal(r.genotype.begin(), r.genotype.end(), o.genotype.begin()))
            return false;
    }
    for (Individual ii : offspring)
    {
        auto &o = tggc.getData(ii);
        // Note: Not all offspring may have been used yet, in which case they should be skipped.
        //       Not entirely sure if this occurs in practice, but better
        //       safe than sorry!
        if (o.genotype.size() != r.genotype.size())
            continue;
        if (!std::equal(r.genotype.begin(), r.genotype.end(), o.genotype.begin()))
            return false;
    }
    return true;
}
void AsynchronousSimulatedECGA::evaluate_solution(size_t idx, Individual ii)
{
    Population &pop = *population;
    GObjectiveFunction &objective_function = *pop.getGlobalData<GObjectiveFunction>();

    // Evaluate it & track time cost
    TimeSpent &ts = cache->tgts.getData(ii);
    ts.t = 0;
    std::optional<std::exception_ptr> maybe_exception;
    try
    {
        objective_function.of->evaluate(ii);
    }
    catch (std::exception &e)
    {
        maybe_exception = std::current_exception();
    }

    // Wait until a processor is available for replacement.
    // sim->simulator->simulate_until([this]() { return sim->num_workers_busy < sim->num_workers; });

    // Once a processor is available, start operating.
    // ++sim->num_workers_busy;
    sim->simulator->insert_event(
        std::make_unique<FunctionalSimResumable>(
            [maybe_exception, idx, ii, this](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                // Perform replacement in population.
                place_in_population(idx, ii, std::nullopt);
                // Free up the worker (simulated work has completed)
                --sim->num_workers_busy;
                sim->processing_queue.push(
                    std::make_tuple(0.0,
                                    std::make_unique<FunctionalSimResumable>(
                                        [this, idx](ISimulator &, double, std::unique_ptr<IResumableSimulated> &) {
                                            sample_and_evaluate_new_solution(idx);
                                        }),
                                    "New solution on " + std::to_string(idx)));

                if (maybe_exception.has_value())
                    std::rethrow_exception(*maybe_exception);
            }),
        sim->simulator->now() + ts.t,
        "Evaluate solution on " + std::to_string(idx));
}

SynchronousSimulatedKernelECGA::SynchronousSimulatedKernelECGA(
    int replacement_strategy,
    std::shared_ptr<SimulatorParameters> sim,
    size_t population_size,
    size_t neighborhood_size,
    std::shared_ptr<ISolutionSamplingDistribution> distribution,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<ISelection> selection,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<SpanLogger> sl) :
    SynchronousSimulatedECGA(replacement_strategy, sim, population_size, distribution, initializer, selection, archive, sl),
    neighborhood_size(neighborhood_size)
{
    initModels();
}
SynchronousSimulatedKernelECGA::SynchronousSimulatedKernelECGA(
    int replacement_strategy,
    std::shared_ptr<IPerformanceCriterion> perf_criterion,
    std::shared_ptr<SimulatorParameters> sim,
    size_t population_size,
    size_t neighborhood_size,
    std::shared_ptr<ISolutionSamplingDistribution> distribution,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<ISelection> selection,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<SpanLogger> sl) :
    SynchronousSimulatedECGA(
        replacement_strategy, perf_criterion, sim, population_size, distribution, initializer, selection, archive, sl),
    neighborhood_size(neighborhood_size)
{
    initModels();
}
void SynchronousSimulatedKernelECGA::initModels()
{
    per_solution_distribution.resize(population_size);
    for (size_t idx = 0; idx < population_size; ++idx)
    {
        per_solution_distribution[idx] = distribution->clone();
    }
}

std::vector<Individual> filterNeighborhoodKNN(TypedGetter<GenotypeCategorical> tggc, Individual reference, std::vector<Individual> &individuals, size_t neighborhood_size)
{
    std::vector<size_t> hamming_distances(individuals.size());
    auto &genotype_r = tggc.getData(reference);
    for (size_t idx = 0; idx < individuals.size(); ++idx)
    {
        auto &genotype_idx = tggc.getData(individuals[idx]);
        size_t min_l = std::min(genotype_r.genotype.size(), genotype_idx.genotype.size());
        size_t max_l = std::max(genotype_r.genotype.size(), genotype_idx.genotype.size());
        hamming_distances[idx] = hamming_distance(genotype_r.genotype.data(), genotype_idx.genotype.data(), min_l) + (max_l - min_l);
    }

    std::vector<size_t> indices(individuals.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::nth_element(indices.begin(), indices.begin() + static_cast<long>(neighborhood_size), indices.end());

    std::vector<Individual> result(neighborhood_size);
    for (size_t idx = 0; idx < neighborhood_size; ++idx)
    {
        result[idx] = individuals[indices[idx]];
    }
    return result;
}

void SynchronousSimulatedKernelECGA::sample_solution(size_t idx, Individual ii)
{
    // doCache();
    if (distribution->should_update())
    {
        Population &pop = *population;
        // Sample new solution
        size_t to_select = population_size;
        if (per_solution_distribution[idx]->should_update())
        {
            auto subselection = filterNeighborhoodKNN(pop.getDataContainer<GenotypeCategorical>(), ii, individuals, neighborhood_size);
            auto subset = selection->select(subselection, to_select);
            per_solution_distribution[idx]->learn(subset);
        }
    }
    per_solution_distribution[idx]->sample(ii);
}

void AsynchronousSimulatedKernelECGA::initModels()
{
    per_solution_distribution.resize(population_size);
    for (size_t idx = 0; idx < population_size; ++idx)
    {
        per_solution_distribution[idx] = distribution->clone();
    }
}
AsynchronousSimulatedKernelECGA::AsynchronousSimulatedKernelECGA(
    int replacement_strategy,
    std::shared_ptr<SimulatorParameters> sim,
    size_t population_size,
    size_t neighborhood_size,
    std::shared_ptr<ISolutionSamplingDistribution> distribution,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<ISelection> selection,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<SpanLogger> sl) :
    AsynchronousSimulatedECGA(
        replacement_strategy, sim, population_size, distribution, initializer, selection, archive, sl),
    neighborhood_size(neighborhood_size)
{
    initModels();
}
AsynchronousSimulatedKernelECGA::AsynchronousSimulatedKernelECGA(
    int replacement_strategy,
    std::shared_ptr<IPerformanceCriterion> perf_criterion,
    std::shared_ptr<SimulatorParameters> sim,
    size_t population_size,
    size_t neighborhood_size,
    std::shared_ptr<ISolutionSamplingDistribution> distribution,
    std::shared_ptr<ISolutionInitializer> initializer,
    std::shared_ptr<ISelection> selection,
    std::shared_ptr<IArchive> archive,
    std::shared_ptr<SpanLogger> sl) :
    AsynchronousSimulatedECGA(
        replacement_strategy, perf_criterion, sim, population_size, distribution, initializer, selection, archive, sl),
    neighborhood_size(neighborhood_size)
{
    initModels();
}
void AsynchronousSimulatedKernelECGA::sample_solution(size_t idx, Individual ii)
{
    // doCache();
    if (distribution->should_update())
    {
        Population &pop = *population;
        // Sample new solution
        size_t to_select = population_size;
        if (per_solution_distribution[idx]->should_update())
        {
            auto subselection = filterNeighborhoodKNN(pop.getDataContainer<GenotypeCategorical>(), ii, individuals, neighborhood_size);
            auto subset = selection->select(subselection, to_select);
            per_solution_distribution[idx]->learn(subset);
        }
    }
    per_solution_distribution[idx]->sample(ii);
}