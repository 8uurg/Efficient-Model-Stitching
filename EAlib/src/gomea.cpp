#include "gomea.hpp"
#include "acceptation_criteria.hpp"
#include "archive.hpp"
#include "base.hpp"
#include "cppassert.h"
#include "utilities.hpp"
#include "scalarize.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <unordered_map>

std::optional<double> LinkageMetric::filter_minimum_threshold()
{
    return {};
}
std::optional<double> LinkageMetric::filter_maximum_threshold()
{
    return {};
}

Matrix<size_t> estimate_bivariate_counts(size_t v_a,
                                         char alphabet_size_v_a,
                                         size_t v_b,
                                         char alphabet_size_v_b,
                                         TypedGetter<GenotypeCategorical> &gg,
                                         std::vector<Individual> &individuals)
{

    Matrix<size_t> counts(0, alphabet_size_v_a, alphabet_size_v_b);

    for (Individual ii : individuals)
    {
        GenotypeCategorical &genotype = gg.getData(ii);
        counts[{genotype.genotype[v_b], genotype.genotype[v_a]}] += 1;
    }

    return counts;
}

struct Entropies
{
    double joint;
    double v_a;
    double v_b;
};

double entropy_part(double p)
{
    if (p <= 0.0 || p >= 1.0)
        return 0.0;
    return -p * std::log2(p);
}

Entropies estimate_bivariate_and_univariate_entropies(size_t v_a,
                                                      char alphabet_size_v_a,
                                                      size_t v_b,
                                                      char alphabet_size_v_b,
                                                      TypedGetter<GenotypeCategorical> &gg,
                                                      std::vector<Individual> &individuals)
{
    Matrix<size_t> bivariate_counts =
        estimate_bivariate_counts(v_a, alphabet_size_v_a, v_b, alphabet_size_v_b, gg, individuals);
    size_t total = individuals.size();
    double HBI = 0.0;
    double HUv_a = 0.0;
    double HUv_b = 0.0;

    // HBI
    for (size_t row = 0; row < bivariate_counts.getHeight(); ++row)
    {
        for (size_t col = 0; col < bivariate_counts.getWidth(); ++col)
        {
            double p = static_cast<double>(bivariate_counts[{row, col}]) / static_cast<double>(total);
            HBI += entropy_part(p);
        }
    }

    // HUv_a
    for (size_t col = 0; col < bivariate_counts.getWidth(); ++col)
    {
        size_t count = 0;
        for (size_t row = 0; row < bivariate_counts.getHeight(); ++row)
        {
            count += bivariate_counts[{row, col}];
        }
        double p = static_cast<double>(count) / static_cast<double>(total);
        HUv_a += entropy_part(p);
    }

    // HUv_b
    for (size_t row = 0; row < bivariate_counts.getHeight(); ++row)
    {
        size_t count = 0;
        for (size_t col = 0; col < bivariate_counts.getWidth(); ++col)
        {
            count += bivariate_counts[{row, col}];
        }
        double p = static_cast<double>(count) / static_cast<double>(total);
        HUv_b += entropy_part(p);
    }

    return Entropies{
        HBI,
        HUv_a,
        HUv_b,
    };
}

double MI::compute_linkage(size_t v_a, size_t v_b, std::vector<Individual> &individuals)
{
    if (!cache.has_value())
    {
        auto genotype_categorical_getter = (*population).getDataContainer<GenotypeCategorical>();
        cache.emplace(
            Cache{genotype_categorical_getter, (*population).getGlobalData<GenotypeCategoricalData>()->alphabet_size});
    }
    Entropies entropies = estimate_bivariate_and_univariate_entropies(v_a,
                                                                      cache.value().alphabet_size[v_a],
                                                                      v_b,
                                                                      cache.value().alphabet_size[v_b],
                                                                      cache.value().genotype_categorical,
                                                                      individuals);

    // Actually compute MI using entropies.
    return entropies.v_a + entropies.v_b - entropies.joint;
}

double NMI::compute_linkage(size_t v_a, size_t v_b, std::vector<Individual> &individuals)
{
    if (!cache.has_value())
    {
        cache.emplace(Cache{(*population).getDataContainer<GenotypeCategorical>(),
                            (*population).getGlobalData<GenotypeCategoricalData>()->alphabet_size});
    }
    Entropies entropies = estimate_bivariate_and_univariate_entropies(v_a,
                                                                      cache.value().alphabet_size[v_a],
                                                                      v_b,
                                                                      cache.value().alphabet_size[v_b],
                                                                      cache.value().genotype_categorical,
                                                                      individuals);

    const double eps = 1e-10;
    // Edge case for two converged variables.
    if (entropies.joint <= eps)
        return 0.0;

    // Actually compute MI using entropies.
    double separate = entropies.v_a + entropies.v_b;
    return separate / entropies.joint - 1;
}

// Random Linkage

double RandomLinkage::compute_linkage(size_t, size_t, std::vector<Individual> &)
{
    Population &population = *(this->population);
    Rng &rng = *population.getGlobalData<Rng>();

    std::uniform_real_distribution<double> p(0.0, 1.0);
    return p(rng.rng);
}

void RandomLinkage::afterRegisterData()
{
    Population &population = *(this->population);
    t_assert(population.isGlobalRegistered<Rng>(), "Random Linkage requires a Random Number generator.");
}

// Fixed Linkage

FixedLinkage::FixedLinkage(SymMatrix<double> linkage,
                           std::optional<double> minimum_threshold,
                           std::optional<double> maximum_threshold) :
    linkage(linkage), minimum_threshold(minimum_threshold), maximum_threshold(maximum_threshold)
{
}
double FixedLinkage::compute_linkage(size_t v_a, size_t v_b, std::vector<Individual> &)
{
    return linkage.get(v_a, v_b);
}
std::optional<double> FixedLinkage::filter_minimum_threshold()
{
    return minimum_threshold;
}
std::optional<double> FixedLinkage::filter_maximum_threshold()
{
    return maximum_threshold;
}

double mergeUPGMA(
    double /* distance_ij */, double distance_ik, double distance_jk, size_t size_i, size_t size_j, size_t /* size_k */)
{
    double weighted = static_cast<double>(size_i) * distance_ik + static_cast<double>(size_j) * distance_jk;
    return weighted / static_cast<double>(size_i + size_j);
}

std::vector<TreeNode> performHierarchicalClustering(SymMatrix<double> linkage, Rng &rng)
{
    std::vector<TreeNode> merges;
    size_t n = linkage.getSize();
    // Every merge reduces the number of elements left by one.
    // As such there are n - 1 such merges to end up at the root.
    merges.reserve(n - 1);

    // The algorithm implemented here is named NN-chain, or nearest-neighbor chain.
    // And is a fast O(n^2) hierarchical clustering algorithm.

    // The first important implementation detail here is that we use representatives,
    // as we are merging nodes, the in the current state each variable only appears once.
    // This results in each variable uniquely mapping to a subset at a point in time.
    // As such we choose to represent each subset by its smallest element contained within.
    // Annoyingly enough, this does mean that translating the output requires some care.

    // The chain is initially empty, first element will be picked randomly as well.
    std::vector<size_t> nn_chain;
    nn_chain.reserve(n);

    // We start off with the univariate marginal product
    // -- i.e. all variables are in subsets on their own.
    std::vector<size_t> node_sizes(n);
    std::fill(node_sizes.begin(), node_sizes.end(), 1);
    // - keep track of unique indices as well so we can output a structure similar to that provided by scipy.
    std::vector<size_t> indices(n);
    std::iota(indices.begin(), indices.end(), 0);
    // and all of them are remaining.
    std::vector<size_t> remaining(n);
    std::iota(remaining.begin(), remaining.end(), 0);

    // Another implementation detail, important for use in Evolutionary Algorithms, is that ties should be
    // broken randomly, as this results in variation in the tree when the metric used is fixed and ties occur.
    // i.e. due to convergence or coincidence. Furthermore: we do not want to favor one pair over another.
    // dependent on the situation.
    std::shuffle(remaining.begin(), remaining.end(), rng.rng);

    // A common operation in this algorithm is finding the nearest neighbor of the current end of the chain
    // to the remaining elements.
    //  (Sidenote: in our case this is the farthest element, as we are working with similarities, not distances)
    auto next = [&nn_chain, &remaining, &linkage]() {
        size_t idx = 0;
        double s = -std::numeric_limits<double>::infinity();
        size_t leader = nn_chain.back();
        for (size_t remaining_idx = 0; remaining_idx < remaining.size(); ++remaining_idx)
        {
            size_t other = remaining[remaining_idx];
            if (linkage[{other, leader}] > s)
            {
                s = linkage[{other, leader}];
                idx = remaining_idx;
            }
        }
        return idx;
    };

    // another aspect than can be modified is the update of distances. in this case we use UPGMA.
    auto merged_distance = [&linkage, &node_sizes](size_t to_merge_i, size_t to_merge_j, size_t k) {
        size_t size_i = node_sizes[to_merge_i];
        size_t size_j = node_sizes[to_merge_j];
        size_t size_k = node_sizes[k];
        return mergeUPGMA(linkage[{to_merge_i, to_merge_j}],
                          linkage[{to_merge_i, k}],
                          linkage[{to_merge_j, k}],
                          size_i,
                          size_j,
                          size_k);
    };

    // What is the next index to use after merging?
    size_t next_merge_index = n;

    // While there is more than one subset remaining.
    while (remaining.size() > 1 || nn_chain.size() > 0)
    {
        if (nn_chain.size() == 0)
        {
            // To get started, pick the last index of the remaining list.
            // This is random, 'remaining' should be shuffled!
            size_t leader = remaining.back();
            remaining.pop_back();
            nn_chain.push_back(leader);
        }
        if (nn_chain.size() == 1)
        {
            // With only one element, the next one in the chain is trivial
            size_t remaining_idx = next();
            size_t leader = remaining[remaining_idx];
            // Quickly remove the new item from remaining.
            std::swap(remaining[remaining_idx], remaining.back());
            remaining.pop_back();
            nn_chain.push_back(leader);
        }

        // In all other cases we need to check if we are closer to the previous element than
        // the next remaining element. If we are: we merge, otherwise the next item is added
        // to the chain.

        size_t leader = nn_chain.back();
        size_t previous = nn_chain[nn_chain.size() - 2];
        double distance_previous = linkage[{leader, previous}];

        // If there are no elements remaining: just merge!
        // this can occur if the tree is more like a list and we started exactly in the wrong node.
        // or if we are merging the last two nodes.
        if (remaining.size() != 0)
        {
            size_t remaining_idx = next();
            size_t next_remaining = remaining[remaining_idx];
            double distance_next_remaining = linkage[{leader, next_remaining}];

            // comparison is flipped here as we are working with similarities instead of disimilarities.
            // normally it would be `distance_next_remaining < distance_previous`
            if (distance_next_remaining > distance_previous)
            {
                // next element is closer, add it to the chain.
                // Quickly remove the new item from remaining.
                std::swap(remaining[remaining_idx], remaining.back());
                remaining.pop_back();
                nn_chain.push_back(next_remaining);
                // and start back from the top -- technically going back to line 193 would work better
                // as the if statements are always false from this point onwards, unless a merge was performed.
                continue;
            }
        }

        // we have found a pair of mutual nearest neighbors: leader and previous.
        // now we need to merge them!
        // Determine the representative.
        size_t representative = std::min(leader, previous);
        size_t not_representative = std::max(leader, previous);

        // Update distances within the chain.
        for (size_t i = 0; i < nn_chain.size() - 2; ++i)
        {
            size_t other = nn_chain[i];
            linkage[{representative, other}] = merged_distance(leader, previous, other);
        }
        // Update distances for those remaining.
        for (size_t i = 0; i < remaining.size(); ++i)
        {
            size_t other = remaining[i];
            linkage[{representative, other}] = merged_distance(leader, previous, other);
        }

        // Update the size of the resultant node.
        node_sizes[representative] += node_sizes[not_representative];

        // Keep track of the merges
        merges.push_back(TreeNode{
            /* .left = */ indices[representative],
            /* .right = */ indices[not_representative],
            /* .distance = */ distance_previous,
            /* .size =  */ node_sizes[representative],
        });

        // get a merge index for this element.
        indices[representative] = next_merge_index++;

        // Remove the last two items
        nn_chain.pop_back();
        nn_chain.pop_back();

        // add the current element back to remaining.
        remaining.push_back(representative);
        // and shuffle this element as well
        std::uniform_int_distribution<size_t> idx(0, remaining.size() - 1);
        std::swap(remaining[idx(rng.rng)], remaining.back());
    }

    return merges;
}

std::vector<std::vector<size_t>> &CategoricalUnivariateFoS::getFoS()
{
    return fos;
}

void CategoricalUnivariateFoS::learnFoS(std::vector<Individual> &)
{
    Population &population = *(this->population);
    GenotypeCategoricalData &genotype_data = *population.getGlobalData<GenotypeCategoricalData>();

    size_t l = genotype_data.l;
    fos.resize(l);
    for (size_t i = 0; i < l; ++i)
    {
        fos[i].resize(1);
        fos[i][0] = i;
    }
}

void CategoricalUnivariateFoS::afterRegisterData()
{
    Population &population = *(this->population);
    t_assert(population.isGlobalRegistered<GenotypeCategoricalData>(),
             "Global Categorical Genotype data (i.e. string length) should be registered.");
}

FoSLearner *CategoricalUnivariateFoS::cloned_ptr()
{
    return new CategoricalUnivariateFoS(*this);
}

CategoricalLinkageTree::CategoricalLinkageTree(std::shared_ptr<LinkageMetric> metric,
                                               FoSOrdering ordering,
                                               bool filter_zeros,
                                               bool filter_maxima,
                                               bool filter_root) :
    metric(metric),
    ordering(ordering),
    filter_minima(filter_zeros),
    filter_maxima(filter_maxima),
    filter_root(filter_root)
{
}
void CategoricalLinkageTree::learnFoS(std::vector<Individual> &individuals)
{
    Population &population = *(this->population);
    GenotypeCategoricalData &genotype_data = *population.getGlobalData<GenotypeCategoricalData>();
    Rng &rng = *population.getGlobalData<Rng>();

    size_t l = genotype_data.l;
    // Compute the linkage matrix using the provided metric.
    SymMatrix<double> linkage_matrix(0, l);
    for (size_t i = 0; i < l; ++i)
    {
        for (size_t j = i + 1; j < l; ++j)
        {
            linkage_matrix.set(i, j, metric->compute_linkage(i, j, individuals));
        }
    }
    // Compute the corresponding hierarchical clustering
    std::vector<TreeNode> merges = performHierarchicalClustering(linkage_matrix, rng);

    fos.resize(2 * l - 1);
    // Initialize to univariate model
    for (size_t i = 0; i < l; ++i)
    {
        fos[i].resize(1);
        fos[i][0] = i;
    }
    // Turn merges into the remaining FoS elements.
    for (size_t i = 0; i < merges.size(); ++i)
    {
        fos[i + l].resize(0);

        if (filter_minima)
        {
            std::optional<double> maybe_threshold = metric->filter_minimum_threshold();
            t_assert(maybe_threshold.has_value(),
                     "minimum filter can only be applied if the required thresholds are specified.");
            double threshold = *maybe_threshold;
            if (merges[i].distance <= threshold)
                continue;
        }

        size_t left = merges[i].left;
        size_t right = merges[i].right;
        fos[i + l].insert(fos[i + l].end(), fos[left].begin(), fos[left].end());
        fos[i + l].insert(fos[i + l].end(), fos[right].begin(), fos[right].end());

        if (filter_maxima)
        {
            std::optional<double> maybe_threshold = metric->filter_maximum_threshold();
            t_assert(maybe_threshold.has_value(),
                     "maximum filter can only be applied if the required thresholds are specified.");
            double threshold = *maybe_threshold;

            if (merges[i].distance >= threshold)
            {
                fos[left].resize(0);
                fos[right].resize(0);
            }
        }
    }

    if (filter_root)
    {
        fos.back().resize(0);
    }

    // Change up the ordering
    switch (ordering)
    {
    case AsIs:
        break;
    case Random:
        std::shuffle(fos.begin(), fos.end(), rng.rng);
        break;
    case SizeIncreasing:
        std::sort(fos.begin(), fos.end(), [](auto &a, auto &b) { return a.size() < b.size(); });
        break;
    }

    // Remove empty subsets
    auto new_end = std::remove_if(fos.begin(), fos.end(), [](auto &a) { return a.size() == 0; });
    fos.erase(new_end, fos.end());
}
std::vector<std::vector<size_t>> &CategoricalLinkageTree::getFoS()
{
    if (ordering == Random)
    {
        Population &population = *this->population;
        Rng &rng = *population.getGlobalData<Rng>();
        std::shuffle(fos.begin(), fos.end(), rng.rng);
    }

    return fos;
}
void CategoricalLinkageTree::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    metric->setPopulation(population);
}

FoSLearner *CategoricalLinkageTree::cloned_ptr()
{
    return new CategoricalLinkageTree(*this);
}

void IIncrementalImprovementOperator::evaluate_change(Individual current,
                                                      Individual /* backup */,
                                                      std::vector<size_t> & /* elements_changed */)
{
    Population &population = *this->population;
    //
    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();
    objective_function.of->evaluate(current);

    // TODO: Add partial evaluations here when supported!
}

CategoricalPopulationSamplingDistribution::CategoricalPopulationSamplingDistribution(Population &population,
                                                                                     std::vector<Individual> pool) :
    population(population), pool(pool)
{
}
bool CategoricalPopulationSamplingDistribution::apply_resample(Individual current, std::vector<size_t> &subset)
{
    if (!cache.has_value())
    {
        auto gg = population.getDataContainer<GenotypeCategorical>();
        Rng &rng = *population.getGlobalData<Rng>();
        cache.emplace(Cache{rng, gg});
    }

    if (pool.size() == 0)
        return false;

    Rng &rng = cache->rng;
    GenotypeCategorical &genotype_current = cache->ggc.getData(current);

    std::uniform_int_distribution<size_t> idx(0, pool.size() - 1);
    Individual donor = pool[idx(rng.rng)];
    GenotypeCategorical &genotype_donor = cache->ggc.getData(donor);

    bool changed_any = false;

    for (size_t idx : subset)
    {
        if (genotype_current.genotype[idx] != genotype_donor.genotype[idx])
            changed_any = true;
        genotype_current.genotype[idx] = genotype_donor.genotype[idx];
    }

    return changed_any;
}

CategoricalDonorSearchDistribution::CategoricalDonorSearchDistribution(Population &population,
                                                                       std::vector<Individual> pool) :
    population(population), pool(pool)
{
}
bool CategoricalDonorSearchDistribution::apply_resample(Individual current, std::vector<size_t> &subset)
{
    if (!cache.has_value())
    {
        auto gg = population.getDataContainer<GenotypeCategorical>();
        Rng &rng = *population.getGlobalData<Rng>();
        cache.emplace(Cache{rng, gg});
    }

    std::vector<size_t> indices(pool.size());
    std::iota(indices.begin(), indices.end(), 0);

    Rng &rng = cache->rng;
    GenotypeCategorical &genotype_current = cache->ggc.getData(current);

    bool changed_any = false;
    size_t indices_idx = 0;

    while (!changed_any && indices_idx < pool.size())
    {
        std::uniform_int_distribution<size_t> idx(indices_idx, pool.size() - 1);
        std::swap(indices[indices_idx], indices[idx(rng.rng)]);

        Individual donor = pool[indices[indices_idx]];
        GenotypeCategorical &genotype_donor = cache->ggc.getData(donor);

        for (size_t idx : subset)
        {
            if (genotype_current.genotype[idx] != genotype_donor.genotype[idx])
                changed_any = true;
            genotype_current.genotype[idx] = genotype_donor.genotype[idx];
        }
        ++indices_idx;
    }

    return changed_any;
}

CategoricalStoredSamplingDistribution::CategoricalStoredSamplingDistribution(Population &population,
                                                                                     std::vector<Individual> pool) :
    population(population)
{
    auto gg = population.getDataContainer<GenotypeCategorical>();
    Rng &rng = *population.getGlobalData<Rng>();
    cache.emplace(Cache{rng, gg});

    // Copy & store the genotypes.
    for (Individual &i: pool)
    {
        auto &d = gg.getData(i);
        this->pool.push_back(d.genotype);
    }
}
bool CategoricalStoredSamplingDistribution::apply_resample(Individual current, std::vector<size_t> &subset)
{
    if (!cache.has_value())
    {
        auto gg = population.getDataContainer<GenotypeCategorical>();
        Rng &rng = *population.getGlobalData<Rng>();
        cache.emplace(Cache{rng, gg});
    }

    if (pool.size() == 0)
        return false;

    Rng &rng = cache->rng;
    GenotypeCategorical &genotype_current = cache->ggc.getData(current);

    std::uniform_int_distribution<size_t> idx(0, pool.size() - 1);
    std::vector<char> &genotype_donor = pool[idx(rng.rng)];

    bool changed_any = false;

    for (size_t idx : subset)
    {
        if (genotype_current.genotype[idx] != genotype_donor[idx])
            changed_any = true;
        genotype_current.genotype[idx] = genotype_donor[idx];
    }

    return changed_any;
}

CategoricalStoredDonorSearchDistribution::CategoricalStoredDonorSearchDistribution(Population &population,
                                                                       std::vector<Individual> pool) :
    population(population)
{
    auto gg = population.getDataContainer<GenotypeCategorical>();
    Rng &rng = *population.getGlobalData<Rng>();
    cache.emplace(Cache{rng, gg});

    // Copy & store the genotypes.
    for (Individual &i: pool)
    {
        auto &d = gg.getData(i);
        this->pool.push_back(d.genotype);
    }
}
bool CategoricalStoredDonorSearchDistribution::apply_resample(Individual current, std::vector<size_t> &subset)
{
    if (!cache.has_value())
    {
        auto gg = population.getDataContainer<GenotypeCategorical>();
        Rng &rng = *population.getGlobalData<Rng>();
        cache.emplace(Cache{rng, gg});
    }

    std::vector<size_t> indices(pool.size());
    std::iota(indices.begin(), indices.end(), 0);

    Rng &rng = cache->rng;
    GenotypeCategorical &genotype_current = cache->ggc.getData(current);

    bool changed_any = false;
    size_t indices_idx = 0;

    while (!changed_any && indices_idx < pool.size())
    {
        std::uniform_int_distribution<size_t> idx(indices_idx, pool.size() - 1);
        std::swap(indices[indices_idx], indices[idx(rng.rng)]);

        std::vector<char> &genotype_donor = pool[indices[indices_idx]];

        for (size_t idx : subset)
        {
            if (genotype_current.genotype[idx] != genotype_donor[idx])
                changed_any = true;
            genotype_current.genotype[idx] = genotype_donor[idx];
        }
        ++indices_idx;
    }

    return changed_any;
}


void GOM::apply(Individual current,
                FoS &fos,
                ISamplingDistribution *distribution,
                IPerformanceCriterion *acceptance_criterion,
                bool &changed,
                bool &improved)
{
    Population &population = *this->population;

    Individual backup = population.newIndividual();
    population.copyIndividual(current, backup);

    changed = false;
    improved = false;

    for (FoSElement &e : fos)
    {
        bool sampling_changed = distribution->apply_resample(current, e);
        if (!sampling_changed)
            continue;

        evaluate_change(current, backup, e);

        short performance_judgement = acceptance_criterion->compare(backup, current);
        if (performance_judgement == 1)
        {
            // Backup is better than current, change made the solution worse, revert.
            population.copyIndividual(backup, current);
        }
        else
        {
            // Solution is improved by change. Update backup.
            population.copyIndividual(current, backup);
            changed = true;
            if (performance_judgement == 2)
                improved = true;
        }
    }

    population.dropIndividual(backup);
}
void FI::apply(Individual current,
               FoS &fos,
               ISamplingDistribution *distribution,
               IPerformanceCriterion *acceptance_criterion,
               bool &changed,
               bool &improved)
{
    Population &population = *this->population;

    Individual backup = population.newIndividual();
    population.copyIndividual(current, backup);

    changed = false;
    improved = false;

    for (FoSElement &e : fos)
    {
        bool sampling_changed = distribution->apply_resample(current, e);
        if (!sampling_changed)
            continue;

        evaluate_change(current, backup, e);

        short performance_judgement = acceptance_criterion->compare(backup, current);
        if (performance_judgement == 2)
        {
            // Solution is improved by change, FI is done!
            changed = true;
            if (performance_judgement == 2)
                improved = true;

            population.dropIndividual(backup);
            return;
        }
        else
        {
            // Backup is better than or equal to current, revert.
            population.copyIndividual(backup, current);
        }
    }

    population.dropIndividual(backup);
}

std::vector<ObjectiveCluster> cluster_mo_gomea(Population &pop,
                                               std::vector<Individual> &pool,
                                               std::vector<size_t> objective_indices,
                                               std::vector<double> objective_ranges,
                                               size_t number_of_clusters)
{
    t_assert(objective_indices.size() > 0, "should provide at least one index");
    t_assert(objective_ranges.size() == objective_indices.size(),
             "ranges should be provided for all objective indices");

    if (number_of_clusters <= 1)
    {
        // no clustering necessary.
        return {ObjectiveCluster{{}, // TODO: Compute mean.
                                 pool}};
    }

    // First perform greedy subset scattering to get some initial leaders.
    // - Provided first point is the best solution for one arbitrary objective
    //   (note: objective indices are assumed to be shuffled!)
    size_t initial = 0;
    double best_obj_value = std::numeric_limits<double>::infinity();
    for (size_t ii = 0; ii < pool.size(); ++ii)
    {
        Objective &o = pop.getData<Objective>(pool[ii]);
        size_t oi = objective_indices[0];

        // Unevaluated - skip.
        if (o.objectives.size() <= oi) {
            continue;
        }
        if (best_obj_value > o.objectives[oi])
        {
            best_obj_value = o.objectives[oi];
            initial = ii;
        }
    }

    // - Get selected leaders
    auto distance = [&pop, &pool, &objective_indices, &objective_ranges](int a, int b) {
        Objective &o_a = pop.getData<Objective>(pool[a]);
        Objective &o_b = pop.getData<Objective>(pool[b]);
        double d = 0.0;
        for (size_t oi = 0; oi < objective_indices.size(); ++oi)
        {
            size_t o = objective_indices[oi];
            // How do we deal with an unevaluated solution?
            // For now - skip dimension, we cannot cluster this solution properly.
            // By skipping distance will be 0 - and it will not be picked as a leader
            // of a cluster. All unevaluated solutions will currently be assigned to
            // a single identical cluster.
            bool is_a_unevaluated = o_a.objectives.size() <= o;
            bool is_b_unevaluated = o_b.objectives.size() <= o;
            if (is_a_unevaluated || is_b_unevaluated) continue;

            double o_a_o = o_a.objectives[o];
            double o_b_o = o_b.objectives[o];

            d += std::pow((o_a_o - o_b_o) / objective_ranges[oi], 2);
        }

        return std::sqrt(d);
    };

    std::vector<size_t> leaders = greedyScatteredSubsetSelection(distance, pool.size(), number_of_clusters, initial);
    // - And use these to determine the cluster centers
    Matrix<double> cluster_centers(0.0, objective_indices.size(), number_of_clusters);
    for (size_t r = 0; r < number_of_clusters; ++r)
    {
        Objective &o_r = pop.getData<Objective>(pool[leaders[r]]);
        for (size_t oi = 0; oi < objective_indices.size(); ++oi)
        {
            size_t o = objective_indices[oi];
            if (o_r.objectives.size() > o)
            {
                cluster_centers[{r, oi}] = o_r.objectives[o] / objective_ranges[oi];
            }
            else
            {
                cluster_centers[{r, oi}] = std::numeric_limits<double>::infinity();
            }
        }
    }

    // Perform k-means iterations.
    double epsilon = std::numeric_limits<double>::max();
    while (epsilon > 1e-10)
    {
        // Initialize some output matrices & tables
        std::vector<size_t> cluster_sizes(number_of_clusters);
        std::fill(cluster_sizes.begin(), cluster_sizes.end(), 0);
        Matrix<double> cluster_centers_new(0.0, objective_indices.size(), number_of_clusters);

        // Add each item to their nearest cluster.
        for (size_t i = 0; i < pool.size(); ++i)
        {
            // Find the nearest cluster
            Objective &o_i = pop.getData<Objective>(pool[i]);
            size_t nearest_cluster_index = 0;
            double d_min = std::numeric_limits<double>::infinity();
            for (size_t ci = 0; ci < number_of_clusters; ++ci)
            {
                double d = 0.0;
                for (size_t oi = 0; oi < objective_indices.size(); ++oi)
                {
                    size_t o = objective_indices[oi];
                    if (o_i.objectives.size() <= o) continue;
                    d += std::pow((cluster_centers[{ci, oi}] - o_i.objectives[o] / objective_ranges[oi]), 2);
                }
                if (d_min > d)
                {
                    nearest_cluster_index = ci;
                    d_min = d;
                }
            }
            // Add to nearest cluster
            for (size_t oi = 0; oi < objective_indices.size(); ++oi)
            {
                size_t o = objective_indices[oi];
                if (o_i.objectives.size() <= o) continue;
                cluster_centers_new[{nearest_cluster_index, oi}] += o_i.objectives[o] / objective_ranges[oi];
            }
            cluster_sizes[nearest_cluster_index] += 1;
        }

        // Compute new cluster centers & epsilon
        epsilon = 0.0;
        for (size_t ci = 0; ci < number_of_clusters; ++ci)
        {
            double c_epsilon = 0.0;
            for (size_t oi = 0; oi < objective_indices.size(); ++oi)
            {
                cluster_centers_new[{ci, oi}] = cluster_centers_new[{ci, oi}] / static_cast<double>(cluster_sizes[ci]);
                c_epsilon += std::pow(cluster_centers_new[{ci, oi}] - cluster_centers[{ci, oi}], 2);
            }
            epsilon += std::sqrt(c_epsilon);
        }

        // Swap out the cluster centers
        cluster_centers.swap(cluster_centers_new);
    }
    Rng &rng = *pop.getGlobalData<Rng>();

    // After this procedure MO-GOMEA's clustering approach turns the points into balanced overlapping clusters.
    size_t size_of_one_cluster = (2 * pool.size()) / number_of_clusters;
    std::vector<ObjectiveCluster> result;
    result.reserve(number_of_clusters);
    for (size_t ci = 0; ci < number_of_clusters; ++ci)
    {
        std::vector<size_t> indices(pool.size());
        std::iota(indices.begin(), indices.end(), 0);
        // Compute distances
        std::vector<double> distances(pool.size());
        for (size_t i = 0; i < pool.size(); ++i)
        {
            Objective &o_i = pop.getData<Objective>(pool[i]);
            double d = 0.0;
            for (size_t oi = 0; oi < objective_indices.size(); ++oi)
            {
                size_t o = objective_indices[oi];
                if (o_i.objectives.size() <= o) continue;
                d += std::pow((cluster_centers[{ci, oi}] - o_i.objectives[o] / objective_ranges[oi]), 2);
            }
            distances[i] = std::sqrt(d);
        }
        // Shuffle first!
        std::shuffle(indices.begin(), indices.end(), rng.rng);
        // Get the `size_of_one_cluster` nearest elements
        std::nth_element(indices.begin(),
                         indices.begin() + static_cast<long>(size_of_one_cluster),
                         indices.end(),
                         [&distances](int a, int b) { return distances[a] < distances[b]; });
        indices.erase(indices.begin() + static_cast<long>(size_of_one_cluster), indices.end());
        // Compute a new cluster center
        std::vector<double> center(objective_indices.size());
        std::fill(center.begin(), center.end(), 0.0);
        for (size_t ii = 0; ii < indices.size(); ++ii)
        {
            size_t i = indices[ii];
            Objective &o_i = pop.getData<Objective>(pool[i]);
            for (size_t oi = 0; oi < objective_indices.size(); ++oi)
            {
                size_t o = objective_indices[oi];
                if (o_i.objectives.size() <= o) continue;
                center[oi] += o_i.objectives[o] / objective_ranges[oi];
            }
        }
        for (size_t oi = 0; oi < objective_indices.size(); ++oi)
        {
            center[oi] = center[oi] / static_cast<double>(size_of_one_cluster);
        }
        std::vector<Individual> cluster_contents(indices.size());
        for (size_t i = 0; i < indices.size(); ++i)
            cluster_contents[i] = pool[indices[i]];
        result.push_back({center, cluster_contents});
    }
    return result;
}

void ensure_valid_ranges(std::vector<double> &objective_ranges)
{
    for (size_t p = 0; p < objective_ranges.size(); ++p)
    {
        if (!std::isfinite(objective_ranges[p]) || objective_ranges[p] == 0.0)
        {
            objective_ranges[p] = 1.0;
        }
    }
}

void determine_extreme_clusters(std::vector<size_t> &objective_indices, std::vector<ObjectiveCluster> &clusters)
{
    // Determine cluster's objective mode
    size_t num_cluster_objectives = clusters[0].centroid.size();
    for (size_t o = 0; o < num_cluster_objectives; ++o)
    {
        size_t extreme_c_for_o = 0;
        double obj = std::numeric_limits<double>::infinity();
        for (size_t ci = 0; ci < clusters.size(); ++ci)
        {
            if (clusters[ci].centroid[o] < obj)
            {
                obj = clusters[ci].centroid[o];
                extreme_c_for_o = ci;
            }
        }
        // Mixing mode -- as documented -- tells whether a solution should be scalarized, or use a single
        // objective criterion.
        clusters[extreme_c_for_o].mixing_mode = static_cast<long>(objective_indices[o]);
    }
}

// Note: individuals in clusters are assumed to all be solutions in pool.
// (The other way around is NOT required: pool can contain solutions not in a cluster)
void determine_cluster_to_use_mo_gomea(Population &population,
                                       std::vector<ObjectiveCluster> &clusters,
                                       std::vector<Individual> &pool,
                                       std::vector<size_t> &objective_indices,
                                       std::vector<double> &objective_ranges)
{
    // First, map the mixing mode to the indices in the pool
    for (size_t ii = 0; ii < pool.size(); ++ii)
    {
        Individual &i = pool[ii];
        ClusterIndex &cli = population.getData<ClusterIndex>(i);
        cli.cluster_index = ii;
    }

    std::vector<size_t> cluster_per_pool(pool.size());
    std::fill(cluster_per_pool.begin(), cluster_per_pool.end(), static_cast<size_t>(-1));
    std::vector<long> pool_cluster_counts(pool.size());
    std::fill(pool_cluster_counts.begin(), pool_cluster_counts.end(), 0);

    // Go over each cluster and determine each solutions' mixing mode
    Rng &rng = *population.getGlobalData<Rng>();
    for (size_t ci = 0; ci < clusters.size(); ++ci)
    {
        for (auto &i : clusters[ci].members)
        {
            ClusterIndex &cli = population.getData<ClusterIndex>(i);
            size_t idx = cli.cluster_index;
            size_t c = pool_cluster_counts[idx];
            bool replace = false;
            if (c == 0)
                replace = true;
            else
            {
                std::uniform_int_distribution<size_t> r(0, c);
                replace = r(rng.rng) == 0;
            }
            if (replace)
            {
                cluster_per_pool[idx] = ci;
            }
            pool_cluster_counts[idx]++;
        }
    }

    // Assign clusters to solutions
    for (size_t ii = 0; ii < pool.size(); ++ii)
    {
        Individual &i = pool[ii];
        ClusterIndex &cli = population.getData<ClusterIndex>(i);
        cli.cluster_index = cluster_per_pool[ii];

        // Determine nearest cluster, if it was not contained in any cluster.
        if (cluster_per_pool[ii] == static_cast<size_t>(-1))
        {
            Objective &obj = population.getData<Objective>(i);
            double d_min = std::numeric_limits<double>::infinity();
            size_t ci_min = 0;
            for (size_t ci = 0; ci < clusters.size(); ++ci)
            {
                double d = 0.0;
                for (size_t oii = 0; oii < objective_indices.size(); ++oii)
                {
                    size_t o = objective_indices[oii];
                    if (obj.objectives.size() <= o) continue;
                    d += std::pow(clusters[ci].centroid[oii] - obj.objectives[o] / objective_ranges[o], 2);
                }
                if (d < d_min)
                {
                    d_min = d;
                    ci_min = ci;
                }
            }
            cli.cluster_index = ci_min;
        }
    }
}

std::vector<double> compute_objective_ranges(Population &population,
                                             std::vector<size_t> objective_indices,
                                             std::vector<Individual> &pool)
{
    std::vector<double> ranges(objective_indices.size());
    std::vector<double> min_v(objective_indices.size());
    std::fill(min_v.begin(), min_v.end(), std::numeric_limits<double>::infinity());
    std::vector<double> max_v(objective_indices.size());
    std::fill(max_v.begin(), max_v.end(), -std::numeric_limits<double>::infinity());

    auto objectiveGetter = population.getDataContainer<Objective>();

    for (auto &i : pool)
    {
        auto &obj = objectiveGetter.getData(i);
        for (size_t oi = 0; oi < objective_indices.size(); ++oi)
        {
            // If solution is not intialized up to this point: ignore!
            if (objective_indices[oi] < obj.objectives.size() && !std::isnan(obj.objectives[objective_indices[oi]]))
            {
                auto objective_value = obj.objectives[objective_indices[oi]];
                min_v[oi] = std::min(min_v[oi], objective_value);
                max_v[oi] = std::max(max_v[oi], objective_value);
                ranges[oi] = max_v[oi] - min_v[oi];
            }
        }
    }

    return ranges;
}

ArchiveAcceptanceCriterion::ArchiveAcceptanceCriterion(std::shared_ptr<IPerformanceCriterion> wrapped,
                                                       std::shared_ptr<IArchive> archive,
                                                       bool accept_if_added,
                                                       bool accept_if_undominated) :
    wrapped(wrapped), archive(archive), accept_if_added(accept_if_added), accept_if_undominated(accept_if_undominated)
{
}
short ArchiveAcceptanceCriterion::compare(Individual &a, Individual &b)
{
    short result_wrapped = wrapped->compare(a, b);

    // Note: we assume a is in some way an original or reference,
    //       and b is a newly seen solution here.
    //       An operator that tries both, and acts on both is not used
    //       in MO-GOMEA, and hence not implemented here.
    auto r = archive->try_add(b);

    if (accept_if_added && r.added)
        return 2;

    if (accept_if_undominated && !r.dominated)
        return 2;

    return result_wrapped;
}
void ArchiveAcceptanceCriterion::setPopulation(std::shared_ptr<Population> population)
{
    IPerformanceCriterion::setPopulation(population);
    wrapped->setPopulation(population);
    archive->setPopulation(population);
}
void ArchiveAcceptanceCriterion::registerData()
{
    IPerformanceCriterion::registerData();
    wrapped->registerData();
    archive->registerData();
}
void ArchiveAcceptanceCriterion::afterRegisterData()
{
    IPerformanceCriterion::afterRegisterData();
    wrapped->afterRegisterData();
    archive->afterRegisterData();
}

BaseGOMEA::BaseGOMEA(size_t population_size,
                     std::shared_ptr<ISolutionInitializer> initializer,
                     std::shared_ptr<IPerformanceCriterion> performance_criterion,
                     std::shared_ptr<IArchive> archive) :
    population_size(population_size),
    initializer(initializer),
    performance_criterion(performance_criterion),
    archive(archive)
{
}

void BaseGOMEA::setPopulation(std::shared_ptr<Population> population)
{
    this->population = population;
    this->initializer->setPopulation(population);
    this->performance_criterion->setPopulation(population);
    this->archive->setPopulation(population);
}
void BaseGOMEA::registerData()
{
    Population &population = *this->population;
    population.registerData<NIS>();

    this->initializer->registerData();
    this->performance_criterion->registerData();
    this->archive->registerData();
}
void BaseGOMEA::afterRegisterData()
{
    this->initializer->afterRegisterData();
    this->performance_criterion->afterRegisterData();
    this->archive->afterRegisterData();
}
std::vector<Individual> &BaseGOMEA::getSolutionPopulation()
{
    return individuals;
}

void BaseGOMEA::initialize()
{
    Population &population = *this->population;
    individuals.resize(0);

    for (size_t i = 0; i < population_size; ++i)
        individuals.push_back(population.newIndividual());

    initializer->initialize(individuals);

    GObjectiveFunction &objective_function = *population.getGlobalData<GObjectiveFunction>();
    for (auto ii : individuals)
    {
        objective_function.of->evaluate(ii);
        archive->try_add(ii);
    }
}

size_t BaseGOMEA::getNISThreshold(const Individual & /* ii */)
{
    return 1 + static_cast<size_t>(std::floor(std::log2(static_cast<double>(population_size))));
}
void BaseGOMEA::onImprovedSolution(const Individual ii)
{
    Population &population = *this->population;
    // Reset NIS
    NIS &nis = population.getData<NIS>(ii);
    nis.nis = 0;
}
void BaseGOMEA::atGenerationEnd()
{
}
void BaseGOMEA::atGenerationStart()
{
}
void BaseGOMEA::improveSolution(Individual &ii)
{
    Population &population = *this->population;
    bool changed = false;
    bool improved = false;

    auto fos = getFOSForGOM(ii);
    auto gom_dist = getDistributionForGOM(ii);
    auto gom_perf = getPerformanceCriterionForGOM(ii);

    GOM gom;
    gom.setPopulation(this->population);
    gom.apply(ii, *as_ptr(fos), as_ptr(gom_dist), as_ptr(gom_perf), changed, improved);
    // Failed improvement from GOM increases NIS.
    NIS &nis = population.getData<NIS>(ii);
    if (!improved)
        nis.nis += 1;

    if (improved)
    {
        onImprovedSolution(ii);
    }

    // If solution hasn't changed, or the NIS threshold has been reached
    // perform Forced Improvements
    if (!changed || nis.nis > getNISThreshold(ii))
    {
        auto fos = getFOSForFI(ii);
        auto fi_dist = this->getDistributionForFI(ii);
        auto gom_perf = this->getPerformanceCriterionForFI(ii);
        FI fi;
        fi.setPopulation(this->population);
        fi.apply(ii, *as_ptr(fos), as_ptr(fi_dist), as_ptr(gom_perf), changed, improved);
        if (improved)
        {
            onImprovedSolution(ii);
        }
    }
    else
    {
        return;
    }

    // If after both GOM and FI no improvement has occurred, replace with another (i.e. an elitist).
    if (!changed)
    {
        population.copyIndividual(getReplacementSolution(ii), ii);
    }
}

void BaseGOMEA::step_normal_generation()
{
    this->atGenerationStart();

    for (auto &ii : individuals)
    {
        improveSolution(ii);
    }

    this->atGenerationEnd();
}

void BaseGOMEA::step()
{
    if (!initialized)
    {
        initialize();
        initialized = true;
        return;
    }
    step_normal_generation();
}
void GOMEA::afterRegisterData()
{
    BaseGOMEA::afterRegisterData();
    foslearner->afterRegisterData();
}
void GOMEA::registerData()
{
    BaseGOMEA::registerData();
    foslearner->registerData();
}
void GOMEA::setPopulation(std::shared_ptr<Population> population)
{
    BaseGOMEA::setPopulation(population);
    foslearner->setPopulation(population);
}
Individual &GOMEA::getReplacementSolution(const Individual & /* ii */)
{
    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx(0, archived.size() - 1);
    return archived[idx(rng->rng)];
}
APtr<IPerformanceCriterion> GOMEA::getPerformanceCriterionForFI(Individual & /* ii */)
{
    return this->performance_criterion.get();
}
APtr<IPerformanceCriterion> GOMEA::getPerformanceCriterionForGOM(Individual & /* ii */)
{
    return this->performance_criterion.get();
}
APtr<ISamplingDistribution> GOMEA::getDistributionForFI(Individual & /* ii */)
{
    if (!generational_data.has_value()) throw std::runtime_error("can only get distributional info during a generation");
    Population &population = *this->population;
    GenerationalData &g = *generational_data;
    return gomSamplingDistributionFactory(population, g, this);
}
APtr<ISamplingDistribution> GOMEA::getDistributionForGOM(Individual & /* ii */)
{
    if (!generational_data.has_value()) throw std::runtime_error("can only get distributional info during a generation");
    Population &population = *this->population;
    GenerationalData &g = *generational_data;
    return gomSamplingDistributionFactory(population, g, this);
}
APtr<FoS> GOMEA::getFOSForFI(Individual & /* ii */)
{
    return &this->foslearner->getFoS();
}
APtr<FoS> GOMEA::getFOSForGOM(Individual & /* ii */)
{
    return &this->foslearner->getFoS();
}
void GOMEA::atGenerationEnd()
{
    // It is invalid to call this method if a generation has not been started yet.
    if (!generational_data.has_value()) throw std::runtime_error("cannot end generation before starting one");

    Population &population = *this->population;
    GenerationalData &g = *generational_data;

    // Return originals back to the pool.
    for (size_t i = 0; i < population_size; ++i)
    {
        population.dropIndividual(g.originals[i]);
    }

    // Reset generational data
    generational_data.reset();
}
void GOMEA::atGenerationStart()
{
    generational_data.emplace(GenerationalData());
    GenerationalData &g = *generational_data;

    Population &population = *this->population;
    // Learn FOS
    foslearner->learnFoS(individuals);
    // Create the model we'll be sampling from.
    g.originals.resize(population_size);
    for (size_t i = 0; i < population_size; ++i)
    {
        g.originals[i] = population.newIndividual();
        population.copyIndividual(individuals[i], g.originals[i]);
    }
    // Get the rng
    rng = population.getGlobalData<Rng>().get();

    std::shuffle(individuals.begin(), individuals.end(), rng->rng);
}
GOMEA::GOMEA(size_t population_size,
             std::shared_ptr<ISolutionInitializer> initializer,
             std::shared_ptr<FoSLearner> foslearner,
             std::shared_ptr<IPerformanceCriterion> performance_criterion,
             std::shared_ptr<IArchive> archive,
             bool autowrap,
             std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
                 gomSamplingDistributionFactory,
             std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
                 fiSamplingDistributionFactory) :
    BaseGOMEA(population_size,
              initializer,
              autowrap ? std::make_shared<ArchiveAcceptanceCriterion>(performance_criterion, archive)
                       : performance_criterion,
              archive),
    foslearner(foslearner),
    gomSamplingDistributionFactory(gomSamplingDistributionFactory),
    fiSamplingDistributionFactory(fiSamplingDistributionFactory)
{
}

//
MO_GOMEA::MO_GOMEA(size_t population_size,
                   size_t number_of_clusters,
                   std::vector<size_t> objective_indices,
                   std::shared_ptr<ISolutionInitializer> initializer,
                   std::shared_ptr<FoSLearner> foslearner,
                   std::shared_ptr<IPerformanceCriterion> performance_criterion,
                   std::shared_ptr<IArchive> archive,
                   std::shared_ptr<GOMEAPlugin> plugin,
                   bool donor_search,
                   bool autowrap) :
    BaseGOMEA(population_size,
              initializer,
              autowrap
                  ? std::make_shared<ArchiveAcceptanceCriterion>(
                        std::make_shared<WrappedOrSingleSolutionPerformanceCriterion>(performance_criterion), archive)
                  : performance_criterion,
              archive),
    number_of_clusters(number_of_clusters),
    objective_indices(objective_indices),
    foslearner(foslearner),
    plugin(plugin),
    donor_search(donor_search)
{
}
void MO_GOMEA::initialize()
{
    BaseGOMEA::initialize();
    plugin->onInitEnd(individuals);
}
void MO_GOMEA::atGenerationStart()
{
    plugin->atGenerationStartBegin(individuals);
    Population &population = *this->population;
    rng = population.getGlobalData<Rng>().get();

    // Get objective ranges
    auto objective_ranges = compute_objective_ranges(population, objective_indices, individuals);

    // Infer clusters
    auto clusters = cluster_mo_gomea(population, individuals, objective_indices, objective_ranges, number_of_clusters);
    // Determine extreme clusters
    determine_extreme_clusters(objective_indices, clusters);
    //
    determine_cluster_to_use_mo_gomea(population, clusters, individuals, objective_indices, objective_ranges);

    // Learn per cluster FOS
    std::vector<std::shared_ptr<FoSLearner>> per_cluster_fos(clusters.size());
    for (size_t ci = 0; ci < clusters.size(); ++ci)
    {
        foslearner->learnFoS(clusters[ci].members);

        per_cluster_fos[ci] = std::shared_ptr<FoSLearner>(foslearner->cloned_ptr());
    }
    TypedGetter<ClusterIndex> gli = population.getDataContainer<ClusterIndex>();
    TypedGetter<UseSingleObjective> guso = population.getDataContainer<UseSingleObjective>();
    for (size_t i = 0; i < population_size; ++i)
    {
        ClusterIndex &cli = gli.getData(individuals[i]);
        UseSingleObjective &uso = guso.getData(individuals[i]);
        uso.index = clusters[cli.cluster_index].mixing_mode;
    }

    // Create copies.
    std::vector<Individual> originals(population_size);
    std::unordered_map<Individual, Individual> remap_to_original;
    for (size_t i = 0; i < population_size; ++i)
    {
        originals[i] = population.newIndividual();
        population.copyIndividual(individuals[i], originals[i]);
        remap_to_original[individuals[i]] = originals[i];
    }
    // Remap the clusters
    std::vector<std::vector<Individual>> per_cluster_originals(clusters.size());
    for (size_t ci = 0; ci < clusters.size(); ++ci)
    {
        per_cluster_originals[ci].resize(clusters[ci].members.size());
        std::transform(clusters[ci].members.begin(),
                       clusters[ci].members.end(),
                       per_cluster_originals[ci].begin(),
                       [&remap_to_original](Individual i) { return remap_to_original[i]; });
    }

    std::shuffle(individuals.begin(), individuals.end(), rng->rng);

    g.emplace(GenerationalData{std::move(per_cluster_fos), std::move(originals), std::move(per_cluster_originals)});
    plugin->atGenerationStartEnd(individuals);
}
void MO_GOMEA::setPopulation(std::shared_ptr<Population> population)
{
    BaseGOMEA::setPopulation(population);
    foslearner->setPopulation(population);
    plugin->setPopulation(population);
}
void MO_GOMEA::registerData()
{
    Population &population = *this->population;
    population.registerData<ClusterIndex>();
    population.registerData<UseSingleObjective>();

    BaseGOMEA::registerData();
    foslearner->registerData();
    plugin->registerData();
}
void MO_GOMEA::afterRegisterData()
{
    BaseGOMEA::afterRegisterData();
    foslearner->afterRegisterData();
    plugin->afterRegisterData();
}

Individual &MO_GOMEA::getReplacementSolution(const Individual & /* ii */)
{
    // Get random from archive
    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx_d(0, archived.size() - 1);
    return std::ref(archived[idx_d(rng->rng)]);
}
APtr<IPerformanceCriterion> MO_GOMEA::getPerformanceCriterionForGOM(Individual & /* ii */)
{
    return performance_criterion.get();
}
APtr<IPerformanceCriterion> MO_GOMEA::getPerformanceCriterionForFI(Individual & /* ii */)
{
    return performance_criterion.get();
}
APtr<ISamplingDistribution> MO_GOMEA::getDistributionForGOM(Individual &ii)
{
    if (!g.has_value()) throw std::runtime_error("can only get distributional info during a generation");
    Population &population = *this->population;
    auto &d = *g;
    auto &ci = population.getData<ClusterIndex>(ii);
    if (donor_search)
        return std::make_unique<CategoricalDonorSearchDistribution>(population,
                                                                    d.per_cluster_originals[ci.cluster_index]);
    else
        return std::make_unique<CategoricalPopulationSamplingDistribution>(population,
                                                                           d.per_cluster_originals[ci.cluster_index]);
}
APtr<ISamplingDistribution> MO_GOMEA::getDistributionForFI(Individual & /* ii */)
{
    if (!g.has_value()) throw std::runtime_error("can only get distributional info during a generation");
    Population &population = *this->population;

    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx_d(0, archived.size() - 1);
    std::vector<Individual> random_from_archive = {archived[idx_d(rng->rng)]};
    return std::make_unique<CategoricalPopulationSamplingDistribution>(population, random_from_archive);
}
APtr<FoS> MO_GOMEA::getFOSForGOM(Individual &ii)
{
    if (!g.has_value()) throw std::runtime_error("can only get distributional info during a generation");
    Population &population = *this->population;
    auto &d = *g;

    auto &ci = population.getData<ClusterIndex>(ii);
    return &d.per_cluster_fos[ci.cluster_index]->getFoS();
}
APtr<FoS> MO_GOMEA::getFOSForFI(Individual &ii)
{
    return getFOSForGOM(ii);
}
void MO_GOMEA::atGenerationEnd()
{
    plugin->atGenerationEndBegin(individuals);
    Population &population = *this->population;
    
    if (!g.has_value()) throw std::runtime_error("cannot end generation before starting one.");
    auto &d = *g;

    // Clean up
    for (size_t i = 0; i < population_size; ++i)
    {
        population.dropIndividual(d.originals[i]);
    }

    g.reset();
    plugin->atGenerationEndEnd(individuals);
}

//

/**
 * @brief Construct KernelGOMEA
 *
 * @param population_size Initial population size
 * @param number_of_clusters Number of clusters - note: only used for determining which solutions perform
 * single-objective acceptance.
 * @param objective_indices Which indices to perform clustering on - note: see number_of_clusters
 * @param initializer Population Initializer
 * @param foslearner Family of Subsets learner, e.g. a linkage tree.
 * @param performance_criterion Acceptance Criterion to use
 * @param archive Tracker for best known solution so-far.
 * @param autowrap Whether the acceptance criterion should be automatically adjusted to account for single-objective
 * acceptance.
 */
KernelGOMEA::KernelGOMEA(size_t population_size,
                         size_t number_of_clusters, // Note: only used for determining which solutions should use a
                                                    // single-objective acceptance criterion
                         std::vector<size_t> objective_indices,
                         std::shared_ptr<ISolutionInitializer> initializer,
                         std::shared_ptr<FoSLearner> foslearner,
                         std::shared_ptr<IPerformanceCriterion> performance_criterion,
                         std::shared_ptr<IArchive> archive,
                         std::shared_ptr<GOMEAPlugin> plugin,
                         bool donor_search,
                         bool autowrap) :
    BaseGOMEA(population_size,
              initializer,
              autowrap
                  ? std::make_shared<ArchiveAcceptanceCriterion>(
                        std::make_shared<WrappedOrSingleSolutionPerformanceCriterion>(performance_criterion), archive)
                  : performance_criterion,
              archive),
    number_of_clusters(number_of_clusters),
    objective_indices(objective_indices),
    foslearner(foslearner),
    plugin(plugin),
    donor_search(donor_search)
{
}
void KernelGOMEA::initialize()
{
    BaseGOMEA::initialize();
    plugin->onInitEnd(individuals);
}

void KernelGOMEA::setPopulation(std::shared_ptr<Population> population)
{
    BaseGOMEA::setPopulation(population);
    foslearner->setPopulation(population);
    plugin->setPopulation(population);
}
void KernelGOMEA::registerData()
{

    Population &population = *this->population;
    population.registerData<ClusterIndex>();
    population.registerData<UseSingleObjective>();
    population.registerData<LinkageKernel>();

    BaseGOMEA::registerData();
    foslearner->registerData();
    plugin->registerData();
}
void KernelGOMEA::afterRegisterData()
{
    BaseGOMEA::afterRegisterData();
    foslearner->afterRegisterData();
    plugin->afterRegisterData();
}

void KernelGOMEA::atGenerationStart()
{
    plugin->atGenerationStartBegin(individuals);
    Population &population = *this->population;
    rng = population.getGlobalData<Rng>().get();

    // Get objective ranges
    auto objective_ranges = compute_objective_ranges(population, objective_indices, individuals);

    if (objective_indices.size() > 1)
    {
        // Infer clusters
        auto clusters =
            cluster_mo_gomea(population, individuals, objective_indices, objective_ranges, number_of_clusters);
        // Determine extreme clusters
        determine_extreme_clusters(objective_indices, clusters);
        determine_cluster_to_use_mo_gomea(population, clusters, individuals, objective_indices, objective_ranges);

        // printClusters(clusters);

        // Assign extreme objectives
        TypedGetter<ClusterIndex> gli = population.getDataContainer<ClusterIndex>();
        TypedGetter<UseSingleObjective> guso = population.getDataContainer<UseSingleObjective>();
        for (size_t i = 0; i < population_size; ++i)
        {
            ClusterIndex &cli = gli.getData(individuals[i]);
            UseSingleObjective &uso = guso.getData(individuals[i]);
            long mixing_mode = clusters[cli.cluster_index].mixing_mode;
            uso.index = mixing_mode;
        }
    }
    // Create copies.
    std::vector<Individual> originals(population_size);
    for (size_t i = 0; i < population_size; ++i)
    {
        originals[i] = population.newIndividual();
        population.copyIndividual(individuals[i], originals[i]);
    }

    // Determine neighborhoods & corresponding FOS.
    // TODO: Make the neighborhood determination & corresponding measure a parameter to Kernel GOMEA.
    TypedGetter<GenotypeCategorical> gc = population.getDataContainer<GenotypeCategorical>();
    bool symmetric = true;

    size_t k = static_cast<size_t>(std::ceil(std::sqrt(population_size)));
    std::vector<size_t> indices(individuals.size());
    std::iota(indices.begin(), indices.end(), 0);
    auto neighborhood_indices = getNeighborhoods(
        [this, &gc](size_t a, size_t b) {
            GenotypeCategorical gca = gc.getData(individuals[a]);
            GenotypeCategorical gcb = gc.getData(individuals[b]);

            return hamming_distance(
                gca.genotype.data(), gcb.genotype.data(), std::min(gca.genotype.size(), gcb.genotype.size()));
        },
        *rng,
        indices,
        k,
        symmetric);
    auto lk = population.getDataContainer<LinkageKernel>();
    for (size_t i = 0; i < population_size; ++i)
    {
        auto &ilk = lk.getData(individuals[i]);
        // Convert neighborhood to originals.
        auto &nii = neighborhood_indices[i];
        // TODO: Remove reference to original solutions, maybe.
        std::vector<Individual> nbi(nii.size());
        std::vector<Individual> anb(nii.size());
        for (size_t idx = 0; idx < nii.size(); ++idx)
        {
            nbi[idx] = originals[nii[idx]];
            anb[idx] = individuals[nii[idx]];
        }
        // Learn linkage
        foslearner->learnFoS(nbi);
        auto foslearner_copy = std::unique_ptr<FoSLearner>(foslearner->cloned_ptr());

        // Set data
        ilk.neighborhood = std::move(nbi);
        ilk.pop_neighborhood = std::move(anb);
        // std::cout << "Neighborhood size: " << ilk.neighborhood.size() << "\n";
        ilk.fos = std::move(foslearner_copy);
    }

    std::shuffle(individuals.begin(), individuals.end(), rng->rng);

    g.emplace(GenerationalData{std::move(originals), std::move(lk)});
    plugin->atGenerationStartEnd(individuals);
}
Individual &KernelGOMEA::getReplacementSolution(const Individual & /* ii */)
{
    // Get random from archive
    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx_d(0, archived.size() - 1);
    return std::ref(archived[idx_d(rng->rng)]);
}
APtr<IPerformanceCriterion> KernelGOMEA::getPerformanceCriterionForGOM(Individual & /* ii */)
{
    return performance_criterion.get();
}
APtr<IPerformanceCriterion> KernelGOMEA::getPerformanceCriterionForFI(Individual & /* ii */)
{
    return performance_criterion.get();
}
APtr<ISamplingDistribution> KernelGOMEA::getDistributionForGOM(Individual &ii)
{
    if (!g.has_value()) throw std::runtime_error("can only get distributional info during a generation");
    Population &population = *this->population;
    auto &d = *g;
    if (donor_search)
        return std::make_unique<CategoricalDonorSearchDistribution>(population, d.lk.getData(ii).neighborhood);
    else
        return std::make_unique<CategoricalPopulationSamplingDistribution>(population, d.lk.getData(ii).neighborhood);
}
APtr<ISamplingDistribution> KernelGOMEA::getDistributionForFI(Individual & /* ii */)
{
    Population &population = *this->population;

    auto &archived = archive->get_archived();
    std::uniform_int_distribution<size_t> idx_d(0, archived.size() - 1);
    std::vector<Individual> random_from_archive = {archived[idx_d(rng->rng)]};
    return std::make_unique<CategoricalPopulationSamplingDistribution>(population, random_from_archive);
}
APtr<FoS> KernelGOMEA::getFOSForGOM(Individual &ii)
{
    if (!g.has_value()) throw std::runtime_error("can only get distributional info during a generation");
    // Note: As the reference to the fos is stored within the array
    // its adress can change, hence we make a copy here.
    // Time spent finding this bug: 8 hours.
    auto &d = *g;
    auto &lkii = d.lk.getData(ii);
    // Note: reference to linkage kernel
    // ! If ever LinkageKernel is made copyable, it will change the address
    // ! of the FOS elements. In that case, make a copy here.
    // ! Otherwise, the current approach (using a pointer) is cheaper and works.
    // auto fos = std::make_unique<FoS>(lkii.fos->getFoS());
    // return fos;
    return &(lkii.fos->getFoS());
}
APtr<FoS> KernelGOMEA::getFOSForFI(Individual &ii)
{
    return getFOSForGOM(ii);
}
void KernelGOMEA::atGenerationEnd()
{
    if (!g.has_value()) throw std::runtime_error("cannot end generation before starting one");

    plugin->atGenerationEndBegin(individuals);
    Population &population = *this->population;
    auto &d = *g;

    // Clean up
    for (size_t i = 0; i < population_size; ++i)
    {
        population.dropIndividual(d.originals[i]);
    }

    // Reset the linkage kernels (to save memory)
    bool reset_linkage_kernels = true;
    if (reset_linkage_kernels)
    {
        auto lk = population.getDataContainer<LinkageKernel>();
        for (auto &ii : individuals)
        {
            auto &iilk = lk.getData(ii);
            iilk.fos.reset();
            iilk.neighborhood.clear();
        }
    }
    g.reset();
    plugin->atGenerationEndEnd(individuals);
}

HoangScalarizationScheme::HoangScalarizationScheme(std::shared_ptr<Scalarizer> scalarizer,
                                                   std::vector<size_t> objective_indices) :
    objective_indices(objective_indices), scalarizer(scalarizer)
{
}

void HoangScalarizationScheme::setPopulation(std::shared_ptr<Population> population)
{
    GOMEAPlugin::setPopulation(population);
    scalarizer->setPopulation(population);
}
void HoangScalarizationScheme::registerData()
{
    Population &population = *this->population;
    population.registerData<ScalarizationWeights>();
    scalarizer->registerData();
}
void HoangScalarizationScheme::afterRegisterData()
{
    scalarizer->afterRegisterData();
}

void HoangScalarizationScheme::onInitEnd(std::vector<Individual> &individuals)
{
    Population &population = *this->population;
    hoang_generate_initial_weights(
        population,
        *scalarizer.get(),
        objective_indices.size(),
        individuals);
}
void HoangScalarizationScheme::atGenerationStartEnd(std::vector<Individual> &individuals)
{
    scalarizer->update_frame_of_reference(individuals);

    Population &population = *this->population;
    hoang_reassign_weights(
        population,
        *scalarizer.get(),
        objective_indices.size(),
        individuals);
}
