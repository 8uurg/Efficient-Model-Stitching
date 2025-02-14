//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#pragma once

#include "acceptation_criteria.hpp"
#include "archive.hpp"
#include "base.hpp"
#include "utilities.hpp"
#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>

// Linkage Learning
class LinkageMetric : public IDataUser
{
  public:
    virtual double compute_linkage(size_t v_a, size_t v_b, std::vector<Individual> &individuals) = 0;
    // TODO: Make these values dependent on the variable pair.
    virtual std::optional<double> filter_minimum_threshold();
    virtual std::optional<double> filter_maximum_threshold();
};

// Mututal Information
class MI : public LinkageMetric
{
  public:
    double compute_linkage(size_t v_a, size_t v_b, std::vector<Individual> &individuals) override;

    // TODO: Implement minimum and maximum threshold
  private:
    struct Cache
    {
        TypedGetter<GenotypeCategorical> &genotype_categorical;
        std::vector<char> &alphabet_size;
    };
    std::optional<Cache> cache;
};

// Normalized Mutual Information
class NMI : public LinkageMetric
{
  public:
    double compute_linkage(size_t v_a, size_t v_b, std::vector<Individual> &individuals) override;

    // TODO: Implement minimum and maximum threshold
  private:
    struct Cache
    {
        TypedGetter<GenotypeCategorical> genotype_categorical;
        std::vector<char> &alphabet_size;
    };
    std::optional<Cache> cache;
};

// Random - can be used to make a random tree.
class RandomLinkage : public LinkageMetric
{
  public:
    double compute_linkage(size_t v_a, size_t v_b, std::vector<Individual> &individuals) override;
    void afterRegisterData() override;
};

// Predetermined / fixed linkage.
class FixedLinkage : public LinkageMetric
{
  public:
    FixedLinkage(SymMatrix<double> linkage,
                 std::optional<double> minimum_threshold = std::nullopt,
                 std::optional<double> maximum_threshold = std::nullopt);
    double compute_linkage(size_t v_a, size_t v_b, std::vector<Individual> &) override;

    std::optional<double> filter_minimum_threshold() override;
    std::optional<double> filter_maximum_threshold() override;

  private:
    SymMatrix<double> linkage;
    std::optional<double> minimum_threshold;
    std::optional<double> maximum_threshold;
};

using FoSElement = std::vector<size_t>;
using FoS = std::vector<FoSElement>;

class FoSLearner : public IDataUser
{
  public:
    virtual void learnFoS(std::vector<Individual> &individuals) = 0;
    virtual FoS &getFoS() = 0;
    virtual FoSLearner *cloned_ptr() = 0;
};

class CategoricalUnivariateFoS : public FoSLearner
{
  public:
    void learnFoS(std::vector<Individual> &individuals) override;
    FoS &getFoS() override;

    void afterRegisterData() override;
    FoSLearner *cloned_ptr() override;

  private:
    FoS fos;
};

double mergeUPGMA(
    double distance_ij, double distance_ik, double distance_jk, size_t size_i, size_t size_j, size_t /* size_k */);

struct TreeNode
{
    size_t left;
    size_t right;
    double distance;
    size_t size;
};

std::vector<TreeNode> performHierarchicalClustering(SymMatrix<double> linkage, Rng &rng);

enum FoSOrdering
{
    AsIs = 0,
    Random = 1,
    SizeIncreasing = 2,
};

class CategoricalLinkageTree : public FoSLearner
{
  public:
    CategoricalLinkageTree(std::shared_ptr<LinkageMetric> metric,
                           FoSOrdering ordering = AsIs,
                           bool filter_zeros = false,
                           bool filter_maxima = false,
                           bool filter_root = true);

    void learnFoS(std::vector<Individual> &individuals) override;
    FoS &getFoS() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    FoSLearner *cloned_ptr() override;

  private:
    std::shared_ptr<LinkageMetric> metric;
    FoSOrdering ordering;
    // Filter merge-distances of zero (unless univariate)
    bool filter_minima;
    // Filter the children if the parent has the maximum scoring value.
    bool filter_maxima;
    //
    bool filter_root;
    FoS fos;
};

class ISamplingDistribution
{
  public:
    virtual ~ISamplingDistribution() = default;
    virtual bool apply_resample(Individual ii, std::vector<size_t> &subset) = 0;
};

template <typename T, // Lambda double(size_t, size_t)
                      // Check lambda signature
          typename = std::enable_if_t<std::is_invocable_r_v<double, T, size_t, size_t>>>
std::vector<size_t> getKNNNeighborhood(
    T &&distance,
    Rng &rng,
    size_t c,
    std::vector<size_t> &pool,
    size_t k,
    double distance_threshold)
{
    std::uniform_real_distribution<double> r01(0, 1);
    std::vector<std::tuple<double, double, size_t>> v(pool.size());
    for (size_t idx = 0; idx < pool.size(); ++idx)
    {
        double d = distance(c, pool[idx]);
        // For every distance smaller or equal - it is included 'for free'.
        if (d <= distance_threshold)
        {
            k += 1;
        }
        v[idx] = std::make_tuple(d, r01(rng.rng), pool[idx]);
    }
    if (pool.size() > k)
    {
        std::nth_element(v.begin(), v.begin() + static_cast<long long int>(k), v.end());
        v.resize(k);
    }
    std::vector<size_t> neighbors(v.size());
    for (size_t idx = 0; idx < v.size(); ++idx)
    {
        neighbors[idx] = std::get<2>(v[idx]);
    }
    return neighbors;
}

template <typename T, // Lambda double(size_t, size_t)
                      // Check lambda signature
          typename = std::enable_if_t<std::is_invocable_r_v<double, T, size_t, size_t>>>
std::vector<std::vector<size_t>> getNeighborhoods(
    T &&distance,
    Rng &rng, std::vector<size_t> &pool,
    size_t k,
    bool symmetric,
    double distance_threshold = -std::numeric_limits<double>::infinity())
{
    std::vector<std::vector<size_t>> neighborhoods(pool.size());
    std::vector<std::set<size_t>> is_neighbor;
    if (symmetric)
        is_neighbor.resize(pool.size());
    for (size_t idx = 0; idx < pool.size(); ++idx)
    {
        std::vector<size_t> neighborhood = getKNNNeighborhood(distance, rng, pool[idx], pool, k, distance_threshold);
        // Append to neighborhood
        neighborhoods[idx].insert(neighborhoods[idx].end(), neighborhood.begin(), neighborhood.end());
        // If symmetric, keep track of which items are neighbors.
        if (symmetric)
            for (auto o_idx : neighborhood)
                is_neighbor[idx].insert(o_idx);
        // If symmetric, add self to neighbors.
        if (symmetric)
        {
            for (size_t o_idx : neighborhood)
            {
                neighborhoods[o_idx].push_back(idx);
                is_neighbor[o_idx].insert(idx);
            }
        }
    }
    return neighborhoods;
}


template <typename T, // Lambda double(size_t, size_t)
                      // Check lambda signature
          typename = std::enable_if_t<std::is_invocable_r_v<double, T, size_t, size_t>>>
std::vector<size_t> getNeighborhood(
    T &&distance,
    Rng &rng, 
    size_t idx,
    std::vector<size_t> &pool,
    size_t k,
    bool symmetric,
    double distance_threshold = -std::numeric_limits<double>::infinity())
{
    std::vector<std::vector<size_t>> neighborhoods(pool.size());
    std::vector<std::set<size_t>> is_neighbor;
    if (symmetric)
    {
        is_neighbor.resize(pool.size());
        for (size_t idx = 0; idx < pool.size(); ++idx)
        {
            std::vector<size_t> neighborhood = getKNNNeighborhood(distance, rng, pool[idx], pool, k, distance_threshold);
            // Append to neighborhood
            neighborhoods[idx].insert(neighborhoods[idx].end(), neighborhood.begin(), neighborhood.end());
            // If symmetric, keep track of which items are neighbors.
            for (auto o_idx : neighborhood)
                    is_neighbor[idx].insert(o_idx);
            // If symmetric, add self to neighbors.
            for (size_t o_idx : neighborhood)
            {
                neighborhoods[o_idx].push_back(idx);
                is_neighbor[o_idx].insert(idx);
            }
            
        }
        return neighborhoods[idx];
    }
    else
    {
        return getKNNNeighborhood(distance, rng, pool[idx], pool, k, distance_threshold);
    }
}

class CategoricalPopulationSamplingDistribution : public ISamplingDistribution
{
  public:
    CategoricalPopulationSamplingDistribution(Population &population, std::vector<Individual> pool);
    bool apply_resample(Individual ii, std::vector<size_t> &subset) override;

  private:
    Population &population;
    std::vector<Individual> pool;

    struct Cache
    {
        Rng &rng;
        TypedGetter<GenotypeCategorical> ggc;
    };
    std::optional<Cache> cache;
};

class CategoricalDonorSearchDistribution : public ISamplingDistribution
{
  public:
    CategoricalDonorSearchDistribution(Population &population, std::vector<Individual> pool);
    bool apply_resample(Individual ii, std::vector<size_t> &subset) override;

  private:
    Population &population;
    std::vector<Individual> pool;

    struct Cache
    {
        Rng &rng;
        TypedGetter<GenotypeCategorical> ggc;
    };
    std::optional<Cache> cache;
};

class CategoricalStoredSamplingDistribution : public ISamplingDistribution
{
  public:
    CategoricalStoredSamplingDistribution(Population &population, std::vector<Individual> pool);
    bool apply_resample(Individual ii, std::vector<size_t> &subset) override;

  private:
    Population &population;
    std::vector<std::vector<char>> pool;

    struct Cache
    {
        Rng &rng;
        TypedGetter<GenotypeCategorical> ggc;
    };
    std::optional<Cache> cache;
};

class CategoricalStoredDonorSearchDistribution : public ISamplingDistribution
{
  public:
    CategoricalStoredDonorSearchDistribution(Population &population, std::vector<Individual> pool);
    bool apply_resample(Individual ii, std::vector<size_t> &subset) override;

  private:
    Population &population;
    std::vector<std::vector<char>> pool;

    struct Cache
    {
        Rng &rng;
        TypedGetter<GenotypeCategorical> ggc;
    };
    std::optional<Cache> cache;
};

class IIncrementalImprovementOperator : public IDataUser
{
  public:
    virtual void apply(Individual ii,
                       FoS &fos,
                       ISamplingDistribution *distribution,
                       IPerformanceCriterion *acceptance_criterion,
                       bool &changed,
                       bool &improved) = 0;

    void evaluate_change(Individual current, Individual backup, std::vector<size_t> &elements_changed);
};

class GOM : public IIncrementalImprovementOperator
{
  public:
    void apply(Individual ii,
               FoS &fos,
               ISamplingDistribution *distribution,
               IPerformanceCriterion *acceptance_criterion,
               bool &changed,
               bool &improved) override;
};

class FI : public IIncrementalImprovementOperator
{
  public:
    void apply(Individual ii,
               FoS &fos,
               ISamplingDistribution *distribution,
               IPerformanceCriterion *acceptance_criterion,
               bool &changed,
               bool &improved) override;
};

struct NIS
{
    size_t nis = 0;
};

class BaseGOMEA : public GenerationalApproach
{
  protected:
    BaseGOMEA(size_t population_size,
              std::shared_ptr<ISolutionInitializer> initializer,
              std::shared_ptr<IPerformanceCriterion> performance_criterion,
              std::shared_ptr<IArchive> archive);

  public:
    // -- Common fields --
    bool initialized = false;            // Is the population initialized & evaluated?
    std::vector<Individual> individuals; // Population of individuals
    const size_t population_size;        // Population size!
    // -- Subcomponents --
    const std::shared_ptr<ISolutionInitializer> initializer; // Genotype initializer
    const std::shared_ptr<IPerformanceCriterion>
        performance_criterion;               // Performance criterion to be used by GOM & others
    const std::shared_ptr<IArchive> archive; // Archive to store best solutions in.

    // -- Changing Methods --
    // The methods listed below often differ between configurations / approaches
    // Override and change them accordingly.

    virtual void atGenerationStart();
    virtual void atGenerationEnd();
    virtual void onImprovedSolution(const Individual);
    virtual size_t getNISThreshold(const Individual & /* ii */);

    virtual APtr<FoS> getFOSForGOM(Individual &ii) = 0;
    virtual APtr<FoS> getFOSForFI(Individual &ii) = 0;
    virtual APtr<ISamplingDistribution> getDistributionForGOM(Individual &ii) = 0;
    virtual APtr<ISamplingDistribution> getDistributionForFI(Individual &ii) = 0;
    virtual APtr<IPerformanceCriterion> getPerformanceCriterionForGOM(Individual &ii) = 0;
    virtual APtr<IPerformanceCriterion> getPerformanceCriterionForFI(Individual &ii) = 0;
    virtual Individual &getReplacementSolution(const Individual &ii) = 0;

    virtual void initialize();
    void improveSolution(Individual &ii);

    virtual void step_normal_generation();
    // Following this point,fewer changes are expected.

    void step() override;

    std::vector<Individual> &getSolutionPopulation() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;
};

// This needs to be moved & renamed to something better.
class ArchiveAcceptanceCriterion : public IPerformanceCriterion
{
  public:
    ArchiveAcceptanceCriterion(std::shared_ptr<IPerformanceCriterion> wrapped,
                               std::shared_ptr<IArchive> archive,
                               bool accept_if_added = true,
                               bool accept_if_undominated = false);

    short compare(Individual &a, Individual &b) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

  private:
    std::shared_ptr<IPerformanceCriterion> wrapped;
    std::shared_ptr<IArchive> archive;

    bool accept_if_added = true;
    bool accept_if_undominated = false;
};

struct GenerationalData
{
    std::vector<Individual> originals;
};

class DefaultGOMSamplingDistributionFactory
{
  public:
    bool donor_search;
    DefaultGOMSamplingDistributionFactory(bool donor_search = true) : donor_search(donor_search)
    {
    }

    APtr<ISamplingDistribution> operator()(Population &population, GenerationalData &g, BaseGOMEA *)
    {
        if (donor_search)
            return std::make_unique<CategoricalDonorSearchDistribution>(population, g.originals);
        else
            return std::make_unique<CategoricalPopulationSamplingDistribution>(population, g.originals);
    }
};

class DefaultFISamplingDistributionFactory
{
  public:
    APtr<ISamplingDistribution> operator()(Population &population, GenerationalData &, BaseGOMEA *g)
    {
        return std::make_unique<CategoricalPopulationSamplingDistribution>(population, g->archive->get_archived());
    }
};

class GOMEA : public BaseGOMEA
{
  private:
    const std::shared_ptr<FoSLearner> foslearner;
    std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
        gomSamplingDistributionFactory;
    std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
        fiSamplingDistributionFactory;

  public:
    GOMEA(size_t population_size,
          std::shared_ptr<ISolutionInitializer> initializer,
          std::shared_ptr<FoSLearner> foslearner,
          std::shared_ptr<IPerformanceCriterion> performance_criterion,
          std::shared_ptr<IArchive> archive,
          bool autowrap = true,
          std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
              gomSamplingDistributionFactory = DefaultGOMSamplingDistributionFactory(),
          std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
              fiSamplingDistributionFactory = DefaultFISamplingDistributionFactory());

    Rng *rng;
    std::optional<GenerationalData> generational_data;

    void atGenerationStart() override;

    void atGenerationEnd() override;

    APtr<FoS> getFOSForGOM(Individual & /* ii */) override;
    APtr<FoS> getFOSForFI(Individual & /* ii */) override;
    APtr<ISamplingDistribution> getDistributionForGOM(Individual & /* ii */) override;
    APtr<ISamplingDistribution> getDistributionForFI(Individual & /* ii */) override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForGOM(Individual & /* ii */) override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForFI(Individual & /* ii */) override;
    Individual &getReplacementSolution(const Individual & /* ii */) override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;
};

/**
 * @brief A plugin for GOMEA, i.e. for setting up scalarizations.
 *
 * For small encapsulated behaviors that do not affect the flow of the algorithm itself,
 * and are instead plugged in in particular places.
 *
 * Generational Loop plugs for each generation in the following order.
 *  [atGenerationStartBegin]
 *  [...]
 *  [atGenerationStartEnd]
 *  GENERATION
 *  [atGenerationEndBegin]
 *  [...]
 *  [atGenerationEndEnd]
 */
class GOMEAPlugin : public IDataUser
{
  public:
    /**
     * @brief Ran at the end of initialization of a generation.
     */
    virtual void onInitEnd(std::vector<Individual> &){};

    /**
     * @brief Ran at the start of a generation, before approach specific operations.
     */
    virtual void atGenerationStartBegin(std::vector<Individual> &){};
    /**
     * @brief Ran at the start of a generation, after approach specific operations.
     */
    virtual void atGenerationStartEnd(std::vector<Individual> &){};
    /**
     * @brief Ran at the end of a generation, before approach specific operations.
     */
    virtual void atGenerationEndBegin(std::vector<Individual> &){};
    /**
     * @brief Ran at the end of a generation, after approach specific operations.
     */
    virtual void atGenerationEndEnd(std::vector<Individual> &){};
};

/**
 * @brief A scalarization weights assignment scheme by (Luong, Alderliesten, and Bosman 2018)
 *
 * This class implements the scalarization assignment scheme described in:
 *    Luong, Ngoc Hoang, Tanja Alderliesten, and Peter A. N. Bosman. 2018.
 *    ‘Improving the Performance of MO-RV-GOMEA on Problems with Many Objectives Using Tchebycheff Scalarizations’.
 *    In Proceedings of the Genetic and Evolutionary Computation Conference, 705–12. GECCO ’18.
 *    New York, NY, USA: Association for Computing Machinery.
 *    https://doi.org/10.1145/3205455.3205498.
 *
 * A rough description is as follows:
 * - Generate randomly distributed vectors (normalized by the sum of their absolute values)
 * - Apply Greedy Subset Scattering (i.e. from utilities) to obtain a well distributed set of vectors.
 * - From the well distributed set of random vectors, in random order,
 *   Assign each to the solution with the best scalarization value for this vector of weights.
 */
class HoangScalarizationScheme : public GOMEAPlugin
{
  private:
    std::vector<size_t> objective_indices;
    std::shared_ptr<Scalarizer> scalarizer;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    // void assignWeights(std::vector<Individual> &individuals,
    //                    Matrix<double> &vectors,
    //                    std::vector<size_t> indices,
    //                    size_t dim);

  public:
    HoangScalarizationScheme(std::shared_ptr<Scalarizer> scalarizer, std::vector<size_t> objective_indices);

    void onInitEnd(std::vector<Individual> &) override;
    void atGenerationStartEnd(std::vector<Individual> &) override;
};

/**
 * @brief Compute objective ranges over a pool of solutions.
 *
 * @param population Population object for accessing solution data.
 * @param objective_indices Indices for which to compute the ranges.
 * @param pool Pool of solutions to compute ranges over.
 * @return std::vector<double> Range of each objective_index provided in pool.
 *
 * Additionally, if the minimum and maximum is important (e.g. for scalarizing solutions)
 * @see compute_objective_min_max_ranges
 */
std::vector<double> compute_objective_ranges(Population &population,
                                             std::vector<size_t> objective_indices,
                                             std::vector<Individual> &pool);

struct ObjectiveCluster
{
    std::vector<double> centroid;
    std::vector<Individual> members;

    // Objective?
    //   -1   : standard mixing mode - i.e. no change.
    //  0...n : single-objective for objective x
    long mixing_mode = -1;
};

struct ClusterIndices
{
    std::vector<size_t> indices;
};
const std::shared_ptr<IDataType> CLUSTERINDICES =
    std::make_shared<DataType<ClusterIndices>>(typeid(ClusterIndices).name());

struct ClusterIndex
{
    size_t cluster_index;
};
const std::shared_ptr<IDataType> CLUSTERINDEX = std::make_shared<DataType<ClusterIndex>>(typeid(ClusterIndex).name());

// Perform clustering over a subpopulation using Objective Space clustering
// using a variant of k-means clustering with overlap.

std::vector<ObjectiveCluster> cluster_mo_gomea(Population &pop,
                                               std::vector<Individual> &pool,
                                               std::vector<size_t> objective_indices,
                                               std::vector<double> objective_ranges,
                                               size_t number_of_clusters);

/**
 * @brief Ensure objective ranges are always valid by replacing invalid ones.
 * 
 * @param objective_ranges The objective ranges to apply this to.
 */
void ensure_valid_ranges(std::vector<double> &objective_ranges);

void determine_extreme_clusters(std::vector<size_t> &objective_indices, std::vector<ObjectiveCluster> &clusters);

void determine_cluster_to_use_mo_gomea(Population &population,
                                       std::vector<ObjectiveCluster> &clusters,
                                       std::vector<Individual> &pool,
                                       std::vector<size_t> &objective_indices,
                                       std::vector<double> &objective_ranges);

struct UseSingleObjective
{
    // Index of objective.
    long index;
};
const std::shared_ptr<IDataType> USESINGLEOBJECTIVE =
    std::make_shared<DataType<UseSingleObjective>>(typeid(UseSingleObjective).name());

class WrappedOrSingleSolutionPerformanceCriterion : public IPerformanceCriterion
{
  public:
    WrappedOrSingleSolutionPerformanceCriterion(std::shared_ptr<IPerformanceCriterion> wrapped) : wrapped(wrapped)
    {
    }

    short compare(Individual &a, Individual &b) override
    {
        doCache();
        // Note: based of a!
        long obj_index = cache->guso.getData(a).index;
        if (obj_index == -1)
        {
            short r = wrapped->compare(a, b);
            // std::cout << "Performed multi-objective test: " << r << " (obj " << obj_index << ")\n";
            return r;
        }
        else
        {
            auto soac = SingleObjectiveAcceptanceCriterion(obj_index);
            soac.setPopulation(population);
            short r = soac.compare(a, b);
            // std::cout << "Performed single-objective test: " << r << " (obj " << obj_index << ")\n";
            return r;
        }
    }

    void doCache()
    {
        if (cache.has_value())
            return;
        auto guso = (*population).getDataContainer<UseSingleObjective>();
        cache.emplace(Cache{guso});
    }

    void setPopulation(std::shared_ptr<Population> population) override
    {
        IPerformanceCriterion::setPopulation(population);
        wrapped->setPopulation(population);
    }

    void registerData() override
    {
        wrapped->registerData();
    }
    void afterRegisterData() override
    {
        wrapped->afterRegisterData();
    }

  private:
    struct Cache
    {
        TypedGetter<UseSingleObjective> guso;
    };
    std::optional<Cache> cache;

    std::shared_ptr<IPerformanceCriterion> wrapped;
};

class MO_GOMEA : public BaseGOMEA
{
  private:
    size_t number_of_clusters;
    std::vector<size_t> objective_indices;
    const std::shared_ptr<FoSLearner> foslearner;
    const std::shared_ptr<GOMEAPlugin> plugin;
    bool donor_search = true;
    Rng *rng;

    void initialize() override;

  public:
    MO_GOMEA(size_t population_size,
             size_t number_of_clusters,
             std::vector<size_t> objective_indices,
             std::shared_ptr<ISolutionInitializer> initializer,
             std::shared_ptr<FoSLearner> foslearner,
             std::shared_ptr<IPerformanceCriterion> performance_criterion,
             std::shared_ptr<IArchive> archive,
             std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
             bool donor_search = true,
             bool autowrap = true);

    struct GenerationalData
    {
        std::vector<std::shared_ptr<FoSLearner>> per_cluster_fos;
        std::vector<Individual> originals;
        std::vector<std::vector<Individual>> per_cluster_originals;
    };
    std::optional<GenerationalData> g;

    void atGenerationStart() override;

    APtr<FoS> getFOSForGOM(Individual &ii) override;
    APtr<FoS> getFOSForFI(Individual &ii) override;
    APtr<ISamplingDistribution> getDistributionForGOM(Individual &ii) override;
    APtr<ISamplingDistribution> getDistributionForFI(Individual & /* ii */) override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForGOM(Individual & /* ii */) override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForFI(Individual & /* ii */) override;
    Individual &getReplacementSolution(const Individual & /* ii */) override;

    void atGenerationEnd() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;
};

struct LinkageKernel
{
    std::vector<Individual> neighborhood;
    std::vector<Individual> pop_neighborhood;
    std::unique_ptr<FoSLearner> fos;

    LinkageKernel()
    {
    }

    LinkageKernel(LinkageKernel &&o)
    {
        *this = o;
    }

    LinkageKernel(const LinkageKernel &o)
    {
        neighborhood = o.neighborhood;
        pop_neighborhood = o.pop_neighborhood;
        if (o.fos != nullptr)
            fos = std::unique_ptr<FoSLearner>(o.fos->cloned_ptr());
        else
            fos = nullptr;
    }

    void operator=(const LinkageKernel &)
    {
        // Linkage Kernel is NOT copied, i.e. it should not be changed when a solution is improved,
        // or backed up.
    }

    void operator=(LinkageKernel &&o)
    {
        // Move is performed through swaps!
        neighborhood.swap(o.neighborhood);
        pop_neighborhood.swap(o.pop_neighborhood);
        fos.swap(o.fos);
    }
};
const std::shared_ptr<IDataType> LINKAGEKERNEL =
    std::make_shared<DataType<LinkageKernel>>(typeid(LinkageKernel).name());

class KernelGOMEA : public BaseGOMEA
{
  private:
    size_t number_of_clusters;
    std::vector<size_t> objective_indices;
    const std::shared_ptr<FoSLearner> foslearner;
    const std::shared_ptr<GOMEAPlugin> plugin;
    bool donor_search = true;
    Rng *rng;

    void initialize() override;

  public:
    KernelGOMEA(size_t population_size,
                size_t number_of_clusters,
                std::vector<size_t> objective_indices,
                std::shared_ptr<ISolutionInitializer> initializer,
                std::shared_ptr<FoSLearner> foslearner,
                std::shared_ptr<IPerformanceCriterion> performance_criterion,
                std::shared_ptr<IArchive> archive,
                std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                bool donor_search = true,
                bool autowrap = true);

    struct GenerationalData
    {
        std::vector<Individual> originals;
        TypedGetter<LinkageKernel> lk;
    };
    std::optional<GenerationalData> g;

    void atGenerationStart() override;

    APtr<FoS> getFOSForGOM(Individual &ii) override;
    APtr<FoS> getFOSForFI(Individual &ii) override;
    APtr<ISamplingDistribution> getDistributionForGOM(Individual &ii) override;
    APtr<ISamplingDistribution> getDistributionForFI(Individual & /* ii */) override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForGOM(Individual & /* ii */) override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForFI(Individual & /* ii */) override;
    Individual &getReplacementSolution(const Individual & /* ii */) override;

    void atGenerationEnd() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;
};
