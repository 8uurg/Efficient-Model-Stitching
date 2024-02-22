#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <queue>
#include <tuple>

#include "archive.hpp"
#include "base.hpp"
#include "cppassert.h"
#include "decentralized.hpp"
#include "gomea.hpp"
#include "running.hpp"

// TODOs
// [ ] What to do with unevaluated solutions for Linkage Learning & other aspects?
// [x] Set up new stepping mechanism / loop for the Synchronous GOMEAs.
// [x] Set up new stepping mechanism / loop for the Asynchronous GOMEAs.
// [x] Optional: Wrap the scheduling & event mechanism such that the implementation does not
//     assume gRPC.
// [x] Alter the Synchronous setup such that it actually does more than just initialize.

// -- State-Machine based operations --

/**
 * @brief State machine based GOM
 */
class StateMachineGOM : public IResumable
{
  private:
    Population &population;
    Individual to_improve;
    DroppingIndividual backup;
    DroppingIndividual current;
    APtr<FoS> fos;
    APtr<ISamplingDistribution> distribution;
    APtr<IPerformanceCriterion> acceptance_criterion;
    std::optional<std::exception_ptr> maybe_exception;
    std::function<void(Individual &, Individual &)> replace_in_population;
    size_t completion_tag;

    size_t idx = 0;
    bool steadystate;

  public:
    StateMachineGOM(Population &population,
                    Individual to_improve,
                    APtr<FoS> fos,
                    APtr<ISamplingDistribution> distribution,
                    APtr<IPerformanceCriterion> acceptance_criterion,
                    std::function<void(Individual &, Individual &)> replace_in_population,
                    size_t completion_tag,
                    bool steadystate,
                    bool *changed,
                    bool *improved,
                    bool *limited);

    bool *changed;
    bool *improved;
    bool *limited;

    static std::unique_ptr<StateMachineGOM> apply(Population &population,
                                                  Individual to_improve,
                                                  APtr<FoS> fos,
                                                  APtr<ISamplingDistribution> distribution,
                                                  APtr<IPerformanceCriterion> acceptance_criterion,
                                                  std::function<void(Individual &, Individual &)> replace_in_population,
                                                  size_t completion_tag,
                                                  bool steadystate,
                                                  bool *changed,
                                                  bool *improved,
                                                  bool *limited);

    void evaluate_change(Individual current, Individual /* backup */, std::vector<size_t> & /* elements_changed */);

    bool resume(Scheduler &wd, std::unique_ptr<IResumable> &resumable) override;
};

/**
 * @brief State machine based FI
 */
class StateMachineFI : public IResumable
{
  private:
    Population &population;
    Individual to_improve;
    DroppingIndividual backup;
    DroppingIndividual current;
    APtr<FoS> fos;
    APtr<ISamplingDistribution> distribution;
    APtr<IPerformanceCriterion> acceptance_criterion;
    std::optional<std::exception_ptr> maybe_exception;
    std::function<void(Individual &, Individual &)> replace_in_population;
    size_t completion_tag;

    size_t idx = 0;
    bool success = false;

  public:
    bool *changed;
    bool *improved;
    bool *limited;

    StateMachineFI(Population &population,
                   Individual to_improve,
                   APtr<FoS> fos,
                   APtr<ISamplingDistribution> distribution,
                   APtr<IPerformanceCriterion> acceptance_criterion,
                   std::function<void(Individual &, Individual &)> replace_in_population,
                   size_t completion_tag,
                   bool *changed,
                   bool *improved,
                   bool *limited);

    static std::unique_ptr<StateMachineFI> apply(Population &population,
                                                 Individual to_improve,
                                                 APtr<FoS> fos,
                                                 APtr<ISamplingDistribution> distribution,
                                                 APtr<IPerformanceCriterion> acceptance_criterion,
                                                 std::function<void(Individual &, Individual &)> replace_in_population,
                                                 size_t completion_tag,
                                                 bool *changed,
                                                 bool *improved,
                                                 bool *limited);

    void evaluate_change(Individual current, Individual /* backup */, std::vector<size_t> & /* elements_changed */);

    bool resume(Scheduler &wd, std::unique_ptr<IResumable> &resumable) override;
};

class IGOMFIData
{
  public:
    virtual ~IGOMFIData() = default;

    virtual APtr<FoS> getFOSForGOM() = 0;
    virtual APtr<ISamplingDistribution> getDistributionForGOM() = 0;
    virtual APtr<IPerformanceCriterion> getPerformanceCriterionForGOM() = 0;
    virtual APtr<FoS> getFOSForFI() = 0;
    virtual APtr<ISamplingDistribution> getDistributionForFI() = 0;
    virtual APtr<IPerformanceCriterion> getPerformanceCriterionForFI() = 0;
};

class DistributedGOMThenMaybeFI : public IResumable
{
  private:
    enum State
    {
        Start = 0,
        GOM = 1,
        FI = 2,
        Completed = 3,
    };

    std::queue<std::tuple<double, std::unique_ptr<IResumable>, std::optional<std::string>>> enqueued;
    State state = Start;

    Population &population;
    Individual to_improve;
    std::unique_ptr<IGOMFIData> kernel_data;
    std::unique_ptr<bool> changed;
    std::unique_ptr<bool> improved;
    std::function<size_t(Individual &ii)> getNISThreshold;
    std::function<Individual(Individual &ii)> getReplacementSolution;
    std::function<void(Individual &, Individual &)> replace_in_population;

    std::function<void(Scheduler &scheduler)> onCompletion;
    size_t completion_tag;
    bool steadystate;
    std::unique_ptr<bool> limited;

  public:
    DistributedGOMThenMaybeFI(Population &population,
                              Individual to_improve,
                              size_t completion_tag,
                              std::function<size_t(Individual &ii)> getNISThreshold,
                              std::function<Individual(Individual &ii)> getReplacementSolution,
                              std::function<void(Individual &, Individual &)> replace_in_population,
                              std::function<void(Scheduler &wd)> onCompletion,
                              std::unique_ptr<IGOMFIData> kernel_data,
                              bool steadystate);

    bool resume(Scheduler &wd, std::unique_ptr<IResumable> &resumable) override;
    bool progress(Scheduler &wd, std::unique_ptr<IResumable> &resumable);

    void onImprovedSolution(Individual &ii);
};

// Simulated Parallel versions of Generational GOMEA

class GOMFIDataBaseGOMEA : public IGOMFIData
{
  private:
    BaseGOMEA *baseGOMEA;
    Individual ii;

  public:
    GOMFIDataBaseGOMEA(BaseGOMEA *baseGOMEA, Individual ii) : baseGOMEA(baseGOMEA), ii(ii)
    {
    }

    APtr<FoS> getFOSForGOM() override
    {
        return baseGOMEA->getFOSForGOM(ii);
    }
    APtr<ISamplingDistribution> getDistributionForGOM() override
    {
        return baseGOMEA->getDistributionForGOM(ii);
    }
    APtr<IPerformanceCriterion> getPerformanceCriterionForGOM() override
    {
        return baseGOMEA->getPerformanceCriterionForGOM(ii);
    }
    APtr<FoS> getFOSForFI() override
    {
        return baseGOMEA->getFOSForFI(ii);
    }
    APtr<ISamplingDistribution> getDistributionForFI() override
    {
        return baseGOMEA->getDistributionForFI(ii);
    }
    APtr<IPerformanceCriterion> getPerformanceCriterionForFI() override
    {
        return baseGOMEA->getPerformanceCriterionForFI(ii);
    }
};

// -- Synchronous --

/**
 * @brief A simulated Parallel GOMEA
 *
 * This implementation uses a simulator to simulate a parallel version
 * of GOMEA. Note that time is progressed through the use of a simulator
 * providing total control on time spent, and simulate a hypothetical
 * setup with many more cores than what we have available in reality.
 *
 * Furthermore, actual waiting time due to parallel progression is halted.
 */
class DistributedSynchronousGOMEA : public GOMEA
{
  private:
    std::shared_ptr<Scheduler> wd;
    bool steadystate;

  protected:
    void step_normal_generation() override;
    void initialize() override;
    void replace_population_individual(size_t idx, Individual replacement, Individual in_population);

    void improveSolution(size_t idx, Individual ii, Scheduler &wd);

  public:
    DistributedSynchronousGOMEA(std::shared_ptr<Scheduler> sp,
                                size_t population_size,
                                std::shared_ptr<ISolutionInitializer> initializer,
                                std::shared_ptr<FoSLearner> foslearner,
                                std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                std::shared_ptr<IArchive> archive,
                                bool autowrap = true,
                                bool steadystate = false,
             std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
                 gomSamplingDistributionFactory = DefaultGOMSamplingDistributionFactory(),
             std::function<APtr<ISamplingDistribution>(Population &, GenerationalData &, BaseGOMEA *)>
                 fiSamplingDistributionFactory = DefaultFISamplingDistributionFactory());

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void step() override
    {
        if (!initialized)
        {
            initialize();
            initialized = true;
        }
        else
        {
            wd->step();
        }
    }
};

/**
 * @brief A simulated Parallel MO-GOMEA
 *
 * For the reasoning behind writing a simulator @sa DistributedParallelSynchronousGOMEA
 */
class DistributedSynchronousMO_GOMEA : public MO_GOMEA
{
  private:
    std::shared_ptr<Scheduler> wd;
    std::shared_ptr<SharedgRPCCompletionQueue> scq;
    bool steadystate;

  protected:
    void step_normal_generation() override;
    void initialize() override;

    void improveSolution(size_t idx, Individual ii, Scheduler &wd);

  public:
    DistributedSynchronousMO_GOMEA(std::shared_ptr<Scheduler> wd,
                                   size_t population_size,
                                   size_t number_of_clusters,
                                   std::vector<size_t> objective_indices,
                                   std::shared_ptr<ISolutionInitializer> initializer,
                                   std::shared_ptr<FoSLearner> foslearner,
                                   std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                   std::shared_ptr<IArchive> archive,
                                   std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                   bool donor_search = true,
                                   bool autowrap = true,
                                   bool steadystate = false);

    void replace_population_individual(size_t idx, Individual replacement, Individual in_population);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void step() override
    {
        if (!initialized)
        {
            initialize();
            initialized = true;
        }
        else
        {
            wd->step();
        }
    }
};

/**
 * @brief A simulated Parallel LK-GOMEA
 *
 * For the reasoning behind writing a simulator @sa DistributedParallelSynchronousGOMEA
 */
class DistributedSynchronousKernelGOMEA : public KernelGOMEA
{
  private:
    std::shared_ptr<Scheduler> wd;
    bool steadystate;

  protected:
    void step_normal_generation() override;
    void initialize() override;

    void improveSolution(size_t idx, Individual ii, Scheduler &wd);

  public:
    DistributedSynchronousKernelGOMEA(std::shared_ptr<Scheduler> wd,
                                      size_t population_size,
                                      size_t number_of_clusters,
                                      std::vector<size_t> objective_indices,
                                      std::shared_ptr<ISolutionInitializer> initializer,
                                      std::shared_ptr<FoSLearner> foslearner,
                                      std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                      std::shared_ptr<IArchive> archive,
                                      std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                      bool donor_search = true,
                                      bool autowrap = true,
                                      bool steadystate = false);

    void replace_population_individual(size_t idx, Individual replacement, Individual in_population);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void step() override
    {
        if (!initialized)
        {
            initialize();
            initialized = true;
        }
        else
        {
            wd->step();
        }
    }
};

// -- Asynchronous --

/**
 * @brief An asynchronous variant of LK-GOMEA
 *
 * A generational - synchronous - implementation has to wait until
 * all evaluations have been performed, leading to potentially unbounded
 * waiting times, and significant underutilization of the available
 * computational resources.
 *
 * A potential approach to increase the utilization of resources in
 * a parallel evolutionary algorithm is to make it asynchronous, i.e.
 * to drop the generational constraint and operate in a steady-state
 * like fashion.
 *
 * This was a simulator for LK-GOMEA which removes the generational constraint,
 * and as such is faced with a few additional complicating factors.
 *
 * - (Parts of) population are potentially unevaluated.
 * - There is no synchronous point at which we can perform particular actions
 *   which makes multi-individual altercations more difficult.
 *   Examples (and how we have resolved these issues for now):
 *
 *   - (Perform linkage learning)
 *     - Already resolved as each linkage kernel requires to learn their
 *       own linkage model.
 *
 *   - Determining single-objective clusters
 *     1. Cluster over copies for each kernel.
 *     2. Copy assignment over to current kernel.
 *
 *   - Assigning scalarization directions to solutions (not implemented)
 *
 */
class DistributedAsynchronousBaseGOMEA : public GenerationalApproach
{
  protected:
    size_t population_size;
    size_t number_of_clusters;
    bool donor_search;
    std::vector<size_t> objective_indices;
    std::shared_ptr<ISolutionInitializer> initializer;
    std::shared_ptr<FoSLearner> foslearner;
    std::shared_ptr<IPerformanceCriterion> performance_criterion;
    std::shared_ptr<IArchive> archive;
    std::shared_ptr<Scheduler> wd;
    Rng *rng;

    bool initialized = false;
    std::vector<Individual> individuals;
    bool steadystate;
    bool sample_from_copies;
    // std::priority_queue<std::tuple<float, Individual, Individual>> event_queue;

    virtual std::unique_ptr<IGOMFIData> learnKernel(const Individual &ii) = 0;

  protected:
    void improveSolution(size_t idx, const Individual &ii, Scheduler &wd);
    void onImprovedSolution(Individual &ii);
    Individual &getReplacementSolution(const Individual & /* ii */);

    size_t getNISThreshold(const Individual & /* ii */);

  public:
    DistributedAsynchronousBaseGOMEA(std::shared_ptr<Scheduler> wd,
                                       size_t population_size,
                                       size_t number_of_clusters,
                                       std::vector<size_t> objective_indices,
                                       std::shared_ptr<ISolutionInitializer> initializer,
                                       std::shared_ptr<FoSLearner> foslearner,
                                       std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                       std::shared_ptr<IArchive> archive,
                                       // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                       bool donor_search = true,
                                       bool autowrap = true,
                                       bool steadystate = false,
                                       bool sample_from_copies = true);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void replace_population_individual(size_t idx, Individual replacement, Individual in_population);

    APtr<ISamplingDistribution> getDistributionForFI(Individual & /* ii */);

    void initialize();

    void run();
    void step() override
    {
        if (!initialized)
        {
            initialize();
            initialized = true;
        }
        else
        {
            wd->step();
        }
    }

    std::vector<Individual> &getSolutionPopulation() override
    {
        return individuals;
    }
    bool terminated() override
    {
        // If there are no more events left to process & no more things scheduled to be
        // processed (all that is left is those waiting for a change...) we have terminated.
        // Sidenote: we may want to keep track of this locally, such that a population sizing
        // scheme can be used, and the actual event loop can take place elsewhere.
        return wd->terminated();
    }
};

enum LKSimpleStrategy
{
    SQRT_SYM = 0,
    SQRT_ASYM = 1,
    RAND_SYM = 2,
    RAND_ASYM = 3,
    RAND_INT_SYM = 4,
    RAND_INT_ASYM = 5,
};

class LKStrategy
{
  public:
    virtual std::vector<std::vector<size_t>> determine_lk_neighborhood(
        Population &population,
        std::vector<Individual> & individuals,
        std::vector<size_t> &subset,
        size_t population_size
    ) = 0;
};

class SimpleStrategy: public LKStrategy
{

  LKSimpleStrategy strategy;
  double distance_threshold;

  public:

    /**
     * @brief Construct a new Simple Strategy.
     * 
     * @param strategy The strategy for picking k.
     * @param distance_threshold The threshold for the distance function such that it counts towards k.
     *    def
     */
    SimpleStrategy(
      LKSimpleStrategy strategy,
      double distance_threshold = -std::numeric_limits<double>::infinity()) : 
      strategy(strategy),
      distance_threshold(distance_threshold)
    {
    }

    std::vector<std::vector<size_t>> determine_lk_neighborhood(
        Population &population,
        std::vector<Individual> &individuals,
        std::vector<size_t> &subset,
        size_t population_size) override {
        
        auto rng = population.getGlobalData<Rng>().get();

        TypedGetter<GenotypeCategorical> gc = population.getDataContainer<GenotypeCategorical>();
        
        bool symmetric = (strategy == SQRT_SYM || strategy == RAND_INT_SYM);

        size_t k;
        if (strategy == RAND_INT_SYM || strategy == RAND_INT_ASYM) {
            // Minimum neighborhood size should be reasonable enough!
            size_t min_neighborhood_size = 4;
            double min_neighborhood_size_dbl = static_cast<double>(min_neighborhood_size);
            double population_size_dbl = static_cast<double>(population_size);
            double max_modes_dbl = std::ceil(population_size_dbl / min_neighborhood_size_dbl);
            size_t max_modes_st = static_cast<size_t>(max_modes_dbl);
            std::uniform_int_distribution<size_t> dst(1, max_modes_st);
            size_t num_modes_picked = dst(rng->rng);
            double num_modes_picked_dbl = static_cast<double>(num_modes_picked);
            double k_dbl = std::ceil(population_size_dbl / num_modes_picked_dbl);
            k = static_cast<size_t>(k_dbl);
        }
        else if (strategy == RAND_SYM || strategy == RAND_ASYM) {
            // Minimum neighborhood size should be reasonable enough!
            size_t min_neighborhood_size = 32;
            double min_neighborhood_size_dbl = static_cast<double>(min_neighborhood_size);
            double population_size_dbl = static_cast<double>(population_size);
            double max_modes_dbl = population_size_dbl / min_neighborhood_size_dbl;
            std::uniform_real_distribution<double> dst(1, max_modes_dbl);
            double num_modes_picked = dst(rng->rng);
            double k_dbl = std::ceil(population_size_dbl / num_modes_picked);
            k = static_cast<size_t>(k_dbl);
        }
        else // (strategy == SQRT_SYM || strategy == SQRT_ASYM)
        {
            // Also used as fallback.
            k = static_cast<size_t>(std::ceil(std::sqrt(population_size)));
        }
        // std::cerr << "using neighborhood of size " << k << " for kernel of solution " << i.i << "." << std::endl;
        
        std::vector<size_t> indices(individuals.size());
        std::iota(indices.begin(), indices.end(), 0);

        std::vector<std::vector<size_t>> neighborhoods;
        if (symmetric) {
            auto neighborhood_indices = getNeighborhoods(
                [&individuals, &gc](size_t a, size_t b) {
                    GenotypeCategorical gca = gc.getData(individuals[a]);
                    GenotypeCategorical gcb = gc.getData(individuals[b]);

                    return hamming_distance(
                        gca.genotype.data(), gcb.genotype.data(), std::min(gca.genotype.size(), gcb.genotype.size()));
                },
                *rng,
                indices,
                k,
                symmetric,
                distance_threshold);

            // Filter the neighborhood.
            for (size_t i = 0; i < subset.size(); ++i)
            {
                size_t j = subset[i];
                neighborhoods.push_back(std::move(neighborhood_indices[j]));
            }
        }
        else
        {
            for (size_t i = 0; i < subset.size(); ++i)
            {
                size_t idx = subset[i];

                auto idx_neighborhood_indices = getNeighborhood(
                  [&individuals, &gc](size_t a, size_t b) {
                      GenotypeCategorical gca = gc.getData(individuals[a]);
                      GenotypeCategorical gcb = gc.getData(individuals[b]);

                      return hamming_distance(
                          gca.genotype.data(), gcb.genotype.data(), std::min(gca.genotype.size(), gcb.genotype.size()));
                  },
                  *rng,
                  idx,
                  indices,
                  k,
                  symmetric,
                  distance_threshold);

                neighborhoods.push_back(std::move(idx_neighborhood_indices));
            }
        }
        return neighborhoods;
    }
};


/**
 * @brief An asynchronous variant of LK-GOMEA
 *
 * A generational - synchronous - implementation has to wait until
 * all evaluations have been performed, leading to potentially unbounded
 * waiting times, and significant underutilization of the available
 * computational resources.
 *
 * A potential approach to increase the utilization of resources in
 * a parallel evolutionary algorithm is to make it asynchronous, i.e.
 * to drop the generational constraint and operate in a steady-state
 * like fashion.
 *
 * This was a simulator for LK-GOMEA which removes the generational constraint,
 * and as such is faced with a few additional complicating factors.
 *
 * - (Parts of) population are potentially unevaluated.
 * - There is no synchronous point at which we can perform particular actions
 *   which makes multi-individual altercations more difficult.
 *   Examples (and how we have resolved these issues for now):
 *
 *   - (Perform linkage learning)
 *     - Already resolved as each linkage kernel requires to learn their
 *       own linkage model.
 *
 *   - Determining single-objective clusters
 *     1. Cluster over copies for each kernel.
 *     2. Copy assignment over to current kernel.
 *
 *   - Assigning scalarization directions to solutions (not implemented)
 *
 */
class DistributedAsynchronousKernelGOMEA : public DistributedAsynchronousBaseGOMEA
{
  protected:
    std::unique_ptr<IGOMFIData> learnKernel(const Individual &ii) override;

  public:
    std::shared_ptr<LKStrategy> lkg_strategy;

    DistributedAsynchronousKernelGOMEA(std::shared_ptr<Scheduler> wd,
                                       size_t population_size,
                                       size_t number_of_clusters,
                                       std::vector<size_t> objective_indices,
                                       std::shared_ptr<ISolutionInitializer> initializer,
                                       std::shared_ptr<FoSLearner> foslearner,
                                       std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                       std::shared_ptr<IArchive> archive,
                                       // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                       bool donor_search = true,
                                       bool autowrap = true,
                                       bool steadystate = false,
                                       bool sample_from_copies = true,
                                       std::optional<std::shared_ptr<LKStrategy>> lkg_strategy = std::nullopt
                                       );
};

/**
 * @brief Asynchronous GOMEA
 *
 * One of the key issues with the aforementioned approach is that many computational resources
 * are spent on linkage learning. Yet, with the removal of the generational barrier, things do
 * become more difficult to comply with the design of the original GOMEA.
 *
 * 1. The FOS should not change while GOM / FI is being performed.
 *   > Kernel GOMEA does not have this problem as all FOS models are separate. Going back to a
 *   > single global model (as originally the case) would not work: we need to force a wait in
 *   > order to be able to update a single global model.
 * 2. When should the model be updated in the first place?
 *   > Traditionally, GOMEA updates its model at the start of a generation. When dropping the
 *   > generational barrier, this point does no longer exist. Kernel GOMEA sidesteps this
 *   > issue by learning a model at the start of GOM. But how can this approach resolve this
 *   > issue?
 *
 * The key idea here is to re-use Kernel GOMEA, and make it cheaper by using a single (global)
 * model:
 * - Rather than using the neighborhood, we always use the full population.
 * - We learn the model once-in-a-while:
 *   - A kernel is 'learnt' before starting GOM, consequently, we can count the number of GOM
 *     applications by counting the number of kernels 'learnt'.
 *   - A traditional synchronous generation consists of `population_size` such steps.
 *   - Learn a new global model every `population_size` times a kernel is 'learnt'.
 * - In any case: Copy over the global model as the kernel.
 *
 * As a sidenote: after applying GOM the population does immediately update, leading to some
 * kind of steady-state GOMEA, similar to what Kernel GOMEA is doing. Dropping the generational
 * barrier here, too, leads to a patch that needs to be applied.
 */
class DistributedAsynchronousGOMEA : public DistributedAsynchronousBaseGOMEA
{
  private:
    long long num_kernel_usages_left = 0;

  protected:
    std::unique_ptr<IGOMFIData> learnKernel(const Individual &ii) override;

  public:
    DistributedAsynchronousGOMEA(std::shared_ptr<Scheduler> wd,
                                         size_t population_size,
                                         size_t number_of_clusters,
                                         std::vector<size_t> objective_indices,
                                         std::shared_ptr<ISolutionInitializer> initializer,
                                         std::shared_ptr<FoSLearner> foslearner,
                                         std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                         std::shared_ptr<IArchive> archive,
                                         // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                         bool donor_search = true,
                                         bool autowrap = true,
                                         bool steadystate = false,
                                         bool sample_from_copies = true);

    void registerData() override;
};

/**
 * @brief Kernel associated data
 */
struct AsyncKernelData : public IGOMFIData
{
    std::vector<DroppingIndividual> copies;
    APtr<FoS> fos;
    APtr<ISamplingDistribution> isd;
    APtr<IPerformanceCriterion> ipc;
    DistributedAsynchronousBaseGOMEA *context;
    Individual ii;

    AsyncKernelData(std::vector<DroppingIndividual> &&copies,
                    APtr<FoS> fos,
                    APtr<ISamplingDistribution> isd,
                    APtr<IPerformanceCriterion> ipc,
                    DistributedAsynchronousBaseGOMEA *context,
                    Individual ii);

    APtr<FoS> getFOSForGOM() override;
    APtr<ISamplingDistribution> getDistributionForGOM() override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForGOM() override;
    APtr<FoS> getFOSForFI() override;
    APtr<ISamplingDistribution> getDistributionForFI() override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForFI() override;
};
