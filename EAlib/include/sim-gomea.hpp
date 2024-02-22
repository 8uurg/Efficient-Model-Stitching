#pragma once

#include <chrono>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <queue>
#include <tuple>

#include "archive.hpp"
#include "base.hpp"
#include "cppassert.h"
#include "gomea.hpp"
#include "running.hpp"
#include "sim.hpp"

// TODOs
// [x] Break on time limit
// [x] Account for evaluations during initialization (sync)
// [ ] Account for overhead (i.e. time spent during linkage learning)
// [x] Finish up asynchronous kernel
// [x] Account for evaluations during initialization (async)
// [ ] What to do with unevaluated solutions for Linkage Learning & other aspects?
// [x] Deal with no-evaluations-are-taking-place kernels.

// -- State-Machine based operations --

/**
 * @brief State machine based GOM
 */
class SimulatedGOM : public IResumableSimulated
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

    size_t idx = 0;

  public:
    SimulatedGOM(Population &population,
                 Individual to_improve,
                 APtr<FoS> fos,
                 APtr<ISamplingDistribution> distribution,
                 APtr<IPerformanceCriterion> acceptance_criterion,
                 std::function<void(Individual &, Individual &)> replace_in_population,
                 bool *changed,
                 bool *improved);

    bool *changed;
    bool *improved;

    static std::unique_ptr<SimulatedGOM> apply(Population &population,
                                               Individual to_improve,
                                               APtr<FoS> fos,
                                               APtr<ISamplingDistribution> distribution,
                                               APtr<IPerformanceCriterion> acceptance_criterion,
                                               std::function<void(Individual &, Individual &)> replace_in_population,
                                               bool *changed,
                                               bool *improved);

    void evaluate_change(Individual current, Individual /* backup */, std::vector<size_t> & /* elements_changed */);

    void resume(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self) override;
};

/**
 * @brief State machine based FI
 */
class SimulatedFI : public IResumableSimulated
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

    size_t idx = 0;
    bool success = false;

  public:
    bool *changed;
    bool *improved;

    SimulatedFI(Population &population,
                Individual to_improve,
                APtr<FoS> fos,
                APtr<ISamplingDistribution> distribution,
                APtr<IPerformanceCriterion> acceptance_criterion,
                std::function<void(Individual &, Individual &)> replace_in_population,
                bool *changed,
                bool *improved);

    static std::unique_ptr<SimulatedFI> apply(Population &population,
                                              Individual to_improve,
                                              APtr<FoS> fos,
                                              APtr<ISamplingDistribution> distribution,
                                              APtr<IPerformanceCriterion> acceptance_criterion,
                                              std::function<void(Individual &, Individual &)> replace_in_population,
                                              bool *changed,
                                              bool *improved);

    void evaluate_change(Individual current, Individual /* backup */, std::vector<size_t> & /* elements_changed */);

    void resume(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self) override;
};

class SimIGOMFIData
{
  public:
    virtual ~SimIGOMFIData() = default;

    virtual APtr<FoS> getFOSForGOM() = 0;
    virtual APtr<ISamplingDistribution> getDistributionForGOM() = 0;
    virtual APtr<IPerformanceCriterion> getPerformanceCriterionForGOM() = 0;
    virtual APtr<FoS> getFOSForFI() = 0;
    virtual APtr<ISamplingDistribution> getDistributionForFI() = 0;
    virtual APtr<IPerformanceCriterion> getPerformanceCriterionForFI() = 0;
};

class SimulatedGOMThenMaybeFI : public IResumableSimulated, public ISimulator
{
  private:
    enum State
    {
        GOM = 1,
        FI = 2,
        Completed = 3,
    };

    std::queue<std::tuple<double, std::unique_ptr<IResumableSimulated>, std::optional<std::string>>> enqueued;
    State state = GOM;

    Population &population;
    Individual to_improve;
    std::unique_ptr<SimIGOMFIData> kernel_data;
    std::unique_ptr<bool> changed;
    std::unique_ptr<bool> improved;
    std::function<size_t(Individual &ii)> getNISThreshold;
    std::function<Individual(Individual &ii)> getReplacementSolution;
    std::function<void(Individual &, Individual &)> replace_in_population;

    std::function<void(ISimulator &simulator, double)> onCompletion;

  public:
    SimulatedGOMThenMaybeFI(Population &population,
                            Individual to_improve,
                            std::function<size_t(Individual &ii)> getNISThreshold,
                            std::function<Individual(Individual &ii)> getReplacementSolution,
                            std::function<void(Individual &, Individual &)> replace_in_population,
                            std::function<void(ISimulator &simulator, double)> onCompletion,
                            std::unique_ptr<SimIGOMFIData> kernel_data);

    void resume(ISimulator &simulator, double at, std::unique_ptr<IResumableSimulated> &self) override;

    void insert_event(std::unique_ptr<IResumableSimulated> e, double at, std::optional<std::string>) override;

    void onImprovedSolution(Individual &ii);
};

// Simulated Parallel versions of Generational GOMEA

/**
 * @brief A wrapper around BaseGOMEA for use in simulations.
 *
 * In partciular, the original implementation has the entire application
 * of GOM and FI within a method of BaseGOMEA. With the usage of state
 * machines instead, this inclusion is more difficult, and some interface
 * is used to provide this data instead.
 *
 * This class is an implementation that wraps an implementation of GOMEA
 * for forwarding these calls from a state machine.
 */
class SimGOMFIDataBaseGOMEA : public SimIGOMFIData
{
  private:
    BaseGOMEA *baseGOMEA;
    Individual ii;

  public:
    SimGOMFIDataBaseGOMEA(BaseGOMEA *baseGOMEA, Individual ii) : baseGOMEA(baseGOMEA), ii(ii)
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
class SimParallelSynchronousGOMEA : public GOMEA
{
  private:
    std::optional<TypedGetter<TimeSpent>> tgts;
    std::shared_ptr<SimulatorParameters> sp;
    std::shared_ptr<SpanLogger> sl;

  protected:
    void step_normal_generation() override;
    void initialize() override;
    void replace_population_individual(size_t idx, Individual replacement, Individual in_population);

    void improveSolution(size_t idx, Individual ii, Simulator &simulator);

  public:
    SimParallelSynchronousGOMEA(std::shared_ptr<SimulatorParameters> sp,
                                size_t population_size,
                                std::shared_ptr<ISolutionInitializer> initializer,
                                std::shared_ptr<FoSLearner> foslearner,
                                std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                std::shared_ptr<IArchive> archive,
                                std::shared_ptr<SpanLogger> sl = NULL,
                                bool donor_search = true,
                                bool autowrap = true);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;
};

/**
 * @brief A simulated Parallel MO-GOMEA
 *
 * For the reasoning behind writing a simulator @sa SimParallelSynchronousGOMEA
 */
class SimParallelSynchronousMO_GOMEA : public MO_GOMEA
{
  private:
    std::optional<TypedGetter<TimeSpent>> tgts;
    std::shared_ptr<SimulatorParameters> sp;
    std::shared_ptr<SpanLogger> sl;

  protected:
    void step_normal_generation() override;
    void initialize() override;

    void improveSolution(size_t idx, Individual ii, Simulator &simulator);

  public:
    SimParallelSynchronousMO_GOMEA(std::shared_ptr<SimulatorParameters> sp,
                                   size_t population_size,
                                   size_t number_of_clusters,
                                   std::vector<size_t> objective_indices,
                                   std::shared_ptr<ISolutionInitializer> initializer,
                                   std::shared_ptr<FoSLearner> foslearner,
                                   std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                   std::shared_ptr<IArchive> archive,
                                   std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                   std::shared_ptr<SpanLogger> sl = NULL,
                                   bool donor_search = true,
                                   bool autowrap = true);

    void replace_population_individual(size_t idx, Individual replacement, Individual in_population);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;
};

/**
 * @brief A simulated Parallel LK-GOMEA
 *
 * For the reasoning behind writing a simulator @sa SimParallelSynchronousGOMEA
 */
class SimParallelSynchronousKernelGOMEA : public KernelGOMEA
{
  private:
    std::optional<TypedGetter<TimeSpent>> tgts;
    std::shared_ptr<SimulatorParameters> sp;
    std::shared_ptr<SpanLogger> sl;

  protected:
    void step_normal_generation() override;
    void initialize() override;

    void improveSolution(size_t idx, Individual ii, Simulator &simulator);

  public:
    SimParallelSynchronousKernelGOMEA(std::shared_ptr<SimulatorParameters> sp,
                                      size_t population_size,
                                      size_t number_of_clusters,
                                      std::vector<size_t> objective_indices,
                                      std::shared_ptr<ISolutionInitializer> initializer,
                                      std::shared_ptr<FoSLearner> foslearner,
                                      std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                      std::shared_ptr<IArchive> archive,
                                      std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                      std::shared_ptr<SpanLogger> sl = NULL,
                                      bool donor_search = true,
                                      bool autowrap = true);

    void replace_population_individual(size_t idx, Individual replacement, Individual in_population);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;
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
 * This is a simulator for LK-GOMEA which removes the generational constraint,
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
class SimParallelAsynchronousKernelGOMEA : public GenerationalApproach
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
    std::shared_ptr<SimulatorParameters> sp;
    std::shared_ptr<SpanLogger> sl;
    Rng *rng;

    bool initialized = false;
    std::vector<Individual> individuals;
    // std::priority_queue<std::tuple<float, Individual, Individual>> event_queue;

    std::optional<TypedGetter<TimeSpent>> tgts;

    virtual std::unique_ptr<SimIGOMFIData> learnKernel(const Individual &ii);

  protected:
    void improveSolution(size_t idx, const Individual &ii, Simulator &simulator);
    void onImprovedSolution(Individual &ii);
    Individual &getReplacementSolution(const Individual & /* ii */);

    size_t getNISThreshold(const Individual & /* ii */);

  public:
    SimParallelAsynchronousKernelGOMEA(std::shared_ptr<SimulatorParameters> sp,
                                       size_t population_size,
                                       size_t number_of_clusters,
                                       std::vector<size_t> objective_indices,
                                       std::shared_ptr<ISolutionInitializer> initializer,
                                       std::shared_ptr<FoSLearner> foslearner,
                                       std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                       std::shared_ptr<IArchive> archive,
                                       // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                       std::shared_ptr<SpanLogger> sl = NULL,
                                       bool donor_search = true,
                                       bool autowrap = true);

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    void replace_population_individual(size_t idx, Individual replacement, Individual in_population);

    APtr<ISamplingDistribution> getDistributionForFI(Individual & /* ii */);

    void initialize();

    void run();
    void step() override;
    void step_usual();

    std::vector<Individual> &getSolutionPopulation() override
    {
        return individuals;
    }
    bool terminated() override
    {
        // If there are no more events left to process & no more things scheduled to be
        // processed (all that is left is those waiting for a change...) we have terminated.
        return sp->simulator->event_queue.empty() && sp->processing_queue.empty();
    }
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
class SimParallelAsynchronousGOMEA : public SimParallelAsynchronousKernelGOMEA
{
  private:
    long long num_kernel_usages_left = 0;
  protected:
    std::unique_ptr<SimIGOMFIData> learnKernel(const Individual &ii) override;

  public:
    SimParallelAsynchronousGOMEA(std::shared_ptr<SimulatorParameters> sp,
                                 size_t population_size,
                                 size_t number_of_clusters,
                                 std::vector<size_t> objective_indices,
                                 std::shared_ptr<ISolutionInitializer> initializer,
                                 std::shared_ptr<FoSLearner> foslearner,
                                 std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                 std::shared_ptr<IArchive> archive,
                                 // std::shared_ptr<GOMEAPlugin> plugin = std::make_shared<GOMEAPlugin>(),
                                 std::shared_ptr<SpanLogger> sl = NULL,
                                 bool donor_search = true,
                                 bool autowrap = true);
};

/**
 * @brief Kernel associated data
 */
struct SimAsyncKernelData : public SimIGOMFIData
{
    std::vector<DroppingIndividual> copies;
    APtr<FoS> fos;
    APtr<ISamplingDistribution> isd;
    APtr<IPerformanceCriterion> ipc;
    SimParallelAsynchronousKernelGOMEA *context;
    Individual ii;

    SimAsyncKernelData(std::vector<DroppingIndividual> &&copies,
                    APtr<FoS> fos,
                    APtr<ISamplingDistribution> isd,
                    APtr<IPerformanceCriterion> ipc,
                    SimParallelAsynchronousKernelGOMEA *context,
                    Individual ii);

    APtr<FoS> getFOSForGOM() override;
    APtr<ISamplingDistribution> getDistributionForGOM() override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForGOM() override;
    APtr<FoS> getFOSForFI() override;
    APtr<ISamplingDistribution> getDistributionForFI() override;
    APtr<IPerformanceCriterion> getPerformanceCriterionForFI() override;
};
