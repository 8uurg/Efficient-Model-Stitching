//  DAEDALUS – Distributed and Automated Evolutionary Deep Architecture Learning with Unprecedented Scalability
// 
// This research code was developed as part of the research programme Open Technology Programme with project number 18373, which was financed by the Dutch Research Council (NWO), Elekta, and Ortec Logiqcare.
// 
// Project leaders: Peter A.N. Bosman, Tanja Alderliesten
// Researchers: Alex Chebykin, Arthur Guijt, Vangelis Kostoulas
// Main code developer: Arthur Guijt

#include "archive.hpp"
#include "base.hpp"
#include "cppassert.h"
#include "ga.hpp"
#include "decentralized.hpp"
#include <optional>

struct PopulationIndex
{
    size_t idx;
};
struct ParentIndices
{
    std::vector<size_t> parent_indices;
};

class DistributedSynchronousSimpleGA : public GenerationalApproach
{
  protected:
    std::shared_ptr<Scheduler> sp;

    const std::shared_ptr<ISolutionInitializer> initializer;
    const std::shared_ptr<ICrossover> crossover;
    const std::shared_ptr<IMutation> mutation;
    const std::shared_ptr<ISelection> parent_selection;
    const std::shared_ptr<IPerformanceCriterion> performance_criterion;

    const std::shared_ptr<IArchive> archive;

    const size_t population_size;
    const size_t offspring_size;

    std::vector<Individual> individuals;
    std::vector<Individual> offspring;

    struct Cache
    {
        Rng &rng;
        std::optional<TypedGetter<PopulationIndex>> population_idx;
        std::optional<TypedGetter<ParentIndices>> parent_indices;
        Population* population;
    };
    std::optional<Cache> cache;

    void doCache()
    {
        if (cache.has_value())
            return;
        
        if (replacement_strategy == 7)
        {
            cache.emplace(Cache{
                *population->getGlobalData<Rng>(),
                population->getDataContainer<PopulationIndex>(),
                population->getDataContainer<ParentIndices>(),
                population.get(),
            });
        }
        else
        {
            cache.emplace(Cache{
                *population->getGlobalData<Rng>(),
                std::nullopt,
                std::nullopt,
                population.get(),
            });
        }
    }

    bool initialized = false;
    virtual void initialize();

    // Replacement strategy: the means by which solutions in the population are replaced.
    int replacement_strategy = 0;
    // Specific replacement strategies!

    // FIFO - Replace in order.
    size_t current_replacement_index = 0;
    void replace_fifo(Individual ii);

    // Replace predetermined index (i.e. offspring index)
    void replace_idx(size_t idx, Individual ii);

    // Replace uniformly
    void replace_uniformly(Individual ii);

    // std::optional<std::shared_ptr<IPerformanceCriterion>> perf_criterion;
    bool replace_if_equal = true;
    bool replace_if_incomparable = true;

    void replace_selection_fifo(Individual ii);
    void replace_selection_idx(size_t idx, Individual ii);
    // familial_replacement calls replace_selection_idxs using metadata assigned to solution.
    void replace_one_idx(Individual ii, std::vector<size_t> &idxs);
    void familial_replacement(Individual ii);
    void population_replacement(Individual ii);
    void replace_selection_uniformly(Individual ii);

    // Generational-like selection
    size_t target_selection_pool_size;
    bool include_population;
    std::vector<Individual> selection_pool;
    const std::shared_ptr<ISelection> generationalish_selection;

    void contender_generational_like_selection(Individual ii);

    void place_in_population(size_t idx, const Individual ii, const std::optional<int> override_replacement_strategy);

    std::vector<Individual> sample_solutions();

    virtual void evaluate_solution(size_t idx, Individual ii);

    std::vector<Individual> sampled_pool;
    virtual void sample_solution(size_t /* idx */, Individual ii)
    {
        if (sampled_pool.size() == 0)
        {
            sampled_pool = sample_solutions();
            t_assert(sampled_pool.size() > 0, "Sampling new offspring should generate at least 1 offspring");
        }

        auto sample = sampled_pool.back();
        population->copyIndividual(sample, ii);
        population->dropIndividual(sample);
        sampled_pool.pop_back();
    }

    virtual void generation();

  public:
    DistributedSynchronousSimpleGA(std::shared_ptr<Scheduler> sp,
                                 size_t population_size,
                                 size_t offspring_size,
                                 int replacement_strategy,
                                 std::shared_ptr<ISolutionInitializer> initializer,
                                 std::shared_ptr<ICrossover> crossover,
                                 std::shared_ptr<IMutation> mutation,
                                 std::shared_ptr<ISelection> parent_selection,
                                 std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                 std::shared_ptr<IArchive> archive);

    DistributedSynchronousSimpleGA(std::shared_ptr<Scheduler> sp,
                                 size_t population_size,
                                 size_t offspring_size,
                                 std::shared_ptr<ISolutionInitializer> initializer,
                                 std::shared_ptr<ICrossover> crossover,
                                 std::shared_ptr<IMutation> mutation,
                                 std::shared_ptr<ISelection> parent_selection,
                                 std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                 std::shared_ptr<IArchive> archive,
                                 bool include_population,
                                 std::shared_ptr<ISelection> generationalish_selection);

    bool terminated() override;
    void step() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    std::vector<Individual> &getSolutionPopulation() override;
};

class DistributedAsynchronousSimpleGA : public DistributedSynchronousSimpleGA
{
    void sample_and_evaluate_new_solution(size_t idx);
    void evaluate_solution(size_t idx, Individual ii) override;

    void initialize() override;
    void generation() override;
    void evaluate_initial_solution(size_t idx, Individual &ii);

    size_t steps_left_until_weight_reassignment = 0;

  public:
    DistributedAsynchronousSimpleGA(std::shared_ptr<Scheduler> sp,
                                  size_t population_size,
                                  size_t offspring_size,
                                  int replacement_strategy,
                                  std::shared_ptr<ISolutionInitializer> initializer,
                                  std::shared_ptr<ICrossover> crossover,
                                  std::shared_ptr<IMutation> mutation,
                                  std::shared_ptr<ISelection> parent_selection,
                                  std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                  std::shared_ptr<IArchive> archive);

    DistributedAsynchronousSimpleGA(std::shared_ptr<Scheduler> sp,
                                  size_t population_size,
                                  size_t offspring_size,
                                  std::shared_ptr<ISolutionInitializer> initializer,
                                  std::shared_ptr<ICrossover> crossover,
                                  std::shared_ptr<IMutation> mutation,
                                  std::shared_ptr<ISelection> parent_selection,
                                  std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                  std::shared_ptr<IArchive> archive,
                                  bool include_population,
                                  std::shared_ptr<ISelection> generationalish_selection);

    bool terminated() override;
};

// Note - not technically a GA.
class DistributedRandomSearch : public GenerationalApproach
{
  protected:
    std::shared_ptr<Scheduler> sp;
    const std::shared_ptr<ISolutionInitializer> initializer;
    std::shared_ptr<IArchive> archive;
    size_t max_pending;
    bool initialized = false;
    std::vector<Individual> pending;
    std::vector<Individual> next_batch;
    size_t batch_idx = 0;

  public:
    DistributedRandomSearch(std::shared_ptr<Scheduler> sp,
                            std::shared_ptr<ISolutionInitializer> initializer,
                            std::shared_ptr<IArchive> archive,
                            size_t max_pending);

    bool terminated() override;

    void evaluate_solution_at(size_t idx);

    void sample_and_evaluate(size_t idx);

    void initialize();

    void step() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    std::vector<Individual> &getSolutionPopulation() override;
};