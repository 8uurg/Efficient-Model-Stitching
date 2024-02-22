#include "archive.hpp"
#include "base.hpp"
#include "cppassert.h"
#include "ga.hpp"
#include "sim.hpp"
#include <optional>

class SimulatedSynchronousSimpleGA : public GenerationalApproach
{
  protected:
    std::shared_ptr<SimulatorParameters> sim;

    const std::shared_ptr<ISolutionInitializer> initializer;
    const std::shared_ptr<ICrossover> crossover;
    const std::shared_ptr<IMutation> mutation;
    const std::shared_ptr<ISelection> parent_selection;
    const std::shared_ptr<IPerformanceCriterion> performance_criterion;

    const std::shared_ptr<IArchive> archive;
    std::shared_ptr<SpanLogger> sl;

    const size_t population_size;
    const size_t offspring_size;

    std::vector<Individual> individuals;
    std::vector<Individual> offspring;

    struct Cache
    {
        TypedGetter<TimeSpent> tgts;
        Rng &rng;
    };
    std::optional<Cache> cache;
    void doCache()
    {
        if (cache.has_value())
            return;
        cache.emplace(Cache{
            population->getDataContainer<TimeSpent>(),
            *population->getGlobalData<Rng>(),
        });
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
    void replace_selection_uniformly(Individual ii);

    // Generational-like selection
    size_t target_selection_pool_size;
    bool include_population;
    std::vector<Individual> selection_pool;
    const std::shared_ptr<ISelection> generationalish_selection;

    void contender_generational_like_selection(Individual ii);

    void place_in_population(size_t idx, const Individual ii, const std::optional<int> override_replacement_strategy);

    std::vector<Individual> sample_solutions();

    virtual void evaluate_initial_solution(size_t idx, Individual &ii);

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
    SimulatedSynchronousSimpleGA(std::shared_ptr<SimulatorParameters> sim,
                                 size_t population_size,
                                 size_t offspring_size,
                                 int replacement_strategy,
                                 std::shared_ptr<ISolutionInitializer> initializer,
                                 std::shared_ptr<ICrossover> crossover,
                                 std::shared_ptr<IMutation> mutation,
                                 std::shared_ptr<ISelection> parent_selection,
                                 std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                 std::shared_ptr<IArchive> archive,
                                 std::shared_ptr<SpanLogger> sl = NULL);

    SimulatedSynchronousSimpleGA(std::shared_ptr<SimulatorParameters> sim,
                                 size_t population_size,
                                 size_t offspring_size,
                                 std::shared_ptr<ISolutionInitializer> initializer,
                                 std::shared_ptr<ICrossover> crossover,
                                 std::shared_ptr<IMutation> mutation,
                                 std::shared_ptr<ISelection> parent_selection,
                                 std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                 std::shared_ptr<IArchive> archive,
                                 bool include_population,
                                 std::shared_ptr<ISelection> generationalish_selection,
                                 std::shared_ptr<SpanLogger> sl = NULL);

    bool terminated() override;
    void step() override;

    void setPopulation(std::shared_ptr<Population> population) override;
    void registerData() override;
    void afterRegisterData() override;

    std::vector<Individual> &getSolutionPopulation() override;
};

class SimulatedAsynchronousSimpleGA : public SimulatedSynchronousSimpleGA
{
    void sample_and_evaluate_new_solution(size_t idx);
    void evaluate_solution(size_t idx, Individual ii) override;

    void initialize() override;
    void generation() override;
    void evaluate_initial_solution(size_t idx, Individual &ii) override;

  public:
    SimulatedAsynchronousSimpleGA(std::shared_ptr<SimulatorParameters> sim,
                                  size_t population_size,
                                  size_t offspring_size,
                                  int replacement_strategy,
                                  std::shared_ptr<ISolutionInitializer> initializer,
                                  std::shared_ptr<ICrossover> crossover,
                                  std::shared_ptr<IMutation> mutation,
                                  std::shared_ptr<ISelection> parent_selection,
                                  std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                  std::shared_ptr<IArchive> archive,
                                  std::shared_ptr<SpanLogger> sl = NULL);

    SimulatedAsynchronousSimpleGA(std::shared_ptr<SimulatorParameters> sim,
                                  size_t population_size,
                                  size_t offspring_size,
                                  std::shared_ptr<ISolutionInitializer> initializer,
                                  std::shared_ptr<ICrossover> crossover,
                                  std::shared_ptr<IMutation> mutation,
                                  std::shared_ptr<ISelection> parent_selection,
                                  std::shared_ptr<IPerformanceCriterion> performance_criterion,
                                  std::shared_ptr<IArchive> archive,
                                  bool include_population,
                                  std::shared_ptr<ISelection> generationalish_selection,
                                  std::shared_ptr<SpanLogger> sl = NULL);
};
